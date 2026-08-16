import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluator import CILEvaluator, summarize_cl_metrics
from utils import MemoryOptimizer, json_safe, save_checkpoint


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 1.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()


class SupportsForward(Protocol):
    def __call__(self, cat_x: torch.Tensor, cont_x: torch.Tensor) -> torch.Tensor:
        ...


class CLMethod(Protocol):
    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        memory_batch: Tuple[torch.Tensor, ...],
    ) -> None:
        ...

    def training_step(
        self,
        model: SupportsForward,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        ...

    def after_task(self, task_id: int, model: SupportsForward, train_loader: DataLoader) -> None:
        ...


@dataclass
class TrainerConfig:
    model_name: str
    method_name: str
    device: torch.device
    run_name: str
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    batch_size: int = 256
    num_workers: int = 0
    epochs_per_task: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    save_checkpoints: bool = False
    memory_samples_per_class: int = 10


@dataclass
class TaskMetrics:
    task_id: int
    train_loss: List[float] = field(default_factory=list)
    eval_current: Optional[Dict[str, float]] = None
    eval_cumulative: Optional[Dict[str, float]] = None
    eval_seen_tasks: List[Dict[str, Any]] = field(default_factory=list)
    eval_next_task: Optional[Dict[str, Any]] = None


class ContinualTrainer:
    def __init__(
        self,
        config: TrainerConfig,
        summary: Dict[str, Any],
        data_manager: Any,
        model: SupportsForward,
        optimizer: torch.optim.Optimizer,
        method: CLMethod,
    ):
        self.config = config
        self.summary = summary
        self.data_manager = data_manager
        self.model = model
        self.optimizer = optimizer
        self.method = method
        self.task_results: List[TaskMetrics] = []
        self.evaluator = CILEvaluator(device=self.config.device)
        self.matrix_evaluations: List[List[Any]] = []

    def _seen_classes(self, train_task_id: int) -> List[int]:
        classes: List[int] = []
        for task_id in range(train_task_id + 1):
            classes.extend(self.data_manager.task_classes.get(task_id, []))
        return sorted(set(int(cls) for cls in classes))

    def run(self) -> Dict[str, Any]:
        print("\n=== Start continual training ===")
        for task_id in sorted(self.data_manager.task_classes.keys()):
            task_result = self.train_task(task_id)
            self.task_results.append(task_result)
            MemoryOptimizer.cleanup_memory()

        payload = {
            "run_name": self.config.run_name,
            "model": self.config.model_name,
            "method": self.config.method_name,
            "epochs_per_task": self.config.epochs_per_task,
            "config_path": self.summary.get("config_path"),
            "config_defaults": self.summary.get("config_defaults", {}),
            "method_config": self.summary.get("method_config", {}),
            "criterion_config": self.summary.get("criterion_config", {}),
            "evaluation_config": {
                "mask_unseen_classes": True,
                "fwt_next_task_mask": None,
            },
            "data": self.summary.get("data", {}),
            "runtime": self.summary.get("runtime", {}),
            "tasks": self.summary.get("tasks", []),
            "task_results": [json_safe(task.__dict__) for task in self.task_results],
            "continual_metrics": summarize_cl_metrics(self.matrix_evaluations),
        }
        self._save_results(payload)
        return payload

    def train_task(self, task_id: int) -> TaskMetrics:
        train_loader = self.data_manager.get_train_loader(
            task_id=task_id,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        memory_batch = self.data_manager.get_memory_samples(
            task_id=task_id,
            samples_per_class=self.config.memory_samples_per_class,
        )

        self.method.before_task(task_id, train_loader, memory_batch)

        task_metrics = TaskMetrics(task_id=task_id)
        for epoch in range(self.config.epochs_per_task):
            epoch_loss = self._train_one_epoch(train_loader)
            task_metrics.train_loss.append(epoch_loss)
            print(
                f"Task {task_id} | Epoch {epoch + 1}/{self.config.epochs_per_task} | "
                f"loss={epoch_loss:.4f}"
            )

        self.method.after_task(task_id, self.model, train_loader)

        allowed_classes = self._seen_classes(task_id)
        seen_evaluations = self.evaluator.evaluate_seen_tasks(
            model=self.model,
            data_manager=self.data_manager,
            train_task_id=task_id,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            allowed_classes=allowed_classes,
        )
        stage_evaluations = list(seen_evaluations)

        next_task_id = task_id + 1
        if next_task_id in self.data_manager.task_classes:
            next_evaluation = self.evaluator.evaluate_single_task(
                model=self.model,
                data_manager=self.data_manager,
                eval_task_id=next_task_id,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                allowed_classes=None,
            )
            next_evaluation.train_task_id = task_id
            stage_evaluations.append(next_evaluation)
            task_metrics.eval_next_task = json_safe(next_evaluation.__dict__)

        self.matrix_evaluations.append(stage_evaluations)
        task_metrics.eval_seen_tasks = [json_safe(evaluation.__dict__) for evaluation in seen_evaluations]
        task_metrics.eval_current = seen_evaluations[-1].metrics if seen_evaluations else None
        task_metrics.eval_cumulative = self.evaluator.evaluate_cumulative(
            model=self.model,
            data_manager=self.data_manager,
            train_task_id=task_id,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            allowed_classes=allowed_classes,
        )

        if self.config.save_checkpoints:
            checkpoint_path = os.path.join(
                self.config.checkpoints_dir,
                f"{self.config.run_name}_task{task_id}.pth",
            )
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                task_id=task_id,
                epoch=self.config.epochs_per_task,
                filepath=checkpoint_path,
            )

        return task_metrics

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch_loss = self.method.training_step(
                model=self.model,
                batch=batch,
                optimizer=self.optimizer,
                device=self.config.device,
            )
            total_loss += batch_loss
            num_batches += 1
        if num_batches == 0:
            return 0.0
        return total_loss / num_batches

    def _save_results(self, payload: Dict[str, Any]) -> str:
        os.makedirs(self.config.results_dir, exist_ok=True)
        output_path = os.path.join(self.config.results_dir, f"{self.config.run_name}_train.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        print(f"Training results saved: {output_path}")
        return output_path


def resolve_method_kwargs(name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    config_defaults = summary.get("config_defaults", {})
    method_kwargs: Dict[str, Any] = {}

    if name == "ewc":
        if "ewc_lambda" in config_defaults:
            method_kwargs["ewc_lambda"] = config_defaults["ewc_lambda"]
        if "fisher_n_batches" in config_defaults:
            method_kwargs["fisher_n_batches"] = config_defaults["fisher_n_batches"]
    elif name == "gem":
        if "gem_memory_strength" in config_defaults:
            method_kwargs["memory_strength"] = config_defaults["gem_memory_strength"]
    elif name == "er":
        if "er_replay_weight" in config_defaults:
            method_kwargs["replay_weight"] = config_defaults["er_replay_weight"]
        if "er_replay_batch_size" in config_defaults:
            method_kwargs["replay_batch_size"] = config_defaults["er_replay_batch_size"]

    return method_kwargs


def load_method(
    name: str,
    summary: Dict[str, Any],
    criterion: Optional[nn.Module] = None,
) -> CLMethod:
    module = importlib.import_module(f"methods.{name}")
    builder = getattr(module, "build_method", None)
    if builder is None:
        raise NotImplementedError(
            f"methods/{name}.py chua co `build_method()`. Hoan thien method nay truoc khi train."
        )

    method_kwargs = resolve_method_kwargs(name, summary)
    if criterion is not None:
        method_kwargs["criterion"] = criterion
    summary["method_config"] = {
        key: value
        for key, value in method_kwargs.items()
        if key != "criterion"
    }
    return builder(**method_kwargs)


def load_model(
    model_name: str,
    summary: Dict[str, Any],
    data_manager: Any,
    device: torch.device,
) -> SupportsForward:
    module = importlib.import_module(f"models.{model_name}")
    builder = getattr(module, "build_model_for_cil", None)
    if builder is None:
        raise NotImplementedError(
            f"models/{model_name}.py chua co `build_model_for_cil(...)`. "
            "Can adapt model sang interface CIL truoc khi train."
        )
    return builder(summary=summary, data_manager=data_manager, device=device)


def build_optimizer(model: SupportsForward, summary: Dict[str, Any]) -> torch.optim.Optimizer:
    runtime = summary.get("runtime", {})
    config_defaults = summary.get("config_defaults", {})
    learning_rate = runtime.get("learning_rate", config_defaults.get("learning_rate", 1e-3))
    weight_decay = runtime.get("weight_decay", config_defaults.get("weight_decay", 0.0))
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_class_weights(summary: Dict[str, Any], data_manager: Any, device: torch.device) -> Optional[torch.Tensor]:
    config_defaults = summary.get("config_defaults", {})
    if not config_defaults.get("use_class_weights", True):
        return None

    counts = torch.bincount(
        torch.as_tensor(data_manager.y_train, dtype=torch.long),
        minlength=data_manager.num_classes,
    ).float()
    counts = counts.clamp_min(1.0)

    power = float(config_defaults.get("class_weight_power", 0.5))
    weights = counts.sum() / (counts * float(data_manager.num_classes))
    weights = weights.pow(power)

    max_weight = config_defaults.get("max_class_weight", 10.0)
    if max_weight is not None:
        weights = weights.clamp(max=float(max_weight))

    weights = weights / weights.mean()
    return weights.to(device)


def build_criterion(summary: Dict[str, Any], data_manager: Any, device: torch.device) -> nn.Module:
    config_defaults = summary.get("config_defaults", {})
    class_weights = build_class_weights(summary, data_manager, device)
    focal_gamma = config_defaults.get("focal_gamma", None)

    if focal_gamma is not None and float(focal_gamma) > 0:
        return FocalLoss(gamma=float(focal_gamma), weight=class_weights)
    return nn.CrossEntropyLoss(weight=class_weights)


def build_trainer_config(summary: Dict[str, Any]) -> TrainerConfig:
    runtime = summary.get("runtime", {})
    config_defaults = summary.get("config_defaults", {})
    return TrainerConfig(
        model_name=summary["model"],
        method_name=summary["method"],
        device=torch.device(summary["device"]),
        run_name=summary["run_name"],
        results_dir=summary.get("results_dir", "results"),
        checkpoints_dir=summary.get("checkpoints_dir", "checkpoints"),
        batch_size=runtime.get("batch_size", config_defaults.get("batch_size", 256)),
        num_workers=runtime.get("num_workers", 0),
        epochs_per_task=config_defaults.get("epochs_per_task", 1),
        learning_rate=config_defaults.get("learning_rate", 1e-3),
        weight_decay=config_defaults.get("weight_decay", 0.0),
        save_checkpoints=config_defaults.get("save_checkpoints", False),
        memory_samples_per_class=config_defaults.get("memory_samples_per_class", 10),
    )


def run_experiment(summary: Dict[str, Any], data_manager: Any) -> Dict[str, Any]:
    config = build_trainer_config(summary)
    model = load_model(
        model_name=config.model_name,
        summary=summary,
        data_manager=data_manager,
        device=config.device,
    )
    criterion = build_criterion(summary, data_manager, config.device)
    summary["criterion_config"] = {
        "name": criterion.__class__.__name__,
        "focal_gamma": summary.get("config_defaults", {}).get("focal_gamma"),
        "use_class_weights": summary.get("config_defaults", {}).get("use_class_weights", True),
        "class_weight_power": summary.get("config_defaults", {}).get("class_weight_power", 0.5),
        "max_class_weight": summary.get("config_defaults", {}).get("max_class_weight", 10.0),
    }
    method = load_method(config.method_name, summary, criterion=criterion)
    optimizer = build_optimizer(model, summary)

    trainer = ContinualTrainer(
        config=config,
        summary=summary,
        data_manager=data_manager,
        model=model,
        optimizer=optimizer,
        method=method,
    )
    return trainer.run()
