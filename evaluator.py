from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import classification_metrics


def empty_metrics() -> Dict[str, float]:
    return {
        "accuracy": float("nan"),
        "macro_precision": float("nan"),
        "macro_recall": float("nan"),
        "macro_f1": float("nan"),
        "weighted_f1": float("nan"),
    }


@dataclass
class TaskEvaluation:
    train_task_id: int
    eval_task_id: int
    metrics: Dict[str, float]
    num_samples: int


class CILEvaluator:
    def __init__(self, device: torch.device):
        self.device = device

    def predict_loader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        allowed_classes: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int]]:
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        allowed_tensor = None
        if allowed_classes is not None:
            allowed_tensor = torch.as_tensor(allowed_classes, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for cat_x, cont_x, targets in loader:
                cat_x = cat_x.to(self.device)
                cont_x = cont_x.to(self.device)
                logits = model(cat_x, cont_x)
                if allowed_tensor is not None:
                    masked_logits = torch.full_like(logits, float("-inf"))
                    masked_logits.index_copy_(1, allowed_tensor, logits.index_select(1, allowed_tensor))
                    logits = masked_logits
                preds = torch.argmax(logits, dim=1)
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        model.train()
        return y_true, y_pred

    def evaluate_loader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        allowed_classes: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        y_true, y_pred = self.predict_loader(model, loader, allowed_classes=allowed_classes)
        return classification_metrics(y_true, y_pred)

    def evaluate_single_task(
        self,
        model: torch.nn.Module,
        data_manager: Any,
        eval_task_id: int,
        batch_size: int = 256,
        num_workers: int = 0,
        allowed_classes: Optional[List[int]] = None,
    ) -> TaskEvaluation:
        try:
            loader = data_manager.get_test_loader(
                task_id=eval_task_id,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        except ValueError:
            return TaskEvaluation(
                train_task_id=-1,
                eval_task_id=eval_task_id,
                metrics=empty_metrics(),
                num_samples=0,
            )

        metrics = self.evaluate_loader(model, loader, allowed_classes=allowed_classes)
        num_samples = len(loader.dataset) if hasattr(loader, "dataset") else 0
        return TaskEvaluation(
            train_task_id=-1,
            eval_task_id=eval_task_id,
            metrics=metrics,
            num_samples=num_samples,
        )

    def evaluate_seen_tasks(
        self,
        model: torch.nn.Module,
        data_manager: Any,
        train_task_id: int,
        batch_size: int = 256,
        num_workers: int = 0,
        allowed_classes: Optional[List[int]] = None,
    ) -> List[TaskEvaluation]:
        evaluations: List[TaskEvaluation] = []
        for eval_task_id in range(train_task_id + 1):
            try:
                loader = data_manager.get_test_loader(
                    task_id=eval_task_id,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                metrics = self.evaluate_loader(model, loader, allowed_classes=allowed_classes)
                num_samples = len(loader.dataset) if hasattr(loader, "dataset") else 0
            except ValueError:
                metrics = empty_metrics()
                num_samples = 0

            evaluations.append(
                TaskEvaluation(
                    train_task_id=train_task_id,
                    eval_task_id=eval_task_id,
                    metrics=metrics,
                    num_samples=num_samples,
                )
            )
        return evaluations

    def evaluate_cumulative(
        self,
        model: torch.nn.Module,
        data_manager: Any,
        train_task_id: int,
        batch_size: int = 256,
        num_workers: int = 0,
        allowed_classes: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        try:
            loader = data_manager.get_cumulative_test_loader(
                task_id=train_task_id,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        except ValueError:
            return empty_metrics()
        return self.evaluate_loader(model, loader, allowed_classes=allowed_classes)


def build_result_matrix(
    evaluations_by_stage: List[List[TaskEvaluation]],
    metric_name: str = "accuracy",
) -> np.ndarray:
    if not evaluations_by_stage:
        return np.empty((0, 0), dtype=float)

    num_tasks = len(evaluations_by_stage)
    matrix = np.full((num_tasks, num_tasks), np.nan, dtype=float)

    for stage_evaluations in evaluations_by_stage:
        for evaluation in stage_evaluations:
            train_task_id = evaluation.train_task_id
            eval_task_id = evaluation.eval_task_id
            matrix[train_task_id, eval_task_id] = evaluation.metrics.get(metric_name, np.nan)

    return matrix


def compute_average_accuracy(result_matrix: np.ndarray) -> List[float]:
    averages: List[float] = []
    for task_id in range(result_matrix.shape[0]):
        row = result_matrix[task_id, : task_id + 1]
        valid = row[~np.isnan(row)]
        averages.append(float(np.mean(valid)) if len(valid) > 0 else float("nan"))
    return averages


def compute_forgetting_per_task(result_matrix: np.ndarray) -> Dict[int, float]:
    num_tasks = result_matrix.shape[0]
    forgetting: Dict[int, float] = {}

    for task_id in range(num_tasks):
        column = result_matrix[:, task_id]
        observed = column[~np.isnan(column)]
        if len(observed) <= 1:
            forgetting[task_id] = float("nan")
            continue
        best_past = float(np.max(observed[:-1]))
        current = float(observed[-1])
        forgetting[task_id] = best_past - current

    return forgetting


def compute_mean_forgetting(result_matrix: np.ndarray) -> float:
    forgetting = compute_forgetting_per_task(result_matrix)
    valid = [value for value in forgetting.values() if not np.isnan(value)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def compute_backward_transfer(result_matrix: np.ndarray) -> float:
    """BWT = final performance on old tasks - performance right after learning them."""
    num_tasks = result_matrix.shape[0]
    if num_tasks <= 1:
        return float("nan")

    final_row = result_matrix[num_tasks - 1]
    values = []
    for task_id in range(num_tasks - 1):
        final_score = final_row[task_id]
        learned_score = result_matrix[task_id, task_id]
        if not np.isnan(final_score) and not np.isnan(learned_score):
            values.append(float(final_score - learned_score))
    if not values:
        return float("nan")
    return float(np.mean(values))


def compute_forward_transfer(result_matrix: np.ndarray) -> float:
    """Raw FWT = performance on task i before training task i.

    This uses R[i-1, i], so trainer must evaluate the next unseen task after
    each stage. No random/init baseline is subtracted.
    """
    num_tasks = result_matrix.shape[0]
    if num_tasks <= 1:
        return float("nan")

    values = []
    for task_id in range(1, num_tasks):
        score_before_learning = result_matrix[task_id - 1, task_id]
        if not np.isnan(score_before_learning):
            values.append(float(score_before_learning))
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize_cl_metrics(
    evaluations_by_stage: List[List[TaskEvaluation]],
    primary_metric: str = "accuracy",
) -> Dict[str, Any]:
    result_matrix = build_result_matrix(evaluations_by_stage, metric_name=primary_metric)
    return {
        "primary_metric": primary_metric,
        "result_matrix": result_matrix.tolist(),
        "average_accuracy": compute_average_accuracy(result_matrix),
        "forgetting_per_task": compute_forgetting_per_task(result_matrix),
        "mean_forgetting": compute_mean_forgetting(result_matrix),
        "backward_transfer": compute_backward_transfer(result_matrix),
        "forward_transfer": compute_forward_transfer(result_matrix),
        "forward_transfer_note": "Raw FWT from R[i-1,i]; no random/init baseline subtracted.",
    }
