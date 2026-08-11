import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from data.ddi_incremental import DDIIncrementalDataManager
from utils import get_device, json_safe, load_config_defaults, set_seed


DEFAULT_CONFIG_PATH = "configs/tddi.yaml"
DEFAULT_RESULTS_DIR = "results"
SUPPORTED_MODELS = {"tddi", "mlp_resnet", "model3"}
SUPPORTED_METHODS = {"finetune", "ewc", "gem", "agem"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main entry point for CIL-DDI experiment setup and orchestration."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=str, default="data/material")
    parser.add_argument("--train-file", type=str, default="converted_ddi_types.csv")
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--valid-file", type=str, default=None)
    parser.add_argument("--target-col", type=str, default="class")
    parser.add_argument("--columns-to-drop", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--classes-per-task", type=int, default=None)
    parser.add_argument("--shuffle-classes", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="tddi", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument(
        "--method", type=str, default="finetune", choices=sorted(SUPPORTED_METHODS)
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="setup",
        choices=["setup", "train"],
        help="`setup`: prepare data and save run metadata. `train`: call trainer hook if implemented.",
    )
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-summary", action="store_true")
    parser.add_argument("--preview-batch", action="store_true")
    return parser


def resolve_setting(
    cli_value: Optional[Any], config_value: Optional[Any], fallback: Optional[Any] = None
) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return fallback


def normalize_columns_to_drop(raw_value: Optional[Any]) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, (list, tuple)):
        columns = []
        for item in raw_value:
            if not item:
                continue
            columns.extend(part.strip() for part in str(item).split(",") if part.strip())
        return columns
    return []


def build_run_name(model_name: str, method_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{method_name}_{timestamp}"


def build_data_manager(args: argparse.Namespace) -> DDIIncrementalDataManager:
    return DDIIncrementalDataManager(
        data_dir=args.data_dir,
        train_filename=args.train_file,
        test_filename=args.test_file,
        valid_filename=args.valid_file,
        columns_to_drop=args.columns_to_drop,
        target_col=args.target_col,
        seed=args.seed,
    )


def summarize_tasks(data_manager: DDIIncrementalDataManager) -> list[Dict[str, Any]]:
    task_rows = []
    for task_id, class_ids in data_manager.task_classes.items():
        task_rows.append(
            {
                "task_id": task_id,
                "num_classes": len(class_ids),
                "classes": class_ids,
                "train_samples": int(len(data_manager.train_task_indices[task_id])),
                "test_samples": int(len(data_manager.test_task_indices[task_id])),
                "valid_samples": int(len(data_manager.valid_task_indices[task_id])),
            }
        )
    return task_rows


def maybe_preview_first_batch(
    data_manager: DDIIncrementalDataManager, batch_size: int, num_workers: int
) -> Optional[Dict[str, Any]]:
    if not data_manager.task_classes:
        return None
    first_task_id = min(data_manager.task_classes.keys())
    loader = data_manager.get_train_loader(
        task_id=first_task_id,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    batch = next(iter(loader), None)
    if batch is None:
        return None
    cat_x, cont_x, y = batch
    return {
        "task_id": first_task_id,
        "cat_shape": list(cat_x.shape),
        "cont_shape": list(cont_x.shape),
        "target_shape": list(y.shape),
    }


def save_run_summary(results_dir: str, run_name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{run_name}_setup.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
    return output_path


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== CIL-DDI Setup Summary ===")
    print(f"Run name        : {summary['run_name']}")
    print(f"Mode            : {summary['mode']}")
    print(f"Model           : {summary['model']}")
    print(f"Method          : {summary['method']}")
    print(f"Device          : {summary['device']}")
    print(f"Train file      : {summary['data']['train_file']}")
    print(f"Target column   : {summary['data']['target_col']}")
    print(f"Num classes     : {summary['data']['num_classes']}")
    print(f"Num tasks       : {summary['data']['num_tasks']}")
    print(f"Categorical cols: {summary['data']['num_categorical']}")
    print(f"Continuous cols : {summary['data']['num_continuous']}")
    print(f"Batch size      : {summary['runtime']['batch_size']}")
    print(f"Num workers     : {summary['runtime']['num_workers']}")

    print("\nTask breakdown:")
    for task_info in summary["tasks"]:
        print(
            f" - Task {task_info['task_id']}: "
            f"{task_info['num_classes']} classes | "
            f"train={task_info['train_samples']} | "
            f"test={task_info['test_samples']} | "
            f"valid={task_info['valid_samples']}"
        )

    batch_preview = summary.get("preview_batch")
    if batch_preview:
        print("\nPreview batch:")
        print(
            f" - Task {batch_preview['task_id']} | "
            f"cat={batch_preview['cat_shape']} | "
            f"cont={batch_preview['cont_shape']} | "
            f"target={batch_preview['target_shape']}"
        )

    if summary.get("summary_path"):
        print(f"\nSaved summary   : {summary['summary_path']}")


def call_trainer_hook(summary: Dict[str, Any], data_manager: DDIIncrementalDataManager) -> None:
    try:
        from trainer import run_experiment
    except ImportError as exc:
        raise NotImplementedError(
            "trainer.py chưa có `run_experiment(...)`. Hoàn thiện trainer trước khi chạy --mode train."
        ) from exc

    run_experiment(summary=summary, data_manager=data_manager)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_defaults = load_config_defaults(args.config)
    args.seed = resolve_setting(args.seed, config_defaults.get("seed"), 42)
    args.batch_size = resolve_setting(args.batch_size, config_defaults.get("batch_size"), 256)
    args.columns_to_drop = normalize_columns_to_drop(args.columns_to_drop)
    args.run_name = args.run_name or build_run_name(args.model, args.method)

    set_seed(args.seed)
    device = get_device()

    data_manager = build_data_manager(args)
    data_manager.prepare_global_data()
    data_manager.split_tasks(
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        shuffle_classes=args.shuffle_classes,
    )

    summary: Dict[str, Any] = {
        "run_name": args.run_name,
        "mode": args.mode,
        "model": args.model,
        "method": args.method,
        "device": str(device),
        "config_path": args.config,
        "config_defaults": config_defaults,
        "data": {
            "data_dir": args.data_dir,
            "train_file": args.train_file,
            "test_file": args.test_file,
            "valid_file": args.valid_file,
            "target_col": args.target_col,
            "columns_to_drop": args.columns_to_drop,
            "num_classes": data_manager.num_classes,
            "num_tasks": data_manager.num_tasks,
            "num_categorical": len(data_manager.categorical_cols),
            "num_continuous": data_manager.num_continuous,
            "categories": data_manager.categories,
        },
        "runtime": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "shuffle_classes": args.shuffle_classes,
            "classes_per_task": args.classes_per_task,
        },
        "tasks": summarize_tasks(data_manager),
    }

    if args.preview_batch:
        summary["preview_batch"] = maybe_preview_first_batch(
            data_manager=data_manager,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    if args.save_summary:
        summary["summary_path"] = save_run_summary(args.results_dir, args.run_name, summary)

    print_summary(summary)

    if args.mode == "train":
        call_trainer_hook(summary=summary, data_manager=data_manager)


if __name__ == "__main__":
    main()
