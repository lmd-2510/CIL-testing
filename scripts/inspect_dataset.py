import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect large CIL-DDI CSV splits safely.")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--valid", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--target-col", type=str, default="class")
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument(
        "--count-classes",
        action="store_true",
        help="Count rows per class. This scans the target column but avoids loading all features.",
    )
    return parser


def scan_schema(path: Path) -> Dict[str, pl.DataType]:
    return pl.scan_csv(path).collect_schema()


def read_head(path: Path, n_rows: int) -> pl.DataFrame:
    return pl.read_csv(path, n_rows=n_rows)


def count_rows(path: Path) -> int:
    return int(pl.scan_csv(path).select(pl.len().alias("rows")).collect().item())


def count_classes(path: Path, target_col: str) -> Optional[pl.DataFrame]:
    schema = scan_schema(path)
    if target_col not in schema:
        return None
    return (
        pl.scan_csv(path)
        .select(pl.col(target_col))
        .group_by(target_col)
        .len()
        .sort(target_col)
        .collect()
    )


def inspect_split(name: str, path_str: str, target_col: str, head: int, do_count_classes: bool) -> Dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    schema = scan_schema(path)
    columns = list(schema.keys())
    target_exists = target_col in schema
    feature_columns = [col for col in columns if col != target_col]

    print(f"\n=== {name.upper()} ===")
    print(f"Path          : {path}")
    print(f"Columns       : {len(columns)}")
    print(f"Feature cols  : {len(feature_columns)}")
    print(f"Has target    : {target_exists}")
    print(f"Target dtype  : {schema.get(target_col) if target_exists else None}")

    print("\nFirst rows:")
    print(read_head(path, head))

    rows = count_rows(path)
    print(f"\nRows          : {rows}")

    class_counts = None
    if do_count_classes:
        class_counts = count_classes(path, target_col)
        if class_counts is not None:
            print(f"Num classes   : {class_counts.height}")
            print("\nClass count sample:")
            print(class_counts.head(20))

    return {
        "name": name,
        "path": str(path),
        "rows": rows,
        "columns": columns,
        "schema": {col: str(dtype) for col, dtype in schema.items()},
        "has_target": target_exists,
        "num_features": len(feature_columns),
        "num_classes": class_counts.height if class_counts is not None else None,
    }


def compare_columns(reference: Dict, others: List[Dict]) -> None:
    reference_columns = reference["columns"]
    for split in others:
        missing = [col for col in reference_columns if col not in split["columns"]]
        extra = [col for col in split["columns"] if col not in reference_columns]

        print(f"\n=== COLUMN CHECK: {reference['name']} vs {split['name']} ===")
        print(f"Same column order: {reference_columns == split['columns']}")
        print(f"Missing columns  : {len(missing)}")
        print(f"Extra columns    : {len(extra)}")
        if missing:
            print(f"Missing sample   : {missing[:10]}")
        if extra:
            print(f"Extra sample     : {extra[:10]}")


def main() -> None:
    args = build_parser().parse_args()

    splits = [
        ("train", args.train),
        ("valid", args.valid),
        ("test", args.test),
    ]
    inspected = []
    for name, path in splits:
        if path is None:
            continue
        inspected.append(
            inspect_split(
                name=name,
                path_str=path,
                target_col=args.target_col,
                head=args.head,
                do_count_classes=args.count_classes,
            )
        )

    if len(inspected) > 1:
        compare_columns(inspected[0], inspected[1:])


if __name__ == "__main__":
    main()
