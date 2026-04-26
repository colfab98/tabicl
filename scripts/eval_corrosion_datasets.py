#!/usr/bin/env python
"""Evaluate TabICL checkpoints on external corrosion benchmark tasks.

The tasks are built from the corrosion_datasets workspace. Continuous corrosion
targets are converted to binary high/low classification tasks by a median split,
so local stage-1 classifier checkpoints can be compared consistently.

This is an evaluation benchmark only. Do not use these results to re-select the
informed-prior values that were derived from the same external datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "scripts"
if str(ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_DIR))

from analyze_structure import GROUPS, Table, load_all_tables, to_float  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "eval_results"
DEFAULT_FEATURE_GROUPS = ("material", "environment", "history", "intervention")
PRIMARY_TARGET_PATTERNS = (
    "corrosion rate",
    "pitting potential",
    "corrosion current",
    "epit",
    "ecorr",
    "ocp",
    "rate",
)


@dataclass
class EvalTask:
    task_id: str
    dataset: str
    table: str
    target: str
    threshold: float
    X: pd.DataFrame
    y: pd.Series
    feature_groups_used: list[str]
    dropped_feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default=None, help="Run suffix such as v15, or full checkpoint dir name.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint name/step inside the run dir: latest, 3350, step-3350, or step-3350.ckpt.",
    )
    parser.add_argument("--local-ckpt-path", type=Path, default=None, help="Explicit local .ckpt path. Overrides --run.")
    parser.add_argument("--local-model-label", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--run-prefix", type=str, default="tabicl_s1mini_generic_")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-mode", choices=("primary", "all"), default="primary")
    parser.add_argument("--dataset", action="append", help="Limit to dataset id. Can be passed multiple times.")
    parser.add_argument("--task", action="append", help="Limit to exact task id. Can be passed multiple times.")
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=2000,
        help="Stratified cap per task before holdout split. Use 0 to disable.",
    )
    parser.add_argument("--max-category-cardinality", type=int, default=80)
    parser.add_argument("--include-electrochem-features", action="store_true")
    parser.add_argument("--compare-pretrained-tabicl", dest="compare_pretrained_tabicl", action="store_true", default=True)
    parser.add_argument("--no-compare-pretrained-tabicl", dest="compare_pretrained_tabicl", action="store_false")
    parser.add_argument("--pretrained-checkpoint-version", type=str, default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--baseline-auto-download", action="store_true", default=True)
    parser.add_argument("--no-baseline-auto-download", dest="baseline_auto_download", action="store_false")
    parser.add_argument("--compare-tabpfn", action="store_true", help="Try to evaluate TabPFNClassifier if tabpfn is installed.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--list-tasks", action="store_true", help="List generated tasks and exit before loading any model.")
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.local_ckpt_path is not None:
        path = args.local_ckpt_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local checkpoint not found: {path}")
        return path

    if not args.run:
        raise ValueError("Provide --run v15 or --local-ckpt-path.")

    run = args.run
    run_dir_name = run if run.startswith(args.run_prefix) else f"{args.run_prefix}{run}"
    run_dir = (args.checkpoint_root / run_dir_name).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")

    checkpoint = args.checkpoint
    if checkpoint == "latest":
        candidates = sorted(
            run_dir.glob("step-*.ckpt"),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        if not candidates:
            raise FileNotFoundError(f"No step-*.ckpt files found in {run_dir}")
        return candidates[-1]

    if checkpoint.isdigit():
        checkpoint = f"step-{checkpoint}.ckpt"
    elif checkpoint.startswith("step-") and not checkpoint.endswith(".ckpt"):
        checkpoint = f"{checkpoint}.ckpt"

    path = run_dir / checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def finite_target_values(table: Table, target_col: str) -> np.ndarray:
    values = np.array([to_float(row.get(target_col)) for row in table.rows], dtype=float)
    return values[np.isfinite(values)]


def choose_target_columns(table: Table, mode: str, min_samples: int, min_class_count: int) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for col in table.columns:
        if table.groups.get(col) != "target":
            continue
        values = finite_target_values(table, col)
        if len(values) < min_samples:
            continue
        threshold = float(np.nanmedian(values))
        labels = values > threshold
        class_counts = np.bincount(labels.astype(int), minlength=2)
        if int(class_counts.min()) < min_class_count:
            continue
        lower_name = col.lower()
        priority = next((i for i, pattern in enumerate(PRIMARY_TARGET_PATTERNS) if pattern in lower_name), 99)
        candidates.append((priority, -len(values), col))

    candidates.sort()
    cols = [col for _, _, col in candidates]
    if mode == "primary":
        return cols[:1]
    return cols


def feature_value_series(table: Table, column: str) -> tuple[pd.Series | None, str]:
    raw = [row.get(column) for row in table.rows]
    numeric = np.array([to_float(value) for value in raw], dtype=float)
    finite_ratio = float(np.isfinite(numeric).mean()) if len(numeric) else 0.0
    if finite_ratio >= 0.8 and np.nanstd(numeric) > 0:
        return pd.Series(numeric, name=column), "numeric"

    text = pd.Series(["" if value is None else str(value).strip() for value in raw], name=column)
    text = text.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "None": pd.NA})
    if text.dropna().nunique() >= 2:
        return text, "categorical"
    return None, "empty"


def build_task(
    table: Table,
    target_col: str,
    *,
    feature_groups: tuple[str, ...],
    max_category_cardinality: int,
    min_samples: int,
    min_class_count: int,
    max_samples_per_task: int,
    random_state: int,
) -> EvalTask | None:
    target_values = np.array([to_float(row.get(target_col)) for row in table.rows], dtype=float)
    valid_target = np.isfinite(target_values)
    if int(valid_target.sum()) < min_samples:
        return None

    threshold = float(np.nanmedian(target_values[valid_target]))
    y_values = np.where(target_values > threshold, "high", "low")
    class_counts = pd.Series(y_values[valid_target]).value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < min_class_count:
        return None

    features: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for col in table.columns:
        group = table.groups.get(col, "metadata")
        if col == target_col or group not in feature_groups:
            continue
        series, kind = feature_value_series(table, col)
        if series is None:
            dropped.append(f"{col}:empty_or_constant")
            continue
        if kind == "categorical" and series.dropna().nunique() > max_category_cardinality:
            dropped.append(f"{col}:high_cardinality")
            continue
        features[col] = series

    if not features:
        return None

    X = pd.DataFrame(features).loc[valid_target].reset_index(drop=True)
    y = pd.Series(y_values[valid_target], name=target_col).reset_index(drop=True)
    class_counts = y.value_counts()
    if len(X) < min_samples or len(class_counts) != 2 or int(class_counts.min()) < min_class_count:
        return None
    if max_samples_per_task and len(X) > max_samples_per_task:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=max_samples_per_task,
            stratify=y,
            random_state=random_state,
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    task_id = f"{table.dataset}__{slugify(table.table)}__{slugify(target_col)}"
    return EvalTask(
        task_id=task_id,
        dataset=table.dataset,
        table=table.table,
        target=target_col,
        threshold=threshold,
        X=X,
        y=y,
        feature_groups_used=sorted(set(feature_groups)),
        dropped_feature_columns=dropped,
    )


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:96] or "task"


def make_tasks(args: argparse.Namespace) -> list[EvalTask]:
    tables = load_all_tables()
    feature_groups = list(DEFAULT_FEATURE_GROUPS)
    if args.include_electrochem_features:
        feature_groups.append("electrochem")
    feature_groups_tuple = tuple(feature_groups)

    selected_datasets = set(args.dataset or [])
    selected_tasks = set(args.task or [])
    tasks: list[EvalTask] = []
    for table in tables:
        if selected_datasets and table.dataset not in selected_datasets:
            continue
        target_cols = choose_target_columns(table, args.target_mode, args.min_samples, args.min_class_count)
        for target_col in target_cols:
            task = build_task(
                table,
                target_col,
                feature_groups=feature_groups_tuple,
                max_category_cardinality=args.max_category_cardinality,
                min_samples=args.min_samples,
                min_class_count=args.min_class_count,
                max_samples_per_task=args.max_samples_per_task,
                random_state=args.random_state,
            )
            if task is None:
                continue
            if selected_tasks and task.task_id not in selected_tasks:
                continue
            tasks.append(task)
    return tasks


def make_tabicl_classifier(
    *,
    model_path: str | None,
    checkpoint_version: str,
    device: str,
    n_estimators: int,
    random_state: int,
    allow_auto_download: bool,
) -> TabICLClassifier:
    kwargs: dict[str, Any] = {
        "device": device,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "allow_auto_download": allow_auto_download,
    }
    if model_path is None:
        kwargs["checkpoint_version"] = checkpoint_version
    else:
        kwargs["model_path"] = model_path
    return TabICLClassifier(**kwargs)


def make_tabpfn_classifier(device: str, random_state: int) -> Any:
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError("tabpfn is not installed in this environment") from exc

    try:
        return TabPFNClassifier(device=device, random_state=random_state)
    except TypeError:
        try:
            return TabPFNClassifier(device=device)
        except TypeError:
            return TabPFNClassifier()


def positive_probability(estimator: Any, X: pd.DataFrame, positive_label: str = "high") -> np.ndarray:
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = [str(cls) for cls in estimator.classes_]
    if positive_label not in classes:
        raise ValueError(f"Positive label {positive_label!r} not found in classes {classes}")
    return proba[:, classes.index(positive_label)]


def evaluate_estimator(
    *,
    model_label: str,
    model_kind: str,
    estimator_factory: Any,
    task: EvalTask,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        task.X,
        task.y,
        test_size=test_size,
        stratify=task.y,
        random_state=random_state,
    )
    estimator = estimator_factory()
    estimator.fit(X_train, y_train)
    y_pred = pd.Series(estimator.predict(X_test)).astype(str)

    y_score = None
    auroc = math.nan
    if hasattr(estimator, "predict_proba"):
        y_score = positive_probability(estimator, X_test, positive_label="high")
        if np.isfinite(y_score).all():
            y_true_binary = (y_test.astype(str).to_numpy() == "high").astype(int)
            auroc = float(roc_auc_score(y_true_binary, y_score))

    model_source = getattr(estimator, "model_path_", "")
    return {
        "model": model_label,
        "model_kind": model_kind,
        "model_source": str(model_source),
        "task_id": task.task_id,
        "dataset": task.dataset,
        "table": task.table,
        "target": task.target,
        "target_threshold_median": task.threshold,
        "n_samples": int(len(task.X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(task.X.shape[1]),
        "positive_label": "high",
        "positive_rate": float((task.y == "high").mean()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_mcc": float(matthews_corrcoef(y_test, y_pred)),
        "test_auroc": auroc,
        "feature_groups": ",".join(task.feature_groups_used),
        "dropped_feature_columns": ";".join(task.dropped_feature_columns),
    }


def default_output_paths(model_label: str, checkpoint_path: Path | None) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_part = checkpoint_path.stem if checkpoint_path is not None else "pretrained"
    base = f"corrosion_eval_{slugify(model_label)}_{slugify(ckpt_part)}_{stamp}"
    return DEFAULT_OUTPUT_DIR / f"{base}.json", DEFAULT_OUTPUT_DIR / f"{base}.csv"


def print_task_list(tasks: list[EvalTask]) -> None:
    rows = [
        {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "table": task.table,
            "target": task.target,
            "n_samples": len(task.X),
            "n_features": task.X.shape[1],
            "positive_rate": float((task.y == "high").mean()),
            "threshold": task.threshold,
        }
        for task in tasks
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No tasks generated.")
    else:
        print(df.to_string(index=False, max_colwidth=80))


def main() -> None:
    args = parse_args()
    tasks = make_tasks(args)
    if args.list_tasks:
        print_task_list(tasks)
        return

    checkpoint_path = resolve_checkpoint(args)
    local_label = args.local_model_label
    if local_label is None:
        if args.run:
            local_label = args.run if args.run.startswith("v") else slugify(args.run)
        else:
            local_label = checkpoint_path.parent.name

    if not tasks:
        raise RuntimeError("No corrosion evaluation tasks were generated. Try --target-mode all or lower --min-samples.")

    jobs: list[dict[str, Any]] = [
        {
            "model_label": local_label,
            "model_kind": "local_tabicl",
            "factory": lambda: make_tabicl_classifier(
                model_path=str(checkpoint_path),
                checkpoint_version=args.pretrained_checkpoint_version,
                device=args.device,
                n_estimators=args.n_estimators,
                random_state=args.random_state,
                allow_auto_download=False,
            ),
        }
    ]

    if args.compare_pretrained_tabicl:
        jobs.append(
            {
                "model_label": "pretrained_tabicl_v2",
                "model_kind": "pretrained_tabicl",
                "factory": lambda: make_tabicl_classifier(
                    model_path=None,
                    checkpoint_version=args.pretrained_checkpoint_version,
                    device=args.device,
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    allow_auto_download=args.baseline_auto_download,
                ),
            }
        )

    if args.compare_tabpfn:
        jobs.append(
            {
                "model_label": "pretrained_tabpfn",
                "model_kind": "pretrained_tabpfn",
                "factory": lambda: make_tabpfn_classifier(args.device, args.random_state),
            }
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        print(f"Task {task.task_id}: n={len(task.X)}, features={task.X.shape[1]}, target={task.target!r}")
        for job in jobs:
            try:
                row = evaluate_estimator(
                    model_label=job["model_label"],
                    model_kind=job["model_kind"],
                    estimator_factory=job["factory"],
                    task=task,
                    test_size=args.test_size,
                    random_state=args.random_state,
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True))
            except Exception as exc:
                error = {
                    "model": job["model_label"],
                    "model_kind": job["model_kind"],
                    "task_id": task.task_id,
                    "dataset": task.dataset,
                    "target": task.target,
                    "error": repr(exc),
                }
                errors.append(error)
                print(json.dumps(error, sort_keys=True))

    output_json, output_csv = default_output_paths(local_label, checkpoint_path)
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "local_checkpoint": str(checkpoint_path),
        "target_mode": args.target_mode,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_samples_per_task": args.max_samples_per_task,
        "n_estimators": args.n_estimators,
        "feature_groups_default": list(DEFAULT_FEATURE_GROUPS),
        "include_electrochem_features": args.include_electrochem_features,
        "rows": rows,
        "errors": errors,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print()
    if rows:
        summary_cols = [
            "model",
            "task_id",
            "n_samples",
            "n_features",
            "test_balanced_accuracy",
            "test_mcc",
            "test_auroc",
        ]
        print(pd.DataFrame(rows)[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved JSON results to {output_json}")
    print(f"Saved CSV results to {output_csv}")
    if errors:
        print(f"Completed with {len(errors)} task/model errors; see JSON for details.")


if __name__ == "__main__":
    main()
