#!/usr/bin/env python
"""Evaluate TabICL checkpoints on external corrosion benchmark tasks.

The tasks are built from the corrosion_datasets workspace. Continuous corrosion
targets are evaluated as native regression tasks by default, with optional
median-binary or quantile-multiclass binning retained for classifier
checkpoints.

This is an evaluation benchmark only. Do not use these results to re-select the
informed-prior values that were derived from the same external datasets.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier, TabICLRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "scripts"
if str(ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_DIR))

from analyze_structure import Table, clean_name, load_all_tables  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "eval_results"
DEFAULT_FEATURE_GROUPS = (
    "material",
    "environment",
    "process_history",
    "exposure_duration",
    "temporal_history",
    "direct_intervention",
    "molecular_descriptor",
)
ELECTROCHEM_FEATURE_GROUPS = ("electrochem_control", "electrochem_downstream")
EVAL_NA_STRINGS = {"", "na", "n/a", "nan", "none", "null", "-", "--"}
EVAL_RATING_TO_SEVERITY = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}
EVAL_UNIT_SUFFIX_RE = (
    r"^[\s,;/()°%+\-.]*"
    r"(?:m|mol|molar|wt|vol|ppm|ppb|ppt|c|f|k|d|day|days|h|hr|hrs|hour|hours|min|s|sec|"
    r"v|mv|a|ma|ua|µa|ohm|cm2|cm|mm|yr|year|years|max|min|approx|approximately|about)*"
    r"[\s,;/()°%+\-.]*$"
)
METRIC_COLUMNS = (
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_spearman",
    "test_pearson",
    "test_nmae_iqr",
    "test_nrmse_iqr",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_f1_macro",
    "test_mcc",
    "test_auroc",
    "test_roc_auc_ovr_macro",
    "test_ordinal_mae",
    "test_ordinal_rmse",
    "test_adjacent_accuracy",
    "test_quadratic_weighted_kappa",
    "test_spearman_pred_class",
    "test_expected_class_mae",
    "test_expected_class_spearman",
)
PRIMARY_METRIC = "test_balanced_accuracy"
ORDINAL_PRIMARY_METRIC = "test_quadratic_weighted_kappa"
REGRESSION_PRIMARY_METRIC = "test_spearman"
LOWER_IS_BETTER_METRICS = {
    "test_mae",
    "test_rmse",
    "test_nmae_iqr",
    "test_nrmse_iqr",
    "test_ordinal_mae",
    "test_ordinal_rmse",
    "test_expected_class_mae",
}
PRIMARY_TARGET_PATTERNS = (
    "corrosion rate",
    "inhibition efficiency",
    "efficiency",
    "pitting potential",
    "repassivation",
    "er.crev",
    "corrosion current",
    "epit",
    "ecorr",
    "ocp",
    "rate (mm/yr)",
    "rate",
)


@dataclass
class EvalTask:
    task_id: str
    dataset: str
    table: str
    target: str
    threshold: float
    target_binning: str
    target_bins: int
    bin_edges: list[float]
    class_labels: list[str]
    class_counts: dict[str, int]
    X: pd.DataFrame
    y: pd.Series
    y_ordinal: pd.Series
    task_family: str
    feature_groups_used: list[str]
    feature_group_counts: dict[str, int]
    dropped_feature_columns: list[str]
    quality_flags: list[str]


@dataclass(frozen=True)
class LocalModelSpec:
    label: str
    checkpoint_path: Path


@dataclass(frozen=True)
class CheckpointEvalSpec:
    checkpoint_name: str
    checkpoint_step: int | None
    local_specs: tuple[LocalModelSpec, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run suffix such as v15, or full checkpoint dir name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated run suffixes evaluated at the same --checkpoint, e.g. v12,v13,v14,v15.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help=(
            "Checkpoint name/step inside the run dir: latest, 3350, step-3350, "
            "step-3350.ckpt, or all for the step checkpoints common to all --run values."
        ),
    )
    parser.add_argument(
        "--min-checkpoint-step",
        type=int,
        default=1000,
        help="Minimum step to include when --checkpoint all is used.",
    )
    parser.add_argument(
        "--checkpoint-step-interval",
        type=int,
        default=1000,
        help=(
            "Only include common checkpoints whose step is divisible by this interval "
            "when --checkpoint all is used. Use 0 to include every common checkpoint."
        ),
    )
    parser.add_argument(
        "--local-ckpt-path",
        action="append",
        type=Path,
        default=[],
        help="Explicit local .ckpt path. Can be passed multiple times and is evaluated in addition to --run.",
    )
    parser.add_argument(
        "--local-model-label",
        action="append",
        type=str,
        default=[],
        help="Label for a local model. Pass once per local model when evaluating multiple local checkpoints.",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--run-prefix", type=str, default="tabicl_s1mini_generic_")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-mode", choices=("primary", "all"), default="primary")
    parser.add_argument(
        "--target-binning",
        choices=("continuous", "median_binary", "quantile_multiclass"),
        default="continuous",
        help=(
            "How continuous targets are evaluated. "
            "continuous keeps native regression targets; "
            "median_binary preserves the existing low/high median split; "
            "quantile_multiclass creates ordered quantile bins."
        ),
    )
    parser.add_argument(
        "--target-bins",
        type=int,
        default=2,
        help="Number of target bins when --target-binning quantile_multiclass is used.",
    )
    parser.add_argument(
        "--regression-output",
        choices=("mean", "median"),
        default="median",
        help="Point prediction extracted from TabICLRegressor for continuous-target evaluation.",
    )
    parser.add_argument("--dataset", action="append", help="Limit to dataset id. Can be passed multiple times.")
    parser.add_argument("--task", action="append", help="Limit to exact task id. Can be passed multiple times.")
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument(
        "--min-numeric-finite-ratio",
        type=float,
        default=0.80,
        help="Minimum finite numeric ratio required to keep a feature as numeric.",
    )
    parser.add_argument(
        "--min-categorical-nonmissing-ratio",
        type=float,
        default=0.80,
        help="Minimum non-missing ratio required to keep a feature as categorical.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=2000,
        help="Stratified cap per task before holdout split. Use 0 to disable.",
    )
    parser.add_argument("--max-category-cardinality", type=int, default=80)
    parser.add_argument(
        "--exclude-quality-flag",
        action="append",
        default=[],
        help="Skip generated tasks containing this quality flag. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-electrochem-features",
        action="store_true",
        help=(
            "Include electrochemical control/setpoint and downstream-response feature groups. "
            "By default these are excluded to keep evaluation leakage-safe."
        ),
    )
    parser.add_argument("--compare-pretrained-tabicl", dest="compare_pretrained_tabicl", action="store_true", default=True)
    parser.add_argument("--no-compare-pretrained-tabicl", dest="compare_pretrained_tabicl", action="store_false")
    parser.add_argument(
        "--pretrained-checkpoint-version",
        type=str,
        default=None,
        help=(
            "Optional pretrained checkpoint filename. Defaults to the TabICL regressor "
            "checkpoint for continuous evaluation and the classifier checkpoint for binned evaluation."
        ),
    )
    parser.add_argument("--baseline-auto-download", action="store_true", default=True)
    parser.add_argument("--no-baseline-auto-download", dest="baseline_auto_download", action="store_false")
    parser.add_argument("--compare-tabpfn", action="store_true", help="Try to evaluate TabPFNClassifier if tabpfn is installed.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None, help="Row-wise CSV with one row per task/model result.")
    parser.add_argument("--output-wide-csv", type=Path, default=None, help="Wide comparison CSV with one row per task.")
    parser.add_argument("--output-summary-csv", type=Path, default=None, help="Per-model aggregate summary CSV.")
    parser.add_argument(
        "--output-plot-dir",
        type=Path,
        default=None,
        help="Directory for SVG metric trend plots when --checkpoint all is used.",
    )
    parser.add_argument(
        "--plot-metric",
        action="append",
        choices=METRIC_COLUMNS,
        default=None,
        help="Metric to plot in --checkpoint all mode. Repeat to plot multiple metrics. Defaults to all metrics.",
    )
    parser.add_argument(
        "--no-checkpoint-plots",
        dest="checkpoint_plots",
        action="store_false",
        default=True,
        help="Disable SVG metric trend plots for --checkpoint all.",
    )
    parser.add_argument("--print-json-lines", action="store_true", help="Print each result/error row as JSONL while running.")
    parser.add_argument("--list-tasks", action="store_true", help="List generated tasks and exit before loading any model.")
    return parser.parse_args([arg for arg in sys.argv[1:] if arg != ""])


def validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint_step_interval < 0:
        raise ValueError("--checkpoint-step-interval must be >= 0.")
    if args.target_binning == "continuous":
        return
    if args.target_binning == "median_binary" and args.target_bins != 2:
        raise ValueError("--target-bins must be 2 when --target-binning median_binary is used.")
    if args.target_binning == "quantile_multiclass" and args.target_bins < 3:
        raise ValueError("--target-bins must be at least 3 when --target-binning quantile_multiclass is used.")


def primary_metric_for_args(args: argparse.Namespace) -> str:
    if args.target_binning == "continuous":
        return REGRESSION_PRIMARY_METRIC
    if args.target_binning == "quantile_multiclass":
        return ORDINAL_PRIMARY_METRIC
    return PRIMARY_METRIC


def metric_sort_specs_for_args(args: argparse.Namespace) -> list[tuple[str, bool]]:
    if args.target_binning == "continuous":
        return [
            (REGRESSION_PRIMARY_METRIC, False),
            ("test_nmae_iqr", True),
            ("test_rmse", True),
        ]
    if args.target_binning == "quantile_multiclass":
        return [
            (ORDINAL_PRIMARY_METRIC, False),
            ("test_ordinal_mae", True),
            (PRIMARY_METRIC, False),
        ]
    return [(PRIMARY_METRIC, False), ("test_mcc", False)]


def expand_runs(args: argparse.Namespace) -> list[str]:
    runs = list(args.run or [])
    if args.runs:
        runs.extend(run.strip() for run in args.runs.split(",") if run.strip())
    return runs


def resolve_run_checkpoint(run: str, args: argparse.Namespace) -> Path:
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


def resolve_run_dir(run: str, args: argparse.Namespace) -> Path:
    run_dir_name = run if run.startswith(args.run_prefix) else f"{args.run_prefix}{run}"
    run_dir = (args.checkpoint_root / run_dir_name).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")
    return run_dir


def checkpoint_step_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"step-(\d+)", path.stem)
    return int(match.group(1)) if match else None


def available_step_checkpoints(run: str, args: argparse.Namespace) -> dict[int, Path]:
    run_dir = resolve_run_dir(run, args)
    checkpoints: dict[int, Path] = {}
    for path in run_dir.glob("step-*.ckpt"):
        step = checkpoint_step_from_path(path)
        if step is not None:
            checkpoints[step] = path
    if not checkpoints:
        raise FileNotFoundError(f"No step-*.ckpt files found in {run_dir}")
    return checkpoints


def resolve_checkpoint_eval_specs(args: argparse.Namespace) -> list[CheckpointEvalSpec]:
    if args.checkpoint != "all":
        local_specs = tuple(resolve_local_model_specs(args))
        checkpoint_steps = {checkpoint_step_from_path(spec.checkpoint_path) for spec in local_specs}
        checkpoint_steps.discard(None)
        checkpoint_step = next(iter(checkpoint_steps)) if len(checkpoint_steps) == 1 else None
        checkpoint_name = (
            f"step-{checkpoint_step}" if checkpoint_step is not None else slugify(args.checkpoint or "mixed_checkpoints")
        )
        return [CheckpointEvalSpec(checkpoint_name=checkpoint_name, checkpoint_step=checkpoint_step, local_specs=local_specs)]

    runs = expand_runs(args)
    if not runs:
        raise ValueError("--checkpoint all requires one or more --run/--runs values.")
    if args.local_ckpt_path:
        raise ValueError("--checkpoint all works on run directories; do not combine it with --local-ckpt-path.")

    labels = list(args.local_model_label or [])
    if labels and len(labels) != len(runs):
        raise ValueError(
            "--local-model-label must be passed once per run when evaluating --checkpoint all; "
            f"got {len(labels)} labels for {len(runs)} runs."
        )

    by_run = {run: available_step_checkpoints(run, args) for run in runs}
    common_steps = sorted(
        step
        for step in set.intersection(*(set(paths) for paths in by_run.values()))
        if step >= args.min_checkpoint_step
        and (args.checkpoint_step_interval == 0 or step % args.checkpoint_step_interval == 0)
    )
    if not common_steps:
        run_list = ", ".join(runs)
        interval_note = (
            "any interval"
            if args.checkpoint_step_interval == 0
            else f"interval {args.checkpoint_step_interval}"
        )
        raise FileNotFoundError(
            f"No common step-*.ckpt checkpoints found across runs at or after "
            f"step {args.min_checkpoint_step} with {interval_note}: {run_list}"
        )

    specs: list[CheckpointEvalSpec] = []
    for step in common_steps:
        local_specs: list[LocalModelSpec] = []
        for index, run in enumerate(runs):
            label = labels[index] if labels else short_run_label(run, args.run_prefix)
            local_specs.append(LocalModelSpec(label=slugify(label), checkpoint_path=by_run[run][step]))
        specs.append(
            CheckpointEvalSpec(
                checkpoint_name=f"step-{step}",
                checkpoint_step=step,
                local_specs=tuple(local_specs),
            )
        )
    return specs


def short_run_label(run: str, run_prefix: str) -> str:
    if run.startswith(run_prefix):
        return run[len(run_prefix) :]
    return run


def checkpoint_label(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    if stem.startswith("step-"):
        return f"step{stem.split('-', 1)[1]}"
    return slugify(stem)


def default_local_ckpt_label(checkpoint_path: Path, run_prefix: str) -> str:
    run_label = short_run_label(checkpoint_path.parent.name, run_prefix)
    return f"{slugify(run_label)}_{checkpoint_label(checkpoint_path)}"


def resolve_local_model_specs(args: argparse.Namespace) -> list[LocalModelSpec]:
    runs = expand_runs(args)
    local_ckpt_paths = list(args.local_ckpt_path or [])
    if not runs and not local_ckpt_paths:
        raise ValueError("Provide --run v15, --runs v12,v13, or --local-ckpt-path.")

    labels = list(args.local_model_label or [])
    n_local_models = len(runs) + len(local_ckpt_paths)
    if labels and len(labels) != n_local_models:
        raise ValueError(
            "--local-model-label must be passed once per local model when evaluating "
            f"multiple models; got {len(labels)} labels for {n_local_models} local models."
        )

    specs: list[LocalModelSpec] = []
    for index, run in enumerate(runs):
        checkpoint_path = resolve_run_checkpoint(run, args)
        label = labels[index] if labels else short_run_label(run, args.run_prefix)
        specs.append(LocalModelSpec(label=slugify(label), checkpoint_path=checkpoint_path))

    for offset, local_ckpt_path in enumerate(local_ckpt_paths):
        path = local_ckpt_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local checkpoint not found: {path}")
        label_index = len(runs) + offset
        label = labels[label_index] if labels else default_local_ckpt_label(path, args.run_prefix)
        specs.append(LocalModelSpec(label=slugify(label), checkpoint_path=path))

    seen: set[str] = set()
    duplicates: set[str] = set()
    for spec in specs:
        if spec.label in seen:
            duplicates.add(spec.label)
        seen.add(spec.label)
    if duplicates:
        dupes = ", ".join(sorted(duplicates))
        raise ValueError(f"Local model labels must be unique; duplicate labels: {dupes}")

    return specs


def normalize_eval_numeric_text(text: str) -> str:
    text = text.strip().replace("−", "-").replace("–", "-").replace("—", "-")
    text = "".join(ch for ch in text if ch not in {"\u202a", "\u202b", "\u202c", "\u202d", "\u202e"})
    return re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)


def eval_to_float(value: Any, *, column: str = "", group: str = "") -> float:
    """Parse scalar corrosion eval values without depending on analysis heuristics."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = normalize_eval_numeric_text(clean_name(value))
    if text.lower() in EVAL_NA_STRINGS:
        return math.nan

    lower = text.lower()
    if group == "target" and "rating" in column.lower() and lower and lower[0] in EVAL_RATING_TO_SEVERITY:
        if re.match(r"^[abcd]\b", lower):
            return EVAL_RATING_TO_SEVERITY[lower[0]]

    range_match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*$", text)
    if range_match:
        return (float(range_match.group(1)) + float(range_match.group(2))) / 2.0

    scalar_match = re.match(r"^\s*[<>=~]*\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", text)
    if scalar_match:
        return float(scalar_match.group(1))

    if not re.match(r"^\s*[<>=~]*\s*[+-]?\d", text):
        return math.nan
    found = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(found) == 1:
        tail = text[text.find(found[0]) + len(found[0]) :]
        if re.match(EVAL_UNIT_SUFFIX_RE, tail, flags=re.IGNORECASE):
            return float(found[0])
    return math.nan


def finite_target_values(table: Table, target_col: str) -> np.ndarray:
    values = np.array(
        [eval_to_float(row.get(target_col), column=target_col, group="target") for row in table.rows],
        dtype=float,
    )
    return values[np.isfinite(values)]


def make_target_bins(
    values: np.ndarray,
    *,
    target_binning: str,
    target_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]] | None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        return None

    if target_binning == "median_binary":
        threshold = float(np.nanmedian(values))
        ordinals = (values > threshold).astype(int)
        labels = np.where(ordinals == 1, "high", "low")
        return labels, ordinals, [threshold], ["low", "high"]

    if target_binning != "quantile_multiclass":
        raise ValueError(f"Unknown target binning mode: {target_binning}")

    if target_bins < 3:
        raise ValueError("--target-bins must be at least 3 for quantile_multiclass.")

    quantiles = [i / target_bins for i in range(1, target_bins)]
    edges = np.quantile(values, quantiles)
    edges = np.asarray(edges, dtype=float)
    unique_edges = np.unique(edges[np.isfinite(edges)])
    if len(unique_edges) != target_bins - 1:
        return None

    ordinals = np.digitize(values, unique_edges, right=True).astype(int)
    if len(np.unique(ordinals)) != target_bins:
        return None
    class_labels = [f"bin_{idx}" for idx in range(target_bins)]
    labels = np.asarray([class_labels[idx] for idx in ordinals], dtype=object)
    return labels, ordinals, [float(edge) for edge in unique_edges], class_labels


def choose_target_columns(
    table: Table,
    mode: str,
    min_samples: int,
    min_class_count: int,
    *,
    target_binning: str,
    target_bins: int,
) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for col in table.columns:
        if table.groups.get(col) != "target":
            continue
        values = finite_target_values(table, col)
        if len(values) < min_samples:
            continue
        if target_binning != "continuous":
            binned = make_target_bins(values, target_binning=target_binning, target_bins=target_bins)
            if binned is None:
                continue
            _, ordinals, _, class_labels = binned
            class_counts = np.bincount(ordinals, minlength=len(class_labels))
            if len(class_counts) != len(class_labels) or int(class_counts.min()) < min_class_count:
                continue
        lower_name = col.lower()
        priority = next((i for i, pattern in enumerate(PRIMARY_TARGET_PATTERNS) if pattern in lower_name), 99)
        candidates.append((priority, -len(values), col))

    candidates.sort()
    cols = [col for _, _, col in candidates]
    if mode == "primary":
        return cols[:1]
    return cols


def feature_value_series(
    table: Table,
    column: str,
    *,
    min_numeric_finite_ratio: float,
    min_categorical_nonmissing_ratio: float,
) -> tuple[pd.Series | None, str, str]:
    group = table.groups.get(column, "metadata")
    raw = [row.get(column) for row in table.rows]
    numeric = np.array([eval_to_float(value, column=column, group=group) for value in raw], dtype=float)
    finite_ratio = float(np.isfinite(numeric).mean()) if len(numeric) else 0.0
    if finite_ratio >= min_numeric_finite_ratio and np.nanstd(numeric) > 0:
        return pd.Series(numeric, name=column), "numeric", ""

    text = pd.Series(["" if value is None else str(value).strip() for value in raw], name=column)
    text = text.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "None": pd.NA})
    nonmissing = int(text.notna().sum())
    nonmissing_ratio = float(nonmissing / len(text)) if len(text) else 0.0
    if nonmissing == 0:
        return None, "empty_or_constant", "empty_or_constant"

    finite_count = int(np.isfinite(numeric).sum())
    if 0 < finite_count and finite_ratio < min_numeric_finite_ratio:
        numeric_like_ratio = float(finite_count / nonmissing)
        if numeric_like_ratio >= 0.80:
            return None, "numeric_sparse", f"numeric_sparse_{finite_ratio:.2f}"

    if nonmissing_ratio < min_categorical_nonmissing_ratio:
        return None, "categorical_sparse", f"categorical_sparse_{nonmissing_ratio:.2f}"

    if text.dropna().nunique() >= 2:
        return text, "categorical", ""
    return None, "empty_or_constant", "empty_or_constant"


def assess_task_quality(table: Table, X: pd.DataFrame, y: pd.Series) -> list[str]:
    flags: list[str] = []
    n_samples, n_features = X.shape
    if n_samples < 80:
        flags.append("small_n")
    if n_features <= 2:
        flags.append("very_few_features")

    pattern_count = int(pd.DataFrame(X.astype(str).fillna("<NA>")).drop_duplicates().shape[0])
    if n_samples and pattern_count / n_samples < 0.10:
        flags.append("few_unique_feature_patterns")

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            cardinality_ratio = float(X[col].dropna().nunique() / n_samples) if n_samples else 0.0
            if cardinality_ratio > 0.25:
                flags.append("high_cardinality_categorical_features")
                break

    if table.dataset == "316l_pitting_passivity":
        flags.append("fixed_material_low_condition_grid")
    elif table.dataset == "mooring_steel_seawater":
        flags.append("small_time_series_condition_table")
    elif table.dataset == "nace_nist_corr_data":
        flags.append("coarse_heterogeneous_corpus")
    elif table.dataset == "am_mpea_corrosion":
        flags.append("small_identity_like_material_table")
    elif table.dataset == "datacor_aluminum_inhibitors":
        flags.append("small_inhibitor_descriptor_table")
    elif table.dataset == "datacortech_aluminum_inhibitors":
        flags.append("literature_compiled_inhibitor_efficiency")
    elif table.dataset in {"mg_az91_inhibitors", "mg_ze41_inhibitors"}:
        flags.append("small_descriptor_heavy_inhibitor_table")
    elif table.dataset == "ni_crevice_repassivation":
        flags.append("docx_extracted_condition_table")

    return sorted(set(flags))


def regression_stratify_labels(values: np.ndarray, max_bins: int = 10) -> np.ndarray | None:
    """Create coarse quantile labels only for preserving target coverage in splits."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        return None

    n_bins = min(max_bins, max(2, len(values) // 10))
    for bins in range(n_bins, 1, -1):
        edges = np.quantile(values, [i / bins for i in range(1, bins)])
        edges = np.unique(edges[np.isfinite(edges)])
        if len(edges) != bins - 1:
            continue
        labels = np.digitize(values, edges, right=True).astype(int)
        counts = np.bincount(labels, minlength=bins)
        if len(counts) == bins and int(counts.min()) >= 2:
            return labels
    return None


def task_family_for_dataset(dataset: str) -> str:
    if "inhibitor" in dataset:
        return "inhibitor_agent"
    if dataset == "mooring_steel_seawater":
        return "time_series_corrosion"
    if dataset == "nace_nist_corr_data":
        return "coarse_corpus"
    return "normal_corrosion"


def build_task(
    table: Table,
    target_col: str,
    *,
    target_binning: str,
    target_bins: int,
    feature_groups: tuple[str, ...],
    max_category_cardinality: int,
    min_numeric_finite_ratio: float,
    min_categorical_nonmissing_ratio: float,
    min_samples: int,
    min_class_count: int,
    max_samples_per_task: int,
    random_state: int,
) -> EvalTask | None:
    target_values = np.array(
        [eval_to_float(row.get(target_col), column=target_col, group="target") for row in table.rows],
        dtype=float,
    )
    valid_target = np.isfinite(target_values)
    if int(valid_target.sum()) < min_samples:
        return None

    finite_targets = target_values[valid_target]
    if target_binning == "continuous":
        y_values = finite_targets.astype(float)
        y_ordinals = finite_targets.astype(float)
        bin_edges: list[float] = []
        class_labels: list[str] = []
        class_counts = pd.Series(dtype=int)
    else:
        binned = make_target_bins(
            finite_targets,
            target_binning=target_binning,
            target_bins=target_bins,
        )
        if binned is None:
            return None
        y_values, y_ordinals, bin_edges, class_labels = binned
        class_counts = pd.Series(y_values).value_counts()
        if len(class_counts) != len(class_labels) or int(class_counts.min()) < min_class_count:
            return None

    features: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for col in table.columns:
        group = table.groups.get(col, "metadata")
        if col == target_col or group not in feature_groups:
            continue
        series, kind, drop_reason = feature_value_series(
            table,
            col,
            min_numeric_finite_ratio=min_numeric_finite_ratio,
            min_categorical_nonmissing_ratio=min_categorical_nonmissing_ratio,
        )
        if series is None:
            dropped.append(f"{col}:{drop_reason or kind}")
            continue
        if kind == "categorical" and series.dropna().nunique() > max_category_cardinality:
            dropped.append(f"{col}:high_cardinality")
            continue
        features[col] = series

    if not features:
        return None

    X = pd.DataFrame(features).loc[valid_target].reset_index(drop=True)
    y = pd.Series(y_values, name=target_col).reset_index(drop=True)
    y_ordinal = pd.Series(y_ordinals, name=f"{target_col}__ordinal").reset_index(drop=True)
    class_counts = y.value_counts()
    if len(X) < min_samples:
        return None
    if target_binning != "continuous":
        if len(class_counts) != len(class_labels) or int(class_counts.min()) < min_class_count:
            return None
    if max_samples_per_task and len(X) > max_samples_per_task:
        stratify = regression_stratify_labels(y.to_numpy(dtype=float)) if target_binning == "continuous" else y
        X, _, y, _, y_ordinal, _ = train_test_split(
            X,
            y,
            y_ordinal,
            train_size=max_samples_per_task,
            stratify=stratify,
            random_state=random_state,
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        y_ordinal = y_ordinal.reset_index(drop=True)
        class_counts = y.value_counts()
        if target_binning != "continuous" and (
            len(class_counts) != len(class_labels) or int(class_counts.min()) < min_class_count
        ):
            return None

    task_id = f"{table.dataset}__{slugify(table.table)}__{slugify(target_col)}"
    quality_flags = assess_task_quality(table, X, y)
    feature_group_counts = Counter(table.groups.get(col, "metadata") for col in X.columns)
    ordered_class_counts = {
        label: int(class_counts.get(label, 0))
        for label in class_labels
    }
    return EvalTask(
        task_id=task_id,
        dataset=table.dataset,
        table=table.table,
        target=target_col,
        threshold=float(bin_edges[0]) if len(bin_edges) == 1 else math.nan,
        target_binning=target_binning,
        target_bins=len(class_labels) if class_labels else 0,
        bin_edges=bin_edges,
        class_labels=class_labels,
        class_counts=ordered_class_counts if target_binning != "continuous" else {},
        X=X,
        y=y,
        y_ordinal=y_ordinal,
        task_family=task_family_for_dataset(table.dataset),
        feature_groups_used=sorted(set(feature_groups)),
        feature_group_counts=dict(sorted(feature_group_counts.items())),
        dropped_feature_columns=dropped,
        quality_flags=quality_flags,
    )


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:96] or "task"


def make_tasks(args: argparse.Namespace) -> list[EvalTask]:
    tables = load_all_tables()
    feature_groups = list(DEFAULT_FEATURE_GROUPS)
    if args.include_electrochem_features:
        feature_groups.extend(ELECTROCHEM_FEATURE_GROUPS)
    feature_groups_tuple = tuple(feature_groups)

    selected_datasets = set(args.dataset or [])
    selected_tasks = set(args.task or [])
    excluded_quality_flags = set(args.exclude_quality_flag or [])
    tasks: list[EvalTask] = []
    for table in tables:
        if selected_datasets and table.dataset not in selected_datasets:
            continue
        target_cols = choose_target_columns(
            table,
            args.target_mode,
            args.min_samples,
            args.min_class_count,
            target_binning=args.target_binning,
            target_bins=args.target_bins,
        )
        for target_col in target_cols:
            task = build_task(
                table,
                target_col,
                target_binning=args.target_binning,
                target_bins=args.target_bins,
                feature_groups=feature_groups_tuple,
                max_category_cardinality=args.max_category_cardinality,
                min_numeric_finite_ratio=args.min_numeric_finite_ratio,
                min_categorical_nonmissing_ratio=args.min_categorical_nonmissing_ratio,
                min_samples=args.min_samples,
                min_class_count=args.min_class_count,
                max_samples_per_task=args.max_samples_per_task,
                random_state=args.random_state,
            )
            if task is None:
                continue
            if excluded_quality_flags and set(task.quality_flags).intersection(excluded_quality_flags):
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


def make_tabicl_regressor(
    *,
    model_path: str | None,
    checkpoint_version: str,
    device: str,
    n_estimators: int,
    random_state: int,
    allow_auto_download: bool,
) -> TabICLRegressor:
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
    return TabICLRegressor(**kwargs)


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


def make_tabpfn_regressor(device: str, random_state: int) -> Any:
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:
        raise RuntimeError("tabpfn is not installed in this environment") from exc

    try:
        return TabPFNRegressor(device=device, random_state=random_state)
    except TypeError:
        try:
            return TabPFNRegressor(device=device)
        except TypeError:
            return TabPFNRegressor()


def positive_probability(estimator: Any, X: pd.DataFrame, positive_label: str = "high") -> np.ndarray:
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = [str(cls) for cls in estimator.classes_]
    if positive_label not in classes:
        raise ValueError(f"Positive label {positive_label!r} not found in classes {classes}")
    return proba[:, classes.index(positive_label)]


def class_to_ordinal(task: EvalTask) -> dict[str, int]:
    return {label: index for index, label in enumerate(task.class_labels)}


def labels_to_ordinals(labels: pd.Series | np.ndarray, task: EvalTask) -> np.ndarray:
    mapping = class_to_ordinal(task)
    values = [mapping.get(str(label), math.nan) for label in labels]
    return np.asarray(values, dtype=float)


def spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return math.nan
    if len(np.unique(y_true[np.isfinite(y_true)])) < 2 or len(np.unique(y_pred[np.isfinite(y_pred)])) < 2:
        return math.nan
    value = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(value) if value is not None and np.isfinite(value) else math.nan


def pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return math.nan
    if len(np.unique(y_true[np.isfinite(y_true)])) < 2 or len(np.unique(y_pred[np.isfinite(y_pred)])) < 2:
        return math.nan
    value = pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")
    return float(value) if value is not None and np.isfinite(value) else math.nan


def aligned_probability_matrix(estimator: Any, X: pd.DataFrame, task: EvalTask) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    if not np.isfinite(proba).all():
        return None
    estimator_classes = [str(cls) for cls in estimator.classes_]
    if any(label not in estimator_classes for label in task.class_labels):
        return None
    aligned = np.zeros((proba.shape[0], len(task.class_labels)), dtype=float)
    for index, label in enumerate(task.class_labels):
        aligned[:, index] = proba[:, estimator_classes.index(label)]
    return aligned


def ordinal_metrics(
    *,
    y_true_ord: np.ndarray,
    y_pred_ord: np.ndarray,
    proba_aligned: np.ndarray | None,
    n_classes: int,
) -> dict[str, float]:
    valid = np.isfinite(y_true_ord) & np.isfinite(y_pred_ord)
    if not valid.any():
        return {
            "test_ordinal_mae": math.nan,
            "test_ordinal_rmse": math.nan,
            "test_adjacent_accuracy": math.nan,
            "test_quadratic_weighted_kappa": math.nan,
            "test_spearman_pred_class": math.nan,
            "test_expected_class_mae": math.nan,
            "test_expected_class_spearman": math.nan,
        }

    true_ord = y_true_ord[valid].astype(int)
    pred_ord = y_pred_ord[valid].astype(int)
    diff = pred_ord - true_ord
    metrics = {
        "test_ordinal_mae": float(np.mean(np.abs(diff))),
        "test_ordinal_rmse": float(np.sqrt(np.mean(diff.astype(float) ** 2))),
        "test_adjacent_accuracy": float(np.mean(np.abs(diff) <= 1)),
        "test_quadratic_weighted_kappa": float(
            cohen_kappa_score(true_ord, pred_ord, labels=list(range(n_classes)), weights="quadratic")
        ),
        "test_spearman_pred_class": spearman_safe(true_ord.astype(float), pred_ord.astype(float)),
        "test_expected_class_mae": math.nan,
        "test_expected_class_spearman": math.nan,
    }

    if proba_aligned is not None and len(proba_aligned) == len(y_true_ord):
        ordinal_axis = np.arange(n_classes, dtype=float)
        expected = proba_aligned @ ordinal_axis
        expected_valid = np.isfinite(y_true_ord) & np.isfinite(expected)
        if expected_valid.any():
            expected_true = y_true_ord[expected_valid].astype(float)
            expected_pred = expected[expected_valid].astype(float)
            metrics["test_expected_class_mae"] = float(np.mean(np.abs(expected_pred - expected_true)))
            metrics["test_expected_class_spearman"] = spearman_safe(expected_true, expected_pred)

    return metrics


def evaluate_estimator(
    *,
    model_label: str,
    model_kind: str,
    estimator_factory: Any,
    task: EvalTask,
    test_size: float,
    random_state: int,
    regression_output: str = "median",
) -> dict[str, Any]:
    if task.target_binning == "continuous":
        y_values = task.y.astype(float)
        stratify = regression_stratify_labels(y_values.to_numpy(dtype=float))
        X_train, X_test, y_train, y_test = train_test_split(
            task.X,
            y_values,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
        estimator = estimator_factory()
        estimator.fit(X_train, y_train)
        y_pred = np.asarray(estimator.predict(X_test, output_type=regression_output), dtype=float)
        y_true = y_test.to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        target_iqr = float(np.subtract(*np.percentile(y_train.to_numpy(dtype=float), [75, 25])))
        nmae_iqr = mae / target_iqr if target_iqr > 0 else math.nan
        nrmse_iqr = rmse / target_iqr if target_iqr > 0 else math.nan

        model_source = getattr(estimator, "model_path_", "")
        return {
            "model": model_label,
            "model_kind": model_kind,
            "model_source": str(model_source),
            "task_id": task.task_id,
            "dataset": task.dataset,
            "table": task.table,
            "target": task.target,
            "target_threshold_median": math.nan,
            "target_binning": task.target_binning,
            "target_bins": 0,
            "target_bin_edges": "",
            "regression_output": regression_output,
            "n_samples": int(len(task.X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(task.X.shape[1]),
            "n_classes": 0,
            "class_labels": "",
            "class_counts": "",
            "positive_label": "",
            "positive_rate": math.nan,
            "task_quality_flags": ",".join(task.quality_flags),
            "task_family": task.task_family,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": float(r2_score(y_true, y_pred)),
            "test_spearman": spearman_safe(y_true, y_pred),
            "test_pearson": pearson_safe(y_true, y_pred),
            "test_nmae_iqr": nmae_iqr,
            "test_nrmse_iqr": nrmse_iqr,
            "feature_groups": ",".join(task.feature_groups_used),
            "feature_group_counts": json.dumps(task.feature_group_counts, sort_keys=True),
            "dropped_feature_columns": ";".join(task.dropped_feature_columns),
        }

    X_train, X_test, y_train, y_test, _, y_test_ordinal = train_test_split(
        task.X,
        task.y,
        task.y_ordinal,
        test_size=test_size,
        stratify=task.y,
        random_state=random_state,
    )
    estimator = estimator_factory()
    estimator.fit(X_train, y_train)
    y_pred = pd.Series(estimator.predict(X_test)).astype(str)
    y_true_ord = y_test_ordinal.to_numpy(dtype=float)
    y_pred_ord = labels_to_ordinals(y_pred, task)

    proba_aligned = aligned_probability_matrix(estimator, X_test, task)
    auroc = math.nan
    roc_auc_ovr_macro = math.nan
    if proba_aligned is not None:
        if len(task.class_labels) == 2:
            y_score = proba_aligned[:, 1]
            y_true_binary = (y_test.astype(str).to_numpy() == task.class_labels[1]).astype(int)
            auroc = float(roc_auc_score(y_true_binary, y_score))
            roc_auc_ovr_macro = auroc
        else:
            y_true_ord_int = y_true_ord.astype(int)
            roc_auc_ovr_macro = float(
                roc_auc_score(
                    y_true_ord_int,
                    proba_aligned,
                    labels=list(range(len(task.class_labels))),
                    multi_class="ovr",
                    average="macro",
                )
            )
            auroc = roc_auc_ovr_macro

    ord_metrics = ordinal_metrics(
        y_true_ord=y_true_ord,
        y_pred_ord=y_pred_ord,
        proba_aligned=proba_aligned,
        n_classes=len(task.class_labels),
    )

    model_source = getattr(estimator, "model_path_", "")
    row = {
        "model": model_label,
        "model_kind": model_kind,
        "model_source": str(model_source),
        "task_id": task.task_id,
        "dataset": task.dataset,
        "table": task.table,
        "target": task.target,
        "target_threshold_median": task.threshold,
        "target_binning": task.target_binning,
        "target_bins": task.target_bins,
        "target_bin_edges": ",".join(format_metric(edge) for edge in task.bin_edges),
        "n_samples": int(len(task.X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(task.X.shape[1]),
        "n_classes": int(len(task.class_labels)),
        "class_labels": ",".join(task.class_labels),
        "class_counts": ",".join(f"{label}:{task.class_counts.get(label, 0)}" for label in task.class_labels),
        "positive_label": task.class_labels[-1],
        "positive_rate": float((task.y == task.class_labels[-1]).mean()),
        "task_quality_flags": ",".join(task.quality_flags),
        "task_family": task.task_family,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_mcc": float(matthews_corrcoef(y_test, y_pred)),
        "test_auroc": auroc,
        "test_roc_auc_ovr_macro": roc_auc_ovr_macro,
        "feature_groups": ",".join(task.feature_groups_used),
        "feature_group_counts": json.dumps(task.feature_group_counts, sort_keys=True),
        "dropped_feature_columns": ";".join(task.dropped_feature_columns),
    }
    row.update(ord_metrics)
    return row


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def default_output_paths(local_specs: list[LocalModelSpec] | tuple[LocalModelSpec, ...], checkpoint_part: str | None = None) -> tuple[Path, Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_labels = [spec.label for spec in local_specs]
    checkpoint_stems = {spec.checkpoint_path.stem for spec in local_specs}
    if len(model_labels) == 1:
        model_part = slugify(model_labels[0])
    else:
        shown_labels = "_".join(slugify(label) for label in model_labels[:4])
        if len(model_labels) > 4:
            shown_labels = f"{shown_labels}_plus{len(model_labels) - 4}"
        model_part = f"compare_{shown_labels}"
    ckpt_part = checkpoint_part or (next(iter(checkpoint_stems)) if len(checkpoint_stems) == 1 else "mixed_checkpoints")
    base = f"corrosion_eval_{model_part}_{slugify(ckpt_part)}_{stamp}"
    return (
        DEFAULT_OUTPUT_DIR / f"{base}.json",
        DEFAULT_OUTPUT_DIR / f"{base}.csv",
        DEFAULT_OUTPUT_DIR / f"{base}_wide.csv",
        DEFAULT_OUTPUT_DIR / f"{base}_summary.csv",
    )


def derived_csv_path(output_csv: Path, suffix: str) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_{suffix}{output_csv.suffix}")


def default_plot_dir(args: argparse.Namespace, output_csv: Path) -> Path:
    if args.target_binning == "continuous":
        return output_csv.with_name("corrosion_eval_compare_regression")
    return output_csv.with_name(f"corrosion_eval_compare_bins{args.target_bins}")


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def format_metric(value: Any) -> str:
    numeric = to_float_or_nan(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def make_wide_results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    model_order = ordered_unique([str(model) for model in df["model"].tolist()])
    group_cols = ["task_id"]
    if "checkpoint_name" in df.columns:
        group_cols = ["checkpoint_name", "checkpoint_step", "task_id"]
    meta_cols = [
        "checkpoint_name",
        "checkpoint_step",
        "task_id",
        "dataset",
        "table",
        "target",
        "target_threshold_median",
        "target_binning",
        "target_bins",
        "target_bin_edges",
        "regression_output",
        "n_samples",
        "n_train",
        "n_test",
        "n_features",
        "n_classes",
        "class_labels",
        "class_counts",
        "positive_rate",
        "task_quality_flags",
    ]
    records: list[dict[str, Any]] = []
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        first = group.iloc[0]
        record = {col: first[col] for col in meta_cols if col in group.columns}
        for model in model_order:
            model_rows = group[group["model"] == model]
            if model_rows.empty:
                continue
            row = model_rows.iloc[0]
            model_key = slugify(model)
            for metric in METRIC_COLUMNS:
                if metric in row:
                    record[f"{metric}__{model_key}"] = row[metric]

        for metric in METRIC_COLUMNS:
            values = []
            for model in model_order:
                model_rows = group[group["model"] == model]
                if model_rows.empty or metric not in model_rows.columns:
                    continue
                value = to_float_or_nan(model_rows.iloc[0][metric])
                if np.isfinite(value):
                    values.append((model, value))
            if values:
                choose_best = min if metric in LOWER_IS_BETTER_METRICS else max
                best_model, best_value = choose_best(values, key=lambda item: item[1])
                record[f"best_model__{metric}"] = best_model
                record[f"best_value__{metric}"] = best_value
        records.append(record)
    return pd.DataFrame(records)


def make_summary_dataframe(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    metric_sort_specs: list[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    if not rows and not errors:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors)
    has_checkpoint = "checkpoint_name" in df.columns or "checkpoint_name" in error_df.columns
    if has_checkpoint:
        checkpoint_names = ordered_unique(
            [str(value) for value in df.get("checkpoint_name", pd.Series(dtype=str)).tolist()]
            + [str(value) for value in error_df.get("checkpoint_name", pd.Series(dtype=str)).tolist()]
        )
    else:
        checkpoint_names = [""]

    records: list[dict[str, Any]] = []
    for checkpoint_name in checkpoint_names:
        if has_checkpoint:
            checkpoint_df = (
                df[df["checkpoint_name"].astype(str) == checkpoint_name]
                if not df.empty and "checkpoint_name" in df
                else pd.DataFrame()
            )
            checkpoint_error_df = (
                error_df[error_df["checkpoint_name"].astype(str) == checkpoint_name]
                if not error_df.empty and "checkpoint_name" in error_df
                else pd.DataFrame()
            )
        else:
            checkpoint_df = df
            checkpoint_error_df = error_df

        row_models = [str(model) for model in checkpoint_df["model"].tolist()] if not checkpoint_df.empty else []
        error_models = (
            [str(model) for model in checkpoint_error_df["model"].tolist()] if not checkpoint_error_df.empty else []
        )
        model_order = ordered_unique(row_models + error_models)
        if not model_order:
            continue

        checkpoint_step = math.nan
        for frame in (checkpoint_df, checkpoint_error_df):
            if not frame.empty and "checkpoint_step" in frame:
                steps = pd.to_numeric(frame["checkpoint_step"], errors="coerce").dropna()
                if not steps.empty:
                    checkpoint_step = int(steps.iloc[0])
                    break

        win_counts: dict[str, dict[str, int]] = {
            metric: {model: 0 for model in model_order} for metric in METRIC_COLUMNS
        }
        if not checkpoint_df.empty:
            for metric in METRIC_COLUMNS:
                if metric not in checkpoint_df.columns:
                    continue
                for _, group in checkpoint_df.groupby("task_id", sort=False):
                    values = pd.to_numeric(group[metric], errors="coerce")
                    if not values.notna().any():
                        continue
                    best_value = values.min() if metric in LOWER_IS_BETTER_METRICS else values.max()
                    winners = group.loc[values == best_value, "model"].astype(str)
                    for winner in set(winners):
                        win_counts[metric][winner] += 1

        baseline_model = "pretrained_tabicl_v2" if "pretrained_tabicl_v2" in model_order else None
        baseline_key = slugify(baseline_model) if baseline_model else None
        baseline_deltas: dict[str, dict[str, float]] = {
            metric: {model: math.nan for model in model_order} for metric in METRIC_COLUMNS
        }
        if baseline_model and not checkpoint_df.empty:
            for metric in METRIC_COLUMNS:
                if metric not in checkpoint_df.columns:
                    continue
                pivot = checkpoint_df.pivot_table(index="task_id", columns="model", values=metric, aggfunc="first")
                if baseline_model not in pivot.columns:
                    continue
                baseline = pd.to_numeric(pivot[baseline_model], errors="coerce")
                for model in model_order:
                    if model not in pivot.columns:
                        continue
                    delta = pd.to_numeric(pivot[model], errors="coerce") - baseline
                    baseline_deltas[metric][model] = float(delta.mean()) if delta.notna().any() else math.nan

        for model in model_order:
            group = (
                checkpoint_df[checkpoint_df["model"].astype(str) == model]
                if not checkpoint_df.empty
                else pd.DataFrame()
            )
            error_group = (
                checkpoint_error_df[checkpoint_error_df["model"].astype(str) == model]
                if not checkpoint_error_df.empty
                else pd.DataFrame()
            )
            if not group.empty:
                model_kind = group.iloc[0].get("model_kind", "")
                model_source = group.iloc[0].get("model_source", "")
            elif not error_group.empty:
                model_kind = error_group.iloc[0].get("model_kind", "")
                model_source = ""
            else:
                model_kind = ""
                model_source = ""

            record: dict[str, Any] = {}
            if has_checkpoint:
                record["checkpoint_name"] = checkpoint_name
                record["checkpoint_step"] = checkpoint_step
            record.update(
                {
                    "model": model,
                    "model_kind": model_kind,
                    "model_source": model_source,
                    "n_success": int(len(group)),
                    "n_tasks": int(group["task_id"].nunique()) if "task_id" in group else 0,
                    "n_errors": int(len(error_group)),
                }
            )
            for metric in METRIC_COLUMNS:
                values = pd.to_numeric(group[metric], errors="coerce") if metric in group else pd.Series(dtype=float)
                record[f"mean_{metric}"] = float(values.mean()) if values.notna().any() else math.nan
                record[f"median_{metric}"] = float(values.median()) if values.notna().any() else math.nan
                if metric in group and "n_samples" in group:
                    weights = pd.to_numeric(group["n_samples"], errors="coerce")
                    valid = values.notna() & weights.notna() & (weights > 0)
                    if valid.any():
                        record[f"weighted_mean_{metric}"] = float(
                            (values[valid] * weights[valid]).sum() / weights[valid].sum()
                        )
                    else:
                        record[f"weighted_mean_{metric}"] = math.nan
                else:
                    record[f"weighted_mean_{metric}"] = math.nan
                record[f"wins_{metric}"] = int(win_counts[metric].get(model, 0))
                if baseline_key:
                    record[f"mean_delta_{metric}_vs_{baseline_key}"] = baseline_deltas[metric].get(model, math.nan)
            records.append(record)

    summary = pd.DataFrame(records)
    if metric_sort_specs is None:
        metric_sort_specs = [(PRIMARY_METRIC, False), ("test_mcc", False)]
    metric_sort_cols = [f"mean_{metric}" for metric, _ in metric_sort_specs]
    metric_sort_pairs = [
        (col, ascending)
        for col, (_, ascending) in zip(metric_sort_cols, metric_sort_specs)
        if col in summary.columns
    ]
    if has_checkpoint and "checkpoint_step" in summary.columns:
        sort_cols = ["checkpoint_step"] + [col for col, _ in metric_sort_pairs]
        ascending = [True] + [ascending for _, ascending in metric_sort_pairs]
        summary = summary.sort_values(sort_cols, ascending=ascending, na_position="last")
    elif metric_sort_pairs:
        sort_cols = [col for col, _ in metric_sort_pairs]
        ascending = [ascending for _, ascending in metric_sort_pairs]
        summary = summary.sort_values(sort_cols, ascending=ascending, na_position="last")
    return summary.reset_index(drop=True)


def ensure_unique_job_labels(jobs: list[dict[str, Any]]) -> None:
    labels = [str(job["model_label"]) for job in jobs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Model labels must be unique across all jobs; duplicate labels: {', '.join(duplicates)}")


def print_result_row(row: dict[str, Any]) -> None:
    if row.get("target_binning") == "continuous":
        print(
            f"  {row['model']:<24} "
            f"spearman={format_metric(row.get('test_spearman'))} "
            f"nmae_iqr={format_metric(row.get('test_nmae_iqr'))} "
            f"mae={format_metric(row.get('test_mae'))} "
            f"rmse={format_metric(row.get('test_rmse'))}"
        )
        return
    if int(row.get("target_bins", 2) or 2) > 2:
        print(
            f"  {row['model']:<24} "
            f"qwk={format_metric(row.get('test_quadratic_weighted_kappa'))} "
            f"ord_mae={format_metric(row.get('test_ordinal_mae'))} "
            f"bal_acc={format_metric(row.get('test_balanced_accuracy'))} "
            f"auroc_ovr={format_metric(row.get('test_roc_auc_ovr_macro'))}"
        )
    else:
        print(
            f"  {row['model']:<24} "
            f"bal_acc={format_metric(row.get('test_balanced_accuracy'))} "
            f"mcc={format_metric(row.get('test_mcc'))} "
            f"auroc={format_metric(row.get('test_auroc'))}"
        )


def print_summary(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    primary_metric: str = PRIMARY_METRIC,
    metric_sort_specs: list[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    summary = make_summary_dataframe(rows, errors, metric_sort_specs=metric_sort_specs)
    if summary.empty:
        print("\nNo successful evaluation rows.")
        return summary

    display_cols = []
    if "checkpoint_name" in summary.columns:
        display_cols.extend(["checkpoint_name", "checkpoint_step"])
    metric_display_cols = [f"mean_{primary_metric}", f"weighted_mean_{primary_metric}"]
    if primary_metric == REGRESSION_PRIMARY_METRIC:
        metric_display_cols.extend(["mean_test_nmae_iqr", "weighted_mean_test_nmae_iqr"])
        metric_display_cols.extend(["mean_test_rmse", "weighted_mean_test_rmse"])
    elif primary_metric != PRIMARY_METRIC:
        metric_display_cols.extend(["mean_test_ordinal_mae", "weighted_mean_test_ordinal_mae"])
        metric_display_cols.extend([f"mean_{PRIMARY_METRIC}", f"weighted_mean_{PRIMARY_METRIC}"])
    display_cols.extend([
        "model",
        "n_success",
        "n_errors",
        *metric_display_cols,
        "mean_test_mcc",
        "mean_test_auroc",
        f"wins_{primary_metric}",
    ])
    display_cols = [col for col in display_cols if col in summary.columns]
    print("\nPer-model summary")
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary


def print_task_list(tasks: list[EvalTask]) -> None:
    rows = [
        {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "table": task.table,
            "target": task.target,
            "target_binning": task.target_binning,
            "target_bins": task.target_bins,
            "bin_edges": ",".join(format_metric(edge) for edge in task.bin_edges),
            "n_samples": len(task.X),
            "n_features": task.X.shape[1],
            "n_classes": len(task.class_labels),
            "class_counts": ",".join(f"{label}:{task.class_counts.get(label, 0)}" for label in task.class_labels),
            "positive_rate": float((task.y == task.class_labels[-1]).mean()) if task.class_labels else math.nan,
            "threshold": task.threshold,
            "task_family": task.task_family,
            "feature_groups": ",".join(task.feature_groups_used),
            "feature_group_counts": json.dumps(task.feature_group_counts, sort_keys=True),
            "quality_flags": ",".join(task.quality_flags),
        }
        for task in tasks
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No tasks generated.")
    else:
        print(df.to_string(index=False, max_colwidth=80))


PLOT_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
    "#bcbd22",
    "#e377c2",
)


def save_metric_trend_svg(summary: pd.DataFrame, metric: str, output_path: Path) -> bool:
    metric_col = f"mean_{metric}"
    required = {"checkpoint_step", "model", metric_col}
    if summary.empty or not required.issubset(summary.columns):
        return False

    plot_df = summary[["checkpoint_step", "model", metric_col]].copy()
    plot_df["checkpoint_step"] = pd.to_numeric(plot_df["checkpoint_step"], errors="coerce")
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["checkpoint_step", metric_col])
    if plot_df["checkpoint_step"].nunique() < 2:
        return False

    models = ordered_unique([str(model) for model in plot_df["model"].tolist()])
    x_values = sorted(float(value) for value in plot_df["checkpoint_step"].unique())
    y_values = plot_df[metric_col].astype(float).tolist()
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= y_pad
    y_max += y_pad
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01

    width, height = 1120, 640
    left, right, top, bottom = 82, 260, 64, 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    title = f"{metric_col} by checkpoint"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="640" viewBox="0 0 1120 640">',
        '<rect width="1120" height="640" fill="white"/>',
        (
            f'<text x="{left}" y="34" font-size="22" font-family="sans-serif" '
            f'font-weight="700">{html.escape(title)}</text>'
        ),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]

    for index in range(6):
        frac = index / 5
        x = left + frac * plot_w
        y = top + plot_h - frac * plot_h
        x_value = x_min + frac * (x_max - x_min)
        y_value = y_min + frac * (y_max - y_min)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 26}" text-anchor="middle" '
            f'font-size="12" font-family="sans-serif">{x_value:.0f}</text>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" font-family="sans-serif">{y_value:.3f}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" '
        'font-size="14" font-family="sans-serif">checkpoint step</text>'
    )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" '
        f'text-anchor="middle" font-size="14" font-family="sans-serif">{html.escape(metric_col)}</text>'
    )

    for index, model in enumerate(models):
        model_df = plot_df[plot_df["model"].astype(str) == model].sort_values("checkpoint_step")
        points = [
            f'{sx(float(row["checkpoint_step"])):.1f},{sy(float(row[metric_col])):.1f}'
            for _, row in model_df.iterrows()
        ]
        if not points:
            continue
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for point in points:
            x_text, y_text = point.split(",")
            parts.append(f'<circle cx="{x_text}" cy="{y_text}" r="3.2" fill="{color}"/>')

        legend_y = top + index * 24
        parts.append(f'<line x1="{left + plot_w + 28}" y1="{legend_y}" x2="{left + plot_w + 54}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{left + plot_w + 62}" y="{legend_y + 4}" font-size="13" '
            f'font-family="sans-serif">{html.escape(model)}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return True


def write_checkpoint_trend_plots(summary: pd.DataFrame, output_dir: Path, metrics: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        output_path = output_dir / f"trend_mean_{metric}.svg"
        if save_metric_trend_svg(summary, metric, output_path):
            written.append(output_path)
    return written


def main() -> None:
    args = parse_args()
    validate_args(args)
    primary_metric = primary_metric_for_args(args)
    metric_sort_specs = metric_sort_specs_for_args(args)
    is_regression_eval = args.target_binning == "continuous"
    pretrained_checkpoint_version = args.pretrained_checkpoint_version or (
        "tabicl-regressor-v2-20260212.ckpt"
        if is_regression_eval
        else "tabicl-classifier-v2-20260212.ckpt"
    )
    tasks = make_tasks(args)
    if args.list_tasks:
        print_task_list(tasks)
        return

    if not tasks:
        raise RuntimeError("No corrosion evaluation tasks were generated. Try --target-mode all or lower --min-samples.")

    checkpoint_specs = resolve_checkpoint_eval_specs(args)
    first_local_specs = checkpoint_specs[0].local_specs
    include_checkpoint_columns = args.checkpoint == "all"
    if args.checkpoint == "all":
        steps = [spec.checkpoint_step for spec in checkpoint_specs if spec.checkpoint_step is not None]
        print(
            f"Evaluating {len(checkpoint_specs)} common checkpoints "
            f"({min(steps)}-{max(steps)} steps) across {len(first_local_specs)} local models."
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for checkpoint_eval in checkpoint_specs:
        jobs: list[dict[str, Any]] = []
        for spec in checkpoint_eval.local_specs:
            model_path = str(spec.checkpoint_path)
            if is_regression_eval:
                jobs.append(
                    {
                        "model_label": spec.label,
                        "model_kind": "local_tabicl_regressor",
                        "factory": lambda model_path=model_path: make_tabicl_regressor(
                            model_path=model_path,
                            checkpoint_version=pretrained_checkpoint_version,
                            device=args.device,
                            n_estimators=args.n_estimators,
                            random_state=args.random_state,
                            allow_auto_download=False,
                        ),
                    }
                )
            else:
                jobs.append(
                    {
                        "model_label": spec.label,
                        "model_kind": "local_tabicl_classifier",
                        "factory": lambda model_path=model_path: make_tabicl_classifier(
                            model_path=model_path,
                            checkpoint_version=pretrained_checkpoint_version,
                            device=args.device,
                            n_estimators=args.n_estimators,
                            random_state=args.random_state,
                            allow_auto_download=False,
                        ),
                    }
                )

        if args.compare_pretrained_tabicl:
            if is_regression_eval:
                jobs.append(
                    {
                        "model_label": "pretrained_tabicl_v2",
                        "model_kind": "pretrained_tabicl_regressor",
                        "factory": lambda: make_tabicl_regressor(
                            model_path=None,
                            checkpoint_version=pretrained_checkpoint_version,
                            device=args.device,
                            n_estimators=args.n_estimators,
                            random_state=args.random_state,
                            allow_auto_download=args.baseline_auto_download,
                        ),
                    }
                )
            else:
                jobs.append(
                    {
                        "model_label": "pretrained_tabicl_v2",
                        "model_kind": "pretrained_tabicl_classifier",
                        "factory": lambda: make_tabicl_classifier(
                            model_path=None,
                            checkpoint_version=pretrained_checkpoint_version,
                            device=args.device,
                            n_estimators=args.n_estimators,
                            random_state=args.random_state,
                            allow_auto_download=args.baseline_auto_download,
                        ),
                    }
                )

        if args.compare_tabpfn:
            tabpfn_factory = (
                (lambda: make_tabpfn_regressor(args.device, args.random_state))
                if is_regression_eval
                else (lambda: make_tabpfn_classifier(args.device, args.random_state))
            )
            jobs.append(
                {
                    "model_label": "pretrained_tabpfn",
                    "model_kind": "pretrained_tabpfn_regressor" if is_regression_eval else "pretrained_tabpfn_classifier",
                    "factory": tabpfn_factory,
                }
            )
        ensure_unique_job_labels(jobs)

        checkpoint_header = f"Checkpoint {checkpoint_eval.checkpoint_name}"
        print(f"\n{checkpoint_header}: evaluating {len(tasks)} tasks x {len(jobs)} models.")
        print("Local checkpoints:")
        for spec in checkpoint_eval.local_specs:
            print(f"  {spec.label}: {spec.checkpoint_path}")

        for task in tasks:
            quality = f", quality_flags={','.join(task.quality_flags)}" if task.quality_flags else ""
            print(
                f"\nTask {task.task_id}: "
                f"n={len(task.X)}, features={task.X.shape[1]}, target={task.target!r}, "
                f"mode={task.target_binning}{'' if is_regression_eval else f', bins={task.target_bins}'}{quality}"
            )
            for job in jobs:
                try:
                    row = evaluate_estimator(
                        model_label=job["model_label"],
                        model_kind=job["model_kind"],
                        estimator_factory=job["factory"],
                        task=task,
                        test_size=args.test_size,
                        random_state=args.random_state,
                        regression_output=args.regression_output,
                    )
                    if include_checkpoint_columns:
                        row["checkpoint_name"] = checkpoint_eval.checkpoint_name
                        row["checkpoint_step"] = checkpoint_eval.checkpoint_step
                    rows.append(row)
                    print_result_row(row)
                    if args.print_json_lines:
                        print(json.dumps(row, sort_keys=True))
                except Exception as exc:
                    error = {
                        "model": job["model_label"],
                        "model_kind": job["model_kind"],
                        "task_id": task.task_id,
                        "dataset": task.dataset,
                        "table": task.table,
                        "target": task.target,
                        "task_quality_flags": ",".join(task.quality_flags),
                        "error": repr(exc),
                    }
                    if include_checkpoint_columns:
                        error["checkpoint_name"] = checkpoint_eval.checkpoint_name
                        error["checkpoint_step"] = checkpoint_eval.checkpoint_step
                    errors.append(error)
                    print(f"  {job['model_label']:<24} ERROR {exc!r}")
                    if args.print_json_lines:
                        print(json.dumps(error, sort_keys=True))

    output_json, output_csv, output_wide_csv, output_summary_csv = default_output_paths(
        first_local_specs,
        checkpoint_part="all_common_checkpoints" if args.checkpoint == "all" else None,
    )
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser().resolve()
        if args.output_wide_csv is None:
            output_wide_csv = derived_csv_path(output_csv, "wide")
        if args.output_summary_csv is None:
            output_summary_csv = derived_csv_path(output_csv, "summary")
    if args.output_wide_csv is not None:
        output_wide_csv = args.output_wide_csv.expanduser().resolve()
    if args.output_summary_csv is not None:
        output_summary_csv = args.output_summary_csv.expanduser().resolve()
    for output_path in (output_json, output_csv, output_wide_csv, output_summary_csv):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = print_summary(
        rows,
        errors,
        primary_metric=primary_metric,
        metric_sort_specs=metric_sort_specs,
    )
    wide_df = make_wide_results_dataframe(rows)
    output_plot_dir: Path | None = None
    plot_paths: list[Path] = []
    if args.checkpoint == "all" and args.checkpoint_plots:
        output_plot_dir = (
            args.output_plot_dir.expanduser().resolve()
            if args.output_plot_dir is not None
            else default_plot_dir(args, output_csv)
        )
        plot_metrics = list(args.plot_metric or METRIC_COLUMNS)
        plot_paths = write_checkpoint_trend_plots(summary_df, output_plot_dir, plot_metrics)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "local_checkpoint": (
            str(first_local_specs[0].checkpoint_path)
            if len(checkpoint_specs) == 1 and len(first_local_specs) == 1
            else None
        ),
        "local_models": [
            {"model": spec.label, "checkpoint": str(spec.checkpoint_path)}
            for spec in first_local_specs
        ]
        if len(checkpoint_specs) == 1
        else [],
        "checkpoint_eval_specs": [
            {
                "checkpoint_name": spec.checkpoint_name,
                "checkpoint_step": spec.checkpoint_step,
                "local_models": [
                    {"model": local.label, "checkpoint": str(local.checkpoint_path)}
                    for local in spec.local_specs
                ],
            }
            for spec in checkpoint_specs
        ],
        "runs": expand_runs(args),
        "checkpoint": args.checkpoint,
        "min_checkpoint_step": args.min_checkpoint_step,
        "checkpoint_step_interval": args.checkpoint_step_interval,
        "target_mode": args.target_mode,
        "target_binning": args.target_binning,
        "target_bins": args.target_bins,
        "regression_output": args.regression_output,
        "pretrained_checkpoint_version": pretrained_checkpoint_version,
        "primary_metric": primary_metric,
        "metric_sort_specs": metric_sort_specs,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_samples_per_task": args.max_samples_per_task,
        "min_numeric_finite_ratio": args.min_numeric_finite_ratio,
        "min_categorical_nonmissing_ratio": args.min_categorical_nonmissing_ratio,
        "excluded_quality_flags": list(args.exclude_quality_flag or []),
        "n_estimators": args.n_estimators,
        "feature_groups_default": list(DEFAULT_FEATURE_GROUPS),
        "electrochem_feature_groups": list(ELECTROCHEM_FEATURE_GROUPS),
        "include_electrochem_features": args.include_electrochem_features,
        "rows": rows,
        "errors": errors,
        "output_files": {
            "json": str(output_json),
            "csv": str(output_csv),
            "wide_csv": str(output_wide_csv),
            "summary_csv": str(output_summary_csv),
            "plot_dir": str(output_plot_dir) if output_plot_dir is not None else None,
            "plots": [str(path) for path in plot_paths],
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    wide_df.to_csv(output_wide_csv, index=False)
    summary_df.to_csv(output_summary_csv, index=False)

    print(f"\nSaved JSON results to {output_json}")
    print(f"Saved row CSV results to {output_csv}")
    print(f"Saved wide comparison CSV to {output_wide_csv}")
    print(f"Saved model summary CSV to {output_summary_csv}")
    if plot_paths:
        print(f"Saved checkpoint trend plots to {output_plot_dir}")
        for path in plot_paths:
            print(f"  {path}")
    if errors:
        print(f"Completed with {len(errors)} task/model errors; see JSON for details.")


if __name__ == "__main__":
    main()
