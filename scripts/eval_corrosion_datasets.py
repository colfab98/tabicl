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
import hashlib
import html
import json
import math
import re
import subprocess
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl.prior.magpie_features import (
    EPIT_MAGPIE_DESCRIPTOR_NAMES,
    EPIT_MAGPIE_MATERIAL_COLUMNS,
    EPIT_MAGPIE_VERSION,
    magpie_descriptors_numpy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "scripts"
if str(ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_DIR))

from analyze_structure import Table, clean_name, load_all_tables  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "eval_results"
EPIT_PIPELINE_TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
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
DEFAULT_EXCLUDED_DATASETS = ("316l_pitting_passivity",)
DEFAULT_EXCLUDED_TASKS = (
    "nace_nist_corr_data__corr_data_database__rate_mm_yr_or_rating_numeric_rate_only",
    "nace_nist_corr_data__corr_data_database__rate_mils_yr_or_rating_numeric_rate_only",
)
DEFAULT_SUMMARY_EXCLUDED_DATASETS = (
    "mooring_steel_seawater",
    "am_mpea_corrosion",
    "mg_az91_inhibitors",
    "mg_ze41_inhibitors",
)
EVAL_NA_STRINGS = {"", "na", "n/a", "nan", "none", "null", "-", "--"}
EVAL_RATING_TO_SEVERITY = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}
CATBOOST_MISSING_CATEGORY = "__CATBOOST_MISSING__"
EXACT_NUMERIC_TEXT_RE = re.compile(r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*$")
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
    "test_pinball_loss",
    "test_npinball_iqr",
    "test_quantile_calibration_mae",
    "test_quantile_calibration_max_error",
    "test_interval_50_coverage_error",
    "test_interval_50_nwidth_iqr",
    "test_interval_50_nwinkler_iqr",
    "test_interval_80_coverage_error",
    "test_interval_80_nwidth_iqr",
    "test_interval_80_nwinkler_iqr",
    "test_interval_90_coverage_error",
    "test_interval_90_nwidth_iqr",
    "test_interval_90_nwinkler_iqr",
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
    "test_pinball_loss",
    "test_npinball_iqr",
    "test_quantile_calibration_mae",
    "test_quantile_calibration_max_error",
    "test_interval_50_coverage_error",
    "test_interval_50_nwidth_iqr",
    "test_interval_50_nwinkler_iqr",
    "test_interval_80_coverage_error",
    "test_interval_80_nwidth_iqr",
    "test_interval_80_nwinkler_iqr",
    "test_interval_90_coverage_error",
    "test_interval_90_nwidth_iqr",
    "test_interval_90_nwinkler_iqr",
    "test_ordinal_mae",
    "test_ordinal_rmse",
    "test_expected_class_mae",
}
DEFAULT_REGRESSION_QUANTILE_ALPHAS = (
    0.05,
    0.10,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
    0.95,
)
REGRESSION_INTERVAL_SPECS = (
    (50, 0.25, 0.75),
    (80, 0.10, 0.90),
    (90, 0.05, 0.95),
)
DEFAULT_REGRESSION_PLOT_METRICS = (
    "test_spearman",
    "test_mae",
    "test_nmae_iqr",
    "test_nrmse_iqr",
    "test_npinball_iqr",
    "test_interval_80_coverage_error",
    "test_pearson",
    "test_r2",
)
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
DATACORTECH_AUTHOR_FEATURES = (
    "pH_2_neutral",
    "ALogP",
    "tpsaEfficiency",
    "bpol",
    "apol",
    "WTPT.5",
    "ALogp2",
    "ATSm1",
    "WTPT.3",
    "XLogP",
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
    split_groups: pd.Series | None
    split_strategy: str
    include_in_summary: bool
    summary_exclusion_reason: str
    fixed_split: EvalSplit | None = None


@dataclass(frozen=True)
class LocalModelSpec:
    label: str
    checkpoint_path: Path


@dataclass(frozen=True)
class CheckpointEvalSpec:
    checkpoint_name: str
    checkpoint_step: int | None
    local_specs: tuple[LocalModelSpec, ...]


@dataclass(frozen=True)
class EvalSplit:
    train_index: np.ndarray
    test_index: np.ndarray
    split_strategy: str


@dataclass
class CachedTabICLModel:
    model: Any
    model_path: Any
    model_config: dict[str, Any]


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
        "--include-latest-common-checkpoint",
        dest="include_latest_common_checkpoint",
        action="store_true",
        default=True,
        help=(
            "When --checkpoint all is interval-filtered, also include the latest common checkpoint "
            "even if it is not on the interval."
        ),
    )
    parser.add_argument(
        "--no-include-latest-common-checkpoint",
        dest="include_latest_common_checkpoint",
        action="store_false",
        help="Do not add the latest common checkpoint outside the interval-filtered set.",
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
    parser.add_argument(
        "--tabicl-feat-shuffle-method",
        choices=("none", "random", "latin", "shift"),
        default="latin",
        help="Feature shuffle method passed to TabICL estimators. Use none for fixed-schema checkpoints.",
    )
    parser.add_argument(
        "--tabicl-norm-methods",
        nargs="+",
        choices=("none", "power", "quantile", "quantile_rtdl", "robust"),
        default=None,
        help=(
            "Feature normalization methods passed to TabICL estimators. If omitted, "
            "the estimator's legacy default (none plus power) is preserved. Pass "
            "'--tabicl-norm-methods none' to explicitly disable power transformation."
        ),
    )
    parser.add_argument(
        "--no-model-cache",
        dest="model_cache",
        action="store_false",
        default=True,
        help="Disable per-checkpoint TabICL model reuse across tasks.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Evaluate each listed train/test split seed and aggregate the resulting rows. "
            "This reruns this evaluator once per seed and writes combined rows, summary, and plots."
        ),
    )
    parser.add_argument(
        "--epit-split-manifest",
        type=Path,
        default=None,
        help=(
            "Fixed EPIT pipeline split manifest. Requires exactly one of "
            "--epit-validation-fold or --epit-final-test."
        ),
    )
    parser.add_argument(
        "--epit-validation-fold",
        type=int,
        choices=range(1, 6),
        default=None,
        help="Saved development fold used as validation; all other development rows are context.",
    )
    parser.add_argument(
        "--epit-final-test",
        action="store_true",
        help=(
            "Use all saved EPIT development rows as context and the untouched "
            "outer final-test rows as test. Requires --epit-split-manifest."
        ),
    )
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
    parser.add_argument(
        "--regression-quantile-alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_REGRESSION_QUANTILE_ALPHAS),
        help=(
            "Quantile levels requested from TabICLRegressor in continuous-target evaluation. "
            "Used for pinball loss, quantile calibration, and 50/80/90 percent interval diagnostics."
        ),
    )
    parser.add_argument(
        "--no-regression-uncertainty",
        dest="regression_uncertainty",
        action="store_false",
        default=True,
        help="Skip quantile predictions and uncertainty metrics for faster continuous-target evaluation.",
    )
    parser.add_argument("--dataset", action="append", help="Limit to dataset id. Can be passed multiple times.")
    parser.add_argument("--task", action="append", help="Limit to exact task id. Can be passed multiple times.")
    parser.add_argument(
        "--datacortech-protocol",
        choices=("author_simple", "benchmark"),
        default="author_simple",
        help=(
            "DatacorTech-only task construction. author_simple is the default and uses the "
            "R-script filters, selected pH/descriptor features, and fixed holdout rows while "
            "preserving the continuous target unless --target-binning changes it. benchmark "
            "uses the stricter leakage-aware molecular-descriptor grouped split and all "
            "leakage-safe features."
        ),
    )
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
    parser.add_argument(
        "--pitting-magpie-features",
        action="store_true",
        help=(
            "For the fixed EPIT pitting task only, mean-impute the 17 composition columns "
            "from the training split and append the fixed ten Magpie-style descriptors."
        ),
    )
    parser.add_argument(
        "--pitting-magpie-model",
        action="append",
        default=[],
        help=(
            "Apply the EPIT Magpie augmentation only to this model label. Can be passed "
            "multiple times to mix native 21- and 31-feature models in one evaluation."
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
    parser.add_argument(
        "--compare-catboost",
        action="store_true",
        help=(
            "Evaluate a conventional CatBoost regressor on the same continuous-target "
            "task table and train/test split as TabICL."
        ),
    )
    parser.add_argument(
        "--compare-catboost-magpie",
        action="store_true",
        help="Also evaluate CatBoost after appending the ten EPIT Magpie descriptors.",
    )
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--catboost-thread-count", type=int, default=-1)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None, help="Row-wise CSV with one row per task/model result.")
    parser.add_argument("--output-wide-csv", type=Path, default=None, help="Wide comparison CSV with one row per task.")
    parser.add_argument("--output-summary-csv", type=Path, default=None, help="Per-model aggregate summary CSV.")
    parser.add_argument(
        "--save-task-tables",
        action="store_true",
        help=(
            "Save each generated task's human-readable model input table before TabICL "
            "preprocessing. The CSV contains retained feature columns plus the target."
        ),
    )
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
        help=(
            "Metric to plot in --checkpoint all mode. Repeat to plot multiple metrics. "
            "Regression defaults include rank, raw MAE, and normalized error metrics; pass "
            "test_rmse explicitly if squared-error-sensitive diagnostics are needed."
        ),
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
    epit_mode_count = int(args.epit_validation_fold is not None) + int(
        args.epit_final_test
    )
    if args.epit_split_manifest is None and epit_mode_count:
        raise ValueError(
            "--epit-validation-fold and --epit-final-test require "
            "--epit-split-manifest."
        )
    if args.epit_split_manifest is not None and epit_mode_count != 1:
        raise ValueError(
            "--epit-split-manifest requires exactly one of "
            "--epit-validation-fold or --epit-final-test."
        )
    if args.epit_split_manifest is not None and args.split_seeds is not None:
        raise ValueError(
            "A fixed EPIT fold cannot be combined with random --split-seeds."
        )
    if args.checkpoint_step_interval < 0:
        raise ValueError("--checkpoint-step-interval must be >= 0.")
    alphas = list(args.regression_quantile_alphas or [])
    if len(alphas) != len(set(alphas)):
        raise ValueError("--regression-quantile-alphas must not contain duplicate levels.")
    if any(alpha <= 0.0 or alpha >= 1.0 for alpha in alphas):
        raise ValueError("--regression-quantile-alphas values must be between 0 and 1.")
    args.regression_quantile_alphas = sorted(alphas)
    if args.compare_catboost or args.compare_catboost_magpie:
        if args.target_binning != "continuous":
            raise ValueError("CatBoost comparison currently supports only --target-binning continuous.")
        if args.catboost_iterations <= 0:
            raise ValueError("--catboost-iterations must be > 0.")
        if args.catboost_depth <= 0:
            raise ValueError("--catboost-depth must be > 0.")
        if args.catboost_learning_rate <= 0:
            raise ValueError("--catboost-learning-rate must be > 0.")
        if args.catboost_l2_leaf_reg < 0:
            raise ValueError("--catboost-l2-leaf-reg must be >= 0.")
        if args.catboost_thread_count == 0:
            raise ValueError("--catboost-thread-count must not be 0.")
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
            ("test_npinball_iqr", True),
            ("test_interval_80_coverage_error", True),
            ("test_mae", True),
            ("test_rmse", True),
        ]
    if args.target_binning == "quantile_multiclass":
        return [
            (ORDINAL_PRIMARY_METRIC, False),
            ("test_ordinal_mae", True),
            (PRIMARY_METRIC, False),
        ]
    return [(PRIMARY_METRIC, False), ("test_mcc", False)]


def default_plot_metrics_for_args(args: argparse.Namespace) -> list[str]:
    if args.target_binning == "continuous":
        return list(DEFAULT_REGRESSION_PLOT_METRICS)
    return list(METRIC_COLUMNS)


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
        runs = expand_runs(args)
        local_ckpt_paths = list(args.local_ckpt_path or [])
        if not runs and not local_ckpt_paths and (
            args.compare_pretrained_tabicl
            or args.compare_tabpfn
            or args.compare_catboost
            or args.compare_catboost_magpie
        ):
            return [
                CheckpointEvalSpec(
                    checkpoint_name="pretrained_baselines",
                    checkpoint_step=None,
                    local_specs=tuple(),
                )
            ]
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
    all_common_steps = sorted(
        step
        for step in set.intersection(*(set(paths) for paths in by_run.values()))
        if step >= args.min_checkpoint_step
    )
    common_steps = [
        step
        for step in all_common_steps
        if args.checkpoint_step_interval == 0 or step % args.checkpoint_step_interval == 0
    ]
    if args.include_latest_common_checkpoint and all_common_steps:
        latest_common_step = all_common_steps[-1]
        if latest_common_step not in common_steps:
            common_steps.append(latest_common_step)
            common_steps.sort()
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


def is_nace_rate_or_rating_target(table: Table, target_col: str) -> bool:
    return (
        table.dataset == "nace_nist_corr_data"
        and table.table == "CORR-DATA_Database"
        and target_col in {"Rate (mm/yr) or Rating", "Rate (mils/yr) or Rating"}
    )


def exact_numeric_rate_or_nan(value: Any) -> float:
    """Parse only exact numeric rate entries, excluding ratings and censored values."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_eval_numeric_text(clean_name(value))
    if text.lower() in EVAL_NA_STRINGS:
        return math.nan
    if not EXACT_NUMERIC_TEXT_RE.match(text):
        return math.nan
    return float(text)


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
    rows: list[dict[str, Any]] | None = None,
    min_numeric_finite_ratio: float,
    min_categorical_nonmissing_ratio: float,
) -> tuple[pd.Series | None, str, str]:
    group = table.groups.get(column, "metadata")
    source_rows = table.rows if rows is None else rows
    raw = [row.get(column) for row in source_rows]
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


def is_default_excluded_task(task: EvalTask) -> bool:
    if task.task_id in DEFAULT_EXCLUDED_TASKS:
        return True
    return task.dataset == "nace_nist_corr_data" and task.task_id.endswith("_numeric_rate_only")


def summary_exclusion_reason_for_task(task_id: str, dataset: str) -> str:
    if dataset in DEFAULT_SUMMARY_EXCLUDED_DATASETS:
        return f"diagnostic_dataset:{dataset}"
    return ""


def normalized_group_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "<NA>"
        return format(numeric, ".12g")
    return str(value).strip()


def dataframe_group_keys(frame: pd.DataFrame) -> pd.Series:
    values = [
        json.dumps([normalized_group_value(value) for value in row], separators=(",", ":"))
        for row in frame.itertuples(index=False, name=None)
    ]
    return pd.Series(values, index=frame.index, dtype="string")


MOLECULAR_DESCRIPTOR_GROUPED_SPLIT_DATASETS = {
    "datacor_aluminum_inhibitors",
    "datacortech_aluminum_inhibitors",
}


def split_groups_for_task(table: Table, X: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if table.dataset not in MOLECULAR_DESCRIPTOR_GROUPED_SPLIT_DATASETS:
        return None, "random_stratified"

    descriptor_cols = [
        col
        for col in X.columns
        if table.groups.get(col) == "molecular_descriptor"
    ]
    if not descriptor_cols:
        return None, "random_stratified"

    groups = dataframe_group_keys(X[descriptor_cols]).reset_index(drop=True)
    if groups.nunique(dropna=False) < 2:
        return None, "random_stratified"
    return groups, "grouped_molecular_descriptor"


def is_datacortech_author_row(row: dict[str, Any]) -> bool:
    return (
        clean_name(row.get("Metal")).casefold() == "al"
        and clean_name(row.get("Synergistic_inhib")).casefold() == "no"
    )


def datacortech_author_holdout_split(n_samples: int) -> EvalSplit:
    if n_samples != 1966:
        raise ValueError(
            "DatacorTech author_simple expects 1966 rows after filtering "
            f"to Metal == 'Al' and Synergistic_inhib == 'No', got {n_samples}."
        )

    train_index: list[int] = []
    test_index: list[int] = []
    for start in range(0, 1700, 100):
        train_index.extend(range(start, start + 80))
        test_index.extend(range(start + 80, start + 100))
    train_index.extend(range(1700, 1880))
    test_index.extend(range(1880, 1900))
    train_index.extend(range(1900, 1966))

    train = np.asarray(train_index, dtype=int)
    test = np.asarray(test_index, dtype=int)
    if len(set(train).intersection(set(test))) != 0 or len(train) + len(test) != n_samples:
        raise ValueError("DatacorTech author_simple holdout indices are inconsistent.")
    return EvalSplit(train_index=train, test_index=test, split_strategy="datacortech_author_holdout")


def datacortech_neutral_ph_series(rows: list[dict[str, Any]]) -> pd.Series:
    values: list[float] = []
    for row in rows:
        ph = eval_to_float(row.get("pH"), column="pH", group="environment")
        values.append(float(4.0 < ph < 10.0) if np.isfinite(ph) else math.nan)
    return pd.Series(values, name="pH_2_neutral")


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


def split_distribution_score(labels: np.ndarray | None, train_index: np.ndarray, test_index: np.ndarray) -> float:
    if labels is None:
        return 0.0

    labels = np.asarray(labels)
    unique_labels = pd.unique(pd.Series(labels))
    if len(unique_labels) <= 1:
        return 0.0

    full = pd.Series(labels).value_counts(normalize=True)
    train = pd.Series(labels[train_index]).value_counts(normalize=True)
    test = pd.Series(labels[test_index]).value_counts(normalize=True)
    score = 0.0
    missing_penalty = 0.0
    for label in unique_labels:
        full_value = float(full.get(label, 0.0))
        train_value = float(train.get(label, 0.0))
        test_value = float(test.get(label, 0.0))
        score += abs(train_value - full_value) + abs(test_value - full_value)
        if train_value == 0.0:
            missing_penalty += 10.0
        if test_value == 0.0:
            missing_penalty += 2.0
    return score + missing_penalty


def task_train_test_indices(
    task: EvalTask,
    *,
    y_values: np.ndarray | pd.Series,
    test_size: float,
    random_state: int,
    stratify: np.ndarray | pd.Series | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(len(task.X))
    if task.split_groups is None:
        train_index, test_index = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
        return np.asarray(train_index), np.asarray(test_index), "random_stratified"

    groups = task.split_groups.astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        train_index, test_index = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
        return np.asarray(train_index), np.asarray(test_index), "random_stratified_fallback"

    y_array = np.asarray(y_values)
    stratify_array = np.asarray(stratify) if stratify is not None else None
    splitter = GroupShuffleSplit(n_splits=128, test_size=test_size, random_state=random_state)
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_score = math.inf
    for train_index, test_index in splitter.split(indices, y_array, groups):
        if len(train_index) == 0 or len(test_index) == 0:
            continue
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups.intersection(test_groups):
            continue
        size_score = abs((len(test_index) / len(indices)) - test_size)
        distribution_score = split_distribution_score(stratify_array, train_index, test_index)
        score = size_score + distribution_score
        if score < best_score:
            best_score = score
            best_split = (np.asarray(train_index), np.asarray(test_index))

    if best_split is None:
        raise RuntimeError(f"Could not create a non-overlapping grouped split for task {task.task_id}")
    return best_split[0], best_split[1], task.split_strategy


def make_task_split(task: EvalTask, *, test_size: float, random_state: int) -> EvalSplit:
    if task.fixed_split is not None:
        return task.fixed_split

    if task.target_binning == "continuous":
        y_values = task.y.astype(float)
        stratify = regression_stratify_labels(y_values.to_numpy(dtype=float))
    else:
        y_values = task.y
        stratify = task.y
    train_index, test_index, split_strategy = task_train_test_indices(
        task,
        y_values=y_values,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    return EvalSplit(
        train_index=np.asarray(train_index),
        test_index=np.asarray(test_index),
        split_strategy=split_strategy,
    )


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
    datacortech_protocol: str,
    feature_groups: tuple[str, ...],
    max_category_cardinality: int,
    min_numeric_finite_ratio: float,
    min_categorical_nonmissing_ratio: float,
    min_samples: int,
    min_class_count: int,
    max_samples_per_task: int,
    random_state: int,
) -> EvalTask | None:
    rows = list(table.rows)
    use_datacortech_author_simple = (
        table.dataset == "datacortech_aluminum_inhibitors"
        and datacortech_protocol == "author_simple"
    )
    if use_datacortech_author_simple:
        rows = [row for row in rows if is_datacortech_author_row(row)]

    strict_numeric_rate_only = target_binning == "continuous" and is_nace_rate_or_rating_target(table, target_col)
    target_parser = exact_numeric_rate_or_nan if strict_numeric_rate_only else (
        lambda value: eval_to_float(value, column=target_col, group="target")
    )
    target_values = np.array(
        [target_parser(row.get(target_col)) for row in rows],
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
    candidate_columns = DATACORTECH_AUTHOR_FEATURES if use_datacortech_author_simple else tuple(table.columns)
    for col in candidate_columns:
        if col == "pH_2_neutral" and use_datacortech_author_simple:
            features[col] = datacortech_neutral_ph_series(rows)
            continue

        if col not in table.columns:
            dropped.append(f"{col}:missing")
            continue

        group = table.groups.get(col, "metadata")
        if col == target_col or (not use_datacortech_author_simple and group not in feature_groups):
            continue
        series, kind, drop_reason = feature_value_series(
            table,
            col,
            rows=rows,
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

    target_slug = slugify(target_col)
    if strict_numeric_rate_only:
        target_slug = f"{target_slug}_numeric_rate_only"
    task_id = f"{table.dataset}__{slugify(table.table)}__{target_slug}"
    quality_flags = assess_task_quality(table, X, y)
    if strict_numeric_rate_only:
        quality_flags = sorted(set(quality_flags + ["numeric_rate_only_target"]))
    fixed_split: EvalSplit | None = None
    if use_datacortech_author_simple:
        fixed_split = datacortech_author_holdout_split(len(X))
        split_groups = None
        split_strategy = fixed_split.split_strategy
        quality_flags = sorted(
            set(quality_flags + ["datacortech_author_simple", "datacortech_author_feature_subset", split_strategy])
        )
    else:
        split_groups, split_strategy = split_groups_for_task(table, X)
        if split_groups is not None:
            quality_flags = sorted(set(quality_flags + [split_strategy]))
    summary_exclusion_reason = summary_exclusion_reason_for_task(task_id, table.dataset)
    feature_group_counts = Counter(
        "environment" if col == "pH_2_neutral" else table.groups.get(col, "metadata")
        for col in X.columns
    )
    ordered_class_counts = {
        label: int(class_counts.get(label, 0))
        for label in class_labels
    }
    feature_groups_used = sorted(
        set(
            "environment" if col == "pH_2_neutral" else table.groups.get(col, "metadata")
            for col in X.columns
        )
    )
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
        feature_groups_used=feature_groups_used,
        feature_group_counts=dict(sorted(feature_group_counts.items())),
        dropped_feature_columns=dropped,
        quality_flags=quality_flags,
        split_groups=split_groups,
        split_strategy=split_strategy,
        include_in_summary=summary_exclusion_reason == "",
        summary_exclusion_reason=summary_exclusion_reason,
        fixed_split=fixed_split,
    )

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_epit_pipeline_indices(
    tasks: list[EvalTask],
    *,
    manifest_path: Path,
) -> tuple[EvalTask, np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate a frozen EPIT split and return its task and outer indices."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema = manifest.get("schema_version")
    if manifest_schema == "epit_split_manifest_v2":
        # Keep the general corrosion evaluator unchanged unless the frozen v2
        # EPIT path is explicitly requested. Module execution and direct
        # script execution expose the sibling package under different names.
        if __package__:
            from scripts.epit_pipeline.artifact_hashes import load_frozen_split
        else:
            from epit_pipeline.artifact_hashes import load_frozen_split

        manifest = load_frozen_split(manifest_path).manifest
    elif manifest_schema != "epit_split_manifest_v1":
        raise RuntimeError("Unsupported EPIT split manifest schema.")
    dataset_meta = manifest.get("dataset", {})
    if dataset_meta.get("task_id") != EPIT_PIPELINE_TASK_ID:
        raise RuntimeError("EPIT split manifest task does not match the evaluator.")
    source_path = REPO_ROOT / str(dataset_meta.get("source_file", ""))
    if not source_path.is_file():
        raise FileNotFoundError(f"EPIT source file not found: {source_path}")
    if _sha256_file(source_path) != str(dataset_meta.get("source_sha256", "")):
        raise RuntimeError("EPIT source file changed after the split was created.")

    matching = [task for task in tasks if task.task_id == EPIT_PIPELINE_TASK_ID]
    if len(matching) != 1:
        raise RuntimeError(
            f"Expected exactly one EPIT task, found {len(matching)}."
        )
    task = matching[0]
    expected_rows = int(dataset_meta.get("usable_rows", -1))
    if len(task.X) != expected_rows:
        raise RuntimeError(
            "EPIT evaluator row count does not match the split manifest."
        )
    if task.target != str(dataset_meta.get("target_column", "")):
        raise RuntimeError("EPIT evaluator target does not match the split manifest.")

    records = manifest.get("rows", [])
    if len(records) != expected_rows:
        raise RuntimeError("EPIT split manifest does not cover every task row.")
    by_index = {int(record["task_row_index"]): record for record in records}
    if sorted(by_index) != list(range(expected_rows)):
        raise RuntimeError("EPIT split manifest row indices are incomplete.")

    development_global = np.asarray(
        sorted(
            index
            for index, record in by_index.items()
            if record["outer_split"] == "development"
        ),
        dtype=int,
    )
    final_global = np.asarray(
        sorted(
            index
            for index, record in by_index.items()
            if record["outer_split"] == "final_test"
        ),
        dtype=int,
    )
    split_design = manifest.get("split_design", {})
    if len(development_global) != int(split_design.get("development_rows", -1)):
        raise RuntimeError("EPIT development row count is inconsistent.")
    if len(final_global) != int(split_design.get("final_test_rows", -1)):
        raise RuntimeError("EPIT final-test row count is inconsistent.")
    if np.intersect1d(development_global, final_global).size:
        raise RuntimeError("EPIT development and final-test rows overlap.")

    return task, development_global, final_global, manifest


def apply_epit_pipeline_fold(
    tasks: list[EvalTask],
    *,
    manifest_path: Path,
    validation_fold: int,
) -> None:
    """Restrict the EPIT task to development rows and apply one saved fold."""
    task, development_global, _, manifest = _load_epit_pipeline_indices(
        tasks,
        manifest_path=manifest_path,
    )
    by_index = {
        int(record["task_row_index"]): record
        for record in manifest.get("rows", [])
    }
    validation_global = np.asarray(
        [
            index
            for index in development_global
            if int(by_index[int(index)]["optuna_validation_fold"])
            == int(validation_fold)
        ],
        dtype=int,
    )
    context_global = np.setdiff1d(
        development_global,
        validation_global,
        assume_unique=True,
    )
    if len(validation_global) == 0 or len(context_global) == 0:
        raise RuntimeError("EPIT saved validation fold is empty.")

    local_index = {
        int(global_index): local
        for local, global_index in enumerate(development_global)
    }
    context_local = np.asarray(
        [local_index[int(index)] for index in context_global],
        dtype=int,
    )
    validation_local = np.asarray(
        [local_index[int(index)] for index in validation_global],
        dtype=int,
    )
    task.X = task.X.iloc[development_global].reset_index(drop=True)
    task.y = task.y.iloc[development_global].reset_index(drop=True)
    task.y_ordinal = task.y_ordinal.iloc[development_global].reset_index(drop=True)
    if task.split_groups is not None:
        task.split_groups = task.split_groups.iloc[development_global].reset_index(drop=True)
    task.fixed_split = EvalSplit(
        train_index=context_local,
        test_index=validation_local,
        split_strategy=f"epit_pipeline_development_fold_{validation_fold}",
    )
    task.split_strategy = task.fixed_split.split_strategy


def apply_epit_pipeline_final_test(
    tasks: list[EvalTask],
    *,
    manifest_path: Path,
) -> None:
    """Use all development rows as context and untouched rows as final test."""
    task, development_global, final_global, _ = _load_epit_pipeline_indices(
        tasks,
        manifest_path=manifest_path,
    )
    task.fixed_split = EvalSplit(
        train_index=development_global,
        test_index=final_global,
        split_strategy="epit_pipeline_final_test",
    )
    task.split_strategy = task.fixed_split.split_strategy


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
    default_excluded_datasets = set(DEFAULT_EXCLUDED_DATASETS)
    tasks: list[EvalTask] = []
    for table in tables:
        if table.dataset in default_excluded_datasets:
            continue
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
                datacortech_protocol=args.datacortech_protocol,
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
            if is_default_excluded_task(task):
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
    feat_shuffle_method: str = "latin",
    norm_methods: list[str] | None = None,
) -> TabICLClassifier:
    kwargs: dict[str, Any] = {
        "device": device,
        "n_estimators": n_estimators,
        "feat_shuffle_method": feat_shuffle_method,
        "norm_methods": norm_methods,
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
    feat_shuffle_method: str = "latin",
    norm_methods: list[str] | None = None,
) -> TabICLRegressor:
    kwargs: dict[str, Any] = {
        "device": device,
        "n_estimators": n_estimators,
        "feat_shuffle_method": feat_shuffle_method,
        "norm_methods": norm_methods,
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


class CatBoostRegressorAdapter:
    """Sklearn-like CatBoost wrapper with explicit categorical handling.

    The shared evaluator passes the same raw task dataframe to every model.
    CatBoost can consume missing numeric values directly, while categorical
    values must be strings or integers and cannot contain pandas missing
    markers. This adapter changes only that representation and learns no
    preprocessing state from the outer test split.
    """

    def __init__(
        self,
        *,
        regressor_class: Any,
        version: str,
        iterations: int,
        depth: int,
        learning_rate: float,
        l2_leaf_reg: float,
        random_state: int,
        thread_count: int,
    ) -> None:
        self._regressor_class = regressor_class
        self._model_kwargs = {
            "allow_writing_files": False,
            "depth": depth,
            "eval_metric": "RMSE",
            "iterations": iterations,
            "l2_leaf_reg": l2_leaf_reg,
            "learning_rate": learning_rate,
            "loss_function": "RMSE",
            "random_seed": random_state,
            "task_type": "CPU",
            "thread_count": thread_count,
            "verbose": False,
        }
        self.model_source_ = (
            f"catboost=={version};iterations={iterations};depth={depth};"
            f"learning_rate={learning_rate:g};l2_leaf_reg={l2_leaf_reg:g};"
            f"thread_count={thread_count}"
        )

    @staticmethod
    def _categorical_columns(frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column].dtype)
        ]

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("CatBoost corrosion evaluation expects a pandas DataFrame.")
        prepared = frame.copy()
        for column in categorical_columns:
            prepared[column] = (
                prepared[column]
                .astype("string")
                .fillna(CATBOOST_MISSING_CATEGORY)
                .astype(str)
            )
        return prepared

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> CatBoostRegressorAdapter:
        self.feature_columns_ = list(X.columns)
        self.categorical_columns_ = self._categorical_columns(X)
        prepared = self._prepare_frame(X, self.categorical_columns_)
        self.model_ = self._regressor_class(**self._model_kwargs)
        self.model_.fit(
            prepared,
            y,
            cat_features=self.categorical_columns_,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("CatBoostRegressorAdapter must be fitted before prediction.")
        if list(X.columns) != self.feature_columns_:
            raise ValueError("CatBoost train and test feature columns do not match.")
        prepared = self._prepare_frame(X, self.categorical_columns_)
        return np.asarray(self.model_.predict(prepared), dtype=float)


def make_catboost_regressor(
    *,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    random_state: int,
    thread_count: int,
) -> CatBoostRegressorAdapter:
    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError(
            "catboost is required for --compare-catboost; install the corrosion-eval extra."
        ) from exc

    return CatBoostRegressorAdapter(
        regressor_class=catboost.CatBoostRegressor,
        version=str(getattr(catboost, "__version__", "unknown")),
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_state=random_state,
        thread_count=thread_count,
    )


def make_cached_tabicl_factory(
    estimator_factory: Any,
    *,
    cache_key: tuple[Any, ...],
    model_cache: dict[tuple[Any, ...], CachedTabICLModel],
) -> Any:
    def cached_factory() -> Any:
        if cache_key not in model_cache:
            loader = estimator_factory()
            loader._resolve_device()
            loader._load_model()
            loader.model_.to(loader.device_)
            model_cache[cache_key] = CachedTabICLModel(
                model=loader.model_,
                model_path=loader.model_path_,
                model_config=loader.model_config_,
            )

        cached = model_cache[cache_key]
        estimator = estimator_factory()

        def load_model_from_cache() -> None:
            model = cached.model
            if hasattr(model, "clear_cache"):
                model.clear_cache()
            estimator.model_ = model
            estimator.model_path_ = cached.model_path
            estimator.model_config_ = cached.model_config
            estimator.model_.eval()

        estimator._load_model = load_model_from_cache
        return estimator

    return cached_factory


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


def alpha_index(alphas: list[float], target_alpha: float) -> int | None:
    for index, alpha in enumerate(alphas):
        if math.isclose(alpha, target_alpha, rel_tol=0.0, abs_tol=1e-9):
            return index
    return None


def empty_regression_uncertainty_metrics() -> dict[str, float]:
    metrics = {
        "test_pinball_loss": math.nan,
        "test_npinball_iqr": math.nan,
        "test_quantile_calibration_mae": math.nan,
        "test_quantile_calibration_max_error": math.nan,
    }
    for label, _, _ in REGRESSION_INTERVAL_SPECS:
        metrics.update(
            {
                f"test_interval_{label}_coverage": math.nan,
                f"test_interval_{label}_coverage_error": math.nan,
                f"test_interval_{label}_mean_width": math.nan,
                f"test_interval_{label}_nwidth_iqr": math.nan,
                f"test_interval_{label}_winkler": math.nan,
                f"test_interval_{label}_nwinkler_iqr": math.nan,
            }
        )
    return metrics


def regression_uncertainty_metrics(
    *,
    y_true: np.ndarray,
    quantiles: np.ndarray | None,
    alphas: list[float],
    target_iqr: float,
) -> dict[str, float]:
    metrics = empty_regression_uncertainty_metrics()
    if quantiles is None:
        return metrics

    y_true = np.asarray(y_true, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    alphas_array = np.asarray(alphas, dtype=float)
    if quantiles.ndim != 2 or quantiles.shape[0] != len(y_true) or quantiles.shape[1] != len(alphas):
        return metrics

    valid_rows = np.isfinite(y_true) & np.isfinite(quantiles).all(axis=1)
    if not valid_rows.any():
        return metrics

    y_valid = y_true[valid_rows]
    q_valid = quantiles[valid_rows]
    errors = y_valid[:, None] - q_valid
    pinball = np.maximum(alphas_array * errors, (alphas_array - 1.0) * errors)
    pinball_loss = float(np.mean(pinball))
    metrics["test_pinball_loss"] = pinball_loss
    metrics["test_npinball_iqr"] = pinball_loss / target_iqr if target_iqr > 0 else math.nan

    observed_cdf = np.mean(y_valid[:, None] <= q_valid, axis=0)
    calibration_errors = np.abs(observed_cdf - alphas_array)
    metrics["test_quantile_calibration_mae"] = float(np.mean(calibration_errors))
    metrics["test_quantile_calibration_max_error"] = float(np.max(calibration_errors))

    for label, lower_alpha, upper_alpha in REGRESSION_INTERVAL_SPECS:
        lower_index = alpha_index(alphas, lower_alpha)
        upper_index = alpha_index(alphas, upper_alpha)
        if lower_index is None or upper_index is None:
            continue

        lower = np.minimum(q_valid[:, lower_index], q_valid[:, upper_index])
        upper = np.maximum(q_valid[:, lower_index], q_valid[:, upper_index])
        nominal_coverage = (upper_alpha - lower_alpha)
        misses_low = y_valid < lower
        misses_high = y_valid > upper
        coverage = float(np.mean((~misses_low) & (~misses_high)))
        width = upper - lower
        tail_alpha = max(1.0 - nominal_coverage, np.finfo(float).eps)
        winkler = width.copy()
        winkler[misses_low] += (2.0 / tail_alpha) * (lower[misses_low] - y_valid[misses_low])
        winkler[misses_high] += (2.0 / tail_alpha) * (y_valid[misses_high] - upper[misses_high])
        mean_width = float(np.mean(width))
        mean_winkler = float(np.mean(winkler))
        metrics.update(
            {
                f"test_interval_{label}_coverage": coverage,
                f"test_interval_{label}_coverage_error": abs(coverage - nominal_coverage),
                f"test_interval_{label}_mean_width": mean_width,
                f"test_interval_{label}_nwidth_iqr": mean_width / target_iqr if target_iqr > 0 else math.nan,
                f"test_interval_{label}_winkler": mean_winkler,
                f"test_interval_{label}_nwinkler_iqr": mean_winkler / target_iqr if target_iqr > 0 else math.nan,
            }
        )

    return metrics


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


def split_metadata(task: EvalTask, train_index: np.ndarray, test_index: np.ndarray, split_strategy: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "split_strategy": split_strategy,
        "split_group_count": 0,
        "split_train_group_count": 0,
        "split_test_group_count": 0,
        "split_group_overlap_count": 0,
        "include_in_summary": bool(task.include_in_summary),
        "summary_exclusion_reason": task.summary_exclusion_reason,
    }
    if task.split_groups is None:
        return metadata

    groups = task.split_groups.astype(str).reset_index(drop=True)
    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])
    metadata.update(
        {
            "split_group_count": int(groups.nunique(dropna=False)),
            "split_train_group_count": int(len(train_groups)),
            "split_test_group_count": int(len(test_groups)),
            "split_group_overlap_count": int(len(train_groups.intersection(test_groups))),
        }
    )
    return metadata


def augment_pitting_magpie_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append descriptors using the same train-fitted mean imputation as TabICL."""

    material_columns = list(EPIT_MAGPIE_MATERIAL_COLUMNS)
    missing_columns = [column for column in material_columns if column not in X_train.columns]
    if missing_columns:
        raise ValueError(f"EPIT Magpie composition columns are missing: {missing_columns}")
    if list(X_train.columns[: len(material_columns)]) != material_columns:
        raise ValueError("EPIT Magpie material columns must be the first 17 model inputs in fixed order.")
    if list(X_test.columns) != list(X_train.columns):
        raise ValueError("Training and test feature columns do not match for EPIT Magpie augmentation.")

    imputer = SimpleImputer(strategy="mean")
    train_material = imputer.fit_transform(X_train.loc[:, material_columns])
    test_material = imputer.transform(X_test.loc[:, material_columns])
    if train_material.shape[1] != len(material_columns):
        raise ValueError("A material column is entirely missing in the training split.")

    X_train_augmented = X_train.copy()
    X_test_augmented = X_test.copy()
    X_train_augmented.loc[:, material_columns] = train_material
    X_test_augmented.loc[:, material_columns] = test_material
    train_descriptors = pd.DataFrame(
        magpie_descriptors_numpy(train_material),
        columns=EPIT_MAGPIE_DESCRIPTOR_NAMES,
        index=X_train_augmented.index,
    )
    test_descriptors = pd.DataFrame(
        magpie_descriptors_numpy(test_material),
        columns=EPIT_MAGPIE_DESCRIPTOR_NAMES,
        index=X_test_augmented.index,
    )
    return (
        pd.concat((X_train_augmented, train_descriptors), axis=1),
        pd.concat((X_test_augmented, test_descriptors), axis=1),
    )


def evaluate_estimator(
    *,
    model_label: str,
    model_kind: str,
    estimator_factory: Any,
    task: EvalTask,
    task_split: EvalSplit | None,
    test_size: float,
    random_state: int,
    regression_output: str = "median",
    regression_quantile_alphas: list[float] | None = None,
    regression_uncertainty: bool = True,
    pitting_magpie_features: bool = False,
) -> dict[str, Any]:
    if pitting_magpie_features and task.task_id != "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg":
        raise ValueError("pitting_magpie_features is only valid for the fixed EPIT pitting task.")
    feature_groups_used = list(task.feature_groups_used)
    feature_group_counts = dict(task.feature_group_counts)
    if pitting_magpie_features:
        feature_groups_used.append("material_descriptor")
        feature_group_counts["material_descriptor"] = len(EPIT_MAGPIE_DESCRIPTOR_NAMES)

    if task.target_binning == "continuous":
        regression_quantile_alphas = list(regression_quantile_alphas or DEFAULT_REGRESSION_QUANTILE_ALPHAS)
        y_values = task.y.astype(float)
        split = task_split or make_task_split(task, test_size=test_size, random_state=random_state)
        train_index = split.train_index
        test_index = split.test_index
        split_strategy = split.split_strategy
        X_train = task.X.iloc[train_index].reset_index(drop=True)
        X_test = task.X.iloc[test_index].reset_index(drop=True)
        if pitting_magpie_features:
            X_train, X_test = augment_pitting_magpie_split(X_train, X_test)
        y_train = y_values.iloc[train_index].reset_index(drop=True)
        y_test = y_values.iloc[test_index].reset_index(drop=True)
        estimator = estimator_factory()
        estimator.fit(X_train, y_train)
        quantile_pred: np.ndarray | None = None
        if regression_uncertainty:
            try:
                predictions = estimator.predict(
                    X_test,
                    output_type=[regression_output, "quantiles"],
                    alphas=regression_quantile_alphas,
                )
                if isinstance(predictions, dict):
                    y_pred = np.asarray(predictions[regression_output], dtype=float)
                    quantile_pred = np.asarray(predictions["quantiles"], dtype=float)
                else:
                    y_pred = np.asarray(predictions, dtype=float)
            except TypeError:
                try:
                    y_pred = np.asarray(estimator.predict(X_test, output_type=regression_output), dtype=float)
                except TypeError:
                    y_pred = np.asarray(estimator.predict(X_test), dtype=float)
        else:
            try:
                y_pred = np.asarray(estimator.predict(X_test, output_type=regression_output), dtype=float)
            except TypeError:
                y_pred = np.asarray(estimator.predict(X_test), dtype=float)
        y_true = y_test.to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        target_iqr = float(np.subtract(*np.percentile(y_train.to_numpy(dtype=float), [75, 25])))
        nmae_iqr = mae / target_iqr if target_iqr > 0 else math.nan
        nrmse_iqr = rmse / target_iqr if target_iqr > 0 else math.nan
        uncertainty_metrics = (
            regression_uncertainty_metrics(
                y_true=y_true,
                quantiles=quantile_pred,
                alphas=regression_quantile_alphas,
                target_iqr=target_iqr,
            )
            if regression_uncertainty
            else empty_regression_uncertainty_metrics()
        )

        model_source = getattr(
            estimator,
            "model_source_",
            getattr(estimator, "model_path_", ""),
        )
        row = {
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
            "regression_uncertainty": bool(regression_uncertainty),
            "regression_quantile_alphas": (
                " ".join(format(alpha, ".6g") for alpha in regression_quantile_alphas)
                if regression_uncertainty
                else ""
            ),
            "n_samples": int(len(task.X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X_train.shape[1]),
            "pitting_magpie_features": bool(pitting_magpie_features),
            "n_classes": 0,
            "class_labels": "",
            "class_counts": "",
            "positive_label": "",
            "positive_rate": math.nan,
            **split_metadata(task, train_index, test_index, split_strategy),
            "task_quality_flags": ",".join(task.quality_flags),
            "task_family": task.task_family,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": float(r2_score(y_true, y_pred)),
            "test_spearman": spearman_safe(y_true, y_pred),
            "test_pearson": pearson_safe(y_true, y_pred),
            "test_nmae_iqr": nmae_iqr,
            "test_nrmse_iqr": nrmse_iqr,
            "feature_groups": ",".join(sorted(feature_groups_used)),
            "feature_group_counts": json.dumps(feature_group_counts, sort_keys=True),
            "dropped_feature_columns": ";".join(task.dropped_feature_columns),
        }
        row.update(uncertainty_metrics)
        return row

    split = task_split or make_task_split(task, test_size=test_size, random_state=random_state)
    train_index = split.train_index
    test_index = split.test_index
    split_strategy = split.split_strategy
    X_train = task.X.iloc[train_index].reset_index(drop=True)
    X_test = task.X.iloc[test_index].reset_index(drop=True)
    if pitting_magpie_features:
        X_train, X_test = augment_pitting_magpie_split(X_train, X_test)
    y_train = task.y.iloc[train_index].reset_index(drop=True)
    y_test = task.y.iloc[test_index].reset_index(drop=True)
    y_test_ordinal = task.y_ordinal.iloc[test_index].reset_index(drop=True)
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

    model_source = getattr(
        estimator,
        "model_source_",
        getattr(estimator, "model_path_", ""),
    )
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
        "n_features": int(X_train.shape[1]),
        "pitting_magpie_features": bool(pitting_magpie_features),
        "n_classes": int(len(task.class_labels)),
        "class_labels": ",".join(task.class_labels),
        "class_counts": ",".join(f"{label}:{task.class_counts.get(label, 0)}" for label in task.class_labels),
        "positive_label": task.class_labels[-1],
        "positive_rate": float((task.y == task.class_labels[-1]).mean()),
        **split_metadata(task, train_index, test_index, split_strategy),
        "task_quality_flags": ",".join(task.quality_flags),
        "task_family": task.task_family,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_mcc": float(matthews_corrcoef(y_test, y_pred)),
        "test_auroc": auroc,
        "test_roc_auc_ovr_macro": roc_auc_ovr_macro,
        "feature_groups": ",".join(sorted(feature_groups_used)),
        "feature_group_counts": json.dumps(feature_group_counts, sort_keys=True),
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


def default_output_paths(
    local_specs: list[LocalModelSpec] | tuple[LocalModelSpec, ...],
    *,
    checkpoint_part: str | None = None,
    stage_tag: str = "unspecified_stage",
) -> tuple[Path, Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_labels = [spec.label for spec in local_specs]
    checkpoint_stems = {spec.checkpoint_path.stem for spec in local_specs}
    if not model_labels:
        model_part = "pretrained_baselines"
    elif len(model_labels) == 1:
        model_part = slugify(model_labels[0])
    else:
        shown_labels = "_".join(slugify(label) for label in model_labels[:4])
        if len(model_labels) > 4:
            shown_labels = f"{shown_labels}_plus{len(model_labels) - 4}"
        model_part = f"compare_{shown_labels}"
    ckpt_part = checkpoint_part or (
        next(iter(checkpoint_stems))
        if len(checkpoint_stems) == 1
        else "pretrained" if not checkpoint_stems else "mixed_checkpoints"
    )
    base = f"corrosion_eval_{model_part}_{slugify(ckpt_part)}_{slugify(stage_tag)}_{stamp}"
    output_dir = DEFAULT_OUTPUT_DIR / base
    return (
        output_dir / "results.json",
        output_dir / "rows.csv",
        output_dir / "wide.csv",
        output_dir / "summary.csv",
    )


def derived_csv_path(output_csv: Path, suffix: str) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_{suffix}{output_csv.suffix}")


def infer_stage_tag(args: argparse.Namespace) -> str:
    stage_tags: list[str] = []
    candidates = [str(args.run_prefix or "")]
    candidates.extend(str(run) for run in (args.run or []))
    candidates.extend(str(path) for path in (args.local_ckpt_path or []))

    for candidate in candidates:
        for match in re.finditer(r"tabicl_(s\d+(?:mini)?)(?:[_/\\-]|$)", candidate):
            tag = match.group(1)
            if tag not in stage_tags:
                stage_tags.append(tag)

    if len(stage_tags) == 1:
        return stage_tags[0]
    if len(stage_tags) > 1:
        return "mixed_stage"
    return "unspecified_stage"


def default_plot_dir(args: argparse.Namespace, output_csv: Path) -> Path:
    return output_csv.parent / "plots"


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
        "regression_uncertainty",
        "regression_quantile_alphas",
        "n_samples",
        "n_train",
        "n_test",
        "n_features",
        "n_classes",
        "class_labels",
        "class_counts",
        "positive_rate",
        "split_strategy",
        "split_group_count",
        "split_train_group_count",
        "split_test_group_count",
        "split_group_overlap_count",
        "include_in_summary",
        "summary_exclusion_reason",
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


def summary_included_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "include_in_summary" not in df.columns:
        return df
    include = df["include_in_summary"].fillna(True).astype(bool)
    return df[include].copy()


def make_summary_dataframe(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    metric_sort_specs: list[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    if not rows and not errors:
        return pd.DataFrame()

    df = summary_included_frame(pd.DataFrame(rows))
    error_df = summary_included_frame(pd.DataFrame(errors))
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


def model_uses_pitting_magpie(args: argparse.Namespace, model_label: str, *, force: bool = False) -> bool:
    requested_labels = {slugify(label) for label in (args.pitting_magpie_model or [])}
    return bool(force or args.pitting_magpie_features or slugify(model_label) in requested_labels)


def validate_pitting_magpie_model_labels(args: argparse.Namespace, available_labels: set[str]) -> None:
    requested_labels = {slugify(label) for label in (args.pitting_magpie_model or [])}
    unknown_labels = sorted(requested_labels.difference(available_labels))
    if unknown_labels:
        raise ValueError(
            "--pitting-magpie-model contains labels that are not part of this evaluation: "
            + ", ".join(unknown_labels)
        )



def print_result_row(row: dict[str, Any]) -> None:
    if row.get("target_binning") == "continuous":
        print(
            f"  {row['model']:<24} "
            f"spearman={format_metric(row.get('test_spearman'))} "
            f"nmae_iqr={format_metric(row.get('test_nmae_iqr'))} "
            f"npinball_iqr={format_metric(row.get('test_npinball_iqr'))} "
            f"cov80={format_metric(row.get('test_interval_80_coverage'))} "
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
        metric_display_cols.extend(["mean_test_mae", "weighted_mean_test_mae"])
        metric_display_cols.extend(["mean_test_nmae_iqr", "weighted_mean_test_nmae_iqr"])
        metric_display_cols.extend(["mean_test_npinball_iqr", "mean_test_interval_80_coverage_error"])
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
            "split_strategy": task.split_strategy,
            "split_group_count": int(task.split_groups.nunique(dropna=False)) if task.split_groups is not None else 0,
            "include_in_summary": task.include_in_summary,
            "summary_exclusion_reason": task.summary_exclusion_reason,
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


def save_model_input_tables(tasks: list[EvalTask], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task in tasks:
        task_dir = output_dir / slugify(task.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        table = task.X.reset_index(drop=True).copy()
        table[task.target] = task.y.reset_index(drop=True)
        output_path = task_dir / "model_input_table.csv"
        table.to_csv(output_path, index=False)
        written.append(output_path)
    return written


REPEATED_SPLIT_VALUE_ARGS = {
    "--output-json",
    "--output-csv",
    "--output-wide-csv",
    "--output-summary-csv",
    "--output-plot-dir",
    "--random-state",
}
REPEATED_SPLIT_DROP_FLAGS = {"--no-checkpoint-plots", "--save-task-tables"}


def strip_repeated_split_driver_args(argv: list[str]) -> list[str]:
    """Remove args controlled by the repeated-split driver before spawning seed runs."""
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--split-seeds":
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        if arg in REPEATED_SPLIT_VALUE_ARGS:
            index += 2
            continue
        if arg in REPEATED_SPLIT_DROP_FLAGS:
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return cleaned


def repeated_split_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    if args.output_csv is not None:
        artifact_dir = args.output_csv.expanduser().resolve().parent
    elif args.output_json is not None:
        artifact_dir = args.output_json.expanduser().resolve().parent
    elif args.output_wide_csv is not None:
        artifact_dir = args.output_wide_csv.expanduser().resolve().parent
    elif args.output_summary_csv is not None:
        artifact_dir = args.output_summary_csv.expanduser().resolve().parent
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = DEFAULT_OUTPUT_DIR / f"corrosion_eval_repeated_splits_{stamp}"

    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else artifact_dir / "results.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else artifact_dir / "rows.csv"
    )
    output_wide_csv = (
        args.output_wide_csv.expanduser().resolve()
        if args.output_wide_csv is not None
        else artifact_dir / "wide.csv"
    )
    output_summary_csv = (
        args.output_summary_csv.expanduser().resolve()
        if args.output_summary_csv is not None
        else artifact_dir / "summary.csv"
    )
    output_plot_dir = (
        args.output_plot_dir.expanduser().resolve()
        if args.output_plot_dir is not None
        else artifact_dir / "plots"
    )
    return output_json, output_csv, output_wide_csv, output_summary_csv, output_plot_dir


def add_split_metadata(records: list[dict[str, Any]], *, seed: int, seed_dir: Path) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["split_seed"] = int(seed)
        item["seed_output_dir"] = str(seed_dir)
        enriched.append(item)
    return enriched


def add_split_counts_to_summary(summary: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if summary.empty or not rows:
        return summary
    row_df = pd.DataFrame(rows)
    if "split_seed" not in row_df.columns or "model" not in row_df.columns:
        return summary

    group_cols = ["model"]
    if "checkpoint_name" in row_df.columns and "checkpoint_name" in summary.columns:
        group_cols = ["checkpoint_name", "model"]
    counts = (
        row_df.groupby(group_cols, dropna=False)["split_seed"]
        .nunique()
        .reset_index(name="n_splits")
    )
    merged = summary.merge(counts, on=group_cols, how="left")
    ordered_cols = list(merged.columns)
    if "n_splits" in ordered_cols:
        ordered_cols.remove("n_splits")
        insert_at = ordered_cols.index("n_success") if "n_success" in ordered_cols else len(ordered_cols)
        ordered_cols.insert(insert_at, "n_splits")
        merged = merged[ordered_cols]
    return merged


def make_repeated_split_wide_dataframe(rows: list[dict[str, Any]], summary: pd.DataFrame) -> pd.DataFrame:
    if not rows or summary.empty:
        return pd.DataFrame()
    row_df = pd.DataFrame(rows)
    has_checkpoint = "checkpoint_name" in row_df.columns and "checkpoint_name" in summary.columns
    group_cols = ["task_id"]
    if has_checkpoint:
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
        "regression_uncertainty",
        "regression_quantile_alphas",
        "n_samples",
        "n_train",
        "n_test",
        "n_features",
        "n_classes",
        "class_labels",
        "class_counts",
        "positive_rate",
        "split_strategy",
        "include_in_summary",
        "summary_exclusion_reason",
        "task_quality_flags",
    ]
    records: list[dict[str, Any]] = []
    for _, group in row_df.groupby(group_cols, sort=False, dropna=False):
        first = group.iloc[0]
        record = {col: first[col] for col in meta_cols if col in group.columns}
        if "split_seed" in group.columns:
            seeds = sorted({int(seed) for seed in pd.to_numeric(group["split_seed"], errors="coerce").dropna()})
            record["split_seeds"] = " ".join(str(seed) for seed in seeds)
            record["n_splits"] = len(seeds)

        if has_checkpoint:
            summary_group = summary[summary["checkpoint_name"].astype(str) == str(first["checkpoint_name"])]
        else:
            summary_group = summary
        for _, summary_row in summary_group.iterrows():
            model = str(summary_row["model"])
            model_key = slugify(model)
            for metric in METRIC_COLUMNS:
                value_col = f"mean_{metric}"
                if value_col in summary_row:
                    record[f"{metric}__{model_key}"] = summary_row[value_col]
        records.append(record)
    return pd.DataFrame(records)


def run_repeated_split_eval(args: argparse.Namespace) -> None:
    seeds = list(args.split_seeds or [])
    if not seeds:
        raise ValueError("--split-seeds must contain at least one seed.")

    output_json, output_csv, output_wide_csv, output_summary_csv, output_plot_dir = repeated_split_output_paths(args)
    artifact_dir = output_json.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for output_path in (output_json, output_csv, output_wide_csv, output_summary_csv):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    task_table_paths: list[Path] = []
    if args.save_task_tables:
        task_table_paths = save_model_input_tables(make_tasks(args), artifact_dir / "task_tables")

    base_argv = strip_repeated_split_driver_args(sys.argv[1:])
    all_rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    seed_outputs: list[dict[str, Any]] = []

    for seed in seeds:
        seed_dir = artifact_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_json = seed_dir / "results.json"
        seed_rows = seed_dir / "rows.csv"
        seed_wide = seed_dir / "wide.csv"
        seed_summary = seed_dir / "summary.csv"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_argv,
            "--random-state",
            str(seed),
            "--output-json",
            str(seed_json),
            "--output-csv",
            str(seed_rows),
            "--output-wide-csv",
            str(seed_wide),
            "--output-summary-csv",
            str(seed_summary),
            "--no-checkpoint-plots",
        ]
        print(f"\nRepeated split seed {seed}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        payload = json.loads(seed_json.read_text(encoding="utf-8"))
        all_rows.extend(add_split_metadata(payload.get("rows", []), seed=seed, seed_dir=seed_dir))
        all_errors.extend(add_split_metadata(payload.get("errors", []), seed=seed, seed_dir=seed_dir))
        seed_outputs.append(
            {
                "split_seed": int(seed),
                "json": str(seed_json),
                "rows_csv": str(seed_rows),
                "wide_csv": str(seed_wide),
                "summary_csv": str(seed_summary),
            }
        )

    primary_metric = primary_metric_for_args(args)
    metric_sort_specs = metric_sort_specs_for_args(args)
    summary_df = print_summary(
        all_rows,
        all_errors,
        primary_metric=primary_metric,
        metric_sort_specs=metric_sort_specs,
    )
    summary_df = add_split_counts_to_summary(summary_df, all_rows)
    wide_df = make_repeated_split_wide_dataframe(all_rows, summary_df)

    plot_paths: list[Path] = []
    if args.checkpoint == "all" and args.checkpoint_plots:
        plot_metrics = list(args.plot_metric or default_plot_metrics_for_args(args))
        plot_paths = write_checkpoint_trend_plots(summary_df, output_plot_dir, plot_metrics)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repeated_split_eval": True,
        "split_seeds": seeds,
        "seed_outputs": seed_outputs,
        "runs": expand_runs(args),
        "checkpoint": args.checkpoint,
        "target_mode": args.target_mode,
        "target_binning": args.target_binning,
        "target_bins": args.target_bins,
        "regression_output": args.regression_output,
        "regression_uncertainty": args.regression_uncertainty,
        "primary_metric": primary_metric,
        "metric_sort_specs": metric_sort_specs,
        "test_size": args.test_size,
        "n_estimators": args.n_estimators,
        "tabicl_feat_shuffle_method": args.tabicl_feat_shuffle_method,
        "compare_catboost": bool(args.compare_catboost),
        "compare_catboost_magpie": bool(args.compare_catboost_magpie),
        "pitting_magpie_features": bool(args.pitting_magpie_features),
        "pitting_magpie_models": sorted({slugify(label) for label in args.pitting_magpie_model}),
        "catboost_settings": {
            "iterations": args.catboost_iterations,
            "depth": args.catboost_depth,
            "learning_rate": args.catboost_learning_rate,
            "l2_leaf_reg": args.catboost_l2_leaf_reg,
            "thread_count": args.catboost_thread_count,
        } if args.compare_catboost or args.compare_catboost_magpie else None,
        "rows": all_rows,
        "errors": all_errors,
        "output_files": {
            "artifact_dir": str(artifact_dir),
            "json": str(output_json),
            "csv": str(output_csv),
            "wide_csv": str(output_wide_csv),
            "summary_csv": str(output_summary_csv),
            "plot_dir": str(output_plot_dir) if plot_paths else None,
            "plots": [str(path) for path in plot_paths],
            "task_tables": [str(path) for path in task_table_paths],
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    wide_df.to_csv(output_wide_csv, index=False)
    summary_df.to_csv(output_summary_csv, index=False)

    print(f"\nSaved repeated-split evaluation artifacts to {artifact_dir}")
    print(f"Saved JSON results to {output_json}")
    print(f"Saved repeated-split row CSV results to {output_csv}")
    print(f"Saved repeated-split wide comparison CSV to {output_wide_csv}")
    print(f"Saved repeated-split model summary CSV to {output_summary_csv}")
    if plot_paths:
        print(f"Saved repeated-split checkpoint trend plots to {output_plot_dir}")
        for path in plot_paths:
            print(f"  {path}")
    if task_table_paths:
        print(f"Saved model input task tables to {artifact_dir / 'task_tables'}")
        for path in task_table_paths:
            print(f"  {path}")
    if all_errors:
        print(f"Completed with {len(all_errors)} task/model errors; see JSON for details.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.split_seeds is not None:
        run_repeated_split_eval(args)
        return

    primary_metric = primary_metric_for_args(args)
    metric_sort_specs = metric_sort_specs_for_args(args)
    is_regression_eval = args.target_binning == "continuous"
    pretrained_checkpoint_version = args.pretrained_checkpoint_version or (
        "tabicl-regressor-v2-20260212.ckpt"
        if is_regression_eval
        else "tabicl-classifier-v2-20260212.ckpt"
    )
    tasks = make_tasks(args)
    if args.epit_split_manifest is not None:
        if args.epit_final_test:
            apply_epit_pipeline_final_test(
                tasks,
                manifest_path=args.epit_split_manifest,
            )
        else:
            apply_epit_pipeline_fold(
                tasks,
                manifest_path=args.epit_split_manifest,
                validation_fold=int(args.epit_validation_fold),
            )
    if args.list_tasks:
        print_task_list(tasks)
        return

    if not tasks:
        raise RuntimeError("No corrosion evaluation tasks were generated. Try --target-mode all or lower --min-samples.")

    summary_task_count = sum(1 for task in tasks if task.include_in_summary)
    diagnostic_task_count = len(tasks) - summary_task_count
    if diagnostic_task_count:
        print(
            f"Generated {len(tasks)} tasks: {summary_task_count} included in checkpoint summaries, "
            f"{diagnostic_task_count} diagnostic-only."
        )
    task_splits = {
        task.task_id: make_task_split(task, test_size=args.test_size, random_state=args.random_state)
        for task in tasks
    }

    checkpoint_specs = resolve_checkpoint_eval_specs(args)
    first_local_specs = checkpoint_specs[0].local_specs
    available_model_labels = {spec.label for spec in first_local_specs}
    if args.compare_pretrained_tabicl:
        available_model_labels.add("pretrained_tabicl_v2")
    if args.compare_tabpfn:
        available_model_labels.add("pretrained_tabpfn")
    if args.compare_catboost:
        available_model_labels.add("catboost")
    if args.compare_catboost_magpie:
        available_model_labels.add("catboost_magpie")
    validate_pitting_magpie_model_labels(args, available_model_labels)

    include_checkpoint_columns = args.checkpoint == "all"
    if args.checkpoint == "all":
        steps = [spec.checkpoint_step for spec in checkpoint_specs if spec.checkpoint_step is not None]
        print(
            f"Evaluating {len(checkpoint_specs)} common checkpoints "
            f"({min(steps)}-{max(steps)} steps) across {len(first_local_specs)} local models."
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    reuse_static_baselines = args.checkpoint == "all" and (
        args.compare_pretrained_tabicl
        or args.compare_catboost
        or args.compare_catboost_magpie
    )
    pretrained_tabicl_reuse_ready = False
    pretrained_tabicl_row_templates_by_task: dict[str, list[dict[str, Any]]] = {}
    pretrained_tabicl_error_templates_by_task: dict[str, list[dict[str, Any]]] = {}
    pretrained_tabicl_reuse_labels: list[str] = []
    for checkpoint_eval in checkpoint_specs:
        model_cache: dict[tuple[Any, ...], CachedTabICLModel] = {}
        jobs: list[dict[str, Any]] = []
        for spec in checkpoint_eval.local_specs:
            model_path = str(spec.checkpoint_path)
            if is_regression_eval:
                factory = lambda model_path=model_path: make_tabicl_regressor(
                    model_path=model_path,
                    checkpoint_version=pretrained_checkpoint_version,
                    device=args.device,
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    allow_auto_download=False,
                    feat_shuffle_method=args.tabicl_feat_shuffle_method,
                    norm_methods=args.tabicl_norm_methods,
                )
                jobs.append(
                    {
                        "model_label": spec.label,
                        "model_kind": "local_tabicl_regressor",
                        "factory": factory,
                        "pitting_magpie_features": model_uses_pitting_magpie(args, spec.label),
                        "cache_key": (
                            "local_tabicl_regressor",
                            model_path,
                            args.device,
                            args.n_estimators,
                            args.tabicl_feat_shuffle_method,
                            tuple(args.tabicl_norm_methods or ()),
                        ),
                    }
                )
            else:
                factory = lambda model_path=model_path: make_tabicl_classifier(
                    model_path=model_path,
                    checkpoint_version=pretrained_checkpoint_version,
                    device=args.device,
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    allow_auto_download=False,
                    feat_shuffle_method=args.tabicl_feat_shuffle_method,
                    norm_methods=args.tabicl_norm_methods,
                )
                jobs.append(
                    {
                        "model_label": spec.label,
                        "model_kind": "local_tabicl_classifier",
                        "factory": factory,
                        "pitting_magpie_features": model_uses_pitting_magpie(args, spec.label),
                        "cache_key": (
                            "local_tabicl_classifier",
                            model_path,
                            args.device,
                            args.n_estimators,
                            args.tabicl_feat_shuffle_method,
                            tuple(args.tabicl_norm_methods or ()),
                        ),
                    }
                )

        if args.compare_pretrained_tabicl and not pretrained_tabicl_reuse_ready:
            if is_regression_eval:
                factory = lambda: make_tabicl_regressor(
                    model_path=None,
                    checkpoint_version=pretrained_checkpoint_version,
                    device=args.device,
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    allow_auto_download=args.baseline_auto_download,
                    feat_shuffle_method=args.tabicl_feat_shuffle_method,
                    norm_methods=args.tabicl_norm_methods,
                )
                jobs.append(
                    {
                        "model_label": "pretrained_tabicl_v2",
                        "model_kind": "pretrained_tabicl_regressor",
                        "factory": factory,
                        "pitting_magpie_features": model_uses_pitting_magpie(args, "pretrained_tabicl_v2"),
                        "cache_key": (
                            "pretrained_tabicl_regressor",
                            pretrained_checkpoint_version,
                            args.device,
                            args.n_estimators,
                            args.tabicl_feat_shuffle_method,
                            tuple(args.tabicl_norm_methods or ()),
                        ),
                        "reuse_static_baseline": reuse_static_baselines,
                    }
                )
            else:
                factory = lambda: make_tabicl_classifier(
                    model_path=None,
                    checkpoint_version=pretrained_checkpoint_version,
                    device=args.device,
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    allow_auto_download=args.baseline_auto_download,
                    feat_shuffle_method=args.tabicl_feat_shuffle_method,
                    norm_methods=args.tabicl_norm_methods,
                )
                jobs.append(
                    {
                        "model_label": "pretrained_tabicl_v2",
                        "model_kind": "pretrained_tabicl_classifier",
                        "factory": factory,
                        "pitting_magpie_features": model_uses_pitting_magpie(args, "pretrained_tabicl_v2"),
                        "cache_key": (
                            "pretrained_tabicl_classifier",
                            pretrained_checkpoint_version,
                            args.device,
                            args.n_estimators,
                            args.tabicl_feat_shuffle_method,
                            tuple(args.tabicl_norm_methods or ()),
                        ),
                        "reuse_static_baseline": reuse_static_baselines,
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
                    "pitting_magpie_features": model_uses_pitting_magpie(args, "pretrained_tabpfn"),
                    "cache_key": None,
                }
            )
        if args.compare_catboost and not pretrained_tabicl_reuse_ready:
            catboost_factory = lambda: make_catboost_regressor(
                iterations=args.catboost_iterations,
                depth=args.catboost_depth,
                learning_rate=args.catboost_learning_rate,
                l2_leaf_reg=args.catboost_l2_leaf_reg,
                random_state=args.random_state,
                thread_count=args.catboost_thread_count,
            )
            jobs.append(
                {
                    "model_label": "catboost",
                    "reuse_static_baseline": reuse_static_baselines,
                    "model_kind": "catboost_regressor",
                    "factory": catboost_factory,
                    "pitting_magpie_features": model_uses_pitting_magpie(args, "catboost"),
                    "cache_key": None,
                }
            )
        if args.compare_catboost_magpie and not pretrained_tabicl_reuse_ready:
            catboost_magpie_factory = lambda: make_catboost_regressor(
                iterations=args.catboost_iterations,
                depth=args.catboost_depth,
                learning_rate=args.catboost_learning_rate,
                l2_leaf_reg=args.catboost_l2_leaf_reg,
                random_state=args.random_state,
                thread_count=args.catboost_thread_count,
            )
            jobs.append(
                {
                    "reuse_static_baseline": reuse_static_baselines,
                    "model_label": "catboost_magpie",
                    "model_kind": "catboost_regressor",
                    "factory": catboost_magpie_factory,
                    "pitting_magpie_features": True,
                    "cache_key": None,
                }
            )

        ensure_unique_job_labels(jobs)
        if args.model_cache:
            for job in jobs:
                cache_key = job.get("cache_key")
                if cache_key is None:
                    continue
                job["factory"] = make_cached_tabicl_factory(
                    job["factory"],
                    cache_key=cache_key,
                    model_cache=model_cache,
                )

        checkpoint_header = f"Checkpoint {checkpoint_eval.checkpoint_name}"
        reused_model_count = len(pretrained_tabicl_reuse_labels) if pretrained_tabicl_reuse_ready else 0
        if reused_model_count:
            print(
                f"\n{checkpoint_header}: evaluating {len(tasks)} tasks x "
                f"{len(jobs) + reused_model_count} models "
                f"({len(jobs)} computed, {reused_model_count} reused)."
            )
        else:
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
                        task_split=task_splits[task.task_id],
                        test_size=args.test_size,
                        random_state=args.random_state,
                        regression_output=args.regression_output,
                        regression_quantile_alphas=args.regression_quantile_alphas,
                        regression_uncertainty=args.regression_uncertainty,
                        pitting_magpie_features=bool(job["pitting_magpie_features"]),
                    )
                    if job.get("reuse_static_baseline"):
                        pretrained_tabicl_row_templates_by_task.setdefault(task.task_id, []).append(dict(row))
                        if job["model_label"] not in pretrained_tabicl_reuse_labels:
                            pretrained_tabicl_reuse_labels.append(job["model_label"])
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
                        "pitting_magpie_features": bool(job["pitting_magpie_features"]),
                        "task_id": task.task_id,
                        "dataset": task.dataset,
                        "table": task.table,
                        "target": task.target,
                        "task_quality_flags": ",".join(task.quality_flags),
                        "split_strategy": task.split_strategy,
                        "include_in_summary": bool(task.include_in_summary),
                        "summary_exclusion_reason": task.summary_exclusion_reason,
                        "error": repr(exc),
                    }
                    if job.get("reuse_static_baseline"):
                        pretrained_tabicl_error_templates_by_task.setdefault(task.task_id, []).append(dict(error))
                        if job["model_label"] not in pretrained_tabicl_reuse_labels:
                            pretrained_tabicl_reuse_labels.append(job["model_label"])
                    if include_checkpoint_columns:
                        error["checkpoint_name"] = checkpoint_eval.checkpoint_name
                        error["checkpoint_step"] = checkpoint_eval.checkpoint_step
                    errors.append(error)
                    print(f"  {job['model_label']:<24} ERROR {exc!r}")
                    if args.print_json_lines:
                        print(json.dumps(error, sort_keys=True))

            if pretrained_tabicl_reuse_ready:
                for template in pretrained_tabicl_row_templates_by_task.get(task.task_id, []):
                    row = dict(template)
                    if include_checkpoint_columns:
                        row["checkpoint_name"] = checkpoint_eval.checkpoint_name
                        row["checkpoint_step"] = checkpoint_eval.checkpoint_step
                    rows.append(row)
                    print_result_row(row)
                    if args.print_json_lines:
                        print(json.dumps(row, sort_keys=True))
                for template in pretrained_tabicl_error_templates_by_task.get(task.task_id, []):
                    error = dict(template)
                    if include_checkpoint_columns:
                        error["checkpoint_name"] = checkpoint_eval.checkpoint_name
                        error["checkpoint_step"] = checkpoint_eval.checkpoint_step
                    errors.append(error)
                    print(f"  {error['model']:<24} ERROR {error['error']} (reused)")
                    if args.print_json_lines:
                        print(json.dumps(error, sort_keys=True))

        if reuse_static_baselines and not pretrained_tabicl_reuse_ready:
            pretrained_tabicl_reuse_ready = True

    output_json, output_csv, output_wide_csv, output_summary_csv = default_output_paths(
        first_local_specs,
        checkpoint_part="all_common_checkpoints" if args.checkpoint == "all" else None,
        stage_tag=infer_stage_tag(args),
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

    task_table_paths: list[Path] = []
    if args.save_task_tables:
        task_table_paths = save_model_input_tables(tasks, output_json.parent / "task_tables")

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
        plot_metrics = list(args.plot_metric or default_plot_metrics_for_args(args))
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
        "include_latest_common_checkpoint": args.include_latest_common_checkpoint,
        "target_mode": args.target_mode,
        "target_binning": args.target_binning,
        "target_bins": args.target_bins,
        "regression_output": args.regression_output,
        "regression_uncertainty": args.regression_uncertainty,
        "regression_quantile_alphas": list(args.regression_quantile_alphas),
        "pretrained_checkpoint_version": pretrained_checkpoint_version,
        "model_cache": args.model_cache,
        "primary_metric": primary_metric,
        "metric_sort_specs": metric_sort_specs,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "datacortech_protocol": args.datacortech_protocol,
        "max_samples_per_task": args.max_samples_per_task,
        "min_numeric_finite_ratio": args.min_numeric_finite_ratio,
        "min_categorical_nonmissing_ratio": args.min_categorical_nonmissing_ratio,
        "default_excluded_datasets": list(DEFAULT_EXCLUDED_DATASETS),
        "default_excluded_tasks": list(DEFAULT_EXCLUDED_TASKS),
        "default_summary_excluded_datasets": list(DEFAULT_SUMMARY_EXCLUDED_DATASETS),
        "excluded_quality_flags": list(args.exclude_quality_flag or []),
        "compare_catboost_magpie": bool(args.compare_catboost_magpie),
        "pitting_magpie_models": sorted({slugify(label) for label in args.pitting_magpie_model}),
        "n_estimators": args.n_estimators,
        "tabicl_feat_shuffle_method": args.tabicl_feat_shuffle_method,
        "compare_catboost": bool(args.compare_catboost),
        "catboost_settings": {
            "iterations": args.catboost_iterations,
            "depth": args.catboost_depth,
            "learning_rate": args.catboost_learning_rate,
            "l2_leaf_reg": args.catboost_l2_leaf_reg,
            "thread_count": args.catboost_thread_count,
        } if args.compare_catboost or args.compare_catboost_magpie else None,
        "feature_groups_default": list(DEFAULT_FEATURE_GROUPS),
        "pitting_magpie_features": bool(args.pitting_magpie_features),
        "pitting_magpie_version": (
            EPIT_MAGPIE_VERSION
            if args.pitting_magpie_features or args.pitting_magpie_model or args.compare_catboost_magpie
            else None
        ),
        "electrochem_feature_groups": list(ELECTROCHEM_FEATURE_GROUPS),
        "include_electrochem_features": args.include_electrochem_features,
        "rows": rows,
        "errors": errors,
        "output_files": {
            "artifact_dir": str(output_json.parent),
            "json": str(output_json),
            "csv": str(output_csv),
            "wide_csv": str(output_wide_csv),
            "summary_csv": str(output_summary_csv),
            "plot_dir": str(output_plot_dir) if output_plot_dir is not None else None,
            "plots": [str(path) for path in plot_paths],
            "task_tables": [str(path) for path in task_table_paths],
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    wide_df.to_csv(output_wide_csv, index=False)
    summary_df.to_csv(output_summary_csv, index=False)

    print(f"\nSaved evaluation artifacts to {output_json.parent}")
    print(f"Saved JSON results to {output_json}")
    print(f"Saved row CSV results to {output_csv}")
    print(f"Saved wide comparison CSV to {output_wide_csv}")
    print(f"Saved model summary CSV to {output_summary_csv}")
    if plot_paths:
        print(f"Saved checkpoint trend plots to {output_plot_dir}")
        for path in plot_paths:
            print(f"  {path}")
    if task_table_paths:
        print(f"Saved model input task tables to {output_json.parent / 'task_tables'}")
        for path in task_table_paths:
            print(f"  {path}")
    if errors:
        print(f"Completed with {len(errors)} task/model errors; see JSON for details.")


if __name__ == "__main__":
    main()
