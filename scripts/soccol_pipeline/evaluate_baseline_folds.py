#!/usr/bin/env python
"""Run the old EPIT all-checkpoint fold comparison on frozen Soccol folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import summarize_checkpoint_comparison  # noqa: E402
from scripts.epit_pipeline.artifact_hashes import load_frozen_split  # noqa: E402
from scripts.epit_pipeline.evaluate_baseline_folds import (  # noqa: E402
    prepare_output_dir,
)
from scripts.eval_corrosion_datasets import SOCCOL_PIPELINE_TASK_ID  # noqa: E402
from tabicl.prior.soccol_schema import SOCCOL_CATEGORICAL_COLUMNS  # noqa: E402


SOCCOL_SPLIT_MANIFEST = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "soccol_pitting_potential"
    / "processed"
    / "splits_v1"
    / "split_manifest.json"
)
DEFAULT_GENERIC_RUN = REPO_ROOT / "checkpoints" / "tabicl_s1_regression_baseline"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "soccol_pipeline"
    / "baseline_folds_v1"
)
VALIDATION_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_MODELS = {
    "generic_baseline",
    "pretrained_tabicl_v2",
    "catboost",
}
EXPECTED_CONTEXT_ROWS = {2577, 2578}
EXPECTED_VALIDATION_ROWS = {644, 645}
EXPECTED_FOLD_PLOTS = {
    "trend_mean_test_spearman.svg",
    "trend_mean_test_mae.svg",
    "trend_mean_test_nmae_iqr.svg",
    "trend_mean_test_nrmse_iqr.svg",
    "trend_mean_test_pearson.svg",
    "trend_mean_test_r2.svg",
}
EXPECTED_AGGREGATE_PLOTS = {
    "trend_mean_test_spearman.svg",
    "trend_mean_test_mae.svg",
    "trend_mean_test_pearson.svg",
    "trend_mean_test_r2.svg",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=SOCCOL_SPLIT_MANIFEST)
    parser.add_argument("--generic-run", type=Path, default=DEFAULT_GENERIC_RUN)
    parser.add_argument("--candidate-run", type=Path, default=None)
    parser.add_argument("--candidate-label", default="soccol_epit")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--auto-output-dir", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-checkpoint-step", type=int, default=500)
    parser.add_argument("--checkpoint-step-interval", type=int, default=500)
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--catboost-thread-count", type=int, default=8)
    parser.add_argument("--print-subcommands", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.split_manifest.expanduser().is_file():
        raise FileNotFoundError(f"Split manifest not found: {args.split_manifest}")
    if not args.generic_run.expanduser().is_dir():
        raise FileNotFoundError(f"Generic run not found: {args.generic_run}")
    if args.candidate_run is not None and not args.candidate_run.expanduser().is_dir():
        raise FileNotFoundError(f"Candidate run not found: {args.candidate_run}")
    positive = {
        "--n-estimators": args.n_estimators,
        "--checkpoint-step-interval": args.checkpoint_step_interval,
        "--catboost-iterations": args.catboost_iterations,
        "--catboost-depth": args.catboost_depth,
        "--catboost-learning-rate": args.catboost_learning_rate,
        "--catboost-thread-count": args.catboost_thread_count,
    }
    for option, value in positive.items():
        if value <= 0:
            raise ValueError(f"{option} must be positive.")
    if args.min_checkpoint_step < 0:
        raise ValueError("--min-checkpoint-step must be nonnegative.")
    if args.catboost_l2_leaf_reg < 0:
        raise ValueError("--catboost-l2-leaf-reg must be nonnegative.")
    load_frozen_split(
        args.split_manifest.expanduser(),
        expected_manifest_schema="soccol_composition_split_manifest_v1",
    )


def fold_command(
    args: argparse.Namespace,
    *,
    fold: int,
    fold_dir: Path,
) -> list[str]:
    run_args = ["--run", str(args.generic_run.expanduser().resolve())]
    label_args = ["--local-model-label", "generic_baseline"]
    if args.candidate_run is not None:
        run_args.extend(["--run", str(args.candidate_run.expanduser().resolve())])
        label_args.extend(["--local-model-label", args.candidate_label])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        *run_args,
        *label_args,
        "--checkpoint",
        "all",
        "--checkpoint-root",
        str(args.checkpoint_root.expanduser().resolve()),
        "--run-prefix",
        "/",
        "--min-checkpoint-step",
        str(args.min_checkpoint_step),
        "--checkpoint-step-interval",
        str(args.checkpoint_step_interval),
        "--include-latest-common-checkpoint",
        "--task",
        SOCCOL_PIPELINE_TASK_ID,
        "--target-mode",
        "primary",
        "--target-binning",
        "continuous",
        "--epit-split-manifest",
        str(args.split_manifest.expanduser().resolve()),
        "--epit-validation-fold",
        str(fold),
        "--device",
        args.device,
        "--n-estimators",
        str(args.n_estimators),
        "--random-state",
        str(args.random_state),
        "--tabicl-feat-shuffle-method",
        "none",
        "--tabicl-norm-methods",
        "none",
        "--regression-output",
        "median",
        "--no-regression-uncertainty",
        "--max-samples-per-task",
        "0",
        "--compare-pretrained-tabicl",
        "--pretrained-checkpoint-version",
        "tabicl-regressor-v2-20260212.ckpt",
        "--compare-catboost",
        "--catboost-iterations",
        str(args.catboost_iterations),
        "--catboost-depth",
        str(args.catboost_depth),
        "--catboost-learning-rate",
        str(args.catboost_learning_rate),
        "--catboost-l2-leaf-reg",
        str(args.catboost_l2_leaf_reg),
        "--catboost-thread-count",
        str(args.catboost_thread_count),
        "--output-json",
        str(fold_dir / "results.json"),
        "--output-csv",
        str(fold_dir / "rows.csv"),
        "--output-wide-csv",
        str(fold_dir / "wide.csv"),
        "--output-summary-csv",
        str(fold_dir / "summary.csv"),
        "--output-plot-dir",
        str(fold_dir / "plots"),
    ]
    for column in SOCCOL_CATEGORICAL_COLUMNS:
        command.extend(["--catboost-categorical-column", column])
    return command


def validate_fold_output(
    fold_dir: Path,
    *,
    fold: int,
    candidate_label: str | None = None,
) -> None:
    result = json.loads((fold_dir / "results.json").read_text(encoding="utf-8"))
    if result.get("errors"):
        raise RuntimeError(f"Fold {fold} reported errors: {result['errors']}")

    rows = pd.read_csv(fold_dir / "rows.csv")
    expected_models = set(EXPECTED_MODELS)
    if candidate_label is not None:
        expected_models.add(candidate_label)
    if set(rows["model"].astype(str)) != expected_models:
        raise RuntimeError(f"Fold {fold} did not produce the expected models.")
    if rows["checkpoint_step"].nunique() < 2:
        raise RuntimeError(f"Fold {fold} did not evaluate a checkpoint series.")
    expected_strategy = f"epit_pipeline_development_fold_{fold}"
    if set(rows["split_strategy"].astype(str)) != {expected_strategy}:
        raise RuntimeError(f"Fold {fold} did not use the frozen development split.")
    context_rows = set(pd.to_numeric(rows["n_train"], errors="raise"))
    if not context_rows <= EXPECTED_CONTEXT_ROWS:
        raise RuntimeError(f"Fold {fold} has an unexpected context size.")
    validation_rows = set(pd.to_numeric(rows["n_test"], errors="raise"))
    if not validation_rows <= EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(f"Fold {fold} has an unexpected validation size.")

    observed_plots = {path.name for path in (fold_dir / "plots").glob("*.svg")}
    missing_plots = EXPECTED_FOLD_PLOTS.difference(observed_plots)
    if missing_plots:
        raise RuntimeError(
            f"Fold {fold} omitted required old-format plots: {sorted(missing_plots)}"
        )


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output_dir = prepare_output_dir(
        args.output_dir,
        auto_output_dir=args.auto_output_dir,
    )
    for fold in VALIDATION_FOLDS:
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        command = fold_command(args, fold=fold, fold_dir=fold_dir)
        if args.print_subcommands:
            print(" ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        validate_fold_output(
            fold_dir,
            fold=fold,
            candidate_label=(
                args.candidate_label if args.candidate_run is not None else None
            ),
        )

    summarize_checkpoint_comparison.run(argparse.Namespace(output_root=output_dir))
    observed_plots = {path.name for path in (output_dir / "plots").glob("*.svg")}
    missing_plots = EXPECTED_AGGREGATE_PLOTS.difference(observed_plots)
    if missing_plots:
        raise RuntimeError(
            "Combined evaluation omitted required old-format plots: "
            f"{sorted(missing_plots)}"
        )
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
