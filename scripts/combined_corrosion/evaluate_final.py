#!/usr/bin/env python
"""Evaluate the combined checkpoint and standard baselines on frozen MPEA queries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "combined_epit_soccol_context__mpea_pitting_potential_mv_sce"
DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "combined_epit_soccol_mpea"
    / "processed"
)
DEFAULT_CANDIDATE_RUN = (
    REPO_ROOT / "checkpoints" / "combined_epit_soccol_mpea" / "final_seed42"
)
DEFAULT_GENERIC_RUN = REPO_ROOT / "checkpoints" / "tabicl_s1_regression_baseline"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "combined_epit_soccol_mpea"
    / "checkpoint_comparison"
)
CATEGORICAL_COLUMNS = (
    "Prep_medium",
    "CP_aeration",
    "CP_agitation",
    "CP_anions_info",
    "test_method",
    "reference_type",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--generic-run", type=Path, default=DEFAULT_GENERIC_RUN)
    parser.add_argument("--combined-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", default="all")
    parser.add_argument("--min-checkpoint-step", type=int, default=0)
    parser.add_argument("--checkpoint-step-interval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--catboost-thread-count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def evaluation_command(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir.expanduser().resolve()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--run",
        str(args.candidate_run.expanduser().resolve()),
        "--run",
        str(args.generic_run.expanduser().resolve()),
        "--local-model-label",
        "combined_corrpfn",
        "--local-model-label",
        "generic_tabicl",
        "--fixed-schema-model",
        "combined_corrpfn",
        "--missing-aware-model",
        "combined_corrpfn",
        "--native-tabicl-baselines",
        "--checkpoint",
        str(args.checkpoint),
        "--min-checkpoint-step",
        str(args.min_checkpoint_step),
        "--checkpoint-step-interval",
        str(args.checkpoint_step_interval),
        "--include-latest-common-checkpoint",
        "--task",
        TASK_ID,
        "--combined-data-dir",
        str(args.combined_data_dir.expanduser().resolve()),
        "--target-mode",
        "primary",
        "--target-binning",
        "continuous",
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
    ]
    for column in CATEGORICAL_COLUMNS:
        command.extend(["--catboost-categorical-column", column])
    command.extend(
        [
            "--output-json",
            str(output_dir / "results.json"),
            "--output-csv",
            str(output_dir / "rows.csv"),
            "--output-wide-csv",
            str(output_dir / "wide.csv"),
            "--output-summary-csv",
            str(output_dir / "summary.csv"),
            "--output-plot-dir",
            str(output_dir / "plots"),
        ]
    )
    return command


def main() -> None:
    args = parse_args()
    subprocess.run(evaluation_command(args), cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
