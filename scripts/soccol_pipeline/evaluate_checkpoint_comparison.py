#!/usr/bin/env python
"""Evaluate current SOCCOL checkpoints and baselines on the frozen query set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOCCOL_TASK_ID = "soccol_pitting_potential__event1__epit_mv_agagcl_3m_kcl"
DEFAULT_CANDIDATE_RUN = (
    REPO_ROOT
    / "checkpoints"
    / "soccol_pipeline_final_v1"
    / "final_soccol_pipeline_optuna_empirical_features_scm_target_v1_trial_0019"
)
DEFAULT_GENERIC_RUN = REPO_ROOT / "checkpoints" / "tabicl_s1_regression_baseline"
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "soccol_pitting_potential"
    / "processed"
    / "splits_v1"
    / "split_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline"
)
CATBOOST_CATEGORICAL_COLUMNS = (
    "Prep_medium",
    "CP_aeration",
    "CP_agitation",
    "CP_anions_info",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-folds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use all N saved development folds separately as context and "
            "average metrics. Omit this option to use the entire development set."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--generic-run", type=Path, default=DEFAULT_GENERIC_RUN)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def evaluation_command(args: argparse.Namespace) -> list[str]:
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / (
            "final_test_context_folds_comparison"
            if args.context_folds is not None
            else "final_test_full_development_comparison"
        )
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--run",
        str(args.candidate_run.expanduser().resolve()),
        "--run",
        str(args.generic_run.expanduser().resolve()),
        "--local-model-label",
        "soccol_trial19",
        "--local-model-label",
        "generic_baseline",
        "--fixed-schema-model",
        "soccol_trial19",
        "--checkpoint",
        "all",
        "--min-checkpoint-step",
        "0",
        "--checkpoint-step-interval",
        "0",
        "--task",
        SOCCOL_TASK_ID,
        "--epit-split-manifest",
        str(args.split_manifest.expanduser().resolve()),
        "--epit-final-test",
        "--device",
        args.device,
        "--n-estimators",
        str(args.n_estimators),
        "--tabicl-feat-shuffle-method",
        "none",
        "--tabicl-norm-methods",
        "none",
        "--no-regression-uncertainty",
        "--compare-pretrained-tabicl",
        "--compare-catboost",
    ]
    for column in CATBOOST_CATEGORICAL_COLUMNS:
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
    if args.context_folds is not None:
        command.extend(["--context-folds", str(args.context_folds)])
    return command


def main() -> None:
    args = parse_args()
    subprocess.run(evaluation_command(args), cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
