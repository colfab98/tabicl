#!/usr/bin/env python
"""Evaluate the current EPIT checkpoints and baselines on the frozen query set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PITTING_TASK_ID = (
    "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
)
DEFAULT_TRIAL_RUN = (
    REPO_ROOT
    / "checkpoints"
    / "epit_pipeline_final_v2"
    / "final_epit_pipeline_optuna_empirical_features_scm_target_v8_trial_0017"
)
DEFAULT_GENERIC_RUN = REPO_ROOT / "checkpoints" / "tabicl_s1_regression_baseline"
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "epit_pipeline"
    / "splits_v2"
    / "split_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "epit_pipeline"
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
            "average metrics; the frozen outer query set never changes."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--trial-run", type=Path, default=DEFAULT_TRIAL_RUN)
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
            "v8_final_test_context_folds"
            if args.context_folds is not None
            else "v8_final_test_comparison"
        )
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--run",
        str(args.trial_run.expanduser().resolve()),
        "--run",
        str(args.generic_run.expanduser().resolve()),
        "--local-model-label",
        "corrpfn_v8",
        "--local-model-label",
        "generic_baseline",
        "--fixed-schema-model",
        "corrpfn_v8",
        "--checkpoint",
        "all",
        "--min-checkpoint-step",
        "0",
        "--checkpoint-step-interval",
        "0",
        "--task",
        PITTING_TASK_ID,
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
        "--catboost-categorical-column",
        "[Cl-] Test Method",
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
    if args.context_folds is not None:
        command.extend(["--context-folds", str(args.context_folds)])
    return command


def main() -> None:
    args = parse_args()
    subprocess.run(evaluation_command(args), cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
