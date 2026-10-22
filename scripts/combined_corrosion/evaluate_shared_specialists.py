#!/usr/bin/env python
"""Compare shared-schema EPIT/Soccol V2, combined V2, and baselines on frozen MPEA queries."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.combined_corrosion.evaluate_final import (
    REPO_ROOT,
    add_evaluation_arguments,
    comparison_command,
)


RUN_ROOT = REPO_ROOT / "checkpoints" / "combined_epit_soccol_mpea"
DEFAULT_EPIT_RUN = RUN_ROOT / "epit_shared_v2_seed42"
DEFAULT_SOCCOL_RUN = RUN_ROOT / "soccol_shared_v2_seed42"
DEFAULT_COMBINED_RUN = RUN_ROOT / "final_v2_seed42"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "corrosion_datasets/analysis/combined_epit_soccol_mpea/shared_specialist_comparison"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epit-run", type=Path, default=DEFAULT_EPIT_RUN)
    parser.add_argument("--soccol-run", type=Path, default=DEFAULT_SOCCOL_RUN)
    parser.add_argument("--combined-run", type=Path, default=DEFAULT_COMBINED_RUN)
    parser.add_argument(
        "--skip-combined", action="store_true",
        help="Omit combined V2 from the comparison.",
    )
    add_evaluation_arguments(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def evaluation_command(args: argparse.Namespace) -> list[str]:
    model_runs = [
        ("epit_shared_v2", args.epit_run),
        ("soccol_shared_v2", args.soccol_run),
    ]
    if not args.skip_combined:
        model_runs.append(("combined_corrpfn_v2", args.combined_run))
    missing_aware_models = tuple(label for label, _ in model_runs)
    model_runs.append(("generic_tabicl", args.generic_run))
    return comparison_command(args, model_runs, missing_aware_models=missing_aware_models)


def main() -> None:
    args = parse_args()
    subprocess.run(evaluation_command(args), cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
