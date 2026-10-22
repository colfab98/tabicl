#!/usr/bin/env python
"""Compare EPIT control/Soccol continuations and baselines on frozen MPEA queries."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.combined_corrosion.evaluate_final import (
    REPO_ROOT,
    add_evaluation_arguments,
    comparison_command,
)


RUN_ROOT = REPO_ROOT / "checkpoints/combined_epit_soccol_mpea"
DEFAULT_CONTROL_RUN = RUN_ROOT / "epit_v2_5000_control_seed42"
DEFAULT_SOCCOL_RUN = RUN_ROOT / "epit_v2_5000_to_soccol_seed42"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "corrosion_datasets/analysis/combined_epit_soccol_mpea/epit5k_continuation_comparison"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, default=DEFAULT_CONTROL_RUN,
                        help="EPIT step-5000 continuation retaining 50%% EPIT / 50%% generic.")
    parser.add_argument("--soccol-run", type=Path, default=DEFAULT_SOCCOL_RUN,
                        help="EPIT step-5000 continuation switched to 100%% Soccol.")
    add_evaluation_arguments(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def evaluation_command(args: argparse.Namespace) -> list[str]:
    return comparison_command(
        args,
        [("epit5k_control", args.control_run), ("epit5k_to_soccol", args.soccol_run),
         ("generic_tabicl", args.generic_run)],
        missing_aware_models=("epit5k_control", "epit5k_to_soccol"),
    )


def main() -> None:
    args = parse_args()
    subprocess.run(evaluation_command(args), cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
