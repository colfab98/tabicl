#!/usr/bin/env python
"""Train and freeze the selected Soccol configuration using the EPIT workflow."""

from __future__ import annotations

import sys
import argparse
from functools import partial
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import train_final as base  # noqa: E402
from scripts.epit_pipeline.artifact_hashes import load_frozen_split  # noqa: E402
from scripts.eval_corrosion_datasets import SOCCOL_PIPELINE_TASK_ID  # noqa: E402
from scripts.soccol_pipeline import run_optuna as soccol_search  # noqa: E402


SOCCOL_PIPELINE_ROOT = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline"
)
SOCCOL_FINAL_ROOT = SOCCOL_PIPELINE_ROOT / "final_v2"
SOCCOL_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "soccol_pipeline_final_v2"
SOCCOL_SPLIT_SCHEMA = "soccol_composition_split_manifest_v1"


def configure(*, reviewed_data: bool = True) -> None:
    """Select reviewed Soccol artifacts, or explicitly reproduce historical v1."""
    soccol_search.configure(reviewed_data=reviewed_data)
    base.__doc__ = __doc__
    base.DEFAULT_FINAL_ROOT = SOCCOL_FINAL_ROOT if reviewed_data else SOCCOL_PIPELINE_ROOT / "final_v1"
    base.DEFAULT_CHECKPOINT_ROOT = SOCCOL_CHECKPOINT_ROOT if reviewed_data else REPO_ROOT / "checkpoints/soccol_pipeline_final_v1"
    base.PITTING_TASK_ID = SOCCOL_PIPELINE_TASK_ID
    base.load_frozen_split = partial(
        load_frozen_split,
        expected_manifest_schema=SOCCOL_SPLIT_SCHEMA,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    mode_parser = soccol_search.data_mode_parser()
    mode, _ = mode_parser.parse_known_args(argv)
    configure(reviewed_data=not mode.legacy_data)
    args = base.parse_args(argv, parents=(mode_parser,), manual_hyperparameters_default=not mode.legacy_data)
    if not mode.legacy_data:
        if args.override_mlp_prob is None:
            args.override_mlp_prob = 0.70
        if args.override_pitting_composition_perturb_strength is None:
            args.override_pitting_composition_perturb_strength = 0.30
        if args.manual_hyperparameters and args.override_informed_target_mix_weight is None:
            args.override_informed_target_mix_weight = 0.28
    return args


def main() -> None:
    base.run(parse_args())


if __name__ == "__main__":
    main()
