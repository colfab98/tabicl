#!/usr/bin/env python
"""Train and freeze the selected Soccol configuration using the EPIT workflow."""

from __future__ import annotations

import sys
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
SOCCOL_FINAL_ROOT = SOCCOL_PIPELINE_ROOT / "final_v1"
SOCCOL_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "soccol_pipeline_final_v1"
SOCCOL_SPLIT_SCHEMA = "soccol_composition_split_manifest_v1"


def configure() -> None:
    """Point the unchanged EPIT final-training workflow at Soccol artifacts."""
    soccol_search.configure()
    base.__doc__ = __doc__
    base.DEFAULT_FINAL_ROOT = SOCCOL_FINAL_ROOT
    base.DEFAULT_CHECKPOINT_ROOT = SOCCOL_CHECKPOINT_ROOT
    base.PITTING_TASK_ID = SOCCOL_PIPELINE_TASK_ID
    base.load_frozen_split = partial(
        load_frozen_split,
        expected_manifest_schema=SOCCOL_SPLIT_SCHEMA,
    )


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
