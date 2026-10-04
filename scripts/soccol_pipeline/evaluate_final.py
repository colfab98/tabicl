#!/usr/bin/env python
"""Evaluate the frozen Soccol model once on its untouched final-test set."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import evaluate_final as base  # noqa: E402
from scripts.epit_pipeline.artifact_hashes import load_frozen_split  # noqa: E402
from scripts.eval_corrosion_datasets import SOCCOL_PIPELINE_TASK_ID  # noqa: E402
from scripts.soccol_pipeline import train_final as soccol_train  # noqa: E402


SOCCOL_DEVELOPMENT_ROWS = 3222
SOCCOL_FINAL_TEST_ROWS = 805
SOCCOL_FINAL_MODEL_LABEL = "final_soccol_model"


def configure() -> None:
    """Point the unchanged EPIT final evaluator at the frozen Soccol model."""
    soccol_train.configure()
    base.__doc__ = __doc__
    base.DEFAULT_FINAL_ROOT = soccol_train.SOCCOL_FINAL_ROOT
    base.PITTING_TASK_ID = SOCCOL_PIPELINE_TASK_ID
    base.FINAL_MODEL_LABEL = SOCCOL_FINAL_MODEL_LABEL
    base.DEVELOPMENT_ROWS = SOCCOL_DEVELOPMENT_ROWS
    base.FINAL_TEST_ROWS = SOCCOL_FINAL_TEST_ROWS
    base.load_frozen_split = partial(
        load_frozen_split,
        expected_manifest_schema=soccol_train.SOCCOL_SPLIT_SCHEMA,
    )


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
