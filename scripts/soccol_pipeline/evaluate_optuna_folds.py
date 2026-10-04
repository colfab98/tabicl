#!/usr/bin/env python
"""Evaluate one Optuna trial on the frozen Soccol development folds."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import evaluate_optuna_folds as base  # noqa: E402
from scripts.epit_pipeline.artifact_hashes import load_frozen_split  # noqa: E402
from scripts.eval_corrosion_datasets import SOCCOL_PIPELINE_TASK_ID  # noqa: E402


SOCCOL_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline"
base.DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "soccol_pitting_potential"
    / "processed"
    / "splits_v1"
    / "split_manifest.json"
)
base.DEFAULT_OUTPUT_ROOT = SOCCOL_ROOT / "optuna_v1" / "evaluations"
base.PITTING_TASK_ID = SOCCOL_PIPELINE_TASK_ID
base.load_frozen_split = partial(
    load_frozen_split,
    expected_manifest_schema="soccol_composition_split_manifest_v1",
)


if __name__ == "__main__":
    base.main()
