#!/usr/bin/env python
"""Evaluate fixed reference models on frozen Soccol development folds."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import evaluate_baseline_folds as base  # noqa: E402
from scripts.epit_pipeline.artifact_hashes import load_frozen_split  # noqa: E402
from scripts.eval_corrosion_datasets import SOCCOL_PIPELINE_TASK_ID  # noqa: E402
from tabicl.prior.soccol_schema import SOCCOL_CATEGORICAL_COLUMNS  # noqa: E402


SOCCOL_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline"
SOCCOL_SPLIT_MANIFEST = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "soccol_pitting_potential"
    / "processed"
    / "splits_v1"
    / "split_manifest.json"
)
SOCCOL_BASELINE_OUTPUT = SOCCOL_ROOT / "baseline_folds_v1"


def configure() -> None:
    """Point the shared fixed-baseline evaluator at the Soccol development task."""
    base.__doc__ = __doc__
    base.DEFAULT_SPLIT_MANIFEST = SOCCOL_SPLIT_MANIFEST
    base.DEFAULT_OUTPUT_DIR = SOCCOL_BASELINE_OUTPUT
    base.LEGACY_OUTPUT_DIR = None
    base.PITTING_TASK_ID = SOCCOL_PIPELINE_TASK_ID
    base.DEVELOPMENT_ROWS = 3222
    base.CONTEXT_ROW_COUNTS = frozenset({2577, 2578})
    base.VALIDATION_ROW_COUNTS = frozenset({644, 645})
    base.BASELINE_SCHEMA_VERSION = "soccol_development_baselines_v1"
    base.CATBOOST_CATEGORICAL_COLUMNS = tuple(SOCCOL_CATEGORICAL_COLUMNS)
    base.load_frozen_split = partial(
        load_frozen_split,
        expected_manifest_schema="soccol_composition_split_manifest_v1",
    )


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
