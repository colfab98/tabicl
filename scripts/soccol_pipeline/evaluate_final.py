#!/usr/bin/env python
"""Evaluate the frozen Soccol model once on its untouched final-test set."""

from __future__ import annotations

import sys
import argparse
from functools import partial
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import evaluate_final as base  # noqa: E402
from scripts.epit_pipeline.artifact_hashes import load_frozen_split  # noqa: E402
from scripts.eval_corrosion_datasets import SOCCOL_PIPELINE_TASK_ID  # noqa: E402
from scripts.soccol_pipeline import train_final as soccol_train  # noqa: E402


SOCCOL_DEVELOPMENT_ROWS = 2950
SOCCOL_FINAL_TEST_ROWS = 805
SOCCOL_FINAL_MODEL_LABEL = "final_soccol_model"


def configure(*, reviewed_data: bool = True) -> None:
    """Evaluate reviewed context against the original outer queries."""
    soccol_train.configure(reviewed_data=reviewed_data)
    base.__doc__ = __doc__
    base.DEFAULT_FINAL_ROOT = soccol_train.base.DEFAULT_FINAL_ROOT
    base.PITTING_TASK_ID = SOCCOL_PIPELINE_TASK_ID
    base.FINAL_MODEL_LABEL = SOCCOL_FINAL_MODEL_LABEL
    base.DEVELOPMENT_ROWS = SOCCOL_DEVELOPMENT_ROWS if reviewed_data else 3222
    base.FINAL_TEST_ROWS = SOCCOL_FINAL_TEST_ROWS
    base.load_frozen_split = partial(
        load_frozen_split,
        expected_manifest_schema=soccol_train.SOCCOL_SPLIT_SCHEMA,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    mode_parser = soccol_train.soccol_search.data_mode_parser()
    mode, _ = mode_parser.parse_known_args(argv)
    configure(reviewed_data=not mode.legacy_data)
    return base.parse_args(argv, parents=(mode_parser,))


def main() -> None:
    base.run(parse_args())


if __name__ == "__main__":
    main()
