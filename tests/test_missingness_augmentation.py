from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tabicl.prior.dataset import SCMPrior
from tabicl.prior.source_ids import (
    MISSINGNESS_SOURCE_EPIT,
    MISSINGNESS_SOURCE_GENERIC,
    MISSINGNESS_SOURCE_SOCCOL,
)
from tabicl.prior.epit_feature_profile import EPIT_FEATURE_PROFILE
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_PROFILE
from tabicl.train.missingness import MissingnessAugmenter
from tabicl.train.train_config import build_parser


def _write_profile(path: Path, pattern: str, *, count: int = 1) -> Path:
    payload = {
        "schema_version": "test_v1",
        "source": "context only",
        "feature_columns_in_order": ["A", "B", "C", "D"],
        "composition_feature_count": 2,
        "patterns_by_source": {
            "EPIT": {
                "row_count": count,
                "patterns": [{"missing_mask": pattern, "count": count}],
            },
            "Soccol": {
                "row_count": count,
                "patterns": [{"missing_mask": pattern, "count": count}],
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_masks_observed_zero_and_noncomposition_from_joint_pattern(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", "1001")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=1,
    )
    X = torch.tensor([[[0.0, 2.0, 3.0, 4.0]]])
    y = torch.tensor([[7.0]])

    augmented, stats = augmenter.augment(
        X,
        torch.tensor([4]),
        generator=torch.Generator().manual_seed(3),
    )

    assert torch.isnan(augmented[0, 0, 0])
    assert augmented[0, 0, 1:3].tolist() == [2.0, 3.0]
    assert torch.isnan(augmented[0, 0, 3])
    assert y.tolist() == [[7.0]]
    assert stats["missing_mask_fraction"] == pytest.approx(0.5)


def test_retains_minimum_observed_composition_and_never_fills_original_nan(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", "1100")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=2,
    )
    X = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [float("nan"), 2.0, 3.0, 4.0],
            ]
        ]
    )

    augmented, _ = augmenter.augment(
        X,
        torch.tensor([4]),
        generator=torch.Generator().manual_seed(5),
    )

    assert torch.isfinite(augmented[0, 0, :2]).sum().item() == 2
    assert torch.isnan(augmented[0, 1, 0])
    assert torch.isfinite(augmented[0, 1, 1])


def test_does_not_mask_inactive_padding_columns(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", "1111")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=0,
    )
    X = torch.tensor([[[1.0, 2.0, 30.0, 40.0]]])

    augmented, _ = augmenter.augment(X, torch.tensor([2]))

    assert torch.isnan(augmented[0, 0, :2]).all()
    assert augmented[0, 0, 2:].tolist() == [30.0, 40.0]


def test_informed_tables_receive_source_matched_masks(tmp_path: Path) -> None:
    payload = {
        "schema_version": "test_v1",
        "source": "context only",
        "feature_columns_in_order": ["A", "B", "C", "D"],
        "composition_feature_count": 2,
        "patterns_by_source": {
            "EPIT": {"row_count": 1, "patterns": [{"missing_mask": "1000", "count": 1}]},
            "Soccol": {"row_count": 1, "patterns": [{"missing_mask": "0100", "count": 1}]},
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=1,
    )
    X = torch.ones(2, 1, 4)

    augmented, _ = augmenter.augment(
        X,
        torch.tensor([4, 4]),
        source_ids=torch.tensor([MISSINGNESS_SOURCE_EPIT, MISSINGNESS_SOURCE_SOCCOL]),
        generator=torch.Generator().manual_seed(9),
    )

    assert torch.isnan(augmented[0, 0, 0])
    assert torch.isfinite(augmented[0, 0, 1])
    assert torch.isfinite(augmented[1, 0, 0])
    assert torch.isnan(augmented[1, 0, 1])


def test_prior_source_ids_distinguish_generic_epit_and_soccol() -> None:
    assert SCMPrior._missingness_source_id({"informed_mode": False}) == MISSINGNESS_SOURCE_GENERIC
    assert SCMPrior._missingness_source_id(
        {"informed_mode": True, "pitting_feature_profile": EPIT_FEATURE_PROFILE}
    ) == MISSINGNESS_SOURCE_EPIT
    assert SCMPrior._missingness_source_id(
        {"informed_mode": True, "pitting_feature_profile": SOCCOL_FEATURE_PROFILE}
    ) == MISSINGNESS_SOURCE_SOCCOL


def test_training_flags_are_opt_in() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])
    enabled = parser.parse_args(
        [
            "--col_missing_value_embedding",
            "true",
            "--missingness_profile_path",
            "profile.json",
        ]
    )

    assert defaults.col_missing_value_embedding is False
    assert defaults.missingness_profile_path is None
    assert defaults.combined_corrosion_prior_config is None
    assert enabled.col_missing_value_embedding is True
    assert enabled.missingness_profile_path == "profile.json"
