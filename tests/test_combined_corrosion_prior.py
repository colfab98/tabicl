from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tabicl.prior.combined_corrosion import (
    COMBINED_FEATURE_COLUMNS,
    COMBINED_FEATURE_INDEX,
    EPIT_FEATURE_COLUMNS,
    CombinedCorrosionPriorDataset,
)
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS
from tabicl.train.train_config import build_parser


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_CONFIG = (
    REPO_ROOT
    / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
    / "combined_prior_config.json"
)


def test_frozen_combined_prior_policy() -> None:
    config = json.loads(PRIOR_CONFIG.read_text(encoding="utf-8"))

    assert tuple(config["feature_columns_in_order"]) == COMBINED_FEATURE_COLUMNS
    assert {
        name: regime["probability"] for name, regime in config["regimes"].items()
    } == {"EPIT": 0.25, "Soccol": 0.5, "generic": 0.25}
    assert config["regimes"]["generic"]["fixed_hp"]["cat_prob"] == 0.2
    assert config["regimes"]["EPIT"]["fixed_hp"]["cat_prob"] == 0.0
    assert config["regimes"]["Soccol"]["fixed_hp"]["cat_prob"] == 0.0
    assert config["policy"]["mpea_used"] is False


def test_source_features_map_by_name_and_leave_unsupported_slots_nan() -> None:
    epit = torch.arange(len(EPIT_FEATURE_COLUMNS), dtype=torch.float32).reshape(1, 1, -1)
    soccol = torch.arange(len(SOCCOL_FEATURE_COLUMNS), dtype=torch.float32).reshape(1, 1, -1)

    mapped_epit = CombinedCorrosionPriorDataset._map_named_features(
        epit, EPIT_FEATURE_COLUMNS
    )
    mapped_soccol = CombinedCorrosionPriorDataset._map_named_features(
        soccol, SOCCOL_FEATURE_COLUMNS
    )

    for source_index, name in enumerate(EPIT_FEATURE_COLUMNS):
        assert mapped_epit[0, 0, COMBINED_FEATURE_INDEX[name]] == source_index
    for source_index, name in enumerate(SOCCOL_FEATURE_COLUMNS):
        assert mapped_soccol[0, 0, COMBINED_FEATURE_INDEX[name]] == source_index
    assert torch.isnan(mapped_epit[0, 0, COMBINED_FEATURE_INDEX["Prep_medium"]])
    assert torch.isnan(mapped_soccol[0, 0, COMBINED_FEATURE_INDEX["test_method"]])
    assert torch.isnan(mapped_soccol[0, 0, COMBINED_FEATURE_INDEX["reference_type"]])


def test_combined_prior_generates_exact_regime_counts_and_width() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    prior = CombinedCorrosionPriorDataset(
        config_path=PRIOR_CONFIG,
        batch_size=4,
        batch_size_per_gp=4,
        max_classes=0,
        min_seq_len=None,
        max_seq_len=64,
        min_train_size=0.1,
        max_train_size=0.9,
        n_jobs=1,
        device="cpu",
    )

    X, y, d, seq_lens, train_sizes, source_ids = prior.get_batch()

    assert X.shape == (4, 64, 50)
    assert y.shape == (4, 64)
    assert d.tolist() == [50] * 4
    assert seq_lens.tolist() == [64] * 4
    assert len(set(train_sizes.tolist())) == 1
    assert source_ids.bincount(minlength=3).tolist() == [1, 1, 2]


def test_combined_prior_training_flag_is_opt_in() -> None:
    parser = build_parser()
    assert parser.parse_args([]).combined_corrosion_prior_config is None
    assert (
        parser.parse_args(
            ["--combined_corrosion_prior_config", "combined_prior.json"]
        ).combined_corrosion_prior_config
        == "combined_prior.json"
    )
