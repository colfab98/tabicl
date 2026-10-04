from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.eval_corrosion_datasets import preprocess_soccol_split
from tabicl.prior.dataset import SCMPrior
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.prior.soccol_feature_generator import sample_soccol_feature_rows
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile
from tabicl.prior.soccol_schema import (
    SOCCOL_BASE_FEATURE_COUNT,
    SOCCOL_CATEGORICAL_COLUMNS,
    SOCCOL_COMPOSITION_COLUMNS,
    SOCCOL_CONTINUOUS_COLUMNS,
    SOCCOL_FEATURE_COLUMNS,
    SOCCOL_FEATURE_PROFILE,
    SOCCOL_FIXED_BLOCK_ALLOCATION,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RULE_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline" / "target_rules_v1"


def test_soccol_feature_profile_samples_finite_37_column_rows():
    profile = load_soccol_feature_profile()
    batch = sample_soccol_feature_rows(512, profile=profile, random_state=42)

    assert profile.n_rows == 4027
    assert batch.features.shape == (512, SOCCOL_BASE_FEATURE_COUNT)
    assert np.isfinite(batch.features).all()
    composition_sums = batch.features[:, : len(SOCCOL_COMPOSITION_COLUMNS)].sum(axis=1)
    assert np.all(np.isclose(composition_sums, 0.0) | np.isclose(composition_sums, 100.0))


def test_soccol_real_preprocessing_uses_context_only_state():
    train = pd.DataFrame(0.0, index=range(3), columns=SOCCOL_FEATURE_COLUMNS)
    test = pd.DataFrame(0.0, index=range(2), columns=SOCCOL_FEATURE_COLUMNS)
    for column in SOCCOL_CATEGORICAL_COLUMNS:
        train[column] = ["known", None, "known"]
        test[column] = ["known", "unseen"]
    train[SOCCOL_CONTINUOUS_COLUMNS[0]] = [1.0, np.nan, 3.0]
    test[SOCCOL_CONTINUOUS_COLUMNS[0]] = [np.nan, 9.0]

    encoded_train, encoded_test = preprocess_soccol_split(train, test)

    assert encoded_train.shape[1] == encoded_test.shape[1] == 37
    assert np.isfinite(encoded_train.to_numpy()).all()
    assert np.isfinite(encoded_test.to_numpy()).all()
    assert encoded_train.loc[1, SOCCOL_CONTINUOUS_COLUMNS[0]] == 2.0
    assert encoded_test.loc[0, SOCCOL_CONTINUOUS_COLUMNS[0]] == 2.0
    for column in SOCCOL_CATEGORICAL_COLUMNS:
        assert encoded_train.loc[1, column] == -1.0
        assert encoded_test.loc[1, column] == -1.0


def test_soccol_empirical_scm_target_generates_finite_dataset():
    artifact = json.loads(
        (RULE_ROOT / "pren_n_coupled_mns_weak_anions.json").read_text(encoding="utf-8")
    )
    family = artifact["rule_family"]
    coefficients = artifact["final_development_calibration"]["coefficients"]
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "pitting_composition_mode": "empirical_features_scm_target",
            "pitting_feature_profile": SOCCOL_FEATURE_PROFILE,
            "pitting_fixed_epit_schema": True,
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 1.0,
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": SOCCOL_FIXED_BLOCK_ALLOCATION,
            "informed_normal_block_allocation_min_counts": SOCCOL_FIXED_BLOCK_ALLOCATION,
            "informed_feature_block_strength": 0.0,
            "pitting_composition_perturb_strength": 0.05,
            "pitting_magpie_features": False,
            "cat_prob": 0.0,
            "permute_features": False,
            "permute_labels": False,
            "pitting_target_rule_scores": {family: 1.0},
            "pitting_target_rule_coefficients": [
                f"{family}.{name}={value}" for name, value in coefficients.items()
            ],
            "pitting_coefficient_variation": 0.0,
            "pitting_coefficient_upper_bounds": {
                family: artifact["coefficient_upper_bounds"]
            },
        }
    )
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )
    params = {
        **fixed_hp,
        "seq_len": 64,
        "train_size": 40,
        "max_features": SOCCOL_BASE_FEATURE_COUNT,
        "num_features": SOCCOL_BASE_FEATURE_COUNT,
        "num_classes": 0,
        "num_layers": 2,
        "hidden_dim": 12,
        "noise_std": 0.0,
        "prior_type": "mlp_scm",
        "informed_mode": True,
        "device": "cpu",
    }
    np.random.seed(41)
    torch.manual_seed(41)

    X, y, feature_count = prior.generate_dataset(params)

    assert X.shape == (64, 37)
    assert y.shape == (64,)
    assert feature_count.item() == 37
    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert prior.last_pitting_target_rule["target_rule_family"] == family
    assert prior.last_pitting_target_rule["soccol_schema"] is True
