import random

import numpy as np
import pandas as pd
import torch

from scripts.eval_corrosion_datasets import augment_pitting_magpie_split
from tabicl.prior.dataset import SCMPrior
from tabicl.prior.magpie_features import (
    EPIT_MAGPIE_DESCRIPTOR_NAMES,
    EPIT_MAGPIE_MATERIAL_COLUMNS,
    EPIT_MAGPIE_TOTAL_FEATURE_COUNT,
    append_magpie_descriptors_torch,
    magpie_descriptors_numpy,
    magpie_descriptors_torch,
)
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.prior.epit_schema import (
    EPIT_BASE_FEATURE_COUNT,
    EPIT_COMPOSITION_COLUMNS,
    EPIT_MATERIAL_FEATURE_COUNT,
)


def _magpie_fixed_hp() -> dict:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "informed_mix_probs": (1.0, 0.0),
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": (24, 3, 1, 0, 0, 0, 0, 0, 0),
            "informed_normal_block_allocation_min_counts": (24, 3, 1, 0, 0, 0, 0, 0, 0),
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 1.0,
            "pitting_material_style_probs": (1.0, 0.0, 0.0, 0.0, 0.0),
            "pitting_material_dirichlet_prob": 0.0,
            "pitting_process_role": "test_method_category",
            "pitting_process_category_count": 3,
            "pitting_fixed_epit_schema": True,
            "cat_prob": 0.0,
            "permute_features": False,
            "permute_labels": False,
        }
    )
    return fixed_hp


def _make_prior(
    fixed_hp: dict,
    feature_count: int,
    *,
    prior_type: str = "informed_scm",
    informed_prior_ratio: float = 0.5,
) -> SCMPrior:
    return SCMPrior(
        batch_size=1,
        batch_size_per_gp=1,
        min_features=feature_count,
        max_features=feature_count,
        max_classes=0,
        max_seq_len=64,
        min_train_size=0.4,
        max_train_size=0.8,
        prior_type=prior_type,
        informed_prior_ratio=informed_prior_ratio,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )


def test_pure_element_descriptors_match_lookup_values():
    pure_fe = np.zeros((1, 17), dtype=float)
    pure_fe[0, 0] = 100.0

    descriptors = magpie_descriptors_numpy(pure_fe)[0]

    assert np.allclose(
        descriptors,
        [1.83, 0.0, 0.0, 132.0, 0.0, 1811.0, 0.0, 6.0, 8.0, 4.0],
    )


def test_numpy_and_torch_descriptor_implementations_match():
    compositions = np.array(
        [
            [70.0, 20.0, 10.0, *([0.0] * 14)],
            [0.0, 20.0, 65.0, 15.0, *([0.0] * 13)],
        ]
    )

    expected = magpie_descriptors_numpy(compositions)
    actual = magpie_descriptors_torch(torch.tensor(compositions, dtype=torch.float64)).numpy()

    assert expected.shape == (2, 10)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_descriptor_calculation_rejects_negative_values_instead_of_repairing_them():
    composition = torch.ones(2, 17)
    composition[0, 3] = -0.1

    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        magpie_descriptors_torch(composition)


def test_append_preserves_all_base_feature_values():
    X = torch.rand(8, EPIT_BASE_FEATURE_COUNT)
    X[:, :EPIT_MATERIAL_FEATURE_COUNT] = (
        torch.softmax(X[:, :EPIT_MATERIAL_FEATURE_COUNT], dim=-1) * 100.0
    )

    augmented = append_magpie_descriptors_torch(X)

    assert augmented.shape == (8, EPIT_MAGPIE_TOTAL_FEATURE_COUNT)
    assert torch.equal(augmented[:, :EPIT_BASE_FEATURE_COUNT], X)


def test_disabled_magpie_flag_reproduces_absent_flag_generation_exactly():
    without_key = _magpie_fixed_hp()
    without_key.pop("pitting_magpie_features", None)
    explicit_false = {**without_key, "pitting_magpie_features": False}

    def generate(fixed_hp: dict):
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        return _make_prior(fixed_hp, EPIT_BASE_FEATURE_COUNT).get_batch()

    absent_batch = generate(without_key)
    false_batch = generate(explicit_false)

    for absent, explicit in zip(absent_batch, false_batch):
        assert torch.equal(absent, explicit)


def test_enabled_magpie_prior_generates_38_features_from_28_feature_schema(monkeypatch):
    import tabicl.prior.dataset as prior_dataset

    seen_widths: list[int] = []
    original_mlp = prior_dataset.MLPSCM

    class RecordingMLP(original_mlp):
        def __init__(self, *args, **kwargs):
            seen_widths.append(int(kwargs["num_features"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(prior_dataset, "MLPSCM", RecordingMLP)
    fixed_hp = {**_magpie_fixed_hp(), "pitting_magpie_features": True}
    random.seed(92)
    np.random.seed(92)
    torch.manual_seed(92)

    X, y, d, _, _ = _make_prior(fixed_hp, EPIT_MAGPIE_TOTAL_FEATURE_COUNT).get_batch()

    assert seen_widths == [EPIT_BASE_FEATURE_COUNT]
    assert X.shape == (1, 64, EPIT_MAGPIE_TOTAL_FEATURE_COUNT)
    assert y.shape == (1, 64)
    assert d.tolist() == [EPIT_MAGPIE_TOTAL_FEATURE_COUNT]
    assert torch.isfinite(X).all()


def test_magpie_enabled_hybrid_generic_task_has_28_active_features_and_no_descriptors(monkeypatch):
    import tabicl.prior.dataset as prior_dataset

    seen_widths: list[int] = []
    original_mlp = prior_dataset.MLPSCM

    class RecordingMLP(original_mlp):
        def __init__(self, *args, **kwargs):
            seen_widths.append(int(kwargs["num_features"]))
            super().__init__(*args, **kwargs)

    def reject_pitting_branch(*args, **kwargs):
        raise AssertionError("Generic hybrid tasks must not enter the pitting/PREN branch.")

    monkeypatch.setattr(prior_dataset, "MLPSCM", RecordingMLP)
    monkeypatch.setattr(
        prior_dataset.SCMPrior,
        "_apply_pitting_potential_profile",
        reject_pitting_branch,
    )
    monkeypatch.setattr(
        prior_dataset.SCMPrior,
        "_apply_informed_pitting_potential_mechanism",
        reject_pitting_branch,
    )
    fixed_hp = {**_magpie_fixed_hp(), "pitting_magpie_features": True}
    random.seed(93)
    np.random.seed(93)
    torch.manual_seed(93)

    X, y, d, _, _ = _make_prior(
        fixed_hp,
        EPIT_MAGPIE_TOTAL_FEATURE_COUNT,
        prior_type="hybrid_scm",
        informed_prior_ratio=0.0,
    ).get_batch()

    assert seen_widths == [EPIT_BASE_FEATURE_COUNT]
    assert X.shape == (1, 64, EPIT_MAGPIE_TOTAL_FEATURE_COUNT)
    assert y.shape == (1, 64)
    assert d.tolist() == [EPIT_BASE_FEATURE_COUNT]
    assert torch.count_nonzero(X[..., EPIT_BASE_FEATURE_COUNT:]) == 0
    assert torch.isfinite(X).all()


def test_eval_augmentation_uses_train_means_and_keeps_original_column_order():
    material_columns = list(EPIT_MAGPIE_MATERIAL_COLUMNS)
    full_columns = list(EPIT_COMPOSITION_COLUMNS)
    legacy_train = np.arange(3 * 17, dtype=float).reshape(3, 17) + 1.0
    legacy_test = np.arange(2 * 17, dtype=float).reshape(2, 17) + 2.0
    legacy_train[1, 1] = np.nan
    legacy_test[0, 1] = np.nan
    X_train = pd.DataFrame(0.0, index=range(3), columns=full_columns)
    X_test = pd.DataFrame(0.0, index=range(2), columns=full_columns)
    X_train.loc[:, material_columns] = legacy_train
    X_test.loc[:, material_columns] = legacy_test
    extra_columns = ["Test Temp. oC", "[Cl-] M", "[Cl-] pH", "[Cl-] Test Method"]
    X_train[extra_columns] = [[20.0, 0.1, 7.0, "A"]] * 3
    X_test[extra_columns] = [[25.0, 0.2, 6.5, "B"]] * 2

    train_augmented, test_augmented = augment_pitting_magpie_split(X_train, X_test)

    expected_cr_mean = np.nanmean(legacy_train[:, 1])
    assert train_augmented.columns[:28].tolist() == full_columns + extra_columns
    assert train_augmented.columns[28:].tolist() == list(EPIT_MAGPIE_DESCRIPTOR_NAMES)
    assert train_augmented.loc[1, "Composition, wt.% Cr"] == expected_cr_mean
    assert test_augmented.loc[0, "Composition, wt.% Cr"] == expected_cr_mean
    assert np.isnan(X_train.loc[1, "Composition, wt.% Cr"])
    assert np.allclose(
        train_augmented.loc[:, EPIT_MAGPIE_DESCRIPTOR_NAMES].to_numpy(),
        magpie_descriptors_numpy(train_augmented.loc[:, material_columns].to_numpy()),
    )
