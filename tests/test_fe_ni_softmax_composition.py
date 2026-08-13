from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tabicl.prior.dataset import (
    EPIT_FE_FAMILY_ID,
    EPIT_FE_NI_FAMILY_PROBS,
    EPIT_NI_CR_MO_FAMILY_ID,
    SCMPrior,
)
from tabicl.prior.magpie_features import (
    EPIT_MAGPIE_TOTAL_FEATURE_COUNT,
    magpie_descriptors_torch,
)
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.train.train_config import build_parser


def _fe_ni_fixed_hp(*, mode: str | None = "fe_ni_softmax") -> dict:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "informed_mix_probs": (1.0, 0.0),
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": (17, 3, 1, 0, 0, 0, 0, 0, 0),
            "informed_normal_block_allocation_min_counts": (17, 3, 1, 0, 0, 0, 0, 0, 0),
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 1.0,
            "pitting_material_style_probs": (1.0, 0.0, 0.0, 0.0, 0.0),
            "pitting_material_dirichlet_prob": 0.0,
            "pitting_process_role": "test_method_category",
            "pitting_process_category_count": 52,
            "pitting_fixed_epit_schema": True,
            "cat_prob": 0.0,
            "permute_features": False,
            "permute_labels": False,
        }
    )
    if mode is None:
        fixed_hp.pop("pitting_composition_mode", None)
    else:
        fixed_hp["pitting_composition_mode"] = mode
    return fixed_hp


def _make_prior(
    fixed_hp: dict,
    *,
    feature_count: int = 21,
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _profile_draw(mode: str, *, rows: int = 4096):
    fixed_hp = _fe_ni_fixed_hp(mode=mode)
    prior = _make_prior(fixed_hp)
    _seed_everything(712)
    X = torch.randn(rows, 21)
    blocks = {
        "material": slice(0, 17),
        "environment": slice(17, 20),
        "process_history": slice(20, 21),
    }
    profiled, info = prior._apply_pitting_potential_profile(
        X, blocks, "normal_corrosion"
    )
    return prior, profiled, info, blocks


def test_fe_ni_rearrangement_is_an_exact_row_permutation() -> None:
    composition = torch.tensor(
        [
            [3.0, 11.0, 7.0, 5.0, 2.0, 13.0, 17.0],
            [23.0, 2.0, 29.0, 19.0, 31.0, 5.0, 7.0],
        ]
    )
    original = composition.clone()
    family_ids = torch.tensor([EPIT_FE_FAMILY_ID, EPIT_NI_CR_MO_FAMILY_ID])

    rearranged = SCMPrior._rearrange_fe_ni_softmax_composition(
        composition, family_ids
    )
    sorted_original = torch.sort(original, dim=-1, descending=True).values

    assert torch.equal(composition, original)
    assert torch.equal(
        torch.sort(rearranged, dim=-1, descending=True).values,
        sorted_original,
    )
    assert torch.equal(rearranged.sum(dim=-1), original.sum(dim=-1))
    assert rearranged[0, 0] == sorted_original[0, 0]
    assert rearranged[1, 2] == sorted_original[1, 0]
    assert rearranged[1, 1] == sorted_original[1, 1]
    assert rearranged[1, 3] == sorted_original[1, 2]
    assert torch.isfinite(rearranged).all()
    assert torch.all(rearranged >= 0.0)


def test_fe_ni_mode_reuses_the_legacy_softmax_percentages_exactly() -> None:
    _, legacy, _, _ = _profile_draw("legacy")
    prior, rearranged, info, _ = _profile_draw("fe_ni_softmax")
    legacy_material = legacy[:, :17]
    rearranged_material = rearranged[:, :17]

    assert torch.equal(
        torch.sort(rearranged_material, dim=-1).values,
        torch.sort(legacy_material, dim=-1).values,
    )
    assert torch.allclose(
        rearranged_material.sum(dim=-1),
        legacy_material.sum(dim=-1),
        rtol=1e-6,
        atol=1e-5,
    )
    assert torch.isfinite(rearranged_material).all()
    assert torch.all(rearranged_material >= 0.0)
    assert info.material_style == "fe_ni_softmax_composition"
    assert info.material_family_ids is not None
    assert torch.equal(
        info.material_family_ids.cpu(), prior.last_pitting_material_family_ids
    )

    family_ids = info.material_family_ids
    assert torch.all(
        (family_ids == EPIT_FE_FAMILY_ID)
        | (family_ids == EPIT_NI_CR_MO_FAMILY_ID)
    )
    fe_rows = family_ids == EPIT_FE_FAMILY_ID
    ni_rows = family_ids == EPIT_NI_CR_MO_FAMILY_ID
    sorted_material = torch.sort(
        rearranged_material, dim=-1, descending=True
    ).values
    assert torch.equal(rearranged_material[fe_rows, 0], sorted_material[fe_rows, 0])
    assert torch.equal(rearranged_material[ni_rows, 2], sorted_material[ni_rows, 0])
    assert torch.equal(rearranged_material[ni_rows, 1], sorted_material[ni_rows, 1])
    assert torch.equal(rearranged_material[ni_rows, 3], sorted_material[ni_rows, 2])
    observed_ni_fraction = float(ni_rows.float().mean())
    assert observed_ni_fraction == pytest.approx(
        EPIT_FE_NI_FAMILY_PROBS[1], abs=0.02
    )

    descriptors = magpie_descriptors_torch(rearranged_material)
    assert torch.isfinite(descriptors).all()


def test_fe_ni_informed_target_is_epit_only_and_uses_rearranged_rows() -> None:
    prior, X, info, blocks = _profile_draw("fe_ni_softmax", rows=128)
    first_generic_target = torch.linspace(-3.0, 2.0, steps=X.shape[0]).square()
    second_generic_target = torch.flip(first_generic_target, dims=(0,))

    def target_from(generic_target: torch.Tensor) -> torch.Tensor:
        _seed_everything(913)
        _, target = prior._apply_informed_corrosion_mechanism(
            X.clone(),
            generic_target,
            blocks,
            "normal_corrosion",
            interaction_strength=0.35,
            intervention_strength=0.0,
            target_mix_weight=0.2,
            profile_info=info,
        )
        return target

    first = target_from(first_generic_target)
    second = target_from(second_generic_target)

    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()
    assert torch.std(first, unbiased=False) > 0.0
    rule = prior.last_pitting_target_rule
    assert rule is not None
    assert rule["target_mix_weight"] == 1.0
    assert rule["configured_target_mix_weight"] == 0.2
    assert rule["target_mix_policy"] == "epit_only_for_fe_ni_softmax"
    assert np.array_equal(
        rule["synthetic_material_family_ids"],
        info.material_family_ids.cpu().numpy(),
    )
    expected, _ = prior._evaluate_fixed_epit_target_rule_tensor(X, rule)
    assert torch.allclose(second, expected)


def test_explicit_legacy_mode_is_seed_identical_to_the_default() -> None:
    def generate(fixed_hp: dict):
        _seed_everything(421)
        return _make_prior(fixed_hp).get_batch()

    default_batch = generate(_fe_ni_fixed_hp(mode=None))
    explicit_batch = generate(_fe_ni_fixed_hp(mode="legacy"))

    for default_value, explicit_value in zip(default_batch, explicit_batch):
        assert torch.equal(default_value, explicit_value)


def test_generic_hybrid_task_is_unchanged_by_fe_ni_mode() -> None:
    def generate(mode: str):
        _seed_everything(422)
        return _make_prior(
            _fe_ni_fixed_hp(mode=mode),
            prior_type="hybrid_scm",
            informed_prior_ratio=0.0,
        ).get_batch()

    legacy_batch = generate("legacy")
    fe_ni_batch = generate("fe_ni_softmax")

    for legacy_value, fe_ni_value in zip(legacy_batch, fe_ni_batch):
        assert torch.equal(legacy_value, fe_ni_value)


def test_fe_ni_mode_generates_finite_magpie_batches() -> None:
    fixed_hp = {
        **_fe_ni_fixed_hp(),
        "pitting_magpie_features": True,
    }
    prior = _make_prior(
        fixed_hp, feature_count=EPIT_MAGPIE_TOTAL_FEATURE_COUNT
    )
    _seed_everything(423)

    X, y, d, _, _ = prior.get_batch()

    assert X.shape == (1, 64, EPIT_MAGPIE_TOTAL_FEATURE_COUNT)
    assert y.shape == (1, 64)
    assert d.tolist() == [EPIT_MAGPIE_TOTAL_FEATURE_COUNT]
    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert prior.last_pitting_material_family_ids is not None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"pitting_fixed_epit_schema": False}, "fixed_epit_schema"),
        ({"informed_physical_marginal_prob": 0.5}, "marginal_prob=1"),
        ({"pitting_material_dirichlet_prob": 1.0}, "does not permit"),
    ],
)
def test_fe_ni_mode_rejects_incompatible_configuration(
    override: dict, message: str
) -> None:
    fixed_hp = {**_fe_ni_fixed_hp(), **override}
    prior = _make_prior(fixed_hp)

    with pytest.raises(ValueError, match=message):
        prior._apply_pitting_potential_profile(
            torch.randn(16, 21),
            {
                "material": slice(0, 17),
                "environment": slice(17, 20),
                "process_history": slice(20, 21),
            },
            "normal_corrosion",
        )


def test_training_cli_accepts_fe_ni_softmax_mode() -> None:
    args = build_parser().parse_args(
        ["--pitting_composition_mode", "fe_ni_softmax"]
    )

    assert args.pitting_composition_mode == "fe_ni_softmax"
