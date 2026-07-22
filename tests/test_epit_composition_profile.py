import numpy as np
import pytest
import torch

import tabicl.prior.dataset as prior_dataset
from tabicl.prior.dataset import SCMPrior
from tabicl.prior.epit_composition_profile import (
    EPIT_COMPOSITION_FAMILY_COUNTS,
    EPIT_COMPOSITION_FAMILY_PROBS,
    EPIT_COMPOSITION_PROFILE,
    load_epit_composition_profile,
    map_epit_latents_to_compositions,
    normalize_epit_family_probabilities,
    sample_epit_compositions,
)
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.train.train_config import build_parser


def test_epit_dataset_v1_profile_loads_expected_static_asset():
    profile = load_epit_composition_profile()

    assert profile.name == EPIT_COMPOSITION_PROFILE
    assert profile.template_values.shape == (403, 24)
    assert profile.observed_template_values.shape == (403, 17)
    assert profile.template_missing_mask.shape == (403, 24)
    assert tuple(np.bincount(profile.template_family_indices)) == EPIT_COMPOSITION_FAMILY_COUNTS
    assert np.isclose(sum(profile.family_probabilities), 1.0)
    assert np.allclose(profile.family_probabilities, EPIT_COMPOSITION_FAMILY_PROBS)
    assert np.all((np.nansum(profile.template_values, axis=1) >= 99.0))
    assert np.all((np.nansum(profile.template_values, axis=1) <= 101.0))


def test_epit_composition_sampling_is_deterministic_and_closed():
    first = sample_epit_compositions(128, random_state=123)
    second = sample_epit_compositions(128, random_state=123)

    assert np.array_equal(first.full_compositions, second.full_compositions)
    assert np.array_equal(first.observed_compositions, second.observed_compositions)
    assert np.array_equal(first.family_indices, second.family_indices)
    assert first.template_ids == second.template_ids
    assert first.full_compositions.shape == (128, 24)
    assert first.observed_compositions.shape == (128, 17)
    assert np.all(first.full_compositions >= 0.0)
    assert np.allclose(first.full_compositions.sum(axis=1), 100.0)


def test_epit_composition_family_probabilities_are_tunable():
    profile = load_epit_composition_profile()
    nicrmo_index = profile.families.index("nicrmo_alloy")
    probabilities = np.zeros(len(profile.families))
    probabilities[nicrmo_index] = 1.0

    batch = sample_epit_compositions(
        64,
        profile=profile,
        family_probabilities=probabilities,
        perturb_strength=0.0,
        random_state=321,
    )

    assert set(batch.family_names) == {"nicrmo_alloy"}
    assert np.all(batch.family_indices == nicrmo_index)
    fe_index = profile.elements.index("Fe")
    assert np.all(batch.full_compositions[:, fe_index] == 0.0)


def test_epit_material_latents_map_deterministically_to_dataset_family_defaults():
    latents = np.random.default_rng(123).normal(size=(10_000, 2))
    first = map_epit_latents_to_compositions(latents, perturb_strength=0.0, random_state=456)
    second = map_epit_latents_to_compositions(latents, perturb_strength=0.0, random_state=456)

    frequencies = np.bincount(first.family_indices, minlength=5) / len(latents)
    assert np.allclose(frequencies, EPIT_COMPOSITION_FAMILY_PROBS, atol=2.0 / len(latents))
    assert np.array_equal(first.full_compositions, second.full_compositions)
    assert np.array_equal(first.template_indices, second.template_indices)
    assert np.allclose(first.full_compositions.sum(axis=1), 100.0)


def _empirical_epit_fixed_hp() -> dict:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "pitting_composition_mode": "empirical",
            "pitting_fixed_epit_schema": True,
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 1.0,
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": (17, 3, 1, 0, 0, 0, 0, 0, 0),
            "informed_normal_block_allocation_min_counts": (17, 3, 1, 0, 0, 0, 0, 0, 0),
            "pitting_process_role": "test_method_category",
            "pitting_process_category_count": 3,
        }
    )
    return fixed_hp


def test_empirical_epit_mode_reduces_scm_to_six_features_then_expands_material_only():
    fixed_hp = _empirical_epit_fixed_hp()
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    context = prior._prepare_epit_latent_scm_context(21)

    assert context.scm_num_features == 6
    assert context.latent_blocks == {
        "material": slice(0, 2),
        "environment": slice(2, 5),
        "process_history": slice(5, 6),
    }
    assert context.output_blocks == {
        "material": slice(0, 17),
        "environment": slice(17, 20),
        "process_history": slice(20, 21),
    }

    torch.manual_seed(7)
    np.random.seed(7)
    latent_X = torch.randn(256, 6)
    non_material = latent_X[:, 2:].clone()
    expanded = prior._expand_epit_material_latents(latent_X, context)
    batch = prior.last_pitting_composition_batch

    assert expanded.shape == (256, 21)
    assert torch.equal(expanded[:, 17:], non_material)
    assert batch is not None
    assert np.allclose(expanded[:, :17].numpy(), batch.observed_compositions, atol=1e-5)
    assert np.allclose(batch.full_compositions.sum(axis=1), 100.0)


def test_empirical_epit_expansion_integrates_with_existing_profile_and_target():
    fixed_hp = _empirical_epit_fixed_hp()
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    context = prior._prepare_epit_latent_scm_context(21)
    X = torch.randn(128, context.scm_num_features)
    y = torch.randn(128)

    X_out, y_out = prior.apply_informed_structure(
        X,
        y,
        {"num_features": 21},
        epit_latent_context=context,
    )

    assert X_out.shape == (128, 21)
    assert y_out.shape == (128,)
    assert torch.isfinite(X_out).all()
    assert torch.isfinite(y_out).all()
    assert prior.last_pitting_composition_batch is not None
    assert prior.last_pitting_target_rule["material_cols"] == list(range(17))
    assert prior.last_pitting_target_rule["temperature_col"] == 17
    assert prior.last_pitting_target_rule["process_col"] == 20


def test_empirical_epit_mode_passes_reduced_width_to_mlp_scm(monkeypatch):
    seen_widths = []
    original_mlp_scm = prior_dataset.MLPSCM

    class RecordingMLPSCM(original_mlp_scm):
        def __init__(self, *args, **kwargs):
            seen_widths.append(int(kwargs["num_features"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(prior_dataset, "MLPSCM", RecordingMLPSCM)
    fixed_hp = _empirical_epit_fixed_hp()
    fixed_hp.update(
        {
            "informed_mix_probs": (1.0, 0.0),
            "cat_prob": 0.0,
            "permute_features": False,
            "permute_labels": False,
        }
    )
    prior = SCMPrior(
        batch_size=1,
        batch_size_per_gp=1,
        min_features=21,
        max_features=21,
        max_classes=0,
        max_seq_len=256,
        min_train_size=0.5,
        max_train_size=0.8,
        prior_type="informed_scm",
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X, y, _, _, _ = prior.get_batch()

    assert seen_widths and set(seen_widths) == {6}
    assert X.shape == (1, 256, 21)
    assert y.shape == (1, 256)
    assert prior.last_pitting_composition_batch is not None


def test_epit_composition_profile_defaults_are_exposed_in_prior_config():
    assert DEFAULT_FIXED_HP["pitting_composition_mode"] == "legacy"
    assert DEFAULT_FIXED_HP["pitting_composition_profile"] == EPIT_COMPOSITION_PROFILE
    assert DEFAULT_FIXED_HP["pitting_composition_family_probs"] == EPIT_COMPOSITION_FAMILY_PROBS
    assert DEFAULT_FIXED_HP["pitting_composition_perturb_strength"] == 0.05
    assert DEFAULT_FIXED_HP["pitting_material_latent_count"] == 2


def test_epit_composition_cli_accepts_profile_overrides():
    args = build_parser().parse_args(
        [
            "--pitting_composition_mode",
            "empirical",
            "--pitting_composition_profile",
            "epit_dataset_v1",
            "--pitting_composition_family_probs",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--pitting_composition_perturb_strength",
            "0.1",
            "--pitting_material_latent_count",
            "3",
        ]
    )

    assert args.pitting_composition_mode == "empirical"
    assert args.pitting_composition_profile == "epit_dataset_v1"
    assert args.pitting_composition_family_probs == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert args.pitting_composition_perturb_strength == 0.1
    assert args.pitting_material_latent_count == 3


@pytest.mark.parametrize(
    "probabilities",
    ([1.0], [-1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]),
)
def test_invalid_epit_family_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        normalize_epit_family_probabilities(probabilities)


def test_invalid_epit_perturb_strength_is_rejected():
    with pytest.raises(ValueError):
        sample_epit_compositions(4, perturb_strength=-0.1, random_state=1)
