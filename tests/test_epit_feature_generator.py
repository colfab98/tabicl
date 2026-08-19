import numpy as np
import torch

from tabicl.prior.dataset import SCMPrior
from tabicl.prior.epit_feature_generator import sample_epit_feature_rows
from tabicl.prior.epit_feature_profile import EPIT_FEATURE_PROFILE, load_epit_feature_profile
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.train.train_config import build_parser


def test_epit_feature_row_sampling_is_deterministic_complete_and_paired():
    profile = load_epit_feature_profile()
    first = sample_epit_feature_rows(512, perturb_strength=0.05, random_state=123)
    second = sample_epit_feature_rows(512, perturb_strength=0.05, random_state=123)

    assert first.profile_name == EPIT_FEATURE_PROFILE
    assert first.features.shape == (512, 21)
    assert np.isfinite(first.features).all()
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.environment_indices, second.environment_indices)
    assert np.array_equal(first.composition_family_indices, second.composition_family_indices)
    assert set(first.compositions.family_names) <= {"fe_alloy", "nicrmo_alloy"}
    assert np.allclose(first.compositions.full_compositions.sum(axis=1), 100.0)
    assert np.array_equal(
        first.environment_values,
        profile.environment_imputed_values[first.environment_indices],
    )
    assert np.array_equal(
        first.test_method_codes,
        profile.test_method_codes[first.environment_indices],
    )
    assert np.array_equal(first.features[:, 17:20], first.environment_values)
    assert np.array_equal(first.features[:, 20], first.test_method_codes)


def test_epit_feature_row_sampling_uses_only_requested_family():
    fe = sample_epit_feature_rows(
        64,
        composition_family_probabilities=(1.0, 0.0),
        perturb_strength=0.0,
        random_state=5,
    )
    ni = sample_epit_feature_rows(
        64,
        composition_family_probabilities=(0.0, 1.0),
        perturb_strength=0.0,
        random_state=5,
    )

    assert set(fe.compositions.family_names) == {"fe_alloy"}
    assert set(ni.compositions.family_names) == {"nicrmo_alloy"}
    assert np.all(fe.composition_family_indices == 0)
    assert np.all(ni.composition_family_indices == 1)


def _empirical_feature_fixed_hp() -> dict:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "pitting_composition_mode": "empirical_features",
            "pitting_feature_profile": EPIT_FEATURE_PROFILE,
            "pitting_fixed_epit_schema": True,
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 1.0,
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": (17, 3, 1, 0, 0, 0, 0, 0, 0),
            "informed_normal_block_allocation_min_counts": (17, 3, 1, 0, 0, 0, 0, 0, 0),
            "pitting_process_role": "test_method_category",
            "pitting_process_category_count": 52,
            "pitting_composition_perturb_strength": 0.05,
        }
    )
    return fixed_hp


def test_empirical_feature_mode_replaces_all_21_columns_before_epit_only_target_logic():
    fixed_hp = _empirical_feature_fixed_hp()
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(256, 21)
    y = torch.randn(256)

    X_out, y_out = prior.apply_informed_structure(X, y, {"num_features": 21})
    batch = prior.last_pitting_feature_batch

    assert batch is not None
    assert X_out.shape == (256, 21)
    assert torch.allclose(X_out, torch.as_tensor(batch.features.copy(), dtype=X_out.dtype))
    assert torch.isfinite(X_out).all()
    assert torch.isfinite(y_out).all()
    assert prior.last_pitting_composition_batch is batch.compositions
    assert torch.equal(
        prior.last_pitting_material_family_ids,
        torch.as_tensor(batch.composition_family_indices.copy()),
    )
    assert prior.last_pitting_target_rule["synthetic_environment_mode"] == "raw"
    assert prior.last_pitting_target_rule["process_category_count"] == 52

    assert prior.last_pitting_target_rule["target_mix_weight"] == 1.0
    assert prior.last_pitting_target_rule["configured_target_mix_weight"] == fixed_hp[
        "informed_target_mix_weight"
    ]
    assert (
        prior.last_pitting_target_rule["target_mix_policy"]
        == "epit_only_for_empirical_features"
    )
    assert torch.allclose(y_out.cpu(), prior.last_pitting_target_component)


def test_empirical_feature_target_is_independent_of_generic_scm_target():
    fixed_hp = _empirical_feature_fixed_hp()
    fixed_hp["informed_target_mix_weight"] = 0.0
    X = torch.randn(256, 21)
    generic_y_a = torch.linspace(-2.0, 2.0, 256)
    generic_y_b = torch.linspace(30.0, -10.0, 256)

    outputs = []
    for generic_y in (generic_y_a, generic_y_b):
        np.random.seed(123)
        torch.manual_seed(123)
        prior = SCMPrior(
            batch_size=1,
            fixed_hp=fixed_hp,
            sampled_hp={},
            n_jobs=1,
            device="cpu",
        )
        X_out, y_out = prior.apply_informed_structure(
            X.clone(), generic_y.clone(), {"num_features": 21}
        )
        expected = SCMPrior.evaluate_fixed_epit_target_rule_numpy(
            X_out.numpy(),
            prior.last_pitting_target_rule,
            environment_mode="raw",
        )
        assert np.allclose(y_out.numpy(), expected, atol=2e-5)
        assert prior.last_pitting_target_rule["target_mix_weight"] == 1.0
        assert prior.last_pitting_target_rule["configured_target_mix_weight"] == 0.0
        outputs.append((X_out, y_out))

    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert float(torch.std(outputs[0][1], unbiased=False)) > 0.0


def test_empirical_fe_ni_threshold_numpy_uses_sampled_family_ids():
    fixed_hp = _empirical_feature_fixed_hp()
    fixed_hp["pitting_target_rule_scores"] = {"fe_ni_cr_threshold": 1.0}
    np.random.seed(42)
    torch.manual_seed(42)
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X_out, y_out = prior.apply_informed_structure(
        torch.randn(4096, 21),
        torch.randn(4096),
        {"num_features": 21},
    )
    rule = prior.last_pitting_target_rule
    expected = SCMPrior.evaluate_fixed_epit_target_rule_numpy(
        X_out.numpy(),
        rule,
        environment_mode="raw",
    )

    assert rule["target_rule_family"] == "fe_ni_cr_threshold"
    assert "synthetic_material_family_ids" in rule
    assert np.allclose(y_out.numpy(), expected, atol=2e-5)


def test_empirical_feature_mode_preserves_fixed_schema_through_batch_generation():
    fixed_hp = _empirical_feature_fixed_hp()
    fixed_hp.update(
        {
            "informed_mix_probs": (1.0, 0.0),
            "cat_prob": 0.0,
            "permute_features": False,
            "permute_labels": False,
        }
    )
    np.random.seed(9)
    torch.manual_seed(9)
    prior = SCMPrior(
        batch_size=1,
        batch_size_per_gp=1,
        min_features=21,
        max_features=21,
        max_classes=0,
        max_seq_len=128,
        min_train_size=0.5,
        max_train_size=0.8,
        prior_type="informed_scm",
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X, y, d, _, _ = prior.get_batch()
    batch = prior.last_pitting_feature_batch

    assert batch is not None
    assert X.shape == (1, 128, 21)
    assert y.shape == (1, 128)
    assert d.tolist() == [21]
    assert torch.unique(X[0, :, 20]).numel() > 1


def test_empirical_feature_mode_is_available_from_training_cli():
    args = build_parser().parse_args(
        [
            "--pitting_composition_mode",
            "empirical_features",
            "--pitting_feature_profile",
            EPIT_FEATURE_PROFILE,
        ]
    )

    assert args.pitting_composition_mode == "empirical_features"
    assert args.pitting_feature_profile == EPIT_FEATURE_PROFILE
