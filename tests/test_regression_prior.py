import numpy as np
import torch

from tabicl.prior.dataset import PriorDataset, SCMPrior
from tabicl.prior.reg2cls import Reg2Cls
from tabicl.prior.prior_config import DEFAULT_FIXED_HP


def test_scm_prior_generates_continuous_targets_when_max_classes_zero():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["cat_prob"] = 0.0

    dataset = PriorDataset(
        batch_size=4,
        batch_size_per_gp=1,
        min_features=2,
        max_features=4,
        max_classes=0,
        min_seq_len=24,
        max_seq_len=32,
        min_train_size=0.4,
        max_train_size=0.6,
        prior_type="mlp_scm",
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X, y, d, _, train_sizes = dataset.prior.get_batch()

    assert X.shape[:2] == y.shape
    assert y.dtype.is_floating_point
    assert torch.isfinite(y).all()
    assert (d > 0).all()

    for yi, train_size in zip(y, train_sizes):
        train_size = int(train_size.item())
        assert torch.std(yi[:train_size], unbiased=False) > 0
        assert torch.std(yi[train_size:], unbiased=False) > 0


def test_molecular_descriptor_marginals_are_finite_and_nonconstant():
    values = torch.linspace(-3.0, 3.0, steps=128)

    for family in [
        "descriptor_standard",
        "descriptor_positive",
        "descriptor_count",
        "descriptor_bounded",
    ]:
        transformed = SCMPrior._corrosion_marginal_values(values, family)
        assert transformed.shape == values.shape
        assert torch.isfinite(transformed).all()
        assert torch.std(transformed, unbiased=False) > 0

    counts = SCMPrior._corrosion_marginal_values(values, "descriptor_count")
    assert torch.allclose(counts, torch.floor(counts))


def test_audit_v2_block_marginals_are_finite_and_nonconstant():
    values = torch.linspace(-3.0, 3.0, steps=128)
    families = [
        "concentration",
        "material_bounded",
        "material_property",
        "environment_bounded",
        "process_score",
        "intervention_dose",
    ]

    for family in families:
        transformed = SCMPrior._corrosion_marginal_values(values, family)
        assert transformed.shape == values.shape
        assert torch.isfinite(transformed).all()
        assert torch.std(transformed, unbiased=False) > 0

    for family in [
        "material_category",
        "environment_category",
        "process_category",
        "process_binary",
        "intervention_category",
        "intervention_binary",
    ]:
        transformed = SCMPrior._corrosion_marginal_values(values, family)
        assert torch.isfinite(transformed).all()
        assert torch.std(transformed, unbiased=False) > 0
        assert torch.allclose(transformed, torch.floor(transformed))


def test_informed_corrosion_mechanism_changes_normal_targets():
    torch.manual_seed(0)
    fixed_hp = dict(DEFAULT_FIXED_HP)
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(96, 8)
    y = torch.randn(96)
    blocks = {
        "material": slice(0, 3),
        "environment": slice(3, 6),
        "process_history": slice(6, 7),
        "exposure_duration": slice(7, 8),
    }

    X_new, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=0.35,
        intervention_strength=0.20,
    )

    assert X_new.shape == X.shape
    assert y_new.shape == y.shape
    assert torch.isfinite(X_new).all()
    assert torch.isfinite(y_new).all()
    assert torch.std(y_new - y, unbiased=False) > 0


def test_informed_corrosion_mechanism_changes_inhibitor_targets_from_descriptors():
    torch.manual_seed(1)
    fixed_hp = dict(DEFAULT_FIXED_HP)
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(96, 10)
    y = torch.randn(96)
    blocks = {
        "environment": slice(0, 2),
        "direct_intervention": slice(2, 3),
        "molecular_descriptor": slice(3, 10),
    }

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "inhibitor_agent",
        interaction_strength=0.35,
        intervention_strength=0.50,
    )

    assert y_new.shape == y.shape
    assert torch.isfinite(y_new).all()
    assert torch.std(y_new - y, unbiased=False) > 0




def test_inhibitor_efficiency_target_family_raises_target_with_descriptor_efficacy():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "inhibitor_efficiency"
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.zeros(64, 1)
    X[:, 0] = torch.linspace(-3.0, 3.0, steps=64)
    y = torch.zeros(64)
    blocks = {"molecular_descriptor": slice(0, 1)}

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "inhibitor_agent",
        interaction_strength=0.0,
        intervention_strength=1.0,
    )

    assert torch.isfinite(y_new).all()
    assert y_new[-1] > y_new[0]

def test_pitting_target_family_raises_target_with_material_passivity():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.zeros(64, 2)
    X[:, 0] = torch.linspace(-3.0, 3.0, steps=64)
    y = torch.zeros(64)
    blocks = {"material": slice(0, 1), "environment": slice(1, 2)}

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=1.0,
        intervention_strength=0.0,
    )

    assert torch.isfinite(y_new).all()
    assert y_new[-1] > y_new[0]


def test_pitting_target_family_lowers_target_with_environment_aggressiveness():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.zeros(64, 2)
    X[:, 1] = torch.linspace(-3.0, 3.0, steps=64)
    y = torch.zeros(64)
    blocks = {"material": slice(0, 1), "environment": slice(1, 2)}

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=1.0,
        intervention_strength=0.0,
    )

    assert torch.isfinite(y_new).all()
    assert y_new[0] > y_new[-1]


def test_informed_block_allocation_ranges_override_fixed_allocation():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_task_family_probs"] = (1.0, 0.0)
    fixed_hp["informed_normal_block_allocation"] = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    fixed_hp["informed_normal_block_allocation_ranges"] = (
        0.8,
        0.8,
        0.2,
        0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")

    blocks = prior._split_informed_blocks(10)

    assert blocks["material"].start == 0
    assert blocks["material"].stop == 8
    assert blocks["environment"].start == 8
    assert blocks["environment"].stop == 10
    assert "process_history" not in blocks


def test_pitting_target_family_honors_epit_coefficients():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    fixed_hp["epit_material_coef"] = 0.0
    fixed_hp["epit_environment_coef"] = 0.0
    fixed_hp["epit_interaction_coef"] = 0.0
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.zeros(64, 2)
    X[:, 0] = torch.linspace(-3.0, 3.0, steps=64)
    X[:, 1] = torch.linspace(3.0, -3.0, steps=64)
    y = torch.zeros(64)
    blocks = {"material": slice(0, 1), "environment": slice(1, 2)}

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=1.0,
        intervention_strength=0.0,
    )

    assert torch.allclose(y_new, y)


def test_pitting_target_family_uses_fixed_epit_coefficients_across_random_seeds():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    fixed_hp["epit_material_coef"] = 0.575
    fixed_hp["epit_environment_coef"] = 0.50
    fixed_hp["epit_interaction_coef"] = 0.775
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.zeros(64, 2)
    X[:, 0] = torch.linspace(-3.0, 3.0, steps=64)
    X[:, 1] = torch.linspace(3.0, -3.0, steps=64)
    y = torch.zeros(64)
    blocks = {"material": slice(0, 1), "environment": slice(1, 2)}

    np.random.seed(1)
    _, y_first = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=1.0,
        intervention_strength=0.0,
    )
    np.random.seed(999)
    _, y_second = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=1.0,
        intervention_strength=0.0,
    )

    assert torch.allclose(y_first, y_second)


def test_flexible_epit_target_uses_convex_standardized_mixture():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(64, 2)
    y = torch.linspace(-2.0, 3.0, steps=64).square()
    blocks = {"material": slice(0, 1), "environment": slice(1, 2)}

    def run(weight: float) -> torch.Tensor:
        torch.manual_seed(123)
        _, mixed = prior._apply_informed_corrosion_mechanism(
            X.clone(),
            y.clone(),
            blocks,
            "normal_corrosion",
            interaction_strength=0.35,
            intervention_strength=0.0,
            target_mix_weight=weight,
        )
        return mixed

    generic = prior._standardize_signal(y)
    informed = run(1.0)
    assert torch.allclose(run(0.0), generic)
    assert torch.allclose(run(0.5), 0.5 * generic + 0.5 * informed)


def test_fixed_epit_target_uses_convex_standardized_mixture():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    fixed_hp["pitting_fixed_epit_schema"] = True
    fixed_hp["pitting_process_category_count"] = 3
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(64, 5)
    X[:, 4] = torch.arange(64) % 3
    y = torch.linspace(-2.0, 3.0, steps=64).square()
    blocks = {
        "material": slice(0, 1),
        "environment": slice(1, 4),
        "process_history": slice(4, 5),
    }

    def run(weight: float) -> torch.Tensor:
        torch.manual_seed(456)
        np.random.seed(456)
        _, mixed = prior._apply_informed_corrosion_mechanism(
            X.clone(),
            y.clone(),
            blocks,
            "normal_corrosion",
            interaction_strength=0.35,
            intervention_strength=0.0,
            target_mix_weight=weight,
        )
        return mixed

    generic = prior._standardize_signal(y)
    informed = run(1.0)
    assert torch.allclose(run(0.0), generic)
    assert torch.allclose(run(0.5), 0.5 * generic + 0.5 * informed)


def test_epit_target_mixture_falls_back_to_generic_for_constant_drive():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    fixed_hp["epit_material_coef"] = 0.0
    fixed_hp["epit_environment_coef"] = 0.0
    fixed_hp["epit_interaction_coef"] = 0.0
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(64, 2)
    y = torch.linspace(-2.0, 3.0, steps=64).square()

    _, mixed = prior._apply_informed_corrosion_mechanism(
        X,
        y,
        {"material": slice(0, 1), "environment": slice(1, 2)},
        "normal_corrosion",
        interaction_strength=0.35,
        intervention_strength=0.0,
        target_mix_weight=1.0,
    )

    assert torch.allclose(mixed, prior._standardize_signal(y))


def _pitting_fixed_hp() -> dict:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "pitting_potential"
    fixed_hp["informed_physical_marginal_profile"] = "pitting_potential_v1"
    fixed_hp["informed_physical_marginal_prob"] = 1.0
    fixed_hp["informed_task_family_probs"] = (1.0, 0.0)
    fixed_hp["informed_mix_probs"] = (1.0, 0.0)
    fixed_hp["informed_normal_block_allocation"] = (0.65, 0.25, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return fixed_hp


def test_masked_dirichlet_material_composition_is_sparse_and_sums_to_100():
    torch.manual_seed(11)
    fixed_hp = _pitting_fixed_hp()
    fixed_hp["pitting_material_dirichlet_concentration"] = 0.50
    fixed_hp["pitting_material_dirichlet_active_prob"] = 0.15
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(48, 8)

    prior._apply_masked_dirichlet_composition_marginal(X, slice(0, 8))

    material = X[:, :8]
    assert torch.isfinite(material).all()
    assert (material >= 0).all()
    assert torch.allclose(material.sum(dim=-1), torch.full((48,), 100.0), atol=1e-4)
    assert (material == 0).float().mean() > 0.40


def test_pitting_potential_v1_profile_records_roles_and_drives_target():
    torch.manual_seed(2)
    fixed_hp = _pitting_fixed_hp()
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(96, 10)
    y = torch.zeros(96)
    blocks = {
        "material": slice(0, 6),
        "environment": slice(6, 9),
        "process_history": slice(9, 10),
    }

    X_profile, info = prior._apply_pitting_potential_profile(X.clone(), blocks, "normal_corrosion")
    assert info.applied
    assert info.material_passivity is not None
    assert info.environment_aggressiveness is not None
    assert info.role_columns
    assert torch.isfinite(X_profile).all()
    assert torch.std(X_profile - X, unbiased=False) > 0

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X_profile,
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=0.5,
        intervention_strength=0.0,
        profile_info=info,
    )

    assert torch.isfinite(y_new).all()
    assert torch.std(y_new - y, unbiased=False) > 0


def test_pitting_process_profile_can_force_category_role_and_count_without_physical_marginals():
    torch.manual_seed(7)
    fixed_hp = _pitting_fixed_hp()
    fixed_hp["informed_physical_marginal_prob"] = 0.0
    fixed_hp["pitting_process_role"] = "test_method_category"
    fixed_hp["pitting_process_category_count"] = 3
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(96, 1)
    blocks = {"process_history": slice(0, 1)}

    X_profile, info = prior._apply_pitting_potential_profile(X.clone(), blocks, "normal_corrosion")

    values = torch.unique(X_profile[:, 0])
    assert info.applied
    assert info.role_columns.get("test_method_category") == [0]
    assert info.categorical_columns == [0]
    assert values.numel() == 3
    assert set(values.tolist()) == {0.0, 1.0, 2.0}
    assert torch.allclose(X_profile[:, 0], torch.floor(X_profile[:, 0]))


def test_pitting_potential_v1_full_generation_is_finite_and_disables_generic_num2cat(monkeypatch):
    seen_cat_probs = []
    original_num2cat = Reg2Cls._num2cat

    def spy_num2cat(self, X):
        seen_cat_probs.append(float(self.hp.get("cat_prob", 0.0)))
        return original_num2cat(self, X)

    monkeypatch.setattr(Reg2Cls, "_num2cat", spy_num2cat)

    fixed_hp = _pitting_fixed_hp()
    fixed_hp["cat_prob"] = 1.0
    dataset = PriorDataset(
        batch_size=2,
        batch_size_per_gp=1,
        min_features=6,
        max_features=10,
        max_classes=0,
        min_seq_len=32,
        max_seq_len=40,
        min_train_size=0.4,
        max_train_size=0.6,
        prior_type="informed_scm",
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X, y, d, _, _ = dataset.prior.get_batch()

    assert seen_cat_probs
    assert all(prob == 0.0 for prob in seen_cat_probs)
    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert (d > 0).all()
    assert y.dtype.is_floating_point


def test_inhibitor_target_family_uses_fixed_coefficients_across_random_seeds():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "inhibitor_efficiency"
    fixed_hp["inhibitor_descriptor_coef"] = 0.65
    fixed_hp["inhibitor_context_coef"] = 0.14
    fixed_hp["inhibitor_environment_interaction_coef"] = 0.365
    fixed_hp["inhibitor_material_interaction_coef"] = 0.24
    fixed_hp["inhibitor_intervention_coef"] = 0.0
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.zeros(64, 3)
    X[:, 0] = torch.linspace(-3.0, 3.0, steps=64)
    X[:, 1] = torch.linspace(3.0, -3.0, steps=64)
    X[:, 2] = torch.linspace(-1.5, 1.5, steps=64)
    y = torch.zeros(64)
    blocks = {
        "material": slice(0, 1),
        "environment": slice(1, 2),
        "molecular_descriptor": slice(2, 3),
    }

    np.random.seed(1)
    _, y_first = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "inhibitor_agent",
        interaction_strength=1.0,
        intervention_strength=1.0,
    )
    np.random.seed(999)
    _, y_second = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "inhibitor_agent",
        interaction_strength=1.0,
        intervention_strength=1.0,
    )

    assert torch.allclose(y_first, y_second)


def test_datacor_inhibitor_allocation_ranges_preserve_descriptor_heavy_shape():
    fixed_hp = _inhibitor_efficiency_fixed_hp()
    fixed_hp["informed_inhibitor_block_allocation_ranges"] = (
        0.05,
        0.08,
        0.05,
        0.08,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.84,
        0.90,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    fixed_hp["informed_inhibitor_block_allocation_min_counts"] = (1, 1, 0, 0, 0, 0, 1, 0, 0)
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")

    for seed, num_features in enumerate((11, 16, 32), start=10):
        np.random.seed(seed)
        family, blocks = prior._split_informed_blocks_with_family(num_features)

        assert family == "inhibitor_agent"
        assert blocks["material"].stop - blocks["material"].start >= 1
        assert blocks["environment"].stop - blocks["environment"].start >= 1
        assert blocks["molecular_descriptor"].stop - blocks["molecular_descriptor"].start >= 1
        assert "direct_intervention" not in blocks
        assert set(blocks) == {"material", "environment", "molecular_descriptor"}
        assert blocks["molecular_descriptor"].stop == num_features


def _inhibitor_efficiency_fixed_hp() -> dict:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["informed_target_family"] = "inhibitor_efficiency"
    fixed_hp["informed_physical_marginal_profile"] = "inhibitor_efficiency_v1"
    fixed_hp["informed_physical_marginal_prob"] = 1.0
    fixed_hp["informed_task_family_probs"] = (0.0, 1.0)
    fixed_hp["informed_mix_probs"] = (1.0, 0.0)
    fixed_hp["informed_inhibitor_block_allocation"] = (0.08, 0.10, 0.0, 0.0, 0.0, 0.02, 0.80, 0.0, 0.0)
    return fixed_hp


def test_datacor_inhibitor_allocation_matches_retained_feature_shape():
    fixed_hp = _inhibitor_efficiency_fixed_hp()
    fixed_hp["informed_inhibitor_block_allocation"] = (0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.875, 0.0, 0.0)
    fixed_hp["informed_inhibitor_block_allocation_min_counts"] = (1, 1, 0, 0, 0, 0, 1, 0, 0)
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")

    expected_counts = {
        8: (1, 1, 6),
        10: (1, 1, 8),
        16: (1, 1, 14),
        32: (2, 2, 28),
    }
    for num_features, (material_count, environment_count, descriptor_count) in expected_counts.items():
        family, blocks = prior._split_informed_blocks_with_family(num_features)

        assert family == "inhibitor_agent"
        assert blocks["material"] == slice(0, material_count)
        assert blocks["environment"] == slice(material_count, material_count + environment_count)
        assert blocks["molecular_descriptor"] == slice(material_count + environment_count, num_features)
        assert blocks["molecular_descriptor"].stop - blocks["molecular_descriptor"].start == descriptor_count
        assert "direct_intervention" not in blocks
        assert set(blocks) == {"material", "environment", "molecular_descriptor"}


def test_inhibitor_efficiency_v1_profile_records_roles_and_drives_target():
    torch.manual_seed(3)
    fixed_hp = _inhibitor_efficiency_fixed_hp()
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(96, 12)
    y = torch.zeros(96)
    blocks = {
        "material": slice(0, 1),
        "environment": slice(1, 2),
        "molecular_descriptor": slice(2, 12),
    }

    X_profile, info = prior._apply_inhibitor_efficiency_profile(X.clone(), blocks, "inhibitor_agent")
    assert info.applied
    assert info.descriptor_efficacy is not None
    assert info.environment_modifier is not None
    assert info.material_modifier is not None
    assert info.role_columns.get("alloy_category") == [0]
    assert info.role_columns.get("ph_condition") == [1]
    assert set(torch.unique(X_profile[:, 0]).tolist()).issubset({0.0, 1.0})
    assert set(torch.unique(X_profile[:, 1]).tolist()).issubset({4.0, 10.0})
    assert torch.isfinite(X_profile).all()
    assert torch.std(X_profile - X, unbiased=False) > 0

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X_profile,
        y.clone(),
        blocks,
        "inhibitor_agent",
        interaction_strength=0.35,
        intervention_strength=0.60,
        profile_info=info,
    )

    assert torch.isfinite(y_new).all()
    assert torch.std(y_new - y, unbiased=False) > 0


def test_inhibitor_efficiency_v1_full_generation_is_finite_and_disables_generic_num2cat(monkeypatch):
    seen_cat_probs = []
    original_num2cat = Reg2Cls._num2cat

    def spy_num2cat(self, X):
        seen_cat_probs.append(float(self.hp.get("cat_prob", 0.0)))
        return original_num2cat(self, X)

    monkeypatch.setattr(Reg2Cls, "_num2cat", spy_num2cat)

    fixed_hp = _inhibitor_efficiency_fixed_hp()
    fixed_hp["cat_prob"] = 1.0
    dataset = PriorDataset(
        batch_size=2,
        batch_size_per_gp=1,
        min_features=8,
        max_features=14,
        max_classes=0,
        min_seq_len=32,
        max_seq_len=40,
        min_train_size=0.4,
        max_train_size=0.6,
        prior_type="informed_scm",
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X, y, d, _, _ = dataset.prior.get_batch()

    assert seen_cat_probs
    assert all(prob == 0.0 for prob in seen_cat_probs)
    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert (d > 0).all()
    assert y.dtype.is_floating_point

