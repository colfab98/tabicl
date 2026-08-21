from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline import train_final
from tabicl.prior.dataset import SCMPrior
from tabicl.prior.epit_feature_generator import sample_epit_feature_rows
from tabicl.prior.epit_feature_profile import EPIT_FEATURE_PROFILE
from tabicl.prior.mlp_scm import MLPSCM
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.prior.tree_scm import TreeSCM
from tabicl.train.train_config import build_parser


class RecordingTrial:
    def __init__(self) -> None:
        self.suggested_names: list[str] = []

    def suggest_categorical(self, name, choices):
        self.suggested_names.append(name)
        return choices[0]

    def suggest_float(self, name, low, high):
        self.suggested_names.append(name)
        return (low + high) / 2.0


def _fixed_hp() -> dict:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "pitting_composition_mode": "empirical_features_scm_target",
            "pitting_feature_profile": EPIT_FEATURE_PROFILE,
            "pitting_fixed_epit_schema": True,
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 1.0,
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": (17, 3, 1, 0, 0, 0, 0, 0, 0),
            "informed_normal_block_allocation_min_counts": (
                17,
                3,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            "informed_feature_block_strength": 0.0,
            "pitting_process_role": "test_method_category",
            "pitting_process_category_count": 52,
            "pitting_composition_perturb_strength": 0.05,
            "pitting_magpie_features": False,
            "cat_prob": 0.0,
            "permute_features": False,
            "permute_labels": False,
        }
    )
    return fixed_hp


def _search_args():
    return search.parse_args(
        [
            "--device",
            "cpu",
            "--nproc-per-node",
            "1",
            "--pitting-composition-mode",
            "empirical_features_scm_target",
            "--prior-n-jobs",
            "1",
        ]
    )


def _value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_generated_physical_features_are_standardized_only_for_scm_target():
    physical = torch.as_tensor(
        sample_epit_feature_rows(
            512,
            perturb_strength=0.05,
            random_state=123,
        ).features.copy(),
        dtype=torch.float32,
    )
    original = physical.clone()

    standardized = SCMPrior._standardize_features_for_scm_target(physical)

    assert torch.equal(physical, original)
    assert torch.isfinite(standardized).all()
    raw_scale = torch.std(physical, dim=0, unbiased=False)
    variable = raw_scale > 1e-6
    constant = ~variable
    assert torch.allclose(
        standardized[:, variable].mean(dim=0),
        torch.zeros(int(variable.sum())),
        atol=2e-5,
    )
    assert torch.allclose(
        standardized[:, variable].std(dim=0, unbiased=False),
        torch.ones(int(variable.sum())),
        atol=2e-5,
    )
    if bool(constant.any()):
        assert torch.equal(
            standardized[:, constant],
            torch.zeros_like(standardized[:, constant]),
        )


@pytest.mark.parametrize(
    ("prior_cls", "prior_type"),
    ((MLPSCM, "mlp_scm"), (TreeSCM, "tree_scm")),
)
def test_target_only_scm_uses_standardized_copy_without_modifying_features(
    prior_cls,
    prior_type,
):
    physical = torch.as_tensor(
        sample_epit_feature_rows(
            96,
            perturb_strength=0.05,
            random_state=17,
        ).features.copy(),
        dtype=torch.float32,
    )
    original = physical.clone()
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=_fixed_hp(),
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )
    params = {
        "prior_type": prior_type,
        "seq_len": len(physical),
        "num_features": physical.shape[1],
        "num_outputs": 1,
        "num_layers": 2,
        "hidden_dim": 12,
        "noise_std": 0.0,
        "device": "cpu",
    }
    np.random.seed(29)
    torch.manual_seed(29)

    target = prior._generate_scm_target_from_physical_features(
        physical,
        params,
        prior_cls,
    )

    assert torch.equal(physical, original)
    assert target.shape == (len(physical),)
    assert torch.isfinite(target).all()
    assert float(torch.std(target, unbiased=False)) > 0.0
    assert torch.equal(
        prior.last_pitting_scm_target_input,
        SCMPrior._standardize_features_for_scm_target(physical),
    )
    assert prior.last_pitting_scm_target_metadata == {
        "target_policy": "scm_target_from_standardized_physical_features_v1",
        "scm_prior_type": prior_type,
        "input_standardization": "per_dataset_column_population_zscore",
        "constant_column_policy": "zero",
        "physical_features_modified": False,
    }


def _generated_dataset(prior_type: str):
    fixed_hp = _fixed_hp()
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )
    params = {
        **fixed_hp,
        "seq_len": 128,
        "train_size": 80,
        "max_features": 21,
        "num_features": 21,
        "num_classes": 0,
        "num_layers": 2,
        "hidden_dim": 12,
        "noise_std": 0.0,
        "prior_type": prior_type,
        "informed_mode": True,
        "device": "cpu",
    }
    np.random.seed(41)
    torch.manual_seed(41)
    result = prior.generate_dataset(params)
    return result, prior


def test_mlp_and_tree_target_heads_start_from_the_same_physical_features():
    (mlp_X, mlp_y, mlp_d), mlp_prior = _generated_dataset("mlp_scm")
    (tree_X, tree_y, tree_d), tree_prior = _generated_dataset("tree_scm")

    assert torch.equal(mlp_X, tree_X)
    assert mlp_d.item() == tree_d.item() == 21
    assert mlp_y.shape == tree_y.shape == (128,)
    assert torch.isfinite(mlp_y).all()
    assert torch.isfinite(tree_y).all()
    assert not torch.equal(mlp_y, tree_y)
    assert mlp_prior.last_pitting_target_rule is None
    assert tree_prior.last_pitting_target_rule is None
    assert mlp_prior.last_pitting_scm_target_metadata["scm_prior_type"] == "mlp_scm"
    assert tree_prior.last_pitting_scm_target_metadata["scm_prior_type"] == "tree_scm"


def _generated_generic_dataset(composition_mode: str):
    fixed_hp = _fixed_hp()
    fixed_hp["pitting_composition_mode"] = composition_mode
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )
    params = {
        **fixed_hp,
        "seq_len": 128,
        "train_size": 80,
        "max_features": 21,
        "num_features": 21,
        "num_classes": 0,
        "num_layers": 2,
        "hidden_dim": 12,
        "noise_std": 0.0,
        "prior_type": "mlp_scm",
        "informed_mode": False,
        "device": "cpu",
    }
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)
    return prior.generate_dataset(params)


def test_new_mode_does_not_change_generic_dataset_generation():
    old_mode = _generated_generic_dataset("empirical_features")
    new_mode = _generated_generic_dataset("empirical_features_scm_target")

    for old_value, new_value in zip(old_mode, new_mode, strict=True):
        assert torch.equal(old_value, new_value)


def test_v6_search_and_final_training_are_bound_to_scm_target_policy(tmp_path: Path):
    args = _search_args()
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    trial = RecordingTrial()
    params = search.sample_params(
        trial,
        composition_mode=args.pitting_composition_mode,
    )
    reconstructed = search.trial_params_from_mapping(
        {
            "informed_prior_ratio": params.informed_prior_ratio,
            "mlp_prob": params.mlp_prob,
            "pitting_composition_perturb_strength": (
                params.pitting_composition_perturb_strength
            ),
        },
        composition_mode=args.pitting_composition_mode,
    )
    fingerprint = search.build_pipeline_fingerprint(args, rules)
    fixed_prior = fingerprint["fixed_prior"]
    command = search.training_command(args, params, tmp_path / "checkpoint", rules)

    assert set(trial.suggested_names) == {
        "informed_prior_ratio",
        "mlp_prob",
        "pitting_composition_perturb_strength",
    }
    assert reconstructed == params
    assert fixed_prior["informed_target_policy"] == (
        "scm_target_from_standardized_physical_features_v1"
    )
    assert fixed_prior["scm_target_input_standardization"] == (
        "per_dataset_column_population_zscore"
    )
    assert fixed_prior["scm_target_constant_column_policy"] == "zero"
    assert fixed_prior["scm_target_modifies_physical_features"] is False
    assert fixed_prior["scm_target_topology"] == "direct_noncausal"
    train_final.verify_empirical_feature_profile_identity(fixed_prior)
    assert _value_after(command, "--pitting_composition_mode") == (
        "empirical_features_scm_target"
    )
    assert _value_after(command, "--prior_n_jobs") == "1"
    assert "--pitting_target_rule_scores" not in command
    assert "--pitting_target_rule_coefficients" not in command

    optuna_launcher = (
        search.REPO_ROOT
        / "scripts"
        / "epit_pipeline"
        / "run_optuna_empirical_features_scm_target.sbatch"
    ).read_text(encoding="utf-8")
    final_launcher = (
        search.REPO_ROOT
        / "scripts"
        / "epit_pipeline"
        / "train_final_empirical_features_scm_target.sbatch"
    ).read_text(encoding="utf-8")
    for launcher in (optuna_launcher, final_launcher):
        assert "epit_pipeline_optuna_empirical_features_scm_target_v6" in launcher
        assert 'NP_SEED="${NP_SEED:-42}"' in launcher
        assert 'TORCH_SEED="${TORCH_SEED:-42}"' in launcher
        assert 'PRIOR_N_JOBS="${PRIOR_N_JOBS:-1}"' in launcher
    assert "--pitting-composition-mode empirical_features_scm_target" in optuna_launcher
    assert "--selected-trial-number" not in final_launcher
    for override in (
        "OVERRIDE_INFORMED_PRIOR_RATIO",
        "OVERRIDE_MLP_PROB",
        "OVERRIDE_PITTING_COMPOSITION_PERTURB_STRENGTH",
        "OVERRIDE_USE_MAGPIE",
        "OVERRIDE_PITTING_COMPOSITION_FAMILY_PROBS",
    ):
        assert override in final_launcher


def test_v6_final_overrides_change_effective_config_not_selected_trial(
    tmp_path: Path,
):
    args = train_final.parse_args(
        [
            "--storage",
            "journal:///unused.log",
            "--override-informed-prior-ratio",
            "0.25",
            "--override-mlp-prob",
            "0.75",
            "--override-pitting-composition-perturb-strength",
            "0.2",
            "--override-use-magpie",
            "true",
            "--override-pitting-composition-family-probs",
            "1",
            "2",
            "3",
            "4",
            "5",
        ]
    )
    args.pitting_composition_mode = "empirical_features_scm_target"
    selected = search.TrialParams(
        use_magpie=False,
        informed_prior_ratio=0.5,
        mlp_prob=0.5,
        informed_feature_block_strength=0.0,
        informed_target_mix_weight=1.0,
        pitting_material_dirichlet_prob=0.0,
        pitting_material_dirichlet_concentration=None,
        pitting_material_dirichlet_active_prob=None,
        pitting_composition_perturb_strength=0.05,
    )

    effective, provenance = train_final.apply_parameter_overrides(
        args,
        selected,
        composition_mode=args.pitting_composition_mode,
    )

    assert selected.use_magpie is False
    assert selected.informed_prior_ratio == 0.5
    assert effective.use_magpie is True
    assert effective.informed_prior_ratio == 0.25
    assert effective.mlp_prob == 0.75
    assert effective.pitting_composition_perturb_strength == 0.2
    assert provenance["trial_parameter_overrides"]["use_magpie"] is True
    assert provenance["fixed_prior_overrides"][
        "pitting_composition_family_probs"
    ] == [1.0, 2.0, 3.0, 4.0, 5.0]

    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    command = search.training_command(
        args,
        effective,
        tmp_path / "checkpoint",
        rules,
    )
    train_final.replace_option_values(
        command,
        "--pitting_composition_family_probs",
        provenance["fixed_prior_overrides"][
            "pitting_composition_family_probs"
        ],
        expected_count=5,
    )

    assert _value_after(command, "--pitting_magpie_features") == "True"
    assert _value_after(command, "--min_features") == "31"
    assert _value_after(command, "--max_features") == "31"
    style_index = command.index("--pitting_material_style_probs")
    assert command[style_index + 1 : style_index + 6] == [
        "1",
        "0",
        "0",
        "0",
        "0",
    ]
    family_index = command.index("--pitting_composition_family_probs")
    assert command[family_index + 1 : family_index + 6] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert "--pitting_target_rule_scores" not in command
    assert "--pitting_target_rule_coefficients" not in command


def test_v6_fixed_magpie_override_generates_31_features():
    fixed_hp = _fixed_hp()
    fixed_hp.update(
        {
            "pitting_magpie_features": True,
            "pitting_material_style_probs": (1.0, 0.0, 0.0, 0.0, 0.0),
            "pitting_composition_family_probs": (1, 2, 3, 4, 5),
            "pitting_composition_perturb_strength": 0.2,
        }
    )
    params = {
        **fixed_hp,
        "seq_len": 64,
        "train_size": 40,
        "max_features": 31,
        "num_features": 31,
        "num_classes": 0,
        "num_layers": 2,
        "hidden_dim": 12,
        "noise_std": 0.0,
        "prior_type": "mlp_scm",
        "informed_mode": True,
        "device": "cpu",
    }
    np.random.seed(23)
    torch.manual_seed(23)
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X, y, d = prior.generate_dataset(params)

    assert X.shape == (64, 31)
    assert y.shape == (64,)
    assert d.item() == 31
    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert prior.last_pitting_scm_target_input.shape == (64, 21)


def test_final_override_runs_use_isolated_default_output_directory():
    args = train_final.parse_args(
        [
            "--study-name",
            "epit_pipeline_optuna_empirical_features_scm_target_v6",
            "--storage",
            "journal:///unused.log",
        ]
    )

    canonical = train_final.output_dir_for(
        args,
        run_name="final_v6_trial_0001",
        isolate_run=False,
    )
    overridden = train_final.output_dir_for(
        args,
        run_name="final_v6_trial_0001_override_abcdef",
        isolate_run=True,
    )

    assert overridden.parent == canonical
    assert overridden.name == "final_v6_trial_0001_override_abcdef"


def test_new_mode_is_available_from_training_cli():
    args = build_parser().parse_args(
        [
            "--pitting_composition_mode",
            "empirical_features_scm_target",
            "--pitting_feature_profile",
            EPIT_FEATURE_PROFILE,
        ]
    )

    assert args.pitting_composition_mode == "empirical_features_scm_target"
    assert args.pitting_feature_profile == EPIT_FEATURE_PROFILE
