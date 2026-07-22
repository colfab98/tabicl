from argparse import Namespace

import numpy as np

import scripts.optuna_pitting_empirical_composition_prior_search as empirical_search
import scripts.optuna_pitting_fixed_pren_prior_search as fixed_pren_search


class RecordingTrial:
    def __init__(self):
        self.suggested_names = []

    def suggest_categorical(self, name, choices):
        self.suggested_names.append(name)
        return choices[len(choices) // 2]

    def suggest_float(self, name, low, high):
        self.suggested_names.append(name)
        return (low + high) / 2.0


def _command_args(mode: str) -> Namespace:
    return Namespace(
        pitting_composition_mode=mode,
        nproc_per_node=1,
        device="cpu",
        np_seed=11,
        torch_seed=12,
        max_steps=20,
        scheduler_total_steps=None,
        physical_profile="pitting_potential_v1",
        prior_n_jobs=1,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
    )


def _values_after(command: list[str], option: str, count: int) -> list[str]:
    start = command.index(option) + 1
    return command[start : start + count]


def test_empirical_search_uses_dataset_defaults_and_only_tunes_perturbation():
    trial = RecordingTrial()
    params = fixed_pren_search.sample_params(trial, pitting_composition_mode="empirical")

    assert np.isclose(sum(fixed_pren_search.EPIT_COMPOSITION_FAMILY_PROBS), 1.0)
    assert fixed_pren_search.EPIT_COMPOSITION_FAMILY_COUNTS == (298, 56, 19, 17, 13)
    assert fixed_pren_search.EPIT_MATERIAL_LATENT_COUNT == 2
    assert params.pitting_material_dirichlet_prob is None
    assert params.pitting_material_dirichlet_concentration is None
    assert params.pitting_material_dirichlet_active_prob is None
    assert 0.0 <= params.pitting_composition_perturb_strength <= 0.15
    assert "pitting_composition_perturb_strength" in trial.suggested_names
    assert "pitting_material_dirichlet_prob" not in trial.suggested_names
    assert "informed_target_mix_weight" in trial.suggested_names
    assert params.informed_feature_block_strength == 0.0
    assert "informed_feature_block_strength" not in trial.suggested_names
    assert "epit_interaction_coef" in trial.suggested_names
    assert set(trial.suggested_names) == {
        "informed_prior_ratio",
        "mlp_prob",
        "informed_target_mix_weight",
        "informed_physical_marginal_prob",
        "pitting_composition_perturb_strength",
        "epit_material_coef",
        "epit_environment_coef",
        "epit_interaction_coef",
    }


def test_empirical_training_command_fixes_family_defaults_and_two_latents(tmp_path):
    params = fixed_pren_search.sample_params(RecordingTrial(), pitting_composition_mode="empirical")
    command = fixed_pren_search.training_command(
        _command_args("empirical"),
        params,
        tmp_path / "checkpoint",
    )

    assert _values_after(command, "--pitting_composition_mode", 1) == ["empirical"]
    assert _values_after(command, "--pitting_composition_profile", 1) == ["epit_dataset_v1"]
    family_probs = [
        float(value) for value in _values_after(command, "--pitting_composition_family_probs", 5)
    ]
    assert np.allclose(family_probs, fixed_pren_search.EPIT_COMPOSITION_FAMILY_PROBS)
    assert _values_after(command, "--pitting_material_latent_count", 1) == ["2"]
    assert _values_after(command, "--informed_feature_block_strength", 1) == ["0"]
    assert np.isclose(
        float(_values_after(command, "--pitting_composition_perturb_strength", 1)[0]),
        params.pitting_composition_perturb_strength,
    )
    assert "--pitting_material_dirichlet_prob" not in command
    assert "--pitting_material_dirichlet_concentration" not in command
    assert "--pitting_material_dirichlet_active_prob" not in command


def test_legacy_search_retains_previous_dirichlet_dimensions(tmp_path):
    trial = RecordingTrial()
    params = fixed_pren_search.sample_params(trial, pitting_composition_mode="legacy")
    command = fixed_pren_search.training_command(
        _command_args("legacy"),
        params,
        tmp_path / "checkpoint",
    )

    assert params.pitting_composition_perturb_strength is None
    assert params.pitting_material_dirichlet_prob is not None
    assert params.pitting_material_dirichlet_concentration is not None
    assert params.pitting_material_dirichlet_active_prob is not None
    assert "pitting_composition_perturb_strength" not in trial.suggested_names
    assert "pitting_material_dirichlet_prob" in trial.suggested_names
    assert "informed_feature_block_strength" in trial.suggested_names
    assert "--pitting_composition_mode" not in command
    assert "--pitting_composition_perturb_strength" not in command
    assert "--pitting_material_dirichlet_prob" in command
    assert "--pitting_material_dirichlet_concentration" in command
    assert "--pitting_material_dirichlet_active_prob" in command


def test_default_study_names_separate_empirical_and_legacy_trials():
    assert (
        fixed_pren_search.default_study_name("empirical")
        == "pitting_fixed_pren_empirical_composition_search"
    )
    assert fixed_pren_search.default_study_name("legacy") == "pitting_fixed_pren_prior_search"


def test_dedicated_empirical_launcher_has_new_run_defaults_and_rejects_legacy():
    args = empirical_search.parse_args([])

    assert args.study_name == "pitting_fixed_pren_empirical_composition_v1"
    assert args.pitting_composition_mode == "empirical"
    assert args.n_trials == 50
    assert args.max_steps == 1000
    assert args.scheduler_total_steps == 10000

    try:
        empirical_search.parse_args(["--pitting-composition-mode", "legacy"])
    except ValueError:
        pass
    else:
        raise AssertionError("Dedicated empirical launcher accepted legacy composition mode.")
