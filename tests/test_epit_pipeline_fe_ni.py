from __future__ import annotations

from pathlib import Path

from scripts.epit_pipeline import run_optuna as search


class RecordingTrial:
    def __init__(self, *, use_magpie: bool = False):
        self.use_magpie = use_magpie
        self.suggested_names: list[str] = []

    def suggest_categorical(self, name, choices):
        self.suggested_names.append(name)
        if name == "use_magpie":
            return self.use_magpie
        return choices[0]

    def suggest_float(self, name, low, high):
        self.suggested_names.append(name)
        return (low + high) / 2.0


def _value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_fe_ni_optuna_removes_inactive_target_and_dirichlet_dimensions() -> None:
    args = search.parse_args(
        ["--pitting-composition-mode", "fe_ni_softmax"]
    )
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    trial = RecordingTrial(use_magpie=True)

    params = search.sample_params(
        trial, composition_mode=args.pitting_composition_mode
    )
    fingerprint = search.build_pipeline_fingerprint(args, rules)

    assert "informed_target_mix_weight" not in trial.suggested_names
    assert "pitting_material_dirichlet_prob" not in trial.suggested_names
    assert "pitting_material_dirichlet_concentration" not in trial.suggested_names
    assert "pitting_material_dirichlet_active_prob" not in trial.suggested_names
    assert params.informed_target_mix_weight == 1.0
    assert params.pitting_material_dirichlet_prob == 0.0
    assert params.pitting_material_dirichlet_concentration is None
    assert params.pitting_material_dirichlet_active_prob is None
    assert "informed_target_mix_weight" not in fingerprint["search_space"]
    assert "pitting_material_dirichlet_prob" not in fingerprint["search_space"]
    assert fingerprint["fixed_prior"]["composition_mode"] == "fe_ni_softmax"
    assert fingerprint["fixed_prior"]["informed_target_policy"] == "epit_only"
    assert fingerprint["fixed_prior"]["informed_target_mix_weight"] == 1.0
    assert fingerprint["fixed_prior"]["pitting_material_dirichlet_prob"] == 0.0


def test_fe_ni_training_command_uses_fixed_mode_and_epit_only_target(
    tmp_path: Path,
) -> None:
    args = search.parse_args(
        [
            "--device",
            "cpu",
            "--nproc-per-node",
            "1",
            "--pitting-composition-mode",
            "fe_ni_softmax",
        ]
    )
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    params = search.sample_params(
        RecordingTrial(), composition_mode=args.pitting_composition_mode
    )

    command = search.training_command(
        args, params, tmp_path / "checkpoint", rules
    )

    assert _value_after(command, "--pitting_composition_mode") == "fe_ni_softmax"
    assert _value_after(command, "--informed_target_mix_weight") == "1"
    assert _value_after(command, "--pitting_material_dirichlet_prob") == "0"


def test_fe_ni_trial_reconstruction_needs_no_inactive_optuna_values() -> None:
    params = search.trial_params_from_mapping(
        {
            "use_magpie": True,
            "informed_prior_ratio": 0.5,
            "mlp_prob": 1.0,
            "informed_feature_block_strength": 0.4,
        },
        composition_mode="fe_ni_softmax",
    )

    assert params.informed_target_mix_weight == 1.0
    assert params.pitting_material_dirichlet_prob == 0.0
    assert params.pitting_material_dirichlet_concentration is None
    assert params.pitting_material_dirichlet_active_prob is None


def test_fe_ni_pipeline_has_separate_versioned_launchers() -> None:
    optuna_launcher = (
        search.REPO_ROOT
        / "scripts"
        / "epit_pipeline"
        / "run_optuna_fe_ni.sbatch"
    ).read_text(encoding="utf-8")
    final_launcher = (
        search.REPO_ROOT
        / "scripts"
        / "epit_pipeline"
        / "train_final_fe_ni.sbatch"
    ).read_text(encoding="utf-8")

    assert "epit_pipeline_optuna_fe_ni_v4" in optuna_launcher
    assert "--pitting-composition-mode fe_ni_softmax" in optuna_launcher
    assert 'SELECTED_TRIAL_NUMBER="${SELECTED_TRIAL_NUMBER:-25}"' in final_launcher
    assert '--selected-trial-number "$SELECTED_TRIAL_NUMBER"' in final_launcher
    assert "--allow-stale-running-trials" in final_launcher
    assert "--allow-missing-selected-trial-artifacts" in final_launcher
    assert "epit_pipeline_optuna_fe_ni_v4" in final_launcher
    assert 'MAX_STEPS="${MAX_STEPS:-10000}"' in final_launcher
    assert 'SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"' in final_launcher
