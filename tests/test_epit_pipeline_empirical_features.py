from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline import train_final


class RecordingTrial:
    def __init__(self) -> None:
        self.suggested_names: list[str] = []

    def suggest_categorical(self, name, choices):
        self.suggested_names.append(name)
        return choices[0]

    def suggest_float(self, name, low, high):
        self.suggested_names.append(name)
        return (low + high) / 2.0


def _args():
    return search.parse_args(
        [
            "--device",
            "cpu",
            "--nproc-per-node",
            "1",
            "--pitting-composition-mode",
            "empirical_features",
            "--prior-n-jobs",
            "1",
        ]
    )


def _value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _values_after(command: list[str], option: str, count: int) -> list[str]:
    start = command.index(option) + 1
    return command[start : start + count]


def test_empirical_feature_optuna_search_has_only_three_active_dimensions():
    args = _args()
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    trial = RecordingTrial()

    params = search.sample_params(
        trial,
        composition_mode=args.pitting_composition_mode,
    )
    fingerprint = search.build_pipeline_fingerprint(args, rules)

    assert set(trial.suggested_names) == {
        "informed_prior_ratio",
        "mlp_prob",
        "pitting_composition_perturb_strength",
    }
    assert params.use_magpie is False
    assert params.informed_feature_block_strength == 0.0
    assert params.informed_target_mix_weight == 1.0
    assert params.pitting_material_dirichlet_prob == 0.0
    assert 0.0 <= params.pitting_composition_perturb_strength <= 0.15
    assert set(fingerprint["search_space"]) == set(trial.suggested_names)
    assert fingerprint["fixed_prior"]["feature_profile"] == search.EPIT_FEATURE_PROFILE
    assert fingerprint["fixed_prior"]["use_magpie"] is False
    assert fingerprint["fixed_prior"]["informed_target_mix_weight"] == 1.0
    assert fingerprint["fixed_prior"]["synthetic_material_families"] == [
        "fe_alloy",
        "nicrmo_alloy",
    ]
    assert np.allclose(
        fingerprint["fixed_prior"]["synthetic_material_family_probabilities"],
        [298.0 / 315.0, 17.0 / 315.0],
    )
    assert fingerprint["proxy_training"]["prior_n_jobs"] == 1


def test_final_training_rechecks_empirical_feature_profile_identity():
    args = _args()
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    fixed_prior = search.build_pipeline_fingerprint(args, rules)["fixed_prior"]

    train_final.verify_empirical_feature_profile_identity(fixed_prior)

    changed = dict(fixed_prior)
    changed["feature_profile_csv_sha256"] = "0" * 64
    with pytest.raises(
        RuntimeError,
        match="differs from the selected Optuna study",
    ):
        train_final.verify_empirical_feature_profile_identity(changed)


def test_empirical_feature_training_command_uses_profile_without_magpie(tmp_path: Path):
    args = _args()
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    params = search.sample_params(
        RecordingTrial(),
        composition_mode=args.pitting_composition_mode,
    )

    command = search.training_command(args, params, tmp_path / "checkpoint", rules)
    eval_command = search.eval_command(
        args,
        params,
        tmp_path / "step-1000.ckpt",
        "trial_0000",
        tmp_path / "evaluation",
    )

    assert _value_after(command, "--pitting_composition_mode") == "empirical_features"
    assert _value_after(command, "--pitting_feature_profile") == search.EPIT_FEATURE_PROFILE
    assert np.allclose(
        [
            float(value)
            for value in _values_after(
                command,
                "--pitting_composition_family_probs",
                5,
            )
        ],
        search.EPIT_COMPOSITION_FAMILY_PROBS,
    )
    assert np.isclose(
        float(_value_after(command, "--pitting_composition_perturb_strength")),
        params.pitting_composition_perturb_strength,
    )
    assert _value_after(command, "--pitting_magpie_features") == "False"
    assert _value_after(command, "--informed_target_mix_weight") == "1"
    assert _value_after(command, "--informed_feature_block_strength") == "0"
    assert _value_after(command, "--pitting_material_dirichlet_prob") == "0"
    assert _value_after(command, "--prior_n_jobs") == "1"
    assert _value_after(command, "--min_features") == "21"
    assert _value_after(command, "--max_features") == "21"
    assert "--pitting_material_style_probs" not in command
    assert "--pitting-magpie-features" not in eval_command


def test_empirical_feature_trial_reconstruction_needs_only_active_dimensions():
    params = search.trial_params_from_mapping(
        {
            "informed_prior_ratio": 0.5,
            "mlp_prob": 0.75,
            "pitting_composition_perturb_strength": 0.05,
        },
        composition_mode="empirical_features",
    )

    assert params.use_magpie is False
    assert params.informed_feature_block_strength == 0.0
    assert params.informed_target_mix_weight == 1.0
    assert params.pitting_material_dirichlet_prob == 0.0
    assert params.pitting_composition_perturb_strength == 0.05


def test_empirical_feature_search_requires_one_prior_worker():
    args = search.parse_args(
        ["--pitting-composition-mode", "empirical_features"]
    )

    with pytest.raises(ValueError, match="prior-n-jobs 1"):
        search.run_search(args)


def test_empirical_feature_pipeline_has_separate_versioned_launchers():
    optuna_launcher = (
        search.REPO_ROOT
        / "scripts"
        / "epit_pipeline"
        / "run_optuna_empirical_features.sbatch"
    ).read_text(encoding="utf-8")
    final_launcher = (
        search.REPO_ROOT
        / "scripts"
        / "epit_pipeline"
        / "train_final_empirical_features.sbatch"
    ).read_text(encoding="utf-8")

    for launcher in (optuna_launcher, final_launcher):
        assert "epit_pipeline_optuna_empirical_features_v5" in launcher
        assert 'NP_SEED="${NP_SEED:-42}"' in launcher
        assert 'TORCH_SEED="${TORCH_SEED:-42}"' in launcher
        assert 'PRIOR_N_JOBS="${PRIOR_N_JOBS:-1}"' in launcher
        assert '--prior-n-jobs "$PRIOR_N_JOBS"' in launcher
    assert "--pitting-composition-mode empirical_features" in optuna_launcher
    assert "--selected-trial-number" not in final_launcher
    assert "--allow-stale-running-trials" not in final_launcher
