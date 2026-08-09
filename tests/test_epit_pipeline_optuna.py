from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import eval_corrosion_datasets as corrosion_eval
from scripts.epit_pipeline import evaluate_optuna_folds
from scripts.epit_pipeline import run_optuna as search
from tabicl.prior.dataset import SCMPrior
from tabicl.prior.prior_config import DEFAULT_FIXED_HP


class RecordingTrial:
    def __init__(self, *, use_magpie: bool = False, dirichlet_prob: float = 0.0):
        self.use_magpie = use_magpie
        self.dirichlet_prob = dirichlet_prob
        self.suggested_names: list[str] = []

    def suggest_categorical(self, name, choices):
        self.suggested_names.append(name)
        if name == "use_magpie":
            return self.use_magpie
        if name == "pitting_material_dirichlet_prob":
            return self.dirichlet_prob
        return choices[0]

    def suggest_float(self, name, low, high):
        self.suggested_names.append(name)
        return (low + high) / 2.0


def _value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _fake_epit_task() -> corrosion_eval.EvalTask:
    rows = 760
    return corrosion_eval.EvalTask(
        task_id=corrosion_eval.EPIT_PIPELINE_TASK_ID,
        dataset="electrochemical_metrics_alloys",
        table="Pitting Potential",
        target="Epit, mV (SCE) Avg.",
        threshold=np.nan,
        target_binning="continuous",
        target_bins=0,
        bin_edges=[],
        class_labels=[],
        class_counts={},
        X=pd.DataFrame({"feature": np.arange(rows)}),
        y=pd.Series(np.arange(rows, dtype=float)),
        y_ordinal=pd.Series(np.arange(rows, dtype=float)),
        task_family="normal_corrosion",
        feature_groups_used=["material"],
        feature_group_counts={"material": 1},
        dropped_feature_columns=[],
        quality_flags=[],
        split_groups=pd.Series([f"group_{index}" for index in range(rows)]),
        split_strategy="unused",
        include_in_summary=True,
        summary_exclusion_reason="",
    )


def test_optuna_defaults_preserve_latest_reference_workflow() -> None:
    args = search.parse_args([])

    assert args.n_trials == 50
    assert args.n_startup_trials == 10
    assert args.max_steps == 1000
    assert args.scheduler_total_steps == 10000
    assert args.nproc_per_node == 2
    assert args.eval_n_estimators == 8
    assert args.split_manifest == evaluate_optuna_folds.DEFAULT_SPLIT_MANIFEST
    assert "splits_v2" in args.split_manifest.parts
    assert "target_rules_v2" in args.target_rule_summary.parts


def test_slurm_launcher_uses_v2_pipeline_artifacts() -> None:
    launcher = (
        search.REPO_ROOT / "scripts" / "epit_pipeline" / "run_optuna.sbatch"
    ).read_text(encoding="utf-8")

    assert "epit_pipeline/splits_v2/split_manifest.json" in launcher
    assert "epit_pipeline/target_rules_v2/calibration_summary.json" in launcher


def test_calibrated_coefficients_replace_old_three_search_dimensions() -> None:
    trial = RecordingTrial(use_magpie=True, dirichlet_prob=1.0)
    params = search.sample_params(trial)

    assert "epit_material_coef" not in trial.suggested_names
    assert "epit_environment_coef" not in trial.suggested_names
    assert "epit_interaction_coef" not in trial.suggested_names
    assert "use_magpie" in trial.suggested_names
    assert "informed_prior_ratio" in trial.suggested_names
    assert "pitting_material_dirichlet_concentration" in trial.suggested_names
    assert params.use_magpie is True


def test_rule_artifacts_feed_scores_probabilities_and_coefficients() -> None:
    args = search.parse_args([])
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )

    assert len(rules.scores) == 7
    assert np.isclose(sum(rules.probabilities.values()), 1.0)
    assert set(rules.scores) == set(rules.coefficients)
    assert rules.probabilities["method_aware"] > rules.probabilities["pren_linear"]


def test_training_command_keeps_reference_settings_and_adds_rules(tmp_path: Path) -> None:
    args = search.parse_args(["--device", "cpu", "--nproc-per-node", "1"])
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    params = search.sample_params(RecordingTrial())
    command = search.training_command(args, params, tmp_path / "checkpoint", rules)

    assert _value_after(command, "--max_steps") == "1000"
    assert _value_after(command, "--scheduler_total_steps") == "10000"
    assert _value_after(command, "--informed_physical_marginal_prob") == "1.0"
    assert _value_after(command, "--pitting_composition_mode") == "legacy"
    assert "--pitting_target_rule_scores" in command
    assert "--pitting_target_rule_coefficients" in command
    assert "--epit_material_coef" not in command
    assert "--epit_environment_coef" not in command
    assert "--epit_interaction_coef" not in command


def test_eval_command_uses_saved_folds_not_random_splits(tmp_path: Path) -> None:
    args = search.parse_args(["--device", "cpu"])
    params = search.sample_params(RecordingTrial())
    command = search.eval_command(
        args,
        params,
        tmp_path / "step-1000.ckpt",
        "trial_0000",
        tmp_path / "evaluation",
    )

    assert "--split-manifest" in command
    assert command[command.index("--validation-folds") + 1 : command.index("--device")] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert "--split-seeds" not in command
    assert "--test-size" not in command


def test_saved_folds_remove_final_test_rows() -> None:
    manifest = search.DEFAULT_SPLIT_MANIFEST
    expected = {
        1: (486, 122),
        2: (486, 122),
        3: (486, 122),
        4: (487, 121),
        5: (487, 121),
    }
    for fold, (context_rows, validation_rows) in expected.items():
        task = _fake_epit_task()
        corrosion_eval.apply_epit_pipeline_fold(
            [task],
            manifest_path=manifest,
            validation_fold=fold,
        )
        assert len(task.X) == 608
        assert len(task.fixed_split.train_index) == context_rows
        assert len(task.fixed_split.test_index) == validation_rows
        assert not set(task.fixed_split.train_index) & set(task.fixed_split.test_index)


def test_legacy_v1_manifest_remains_evaluable() -> None:
    task = _fake_epit_task()
    legacy_manifest = search.PIPELINE_ROOT / "splits_v1" / "split_manifest.json"

    corrosion_eval.apply_epit_pipeline_fold(
        [task],
        manifest_path=legacy_manifest,
        validation_fold=1,
    )

    assert len(task.X) == 608
    assert len(task.fixed_split.train_index) == 486
    assert len(task.fixed_split.test_index) == 122


def test_fold_evaluator_command_passes_fixed_manifest(tmp_path: Path) -> None:
    args = evaluate_optuna_folds.parse_args(
        [
            "--local-ckpt-path",
            str(tmp_path / "step-1000.ckpt"),
            "--split-manifest",
            str(search.DEFAULT_SPLIT_MANIFEST),
        ]
    )
    command = evaluate_optuna_folds.fold_command(
        args,
        fold=3,
        fold_dir=tmp_path / "fold_3",
    )

    assert _value_after(command, "--epit-validation-fold") == "3"
    assert _value_after(command, "--epit-split-manifest") == str(
        search.DEFAULT_SPLIT_MANIFEST
    )
    assert "--no-compare-pretrained-tabicl" in command


def test_prior_uses_pipeline_coefficient_override() -> None:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "pitting_fixed_epit_schema": True,
            "pitting_process_category_count": 4,
            "pitting_target_rule_scores": {"pren_linear": 1.0},
            "pitting_target_rule_coefficients": (
                "pren_linear.material_passivity=0.5",
                "pren_linear.environment_aggressiveness=0.3",
                "pren_linear.material_chloride_interaction=0.2",
            ),
        }
    )
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )
    X = torch.zeros((20, 21), dtype=torch.float32)
    X[:, 1] = torch.linspace(10.0, 25.0, 20)
    X[:, 3] = torch.linspace(0.0, 5.0, 20)
    X[:, 17] = torch.linspace(20.0, 80.0, 20)
    X[:, 18] = torch.logspace(-4, 0, 20)
    X[:, 19] = torch.linspace(2.0, 10.0, 20)
    X[:, 20] = torch.arange(20) % 4
    rule = prior._sample_fixed_epit_target_rule(
        X,
        {
            "material": slice(0, 17),
            "environment": slice(17, 20),
            "process_history": slice(20, 21),
        },
        target_mix_weight=1.0,
        profile_info=None,
    )

    assert rule["target_rule_coefficients"] == {
        "material_passivity": 0.5,
        "environment_aggressiveness": 0.3,
        "material_chloride_interaction": 0.2,
    }
