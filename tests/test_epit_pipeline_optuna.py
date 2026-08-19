from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import eval_corrosion_datasets as corrosion_eval
from scripts.epit_pipeline import evaluate_baseline_folds
from scripts.epit_pipeline import evaluate_final
from scripts.epit_pipeline import evaluate_optuna_folds
from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline import train_final
from scripts.epit_pipeline.artifact_hashes import (
    FINAL_MODEL_LOCK_NAME,
    build_final_model_lock,
    load_frozen_final_model,
)
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


class MutableStudy:
    def __init__(self, *, trials=()):
        self.user_attrs = {}
        self.trials = list(trials)

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


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

    assert args.study_name == "epit_pipeline_optuna_v3"
    assert args.n_trials == 50
    assert args.n_startup_trials == 10
    assert args.max_steps == 1000
    assert args.scheduler_total_steps == 10000
    assert args.nproc_per_node == 2
    assert args.eval_n_estimators == 8
    assert args.split_manifest == evaluate_optuna_folds.DEFAULT_SPLIT_MANIFEST
    assert "splits_v2" in args.split_manifest.parts
    assert "target_rules_v2" in args.target_rule_summary.parts


def test_optuna_study_fingerprint_rejects_mixed_pipeline() -> None:
    args = search.parse_args([])
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    fingerprint = search.build_pipeline_fingerprint(args, rules)
    other_worker_args = search.parse_args(["--random-seed", "999"])
    assert search.build_pipeline_fingerprint(other_worker_args, rules) == fingerprint
    study = MutableStudy()

    fingerprint_sha256 = search.bind_study_pipeline_fingerprint(
        study,
        fingerprint,
    )

    assert study.user_attrs[search.STUDY_FINGERPRINT_ATTR] == fingerprint
    assert (
        study.user_attrs[search.STUDY_FINGERPRINT_SHA256_ATTR]
        == fingerprint_sha256
    )
    changed = json.loads(json.dumps(fingerprint))
    changed["proxy_training"]["max_steps"] += 1
    with pytest.raises(RuntimeError, match="different split.*configuration"):
        search.bind_study_pipeline_fingerprint(study, changed)

    legacy_study = MutableStudy(trials=[object()])
    with pytest.raises(RuntimeError, match="no EPIT pipeline fingerprint"):
        search.bind_study_pipeline_fingerprint(legacy_study, fingerprint)


def test_stage4_pins_trial36_with_full_stage1_settings(tmp_path: Path) -> None:
    args = train_final.parse_args(
        [
            "--storage",
            "journal:///unused.log",
            "--selected-trial-number",
            "36",
            "--allow-stale-running-trials",
            "--allow-missing-selected-trial-artifacts",
        ]
    )
    train_final.validate_args(args)
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    params = search.TrialParams(
        use_magpie=True,
        informed_prior_ratio=0.5,
        mlp_prob=1.0,
        informed_feature_block_strength=0.70860372783741,
        informed_target_mix_weight=0.5877153183423934,
        pitting_material_dirichlet_prob=0.0,
        pitting_material_dirichlet_concentration=None,
        pitting_material_dirichlet_active_prob=None,
    )
    command = search.training_command(
        args,
        params,
        tmp_path / "checkpoints",
        rules,
    )
    train_final.replace_option(
        command,
        "--save_temp_every",
        args.save_temp_every,
    )
    train_final.replace_option(
        command,
        "--save_perm_every",
        args.save_perm_every,
    )

    assert args.max_steps == 10000
    assert args.scheduler_total_steps == 10000
    assert "--nproc_per_node=1" in command
    assert _value_after(command, "--max_steps") == "10000"
    assert _value_after(command, "--scheduler_total_steps") == "10000"
    assert _value_after(command, "--batch_size") == "512"
    assert _value_after(command, "--micro_batch_size") == "4"
    assert _value_after(command, "--lr") == "1e-4"
    assert _value_after(command, "--scheduler") == "cosine_warmup"
    assert _value_after(command, "--warmup_proportion") == "0.02"
    assert _value_after(command, "--gradient_clipping") == "1.0"
    assert _value_after(command, "--save_temp_every") == "100"
    assert _value_after(command, "--save_perm_every") == "500"
    assert _value_after(command, "--informed_prior_ratio") == "0.5"
    assert _value_after(command, "--informed_target_mix_weight") == "0.5877153183"
    assert _value_after(command, "--pitting_magpie_features") == "True"
    assert _value_after(command, "--min_features") == "31"
    assert _value_after(command, "--max_features") == "31"
    assert "--pitting_target_rule_scores" in command
    assert "--pitting_target_rule_coefficients" in command
    assert "--epit_material_coef" not in command
    assert "--epit_environment_coef" not in command
    assert "--epit_interaction_coef" not in command

    launcher = (
        search.REPO_ROOT / "scripts" / "epit_pipeline" / "train_final.sbatch"
    ).read_text(encoding="utf-8")
    assert 'SELECTED_TRIAL_NUMBER="${SELECTED_TRIAL_NUMBER:-36}"' in launcher
    assert 'MAX_STEPS="${MAX_STEPS:-10000}"' in launcher
    assert 'SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"' in launcher
    assert "--selected-trial-number" in launcher
    assert "--allow-stale-running-trials" in launcher
    assert "--allow-missing-selected-trial-artifacts" in launcher


def test_stage4_explicit_selection_requires_best_completed_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import optuna
    from optuna.trial import TrialState

    lower = SimpleNamespace(number=12, state=TrialState.COMPLETE, value=0.60)
    best = SimpleNamespace(number=36, state=TrialState.COMPLETE, value=0.69)
    stale = SimpleNamespace(number=47, state=TrialState.RUNNING, value=None)
    study = SimpleNamespace(trials=[lower, best, stale])
    monkeypatch.setattr(optuna, "load_study", lambda **kwargs: study)
    monkeypatch.setattr(
        search.original.search_utils,
        "build_optuna_storage",
        lambda storage: storage,
    )

    args = train_final.parse_args(
        [
            "--storage",
            "journal:///unused.log",
            "--selected-trial-number",
            "36",
            "--allow-stale-running-trials",
        ]
    )
    _, selected = train_final.load_selected_trial(args)
    assert selected is best

    args.selected_trial_number = 12
    with pytest.raises(RuntimeError, match="not the best completed trial"):
        train_final.load_selected_trial(args)


def test_stage4_validates_missing_worker_files_from_journal(tmp_path: Path) -> None:
    search_args = search.parse_args([])
    rules = search.load_target_rule_config(
        summary_path=search_args.target_rule_summary,
        split_manifest_path=search_args.split_manifest,
    )
    fingerprint = search.build_pipeline_fingerprint(search_args, rules)
    fingerprint_sha256 = search.canonical_json_sha256(fingerprint)
    params = search.TrialParams(
        use_magpie=True,
        informed_prior_ratio=0.5,
        mlp_prob=1.0,
        informed_feature_block_strength=0.70860372783741,
        informed_target_mix_weight=0.5877153183423934,
        pitting_material_dirichlet_prob=0.0,
        pitting_material_dirichlet_concentration=None,
        pitting_material_dirichlet_active_prob=None,
    )
    attrs = {
        search.STUDY_FINGERPRINT_SHA256_ATTR: fingerprint_sha256,
        "pipeline_artifact_identity": search.pipeline_artifact_identity(rules),
        "pitting_magpie_features": True,
        "tabicl_norm_methods": ["none"],
        "sampler_seed": 42,
        "mean_test_mae": 210.36,
        "median_test_mae": 196.70,
        "mean_test_rmse": 305.86,
        "mean_test_r2": 0.363,
        "median_test_spearman": 0.716,
        "std_test_spearman": 0.183,
    }
    for path_attr, sha_attr in (
        ("summary_csv", "summary_csv_sha256"),
        ("summary_json", "summary_json_sha256"),
        ("rows_csv", "rows_csv_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
        ("trial_result_json", "trial_result_json_sha256"),
    ):
        attrs[path_attr] = str(tmp_path / path_attr)
        attrs[sha_attr] = "0" * 64
    trial = SimpleNamespace(number=36, value=0.6963, user_attrs=attrs)

    provenance = train_final.verify_selected_trial_journal_record(
        trial,
        rules=rules,
        params=params,
        study_fingerprint=fingerprint,
        study_fingerprint_sha256=fingerprint_sha256,
    )

    assert provenance["verification_mode"] == "optuna_journal_record"
    assert provenance["immutable_files_verified"] is False
    assert len(provenance["missing_local_artifacts"]) == 5
    assert provenance["validation_folds"] == [1, 2, 3, 4, 5]


def test_stage4_rejects_study_from_another_artifact_set() -> None:
    args = search.parse_args([])
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    changed_rules = replace(rules, source_sha256="0" * 64)
    changed_fingerprint = search.build_pipeline_fingerprint(args, changed_rules)
    study = MutableStudy()
    search.bind_study_pipeline_fingerprint(study, changed_fingerprint)

    with pytest.raises(RuntimeError, match="different split, dataset, or target"):
        train_final.verify_study_pipeline_identity(study, rules)


def test_stage4_verifies_selected_trial_files_and_objective(tmp_path: Path) -> None:
    args = search.parse_args([])
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    params = search.TrialParams(
        use_magpie=False,
        informed_prior_ratio=0.5,
        mlp_prob=0.25,
        informed_feature_block_strength=0.4,
        informed_target_mix_weight=0.6,
        pitting_material_dirichlet_prob=0.0,
        pitting_material_dirichlet_concentration=None,
        pitting_material_dirichlet_active_prob=None,
    )
    fingerprint = search.build_pipeline_fingerprint(args, rules)
    fingerprint_sha256 = search.canonical_json_sha256(fingerprint)
    objective = 0.42
    checkpoint = (tmp_path / "step-1000.ckpt").resolve()
    summary_csv = (tmp_path / "summary.csv").resolve()
    summary_json = (tmp_path / "summary.json").resolve()
    rows_csv = (tmp_path / "rows.csv").resolve()
    trial_result_path = (tmp_path / "trial_result.json").resolve()
    checkpoint.write_bytes(b"checkpoint")
    summary_csv.write_text("mean_test_spearman\n0.42\n", encoding="utf-8")
    rows_csv.write_text("fold,prediction\n1,0.0\n", encoding="utf-8")
    evaluation = {
        "development_rows_only": True,
        "validation_folds": list(search.FIXED_VALIDATION_FOLDS),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": rules.split_manifest_sha256,
        "split_lock_sha256": rules.split_lock_sha256,
        "source_sha256": rules.source_sha256,
        "source": {
            "local_ckpt_path": str(checkpoint),
            "local_ckpt_sha256": train_final.sha256_file(checkpoint),
        },
        "settings": {
            "tabicl_norm_methods": ["none"],
            "tabicl_feat_shuffle_method": "none",
            "pitting_magpie_features": False,
        },
        "summary_csv": str(summary_csv),
        "rows_csv": str(rows_csv),
        "summary": [{"mean_test_spearman": objective}],
    }
    summary_json.write_text(json.dumps(evaluation), encoding="utf-8")
    trial_result = {
        "status": "completed",
        "trial_number": 3,
        "sampler_seed": 123,
        "params": search.trial_params_payload(params),
        "pipeline_fingerprint_sha256": fingerprint_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": train_final.sha256_file(checkpoint),
        "summary_csv": str(summary_csv),
        "summary_csv_sha256": train_final.sha256_file(summary_csv),
        "summary_json": str(summary_json),
        "summary_json_sha256": train_final.sha256_file(summary_json),
        "rows_csv": str(rows_csv),
        "rows_csv_sha256": train_final.sha256_file(rows_csv),
        "mean_test_spearman": objective,
    }
    trial_result_path.write_text(json.dumps(trial_result), encoding="utf-8")
    trial = SimpleNamespace(
        number=3,
        value=objective,
        user_attrs={
            search.STUDY_FINGERPRINT_SHA256_ATTR: fingerprint_sha256,
            "pipeline_artifact_identity": search.pipeline_artifact_identity(rules),
            "pitting_magpie_features": False,
            "sampler_seed": 123,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": train_final.sha256_file(checkpoint),
            "summary_csv": str(summary_csv),
            "summary_csv_sha256": train_final.sha256_file(summary_csv),
            "summary_json": str(summary_json),
            "summary_json_sha256": train_final.sha256_file(summary_json),
            "rows_csv": str(rows_csv),
            "rows_csv_sha256": train_final.sha256_file(rows_csv),
            "trial_result_json": str(trial_result_path),
            "trial_result_json_sha256": train_final.sha256_file(
                trial_result_path
            ),
        },
    )

    assert train_final.verify_selected_trial_evaluation(
        trial,
        split_manifest=args.split_manifest.resolve(),
        rules=rules,
        params=params,
        study_fingerprint_sha256=fingerprint_sha256,
    ) == evaluation

    trial.value = 0.43
    with pytest.raises(RuntimeError, match="fold Spearman.*does not match"):
        train_final.verify_selected_trial_evaluation(
            trial,
            split_manifest=args.split_manifest.resolve(),
            rules=rules,
            params=params,
            study_fingerprint_sha256=fingerprint_sha256,
        )

    trial.value = objective
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        train_final.verify_selected_trial_evaluation(
            trial,
            split_manifest=args.split_manifest.resolve(),
            rules=rules,
            params=params,
            study_fingerprint_sha256=fingerprint_sha256,
        )


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
    assert _value_after(command, "--tabicl-norm-methods") == "none"


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
    assert _value_after(command, "--tabicl-norm-methods") == "none"


def test_baseline_fold_command_uses_only_development_references(
    tmp_path: Path,
) -> None:
    args = evaluate_baseline_folds.parse_args(
        ["--device", "cpu", "--output-dir", str(tmp_path / "baselines")]
    )
    command = evaluate_baseline_folds.fold_command(
        args,
        fold=3,
        fold_dir=tmp_path / "fold_3",
    )

    assert _value_after(command, "--local-ckpt-path") == str(
        evaluate_baseline_folds.DEFAULT_GENERIC_CHECKPOINT.resolve()
    )
    assert _value_after(command, "--local-model-label") == "generic_baseline"
    assert "--epit-final-test" not in command
    assert _value_after(command, "--epit-validation-fold") == "3"
    assert _value_after(command, "--epit-split-manifest") == str(
        search.DEFAULT_SPLIT_MANIFEST.resolve()
    )
    assert "--compare-pretrained-tabicl" in command
    assert "--compare-catboost" in command
    assert "--compare-catboost-magpie" not in command
    assert _value_after(command, "--tabicl-feat-shuffle-method") == "none"
    assert _value_after(command, "--tabicl-norm-methods") == "none"
    assert _value_after(command, "--max-samples-per-task") == "0"


def test_baseline_launcher_uses_frozen_development_folds() -> None:
    launcher = (
        search.REPO_ROOT
        / "scripts"
        / "epit_pipeline"
        / "evaluate_baseline_folds.sbatch"
    ).read_text(encoding="utf-8")

    assert "splits_v2/split_manifest.json" in launcher
    assert "baseline_folds_v2" in launcher
    assert "tabicl_s1_regression_baseline/step-1000.ckpt" in launcher
    assert "--auto-output-dir" in launcher
    assert "evaluate_baseline_folds" in launcher


def test_baseline_output_dir_gets_fresh_suffix_when_requested(tmp_path: Path) -> None:
    output_dir = tmp_path / "baselines"
    output_dir.mkdir()
    (output_dir / "old-result.csv").write_text("old\n", encoding="utf-8")

    selected = evaluate_baseline_folds.prepare_output_dir(
        output_dir,
        auto_output_dir=True,
    )

    assert selected == tmp_path / "baselines_run_2"
    assert selected.is_dir()
    assert (output_dir / "old-result.csv").read_text(encoding="utf-8") == "old\n"


def test_explicit_none_normalization_reaches_tabicl_estimator(tmp_path: Path) -> None:
    estimator = corrosion_eval.make_tabicl_regressor(
        model_path=str(tmp_path / "not_loaded_until_fit.ckpt"),
        checkpoint_version="unused.ckpt",
        device="cpu",
        n_estimators=1,
        random_state=42,
        allow_auto_download=False,
        feat_shuffle_method="none",
        norm_methods=["none"],
    )

    assert estimator.norm_methods == ["none"]


def test_legacy_evaluator_normalization_default_is_unchanged(tmp_path: Path) -> None:
    estimator = corrosion_eval.make_tabicl_regressor(
        model_path=str(tmp_path / "not_loaded_until_fit.ckpt"),
        checkpoint_version="unused.ckpt",
        device="cpu",
        n_estimators=1,
        random_state=42,
        allow_auto_download=False,
        feat_shuffle_method="latin",
    )

    assert estimator.norm_methods is None


def test_final_split_uses_608_context_and_152_untouched_rows() -> None:
    task = _fake_epit_task()
    corrosion_eval.apply_epit_pipeline_final_test(
        [task],
        manifest_path=search.DEFAULT_SPLIT_MANIFEST,
    )

    split = task.fixed_split
    assert split is not None
    assert len(task.X) == 760
    assert len(split.train_index) == 608
    assert len(split.test_index) == 152
    assert not set(split.train_index) & set(split.test_index)
    assert split.split_strategy == "epit_pipeline_final_test"


def test_trial_params_can_be_reconstructed_from_optuna_values() -> None:
    values = {
        "use_magpie": True,
        "informed_prior_ratio": 0.75,
        "mlp_prob": 0.5,
        "informed_feature_block_strength": 0.4,
        "informed_target_mix_weight": 0.6,
        "pitting_material_dirichlet_prob": 1.0,
        "pitting_material_dirichlet_concentration": 0.5,
        "pitting_material_dirichlet_active_prob": 0.7,
    }

    params = search.trial_params_from_mapping(values)

    assert params.use_magpie is True
    assert params.pitting_material_dirichlet_concentration == 0.5
    assert params.pitting_material_dirichlet_active_prob == 0.7


def _write_frozen_final_model(tmp_path: Path):
    rules = search.load_target_rule_config(
        summary_path=search.DEFAULT_TARGET_RULE_SUMMARY,
        split_manifest_path=search.DEFAULT_SPLIT_MANIFEST,
    )
    checkpoint = tmp_path / "step-10000.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    manifest_path = tmp_path / train_final.FINAL_MODEL_MANIFEST_NAME
    manifest = {
        "schema_version": "epit_final_model_manifest_v1",
        "final_test_rows_used": False,
        "split": {
            "manifest": str(search.DEFAULT_SPLIT_MANIFEST.resolve()),
            "manifest_sha256": rules.split_manifest_sha256,
            "lock": rules.split_lock_path,
            "lock_sha256": rules.split_lock_sha256,
            "source_sha256": rules.source_sha256,
            "development_rows": 608,
            "final_test_rows": 152,
        },
        "target_rules": {
            "summary": rules.summary_path,
            "summary_sha256": rules.summary_sha256,
            "artifact_sha256s": rules.artifact_sha256s,
        },
        "selected_checkpoint": {
            "path": str(checkpoint),
            "sha256": train_final.sha256_file(checkpoint),
            "step": 10000,
        },
        "evaluation_configuration": {
            "task_id": evaluate_final.PITTING_TASK_ID,
            "device": "cpu",
            "random_state": 42,
            "n_estimators": 8,
            "tabicl_feat_shuffle_method": "none",
            "tabicl_norm_methods": ["none"],
            "regression_output": "median",
            "regression_uncertainty": False,
            "pitting_magpie_features": True,
            "expected_n_features": 31,
            "max_samples_per_task": 0,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lock = build_final_model_lock(
        manifest_path=manifest_path,
        immutable_artifacts={"selected_checkpoint": checkpoint},
    )
    (tmp_path / FINAL_MODEL_LOCK_NAME).write_text(
        json.dumps(lock),
        encoding="utf-8",
    )
    return load_frozen_final_model(manifest_path), checkpoint


def test_frozen_final_command_cannot_change_split_or_power(tmp_path: Path) -> None:
    frozen, _ = _write_frozen_final_model(tmp_path)

    command = evaluate_final.final_evaluation_command(
        frozen,
        output_dir=tmp_path / "evaluation",
    )

    assert "--epit-final-test" in command
    assert "--epit-validation-fold" not in command
    assert _value_after(command, "--tabicl-norm-methods") == "none"
    assert _value_after(command, "--tabicl-feat-shuffle-method") == "none"
    assert "--no-compare-pretrained-tabicl" in command


def test_final_model_lock_detects_checkpoint_changes(tmp_path: Path) -> None:
    frozen, checkpoint = _write_frozen_final_model(tmp_path)
    assert frozen.checkpoint_path == checkpoint.resolve()

    checkpoint.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="artifact changed"):
        load_frozen_final_model(frozen.manifest_path)


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
