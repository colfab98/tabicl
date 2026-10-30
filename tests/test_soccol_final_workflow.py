"""Exercise reviewed final selection and frozen queries without training a model."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as evaluation
from scripts.epit_pipeline import evaluate_final as final_eval
from scripts.epit_pipeline import evaluate_optuna_folds as folds
from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline import train_final as final_train
from scripts.epit_pipeline.artifact_hashes import load_frozen_final_model
from scripts.soccol_pipeline import evaluate_final as soccol_eval
from scripts.soccol_pipeline import run_optuna as soccol_search
from scripts.soccol_pipeline import train_final as soccol_train
from scripts.soccol_pipeline.data import load_soccol_dataset
from tabicl.train.train_config import build_parser


@pytest.fixture
def reviewed():
    modules = (search, final_train, final_eval, folds)
    originals = {module: vars(module).copy() for module in modules}
    try:
        soccol_eval.configure()
        # Match the Soccol fold entry point without its import-time global changes.
        folds.load_frozen_split = search.load_frozen_split
        folds.PITTING_TASK_ID = evaluation.SOCCOL_PIPELINE_TASK_ID
        yield
    finally:
        for module, original in originals.items():
            for name, value in original.items():
                if vars(module).get(name) is not value:
                    setattr(module, name, value)


@pytest.mark.parametrize("ratio", [.25, .50, .75, 1.])
def test_final_defaults_and_ratio_overrides_keep_reviewed_inputs(reviewed, tmp_path, ratio):
    args = soccol_train.parse_args([
        "--storage", "journal:///unused.log", "--override-informed-prior-ratio", str(ratio),
    ])
    assert args.study_name == "soccol_pipeline_optuna_empirical_features_scm_target_v5"
    assert args.selected_trial_number is None
    assert args.allow_missing_selected_trial_artifacts
    assert args.max_steps == 10000
    assert args.checkpoint_root.name == "soccol_pipeline_final_v2"
    assert args.target_rule_summary.parent.name == "target_rules_v2"
    args.pitting_composition_mode = "empirical_features_scm_target"
    selected = search.trial_params_from_mapping(
        {"mlp_prob": .37, "informed_target_mix_weight": .42, "pitting_composition_perturb_strength": .12},
        composition_mode=args.pitting_composition_mode, fixed_informed_prior_ratio=1.,
    )
    effective, provenance = final_train.apply_parameter_overrides(
        args, selected, composition_mode=args.pitting_composition_mode,
    )
    assert effective.informed_prior_ratio == ratio
    assert selected.informed_prior_ratio == 1.
    assert provenance["trial_parameter_overrides"] == {"informed_prior_ratio": ratio}
    rules = search.load_target_rule_config(summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest)
    command = search.training_command(args, effective, tmp_path, rules)
    entry = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
    config = build_parser().parse_args(command[entry + 1:])
    assert config.informed_prior_ratio == ratio
    assert config.min_features == config.max_features == 58
    assert config.pitting_feature_profile == "soccol_pitting_features_v3"
    assert config.col_missing_value_embedding
    assert config.pitting_composition_perturb_strength == .12
    assert config.pitting_coefficient_variation == .8
    assert config.pitting_coefficient_variation_seed == 42
    eval_args = soccol_eval.parse_args([])
    assert eval_args.final_model_manifest.parent.name == args.study_name
    assert eval_args.final_model_manifest.parent.parent.name == "final_v2"
    assert eval_args.context_mode == "full"


@pytest.mark.parametrize("missing_proxy_files", [False, True])
def test_reviewed_final_selection_and_queries_end_to_end(reviewed, monkeypatch, tmp_path, missing_proxy_files):
    """Use real study/manifest/summary code; replace only external model jobs.

    All scores and checkpoint bytes below are artificial test fixtures. No
    training or prediction is performed on development or final-test targets.
    """
    final_queries_seen = []
    training_commands = []

    def fake_predictions(command, **kwargs):
        args = evaluation.parse_args(command[2:])
        assert args.soccol_reviewed_data
        tasks = evaluation.make_tasks(args)
        if args.epit_final_test:
            evaluation.apply_epit_pipeline_final_test(
                tasks, manifest_path=args.epit_split_manifest, context_fold=args.epit_final_context_fold,
            )
            final_queries_seen.append(tasks[0].fixed_split.test_index.tolist())
        else:
            assert args.epit_validation_fold in range(1, 6)
            evaluation.apply_epit_pipeline_fold(
                tasks, manifest_path=args.epit_split_manifest, validation_fold=args.epit_validation_fold,
            )
        task = tasks[0]
        split = task.fixed_split
        label = args.local_model_label[0]
        score = {"trial_0000": .4, "trial_0001": .6,
                 "final_step_00500": .8, "final_step_10000": .7}.get(label, .5)
        row = {
            "model": label, "task_id": task.task_id, "n_train": len(split.train_index),
            "n_test": len(split.test_index), "n_features": task.X.shape[1],
            "split_strategy": split.split_strategy, "pitting_magpie_features": False,
            "test_spearman": score, "test_mae": 1., "test_rmse": 2.,
            "test_r2": .3, "test_pearson": .4,
        }
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(args.output_csv, index=False)
        if args.epit_final_test:
            payload = {
                "tabicl_feat_shuffle_method": "none", "n_estimators": args.n_estimators,
                "random_state": args.random_state, "max_samples_per_task": 0,
                "regression_uncertainty": False, "regression_output": "median",
                "local_checkpoint": str(args.local_ckpt_path[0].resolve()), "errors": [], "rows": [row],
                "soccol_final_split": evaluation._reviewed_soccol_final_provenance(tasks),
                "missing_aware_models": args.missing_aware_model,
            }
            args.output_json.write_text(json.dumps(payload))

    def fake_job(command, **kwargs):
        if "--checkpoint_dir" in command:
            training_commands.append(command)
            entry = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
            config = build_parser().parse_args(command[entry + 1:])
            assert config.min_features == config.max_features == 58
            assert config.col_missing_value_embedding
            directory = Path(config.checkpoint_dir)
            directory.mkdir(parents=True, exist_ok=True)
            steps = {config.max_steps}
            if config.max_steps == 10000:
                steps.add(500)
            for step in steps:
                (directory / f"step-{step}.ckpt").write_bytes(b"test fixture, not a trained model")
        else:
            assert command[1].endswith("soccol_pipeline/evaluate_optuna_folds.py")
            with monkeypatch.context() as local:
                local.setattr(sys, "argv", command[1:])
                folds.main()

    monkeypatch.setattr(search.original.search_utils, "run_command", fake_job)
    monkeypatch.setattr(folds.subprocess, "run", fake_predictions)
    storage = f"journal://{tmp_path / 'study.log'}"
    optuna_args = soccol_search.parse_args([
        "--storage", storage, "--n-trials", "2", "--device", "cpu", "--base-run-name", "test_only",
        "--work-dir", str(tmp_path / "trials"), "--checkpoint-root", str(tmp_path / "proxy_checkpoints"),
        "--eval-output-root", str(tmp_path / "folds"),
    ])
    search.run_search(optuna_args)
    assert not final_queries_seen
    train_args = soccol_train.parse_args([
        "--storage", storage, "--device", "cpu", "--output-dir", str(tmp_path / "final"),
        "--checkpoint-root", str(tmp_path / "final_checkpoints"),
    ])
    if missing_proxy_files:
        _, selected_trial = final_train.load_selected_trial(train_args)
        assert selected_trial.number == 1 and train_args.selected_trial_number is None
        for key in ("summary_csv", "summary_json", "rows_csv", "checkpoint_path", "trial_result_json"):
            path = Path(selected_trial.user_attrs[key])
            assert path.is_relative_to(tmp_path)
            path.unlink()
    manifest_path = final_train.run(train_args)
    assert not final_queries_seen
    frozen = load_frozen_final_model(manifest_path)
    manifest = frozen.manifest
    assert manifest["study"]["selected_trial_number"] == 1
    assert manifest["study"]["selection_mode"] == "automatic_current_best_completed_trial"
    assert manifest["study"]["selected_trial_params"]["informed_prior_ratio"] == 1.
    assert manifest["selected_checkpoint"]["step"] == 500
    assert manifest["training"]["max_steps"] == 10000
    assert manifest["split"]["original_development_rows"] == 3222
    assert manifest["split"]["development_rows"] == 2950
    assert manifest["split"]["final_test_rows"] == 805
    assert len(manifest["split"]["final_query_row_ids"]) == 805
    assert len(training_commands) == 3
    assert all("--epit-final-test" not in record["evaluation_command"]
               for record in manifest["checkpoint_selection"]["candidates"])
    proxy = manifest["study"]["selected_trial_fold_evaluation"]
    if missing_proxy_files:
        assert proxy["verification_mode"] == "optuna_journal_record"
        assert len(proxy["missing_local_artifacts"]) == 5
    assert proxy["source_sha256"] == manifest["split"]["source_sha256"]
    assert proxy["soccol_development_selection"] == manifest["split"]["soccol_development_selection"]

    eval_args = soccol_eval.parse_args(["--final-model-manifest", str(manifest_path)])
    result = json.loads(final_eval.run(eval_args).read_text())
    assert result["development_context_rows"] == 2950
    assert result["final_test_rows"] == 805
    assert not result["selection_or_tuning_performed"]
    assert final_eval.development_fold_counts(frozen) == {1: 622, 2: 615, 3: 638, 4: 445, 5: 630}
    fold_args = soccol_eval.parse_args([
        "--final-model-manifest", str(manifest_path), "--context-mode", "development-folds",
    ])
    fold_result = json.loads(final_eval.run(fold_args).read_text())
    assert [row["n_train"] for row in fold_result["fold_results"]] == [622, 615, 638, 445, 630]
    expected_queries = manifest["split"]["final_query_task_row_indices"]
    assert len(final_queries_seen) == 6
    assert all(queries == expected_queries for queries in final_queries_seen)
    dataset = load_soccol_dataset()
    frozen_indices = np.flatnonzero(dataset.assignments.outer_split.eq("final_test")).tolist()
    assert expected_queries == frozen_indices
    assert dataset.rows.iloc[expected_queries].composition_status.ne("accepted").sum() == 10

    payload = json.loads((tmp_path / "final/final_test_evaluation/results.json").read_text())
    config = final_eval.validate_frozen_model(frozen)
    # v2 used identical reviewed real inputs; only synthetic composition weights
    # changed in v3. Its frozen models must remain evaluable on the same queries.
    historical = deepcopy(frozen)
    historical.manifest["evaluation_configuration"]["soccol_feature_profile"] = "soccol_pitting_features_v2"
    final_eval.validate_frozen_model(historical)
    historical.manifest["evaluation_configuration"]["soccol_feature_profile"] = "soccol_pitting_features_v1"
    with pytest.raises(RuntimeError, match="reviewed schema"):
        final_eval.validate_frozen_model(historical)
    for key in ("context_task_row_indices", "query_task_row_indices"):
        forged = deepcopy(payload)
        forged["soccol_final_split"][key][0] = 0  # Keep counts equal; change membership.
        if forged == payload:
            forged["soccol_final_split"][key][0] = 1
        with pytest.raises(RuntimeError, match="membership"):
            final_eval.validate_final_results(forged, frozen=frozen, config=config)
    changed_split = deepcopy(manifest["split"])
    frozen.manifest["split"]["development_rows"] = 3222
    with pytest.raises(RuntimeError, match="context/query selection"):
        final_eval.validate_frozen_model(frozen)
    frozen.manifest["split"] = changed_split
    malformed_fold = deepcopy(proxy)
    malformed_fold["soccol_development_selection"]["source_rows"] = 3222
    with pytest.raises(RuntimeError, match="accepted development selection"):
        search.validate_reviewed_soccol_fold_selection(malformed_fold)


def test_legacy_final_route_requires_explicit_selection(reviewed):
    args = soccol_train.parse_args(["--legacy-data", "--storage", "journal:///unused.log"])
    assert args.study_name.endswith("_v1")
    assert args.checkpoint_root.name == "soccol_pipeline_final_v1"
    args = soccol_eval.parse_args(["--legacy-data"])
    assert args.final_model_manifest.parent.parent.name == "final_v1"
    assert final_eval.DEVELOPMENT_ROWS == 3222
    assert final_eval.FINAL_TEST_ROWS == 805
