"""Exercise real reviewed split/evaluation orchestration with mock model scores."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline.artifact_hashes import build_final_model_lock, sha256_file
from scripts.soccol_pipeline import evaluate_baseline_folds as baselines
from scripts.soccol_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.soccol_pipeline.data import load_soccol_dataset, reviewed_final_split_metadata, select_accepted_development
from tabicl.prior.soccol_reviewed_schema import SOCCOL_REVIEWED_FEATURE_COLUMNS, SOCCOL_REVIEWED_CATEGORICAL_COLUMNS


@pytest.fixture
def mock_models(monkeypatch):
    calls = []

    def evaluate(**kwargs):
        task, split = kwargs["task"], kwargs["task_split"]
        assert tuple(task.X.columns) == SOCCOL_REVIEWED_FEATURE_COLUMNS
        source_indices = np.arange(len(task.X))
        if split.split_strategy.startswith("epit_pipeline_development_fold_"):
            source_indices = np.asarray(task.X.attrs["soccol_development_selection"]["source_task_row_indices"])
        calls.append((kwargs["model_label"], source_indices[split.train_index].tolist(), source_indices[split.test_index].tolist()))
        return {
            "model": kwargs["model_label"], "model_kind": kwargs["model_kind"], "task_id": task.task_id,
            "target_binning": "continuous", "n_samples": len(task.y), "n_features": 58,
            "n_train": len(split.train_index), "n_test": len(split.test_index),
            "split_strategy": split.split_strategy, "pitting_magpie_features": False,
            **{metric: .3 for metric in ("test_spearman", "test_mae", "test_rmse", "test_r2", "test_pearson", "test_nmae_iqr", "test_nrmse_iqr")},
        }

    def invoke(command, **kwargs):
        assert command[1].endswith("eval_corrosion_datasets.py")
        with monkeypatch.context() as local:
            local.setattr(sys, "argv", command[1:])
            ev.main()

    monkeypatch.setattr(ev, "evaluate_estimator", evaluate)
    monkeypatch.setattr(comparison.subprocess, "run", invoke)
    return calls


def test_baselines_use_exact_accepted_development_folds(tmp_path, mock_models):
    checkpoint = tmp_path / "step-1000.ckpt"
    checkpoint.write_bytes(b"mock checkpoint")
    args = baselines.parse_args(["--generic-checkpoint", str(checkpoint), "--device", "cpu", "--output-dir", str(tmp_path / "results")])
    output = baselines.run(args)
    result = json.loads(output.read_text())
    assert not result["final_test_rows_used"] and result["final_test_rows_excluded"] == 805
    assert result["development_rows"] == 2950
    assert result["settings"]["catboost_categorical_columns"] == list(SOCCOL_REVIEWED_CATEGORICAL_COLUMNS)
    assert Counter(call[0] for call in mock_models) == {label: 5 for label in baselines.baseline.EXPECTED_MODELS}
    selection = select_accepted_development(load_soccol_dataset()).identity
    for fold in range(1, 6):
        expected = baselines.fold_membership(selection, fold)
        for _, context, queries in mock_models[(fold-1)*3:fold*3]:
            assert context == expected["context_task_row_indices"]
            assert queries == expected["query_task_row_indices"]
    rows = pd.read_csv(output.parent / "fold_1/rows.csv")
    payload = json.loads((output.parent / "fold_1/results.json").read_text())
    payload["soccol_development_split"]["query_task_row_indices"][0] = -1
    with pytest.raises(RuntimeError, match="membership"):
        baselines.validate_fold_results(payload, rows, selection=selection, fold=1)


@pytest.mark.parametrize("context_folds", [None, 5])
def test_comparison_all_models_keep_frozen_queries_and_checkpoint_endpoints(tmp_path, mock_models, context_folds):
    candidate, generic = tmp_path / "candidate", tmp_path / "generic"
    candidate.mkdir(); generic.mkdir()
    for directory, steps in ((candidate, [500, 1400]), (generic, [500, 1000, 2000])):
        for step in steps:
            (directory / f"step-{step}.ckpt").write_bytes(b"mock checkpoint")
    argv = ["--candidate-run", str(candidate), "--generic-run", str(generic), "--output-dir", str(tmp_path / "results")]
    if context_folds:
        argv += ["--context-folds", str(context_folds)]
    args = comparison.parse_args(argv)
    output = comparison.run(args)
    payload = json.loads((output / "results.json").read_text())
    metadata = reviewed_final_split_metadata()
    selection = metadata["soccol_development_selection"]
    for _, context, queries in mock_models:
        assert queries == metadata["final_query_task_row_indices"]
        assert set(context) <= set(selection["source_task_row_indices"])
        assert len(context) in ([622, 615, 638, 445, 630] if context_folds else [2950])
    rows = payload["rows"]
    assert {row["checkpoint_step"] for row in rows if row["model"] == "soccol_reviewed"} == {500, 1400}
    assert {row["checkpoint_step"] for row in rows if row["model"] == "generic_baseline"} == {500, 1000, 2000}
    assert all((output / "plots" / name).stat().st_size > 0 for name in comparison.REQUIRED_PLOTS)
    if context_folds is None:
        bad = deepcopy(payload)
        bad["soccol_final_split"]["query_task_row_indices"][0] = -1
        with pytest.raises(RuntimeError, match="membership"):
            comparison.validate_results(args, bad)
    with pytest.raises(FileExistsError, match="not empty"):
        comparison.run(args)


def test_preview_needs_no_soccol_checkpoint_and_creates_no_results(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("A preview must not launch evaluation.")
    monkeypatch.setattr(comparison.subprocess, "run", unexpected)
    output = tmp_path / "preview"
    comparison.run(comparison.parse_args(["--candidate-run", str(tmp_path / "future_run"), "--output-dir", str(output), "--dry-run"]))
    baselines.run(baselines.parse_args(["--generic-checkpoint", str(tmp_path / "future_generic.ckpt"), "--output-dir", str(output), "--dry-run"]))
    printed = capsys.readouterr().out
    assert printed.count("--epit-validation-fold") == 5
    assert printed.count("--epit-final-test") == 1
    assert not output.exists()


def test_frozen_manifest_finds_reviewed_run_and_rejects_historical_schema(tmp_path):
    checkpoint = tmp_path / "new_trial_0042" / "step-2500.ckpt"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"mock frozen checkpoint")
    manifest_path = tmp_path / "final_model_manifest.json"
    manifest = {
        "schema_version": "epit_final_model_manifest_v1", "final_test_rows_used": False,
        "split": reviewed_final_split_metadata(),
        "selected_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "evaluation_configuration": {"task_id": comparison.SOCCOL_TASK_ID, "tabicl_norm_methods": ["none"],
                                     "soccol_reviewed_data": True, "soccol_feature_profile": "soccol_pitting_features_v3",
                                     "expected_n_features": 58, "col_missing_value_embedding": False},
    }
    def save():
        manifest_path.write_text(json.dumps(manifest))
        lock = build_final_model_lock(manifest_path=manifest_path, immutable_artifacts={"selected_checkpoint": checkpoint})
        (tmp_path / "final_model_lock.json").write_text(json.dumps(lock))
    save()
    args = comparison.parse_args(["--final-model-manifest", str(manifest_path)])
    assert comparison.candidate_run(args) == checkpoint.parent
    manifest["evaluation_configuration"]["soccol_reviewed_data"] = False
    save()
    with pytest.raises(RuntimeError, match="reviewed Soccol"):
        comparison.candidate_run(args)


def test_default_never_falls_back_to_trial19_and_legacy_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "DEFAULT_FINAL_MODEL_MANIFEST", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="not available yet"):
        comparison.evaluation_command(comparison.parse_args([]))
    command = comparison.evaluation_command(comparison.parse_args(["--legacy-data"]))
    assert "--soccol-reviewed-data" not in command and "--missing-aware-model" not in command
    assert str(comparison.LEGACY_CANDIDATE_RUN) in command


def test_candidate_labels_match_evaluator_normalization():
    args = comparison.parse_args(["--candidate-label", "Soccol 25 percent", "--candidate-run", "/tmp/future_run"])
    assert args.candidate_label == ev.slugify("Soccol 25 percent")
    command = comparison.evaluation_command(args)
    assert "--missing-aware-model" not in command
    assert "--native-tabicl-baselines" in command
    with pytest.raises(SystemExit):
        comparison.parse_args(["--candidate-label", "generic_baseline"])
