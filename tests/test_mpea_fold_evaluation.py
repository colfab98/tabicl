"""Protect the separate MPEA benchmark's outer folds, inputs, and aggregation."""
from __future__ import annotations

from collections import Counter
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from scripts import eval_corrosion_datasets as ev
from scripts.combined_corrosion import evaluate_mpea_folds as mfe
from scripts.combined_corrosion.evaluate_final import DEFAULT_DATA_DIR


def test_mpea_loader_uses_only_mpea_and_verifies_frozen_inputs(tmp_path):
    for name in (ev.COMBINED_MANIFEST_NAME, ev.COMBINED_QUERY_NAME,
                 ev.COMBINED_SCORING_NAME, ev.COMBINED_REGISTRY_NAME):
        shutil.copyfile(DEFAULT_DATA_DIR / name, tmp_path / name)
    task, ids, hashes = mfe.load_mpea(tmp_path)
    assert task.X.shape == (267, 50)
    assert len(ids) == len(set(ids)) == 267
    assert all(row_id.startswith("mpea_") for row_id in ids)
    assert task.fixed_split is None
    assert task.target not in task.X
    assert task.X.isna().any().any()
    assert ev.COMBINED_CONTEXT_NAME not in hashes
    with (tmp_path / ev.COMBINED_QUERY_NAME).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="artifact changed"):
        mfe.load_mpea(tmp_path)


def test_frozen_folds_cover_each_query_once_without_close_composition_overlap(tmp_path):
    task, ids, hashes = mfe.load_mpea(DEFAULT_DATA_DIR)
    original_X = task.X.copy(deep=True)
    path = tmp_path / "folds.json"
    manifest = mfe.freeze_folds(task, ids, hashes, path, 42)
    original_bytes = path.read_bytes()
    assert mfe.freeze_folds(task, ids, hashes, path, 42) == manifest
    assert path.read_bytes() == original_bytes
    tasks = mfe.fold_tasks(task, manifest)
    seen = []
    vectors = task.X[list(mfe.COMBINED_COMPOSITION_COLUMNS)].fillna(0).to_numpy().round(2)
    distances = squareform(pdist(vectors, metric="cityblock"))
    for fold_task in tasks:
        split = fold_task.fixed_split
        context, query = split.train_index, split.test_index
        assert not set(context) & set(query)
        assert set(context) | set(query) == set(range(267))
        assert distances[np.ix_(context, query)].min() > 1.0
        assert split.preprocessing_index is None  # Fit on this context only.
        assert len(query) in (53, 54)
        seen.extend(query)
    assert Counter(seen) == Counter(range(267))
    pd.testing.assert_frame_equal(task.X, original_X)
    with pytest.raises(ValueError, match="Frozen MPEA folds differ"):
        mfe.freeze_folds(task, ids, hashes, path, 43)
    assert path.read_bytes() == original_bytes


def test_grouping_keeps_transitive_neighbours_together_and_ignores_targets():
    X = pd.DataFrame(0.0, index=range(12), columns=mfe.COMBINED_FEATURE_COLUMNS)
    X["Fe"] = [0, .7, 1.4, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    groups, folds = mfe.composition_folds(X, 42)
    assert groups[0] == groups[1] == groups[2]
    assert folds[0] == folds[1] == folds[2]
    assert len(set(folds)) == 5


def complete_rows():
    return [
        {"checkpoint_name": "step-1000", "checkpoint_step": 1000, "model": model,
         "model_kind": "test", "task_id": f"mpea_fold_{fold}", "outer_fold": fold,
         "n_samples": 267, "n_test": 20 * fold, "test_spearman": fold / 10,
         "test_rmse": float(fold)}
        for model in mfe.MODEL_LABELS for fold in range(1, 6)
    ]


def test_means_are_unweighted_fold_metrics_and_sd_is_sample_sd():
    summary = mfe.summarize(complete_rows())
    assert len(summary) == 7
    assert (summary.n_folds == 5).all()
    assert (summary.n_success == 5).all()
    np.testing.assert_allclose(summary.mean_test_spearman, .3)
    np.testing.assert_allclose(summary.std_test_spearman, np.std([.1, .2, .3, .4, .5], ddof=1))
    assert (summary.n_valid_folds_test_spearman == 5).all()


@pytest.mark.parametrize("change", ["missing_fold", "duplicate_fold", "missing_model"])
def test_partial_or_duplicate_results_cannot_be_averaged(change):
    rows = complete_rows()
    if change == "missing_fold":
        rows.pop()
    elif change == "duplicate_fold":
        rows.append(rows[0].copy())
    else:
        rows = [row for row in rows if row["model"] != "catboost"]
    with pytest.raises(ValueError, match="every scheduled model on all five folds"):
        mfe.summarize(rows)


def test_undefined_correlations_are_not_silently_dropped_from_average():
    rows = complete_rows()
    rows[0]["test_spearman"] = np.nan
    summary = mfe.summarize(rows).set_index("model")
    assert np.isnan(summary.loc[mfe.MODEL_LABELS[0], "mean_test_spearman"])
    assert summary.loc[mfe.MODEL_LABELS[0], "n_valid_folds_test_spearman"] == 4
    assert summary.loc["catboost", "mean_test_spearman"] == pytest.approx(.3)


def test_seven_models_preserve_inference_protocol_and_continue_past_shorter_runs(tmp_path):
    argv = ["--device", "cpu"]
    for name in ("combined", "epit", "soccol", "g50", "generic"):
        path = tmp_path / name
        path.mkdir()
        for step in ([500, 1000] if name == "g50" else [500, 1000, 1500]):
            (path / f"step-{step}.ckpt").touch()
        argv.extend([f"--{name}-run", str(path)])
    args = mfe.shared_options(mfe.parse_args(argv))
    specs = ev.resolve_checkpoint_eval_specs(args)
    assert [spec.checkpoint_step for spec in specs] == [500, 1000, 1500]
    assert {local.label for local in specs[-1].local_specs} == (
        set(mfe.MODEL_LABELS) - {"catboost", "pretrained_tabicl_v2", "combined_g50_epit25_soccol25"}
    )
    assert args.task == [mfe.TASK_ID]
    assert args.combined_data_dir is None
    assert args.compare_pretrained_tabicl and args.compare_catboost
    assert set(args.local_model_label) == set(mfe.MODEL_LABELS) - {"catboost", "pretrained_tabicl_v2"}
    for label in mfe.MISSING_AWARE_MODELS:
        assert ev.tabicl_inference_settings(args, label) == ("none", ["none"])
        assert ev.expected_missing_value_mode(args, label) is True
    for label in ("generic_tabicl", "pretrained_tabicl_v2"):
        assert ev.tabicl_inference_settings(args, label, pretrained=label.startswith("pretrained")) == ("latin", None)
        assert ev.expected_missing_value_mode(args, label) is False


def test_output_has_one_averaged_wide_row_per_checkpoint_and_five_fold_tables(tmp_path):
    mfe.write_results(tmp_path, complete_rows(), {"scope": "test"})
    wide = pd.read_csv(tmp_path / "wide.csv")
    assert len(wide) == 1
    assert wide.iloc[0]["mean_test_spearman__catboost"] == pytest.approx(.3)
    for fold in range(1, 6):
        rows = pd.read_csv(tmp_path / f"outer_fold_{fold}" / "rows.csv")
        assert len(rows) == 7
        assert set(rows.outer_fold) == {fold}
    payload = json.loads((tmp_path / "results.json").read_text())
    assert len(payload["rows"]) == 35


def test_later_steps_require_all_folds_only_for_scheduled_models(tmp_path):
    early = complete_rows()
    late_models = {"generic_tabicl", "pretrained_tabicl_v2", "catboost"}
    late = [
        {**row, "checkpoint_name": "step-2000", "checkpoint_step": 2000}
        for row in early if row["model"] in late_models
    ]
    expected = {"step-1000": set(mfe.MODEL_LABELS), "step-2000": late_models}
    summary = mfe.summarize(early + late, expected)
    assert set(summary.loc[summary.checkpoint_step == 2000, "model"]) == late_models
    assert (summary.n_folds == 5).all()
    with pytest.raises(ValueError, match="every scheduled model"):
        mfe.summarize(early + late[:-1], expected)
    with pytest.raises(ValueError, match="every scheduled model"):
        mfe.summarize(early + [row for row in late if row["model"] != "catboost"], expected)

    metadata = {"checkpoint_evaluations": [
        {"name": name, "models": {label: "unused.ckpt" for label in models
                                  if label not in {"pretrained_tabicl_v2", "catboost"}}}
        for name, models in expected.items()
    ]}
    mfe.write_results(tmp_path, early + late, metadata)
    wide = pd.read_csv(tmp_path / "wide.csv").set_index("checkpoint_step")
    assert np.isnan(wide.loc[2000, "mean_test_spearman__combined_corrpfn_v2"])
    assert wide.loc[2000, "mean_test_spearman__generic_tabicl"] == pytest.approx(.3)
    for fold in range(1, 6):
        rows = pd.read_csv(tmp_path / f"outer_fold_{fold}" / "rows.csv")
        assert set(rows.loc[rows.checkpoint_step == 2000, "model"]) == late_models


def test_mpea_driver_completes_unequal_runs_and_keeps_frozen_queries(tmp_path, monkeypatch):
    output = tmp_path / "results"
    argv = ["--device", "cpu", "--output-dir", str(output),
            "--split-manifest", str(tmp_path / "folds.json")]
    for name in ("combined", "epit", "soccol", "g50", "generic"):
        run = tmp_path / name
        run.mkdir()
        for step in ([500, 1000] if name == "generic" else [500]):
            (run / f"step-{step}.ckpt").touch()
        argv.extend([f"--{name}-run", str(run)])
    seen_queries = {}
    calls = Counter()

    def fake_evaluate(args, tasks, label, path=None):
        calls[label] += 1
        rows = []
        for fold, task in enumerate(tasks, 1):
            query = tuple(task.fixed_split.test_index)
            assert seen_queries.setdefault(fold, query) == query
            rows.append({
                "model": label, "model_kind": "test", "outer_fold": fold,
                "task_id": task.task_id, "test_spearman": .3, "n_samples": 267,
            })
        return rows

    monkeypatch.setattr(mfe, "evaluate_model", fake_evaluate)
    mfe.main(argv)
    payload = json.loads((output / "results.json").read_text())
    assert payload["completed_checkpoints"] == ["step-500", "step-1000"]
    assert calls["generic_tabicl"] == 2
    assert all(calls[label] == 1 for label in set(mfe.MODEL_LABELS) - {"generic_tabicl"})
    late = [row for row in payload["rows"] if row["checkpoint_step"] == 1000]
    assert Counter(row["model"] for row in late) == {
        "generic_tabicl": 5, "pretrained_tabicl_v2": 5, "catboost": 5,
    }
    assert sorted(index for query in seen_queries.values() for index in query) == list(range(267))
