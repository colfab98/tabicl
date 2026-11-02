"""Checkpoint sweeps must retain every run's available steps and frozen queries."""
from collections import Counter
import json
import sys

import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as ev
from scripts.combined_corrosion.evaluate_final import DEFAULT_DATA_DIR


def comparison_args(tmp_path, steps_by_model, *extra):
    argv = ["--checkpoint", "all", "--min-checkpoint-step", "0",
            "--checkpoint-step-interval", "0", "--device", "cpu"]
    for label, steps in steps_by_model.items():
        directory = tmp_path / label
        directory.mkdir(exist_ok=True)
        for step in steps:
            (directory / f"step-{step}.ckpt").touch()
        argv.extend(["--run", str(directory), "--local-model-label", label])
    return argv + list(extra)


@pytest.mark.parametrize(("extra", "expected"), [
    ([], {500: {"early"}, 1000: {"early", "late"}, 1400: {"early"},
          2000: {"late"}, 2300: {"late"}}),
    (["--checkpoint-step-interval", "1000"],
     {1000: {"early", "late"}, 1400: {"early"}, 2000: {"late"}, 2300: {"late"}}),
    (["--checkpoint-step-interval", "1000", "--no-include-latest-common-checkpoint"],
     {1000: {"early", "late"}, 2000: {"late"}}),
    (["--min-checkpoint-step", "2000"], {2000: {"late"}, 2300: {"late"}}),
    (["--max-checkpoint-step", "1400"],
     {500: {"early"}, 1000: {"early", "late"}, 1400: {"early"}}),
    (["--max-checkpoint-step", "1600", "--checkpoint-step-interval", "1000"],
     {1000: {"early", "late"}, 1400: {"early"}}),
    (["--max-checkpoint-step", "1600", "--checkpoint-step-interval", "1000",
      "--no-include-latest-checkpoint"], {1000: {"early", "late"}}),
])
def test_checkpoint_union_respects_filters_and_each_runs_endpoint(tmp_path, extra, expected):
    args = ev.parse_args(comparison_args(
        tmp_path, {"early": [500, 1000, 1400], "late": [1000, 2000, 2300]}, *extra,
    ))
    specs = ev.resolve_checkpoint_eval_specs(args)
    assert {spec.checkpoint_step: {local.label for local in spec.local_specs}
            for spec in specs} == expected
    for spec in specs:
        assert all(local.checkpoint_path.name == f"step-{spec.checkpoint_step}.ckpt"
                   for local in spec.local_specs)


def test_disjoint_runs_are_evaluated_but_explicit_missing_steps_fail(tmp_path):
    argv = comparison_args(tmp_path, {"early": [500], "late": [1000]})
    assert [spec.checkpoint_step for spec in ev.resolve_checkpoint_eval_specs(ev.parse_args(argv))] == [
        500, 1000,
    ]
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        ev.resolve_checkpoint_eval_specs(ev.parse_args(argv + ["--checkpoint", "1000"]))
    with pytest.raises(FileNotFoundError, match="No step-"):
        ev.resolve_checkpoint_eval_specs(ev.parse_args(argv + ["--min-checkpoint-step", "1500"]))
    with pytest.raises(FileNotFoundError, match="through 400"):
        ev.resolve_checkpoint_eval_specs(ev.parse_args(argv + ["--max-checkpoint-step", "400"]))


@pytest.mark.parametrize("extra", [
    ["--max-checkpoint-step", "-1"],
    ["--min-checkpoint-step", "1000", "--max-checkpoint-step", "500"],
    ["--checkpoint", "latest", "--max-checkpoint-step", "5000"],
])
def test_invalid_checkpoint_caps_are_rejected(extra):
    with pytest.raises(ValueError, match="max-checkpoint-step"):
        ev.validate_args(ev.parse_args(["--checkpoint", "all", *extra]))


@pytest.mark.parametrize("min_step", [0, 1000])
def test_shared_driver_keeps_queries_and_baselines_for_unequal_runs(tmp_path, monkeypatch, min_step):
    output = tmp_path / "results"
    argv = comparison_args(
        tmp_path, {"early": [500], "late": [1000, 1500]},
        "--min-checkpoint-step", str(min_step),
        "--fixed-schema-model", "early", "--fixed-schema-model", "late",
        "--missing-aware-model", "early", "--missing-aware-model", "late",
        "--target-binning", "continuous", "--compare-pretrained-tabicl", "--compare-catboost",
        "--output-json", str(output / "results.json"),
        "--output-csv", str(output / "rows.csv"),
        "--output-plot-dir", str(output / "plots"),
    )
    task = ev._make_combined_pipeline_task(DEFAULT_DATA_DIR)
    monkeypatch.setattr(ev, "make_tasks", lambda args: [task])
    calls = Counter()

    def fake_evaluate(**kwargs):
        assert kwargs["task"] is task
        assert kwargs["task_split"].test_index.tolist() == list(range(4787, 5054))
        label = kwargs["model_label"]
        calls[label] += 1
        return {
            "model": label, "model_kind": kwargs["model_kind"], "task_id": task.task_id,
            "target_binning": "continuous", "n_samples": len(task.y), "test_spearman": .3,
        }

    monkeypatch.setattr(ev, "evaluate_estimator", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["eval_corrosion_datasets.py", *argv])
    ev.main()
    payload = json.loads((output / "results.json").read_text())
    assert not payload["errors"]
    expected_steps = [500, 1000, 1500] if min_step == 0 else [1000, 1500]
    assert [spec["checkpoint_step"] for spec in payload["checkpoint_eval_specs"]] == expected_steps
    assert calls == Counter({"early": int(min_step == 0), "late": 2,
                             "pretrained_tabicl_v2": 1, "catboost": 1})
    rows = pd.DataFrame(payload["rows"])
    assert set(rows.loc[rows.model == "late", "checkpoint_step"]) == {1000, 1500}
    for label in ("pretrained_tabicl_v2", "catboost"):
        assert rows.loc[rows.model == label, "checkpoint_step"].tolist() == expected_steps
    wide = pd.read_csv(output / "rows_wide.csv").set_index("checkpoint_step")
    if min_step == 0:
        assert pd.isna(wide.loc[1500, "test_spearman__early"])
        assert pd.isna(wide.loc[500, "test_spearman__late"])
    assert wide.loc[1500, "test_spearman__late"] == pytest.approx(.3)
    assert (output / "plots/trend_mean_test_spearman.svg").is_file()
