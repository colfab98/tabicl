from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import eval_corrosion_datasets as corrosion_eval
from scripts.combined_corrosion import evaluate_epit_continuations, evaluate_final, evaluate_shared_specialists


def shared_args(monkeypatch, command):
    monkeypatch.setattr(sys, "argv", command[1:])
    return corrosion_eval.parse_args()


def test_shared_specialists_reuse_existing_frozen_query_protocol(monkeypatch) -> None:
    original = shared_args(
        monkeypatch,
        evaluate_final.evaluation_command(evaluate_final.parse_args(["--device", "cpu"])),
    )
    options = evaluate_shared_specialists.parse_args(["--device", "cpu"])
    updated = shared_args(monkeypatch, evaluate_shared_specialists.evaluation_command(options))

    changed_fields = {
        "run", "local_model_label", "fixed_schema_model", "missing_aware_model",
        "output_json", "output_csv", "output_wide_csv", "output_summary_csv", "output_plot_dir",
    }
    assert {
        key: value for key, value in vars(updated).items() if key not in changed_fields
    } == {
        key: value for key, value in vars(original).items() if key not in changed_fields
    }
    assert updated.task == [corrosion_eval.COMBINED_PIPELINE_TASK_ID]
    assert updated.local_model_label == [
        "epit_shared_v2", "soccol_shared_v2", "combined_corrpfn_v2", "generic_tabicl",
    ]
    assert updated.run == [
        str(path.resolve()) for path in (
            options.epit_run, options.soccol_run, options.combined_run, options.generic_run,
        )
    ]
    assert updated.output_json.parent == options.output_dir
    assert updated.output_json.parent != original.output_json.parent


def test_shared_specialists_preserve_missing_aware_and_native_baseline_settings(monkeypatch) -> None:
    args = shared_args(
        monkeypatch,
        evaluate_shared_specialists.evaluation_command(evaluate_shared_specialists.parse_args([])),
    )
    labels = set(args.local_model_label)
    corrosion_eval.validate_fixed_schema_model_labels(args, labels)
    corrosion_eval.validate_missing_aware_model_labels(args, labels)
    for label in ("epit_shared_v2", "soccol_shared_v2", "combined_corrpfn_v2"):
        assert corrosion_eval.model_uses_fixed_schema(args, label)
        assert corrosion_eval.expected_missing_value_mode(args, label) is True
        assert corrosion_eval.tabicl_inference_settings(args, label) == ("none", ["none"])
    for label, pretrained in (("generic_tabicl", False), ("pretrained_tabicl_v2", True)):
        assert not corrosion_eval.model_uses_fixed_schema(args, label)
        assert corrosion_eval.expected_missing_value_mode(args, label, pretrained=pretrained) is False
        assert corrosion_eval.tabicl_inference_settings(args, label, pretrained=pretrained) == ("latin", None)
    assert args.compare_pretrained_tabicl
    assert args.compare_catboost


@pytest.mark.parametrize("skip_combined", [False, True])
def test_shared_specialists_continue_past_shorter_runs_without_swapping_labels(
    tmp_path: Path, monkeypatch, skip_combined: bool
) -> None:
    argv = ["--device", "cpu", "--output-dir", str(tmp_path / "results")]
    expected_runs = {}
    for name, label, steps in (
        ("epit", "epit_shared_v2", [500, 1000, 1500]),
        ("soccol", "soccol_shared_v2", [500, 1000, 1500]),
        ("combined", "combined_corrpfn_v2", [500]),
        ("generic", "generic_tabicl", [500, 1000, 1500]),
    ):
        run = tmp_path / name
        run.mkdir()
        for step in steps:
            (run / f"step-{step}.ckpt").touch()
        argv.extend([f"--{name}-run", str(run)])
        if name != "combined" or not skip_combined:
            expected_runs[label] = run
    if skip_combined:
        argv.append("--skip-combined")

    options = evaluate_shared_specialists.parse_args(argv)
    args = shared_args(monkeypatch, evaluate_shared_specialists.evaluation_command(options))
    specs = corrosion_eval.resolve_checkpoint_eval_specs(args)

    assert [spec.checkpoint_step for spec in specs] == [500, 1000, 1500]
    for spec in specs:
        assert {local.label: local.checkpoint_path for local in spec.local_specs} == {
            label: run / f"step-{spec.checkpoint_step}.ckpt"
            for label, run in expected_runs.items()
            if (run / f"step-{spec.checkpoint_step}.ckpt").exists()
        }
    if skip_combined:
        assert "combined_corrpfn_v2" not in args.fixed_schema_model
        assert "combined_corrpfn_v2" not in args.missing_aware_model


def test_continuations_preserve_existing_evaluation_protocol(monkeypatch):
    original = shared_args(
        monkeypatch,
        evaluate_final.evaluation_command(evaluate_final.parse_args(["--device", "cpu"])),
    )
    options = evaluate_epit_continuations.parse_args(["--device", "cpu"])
    updated = shared_args(monkeypatch, evaluate_epit_continuations.evaluation_command(options))
    changed_fields = {
        "run", "local_model_label", "fixed_schema_model", "missing_aware_model",
        "output_json", "output_csv", "output_wide_csv", "output_summary_csv", "output_plot_dir",
    }
    assert {key: value for key, value in vars(updated).items() if key not in changed_fields} == {
        key: value for key, value in vars(original).items() if key not in changed_fields
    }
    assert updated.local_model_label == ["epit5k_control", "epit5k_to_soccol", "generic_tabicl"]
    assert [Path(run).name for run in updated.run] == [
        "epit_v2_5000_control_seed42", "epit_v2_5000_to_soccol_seed42", "tabicl_s1_regression_baseline",
    ]
    labels = set(updated.local_model_label)
    corrosion_eval.validate_fixed_schema_model_labels(updated, labels)
    corrosion_eval.validate_missing_aware_model_labels(updated, labels)
    for label in ("epit5k_control", "epit5k_to_soccol"):
        assert corrosion_eval.model_uses_fixed_schema(updated, label)
        assert corrosion_eval.expected_missing_value_mode(updated, label) is True
        assert corrosion_eval.tabicl_inference_settings(updated, label) == ("none", ["none"])
    for label, pretrained in (("generic_tabicl", False), ("pretrained_tabicl_v2", True)):
        assert not corrosion_eval.model_uses_fixed_schema(updated, label)
        assert corrosion_eval.expected_missing_value_mode(updated, label, pretrained=pretrained) is False
        assert corrosion_eval.tabicl_inference_settings(updated, label, pretrained=pretrained) == ("latin", None)
    assert updated.compare_catboost and updated.compare_pretrained_tabicl
    assert updated.output_json.parent == options.output_dir
    assert options.output_dir != evaluate_final.DEFAULT_OUTPUT_DIR


@pytest.mark.parametrize(("checkpoint", "steps"), [("all", [5000, 5500, 6000, 6500]), ("6000", [6000])])
def test_continuations_resolve_only_the_requested_models(tmp_path, monkeypatch, checkpoint, steps):
    argv = ["--checkpoint", checkpoint]
    expected_runs = {}
    for option, label, saved_steps in (
        ("control", "epit5k_control", [5500, 6000, 6500]),
        ("soccol", "epit5k_to_soccol", [5500, 6000]),
        ("generic", "generic_tabicl", [5000, 5500, 6000, 6500]),
    ):
        directory = tmp_path / option
        directory.mkdir()
        for step in saved_steps:
            (directory / f"step-{step}.ckpt").touch()
        argv.extend([f"--{option}-run", str(directory)])
        expected_runs[label] = directory
    args = shared_args(monkeypatch, evaluate_epit_continuations.evaluation_command(
        evaluate_epit_continuations.parse_args(argv)
    ))
    specifications = corrosion_eval.resolve_checkpoint_eval_specs(args)
    assert [spec.checkpoint_step for spec in specifications] == steps
    for spec in specifications:
        assert {model.label: model.checkpoint_path for model in spec.local_specs} == {
            label: directory / f"step-{spec.checkpoint_step}.ckpt"
            for label, directory in expected_runs.items()
            if (directory / f"step-{spec.checkpoint_step}.ckpt").exists()
        }
