"""Selected-pair S2 must retain the old control settings and evaluation membership."""
import json
from pathlib import Path

import pytest

from scripts.epit_pipeline import evaluate_finetuning, finetune, finetune_selected_pairs as selected
from scripts.epit_pipeline import train_selected_pairs
from tests.test_epit_finetuning import legacy_parent, stage_args, dry_plan
from tests.test_epit_reviewed_comparisons import mock_models


def new_plan(tmp_path, parent):
    return selected.build_plan(selected.parse_args([
        "--finetune-from", str(parent), "--checkpoint-root", str(tmp_path / "checkpoints"),
        "--output-dir", str(tmp_path / "output"),
    ]))["training"]


def test_selected_recipe_matches_scratch_and_s2_control(tmp_path, legacy_parent, capsys):
    informed = new_plan(tmp_path, legacy_parent)
    control = dry_plan(stage_args(tmp_path, legacy_parent, 0), capsys)
    a, b = [vars(finetune.training_config(s["train_command"])) for s in (control, informed)]
    assert {key for key in a if a[key] != b[key]} == {
        "checkpoint_dir", "epit_selected_pair_dir", "epit_target_version", "informed_prior_ratio",
        "informed_target_mix_weight", "informed_target_mix_policy", "pitting_coefficient_variation",
    }
    from argparse import Namespace
    scratch = train_selected_pairs.build_plan(Namespace(
        recipe_dir=selected.DEFAULT_RECIPE, run_name="test_selected_s2_dry_comparison",
        device="cuda", max_steps=10000, scheduler_total_steps=10000, nproc_per_node=1, workers=4))
    assert informed["selected_pair_finetuning"]["recipe_identity"] == scratch["recipe_identity"]
    assert b["informed_prior_ratio"] == pytest.approx(2300 / 7600)
    assert not (tmp_path / "checkpoints").exists()


def test_launcher_records_actual_recipe_and_refuses_overwrite(tmp_path, legacy_parent, monkeypatch):
    import torch
    from scripts.epit_pipeline import train_final
    calls = []
    def run(command, *, cwd, log_path):
        config = finetune.training_config(command)
        calls.append(config)
        log_path.write_text("mock training\n")
        for step in (0, 2000):
            torch.save({"curr_step": step}, Path(config.checkpoint_dir) / f"step-{step}.ckpt")
    monkeypatch.setattr(train_final.search.original.search_utils, "run_command", run)
    monkeypatch.setattr(train_final, "evaluation_record", lambda *a, **kw: pytest.fail("Unexpected evaluation"))
    argv = ["--finetune-from", str(legacy_parent), "--checkpoint-root", str(tmp_path / "checkpoints"),
            "--output-dir", str(tmp_path / "output")]
    selected.main(argv)
    saved = json.loads((Path(calls[0].checkpoint_dir) / "finetuning.json").read_text())
    selected.validate_selected_stage(saved)
    assert saved["selected_pair_finetuning"]["all_data_recipe"]
    assert "epit_target_recipe" not in saved["overrides"]
    assert json.loads((tmp_path / "output/terminal_checkpoint.json").read_text())["step"] == 2000
    with pytest.raises(FileExistsError):
        selected.main(argv)
    assert len(calls) == 1


def stage_directories(tmp_path, parent, capsys):
    informed = new_plan(tmp_path, parent)
    control = dry_plan(stage_args(tmp_path, parent, 0), capsys)
    # Historical control used an earlier dispatcher. Scientific settings remain fixed.
    control["overrides"]["finetuning"]["implementation_sha256"]["run.py"] = "historical"
    directories = []
    for stage in (informed, control):
        directory = Path(stage["checkpoint_dir"])
        directory.mkdir(parents=True)
        (directory / "finetuning.json").write_text(json.dumps(stage))
        for step in (0, 100):
            (directory / f"step-{step}.ckpt").write_bytes(b"mock")
        directories.append(directory)
    return directories


def eval_args(tmp_path, directories, cv=True):
    return evaluate_finetuning.parse_args([
        "--finetune-run", str(directories[0]), "--control-run", str(directories[1]),
        "--output-dir", str(tmp_path / "evaluation"), "--device", "cpu",
        "--no-compare-catboost", *(["--exploratory-cv-folds", "5"] if cv else []),
    ])


def test_selected_s2_cv_uses_same_folds_and_preserves_step_zero(tmp_path, legacy_parent, capsys, mock_models):
    directories = stage_directories(tmp_path, legacy_parent, capsys)
    args = eval_args(tmp_path, directories)
    output = evaluate_finetuning.run(args)
    payload = json.loads((output / "results.json").read_text())
    assert not payload["primary_comparison_complete"]  # No 2000-step control yet.
    assert payload["exploratory_cv"]
    from scripts.epit_pipeline.exploratory_cv import fold_membership
    memberships = [fold_membership(args.split_manifest, fold) for fold in range(1, 6)]
    for _, context, queries in mock_models:
        assert {"context_task_row_indices": context, "query_task_row_indices": queries} in memberships
    for label in ("epit_finetuned", "generic_continuation"):
        for step in (0, 100):
            assert {row["cv_fold"] for row in payload["rows"]
                    if row["model"] == label and row["checkpoint_step"] == step} == set(range(1, 6))


@pytest.mark.parametrize("change", ["lr", "seed", "parent", "ratio", "target", "final_test"])
def test_selected_comparison_rejects_unmatched_control_or_recipe(tmp_path, legacy_parent, capsys, change):
    directories = stage_directories(tmp_path, legacy_parent, capsys)
    path = directories[0 if change in ("ratio", "target") else 1] / "finetuning.json"
    stage = json.loads(path.read_text())
    if change in ("lr", "seed", "ratio", "target"):
        option, value = {"lr": ("--lr", "0.01"), "seed": ("--np_seed", "44"),
                         "ratio": ("--informed_prior_ratio", "1"),
                         "target": ("--informed_target_mix_weight", "0.5")}[change]
        stage["train_command"][stage["train_command"].index(option) + 1] = value
    elif change == "parent":
        stage["overrides"]["finetuning"]["parent_sha256"] = "different"
    path.write_text(json.dumps(stage))
    with pytest.raises(ValueError):
        evaluate_finetuning.validate_stages(evaluate_finetuning.comparison_args(
            eval_args(tmp_path, directories, cv=change != "final_test")))
