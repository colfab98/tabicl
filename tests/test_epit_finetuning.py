"""Fine-tuning initialization, restart, fixed endpoint, and frozen-query evaluation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.epit_pipeline import finetune, train_final, evaluate_finetuning
from scripts.epit_pipeline.artifact_hashes import load_frozen_final_model
from scripts.epit_pipeline.evaluation_identity import frozen_membership
from tabicl.train.run import Trainer
from tabicl.train.optim import get_scheduler
from tests.test_paired_prior_streams import small_training_config
from tests.test_epit_reviewed_comparisons import mock_models


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def legacy_parent(tmp_path):
    """Small metadata fixture, not an actual trained 10k checkpoint."""
    args = finetune.parse_args([])
    args.pitting_composition_mode = "empirical_features_scm_target"
    rules = train_final.search.load_target_rule_config(
        summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest)
    params = train_final.search.trial_params_from_mapping(
        {"mlp_prob": .7, "informed_target_mix_weight": finetune.TRIAL27_TARGET_WEIGHT,
         "pitting_composition_perturb_strength": .3},
        composition_mode=args.pitting_composition_mode, fixed_informed_prior_ratio=1.)
    command = train_final.search.training_command(args, params, tmp_path / "stage", rules)
    config = Trainer.model_configuration(finetune.training_config(command))
    config.pop("col_feature_group")
    config.pop("col_missing_value_embedding")
    path = tmp_path / "parent" / "step-10000.ckpt"
    path.parent.mkdir()
    torch.save({"curr_step": 10000, "config": config, "state_dict": {"test": torch.ones(1)}}, path)
    return path


def stage_args(tmp_path, parent, ratio=1, *extra):
    return finetune.parse_args([
        "--finetune-from", str(parent), "--override-informed-prior-ratio", str(ratio),
        "--final-run-name", f"stage_{ratio}", "--checkpoint-root", str(tmp_path / "checkpoints"),
        "--output-dir", str(tmp_path / f"output_{ratio}"), *extra,
    ])


def dry_plan(args, capsys):
    args.dry_run = True
    train_final.run(args)
    return json.loads(capsys.readouterr().out)["training"]


def test_both_commands_use_identical_fixed_s2_settings(tmp_path, legacy_parent, capsys):
    plans = [dry_plan(stage_args(tmp_path, legacy_parent, ratio), capsys) for ratio in (1, 0)]
    configs = [vars(finetune.training_config(plan["train_command"])) for plan in plans]
    assert {key for key in configs[0] if configs[0][key] != configs[1][key]} == {
        "checkpoint_dir", "informed_prior_ratio"}
    for config in configs:
        for key, value in {**finetune.S2_SETTINGS, **finetune.RETAINED_TABLE_SETTINGS}.items():
            assert config[key] == value
        assert config["only_load_model"] and config["save_initial_checkpoint"]
        assert config["checkpoint_path"] == str(legacy_parent)
        assert config["np_seed"] == config["torch_seed"] == config["prior_stream_seed"] == 43
    assert plans[0]["overrides"]["finetuning"] == plans[1]["overrides"]["finetuning"]
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("overrides", [
    ["--max-steps", "1000"], ["--np-seed", "44"],
    ["--override-informed-prior-ratio", ".5"], ["--override-mlp-prob", ".9"],
])
def test_predeclared_recipe_rejects_changed_settings(overrides):
    with pytest.raises(ValueError):
        finetune.validate_recipe(finetune.parse_args(overrides))


@pytest.mark.parametrize("change", ["step", "informed", "architecture", "regression"])
def test_parent_checks_reject_incompatible_weights(tmp_path, legacy_parent, change, capsys):
    state = torch.load(legacy_parent, weights_only=True)
    if change == "step":
        state["curr_step"] = 9000
    elif change == "informed":
        state["paired_prior_stream"] = {"version": "original_tabicl_mixture_v1", "group_counts": [64, 64]}
    elif change == "architecture":
        state["config"]["row_rope_base"] = 123.
    else:
        state["config"]["max_classes"] = 10
    torch.save(state, legacy_parent)
    with pytest.raises(ValueError):
        dry_plan(stage_args(tmp_path, legacy_parent), capsys)


def test_s2_lr_curve_starts_nonzero_and_ends_at_declared_floor():
    from tabicl.train.train_config import build_parser
    config = build_parser().parse_args([])
    for key, value in finetune.S2_SETTINGS.items():
        setattr(config, key, value)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=config.lr)
    scheduler = get_scheduler(config, optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    for step in range(1, 2001):
        optimizer.step()
        scheduler.step()
        if step == 1000:
            assert optimizer.param_groups[0]["lr"] == pytest.approx(8.75e-6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)


def stop_workers(trainer):
    iterator = trainer.dataloader._iterator
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


@pytest.mark.parametrize("workers", [0, 2])
def test_real_weights_step_zero_and_exact_resume(tmp_path, workers):
    config = small_training_config(tmp_path / "parent", "epit")
    config.original_prior_mix = True
    config.col_missing_value_embedding = False
    config.permute_features = True
    config.informed_prior_ratio = 0.
    parent = Trainer(config)
    parent.run_batch(next(parent.dataloader))
    parent.curr_step = 1
    parent.save_checkpoint("step-1.ckpt")
    parent_state = deepcopy(parent.raw_model.state_dict())
    stage = deepcopy(config)
    stage.checkpoint_dir = str(tmp_path / "stage")
    stage.checkpoint_path = str(tmp_path / "parent/step-1.ckpt")
    stage.only_load_model = stage.save_initial_checkpoint = True
    stage.informed_prior_ratio = 1.
    stage.epit_target_version = "v5"
    stage.informed_target_mix_policy = "uniform_per_table"
    stage.lr = 2e-5
    stage.scheduler = "polynomial_decay_warmup"
    stage.poly_decay_lr_end, stage.poly_decay_power = 5e-6, 2.
    stage.warmup_proportion = 0.
    stage.scheduler_total_steps = 2000
    stage.max_steps = 3  # Only three updates on a tiny CPU model.
    stage.save_temp_every = stage.save_perm_every = 1
    stage.np_seed = stage.torch_seed = stage.python_seed = stage.prior_stream_seed = 43
    stage.dataloader_num_workers = workers
    trained = Trainer(stage)
    if workers:
        trained.dataloader.kwargs["multiprocessing_context"] = "spawn"
    assert trained.curr_step == 0 and not trained.optimizer.state
    assert trained.optimizer.param_groups[0]["lr"] == 2e-5
    for name, weight in parent_state.items():
        torch.testing.assert_close(trained.raw_model.state_dict()[name], weight, rtol=0, atol=0)
    try:
        trained.train()
    finally:
        stop_workers(trained)
    zero_path = tmp_path / "stage/step-0.ckpt"
    zero_hash = train_final.sha256_file(zero_path)
    zero = torch.load(zero_path, weights_only=True)
    assert zero["curr_step"] == 0 and not zero["optimizer_state"]["state"]
    assert zero["sampling_state"]["ranks"][0]["loader"]["consumed"] == 0
    for name, weight in parent_state.items():
        torch.testing.assert_close(zero["state_dict"][name], weight, rtol=0, atol=0)
    expected = torch.load(tmp_path / "stage/step-3.ckpt", weights_only=True)
    for step in (0, 1):
        resumed_config = deepcopy(stage)
        resumed_config.only_load_model = False
        resumed_config.checkpoint_path = str(tmp_path / f"stage/step-{step}.ckpt")
        resumed = Trainer(resumed_config)
        if workers:
            resumed.dataloader.kwargs["multiprocessing_context"] = "spawn"
        assert resumed.curr_step == step
        try:
            resumed.train()
        finally:
            stop_workers(resumed)
        for name, weight in expected["state_dict"].items():
            torch.testing.assert_close(resumed.raw_model.state_dict()[name], weight, rtol=0, atol=0)
        assert resumed.scheduler.state_dict() == expected["scheduler_state"]
        for name, state in expected["optimizer_state"]["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(resumed.optimizer.state_dict()["state"][name][key], value, rtol=0, atol=0)
        assert train_final.sha256_file(zero_path) == zero_hash


def test_pipeline_freezes_terminal_without_any_score_selection(tmp_path, legacy_parent, monkeypatch):
    args = stage_args(tmp_path, legacy_parent)
    calls = []
    def train(command, *, cwd, log_path):
        config = finetune.training_config(command)
        calls.append(config)
        log_path.write_text(" ".join(command) + "\n")
        for step in (0, 100, 2000):
            torch.save({"curr_step": step}, Path(config.checkpoint_dir) / f"step-{step}.ckpt")
    monkeypatch.setattr(train_final.search.original.search_utils, "run_command", train)
    monkeypatch.setattr(train_final, "evaluation_record", lambda *a, **kw: pytest.fail("Unexpected model selection"))
    frozen = load_frozen_final_model(train_final.run(args))
    assert len(calls) == 1
    assert frozen.manifest["selected_checkpoint"]["step"] == 2000
    assert frozen.manifest["checkpoint_selection"]["policy"] == "predeclared_terminal_step"
    assert frozen.manifest["checkpoint_selection"]["candidates"] == []
    assert not frozen.manifest["final_test_rows_used"]
    command = train_final.training_resume_command(
        frozen.manifest["training"]["train_command"], Path(calls[0].checkpoint_dir) / "step-100.ckpt")
    resumed = finetune.training_config(command)
    assert not resumed.only_load_model and resumed.save_initial_checkpoint
    assert resumed.checkpoint_path.endswith("step-100.ckpt")
    assert command.count("--checkpoint_path") == 1


def test_evaluation_includes_zero_and_keeps_frozen_queries(tmp_path, legacy_parent, capsys, mock_models):
    directories = []
    for ratio in (1, 0):
        plan = dry_plan(stage_args(tmp_path, legacy_parent, ratio), capsys)
        directory = Path(plan["checkpoint_dir"])
        directory.mkdir(parents=True)
        (directory / "finetuning.json").write_text(json.dumps(plan))
        for step in (0, 100, 2000):
            (directory / f"step-{step}.ckpt").write_bytes(b"mock")
        directories.append(directory)
    args = evaluate_finetuning.parse_args([
        "--finetune-run", str(directories[0]), "--control-run", str(directories[1]),
        "--output-dir", str(tmp_path / "evaluation"), "--device", "cpu",
    ])
    output = evaluate_finetuning.run(args)
    payload = json.loads((output / "results.json").read_text())
    assert payload["primary_comparison_step"] == 2000 and payload["primary_comparison_complete"]
    for label in ("epit_finetuned", "generic_continuation"):
        assert {r["checkpoint_step"] for r in payload["rows"] if r["model"] == label} == {0, 100, 2000}
    frozen = frozen_membership(args.split_manifest)
    for _, context, queries in mock_models:
        assert context == frozen["context_task_row_indices"]
        assert queries == frozen["query_task_row_indices"]
    bad = json.loads((directories[1] / "finetuning.json").read_text())
    bad["train_command"][bad["train_command"].index("--np_seed") + 1] = "44"
    (directories[1] / "finetuning.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="beyond the prior share"):
        evaluate_finetuning.validate_stages(evaluate_finetuning.comparison_args(args))
