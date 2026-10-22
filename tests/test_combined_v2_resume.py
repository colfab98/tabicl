from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.combined_corrosion.validate_v2_run import validate_run
from scripts.combined_corrosion.continue_epit_v2 import prepare_continuation
from scripts.combined_corrosion.train_specialist_shared_v2 import build_specialist_config
from tabicl.train.run import Trainer
from tabicl.train.sampling import load_rng_state_dict, rng_state_dict
from tabicl.train.train_config import build_parser


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
WRAPPER = ROOT / "scripts/combined_corrosion/train_final_v2.sbatch"


@pytest.fixture(scope="module")
def saved_training(tmp_path_factory):
    torch.set_num_threads(2)
    directory = tmp_path_factory.mktemp("v2_resume")
    config = build_parser().parse_args([
        "--device", "cpu", "--prior_device", "cpu", "--amp", "false",
        "--python_seed", "42", "--prior_n_jobs", "1", "--dataloader_num_workers", "0",
        "--batch_size", "4", "--micro_batch_size", "2", "--max_seq_len", "16",
        "--min_features", "50", "--max_features", "50", "--max_classes", "0",
        "--num_quantiles", "9", "--embed_dim", "16", "--ff_factor", "2",
        "--col_num_blocks", "1", "--col_nhead", "2", "--col_num_inds", "4",
        "--row_num_blocks", "1", "--row_nhead", "2", "--row_num_cls", "2",
        "--icl_num_blocks", "1", "--icl_nhead", "2",
        "--warmup_proportion", "0.02", "--scheduler_total_steps", "10000",
        "--max_steps", "10000", "--checkpoint_dir", str(directory),
        "--combined_corrosion_prior_config", str(DATA / "combined_prior_config.json"),
        "--col_missing_value_embedding", "true",
        "--missingness_profile_path", str(DATA / "context_missingness_profile.json"),
        "--missingness_query_column_probability", "0.8",
        "--missingness_preserve_observed_composition", "true",
    ])
    trainer = Trainer(config)
    trainer.run_batch(trainer.dataloader.dataset.get_batch())
    trainer.curr_step = 1
    trainer.save_checkpoint("step-1.ckpt")
    trainer.run_batch(trainer.dataloader.dataset.get_batch())
    trainer.curr_step = 2
    trainer.save_checkpoint("step-2.ckpt")
    return config, directory / "step-2.ckpt"


def test_resume_restores_actual_training_state_and_advances(saved_training):
    config, path = saved_training
    before = path.read_bytes()
    assert validate_run(config, mode="resume") == path
    resumed_config = copy.copy(config)
    resumed_config.checkpoint_path = str(path)
    resumed = Trainer(resumed_config)
    checkpoint = torch.load(path, weights_only=True)
    assert resumed.curr_step == 2
    assert resumed.scheduler.state_dict() == checkpoint["scheduler_state"]
    for name, value in resumed.raw_model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["state_dict"][name], rtol=0, atol=0)
    restored_optimizer = resumed.optimizer.state_dict()
    assert restored_optimizer["param_groups"] == checkpoint["optimizer_state"]["param_groups"]
    for index, state in restored_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, checkpoint["optimizer_state"]["state"][index][name], rtol=0, atol=0)
    resumed.run_batch(resumed.dataloader.dataset.get_batch())
    assert resumed.scheduler.last_epoch == 3
    assert all(torch.isfinite(value).all() for value in resumed.raw_model.parameters())
    assert path.read_bytes() == before


@pytest.mark.parametrize(("key", "value", "message"), [
    ("missingness_query_column_probability", 0.0, "Missingness"),
    ("missingness_preserve_observed_composition", False, "Missingness"),
    ("scheduler_total_steps", 20000, "original 10,000-step"),
    ("warmup_proportion", 0.1, "original 10,000-step"),
    ("only_load_model", True, "not only model"),
    ("max_steps", 2, "checkpoint step"),
    ("embed_dim", 32, "architecture"),
])
def test_resume_rejects_changed_settings(saved_training, key, value, message):
    config = copy.copy(saved_training[0])
    setattr(config, key, value)
    with pytest.raises(ValueError, match=message):
        validate_run(config, mode="resume")


def test_resume_rejects_changed_prior(saved_training, tmp_path):
    config = copy.copy(saved_training[0])
    prior = tmp_path / "changed_prior.json"
    prior.write_bytes(Path(config.combined_corrosion_prior_config).read_bytes() + b"\n")
    config.combined_corrosion_prior_config = str(prior)
    with pytest.raises(ValueError, match="Prior configuration"):
        validate_run(config, mode="resume")


@pytest.mark.parametrize("change", ["missing_optimizer", "wrong_step", "wrong_lr"])
def test_resume_rejects_incomplete_or_inconsistent_checkpoint(saved_training, tmp_path, change):
    config = copy.copy(saved_training[0])
    config.checkpoint_dir = str(tmp_path)
    checkpoint = torch.load(saved_training[1], weights_only=True)
    if change == "missing_optimizer":
        del checkpoint["optimizer_state"]
    elif change == "wrong_step":
        checkpoint["curr_step"] = 1
    else:
        checkpoint["optimizer_state"]["param_groups"][0]["lr"] *= 2
    torch.save(checkpoint, tmp_path / "step-2.ckpt")
    with pytest.raises(ValueError):
        validate_run(config, mode="resume")


def test_fresh_and_resume_require_opposite_directory_states(saved_training, tmp_path):
    config = copy.copy(saved_training[0])
    with pytest.raises(FileExistsError, match="Use --resume"):
        validate_run(config, mode="fresh")
    config.checkpoint_dir = str(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="refusing to start from scratch"):
        validate_run(config, mode="resume")
    assert validate_run(config, mode="fresh") is None
    assert not Path(config.checkpoint_dir).exists()


def test_explicit_missing_checkpoint_never_silently_starts_fresh(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.config = build_parser().parse_args(["--checkpoint_path", str(tmp_path / "missing.ckpt")])
    with pytest.raises(FileNotFoundError, match="Explicit checkpoint"):
        trainer.load_checkpoint()


def test_wrapper_dry_run_keeps_fresh_directory_untouched(tmp_path):
    directory = tmp_path / "new_run"
    result = subprocess.run(["bash", str(WRAPPER), "--dry-run"], cwd=ROOT,
                            env={**os.environ, "CHECKPOINT_DIR": str(directory)},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--missingness_query_column_probability 0.8" in result.stdout
    assert "--missingness_preserve_observed_composition True" in result.stdout
    assert "--scheduler_total_steps 10000" in result.stdout
    assert not directory.exists()


def test_wrapper_refuses_concurrent_writer(tmp_path):
    with (tmp_path / ".training.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(WRAPPER), "--resume"], cwd=ROOT,
                                env={**os.environ, "CHECKPOINT_DIR": str(tmp_path)},
                                capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Another training process holds the run lock" in result.stderr


def test_wrapper_rejects_unknown_option():
    result = subprocess.run(["bash", str(WRAPPER), "--resuem"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def _stop_workers(trainer):
    iterator = trainer.dataloader._iterator
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


@pytest.mark.parametrize("workers", [0, 4])
def test_sampling_restart_matches_uninterrupted_training(saved_training, tmp_path, workers):
    """Real priors, coefficient jitter, masks, dropout, optimizer and prefetch."""
    config = copy.copy(saved_training[0])
    config.checkpoint_dir = str(tmp_path)
    config.dataloader_num_workers = workers
    config.dataloader_prefetch_factor = 4
    config.dropout = 0.1
    # Tiny test tables still need >=2 targets on each side of the split.
    config.min_train_size = 0.25
    config.max_train_size = 0.75
    trainer = Trainer(config)
    if workers:
        trainer.dataloader.kwargs["multiprocessing_context"] = "spawn"
    batches = []
    try:
        for step in range(1, 9):
            batch = next(trainer.dataloader)
            batches.append([value.clone() for value in batch])
            trainer.run_batch(batch)
            trainer.curr_step = step
            if step in (1, 5):  # Partial first worker cycle, then a non-zero worker offset.
                trainer.save_checkpoint(f"step-{step}.ckpt")
        expected_model = copy.deepcopy(trainer.raw_model.state_dict())
        expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        expected_draws = (random.random(), np.random.rand(), torch.rand(4))
    finally:
        _stop_workers(trainer)

    for start in (1, 5):
        resumed_config = copy.copy(config)
        resumed_config.checkpoint_path = str(tmp_path / f"step-{start}.ckpt")
        resumed = Trainer(resumed_config)
        if workers:
            resumed.dataloader.kwargs["multiprocessing_context"] = "spawn"
        try:
            for index in range(start, 8):
                batch = next(resumed.dataloader)
                for actual, expected in zip(batch, batches[index], strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                resumed.run_batch(batch)
            for name, value in resumed.raw_model.state_dict().items():
                torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
            optimizer = resumed.optimizer.state_dict()
            assert optimizer["param_groups"] == expected_optimizer["param_groups"]
            for index, state in optimizer["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, expected_optimizer["state"][index][name], rtol=0, atol=0)
            assert random.random() == expected_draws[0]
            assert np.random.rand() == expected_draws[1]
            torch.testing.assert_close(torch.rand(4), expected_draws[2], rtol=0, atol=0)
        finally:
            _stop_workers(resumed)

    # Recording state must not change fresh-run samples or training behavior.
    legacy_config = copy.copy(config)
    legacy_config.checkpoint_dir = str(tmp_path / "legacy")
    legacy = Trainer(legacy_config)
    kwargs = dict(legacy.dataloader.kwargs)
    if workers:
        kwargs["multiprocessing_context"] = "spawn"
    iterator = iter(DataLoader(dataset=legacy.dataloader.dataset, **kwargs))
    try:
        for expected_batch in batches:
            batch = next(iterator)
            for actual, expected in zip(batch, expected_batch, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            legacy.run_batch(batch)
        for name, value in legacy.raw_model.state_dict().items():
            torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def test_old_checkpoints_still_resume(saved_training, tmp_path):
    config = copy.copy(saved_training[0])
    config.checkpoint_path = str(tmp_path / "old.ckpt")
    checkpoint = torch.load(saved_training[1], weights_only=True)
    del checkpoint["sampling_state"]
    del checkpoint["scaler_state"]
    torch.save(checkpoint, config.checkpoint_path)
    with pytest.warns(UserWarning, match="old synthetic stream cannot be recovered"):
        resumed = Trainer(config)
    assert resumed.curr_step == 2


def test_sampling_restart_rejects_changed_worker_count(saved_training):
    config = copy.copy(saved_training[0])
    config.dataloader_num_workers = 4
    with pytest.raises(ValueError, match="same worker count"):
        Trainer(config)


def test_loading_only_weights_does_not_restore_sampling(saved_training):
    config = copy.copy(saved_training[0])
    config.only_load_model = True
    config.dataloader_num_workers = 4
    resumed = Trainer(config)
    assert resumed.curr_step == 0
    assert resumed.dataloader.state_dict()["consumed"] == 0


def test_rng_restore_accepts_pin_memory_sequence_conversion():
    state = rng_state_dict()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    state["python"] = [state["python"][0], list(state["python"][1]), state["python"][2]]
    load_rng_state_dict(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_preflight_rejects_changed_sampling_workers(saved_training):
    config = copy.copy(saved_training[0])
    config.dataloader_num_workers = 4
    with pytest.raises(ValueError, match="same worker count"):
        validate_run(config, mode="resume")


@pytest.fixture
def continuation_parent(saved_training, tmp_path):
    config = copy.copy(saved_training[0])
    parent = tmp_path / "parent"
    parent.mkdir()
    prior = parent / "prior_config.json"
    prior.write_text(json.dumps(build_specialist_config(DATA / "combined_prior_config.json", "EPIT")))
    config.checkpoint_dir = str(parent)
    config.combined_corrosion_prior_config = str(prior)
    config.min_train_size, config.max_train_size = 0.25, 0.75
    trainer = Trainer(config)
    trainer.run_batch(trainer.dataloader.dataset.get_batch())
    trainer.curr_step = 1
    trainer.save_checkpoint("step-1.ckpt")
    checkpoint = torch.load(parent / "step-1.ckpt", weights_only=True)
    del checkpoint["sampling_state"]  # Match the historical EPIT checkpoint format.
    torch.save(checkpoint, parent / "step-1.ckpt")
    trainer.run_batch(trainer.dataloader.dataset.get_batch())
    trainer.curr_step = 2
    trainer.save_checkpoint("step-2.ckpt")  # Ensure the explicit parent wins over latest.
    target = tmp_path / "soccol_prior.json"
    target.write_text(json.dumps(build_specialist_config(DATA / "combined_prior_config.json", "Soccol")))
    config.checkpoint_dir = str(tmp_path / "branch")
    return config, parent / "step-1.ckpt", target


@pytest.mark.parametrize("arm", ["control", "soccol"])
def test_continuation_preserves_parent_training_state_and_can_resume(continuation_parent, arm):
    config, parent, soccol_prior = continuation_parent
    before = parent.read_bytes()
    if arm == "soccol":
        config.combined_corrosion_prior_config = str(soccol_prior)
    path = prepare_continuation(config, arm=arm, mode="fresh", parent_checkpoint=parent, dry_run=True)
    assert path == parent
    assert not Path(config.checkpoint_dir).exists()
    assert prepare_continuation(config, arm=arm, mode="fresh", parent_checkpoint=parent) == parent
    manifest = json.loads((Path(config.checkpoint_dir) / "continuation.json").read_text())
    assert manifest["parent_checkpoint"] == str(parent)
    assert manifest["arm"] == arm
    config.checkpoint_path = str(parent)
    with pytest.warns(UserWarning, match="no sampling state"):
        resumed = Trainer(config)
    checkpoint = torch.load(parent, weights_only=True)
    assert resumed.curr_step == 1
    assert resumed.scheduler.state_dict() == checkpoint["scheduler_state"]
    optimizer = resumed.optimizer.state_dict()
    assert optimizer["param_groups"] == checkpoint["optimizer_state"]["param_groups"]
    for index, state in optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, checkpoint["optimizer_state"]["state"][index][name], rtol=0, atol=0)
    for name, value in resumed.raw_model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["state_dict"][name], rtol=0, atol=0)
    expected_counts = {"control": {"generic": 2, "EPIT": 2, "Soccol": 0},
                       "soccol": {"generic": 0, "EPIT": 0, "Soccol": 4}}
    assert resumed.dataloader.dataset._batch_counts(4) == expected_counts[arm]
    resumed.run_batch(next(resumed.dataloader))
    resumed.curr_step = 2
    resumed.save_checkpoint("step-2.ckpt")
    assert prepare_continuation(config, arm=arm, mode="resume", parent_checkpoint=parent).name == "step-2.ckpt"
    with pytest.raises(FileExistsError, match="Use --resume"):
        prepare_continuation(config, arm=arm, mode="fresh", parent_checkpoint=parent)
    assert parent.read_bytes() == before


def test_continuation_rejects_generator_change_and_missing_provenance(continuation_parent):
    config, parent, target = continuation_parent
    config.combined_corrosion_prior_config = str(target)
    with pytest.raises(FileNotFoundError, match="continuation.json"):
        prepare_continuation(config, arm="soccol", mode="resume", parent_checkpoint=parent)
    prior = json.loads(target.read_text())
    prior["regimes"]["Soccol"]["fixed_hp"]["pitting_coefficient_variation"] = 0.4
    target.write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="only the specified prior weights"):
        prepare_continuation(config, arm="soccol", mode="fresh", parent_checkpoint=parent)


def test_continuation_refuses_original_directory(continuation_parent):
    config, parent, _ = continuation_parent
    config.checkpoint_dir = str(parent.parent)
    with pytest.raises(ValueError, match="separate output directory"):
        prepare_continuation(config, arm="control", mode="fresh", parent_checkpoint=parent)


def test_continuation_requires_compatible_sampling_parent(continuation_parent):
    config, parent, _ = continuation_parent
    with pytest.raises(ValueError, match="without sampling state"):
        prepare_continuation(config, arm="control", mode="fresh", parent_checkpoint=parent.with_name("step-2.ckpt"))
