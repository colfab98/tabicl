"""Mixture runs consume common ordered streams, including across workers/resume."""

import copy
import random

import numpy as np
import pytest
import torch

from tabicl.prior import paired_streams as streams
from tabicl.prior.dataset import EPIT_TARGET_RULE_COEFFICIENTS, SOCCOL_TARGET_RULE_COEFFICIENTS
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_FEATURE_PROFILE, EPIT_EXPANDED_BLOCK_ALLOCATION
from tabicl.prior.soccol_reviewed_schema import SOCCOL_REVIEWED_FEATURE_PROFILE, SOCCOL_REVIEWED_BLOCK_ALLOCATION
from tabicl.train.sampling import CheckpointablePriorLoader


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_prior(family="epit", ratio=.5, **kwargs):
    soccol = family == "soccol"
    coefficients = (SOCCOL_TARGET_RULE_COEFFICIENTS if soccol else EPIT_TARGET_RULE_COEFFICIENTS)["pren_n_linear"]
    blocks = SOCCOL_REVIEWED_BLOCK_ALLOCATION if soccol else EPIT_EXPANDED_BLOCK_ALLOCATION
    hp = {**DEFAULT_FIXED_HP,
          "pitting_composition_mode": "empirical_features_scm_target",
          "pitting_feature_profile": SOCCOL_REVIEWED_FEATURE_PROFILE if soccol else EPIT_EXPANDED_FEATURE_PROFILE,
          "pitting_fixed_epit_schema": True, "informed_target_family": "pitting_potential",
          "informed_physical_marginal_profile": "pitting_potential_v1", "informed_physical_marginal_prob": 1.,
          "informed_task_family_probs": (1., 0.), "informed_normal_block_allocation": blocks,
          "informed_normal_block_allocation_min_counts": blocks, "informed_target_mix_weight": .4,
          "informed_mix_probs": (1., 0.), "informed_feature_block_strength": 0.,
          "pitting_target_rule_scores": {"pren_n_linear": 1.},
          "pitting_target_rule_coefficients": [f"pren_n_linear.{k}={v}" for k, v in coefficients.items()],
          "pitting_coefficient_variation": .8,
          "pitting_coefficient_upper_bounds": {"pren_n_linear": {key: 1. for key in coefficients}},
          "cat_prob": 0., "permute_features": False, "pitting_process_category_count": 50,
          "num_layers": 2, "hidden_dim": 12, "noise_std": .01,
          **kwargs.pop("hp", {})}
    width = 58 if soccol else 70
    return streams.PairedPriorDataset(batch_size=8, batch_size_per_gp=2,
        min_features=width, max_features=width, max_seq_len=32,
        min_train_size=.5, max_train_size=.8, scm_fixed_hp=hp, scm_sampled_hp={},
        informed_prior_ratio=ratio, return_source_ids=True, **kwargs)


def fake_generate(self, params):
    shape = params["seq_len"], params["num_features"]
    # Consume all three RNGs and different draw counts for each generator.
    random.random()
    np.random.random(1 if params["informed_mode"] else 7)
    X = torch.randn(shape)
    if params["informed_mode"]:
        X[0, -1] = float("nan")
    return X, torch.randn(shape[0]), torch.tensor(shape[1])


def assert_batch_equal(left, right):
    for a, b in zip(left, right):
        if a.is_nested:
            for x, y in zip(a.unbind(), b.unbind()):
                torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_mixtures_share_every_source_table_not_only_the_initial_seed(monkeypatch, family):
    monkeypatch.setattr(streams._PairedSCMPrior, "generate_dataset", fake_generate)
    reference = {}
    for ratio in [0., .25, .5, .75, 1.]:
        prior = make_prior(family, ratio)
        seen = {0: [], 1: []}
        for _ in range(4):
            batch = prior.get_batch()
            assert sum(source == 1 for source, _ in prior.last_batch_ids) == round(8 * ratio)
            for i, key in enumerate(prior.last_batch_ids):
                values = (batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i])
                if key in reference:
                    assert_batch_equal(values, reference[key])
                else:
                    reference[key] = values
                seen[key[0]].append(key[1])
        for ids in seen.values():
            assert sorted(ids) == list(range(len(ids)))


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_real_informed_tables_match_across_ratios_and_repeat(family):
    mixed = make_prior(family, .5)
    pure = make_prior(family, 1.)
    # The same source group has the same table values, targets and NaN mask.
    for left, right in zip(mixed._group(1, 0), pure._group(1, 0)):
        assert_batch_equal(left[:3], right[:3])
    state = pure.sampling_state_dict()
    assert state["next_batch"] is None


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_optuna_mlp_and_formula_changes_preserve_physical_draws(family):
    a = make_prior(family, 1., hp={"informed_mix_probs": (1., 0.), "informed_target_mix_weight": .1})
    b = make_prior(family, 1., hp={"informed_mix_probs": (0., 1.), "informed_target_mix_weight": .9})
    aa, bb = a._group(1, 0), b._group(1, 0)
    for x, y in zip(aa, bb):
        torch.testing.assert_close(x[0], y[0], rtol=0, atol=0, equal_nan=True)
        assert not torch.equal(x[1], y[1])
    for key in ["target_rule_family", "target_rule_coefficients", "coefficient_variation"]:
        assert a.priors[1].last_pitting_target_rule[key] == b.priors[1].last_pitting_target_rule[key]


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_zero_perturbation_does_not_change_environment_draws(family):
    a = make_prior(family, 1., hp={"pitting_composition_perturb_strength": 0.})
    b = make_prior(family, 1., hp={"pitting_composition_perturb_strength": .2})
    a._group(1, 0)
    b._group(1, 0)
    x, y = a.priors[1].last_pitting_feature_batch, b.priors[1].last_pitting_feature_batch
    np.testing.assert_array_equal(x.features[:, 24:], y.features[:, 24:])
    np.testing.assert_array_equal(x.compositions.template_indices, y.compositions.template_indices)


def test_real_generic_recipe_and_tables_are_independent_of_specialist_settings():
    a = make_prior("epit", 0., prior_type="mix_scm")
    b = make_prior("soccol", .5, hp={"informed_mix_probs": (0., 1.), "informed_target_mix_weight": .9})
    for prior in [a.priors[0], b.priors[0]]:
        assert (prior.min_features, prior.max_features) == (2, 100)
        assert prior.fixed_hp["cat_prob"] == .2
        assert prior.fixed_hp["permute_features"] is True
        assert prior._get_prior_mix_probs().tolist() == [.7, .3]
    for left, right in zip(a._group(0, 0), b._group(0, 0)):
        assert_batch_equal(left[:3], right[:3])
        assert torch.isfinite(left[0]).all()
        assert torch.isfinite(right[0]).all()


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_native_informed_missingness_survives_without_shared_schema(family):
    prior = make_prior(family)
    for X, y, width, *_ in prior._group(1, 0):
        assert X.shape[1] == int(width) == (70 if family == "epit" else 58)
        assert torch.isnan(X).any()
        assert torch.isfinite(X[:, :24]).all()
        assert torch.isfinite(y).all()


@pytest.mark.parametrize("family,source", [("epit", 0), ("epit", 1), ("soccol", 1)])
def test_invalid_tables_stop_after_bounded_retries(monkeypatch, family, source):
    prior = make_prior(family)
    attempts = []

    def reject(X, y, train_size):
        attempts.append(y.clone())
        return False

    monkeypatch.setattr(streams, "MAX_GENERATION_ATTEMPTS", 3)
    monkeypatch.setattr(prior.priors[source], "regression_sanity_check", reject)
    with pytest.raises(RuntimeError, match=r"failed after 3 attempts:.*task_seed=\d+"):
        prior._group(source, 0)
    assert len(attempts) == 3


def test_epit_step_460_bad_mlp_retries_and_stays_identical_across_ratios(monkeypatch, tmp_path):
    """Reproduce the exact table that stalled v11 trial 4 for over 11 hours."""
    from scripts.epit_pipeline import run_optuna as search
    from tabicl.train.run import Trainer
    from tabicl.train.train_config import build_parser

    args = search.parse_args([])
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest,
    )
    params = search.trial_params_from_mapping({
        "mlp_prob": 0.4669845975830962,
        "informed_target_mix_weight": 0.9248433885665498,
        "pitting_composition_perturb_strength": 0.2670749377030415,
    }, composition_mode=args.pitting_composition_mode, fixed_informed_prior_ratio=1.)
    command = search.training_command(args, params, tmp_path, rules)
    entry = next(i for i, value in enumerate(command) if value.endswith("/train/run.py"))
    config = build_parser().parse_args(command[entry + 1:])
    config.device = "cpu"
    config.dataloader_num_workers = 0
    original = streams._PairedSCMPrior._generate_scm_target_from_physical_features
    attempts = []

    def observe(self, X, params, cls):
        target = original(self, X, params, cls)
        if params["paired_task_seed"] == 11044767929213912588:
            attempts.append((params["generation_attempt"], bool(torch.isfinite(target).all())))
        return target

    monkeypatch.setattr(streams._PairedSCMPrior, "_generate_scm_target_from_physical_features", observe)
    # A broken fix fails quickly instead of hanging this regression test.
    monkeypatch.setattr(streams, "MAX_GENERATION_ATTEMPTS", 5)
    tables = []
    for ratio in (1., .5):
        trainer = Trainer.__new__(Trainer)
        trainer.config = copy.deepcopy(config)
        trainer.config.informed_prior_ratio = ratio
        trainer.missingness_augmenter = None
        trainer.ddp_rank, trainer.ddp_world_size, trainer.master_process = 0, 1, False
        trainer.configure_prior()
        # Source group 58799 contains the bad table in source batch 459.
        tables.append(trainer.dataloader.dataset._group(1, 58799))
    assert [attempt for attempt, _ in attempts] == [0, 1, 0, 1]
    assert attempts[0][1] is False  # Original first draw really is invalid.
    for left, right in zip(*tables):
        assert_batch_equal(left[:3], right[:3])
        assert torch.isfinite(left[1]).all() and left[1].std() > 0


def loader(prior, workers):
    return CheckpointablePriorLoader(dataset=prior, batch_size=None, num_workers=workers)


@pytest.mark.parametrize("workers", [0, 2])
def test_worker_independence_and_resume_after_prefetch(monkeypatch, workers):
    monkeypatch.setattr(streams._PairedSCMPrior, "generate_dataset", fake_generate)
    expected = loader(make_prior(), 0)
    original = loader(make_prior(), workers)
    for _ in range(3):
        assert_batch_equal(next(expected), next(original))
    state = original.state_dict()
    restored = loader(make_prior(), workers)
    restored.load_state_dict(state)
    for _ in range(3):
        assert_batch_equal(next(original), next(restored))
    changed = loader(make_prior(seed=43), workers)
    with pytest.raises(ValueError, match="same worker count and dataset settings"):
        changed.load_state_dict(state)


def test_different_gpu_partition_preserves_source_ids(monkeypatch):
    monkeypatch.setattr(streams._PairedSCMPrior, "generate_dataset", fake_generate)
    ranks = [make_prior(rank=rank, world_size=2) for rank in range(2)]
    assert ranks[0].checkpoint_identity() == ranks[1].checkpoint_identity()
    assert ranks[0].sampling_config() != ranks[1].sampling_config()
    for prior in ranks:
        prior.get_batch()
    for source in [0, 1]:
        ids = [i for prior in ranks for s, i in prior.last_batch_ids if s == source]
        assert sorted(ids) == list(range(8))


def test_nonrepresentable_ratio_rejected():
    with pytest.raises(ValueError, match="integer number"):
        make_prior(ratio=.33)


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_real_tables_independent_of_spawned_worker_assignment(family):
    direct = loader(make_prior(family), 0)
    parallel = loader(make_prior(family), 2)
    parallel.kwargs["multiprocessing_context"] = "spawn"
    for _ in range(2):
        assert_batch_equal(next(direct), next(parallel))


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_optuna_and_final_emit_identical_stream_and_generator_settings(tmp_path, family):
    from scripts.epit_pipeline import run_optuna as search, train_final as final
    from scripts.soccol_pipeline import run_optuna as soccol_search
    from tabicl.train.train_config import build_parser

    originals = vars(search).copy()
    try:
        args = (soccol_search if family == "soccol" else search).parse_args([])
        rules = search.load_target_rule_config(summary_path=args.target_rule_summary,
                                              split_manifest_path=args.split_manifest)
        values = {"mlp_prob": .37, "informed_target_mix_weight": .42,
                  "pitting_composition_perturb_strength": .1}
        params = search.trial_params_from_mapping(values,
                    composition_mode=args.pitting_composition_mode, fixed_informed_prior_ratio=1.)
        final_args = final.parse_args(["--storage", "journal:///unused.log"])
        final_args.pitting_composition_mode = args.pitting_composition_mode
        configs = []
        for config in [args, final_args]:
            command = search.training_command(config, params, tmp_path, rules)
            entry = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
            parsed = build_parser().parse_args(command[entry + 1:])
            assert parsed.paired_prior_streams and parsed.prior_stream_seed == parsed.python_seed == 42
            configs.append(vars(parsed))
        for key in configs[0]:
            if key not in {"max_steps", "save_temp_every", "save_perm_every"}:
                assert configs[0][key] == configs[1][key], key
        fingerprint = search.build_pipeline_fingerprint(args, rules)
        assert fingerprint["fixed_prior"]["paired_prior_streams"] == streams.stream_policy(42)
        args.prior_stream_seed = 43
        assert search.build_pipeline_fingerprint(args, rules) != fingerprint
    finally:
        for name, value in originals.items():
            if vars(search).get(name) is not value:
                setattr(search, name, value)


def small_training_config(tmp_path, family, workers=0):
    from tabicl.train.train_config import build_parser

    profile = make_prior(family, .5)
    width = profile.priors[1].max_features
    config = build_parser().parse_args([
        "--device", "cpu", "--prior_device", "cpu", "--amp", "false", "--wandb_log", "false",
        "--python_seed", "42", "--prior_n_jobs", "1", "--dataloader_num_workers", str(workers),
        "--paired_prior_streams", "true", "--prior_type", "hybrid_scm",
        "--batch_size", "4", "--batch_size_per_gp", "2", "--micro_batch_size", "2", "--max_seq_len", "16",
        "--min_train_size", "0.4", "--max_train_size", "0.6",
        "--min_features", str(width), "--max_features", str(width), "--max_classes", "0",
        "--num_quantiles", "9", "--embed_dim", "16", "--ff_factor", "2",
        "--col_num_blocks", "1", "--col_nhead", "2", "--col_num_inds", "4",
        "--row_num_blocks", "1", "--row_nhead", "2", "--row_num_cls", "2",
        "--icl_num_blocks", "1", "--icl_nhead", "2", "--col_missing_value_embedding", "true",
        "--warmup_proportion", "0", "--scheduler_total_steps", "4", "--max_steps", "4",
        "--checkpoint_dir", str(tmp_path),
    ])
    for key, value in profile.priors[1].fixed_hp.items():
        if hasattr(config, key) and key not in {"paired_prior_streams"}:
            setattr(config, key, copy.deepcopy(value))
    config.pitting_target_rule_scores = ["pren_n_linear=1.0"]
    return config


@pytest.mark.parametrize("family,workers", [("epit", 0), ("epit", 2), ("soccol", 0), ("soccol", 2)])
def test_real_training_resume_reproduces_batches_weights_and_optimizer(tmp_path, family, workers):
    from tabicl.train.run import Trainer

    config = small_training_config(tmp_path, family, workers)
    uninterrupted = Trainer(config)
    if workers:
        uninterrupted.dataloader.kwargs["multiprocessing_context"] = "spawn"
    for step in range(2):
        uninterrupted.run_batch(next(uninterrupted.dataloader))
        uninterrupted.curr_step = step + 1
    uninterrupted.save_checkpoint("step-2.ckpt")
    expected_batches = []
    for _ in range(2):
        batch = next(uninterrupted.dataloader)
        expected_batches.append(batch)
        uninterrupted.run_batch(batch)
    resumed_config = copy.deepcopy(config)
    resumed_config.max_steps = 6  # Longer stopping point does not enter dataset seeds.
    resumed = Trainer(resumed_config)
    if workers:
        resumed.dataloader.kwargs["multiprocessing_context"] = "spawn"
    assert resumed.curr_step == 2
    for expected in expected_batches:
        batch = next(resumed.dataloader)
        assert_batch_equal(batch, expected)
        resumed.run_batch(batch)
    for key, value in uninterrupted.raw_model.state_dict().items():
        torch.testing.assert_close(value, resumed.raw_model.state_dict()[key], rtol=0, atol=0)
    assert uninterrupted.scheduler.state_dict() == resumed.scheduler.state_dict()
    left, right = uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    for parameter, state in left["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right["state"][parameter][key], rtol=0, atol=0)
    resumed_config.prior_stream_seed = 43
    with pytest.raises(ValueError, match="matching recipe/profile/seed"):
        Trainer(resumed_config)


def configured_prior(config, rank=0, world_size=1):
    """Exercise the trainer's seed wiring, not just the dataset constructor."""
    from tabicl.train.run import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.config = copy.deepcopy(config)
    trainer.config.batch_size //= world_size
    trainer.missingness_augmenter = None
    trainer.ddp_rank, trainer.ddp_world_size, trainer.master_process = rank, world_size, False
    trainer.configure_prior()
    return trainer.dataloader.dataset


@pytest.mark.parametrize("family", ["epit", "soccol"])
@pytest.mark.parametrize("ratio", [0., .25, .5, .75, 1.])
def test_trainer_partitions_identical_real_tables_across_processes(tmp_path, family, ratio):
    config = small_training_config(tmp_path, family)
    config.batch_size, config.informed_prior_ratio = 16, ratio
    single = configured_prior(config)
    ranks = [configured_prior(config, rank, 2) for rank in range(2)]
    assert ranks[0].checkpoint_identity() == ranks[1].checkpoint_identity()
    for _ in range(2):
        batch = single.get_batch()
        expected = {key: tuple(t[i] for t in batch) for i, key in enumerate(single.last_batch_ids)}
        seen = set()
        for rank in ranks:
            actual = rank.get_batch()
            for i, key in enumerate(rank.last_batch_ids):
                assert key not in seen
                seen.add(key)
                assert_batch_equal(tuple(t[i] for t in actual), expected[key])
        assert seen == set(expected)


def test_nonpaired_coefficient_seed_keeps_legacy_rank_offset(tmp_path):
    config = small_training_config(tmp_path, "epit")
    config.paired_prior_streams = False
    prior = configured_prior(config, 1, 2)
    assert prior.prior.fixed_hp["pitting_coefficient_variation_seed"] == 43


def test_paired_training_never_skips_an_oom_table(tmp_path, monkeypatch):
    from tabicl.train.run import Trainer

    trainer = Trainer(small_training_config(tmp_path, "epit"))
    batch = next(trainer.dataloader)
    calls = []

    def out_of_memory(*args):
        calls.append(args[1])
        raise torch.cuda.OutOfMemoryError("injected OOM")

    monkeypatch.setattr(trainer, "run_micro_batch", out_of_memory)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected OOM"):
        trainer.run_batch(batch)
    assert calls == [0]
    assert not trainer.optimizer.state


def _distributed_training_worker(rank, root, config):
    """Use production setup/updates/checkpoints, replacing only CUDA transport."""
    import os
    from datetime import timedelta
    from pathlib import Path
    from unittest.mock import patch

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from tabicl.train import run

    torch.set_num_threads(1)
    root = Path(root)
    reference = torch.load(root / "single.pt", weights_only=False)

    class CPUTrainer(run.Trainer):
        def configure_ddp(self):
            super().configure_ddp()
            self.config.device = "cpu"

    def initialize(backend):
        assert backend == "nccl"
        if not dist.is_initialized():
            dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                                    rank=rank, world_size=2, timeout=timedelta(seconds=60))

    with patch.dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2"), \
         patch.object(run, "init_process_group", initialize), \
         patch.object(torch.cuda, "set_device"), \
         patch.object(run, "DDP", lambda model, **kw: DistributedDataParallel(
             model, broadcast_buffers=kw["broadcast_buffers"])):
        try:
            trainer = CPUTrainer(copy.deepcopy(config))
            for step in range(3):
                batch = next(trainer.dataloader)
                for i, key in enumerate(trainer.dataloader.dataset.last_batch_ids):
                    assert_batch_equal(tuple(t[i] for t in batch), reference["tables"][step][key])
                trainer.run_batch(batch)
                trainer.curr_step = step + 1
                if step == 1:
                    states = [None, None]
                    dist.all_gather_object(states, trainer.sampling_state_dict())
                    if rank == 0:
                        trainer.save_checkpoint("step-2.ckpt", sampling_states=states)
                    dist.barrier()
            for name, parameter in trainer.raw_model.state_dict().items():
                torch.testing.assert_close(parameter, reference["model"][name], rtol=1e-5, atol=1e-6)
            assert trainer.scheduler.state_dict() == reference["scheduler"]
            for key, state in trainer.optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, reference["optimizer"]["state"][key][name],
                                               rtol=1e-4, atol=1e-6)
            # Each rank must reload the same checkpoint, including its own stream boundary.
            restored = CPUTrainer(copy.deepcopy(config))
            assert restored.curr_step == 2
            restored.run_batch(next(restored.dataloader))
            assert_batch_equal(tuple(trainer.raw_model.state_dict().values()),
                               tuple(restored.raw_model.state_dict().values()))
            assert trainer.scheduler.state_dict() == restored.scheduler.state_dict()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_two_process_training_matches_single_and_resumes(tmp_path, family):
    from tabicl.train.run import Trainer

    config = small_training_config(tmp_path / "single", family)
    config.batch_size, config.informed_prior_ratio = 16, .5
    single = Trainer(copy.deepcopy(config))
    tables = []
    for _ in range(3):
        batch = next(single.dataloader)
        tables.append({key: tuple(t[i].clone() for t in batch)
                       for i, key in enumerate(single.dataloader.dataset.last_batch_ids)})
        single.run_batch(batch)
    torch.save({"tables": tables, "model": single.raw_model.state_dict(),
                "optimizer": single.optimizer.state_dict(), "scheduler": single.scheduler.state_dict()},
               tmp_path / "single.pt")
    config.checkpoint_dir = str(tmp_path / "distributed")
    torch.multiprocessing.spawn(_distributed_training_worker, args=(str(tmp_path), config),
                                nprocs=2, join=True)
