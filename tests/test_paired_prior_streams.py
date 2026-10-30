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


@pytest.mark.parametrize("family,workers", [("epit", 0), ("epit", 2), ("soccol", 0), ("soccol", 2)])
def test_real_training_resume_reproduces_batches_weights_and_optimizer(tmp_path, family, workers):
    from tabicl.train.run import Trainer
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
