"""Single-rule table ablation: physical parity, generic identity and resume."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl.prior.epit_schema import EPIT_COMPOSITION_ELEMENTS
from tabicl.prior.selected_pairs_single_rule import SingleRuleSelectedPairSampler, install_single_rule_sampler
from tests.test_paired_prior_streams import make_prior, assert_batch_equal, small_training_config

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "epit": ROOT / "corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k",
    "soccol": ROOT / "corrosion_datasets/analysis/soccol_pipeline/selected_pairs_allaccepted_generic10k",
}


def module(family):
    if family == "epit":
        from tabicl.prior import epit_selected_pairs
        return epit_selected_pairs
    from tabicl.prior import soccol_selected_pairs
    return soccol_selected_pairs


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=["epit", "soccol"])
def samplers(request):
    base = module(request.param).SelectedPairSampler(RECIPES[request.param])
    return request.param, base, SingleRuleSelectedPairSampler(base)


def test_each_rule_uses_multiple_retained_parents_and_existing_targets(samplers):
    family, base, sampler = samplers
    original = base.pairs.copy(deep=True)
    # Find a deterministic draw selecting each rule, including small pools.
    seeds = {}
    for seed in range(10000):
        formula = base.pairs.iloc[np.random.default_rng(seed).integers(len(base.pairs))].formula
        seeds.setdefault(formula, seed)
        if len(seeds) == len(sampler.formula_positions):
            break
    assert set(seeds) == set(base.pairs.formula)
    for formula, seed in seeds.items():
        batch = sampler.sample(1024, perturb_strength=0., random_state=seed)
        expected = base.pairs.iloc[batch.pair_indices]
        assert set(batch.formulas) == {formula}
        assert len(set(batch.parent_indices)) > 1
        np.testing.assert_array_equal(batch.parent_indices, expected.task_row_index)
        np.testing.assert_allclose(batch.target_mV, expected.prediction_mV, rtol=1e-10, atol=1e-8)
    pd.testing.assert_frame_equal(base.pairs, original)
    assert sampler.ratio == base.ratio


def test_perturbations_and_replay_preserve_physical_schema(samplers):
    family, base, sampler = samplers
    a = sampler.sample(1024, perturb_strength=.3, random_state=42)
    b = sampler.sample(1024, perturb_strength=.3, random_state=42)
    zero = sampler.sample(1024, perturb_strength=0., random_state=42)
    np.testing.assert_array_equal(a.parent_indices, zero.parent_indices)
    np.testing.assert_array_equal(a.target_mV, b.target_mV)
    np.testing.assert_array_equal(a.features, b.features)
    assert a.features.shape == (1024, 70 if family == "epit" else 58)
    assert np.isfinite(a.target_mV).all()
    columns = list(EPIT_COMPOSITION_ELEMENTS)
    physical = a.physical[columns].to_numpy()
    original = zero.physical[columns].to_numpy()
    np.testing.assert_allclose(physical.sum(axis=1), 100.)
    assert (physical[original == 0] == 0).all()
    assert not np.allclose(physical, original)
    pd.testing.assert_frame_equal(a.physical.drop(columns=columns), zero.physical.drop(columns=columns))


class TinySampler:
    """Unequal rule pools expose accidental uniform-over-formulas sampling."""
    def __init__(self):
        self.pairs = pd.DataFrame({"task_row_index": range(10), "formula": ["a"] + ["b"] * 3 + ["c"] * 6})
        self.path, self.ratio, self.identity = Path("test-only"), .3, {"informed_ratio": .3}

    def sample(self, n, *, perturb_strength, random_state):
        indices = random_state.integers(len(self.pairs), size=n)
        selected = self.pairs.iloc[indices]
        return SimpleNamespace(pair_indices=indices, parent_indices=selected.task_row_index.to_numpy(),
                               formulas=selected.formula.to_numpy())


def test_pair_marginal_weights_are_preserved_with_unequal_rule_pools():
    sampler = SingleRuleSelectedPairSampler(TinySampler())
    rng = np.random.default_rng(42)
    counts = np.zeros(10)
    for _ in range(5000):
        batch = sampler.sample(8, perturb_strength=.3, random_state=rng)
        assert len(set(batch.formulas)) == 1
        counts += np.bincount(batch.pair_indices, minlength=10)
    np.testing.assert_allclose(counts / counts.sum(), .1, atol=.015, rtol=0)


def make_selected(family):
    return make_prior(family, dataset_cls=module(family).SelectedPairPriorMixture,
                      selected_pair_dir=RECIPES[family],
                      hp={"pitting_coefficient_variation": 0., "pitting_composition_perturb_strength": .3})


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_actual_mixture_keeps_generic_tables_and_allocation_identical(family):
    old = make_selected(family)
    new = install_single_rule_sampler(make_selected(family))
    assert old.checkpoint_identity() != new.checkpoint_identity()
    for _ in range(4):
        before, after = old.get_batch(), new.get_batch()
        assert old.last_batch_ids == new.last_batch_ids
        for i, (source, _) in enumerate(new.last_batch_ids):
            if source == 0:
                assert_batch_equal(tuple(t[i] for t in before), tuple(t[i] for t in after))
        assert torch.isfinite(after[1]).all()
        assert all(torch.isfinite(x).all() for x in after[0].unbind())
    state = new.sampling_state_dict()
    expected = new.get_batch()
    restored = install_single_rule_sampler(make_selected(family))
    restored.load_sampling_state_dict(state)
    assert_batch_equal(expected, restored.get_batch())


def config_for(tmp_path, family, workers):
    config = small_training_config(tmp_path, family, workers)
    config.original_prior_mix = True
    config.col_missing_value_embedding = False
    setattr(config, f"{family}_selected_pair_dir", str(RECIPES[family]))
    config.pitting_coefficient_variation = 0.
    config.pitting_composition_perturb_strength = .3
    config.informed_target_mix_policy = "fixed"
    config.informed_target_mix_weight = 1.
    config.permute_features = True
    return config


@pytest.mark.parametrize("family", ["epit", "soccol"])
@pytest.mark.parametrize("workers", [0, 2])
def test_checkpoint_resume_reproduces_training_and_rejects_old_policy(tmp_path, family, workers):
    from tabicl.train.run import Trainer
    from tabicl.train.run_selected_pairs_single_rule import SingleRuleTrainer
    config = config_for(tmp_path, family, workers)
    first = SingleRuleTrainer(config)
    if workers:
        first.dataloader.kwargs["multiprocessing_context"] = "spawn"
    first.run_batch(next(first.dataloader))
    first.curr_step = 1
    first.save_checkpoint("step-1.ckpt")
    expected = next(first.dataloader)
    first.run_batch(expected)
    restored = SingleRuleTrainer(deepcopy(config))
    if workers:
        restored.dataloader.kwargs["multiprocessing_context"] = "spawn"
    assert restored.curr_step == 1
    batch = next(restored.dataloader)
    assert_batch_equal(expected, batch)
    restored.run_batch(batch)
    for key, value in first.raw_model.state_dict().items():
        torch.testing.assert_close(value, restored.raw_model.state_dict()[key], rtol=0, atol=0)
    assert first.scheduler.state_dict() == restored.scheduler.state_dict()
    for key, state in first.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, restored.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
    # Old and new runs cannot accidentally resume each other's sampling policy.
    with pytest.raises(ValueError, match="matching recipe/profile/seed"):
        Trainer(deepcopy(config))
    old_config = config_for(tmp_path / "old", family, 0)
    old = Trainer(old_config)
    old.save_checkpoint("step-0.ckpt")
    with pytest.raises(ValueError, match="matching recipe/profile/seed"):
        SingleRuleTrainer(deepcopy(old_config))


@pytest.mark.parametrize("family", ["epit", "soccol"])
@pytest.mark.parametrize("processes", [1, 2])
def test_launcher_changes_only_entry_point_and_records_new_recipe(family, processes):
    if family == "epit":
        from scripts.epit_pipeline import train_selected_pairs as old, train_selected_pairs_single_rule as new
    else:
        from scripts.soccol_pipeline import train_selected_pairs as old, train_selected_pairs_single_rule as new
    args = Namespace(recipe_dir=RECIPES[family], run_name="single_rule_plan_test_only", device="cuda",
                     max_steps=500, scheduler_total_steps=10000, nproc_per_node=processes, workers=4)
    before = old.build_plan(args)
    after = new.build_plan(args)
    command = list(after["command"])
    command[command.index(str(ROOT / "src/tabicl/train/run_selected_pairs_single_rule.py"))] = str(ROOT / "src/tabicl/train/run.py")
    assert command == before["command"]
    assert after["informed_proportion"] == before["informed_proportion"]
    assert after["recipe_identity"]["artifacts"] == before["recipe_identity"]["artifacts"]
    assert after["table_sampling_policy"]["rule_scope"] == "one rule for all context and query rows in each table"
    assert after["max_steps"] == 500 and after["scheduler_total_steps"] == 10000
    assert new.DEFAULT_NAME != old.DEFAULT_NAME
    assert not after["automatic_evaluation"]


def test_launcher_refuses_existing_run(tmp_path, monkeypatch):
    from scripts.epit_pipeline import train_selected_pairs as old, train_selected_pairs_single_rule as new
    monkeypatch.setattr(old, "ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoints/epit_selected_pairs/existing"
    checkpoint.mkdir(parents=True)
    (checkpoint / "step-1.ckpt").write_bytes(b"existing run")
    with pytest.raises(FileExistsError, match="Run already exists"):
        new.build_plan(Namespace(recipe_dir=RECIPES["epit"], run_name="existing"))
