"""Opt-in table-level rule selection, shared by the EPIT and Soccol recipes."""
from copy import copy, deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np


class SingleRuleSelectedPairSampler:
    """Keep every retained pair; group rows by formula only within a table.

    A uniformly drawn retained pair selects the table's formula. All table
    rows are then sampled uniformly, with replacement, from that formula's
    retained pairs. Thus every original pair has the same marginal weight.
    Physical perturbation and formula evaluation use the original sampler.
    """

    def __init__(self, sampler):
        self.base_sampler = sampler
        self.pairs = sampler.pairs
        self.path = sampler.path
        self.ratio = sampler.ratio
        self.formula_positions = {
            name: np.flatnonzero(self.pairs.formula.to_numpy() == name)
            for name in sorted(self.pairs.formula.unique())
        }
        if not self.formula_positions:
            raise ValueError("Single-rule tables require retained sample/formula pairs.")
        self.policy = {
            "version": "selected_pairs_single_rule_per_table_v1",
            "formula_selection": "probability proportional to retained-pair count",
            "parent_selection": "uniform retained pairs for selected formula, with replacement",
            "rule_scope": "one rule for all context and query rows in each table",
            "retained_pair_counts": {name: len(indices) for name, indices in self.formula_positions.items()},
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.identity = {**deepcopy(sampler.identity), "table_sampling_policy": deepcopy(self.policy)}

    def sample(self, n, *, perturb_strength, random_state):
        if n <= 0 or int(n) != n or not math.isfinite(perturb_strength) or perturb_strength < 0:
            raise ValueError("Invalid sample count or perturbation strength.")
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        representative = int(rng.integers(len(self.pairs)))
        formula = self.pairs.iloc[representative].formula
        positions = self.formula_positions[formula]
        # A shallow view preserves the original dataset, coefficients, category
        # encodings and Soccol parent-index mapping without mutating the recipe.
        view = copy(self.base_sampler)
        view.pairs = self.pairs.iloc[positions]
        batch = view.sample(n, perturb_strength=perturb_strength, random_state=rng)
        batch.pair_indices = positions[batch.pair_indices]
        if not np.all(batch.formulas == formula):
            raise RuntimeError("A single-rule table contains different target formulas.")
        return batch


def install_single_rule_sampler(dataset):
    """Adapt a selected-pair stream before workers start or checkpoints load."""
    from .epit_selected_pairs import SelectedPairPriorMixture

    # Soccol inherits the same allocation/stream implementation.
    if not isinstance(dataset, SelectedPairPriorMixture):
        raise ValueError("Single-rule training requires an EPIT or Soccol selected-pair prior.")
    prior = dataset.priors[1]
    if isinstance(prior.sampler, SingleRuleSelectedPairSampler):
        raise ValueError("Single-rule sampling is already installed.")
    prior.sampler = SingleRuleSelectedPairSampler(prior.sampler)
    dataset._config["selected_pair_recipe"] = deepcopy(prior.sampler.identity)
    return dataset
