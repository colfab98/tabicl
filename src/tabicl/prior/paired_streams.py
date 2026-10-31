"""Indexed generic/informed streams shared by mixture and specialist runs.

Stream IDs count tables, not optimizer steps. Groups retain the prior's shared
hyperparameters and compatible micro-batch shapes. Worker scheduling, mixture
weights, and other tables' random draws never enter a table's seed.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import random
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset, get_worker_info

from .checkpoint import configuration_value, empirical_prior_identity
from .dataset import SCMPrior
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


STREAM_VERSION = "paired_generic_informed_v2"
MAX_GENERATION_ATTEMPTS = 100
GENERIC_RECIPE = {
    "min_features": 2, "max_features": 100,
    "mix_probs": [0.7, 0.3], "cat_prob": 0.2, "permute_features": True,
}


@lru_cache(maxsize=1)
def implementation_identity():
    names = ("paired_streams.py", "checkpoint.py", "dataset.py", "prior_config.py",
             "hp_sampling.py", "mlp_scm.py", "tree_scm.py", "reg2cls.py",
             "epit_feature_generator.py", "epit_composition_profile.py",
             "soccol_feature_generator.py", "target_variation.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def stream_policy(seed=42):
    return {
        "version": STREAM_VERSION, "seed": int(seed),
        "group_size": 4, "allocation": "exact_groups_per_batch",
        "identity": "source_and_table_index_independent_of_ratio_and_workers",
        "generic_recipe": GENERIC_RECIPE.copy(),
        "scm_target_rng": "separate_from_physical_features_and_formula",
        "retry_policy": "first_attempt_unchanged_then_attempt_indexed_scm_seed",
        "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
        "implementation_sha256": implementation_identity(),
    }


def keyed_seed(*parts):
    return int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0])


@contextmanager
def seeded(seed):
    """Keep CPU generation independent of training and other task RNGs."""
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(np.random.SeedSequence(seed).generate_state(4))
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


class _PairedSCMPrior(SCMPrior):
    def generate_dataset(self, params):
        return super().generate_dataset(params, max_attempts=MAX_GENERATION_ATTEMPTS)

    def _generate_scm_target_from_physical_features(self, X, params, prior_cls):
        # MLP/tree choices must not change physical draws or the sampled formula.
        # Keep the original first attempt, but do not rebuild the same failing
        # network forever when rejection sampling asks for another attempt.
        attempt = int(params.get("generation_attempt", 0))
        seed_parts = (params["paired_task_seed"], 1)
        if attempt:
            seed_parts += (attempt,)
        with seeded(keyed_seed(*seed_parts)):
            return super()._generate_scm_target_from_physical_features(X, params, prior_cls)


class PairedPriorDataset(IterableDataset):
    def __init__(
        self, *, batch_size=512, batch_size_per_gp=4, min_features=70,
        max_features=70, max_classes=0, min_seq_len=None, max_seq_len=1024,
        log_seq_len=False, seq_len_per_gp=False, min_train_size=0.1,
        max_train_size=0.9, replay_small=False, prior_type="hybrid_scm",
        scm_fixed_hp=None, scm_sampled_hp=None, n_jobs=1,
        informed_prior_ratio=1.0, return_source_ids=False, device="cpu",
        seed=42, rank=0, world_size=1,
    ):
        super().__init__()
        if device != "cpu" or n_jobs != 1 or max_classes != 0:
            raise ValueError("Paired streams require CPU regression priors with prior_n_jobs=1.")
        if prior_type not in {"hybrid_scm", "informed_scm", "mix_scm"}:
            raise ValueError("Paired streams require hybrid_scm, informed_scm, or mix_scm.")
        if batch_size_per_gp <= 0 or batch_size % batch_size_per_gp:
            raise ValueError("Paired batch size must be divisible by its group size.")
        ratio = {"informed_scm": 1.0, "mix_scm": 0.0}.get(prior_type, informed_prior_ratio)
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("Informed ratio must lie in [0, 1].")
        groups = batch_size // batch_size_per_gp
        if not math.isclose(groups * ratio, round(groups * ratio), abs_tol=1e-9):
            raise ValueError("Exact paired proportions require an integer number of informed groups per rank/batch.")
        if seed < 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid stream seed or distributed rank.")
        self.batch_size, self.group_size = batch_size, batch_size_per_gp
        self.seed, self.rank, self.world_size = int(seed), rank, world_size
        self.group_counts = (groups - round(groups * ratio), round(groups * ratio))
        self.return_source_ids = return_source_ids
        self._next_batch = None
        self.last_batch_ids = []
        fixed = copy.deepcopy(DEFAULT_FIXED_HP if scm_fixed_hp is None else scm_fixed_hp)
        sampled = DEFAULT_SAMPLED_HP if scm_sampled_hp is None else scm_sampled_hp
        generic_fixed = copy.deepcopy(DEFAULT_FIXED_HP)
        generic_fixed.update({key: value for key, value in GENERIC_RECIPE.items() if key not in {"min_features", "max_features"}})
        fixed["paired_prior_streams"] = True
        common = dict(
            batch_size=batch_size_per_gp, batch_size_per_gp=batch_size_per_gp,
            max_classes=0, min_seq_len=min_seq_len, max_seq_len=max_seq_len,
            log_seq_len=log_seq_len, seq_len_per_gp=True,
            min_train_size=min_train_size, max_train_size=max_train_size,
            replay_small=replay_small, n_jobs=1, device="cpu",
        )
        # Generic defaults are independent of every informed Optuna setting.
        self.priors = [
            _PairedSCMPrior(**common, min_features=2, max_features=100,
                            prior_type="mix_scm", fixed_hp=generic_fixed,
                            sampled_hp=DEFAULT_SAMPLED_HP),
            _PairedSCMPrior(**common, min_features=min_features, max_features=max_features,
                            prior_type="informed_scm", fixed_hp=fixed, sampled_hp=sampled)
            if self.group_counts[1] else None,
        ]
        self._config = configuration_value({
            "version": STREAM_VERSION, "seed": self.seed,
            "implementation_sha256": implementation_identity(),
            "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
            "batch_size": batch_size, "group_size": self.group_size,
            "group_counts": self.group_counts, "rank": rank, "world_size": world_size,
            "return_source_ids": return_source_ids, "shape_settings": common,
            "informed_features": [min_features, max_features],
            "generic_fixed": generic_fixed, "generic_sampled": DEFAULT_SAMPLED_HP,
            "informed_fixed": fixed if self.group_counts[1] else None,
            "informed_sampled": sampled if self.group_counts[1] else None,
            "profile": empirical_prior_identity(fixed) if self.group_counts[1] else None,
        })

    def sampling_config(self):
        return copy.deepcopy(self._config)

    def checkpoint_identity(self):
        # The master writes one checkpoint for every rank. Rank-specific loader
        # configurations are checked separately in sampling_state['ranks'].
        identity = self.sampling_config()
        identity.pop("rank")
        return identity

    def sampling_state_dict(self):
        return {"version": STREAM_VERSION, "next_batch": self._next_batch}

    def load_sampling_state_dict(self, state):
        if state["version"] != STREAM_VERSION:
            raise ValueError("Incompatible paired stream checkpoint.")
        self._next_batch = state["next_batch"]

    def _group(self, source, group_id):
        prior = self.priors[source]
        # Shared metadata are keyed only by source group, never by batch/ratio.
        with seeded(keyed_seed(self.seed, source, group_id, 0)):
            seq_len = prior.sample_seq_len(prior.min_seq_len, prior.max_seq_len,
                                          log=prior.log_seq_len, replay_small=prior.replay_small)
            train_size = prior.sample_train_size(prior.min_train_size, prior.max_train_size, seq_len)
            if train_size < 2 or seq_len - train_size < 2:
                raise ValueError("Paired regression tables need at least two context and two query rows; adjust sequence/split limits.")
            sampled = prior.hp_sampling()
            prior_type, informed = prior.get_prior()
            width = round(np.random.uniform(prior.min_features, prior.max_features))
            hp = {k: v() if callable(v) else v for k, v in sampled.items()}
        rows = []
        for within in range(self.group_size):
            task_id = group_id * self.group_size + within
            task_seed = keyed_seed(self.seed, source, task_id, 1)
            params = {**prior.fixed_hp, **hp, "seq_len": seq_len, "train_size": train_size,
                      "max_features": prior.max_features, "num_features": width,
                      "num_classes": 0, "prior_type": prior_type, "informed_mode": informed,
                      "device": "cpu", "paired_task_seed": task_seed}
            with seeded(task_seed):
                if prior.coefficient_variation:
                    prior._coefficient_rng = np.random.default_rng(keyed_seed(
                        task_seed, 2, prior.fixed_hp.get("pitting_coefficient_variation_seed", 42)))
                    worker = get_worker_info()
                    prior._coefficient_worker_seed = None if worker is None else worker.seed
                X, y, d = prior.generate_dataset(params)
            rows.append((X, y, d, seq_len, train_size, prior._missingness_source_id(params)))
            self.last_batch_ids.append((source, task_id))
        return rows

    def get_batch(self, batch_size=None):
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError("Changing paired batch size changes the stream allocation; start a new run.")
        worker = get_worker_info()
        if self._next_batch is None:
            self._next_batch = 0 if worker is None else worker.id
        step = self._next_batch
        groups = []
        for source, count in enumerate(self.group_counts):
            first = (step * self.world_size + self.rank) * count
            groups.extend((source, first + offset) for offset in range(count))
        # Shuffle whole groups; retain equal shapes/splits within micro-batches.
        np.random.default_rng(keyed_seed(self.seed, step, self.rank, 3)).shuffle(groups)
        self.last_batch_ids = []
        rows = [row for source, group_id in groups for row in self._group(source, group_id)]
        self._next_batch += 1 if worker is None else worker.num_workers
        X, y, d, lengths, train_sizes, sources = zip(*rows)
        X_batch = torch.stack(X) if len({tuple(x.shape) for x in X}) == 1 else nested_tensor(list(X))
        y_batch = torch.stack(y) if len({tuple(t.shape) for t in y}) == 1 else nested_tensor(list(y))
        batch = (X_batch, y_batch, torch.stack(d),
                 torch.tensor(lengths), torch.tensor(train_sizes))
        if self.return_source_ids:
            batch += (torch.tensor(sources),)
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()
