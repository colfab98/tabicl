"""Original SCM batches with an independent informed prior mixed in.

Generic generation calls SCMPrior.get_batch directly: original batch-level
split, group/subgroup hyperparameters, preprocessing and rejection sampling.
Only batch seeds and selection of source tables belong to the mixture layer.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nested import nested_tensor
from torch.utils.data import get_worker_info

from .dataset import SCMPrior
from .checkpoint import configuration_value
from .paired_streams import (
    GENERIC_RECIPE, MAX_GENERATION_ATTEMPTS, PairedPriorDataset, _PairedSCMPrior, keyed_seed, seeded,
    implementation_identity,
)
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


VERSION = "original_tabicl_mixture_v1"


def stream_policy(seed=42):
    return {
        "version": VERSION, "seed": int(seed), "group_size": 4,
        "allocation": "exact_source_groups_per_global_batch",
        "generic_recipe": copy.deepcopy(GENERIC_RECIPE),
        "generic_generation": "original_scm_full_batch_plan_prefix",
        "generation_attempts": {"generic": None, "informed": MAX_GENERATION_ATTEMPTS},
        "split": "one_original_split_per_global_batch",
        "informed_input": "context_mean_imputation_then_original_feature_preprocessing",
        "informed_target_rng": "separate_from_physical_features",
        "grouped_padding": "wrap_within_each_table_real_features_v1",
        # Training fixes must invalidate study/resume identities too, even when
        # they leave the synthetic samples and model parameter shapes unchanged.
        "training_model_sha256": {
            name: hashlib.sha256((Path(__file__).parents[1] / "model" / name).read_bytes()).hexdigest()
            for name in ("tabicl.py", "embedding.py")
        },
        "implementation_sha256": {
            **implementation_identity(),
            "original_mixture.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


class _InformedPrior(_PairedSCMPrior):
    def generate_dataset(self, params):
        task_seed = keyed_seed(self.batch_seed, self.table_index, 1)
        self.table_index += 1
        with seeded(task_seed):
            if self.coefficient_variation:
                self._coefficient_rng = np.random.default_rng(keyed_seed(
                    task_seed, 2, self.fixed_hp.get("pitting_coefficient_variation_seed", 42)))
                worker = get_worker_info()
                self._coefficient_worker_seed = None if worker is None else worker.seed
            return super().generate_dataset({**params, "paired_task_seed": task_seed})


class OriginalPriorMixture(PairedPriorDataset):
    """Checkpointable mixture; ranks receive slices of the same global batch.

Generation is replayed on each rank to retain the original sequential generic
RNG behavior. GPU work is partitioned without changing tables or the split.
DataLoader workers use batch indices, so prefetching does not change samples.
"""
    def __init__(self, **kwargs):
        if kwargs.get("seq_len_per_gp", False):
            raise ValueError("Original mixture requires batch-level context/query splits.")
        super().__init__(**kwargs)
        global_size = self.batch_size * self.world_size
        shape = dict(self._config["shape_settings"])
        shape.update(batch_size=global_size, seq_len_per_gp=False)
        # configuration_value is unnecessary for these primitive shape settings.
        fixed = copy.deepcopy(DEFAULT_FIXED_HP)
        fixed.update({k: v for k, v in GENERIC_RECIPE.items() if k not in {"min_features", "max_features"}})
        generic = SCMPrior(**shape, min_features=2, max_features=100,
                           prior_type="mix_scm", fixed_hp=fixed, sampled_hp=DEFAULT_SAMPLED_HP)
        informed = None
        if self.group_counts[1]:
            old = self.priors[1]
            informed_fixed = copy.deepcopy(old.fixed_hp)
            informed_fixed.update(original_tabicl_input=True, permute_features=True)
            informed = _InformedPrior(**shape, min_features=old.min_features,
                max_features=old.max_features, prior_type="informed_scm",
                fixed_hp=informed_fixed, sampled_hp=old.sampled_hp)
        self.priors = [generic, informed]
        self._config.update(version=VERSION, policy=stream_policy(self.seed),
                            shape_settings=shape, global_batch_size=global_size,
                            max_generation_attempts={"generic": None, "informed": MAX_GENERATION_ATTEMPTS},
                            informed_fixed=configuration_value(informed.fixed_hp) if informed else None)

    def sampling_state_dict(self):
        return {"version": VERSION, "next_batch": self._next_batch}

    def load_sampling_state_dict(self, state):
        if state["version"] != VERSION:
            raise ValueError("Cannot resume historical paired streams as original mixtures.")
        self._next_batch = state["next_batch"]

    def _global_group_counts(self, step):
        return tuple(count * self.world_size for count in self.group_counts)

    def get_batch(self, batch_size=None):
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError("Changing batch size requires a new run.")
        worker = get_worker_info()
        if self._next_batch is None:
            self._next_batch = 0 if worker is None else worker.id
        step = self._next_batch
        global_size = self.batch_size * self.world_size
        batch_seed = keyed_seed(self.seed, step, 11)
        groups = []
        group_counts = self._global_group_counts(step)
        for source, count in enumerate(group_counts):
            if not count:
                continue
            prior = self.priors[source]
            prior.batch_seed, prior.table_index = batch_seed, 0
            # Both priors begin with the same original sequence/split draws.
            # Subsequent draws are isolated; informed settings never enter generic.
            with seeded(batch_seed):
                batch = prior.get_batch(global_size, prefix_size=count * self.group_size)
            for group in range(count):
                group_rows = []
                for offset in range(self.group_size):
                    i = group * self.group_size + offset
                    group_rows.append((*[part[i] for part in batch[:5]], source, i))
                groups.append(group_rows)
        # At 0% retain the original generic ordering exactly.
        if all(group_counts):
            np.random.default_rng(keyed_seed(self.seed, step, 12)).shuffle(groups)
        rows = [row for group in groups for row in group]
        rows = rows[self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
        self.last_batch_ids = [(row[5], step * global_size + row[6]) for row in rows]
        X, y, d, lengths, splits, sources, _ = zip(*rows)
        X = torch.stack(X) if len({tuple(x.shape) for x in X}) == 1 else nested_tensor(list(X))
        result = (X, torch.stack(y), torch.stack(d), torch.stack(lengths), torch.stack(splits))
        self._next_batch += 1 if worker is None else worker.num_workers
        if self.return_source_ids:
            result += (torch.tensor(sources),)
        return result
