"""Checkpoint RNGs and the consumed boundary of a prefetched prior stream."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def rng_state_dict(*, include_cuda=False):
    """Use only tensors and Python values, compatible with weights_only loading."""
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), position, has_gauss, cached_gauss),
        "torch": torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state_dict(state):
    # DataLoader's pin-memory traversal may convert tuples to lists.
    version, internal_state, gauss = state["python"]
    random.setstate((version, tuple(internal_state), gauss))
    algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG state requires the same number of visible CUDA devices.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _identity(value):
    # default_convert would turn Python RNG state tuples into lists.
    return value


class _WorkerSnapshots(IterableDataset):
    def __init__(self, dataset, worker_states, next_worker):
        self.dataset = dataset
        self.worker_states = worker_states
        self.next_worker = next_worker

    def __iter__(self):
        worker = get_worker_info()
        state = self.worker_states.get(worker.id)
        if state is not None:
            self.dataset.load_sampling_state_dict(state["prior"])
            load_rng_state_dict(state["rng"])
        # A new DataLoader starts at worker zero. Empty slots realign its round
        # robin to the next unconsumed worker without advancing any generator.
        if worker.id < self.next_worker:
            yield None
        for batch in self.dataset:
            yield batch, worker.id, {
                "rng": rng_state_dict(),
                "prior": self.dataset.sampling_state_dict(),
            }


class CheckpointablePriorLoader:
    """Resume CPU priors without serializing large prefetched batches.

    Workers attach their state immediately after each batch. Only states from
    batches delivered to training are checkpointed; unconsumed prefetch is
    regenerated on restart. Worker identities and original worker seeds stay
    fixed, including when resuming between round-robin cycles.
    """

    def __init__(self, *, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self.num_workers = kwargs["num_workers"]
        self._iterator = None
        self._seed_state = None
        self._worker_states = {}
        self._consumed = 0
        self._restarting = False

    def __iter__(self):
        if self._iterator is not None:
            return self
        kwargs = dict(self.kwargs)
        dataset = self.dataset
        if self.num_workers:
            dataset = _WorkerSnapshots(dataset, self._worker_states, self._consumed % self.num_workers)
        if self._restarting and self._seed_state is not None:
            # Reproduce worker seeds without spending another draw from the
            # main training RNG, already restored at its checkpoint boundary.
            kwargs["generator"] = torch.Generator().set_state(self._seed_state.cpu())
        else:
            self._seed_state = torch.get_rng_state()
        self._iterator = iter(DataLoader(dataset=dataset, collate_fn=_identity, **kwargs))
        return self

    def __next__(self):
        if self._iterator is None:
            iter(self)
        item = next(self._iterator)
        if self.num_workers:
            while item is None:
                item = next(self._iterator)
            batch, worker_id, state = item
            if worker_id != self._consumed % self.num_workers:
                raise RuntimeError("Prior loader requires ordered worker delivery.")
            self._worker_states[worker_id] = state
        else:
            batch = item
        self._consumed += 1
        return batch

    def state_dict(self):
        return {
            "version": 1,
            "num_workers": self.num_workers,
            "dataset_config": self.dataset.sampling_config(),
            "seed_state": self._seed_state,
            "consumed": self._consumed,
            "workers": copy.deepcopy(self._worker_states),
            "prior": self.dataset.sampling_state_dict(),
        }

    def load_state_dict(self, state):
        if self._iterator is not None:
            raise ValueError("Restore prior sampling state before starting the loader.")
        if state["version"] != 1:
            raise ValueError("Unsupported prior sampling checkpoint version.")
        if state["num_workers"] != self.num_workers or state["dataset_config"] != self.dataset.sampling_config():
            raise ValueError("Prior sampling restore requires the same worker count and dataset settings.")
        self._seed_state = state["seed_state"]
        self._consumed = state["consumed"]
        self._worker_states = copy.deepcopy(state["workers"])
        self.dataset.load_sampling_state_dict(state["prior"])
        self._restarting = True
