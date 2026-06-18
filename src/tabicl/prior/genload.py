"""
Utilities for generating and loading prior datasets from disk for training.

This module implements a workflow for generating synthetic tabular data from prior
distributions, saving it efficiently to disk in a sparse format, and loading it during
distributed training.

This module provides two main classes:
- SavePriorDataset: Generates and saves batches of prior data to disk
- LoadPriorDataset: Loads pre-generated prior data from disk for distributed training

The data is saved in a sparse format to reduce storage requirements and loaded
on demand during training. The module supports distributed training by allowing
different processes to load different batches in a coordinated way.

The saved data includes:
- X: Input features in sparse format or nested tensor (for variable-length sequences)
- y: Target labels as regular tensor or nested tensor (for variable-length sequences)
- d: Number of features per dataset
- seq_lens: Sequence length for each dataset
- train_sizes: Position at which to split training and evaluation data
- batch_size: Number of datasets in the batch
"""

from __future__ import annotations

import time
import json
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from torch.utils.data import IterableDataset

from tabicl.prior.dataset import PriorDataset
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def dense2sparse(
    dense_tensor: torch.Tensor, row_lengths: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert a dense tensor with trailing zeros into a compact 1D representation.

    Parameters
    ----------
    dense_tensor : torch.Tensor
        Input tensor of shape ``(num_rows, num_cols)`` where each row may contain
        trailing zeros beyond the valid entries.

    row_lengths : torch.Tensor
        Tensor of shape ``(num_rows,)`` specifying the number of valid entries
        in each row of the dense tensor.

    dtype : torch.dtype, default=torch.float32
        Output data type for the sparse representation.

    Returns
    -------
    torch.Tensor
        1D tensor of shape ``(sum(row_lengths),)`` containing only the valid entries.
    """

    assert dense_tensor.dim() == 2, "dense_tensor must be 2D"
    num_rows, num_cols = dense_tensor.shape
    assert row_lengths.shape[0] == num_rows, "row_lengths must match number of rows"
    assert (row_lengths <= num_cols).all(), "row_lengths cannot exceed number of columns"

    indices = torch.arange(num_cols, device=dense_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    sparse = dense_tensor[mask].to(dtype)

    return sparse


def sparse2dense(
    sparse_tensor: torch.Tensor,
    row_lengths: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a dense tensor from its sparse representation.

    This function is the inverse of dense2sparse, reconstructing a padded dense
    tensor from a compact 1D representation and the corresponding row lengths.
    Unused entries in the output are filled with zeros.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        1D tensor containing the valid entries from the original dense tensor.

    row_lengths : torch.Tensor
        Number of valid entries for each row in the output tensor.

    max_len : int, optional
        Maximum length for each row in the output. If None, uses
        ``max(row_lengths)``.

    dtype : torch.dtype, default=torch.float32
        Output data type for the dense representation.

    Returns
    -------
    torch.Tensor
        Dense tensor of shape ``(num_rows, max_len)`` with zeros padding.
    """

    assert sparse_tensor.dim() == 1, "data must be 1D"
    assert row_lengths.sum() == len(sparse_tensor), "data length must match sum of row_lengths"

    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)

    return dense


class SliceNestedTensor:
    """A wrapper for nested tensors that supports slicing along the first dimension.

    This class wraps PyTorch's nested tensor and provides slicing operations
    along the first dimension, which are not natively supported by nested tensors.
    It maintains compatibility with other nested tensor operations by forwarding
    attribute access to the wrapped tensor.

    Parameters
    ----------
    nested_tensor : torch.Tensor
        A nested tensor to wrap
    """

    def __init__(self, nested_tensor):
        self.nested_tensor = nested_tensor
        self.is_nested = nested_tensor.is_nested

    def __getitem__(self, idx):
        """Support slicing operations along the first dimension."""
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = self.nested_tensor.size(0) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step

            indices = list(range(start, stop, step))
            return SliceNestedTensor(torch.nested.nested_tensor([self.nested_tensor[i] for i in indices]))
        elif isinstance(idx, int):
            return self.nested_tensor[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped nested tensor."""
        return getattr(self.nested_tensor, name)

    def __len__(self):
        """Return the length of the first dimension."""
        return self.nested_tensor.size(0)

    def to(self, *args, **kwargs):
        """Support the to() method for device/dtype conversion."""
        return SliceNestedTensor(self.nested_tensor.to(*args, **kwargs))


def cat_slice_nested_tensors(tensors: List, dim=0) -> SliceNestedTensor:
    """Concatenate a list of SliceNestedTensor objects along dimension dim.

    Parameters
    ----------
    tensors : list
        List of tensors to concatenate.

    dim : int, default=0
        Dimension along which to concatenate.

    Returns
    -------
    SliceNestedTensor
        Concatenated tensor wrapped in SliceNestedTensor.
    """
    # Extract the wrapped nested tensors
    nested_tensors = [t.nested_tensor if isinstance(t, SliceNestedTensor) else t for t in tensors]
    return SliceNestedTensor(torch.cat(nested_tensors, dim=dim))


class LoadPriorDataset(IterableDataset):
    """Load pre-generated prior data sequentially for distributed training.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the batch files.

    batch_size : int, default=512
        Number of datasets to return in each iteration.

    ddp_world_size : int, default=1
        Total number of distributed processes.

    ddp_rank : int, default=0
        Rank of current process.

    start_from : int, default=0
        Batch index to start loading from.

    max_batches : int, optional
        Maximum number of batches to load. If None, load indefinitely.

    timeout : int, default=60
        Maximum time in seconds to wait for a batch file.

    delete_after_load : bool, default=False
        Whether to delete batch files after loading them.

    device : str, default='cpu'
        Device to load tensors to.
    """

    def __init__(
        self,
        data_dir,
        batch_size=512,
        ddp_world_size=1,
        ddp_rank=0,
        start_from=0,
        max_batches=None,
        timeout=60,
        delete_after_load=False,
        device="cpu",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.current_idx = ddp_rank + start_from
        self.max_batches = max_batches
        self.timeout = timeout
        self.delete_after_load = delete_after_load
        self.device = device

        # Load metadata if available
        self.metadata = None
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load or parse metadata.json: {e}")

        # Buffer for storing datasets that haven't been returned yet
        self.buffer_X = None
        self.buffer_y = None
        self.buffer_d = None
        self.buffer_seq_lens = None
        self.buffer_train_sizes = None
        self.buffer_size = 0

    def __iter__(self):
        return self

    def _load_batch_file(self):
        """Load a single batch file from disk.

        Returns
        -------
        tuple
            A tuple containing (X, y, d, seq_lens, train_sizes, batch_size).
        """
        batch_file = self.data_dir / f"batch_{self.current_idx:06d}.pt"

        # Try loading the file for up to timeout seconds
        wait_time = 0
        while not batch_file.exists():
            if wait_time >= self.timeout:
                raise RuntimeError(f"Timeout waiting for batch file {batch_file}")
            time.sleep(5)
            wait_time += 5

        batch = torch.load(batch_file, map_location=self.device, weights_only=True)
        X = batch["X"]
        y = batch["y"]
        d = batch["d"]
        seq_lens = batch["seq_lens"]
        train_sizes = batch["train_sizes"]
        batch_size = batch["batch_size"]

        if X.is_nested:
            # Wrap nested tensors with SliceNestedTensor
            X = SliceNestedTensor(X)
            y = SliceNestedTensor(y)
        else:
            # Convert sparse tensor to dense
            X = sparse2dense(X, d.repeat_interleave(seq_lens[0]), dtype=torch.float32).view(batch_size, seq_lens[0], -1)

        # Delete file if requested
        if self.delete_after_load and batch_file.exists():
            batch_file.unlink()

        # Prepare next index for this process
        self.current_idx += self.ddp_world_size

        return X, y, d, seq_lens, train_sizes, batch_size

    def __next__(self):
        """Load datasets until we have at least batch_size, then return exactly batch_size.

        This method accumulates datasets from multiple files if necessary to return
        the exact number of datasets specified in batch_size. Any extra datasets are
        kept in a buffer for the next iteration.

        Returns
        -------
        X : Tensor or NestedTensor
            Input features ``[batch_size, seq_len, features]`` or nested tensor.

        y : Tensor or NestedTensor
            Target labels ``[batch_size, seq_len]`` or nested tensor.

        d : Tensor
            Number of features per dataset.

        seq_lens : Tensor
            Sequence length for each dataset.

        train_sizes : Tensor
            Position at which to split training and evaluation data.
        """
        # Check if we've reached the maximum number of batches and have no buffered data
        if self.max_batches is not None and self.current_idx >= self.max_batches and (self.buffer_size == 0):
            raise StopIteration

        # Initialize or use existing buffer
        if self.buffer_size == 0:
            # Load the first batch
            X, y, d, seq_lens, train_sizes, file_batch_size = self._load_batch_file()
            self.buffer_X = X
            self.buffer_y = y
            self.buffer_d = d
            self.buffer_seq_lens = seq_lens
            self.buffer_train_sizes = train_sizes
            self.buffer_size = file_batch_size

        # Keep loading files until we have enough data or no more files
        while self.buffer_size < self.batch_size:
            # Check if we've reached max_batches
            if self.max_batches is not None and self.current_idx >= self.max_batches:
                # If we can't get a full batch, return what we have
                break

            try:
                # Load another batch and append to buffer
                X, y, d, seq_lens, train_sizes, file_batch_size = self._load_batch_file()

                # Concatenate with existing buffer
                if self.buffer_X is None:
                    # If buffer is empty, directly assign
                    self.buffer_X = X
                    self.buffer_y = y
                    self.buffer_d = d
                    self.buffer_seq_lens = seq_lens
                    self.buffer_train_sizes = train_sizes
                    self.buffer_size = file_batch_size
                else:
                    # Otherwise concatenate, handling SliceNestedTensor if needed
                    if isinstance(X, SliceNestedTensor):
                        self.buffer_X = cat_slice_nested_tensors([self.buffer_X, X], dim=0)
                        self.buffer_y = cat_slice_nested_tensors([self.buffer_y, y], dim=0)
                    else:
                        self.buffer_X = torch.cat([self.buffer_X, X], dim=0)
                        self.buffer_y = torch.cat([self.buffer_y, y], dim=0)

                    self.buffer_d = torch.cat([self.buffer_d, d], dim=0)
                    self.buffer_seq_lens = torch.cat([self.buffer_seq_lens, seq_lens], dim=0)
                    self.buffer_train_sizes = torch.cat([self.buffer_train_sizes, train_sizes], dim=0)
                    self.buffer_size += file_batch_size
            except Exception as e:
                # If we can't load more files, use what we have
                print(f"Warning: Could not load more files: {str(e)}")
                break

        # Extract batch_size datasets (or all if we have fewer)
        output_size = min(self.batch_size, self.buffer_size)

        # Prepare output
        X_out = self.buffer_X[:output_size]
        y_out = self.buffer_y[:output_size]
        d_out = self.buffer_d[:output_size]
        seq_lens_out = self.buffer_seq_lens[:output_size]
        train_sizes_out = self.buffer_train_sizes[:output_size]

        # Update buffer with remaining data
        if output_size < self.buffer_size:
            self.buffer_X = self.buffer_X[output_size:]
            self.buffer_y = self.buffer_y[output_size:]
            self.buffer_d = self.buffer_d[output_size:]
            self.buffer_seq_lens = self.buffer_seq_lens[output_size:]
            self.buffer_train_sizes = self.buffer_train_sizes[output_size:]
            self.buffer_size -= output_size
        else:
            # Buffer is now empty
            self.buffer_X = None
            self.buffer_y = None
            self.buffer_d = None
            self.buffer_seq_lens = None
            self.buffer_train_sizes = None
            self.buffer_size = 0

        if isinstance(X_out, SliceNestedTensor):
            X_out = X_out.nested_tensor
            y_out = y_out.nested_tensor

        return X_out, y_out, d_out, seq_lens_out, train_sizes_out

    def __repr__(self) -> str:
        """Return a string representation of the LoadPriorDataset.

        Returns
        -------
        str
            A formatted string with dataset parameters.
        """
        repr_str = (
            f"LoadPriorDataset(\n"
            f"  data_dir: {self.data_dir}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  ddp_world_size: {self.ddp_world_size}\n"
            f"  ddp_rank: {self.ddp_rank}\n"
            f"  start_from: {self.current_idx - self.ddp_rank}\n"
            f"  max_batches: {self.max_batches or 'Infinite'}\n"
            f"  timeout: {self.timeout}\n"
            f"  delete_after_load: {self.delete_after_load}\n"
            f"  device: {self.device}\n"
        )
        if self.metadata:
            repr_str += "  Loaded Metadata:\n"
            repr_str += f"    prior_type: {self.metadata.get('prior_type', 'N/A')}\n"
            repr_str += f"    batch_size (generated): {self.metadata.get('batch_size', 'N/A')}\n"
            repr_str += f"    batch_size_per_gp: {self.metadata.get('batch_size_per_gp', 'N/A')}\n"
            repr_str += f"    min features: {self.metadata.get('min_features', 'N/A')}\n"
            repr_str += f"    max features: {self.metadata.get('max_features', 'N/A')}\n"
            repr_str += f"    max classes: {self.metadata.get('max_classes', 'N/A')}\n"
            repr_str += f"    seq_len: {self.metadata.get('min_seq_len', 'N/A') or 'None'} - {self.metadata.get('max_seq_len', 'N/A')}\n"
            repr_str += f"    log_seq_len: {self.metadata.get('log_seq_len', 'N/A')}\n"
            repr_str += f"    sequence length varies across groups: {self.metadata.get('seq_len_per_gp', 'N/A')}\n"
            repr_str += f"    train_size: {self.metadata.get('min_train_size', 'N/A')} - {self.metadata.get('max_train_size', 'N/A')}\n"
            repr_str += f"    replay_small: {self.metadata.get('replay_small', 'N/A')}\n"
        repr_str += ")"

        return repr_str


class SavePriorDataset:
    """Generates and saves batches of prior datasets to disk.

    The datasets are saved as individual batch files in the specified directory
    using an atomic file writing pattern to ensure data integrity.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration for dataset generation
    """

    def __init__(self, args):
        self.args = args
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_metadata()

        scm_fixed_hp = dict(DEFAULT_FIXED_HP)
        if self.args.mix_probs is not None:
            scm_fixed_hp["mix_probs"] = tuple(self.args.mix_probs)
        if self.args.informed_mix_probs is not None:
            scm_fixed_hp["informed_mix_probs"] = tuple(self.args.informed_mix_probs)
        if self.args.informed_block_allocation is not None and self.args.informed_normal_block_allocation is None:
            material, environment, electrochem, history, intervention = self.args.informed_block_allocation
            scm_fixed_hp["informed_normal_block_allocation"] = (
                material,
                environment,
                history,
                0.0,
                0.0,
                intervention,
                0.0,
                electrochem,
                0.0,
            )
        if self.args.informed_task_family_probs is not None:
            scm_fixed_hp["informed_task_family_probs"] = tuple(self.args.informed_task_family_probs)
        if self.args.informed_normal_block_allocation is not None:
            scm_fixed_hp["informed_normal_block_allocation"] = tuple(self.args.informed_normal_block_allocation)
        if self.args.informed_inhibitor_block_allocation is not None:
            scm_fixed_hp["informed_inhibitor_block_allocation"] = tuple(self.args.informed_inhibitor_block_allocation)
        if self.args.informed_normal_block_allocation_ranges is not None:
            scm_fixed_hp["informed_normal_block_allocation_ranges"] = tuple(
                self.args.informed_normal_block_allocation_ranges
            )
        if self.args.informed_inhibitor_block_allocation_ranges is not None:
            scm_fixed_hp["informed_inhibitor_block_allocation_ranges"] = tuple(
                self.args.informed_inhibitor_block_allocation_ranges
            )
        if self.args.informed_normal_block_allocation_min_counts is not None:
            scm_fixed_hp["informed_normal_block_allocation_min_counts"] = tuple(
                self.args.informed_normal_block_allocation_min_counts
            )
        if self.args.informed_inhibitor_block_allocation_min_counts is not None:
            scm_fixed_hp["informed_inhibitor_block_allocation_min_counts"] = tuple(
                self.args.informed_inhibitor_block_allocation_min_counts
            )
        if self.args.informed_feature_block_strength is not None:
            scm_fixed_hp["informed_feature_block_strength"] = self.args.informed_feature_block_strength
        if self.args.informed_interaction_strength is not None:
            scm_fixed_hp["informed_interaction_strength"] = self.args.informed_interaction_strength
        if self.args.informed_history_strength is not None:
            scm_fixed_hp["informed_history_strength"] = self.args.informed_history_strength
        if self.args.informed_intervention_strength is not None:
            scm_fixed_hp["informed_intervention_strength"] = self.args.informed_intervention_strength
        if self.args.informed_target_family is not None:
            scm_fixed_hp["informed_target_family"] = self.args.informed_target_family
        if self.args.informed_physical_marginal_prob is not None:
            scm_fixed_hp["informed_physical_marginal_prob"] = self.args.informed_physical_marginal_prob
        if self.args.informed_physical_marginal_profile is not None:
            scm_fixed_hp["informed_physical_marginal_profile"] = self.args.informed_physical_marginal_profile
        if self.args.pitting_material_dirichlet_prob is not None:
            scm_fixed_hp["pitting_material_dirichlet_prob"] = self.args.pitting_material_dirichlet_prob
        if self.args.pitting_material_dirichlet_concentration is not None:
            scm_fixed_hp["pitting_material_dirichlet_concentration"] = self.args.pitting_material_dirichlet_concentration
        if self.args.pitting_material_dirichlet_active_prob is not None:
            scm_fixed_hp["pitting_material_dirichlet_active_prob"] = self.args.pitting_material_dirichlet_active_prob
        pitting_process_role = getattr(self.args, "pitting_process_role", None)
        if pitting_process_role is not None:
            scm_fixed_hp["pitting_process_role"] = pitting_process_role
        pitting_process_category_count = getattr(self.args, "pitting_process_category_count", None)
        if pitting_process_category_count is not None:
            scm_fixed_hp["pitting_process_category_count"] = pitting_process_category_count
        pitting_fixed_epit_schema = getattr(self.args, "pitting_fixed_epit_schema", None)
        if pitting_fixed_epit_schema is not None:
            scm_fixed_hp["pitting_fixed_epit_schema"] = pitting_fixed_epit_schema
        cat_prob = getattr(self.args, "cat_prob", None)
        if cat_prob is not None:
            scm_fixed_hp["cat_prob"] = cat_prob
        permute_features = getattr(self.args, "permute_features", None)
        if permute_features is not None:
            scm_fixed_hp["permute_features"] = permute_features
        if self.args.epit_material_coef is not None:
            scm_fixed_hp["epit_material_coef"] = self.args.epit_material_coef
        if self.args.epit_environment_coef is not None:
            scm_fixed_hp["epit_environment_coef"] = self.args.epit_environment_coef
        if self.args.epit_interaction_coef is not None:
            scm_fixed_hp["epit_interaction_coef"] = self.args.epit_interaction_coef
        if self.args.inhibitor_descriptor_coef_scale is not None:
            scm_fixed_hp["inhibitor_descriptor_coef_scale"] = self.args.inhibitor_descriptor_coef_scale
            if self.args.inhibitor_descriptor_coef is None:
                scm_fixed_hp["inhibitor_descriptor_coef"] = 0.65 * self.args.inhibitor_descriptor_coef_scale
        if self.args.inhibitor_context_coef_scale is not None:
            scm_fixed_hp["inhibitor_context_coef_scale"] = self.args.inhibitor_context_coef_scale
            if self.args.inhibitor_context_coef is None:
                scm_fixed_hp["inhibitor_context_coef"] = 0.14 * self.args.inhibitor_context_coef_scale
        if self.args.inhibitor_environment_interaction_coef_scale is not None:
            scm_fixed_hp["inhibitor_environment_interaction_coef_scale"] = (
                self.args.inhibitor_environment_interaction_coef_scale
            )
            if self.args.inhibitor_environment_interaction_coef is None:
                scm_fixed_hp["inhibitor_environment_interaction_coef"] = (
                    0.365 * self.args.inhibitor_environment_interaction_coef_scale
                )
        if self.args.inhibitor_material_interaction_coef_scale is not None:
            scm_fixed_hp["inhibitor_material_interaction_coef_scale"] = (
                self.args.inhibitor_material_interaction_coef_scale
            )
            if self.args.inhibitor_material_interaction_coef is None:
                scm_fixed_hp["inhibitor_material_interaction_coef"] = (
                    0.24 * self.args.inhibitor_material_interaction_coef_scale
                )
        if self.args.inhibitor_intervention_coef_scale is not None:
            scm_fixed_hp["inhibitor_intervention_coef_scale"] = self.args.inhibitor_intervention_coef_scale
            if self.args.inhibitor_intervention_coef is None:
                scm_fixed_hp["inhibitor_intervention_coef"] = 0.375 * self.args.inhibitor_intervention_coef_scale
        if self.args.inhibitor_descriptor_coef is not None:
            scm_fixed_hp["inhibitor_descriptor_coef"] = self.args.inhibitor_descriptor_coef
        if self.args.inhibitor_context_coef is not None:
            scm_fixed_hp["inhibitor_context_coef"] = self.args.inhibitor_context_coef
        if self.args.inhibitor_environment_interaction_coef is not None:
            scm_fixed_hp["inhibitor_environment_interaction_coef"] = self.args.inhibitor_environment_interaction_coef
        if self.args.inhibitor_material_interaction_coef is not None:
            scm_fixed_hp["inhibitor_material_interaction_coef"] = self.args.inhibitor_material_interaction_coef
        if self.args.inhibitor_intervention_coef is not None:
            scm_fixed_hp["inhibitor_intervention_coef"] = self.args.inhibitor_intervention_coef

        self.prior = PriorDataset(
            batch_size=self.args.batch_size,
            batch_size_per_gp=self.args.batch_size_per_gp,
            min_features=self.args.min_features,
            max_features=self.args.max_features,
            max_classes=self.args.max_classes,
            min_seq_len=self.args.min_seq_len,
            max_seq_len=self.args.max_seq_len,
            log_seq_len=self.args.log_seq_len,
            seq_len_per_gp=self.args.seq_len_per_gp,
            min_train_size=self.args.min_train_size,
            max_train_size=self.args.max_train_size,
            replay_small=self.args.replay_small,
            prior_type=self.args.prior_type,
            scm_fixed_hp=scm_fixed_hp,
            scm_sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=self.args.n_jobs,
            num_threads_per_generate=self.args.num_threads_per_generate,
            informed_prior_ratio=self.args.informed_prior_ratio,
            device=self.args.device,
        )
        print(self.prior)

    def save_metadata(self):
        """Save metadata about the dataset generation configuration to a JSON file."""
        metadata = {
            "prior_type": self.args.prior_type,
            "batch_size": self.args.batch_size,
            "batch_size_per_gp": self.args.batch_size_per_gp,
            "min_seq_len": self.args.min_seq_len,
            "max_seq_len": self.args.max_seq_len,
            "log_seq_len": self.args.log_seq_len,
            "seq_len_per_gp": self.args.seq_len_per_gp,
            "min_features": self.args.min_features,
            "max_features": self.args.max_features,
            "max_classes": self.args.max_classes,
            "min_train_size": self.args.min_train_size,
            "max_train_size": self.args.max_train_size,
            "replay_small": self.args.replay_small,
            "informed_prior_ratio": self.args.informed_prior_ratio,
            "mix_probs": self.args.mix_probs,
            "informed_mix_probs": self.args.informed_mix_probs,
            "informed_block_allocation": self.args.informed_block_allocation,
            "informed_task_family_probs": self.args.informed_task_family_probs,
            "informed_normal_block_allocation": self.args.informed_normal_block_allocation,
            "informed_normal_block_allocation_ranges": self.args.informed_normal_block_allocation_ranges,
            "informed_normal_block_allocation_min_counts": self.args.informed_normal_block_allocation_min_counts,
            "informed_inhibitor_block_allocation": self.args.informed_inhibitor_block_allocation,
            "informed_inhibitor_block_allocation_ranges": self.args.informed_inhibitor_block_allocation_ranges,
            "informed_inhibitor_block_allocation_min_counts": self.args.informed_inhibitor_block_allocation_min_counts,
            "informed_feature_block_strength": self.args.informed_feature_block_strength,
            "informed_interaction_strength": self.args.informed_interaction_strength,
            "informed_history_strength": self.args.informed_history_strength,
            "informed_intervention_strength": self.args.informed_intervention_strength,
            "informed_target_family": self.args.informed_target_family,
            "informed_physical_marginal_prob": self.args.informed_physical_marginal_prob,
            "informed_physical_marginal_profile": self.args.informed_physical_marginal_profile,
            "pitting_material_dirichlet_prob": self.args.pitting_material_dirichlet_prob,
            "pitting_material_dirichlet_concentration": self.args.pitting_material_dirichlet_concentration,
            "pitting_material_dirichlet_active_prob": self.args.pitting_material_dirichlet_active_prob,
            "pitting_process_role": getattr(self.args, "pitting_process_role", None),
            "pitting_process_category_count": getattr(self.args, "pitting_process_category_count", None),
            "pitting_fixed_epit_schema": getattr(self.args, "pitting_fixed_epit_schema", None),
            "cat_prob": getattr(self.args, "cat_prob", None),
            "permute_features": getattr(self.args, "permute_features", None),
            "epit_material_coef": self.args.epit_material_coef,
            "epit_environment_coef": self.args.epit_environment_coef,
            "epit_interaction_coef": self.args.epit_interaction_coef,
            "inhibitor_descriptor_coef_scale": self.args.inhibitor_descriptor_coef_scale,
            "inhibitor_context_coef_scale": self.args.inhibitor_context_coef_scale,
            "inhibitor_environment_interaction_coef_scale": self.args.inhibitor_environment_interaction_coef_scale,
            "inhibitor_material_interaction_coef_scale": self.args.inhibitor_material_interaction_coef_scale,
            "inhibitor_intervention_coef_scale": self.args.inhibitor_intervention_coef_scale,
            "inhibitor_descriptor_coef": self.args.inhibitor_descriptor_coef,
            "inhibitor_context_coef": self.args.inhibitor_context_coef,
            "inhibitor_environment_interaction_coef": self.args.inhibitor_environment_interaction_coef,
            "inhibitor_material_interaction_coef": self.args.inhibitor_material_interaction_coef,
            "inhibitor_intervention_coef": self.args.inhibitor_intervention_coef,
        }
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_batch_sparse(self, batch_idx, X, y, d, seq_lens, train_sizes):
        """Save batch data in sparse format for efficient storage.

        This method handles the conversion between dense and sparse tensor formats
        when appropriate and saves the batch data to a PyTorch file. It uses an atomic
        write pattern (writing to a temporary file and then renaming) to ensure data
        integrity even if the process is interrupted during saving.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch used for file naming.

        X : torch.Tensor
            Input features tensor, either in dense format
            ``[batch_size, seq_len, features]`` or in nested tensor format for
            variable sequence lengths.

        y : torch.Tensor
            Target labels tensor.

        d : torch.Tensor
            Number of features for each dataset.

        seq_lens : torch.Tensor
            Sequence length for each dataset.

        train_sizes : torch.Tensor
            Position at which to split training and evaluation data.
        """

        if self.args.seq_len_per_gp:
            # X and y are nested tensors and they are already sparse
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(X.view(-1, H), d.repeat_interleave(T), dtype=torch.float32)

        # Create temporary file first
        batch_file = self.save_dir / f"batch_{batch_idx:06d}.pt"
        temp_file = self.save_dir / f"batch_{batch_idx:06d}.pt.tmp"
        torch.save(
            {"X": X, "y": y, "d": d, "seq_lens": seq_lens, "train_sizes": train_sizes, "batch_size": B},
            temp_file,
        )
        # Atomic rename to ensure file integrity
        temp_file.replace(batch_file)

    def run(self):
        """Generate and save batches of prior datasets."""
        print(f"Save directory: {self.save_dir}")
        print(f"Generating {self.args.num_batches} batches starting from index {self.args.resume_from}")

        for batch_idx in tqdm(
            range(self.args.resume_from, self.args.resume_from + self.args.num_batches),
            desc="Generating batches",
        ):
            X, y, d, seq_lens, train_sizes = self.prior.get_batch()
            # Move tensors to CPU before saving
            X = X.cpu()
            y = y.cpu()
            d = d.cpu()
            seq_lens = seq_lens.cpu()
            train_sizes = train_sizes.cpu()
            self.save_batch_sparse(batch_idx, X, y, d, seq_lens, train_sizes)


if __name__ == "__main__":

    def str2bool(value):
        return value.lower() == "true"

    def false_or_float(value):
        if isinstance(value, str) and value.lower() == "false":
            return 0.0
        return float(value)

    def train_size_type(value):
        """Custom type function to handle both int and float train sizes."""
        value = float(value)
        if 0 < value < 1:
            return value
        elif value.is_integer():
            return int(value)
        else:
            raise argparse.ArgumentTypeError(
                "Train size must be either an integer (absolute position) "
                "or a float between 0 and 1 (ratio of sequence length)."
            )

    parser = argparse.ArgumentParser(description="Generate training prior datasets")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save the generated data")
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of batches to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this batch index")
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size", type=train_size_type, default=0.1, help="Minimum training size position/ratio"
    )
    parser.add_argument(
        "--max_train_size", type=train_size_type, default=0.9, help="Maximum training size position/ratio"
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="mix_scm",
        choices=["mlp_scm", "tree_scm", "mix_scm", "informed_scm", "hybrid_scm"],
        help="Type of prior to use",
    )
    parser.add_argument(
        "--informed_prior_ratio",
        type=false_or_float,
        default=0.5,
        help="For prior_type=hybrid_scm, probability of sampling informed subgroups.",
    )
    parser.add_argument("--mix_probs", type=float, nargs=2, default=None)
    parser.add_argument("--informed_mix_probs", type=float, nargs=2, default=None)
    parser.add_argument("--informed_block_allocation", type=float, nargs=5, default=None)
    parser.add_argument("--informed_task_family_probs", type=float, nargs=2, default=None)
    parser.add_argument("--informed_normal_block_allocation", type=float, nargs=9, default=None)
    parser.add_argument("--informed_inhibitor_block_allocation", type=float, nargs=9, default=None)
    parser.add_argument("--informed_normal_block_allocation_ranges", type=float, nargs=18, default=None)
    parser.add_argument("--informed_inhibitor_block_allocation_ranges", type=float, nargs=18, default=None)
    parser.add_argument("--informed_normal_block_allocation_min_counts", type=int, nargs=9, default=None)
    parser.add_argument("--informed_inhibitor_block_allocation_min_counts", type=int, nargs=9, default=None)
    parser.add_argument("--informed_feature_block_strength", type=false_or_float, default=None)
    parser.add_argument("--informed_interaction_strength", type=false_or_float, default=None)
    parser.add_argument("--informed_history_strength", type=false_or_float, default=None)
    parser.add_argument("--informed_intervention_strength", type=false_or_float, default=None)
    parser.add_argument(
        "--informed_target_family",
        type=str,
        default=None,
        choices=("generic_corrosion", "pitting_potential", "inhibitor_efficiency"),
    )
    parser.add_argument("--informed_physical_marginal_prob", type=false_or_float, default=None)
    parser.add_argument("--informed_physical_marginal_profile", type=str, default=None)
    parser.add_argument("--pitting_material_dirichlet_prob", type=false_or_float, default=None)
    parser.add_argument("--pitting_material_dirichlet_concentration", type=float, default=None)
    parser.add_argument("--pitting_material_dirichlet_active_prob", type=float, default=None)
    parser.add_argument(
        "--pitting_process_role",
        type=str,
        default=None,
        choices=(
            "test_method_category",
            "heat_treatment_category",
            "microstructure_category",
            "surface_process_score",
            "exposure_history_proxy",
        ),
    )
    parser.add_argument("--pitting_process_category_count", type=int, default=None)
    parser.add_argument("--pitting_fixed_epit_schema", default=None, type=str2bool)
    parser.add_argument("--cat_prob", type=float, default=None)
    parser.add_argument("--permute_features", default=None, type=str2bool)
    parser.add_argument("--epit_material_coef", type=float, default=None)
    parser.add_argument("--epit_environment_coef", type=float, default=None)
    parser.add_argument("--epit_interaction_coef", type=float, default=None)
    parser.add_argument("--inhibitor_descriptor_coef_scale", type=float, default=None)
    parser.add_argument("--inhibitor_context_coef_scale", type=float, default=None)
    parser.add_argument("--inhibitor_environment_interaction_coef_scale", type=float, default=None)
    parser.add_argument("--inhibitor_material_interaction_coef_scale", type=float, default=None)
    parser.add_argument("--inhibitor_intervention_coef_scale", type=float, default=None)
    parser.add_argument("--inhibitor_descriptor_coef", type=float, default=None)
    parser.add_argument("--inhibitor_context_coef", type=float, default=None)
    parser.add_argument("--inhibitor_environment_interaction_coef", type=float, default=None)
    parser.add_argument("--inhibitor_material_interaction_coef", type=float, default=None)
    parser.add_argument("--inhibitor_intervention_coef", type=float, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for generation"
    )

    args = parser.parse_args()
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)
    saver = SavePriorDataset(args)
    saver.run()
