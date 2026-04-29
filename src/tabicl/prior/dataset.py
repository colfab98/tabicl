"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np
from scipy.stats import loguniform
import joblib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset, get_worker_info

from .mlp_scm import MLPSCM
from .tree_scm import TreeSCM

from .hp_sampling import HpSamplerList
from .reg2cls import Reg2Cls
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class Prior:
    """Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    min_features : int, default=2
        Minimum number of features per dataset.

    max_features : int, default=100
        Maximum number of features per dataset.

    max_classes : int, default=10
        Maximum number of target classes.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
    ):
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
        """Check if the training size range is valid.

        Parameters
        ----------
        min_train_size : int or float
            Minimum training size (position or ratio).

        max_train_size : int or float
            Maximum training size (position or ratio).

        Raises
        ------
        AssertionError
            If training size range is invalid.

        ValueError
            If training size types are mismatched or invalid.
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert 0 < min_train_size < max_train_size < 1, "0 < min_train_size < max_train_size < 1"
        else:
            raise ValueError("Both training sizes must be of the same type (int or float)")

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False
    ) -> int:
        """Select a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length.

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes.

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets.

        Returns
        -------
        int
            The sampled sequence length.
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(min_train_size: Union[int, float], max_train_size: Union[int, float], seq_len: int) -> int:
        """Select a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int or float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int or float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length.

        Returns
        -------
        int
            The sampled training size position.

        Raises
        ------
        ValueError
            If training size range has incompatible types.
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """Adjust the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples).

        max_features : int
            Original maximum number of features.

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility.
        """
        if seq_len <= 10240:
            return min(100, max_features)
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """Remove features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape ``(B, T, H)`` where B is batch size,
            T is sequence length, and H is feature dimensionality.

        d : Tensor
            Number of features per dataset of shape ``(B,)``, indicating how many
            features are actually used in each dataset (rest is padding).

        Returns
        -------
        X_new : Tensor
            The filtered tensor with non-informative features removed.

        d_new : Tensor
            The updated feature count per dataset.
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        return X_new, d_new

    @staticmethod
    def sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2) -> bool:
        """Verify that both train and test sets contain all classes.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape ``(B, T, H)``.

        y : Tensor
            Target labels tensor of shape ``(B, T)``.

        train_size : int
            Position to split the data into train and test sets.

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits.

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets.

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise.
        """

        def is_valid_split(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True


class SCMPrior(Prior):
    """Generate synthetic datasets using Structural Causal Models (SCM).

    The data generation process follows a hierarchical structure:

    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary
       transformations and checks.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics.

    batch_size_per_subgp : int, optional
        Number of datasets per subgroup, with more similar causal structures.
        If None, defaults to batch_size_per_gp.

    min_features : int, default=2
        Minimum number of features per dataset.

    max_features : int, default=100
        Maximum number of features per dataset.

    max_classes : int, default=10
        Maximum number of target classes.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        If True, sample sequence length per group, allowing variable-sized datasets.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', or 'mix_scm'.
        'mix_scm' randomly selects between 'mlp_scm' and 'tree_scm' based on
        probabilities.

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters.

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation.

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation.

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        informed_prior_ratio: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.prior_type = prior_type
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        if not 0.0 <= informed_prior_ratio <= 1.0:
            raise ValueError("informed_prior_ratio must be in [0, 1].")
        self.informed_prior_ratio = informed_prior_ratio
        self.device = device

    def hp_sampling(self) -> Dict[str, Any]:
        """Sample hyperparameters for dataset generation.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones.
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device)
        return hp_sampler.sample()

    @staticmethod
    def _normalize_two_way_probs(probs: Any, fallback: Tuple[float, float]) -> np.ndarray:
        """Normalize and validate a two-way probability vector."""
        raw = probs if probs is not None else fallback
        values = np.array(raw, dtype=float).reshape(-1)
        if values.size != 2:
            values = np.array(fallback, dtype=float)
        values = np.clip(values, a_min=0.0, a_max=None)
        total = values.sum()
        if total <= 0:
            values = np.array(fallback, dtype=float)
            total = values.sum()
        return values / total

    def _get_prior_mix_probs(self, informed: bool = False) -> np.ndarray:
        """Return sampling probabilities for (mlp_scm, tree_scm)."""
        if informed:
            return self._normalize_two_way_probs(self.fixed_hp.get("informed_mix_probs"), fallback=(0.9, 0.1))
        return self._normalize_two_way_probs(self.fixed_hp.get("mix_probs"), fallback=(0.7, 0.3))

    @staticmethod
    def _rank_uniform(values: Tensor) -> Tensor:
        """Map a latent feature column to empirical quantiles in (0, 1)."""
        if values.numel() <= 1:
            return torch.full_like(values, 0.5)
        clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        order = torch.argsort(clean)
        ranks = torch.empty_like(clean)
        ranks[order] = torch.arange(1, clean.numel() + 1, device=clean.device, dtype=clean.dtype)
        return ranks / float(clean.numel() + 1)

    @staticmethod
    def _log_uniform_from_rank(u: Tensor, low: float, high: float) -> Tensor:
        return torch.exp(math.log(low) + u * (math.log(high) - math.log(low)))

    @staticmethod
    def _piecewise_ph_from_rank(u: Tensor) -> Tensor:
        acidic = (u / 0.20).clamp(0.0, 1.0) * 6.0
        near_neutral = 6.0 + ((u - 0.20) / 0.60).clamp(0.0, 1.0) * 3.0
        alkaline = 9.0 + ((u - 0.80) / 0.20).clamp(0.0, 1.0) * 5.0
        return torch.where(u < 0.20, acidic, torch.where(u < 0.80, near_neutral, alkaline))

    @classmethod
    def _corrosion_marginal_values(cls, values: Tensor, family: str) -> Tensor:
        """Transform one latent column into a broad corrosion-like marginal family."""
        u = cls._rank_uniform(values).clamp(1e-6, 1.0 - 1e-6)

        if family == "ph":
            return cls._piecewise_ph_from_rank(u)
        if family == "chloride":
            return cls._log_uniform_from_rank(u, 1e-3, 1e5)
        if family == "salinity":
            return cls._log_uniform_from_rank(u, 1e-3, 3.5e1)
        if family == "temperature":
            return -10.0 + 130.0 * u
        if family == "potential":
            return -1.5 + 3.0 * u
        if family == "current_density":
            return cls._log_uniform_from_rank(u, 1e-9, 1e-1)
        if family == "resistance":
            return cls._log_uniform_from_rank(u, 1e-2, 1e6)
        if family == "exposure_time":
            return cls._log_uniform_from_rank(u, 1e-2, 1e5)
        if family == "cycle_count":
            return torch.floor(cls._log_uniform_from_rank(u, 1.0, 1e6))
        if family == "material_fraction":
            return 100.0 * u
        if family == "material_property":
            return cls._log_uniform_from_rank(u, 1e-2, 1e3)
        if family == "prior_damage":
            return u
        if family == "intervention_binary":
            return (u > 0.5).to(dtype=values.dtype)
        if family == "intervention_category":
            n_categories = int(np.random.choice([2, 3, 4]))
            return torch.floor(u * n_categories).clamp(max=n_categories - 1)
        if family == "intervention_dose":
            return u

        raise ValueError(f"Unknown corrosion marginal family: {family}")

    @staticmethod
    def _apply_composition_marginal(X: Tensor, feature_slice: slice) -> None:
        width = feature_slice.stop - feature_slice.start
        if width <= 1:
            return
        logits = torch.nan_to_num(X[:, feature_slice], nan=0.0, posinf=0.0, neginf=0.0)
        sharpness = float(np.random.uniform(0.6, 1.8))
        X[:, feature_slice] = torch.softmax(logits * sharpness, dim=-1) * 100.0

    def apply_informed_physical_marginals(self, X: Tensor, blocks: Dict[str, slice]) -> Tensor:
        """Apply broad physical marginal shapes to informed corrosion-like feature blocks."""
        marginal_prob = float(self.fixed_hp.get("informed_physical_marginal_prob", 0.0))
        marginal_prob = float(np.clip(marginal_prob, 0.0, 1.0))
        if marginal_prob <= 0.0 or np.random.random() >= marginal_prob:
            return X

        profile = str(self.fixed_hp.get("informed_physical_marginal_profile", "corrosion_broad")).lower()
        if profile in {"false", "none", "off", "disabled"}:
            return X
        if profile not in {"corrosion_broad", "broad", "corrosion"}:
            raise ValueError(f"Unknown informed physical marginal profile: {profile}")

        X = X.clone()

        material_slice = blocks.get("material")
        if material_slice is not None and material_slice.stop > material_slice.start:
            start = material_slice.start
            if material_slice.stop - material_slice.start >= 2 and np.random.random() < 0.70:
                width = material_slice.stop - material_slice.start
                comp_width = min(width, max(2, int(round(width * np.random.uniform(0.50, 1.00)))))
                self._apply_composition_marginal(X, slice(start, start + comp_width))
                start += comp_width
            for col in range(start, material_slice.stop):
                family = str(np.random.choice(["material_fraction", "material_property"]))
                X[:, col] = self._corrosion_marginal_values(X[:, col], family)

        block_families = {
            "environment": ["ph", "chloride", "salinity", "temperature"],
            "electrochem": ["potential", "current_density", "resistance"],
            "history": ["exposure_time", "cycle_count", "prior_damage"],
            "intervention": ["intervention_binary", "intervention_category", "intervention_dose"],
        }
        for block_name, families in block_families.items():
            feature_slice = blocks.get(block_name)
            if feature_slice is None or feature_slice.stop <= feature_slice.start:
                continue
            for col in range(feature_slice.start, feature_slice.stop):
                family = str(np.random.choice(families))
                X[:, col] = self._corrosion_marginal_values(X[:, col], family)

        return X

    def _split_informed_blocks(self, num_features: int) -> Dict[str, slice]:
        """Split features into coarse domain blocks."""
        names = ["material", "environment", "electrochem", "history", "intervention"]
        if num_features <= 0:
            return {}

        allocation = np.asarray(
            self.fixed_hp.get("informed_block_allocation", (0.70, 0.18, 0.10, 0.02, 0.0)), dtype=float
        )
        if allocation.shape != (len(names),):
            raise ValueError(
                "informed_block_allocation must contain five weights in "
                "material, environment, electrochem, history, intervention order."
            )
        if not np.all(np.isfinite(allocation)) or np.any(allocation < 0.0):
            raise ValueError("informed_block_allocation weights must be finite and non-negative.")
        if allocation.sum() <= 0.0:
            raise ValueError("At least one informed_block_allocation weight must be positive.")

        weights = allocation / allocation.sum()
        raw_counts = weights * num_features
        counts = np.floor(raw_counts).astype(int)
        counts[weights <= 0.0] = 0

        remaining = int(num_features - counts.sum())
        if remaining > 0:
            for idx in np.argsort(-(raw_counts - counts)):
                if weights[idx] <= 0.0:
                    continue
                counts[idx] += 1
                remaining -= 1
                if remaining == 0:
                    break
        while remaining > 0:
            idx = int(np.argmax(weights))
            counts[idx] += 1
            remaining -= 1

        blocks: Dict[str, slice] = {}
        start = 0
        for name, width in zip(names, counts.tolist()):
            end = min(num_features, start + width)
            if end > start:
                blocks[name] = slice(start, end)
            start = end
            if start >= num_features:
                break
        return blocks

    def apply_informed_structure(self, X: Tensor, y: Tensor, params: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """Inject block-wise and interaction structure for informed synthetic priors."""
        num_features = int(params["num_features"])
        if num_features <= 1:
            return X, y

        blocks = self._split_informed_blocks(num_features)
        if not blocks:
            return X, y

        X = X.clone()
        y = y.clone()

        block_strength = float(self.fixed_hp.get("informed_feature_block_strength", 0.30))
        block_strength = float(np.clip(block_strength, 0.0, 0.95))
        interaction_strength = float(self.fixed_hp.get("informed_interaction_strength", 0.35))
        history_strength = float(self.fixed_hp.get("informed_history_strength", 0.70))
        history_strength = float(np.clip(history_strength, 0.0, 0.99))
        intervention_strength = float(self.fixed_hp.get("informed_intervention_strength", 0.20))

        # Features in the same block share a latent component.
        for feature_slice in blocks.values():
            if feature_slice.stop - feature_slice.start <= 1:
                continue
            shared = torch.randn(X.shape[0], 1, device=X.device, dtype=X.dtype)
            X[:, feature_slice] = (1.0 - block_strength) * X[:, feature_slice] + block_strength * shared

        def block_mean(name: str) -> Optional[Tensor]:
            feature_slice = blocks.get(name)
            if feature_slice is None or feature_slice.stop <= feature_slice.start:
                return None
            return X[:, feature_slice].mean(dim=-1, keepdim=True)

        material = block_mean("material")
        environment = block_mean("environment")
        electrochem = block_mean("electrochem")
        history = block_mean("history")
        intervention = block_mean("intervention")

        # Coupled material x environment signal.
        if material is not None and environment is not None:
            coupled = torch.tanh(material * environment)
            y = y + interaction_strength * coupled.squeeze(-1)
            electro_slice = blocks.get("electrochem")
            if electro_slice is not None and electro_slice.stop > electro_slice.start:
                X[:, electro_slice] = X[:, electro_slice] + 0.5 * interaction_strength * coupled

        # Path dependence via autoregressive history features.
        history_slice = blocks.get("history")
        if history_slice is not None and history_slice.stop > history_slice.start and X.shape[0] > 2:
            hist = X[:, history_slice].clone()
            for t in range(1, hist.shape[0]):
                hist[t] = history_strength * hist[t - 1] + (1.0 - history_strength) * hist[t]
            X[:, history_slice] = hist
            y = y + 0.2 * hist.mean(dim=-1)

        # Interventions reduce damage conditionally on environment severity.
        if intervention is not None and environment is not None:
            intervention_gate = torch.sigmoid(intervention).squeeze(-1)
            env_severity = torch.relu(environment).squeeze(-1)
            y = y - intervention_strength * intervention_gate * env_severity

        X = self.apply_informed_physical_marginals(X, blocks)

        return torch.nan_to_num(X), torch.nan_to_num(y)

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        X : Tensor
            Features tensor of shape ``(seq_len, max_features)``.

        y : Tensor
            Labels tensor of shape ``(seq_len,)``.

        d : Tensor
            Number of active features after filtering (scalar Tensor).
        """

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        while True:
            X, y = prior_cls(**params)()
            if params.get("informed_mode", False):
                X, y = self.apply_informed_structure(X, y, params)
            X, y = Reg2Cls(params)(X, y)

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            # Only keep valid datasets with sufficient features and balanced classes
            X, d = self.delete_unique_features(X, d)
            if (d > 0).all() and self.sanity_check(X, y, params["train_size"]):
                return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size.

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is
            ``(batch_size, seq_len, max_features)``.
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Labels tensor. If seq_len_per_gp=False, shape is ``(batch_size, seq_len)``.
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape ``(batch_size,)``.

        seq_lens : Tensor
            Sequence length for each dataset, shape ``(batch_size,)``.

        train_sizes : Tensor
            Position for train/test split for each dataset, shape ``(batch_size,)``.
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
            )
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
                )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type, subgp_informed_mode = self.get_prior()
                subgp_num_features = round(np.random.uniform(self.min_features, gp_max_features))
                subgp_sampled_hp = {k: v() if callable(v) else v for k, v in group_sampled_hp.items()}

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    # Each dataset has its own number of classes
                    if np.random.random() > 0.5:
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)
                    else:
                        ds_num_classes = 2

                    # Create parameters dictionary for this dataset
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "informed_mode": subgp_informed_mode,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                    }
                    param_list.append(params)

        # Use joblib to generate datasets in parallel.
        # When PriorDataset runs inside a DataLoader worker, we're already in a multiprocessing
        # context, so loky would be downgraded to n_jobs=1. In that nested case we switch to the
        # threading backend to preserve on-the-fly parallelism and DataLoader overlap.
        if self.n_jobs > 1 and self.device == "cpu":
            worker_info = get_worker_info()
            backend = "threading" if worker_info is not None else "loky"
            parallel_config = {"n_jobs": self.n_jobs, "backend": backend}
            if backend == "loky":
                parallel_config["inner_max_num_threads"] = self.num_threads_per_generate

            with joblib.parallel_config(**parallel_config):
                results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)
        else:
            results = [self.generate_dataset(params) for params in param_list]

        X_list, y_list, d_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

        return X, y, d, seq_lens, train_sizes

    def get_prior(self) -> Tuple[str, bool]:
        """Determine which base prior to use and whether informed structure is enabled.

        Returns
        -------
        tuple[str, bool]
            (selected prior type, informed_mode)
        """
        if self.prior_type == "mix_scm":
            return np.random.choice(["mlp_scm", "tree_scm"], p=self._get_prior_mix_probs(informed=False)), False
        if self.prior_type == "informed_scm":
            return np.random.choice(["mlp_scm", "tree_scm"], p=self._get_prior_mix_probs(informed=True)), True
        if self.prior_type == "hybrid_scm":
            informed_mode = bool(np.random.random() < self.informed_prior_ratio)
            probs = self._get_prior_mix_probs(informed=informed_mode)
            return np.random.choice(["mlp_scm", "tree_scm"], p=probs), informed_mode
        return self.prior_type, False


class DummyPrior(Prior):
    """Create purely random data for testing and debugging.

    This is useful for testing and debugging without the computational overhead
    of SCM-based generation.

    Parameters
    ----------
    batch_size : int, default=256
        Number of datasets to generate.

    min_features : int, default=2
        Minimum number of features per dataset.

    max_features : int, default=100
        Maximum number of features per dataset.

    max_classes : int, default=10
        Maximum number of target classes.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    device : str, default="cpu"
        Computation device.
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size.

        Returns
        -------
        X : Tensor
            Features tensor of shape ``(batch_size, seq_len, max_features)``.
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape ``(batch_size, seq_len)``.
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape ``(batch_size,)``.
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape ``(batch_size,)``.
            All datasets share the same sequence length.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape ``(batch_size,)``.
            All datasets share the same split position.
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
        train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

        num_classes = np.random.randint(2, self.max_classes + 1)
        y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)
        train_sizes = torch.full((batch_size,), train_size, device=self.device)

        return X, y, d, seq_lens, train_sizes


class PriorDataset(IterableDataset):
    """Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics.

    batch_size_per_subgp : int, optional
        Number of datasets per subgroup, with more similar causal structures.
        If None, defaults to batch_size_per_gp.

    min_features : int, default=2
        Minimum number of features per dataset.

    max_features : int, default=100
        Maximum number of features per dataset.

    max_classes : int, default=10
        Maximum number of target classes.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        If True, sample sequence length per group, allowing variable-sized datasets.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm',
        'informed_scm', 'hybrid_scm', or 'dummy'.

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models
         - 'informed_scm': SCM + informed block/coupling structure
         - 'hybrid_scm': Mix of generic and informed SCM samples

        2. Dummy: Randomly generated datasets for debugging

    informed_prior_ratio : float, default=0.5
        For 'hybrid_scm', probability of sampling an informed subgroup.

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors.

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation.

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation.

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        informed_prior_ratio: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__()
        if prior_type == "dummy":
            self.prior = DummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
            )
        elif prior_type in ["mlp_scm", "tree_scm", "mix_scm", "informed_scm", "hybrid_scm"]:
            self.prior = SCMPrior(
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                seq_len_per_gp=seq_len_per_gp,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                prior_type=prior_type,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                informed_prior_ratio=informed_prior_ratio,
                device=device,
            )
        else:
            raise ValueError(
                "Unknown prior type "
                f"'{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', "
                "'informed_scm', 'hybrid_scm', or 'dummy'."
            )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call.

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len, max_features)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of
            ``(batch_size, seq_len, max_features)``.

        y : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of ``(batch_size, seq_len)``.

        d : Tensor
            Number of active features per dataset of shape ``(batch_size,)``.

        seq_lens : Tensor
            Sequence length for each dataset of shape ``(batch_size,)``.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape ``(batch_size,)``.
        """
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> "PriorDataset":
        """Return an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator.
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return the next batch from the iterator.

        Since this is an infinite iterator, it never raises StopIteration
        and instead continuously generates new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters.
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout
