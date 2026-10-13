"""EPIT/Soccol/generic prior mixture for the 50-column corrosion schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import Prior, SCMPrior
from .epit_schema import EPIT_COMPOSITION_ELEMENTS
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from .soccol_schema import SOCCOL_FEATURE_COLUMNS
from .source_ids import (
    MISSINGNESS_SOURCE_EPIT,
    MISSINGNESS_SOURCE_GENERIC,
    MISSINGNESS_SOURCE_SOCCOL,
)


COMBINED_COMPOSITION_COLUMNS = (
    "B", "C", "N", "Mg", "Al", "Si", "P", "S", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Y", "Nb", "Mo", "Ce", "Gd", "Ta", "W", "Re",
)
COMBINED_NUMERIC_COLUMNS = (
    "Prep_grinding_grit", "Prep_Ra_micron", "Prep_pH", "Prep_redox", "Prep_time",
    "CP_time", "CP_temp", "CP_pH", "Test_area_cm2", "scan_rate", "CP_Cl", "CP_Br",
    "CP_OH", "CP_SO4", "CP_CO3", "CP_NO3", "CP_PO4", "CP_MoO4", "CP_CrO4",
    "CP_ion_other",
)
COMBINED_CATEGORICAL_COLUMNS = (
    "Prep_medium", "CP_aeration", "CP_agitation", "CP_anions_info", "test_method",
    "reference_type",
)
COMBINED_FEATURE_COLUMNS = (
    *COMBINED_COMPOSITION_COLUMNS,
    *COMBINED_NUMERIC_COLUMNS,
    *COMBINED_CATEGORICAL_COLUMNS,
)
EPIT_FEATURE_COLUMNS = (
    *EPIT_COMPOSITION_ELEMENTS,
    "CP_temp",
    "CP_Cl",
    "CP_pH",
    "test_method",
)
COMBINED_FEATURE_INDEX = {
    name: index for index, name in enumerate(COMBINED_FEATURE_COLUMNS)
}


class CombinedCorrosionPriorDataset(IterableDataset):
    """Mix frozen generic, EPIT-informed, and Soccol-informed prior regimes."""

    REGIME_ORDER = ("generic", "EPIT", "Soccol")

    def __init__(
        self,
        *,
        config_path: str | Path,
        batch_size: int,
        batch_size_per_gp: int,
        max_classes: int,
        min_seq_len: int | None,
        max_seq_len: int,
        min_train_size: int | float,
        max_train_size: int | float,
        n_jobs: int,
        device: str,
        seed_offset: int = 0,
    ) -> None:
        super().__init__()
        if max_classes != 0:
            raise ValueError("The combined corrosion prior supports regression only.")
        if min_seq_len is not None:
            raise ValueError(
                "The combined corrosion prior requires --min_seq_len to remain unset "
                "so all source regimes use the same fixed sequence length."
            )

        self.config_path = Path(config_path).resolve()
        config_bytes = self.config_path.read_bytes()
        config = json.loads(config_bytes)
        if config.get("schema_version") != "combined_corrosion_prior_v1":
            raise ValueError("Unsupported combined corrosion prior configuration.")
        if tuple(config.get("feature_columns_in_order", ())) != COMBINED_FEATURE_COLUMNS:
            raise ValueError("Combined corrosion prior feature order does not match the 50-column schema.")

        regimes = config.get("regimes", {})
        if set(regimes) != set(self.REGIME_ORDER):
            raise ValueError(f"Combined corrosion prior must define {self.REGIME_ORDER}.")
        probabilities = np.asarray(
            [float(regimes[name]["probability"]) for name in self.REGIME_ORDER],
            dtype=float,
        )
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("Combined corrosion regime probabilities must be finite and non-negative.")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Combined corrosion regime probabilities must sum to one.")

        self.batch_size = int(batch_size)
        self.batch_size_per_gp = int(batch_size_per_gp)
        self.max_features = len(COMBINED_FEATURE_COLUMNS)
        self.max_seq_len = int(max_seq_len)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.probabilities = probabilities
        self.regimes = regimes
        self.priors: dict[str, SCMPrior] = {}

        for name in self.REGIME_ORDER:
            regime = regimes[name]
            num_features = int(regime["num_features"])
            fixed_hp = dict(DEFAULT_FIXED_HP)
            fixed_hp.update(regime.get("fixed_hp", {}))
            if float(fixed_hp.get("pitting_coefficient_variation", 0.0)):
                fixed_hp["pitting_coefficient_variation_seed"] = (
                    int(fixed_hp.get("pitting_coefficient_variation_seed", 42))
                    + int(seed_offset)
                )
            self.priors[name] = SCMPrior(
                batch_size=max(1, self.batch_size),
                batch_size_per_gp=self.batch_size_per_gp,
                min_features=num_features,
                max_features=num_features,
                max_classes=0,
                min_seq_len=None,
                max_seq_len=self.max_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                prior_type=str(regime["prior_type"]),
                fixed_hp=fixed_hp,
                sampled_hp=DEFAULT_SAMPLED_HP,
                n_jobs=n_jobs,
                informed_prior_ratio=1.0 if name != "generic" else 0.0,
                return_source_ids=False,
                device=device,
            )

        self.provenance = {
            "config_path": str(self.config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "schema_version": config["schema_version"],
            "feature_count": self.max_features,
            "seed_offset": int(seed_offset),
            "regime_probabilities": {
                name: float(probability)
                for name, probability in zip(self.REGIME_ORDER, probabilities, strict=True)
            },
            "source_commands": config.get("source_commands", {}),
        }

    def _batch_counts(self, batch_size: int) -> dict[str, int]:
        expected = self.probabilities * batch_size
        counts = np.floor(expected).astype(int)
        remainder = batch_size - int(counts.sum())
        if remainder:
            order = np.argsort(-(expected - counts), kind="stable")
            counts[order[:remainder]] += 1
        return dict(zip(self.REGIME_ORDER, counts.tolist(), strict=True))

    @staticmethod
    def _map_named_features(X: Tensor, source_columns: tuple[str, ...]) -> Tensor:
        if X.shape[-1] != len(source_columns):
            raise ValueError("Source prior width does not match its frozen feature schema.")
        mapped = torch.full(
            (*X.shape[:-1], len(COMBINED_FEATURE_COLUMNS)),
            torch.nan,
            dtype=X.dtype,
            device=X.device,
        )
        for source_index, name in enumerate(source_columns):
            mapped[..., COMBINED_FEATURE_INDEX[name]] = X[..., source_index]
        return mapped

    def _map_regime(self, name: str, X: Tensor) -> Tensor:
        if name == "generic":
            if X.shape[-1] != self.max_features:
                raise ValueError("Generic combined-corrosion tables must contain 50 features.")
            return X
        if name == "EPIT":
            return self._map_named_features(X, EPIT_FEATURE_COLUMNS)
        return self._map_named_features(X, SOCCOL_FEATURE_COLUMNS)

    def get_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.batch_size)
        counts = self._batch_counts(batch_size)
        X_parts: list[Tensor] = []
        y_parts: list[Tensor] = []
        source_parts: list[Tensor] = []
        source_id = {
            "generic": MISSINGNESS_SOURCE_GENERIC,
            "EPIT": MISSINGNESS_SOURCE_EPIT,
            "Soccol": MISSINGNESS_SOURCE_SOCCOL,
        }

        for name in self.REGIME_ORDER:
            count = counts[name]
            if count == 0:
                continue
            X, y, _, _, _ = self.priors[name].get_batch(count)
            X_parts.append(self._map_regime(name, X))
            y_parts.append(y)
            source_parts.append(
                torch.full((count,), source_id[name], dtype=torch.long, device=self.device)
            )

        X = torch.cat(X_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        source_ids = torch.cat(source_parts, dim=0)
        permutation = torch.randperm(batch_size, device=X.device)
        X = X[permutation]
        y = y[permutation]
        source_ids = source_ids[permutation]
        d = torch.full((batch_size,), self.max_features, dtype=torch.long, device=self.device)
        seq_lens = torch.full((batch_size,), self.max_seq_len, dtype=torch.long, device=self.device)
        train_size = Prior.sample_train_size(
            self.min_train_size,
            self.max_train_size,
            self.max_seq_len,
        )
        train_sizes = torch.full((batch_size,), train_size, dtype=torch.long, device=self.device)
        return X, y, d, seq_lens, train_sizes, source_ids

    def __iter__(self) -> "CombinedCorrosionPriorDataset":
        return self

    def __next__(self):
        return self.get_batch()

    def __repr__(self) -> str:
        probabilities = ", ".join(
            f"{name}={probability:.2f}"
            for name, probability in zip(self.REGIME_ORDER, self.probabilities, strict=True)
        )
        return (
            "CombinedCorrosionPriorDataset(\n"
            f"  regimes: {probabilities}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  features: {self.max_features}\n"
            f"  seq_len: {self.max_seq_len}\n"
            f"  device: {self.device}\n"
            ")"
        )

