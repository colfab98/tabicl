"""Structured missing-value augmentation for missing-aware TabICL training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from tabicl.prior.source_ids import MISSINGNESS_SOURCE_GENERIC, SOURCE_ID_BY_NAME


class MissingnessAugmenter:
    """Sample source-shaped feature masks from a frozen empirical profile."""

    def __init__(
        self,
        profile_path: str | Path,
        *,
        augmentation_probability: float = 0.8,
        min_observed_composition: int = 2,
    ) -> None:
        if not 0.0 < augmentation_probability <= 1.0:
            raise ValueError("augmentation_probability must be in (0, 1].")
        if min_observed_composition < 0:
            raise ValueError("min_observed_composition must be non-negative.")

        self.profile_path = Path(profile_path).resolve()
        profile_bytes = self.profile_path.read_bytes()
        profile = json.loads(profile_bytes)
        self.schema_version = str(profile["schema_version"])
        self.feature_columns = tuple(profile["feature_columns_in_order"])
        self.composition_feature_count = int(profile["composition_feature_count"])
        self.augmentation_probability = float(augmentation_probability)
        self.min_observed_composition = int(min_observed_composition)

        if not 0 <= self.composition_feature_count <= len(self.feature_columns):
            raise ValueError("Invalid composition_feature_count in missingness profile.")

        self.source_names: list[str] = []
        self.source_row_counts: list[int] = []
        self.pattern_masks: dict[str, Tensor] = {}
        self.pattern_counts: dict[str, Tensor] = {}
        for source, source_profile in profile["patterns_by_source"].items():
            patterns = source_profile["patterns"]
            masks = []
            counts = []
            for pattern in patterns:
                bitstring = str(pattern["missing_mask"])
                if len(bitstring) != len(self.feature_columns) or set(bitstring) - {"0", "1"}:
                    raise ValueError(f"Invalid missingness pattern for source {source!r}.")
                count = int(pattern["count"])
                if count <= 0:
                    raise ValueError(f"Missingness pattern count must be positive for source {source!r}.")
                masks.append([character == "1" for character in bitstring])
                counts.append(count)

            row_count = int(source_profile["row_count"])
            if sum(counts) != row_count:
                raise ValueError(f"Pattern counts do not sum to row_count for source {source!r}.")
            self.source_names.append(str(source))
            self.source_row_counts.append(row_count)
            self.pattern_masks[str(source)] = torch.tensor(masks, dtype=torch.bool)
            self.pattern_counts[str(source)] = torch.tensor(counts, dtype=torch.float64)

        if not self.source_names or sum(self.source_row_counts) <= 0:
            raise ValueError("Missingness profile contains no source patterns.")
        missing_sources = set(SOURCE_ID_BY_NAME) - set(self.source_names)
        if missing_sources:
            raise ValueError(
                "Missingness profile must define source patterns for "
                f"{sorted(missing_sources)}."
            )

        self.provenance: dict[str, Any] = {
            "profile_path": str(self.profile_path),
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "schema_version": self.schema_version,
            "feature_count": len(self.feature_columns),
            "composition_feature_count": self.composition_feature_count,
            "augmentation_probability": self.augmentation_probability,
            "min_observed_composition": self.min_observed_composition,
            "sources": dict(zip(self.source_names, self.source_row_counts, strict=True)),
            "source_routing": {
                "generic": "sampled from empirical source row counts",
                "informed": "matched to the informed EPIT or Soccol generator",
            },
        }

    def validate_max_features(self, max_features: int) -> None:
        if max_features > len(self.feature_columns):
            raise ValueError(
                f"Missingness profile defines {len(self.feature_columns)} features, "
                f"but training allows {max_features}. Set --max_features no higher than the profile width."
            )

    def _sample_mask(
        self,
        X: Tensor,
        generator: torch.Generator | None,
        source_ids: Tensor | None,
    ) -> Tensor:
        batch_size, seq_len, feature_count = X.shape
        device = X.device
        source_weights = torch.tensor(self.source_row_counts, dtype=torch.float64, device=device)
        source_indices = torch.multinomial(
            source_weights,
            batch_size,
            replacement=True,
            generator=generator,
        )
        if source_ids is not None:
            if source_ids.ndim != 1 or source_ids.shape[0] != batch_size:
                raise ValueError("source_ids must contain one source ID per table.")
            source_ids = source_ids.to(device=device, dtype=torch.long)
            valid_source_ids = {MISSINGNESS_SOURCE_GENERIC, *SOURCE_ID_BY_NAME.values()}
            if not set(source_ids.unique().tolist()).issubset(valid_source_ids):
                raise ValueError("source_ids contains an unsupported source ID.")
            for source_name, source_id in SOURCE_ID_BY_NAME.items():
                profile_index = self.source_names.index(source_name)
                source_indices = torch.where(
                    source_ids == source_id,
                    torch.full_like(source_indices, profile_index),
                    source_indices,
                )
        augment_tables = torch.rand(batch_size, device=device, generator=generator) < self.augmentation_probability
        mask = torch.zeros_like(X, dtype=torch.bool)

        for source_index, source in enumerate(self.source_names):
            table_indices = torch.nonzero(
                (source_indices == source_index) & augment_tables,
                as_tuple=False,
            ).flatten()
            if table_indices.numel() == 0:
                continue
            counts = self.pattern_counts[source].to(device=device)
            sampled = torch.multinomial(
                counts,
                int(table_indices.numel()) * seq_len,
                replacement=True,
                generator=generator,
            ).reshape(int(table_indices.numel()), seq_len)
            patterns = self.pattern_masks[source].to(device=device)[:, :feature_count]
            mask[table_indices] = patterns[sampled]
        return mask

    def _protect_composition(
        self,
        mask: Tensor,
        X: Tensor,
        d: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        composition_count = min(self.composition_feature_count, X.shape[-1])
        if composition_count == 0 or self.min_observed_composition == 0:
            return mask

        batch_size, seq_len, _ = X.shape
        active_composition_count = d.clamp(min=0, max=composition_count).long()
        active_composition = (
            torch.arange(composition_count, device=X.device).view(1, 1, -1)
            < active_composition_count.view(batch_size, 1, 1)
        ).expand(-1, seq_len, -1)
        original_missing = torch.isnan(X[..., :composition_count])
        composition_mask = mask[..., :composition_count]
        observed = (active_composition & ~(original_missing | composition_mask)).sum(dim=-1)
        required = active_composition_count.clamp(max=self.min_observed_composition).view(-1, 1)
        needed = (required - observed).clamp(min=0)
        candidates = composition_mask & active_composition & ~original_missing

        for _ in range(self.min_observed_composition):
            can_restore = (needed > 0) & candidates.any(dim=-1)
            if not torch.any(can_restore):
                break
            scores = torch.rand(
                candidates.shape,
                device=X.device,
                dtype=torch.float32,
                generator=generator,
            ).masked_fill(~candidates, -1.0)
            selected = F.one_hot(scores.argmax(dim=-1), num_classes=composition_count).bool()
            selected &= can_restore.unsqueeze(-1)
            composition_mask &= ~selected
            candidates &= ~selected
            needed -= can_restore.long()

        mask[..., :composition_count] = composition_mask
        return mask

    def augment(
        self,
        X: Tensor,
        d: Tensor,
        *,
        source_ids: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Apply sampled masks to X only; targets are never accepted or modified."""
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, rows, features).")
        if d.ndim != 1 or d.shape[0] != X.shape[0]:
            raise ValueError("d must contain one active-feature count per table.")
        if X.shape[-1] > len(self.feature_columns):
            raise ValueError("Training batch is wider than the missingness profile.")
        if not X.is_floating_point():
            raise TypeError("Missingness augmentation requires a floating-point feature tensor.")

        active = (
            torch.arange(X.shape[-1], device=X.device).view(1, 1, -1)
            < d.long().view(-1, 1, 1)
        ).expand(-1, X.shape[1], -1)
        original_missing = torch.isnan(X)
        mask = self._sample_mask(X, generator, source_ids) & active & ~original_missing
        mask = self._protect_composition(mask, X, d, generator)

        composition_count = min(self.composition_feature_count, X.shape[-1])
        composition_active = active[..., :composition_count]
        other_active = active[..., composition_count:]
        stats = {
            "missing_mask_fraction": float(mask.sum().item() / max(1, active.sum().item())),
            "composition_missing_mask_fraction": float(
                mask[..., :composition_count].sum().item() / max(1, composition_active.sum().item())
            ),
            "other_missing_mask_fraction": float(
                mask[..., composition_count:].sum().item() / max(1, other_active.sum().item())
            ),
        }
        return X.masked_fill(mask, torch.nan), stats
