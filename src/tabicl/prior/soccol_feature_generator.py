"""Sample complete 37-column Soccol physical inputs for synthetic training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .soccol_feature_profile import SoccolFeatureProfile, load_soccol_feature_profile
from .soccol_schema import (
    SOCCOL_BASE_FEATURE_COUNT,
    SOCCOL_COMPOSITION_COLUMNS,
    SOCCOL_FEATURE_PROFILE,
    soccol_feature_slices,
)


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class SoccolCompositionBatch:
    observed_compositions: np.ndarray
    template_indices: np.ndarray
    template_ids: tuple[str, ...]


@dataclass(frozen=True)
class SoccolFeatureBatch:
    profile_name: str
    compositions: SoccolCompositionBatch
    composition_family_indices: np.ndarray
    context_indices: np.ndarray
    features: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])


def sample_soccol_feature_rows(
    n_samples: int,
    *,
    profile: SoccolFeatureProfile | None = None,
    perturb_strength: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> SoccolFeatureBatch:
    """Sample composition and non-composition rows independently as before."""
    if int(n_samples) != n_samples or int(n_samples) <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not np.isfinite(perturb_strength) or float(perturb_strength) < 0.0:
        raise ValueError("perturb_strength must be finite and non-negative.")
    profile = load_soccol_feature_profile() if profile is None else profile
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    count = int(n_samples)
    composition_indices = rng.integers(0, profile.n_rows, size=count, dtype=np.int64)
    context_indices = rng.integers(0, profile.n_rows, size=count, dtype=np.int64)
    slices = soccol_feature_slices()
    compositions = profile.feature_values[composition_indices, slices["material"]].copy()
    if float(perturb_strength) > 0.0:
        noise = rng.normal(0.0, float(perturb_strength), size=compositions.shape)
        compositions *= np.where(compositions > 0.0, np.exp(noise), 1.0)
    totals = compositions.sum(axis=1, keepdims=True)
    if np.any(totals < 0.0) or not np.isfinite(totals).all():
        raise ValueError("Sampled Soccol compositions have invalid total mass.")
    nonzero = totals[:, 0] > 0.0
    compositions[nonzero] = 100.0 * compositions[nonzero] / totals[nonzero]

    features = profile.feature_values[context_indices].copy()
    features[:, slices["material"]] = compositions
    if features.shape != (count, SOCCOL_BASE_FEATURE_COUNT) or not np.isfinite(features).all():
        raise ValueError(f"Sampled Soccol feature rows are invalid: {features.shape}")
    composition_batch = SoccolCompositionBatch(
        observed_compositions=_readonly(compositions),
        template_indices=_readonly(composition_indices.copy()),
        template_ids=tuple(profile.row_ids[index] for index in composition_indices),
    )
    return SoccolFeatureBatch(
        profile_name=SOCCOL_FEATURE_PROFILE,
        compositions=composition_batch,
        composition_family_indices=_readonly(np.zeros(count, dtype=np.int64)),
        context_indices=_readonly(context_indices),
        features=_readonly(features),
    )


__all__ = [
    "SOCCOL_FEATURE_PROFILE",
    "SoccolCompositionBatch",
    "SoccolFeatureBatch",
    "sample_soccol_feature_rows",
]
