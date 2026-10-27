"""Sample versioned Soccol composition and joint environment/history banks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .soccol_feature_profile import (
    SOCCOL_FEATURE_PROFILE, SoccolFeatureProfile, load_soccol_feature_profile,
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
    latent_features: np.ndarray | None = None

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
    """Draw a profile's composition template and one independent context row.

    Current v3 weights distinct compositions equally. Explicit v1/v2 profiles
    retain their original observation-weighted composition draws.
    Every non-composition field, including solution, preparation, heat treatment
    and missing values, shares the same context index. The finite internal copy
    uses that same row; it never triggers another environment draw.
    """
    if int(n_samples) != n_samples or int(n_samples) <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not np.isfinite(perturb_strength) or float(perturb_strength) < 0.0:
        raise ValueError("perturb_strength must be finite and non-negative.")
    profile = load_soccol_feature_profile() if profile is None else profile
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    count = int(n_samples)
    templates = profile.composition_template_indices
    if templates is None:
        templates = np.arange(profile.n_rows, dtype=np.int64)
    composition_indices = templates[rng.integers(0, len(templates), size=count, dtype=np.int64)]
    context_indices = rng.integers(0, profile.n_rows, size=count, dtype=np.int64)
    material = slice(0, profile.composition_count)
    compositions = profile.feature_values[composition_indices, material].copy()
    if float(perturb_strength) > 0.0:
        noise = rng.normal(0.0, float(perturb_strength), size=compositions.shape)
        compositions *= np.where(compositions > 0.0, np.exp(noise), 1.0)
    totals = compositions.sum(axis=1, keepdims=True)
    if np.any(totals < 0.0) or not np.isfinite(totals).all():
        raise ValueError("Sampled Soccol compositions have invalid total mass.")
    nonzero = totals[:, 0] > 0.0
    compositions[nonzero] = 100.0 * compositions[nonzero] / totals[nonzero]

    # One row selection preserves relationships across all context fields and
    # their missing values; replace only its independently drawn composition.
    features = profile.feature_values[context_indices].copy()
    features[:, material] = compositions
    latent_source = profile.feature_values if profile.latent_feature_values is None else profile.latent_feature_values
    latent = latent_source[context_indices].copy()
    latent[:, material] = compositions
    if features.shape != (count, profile.feature_values.shape[1]) or np.isinf(features).any() or not np.isfinite(latent).all():
        raise ValueError(f"Sampled Soccol feature rows are invalid: {features.shape}")
    composition_batch = SoccolCompositionBatch(
        observed_compositions=_readonly(compositions),
        template_indices=_readonly(composition_indices.copy()),
        template_ids=tuple(profile.row_ids[index] for index in composition_indices),
    )
    return SoccolFeatureBatch(
        profile_name=profile.name,
        compositions=composition_batch,
        composition_family_indices=_readonly(np.zeros(count, dtype=np.int64)),
        context_indices=_readonly(context_indices),
        features=_readonly(features),
        latent_features=_readonly(latent),
    )


__all__ = [
    "SOCCOL_FEATURE_PROFILE",
    "SoccolCompositionBatch",
    "SoccolFeatureBatch",
    "sample_soccol_feature_rows",
]
