from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tabicl.prior.epit_composition_profile import (
    EPIT_COMPOSITION_DEFAULT_PERTURB_STRENGTH,
    EpitCompositionBatch,
    sample_epit_compositions,
)
from tabicl.prior.epit_feature_profile import (
    EPIT_FEATURE_PROFILE,
    EpitFeatureProfile,
    load_epit_feature_profile,
)


@dataclass(frozen=True)
class EpitFeatureBatch:
    """A complete physical 21-column EPIT feature batch."""

    profile_name: str
    compositions: EpitCompositionBatch
    composition_family_indices: np.ndarray
    environment_indices: np.ndarray
    environment_ids: tuple[str, ...]
    environment_values: np.ndarray
    test_method_codes: np.ndarray
    test_methods: tuple[str, ...]
    features: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _normalize_family_probabilities(
    profile: EpitFeatureProfile,
    probabilities: Sequence[float] | None,
) -> np.ndarray:
    if probabilities is None:
        values = np.bincount(
            profile.composition_family_indices,
            minlength=len(profile.families),
        ).astype(np.float64)
    else:
        values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(profile.families),):
        raise ValueError(
            f"Expected {len(profile.families)} EPIT feature-family probabilities, "
            f"got shape {values.shape}."
        )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("EPIT feature-family probabilities must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("At least one EPIT feature-family probability must be positive.")
    return values / total


def sample_epit_feature_rows(
    n_samples: int,
    *,
    profile: EpitFeatureProfile | None = None,
    composition_family_probabilities: Sequence[float] | None = None,
    perturb_strength: float = EPIT_COMPOSITION_DEFAULT_PERTURB_STRENGTH,
    random_state: int | np.random.Generator | None = None,
) -> EpitFeatureBatch:
    """Sample complete Fe/Ni--Cr--Mo physical inputs for the fixed EPIT schema.

    Composition templates and environment records are sampled independently.
    Each environment draw keeps temperature, chloride, pH, and test method from
    one observed row. Missing numeric environment values use the exact means
    recorded from the direct evaluator in the versioned profile.
    """

    if int(n_samples) != n_samples or int(n_samples) <= 0:
        raise ValueError("n_samples must be a positive integer.")
    profile = load_epit_feature_profile() if profile is None else profile
    family_probabilities = _normalize_family_probabilities(
        profile,
        composition_family_probabilities,
    )
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    full_family_probabilities = np.zeros(
        len(profile.composition_profile.families),
        dtype=np.float64,
    )
    for family_index, family in enumerate(profile.families):
        composition_family_index = profile.composition_profile.families.index(family)
        full_family_probabilities[composition_family_index] = family_probabilities[family_index]

    compositions = sample_epit_compositions(
        int(n_samples),
        profile=profile.composition_profile,
        family_probabilities=full_family_probabilities,
        perturb_strength=perturb_strength,
        random_state=rng,
    )
    family_to_index = {family: index for index, family in enumerate(profile.families)}
    composition_family_indices = np.asarray(
        [family_to_index[family] for family in compositions.family_names],
        dtype=np.int64,
    )

    environment_indices = rng.integers(
        0,
        profile.n_environment_rows,
        size=int(n_samples),
        dtype=np.int64,
    )
    environment_values = profile.environment_imputed_values[environment_indices].copy()
    test_method_codes = profile.test_method_codes[environment_indices].copy()
    features = np.column_stack(
        (
            compositions.observed_compositions,
            environment_values,
            test_method_codes.astype(np.float64),
        )
    ).astype(np.float64, copy=False)
    if features.shape != (int(n_samples), 21) or not np.isfinite(features).all():
        raise ValueError(f"Sampled EPIT feature rows have invalid shape or values: {features.shape}")

    return EpitFeatureBatch(
        profile_name=profile.name,
        compositions=compositions,
        composition_family_indices=_readonly(composition_family_indices),
        environment_indices=_readonly(environment_indices),
        environment_ids=tuple(profile.environment_ids[index] for index in environment_indices),
        environment_values=_readonly(environment_values),
        test_method_codes=_readonly(test_method_codes),
        test_methods=tuple(profile.test_methods[index] for index in environment_indices),
        features=_readonly(features),
    )


__all__ = [
    "EPIT_FEATURE_PROFILE",
    "EpitFeatureBatch",
    "sample_epit_feature_rows",
]
