"""Opt-in coefficient sampling; disabled by default in training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CoefficientDraw:
    coefficients: dict[str, float]
    attempts: int
    rejected_draws: int
    used_fallback: bool


def sample_coefficients(
    coefficients: Mapping[str, float],
    variation: float,
    rng: np.random.Generator,
    *,
    upper_bounds: Mapping[str, float] | None = None,
    max_attempts: int = 100,
) -> CoefficientDraw:
    """Multiply by U(1-v, 1+v), normalize, and reject upper-bound violations.

    Zeros remain zero. ``variation`` is a multiplier range, not a variance;
    normalization and rejection can shift the mean. Zero variation returns
    an exact copy without consuming random numbers. Exhausted retries return
    the original valid coefficients and explicitly flag the fallback.
    """
    if not np.isfinite(variation) or not 0 <= variation < 1:
        raise ValueError("variation must be finite and in [0, 1).")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")
    names = list(coefficients)
    values = np.asarray([coefficients[name] for name in names], dtype=float)
    if not names or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Coefficients must be finite, nonnegative, and nonempty.")
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Coefficients must sum to one.")
    if upper_bounds is not None and set(upper_bounds) != set(names):
        raise ValueError("Upper bounds must have exactly the coefficient keys.")
    bounds = np.asarray(
        [upper_bounds[name] if upper_bounds is not None else 1.0 for name in names],
        dtype=float,
    )
    if not np.isfinite(bounds).all() or np.any(bounds < 0) or np.any(bounds > 1):
        raise ValueError("Upper bounds must be finite and in [0, 1].")
    if np.any(values > bounds):
        raise ValueError("Original coefficients exceed an upper bound.")
    original = dict(zip(names, map(float, values)))
    if variation == 0:
        return CoefficientDraw(original, 0, 0, False)
    for attempt in range(1, max_attempts + 1):
        sampled = values * rng.uniform(1 - variation, 1 + variation, size=len(values))
        sampled /= sampled.sum()
        if np.all(sampled <= bounds):
            return CoefficientDraw(
                dict(zip(names, map(float, sampled))), attempt, attempt - 1, False
            )
    return CoefficientDraw(original, max_attempts, max_attempts, True)
