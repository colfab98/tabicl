"""Fixed Magpie-style composition descriptors for the EPIT alloy schema.

The lookup values below were extracted once from Matminer 0.10.0
``MagpieData``.  Matminer is intentionally not a runtime dependency: synthetic
training calls these calculations for every generated dataset, so the small
fixed table is kept locally and evaluated with vectorized NumPy or Torch
operations.

The 17 EPIT material inputs are weight percentages.  Descriptor calculations
therefore convert a copy of those values to atomic fractions using the lookup
atomic weights.  The original material inputs are never modified.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
from torch import Tensor


EPIT_MAGPIE_ELEMENTS: Final[tuple[str, ...]] = (
    "Fe",
    "Cr",
    "Ni",
    "Mo",
    "W",
    "Nb",
    "Al",
    "V",
    "Ta",
    "Re",
    "Ce",
    "Ti",
    "Co",
    "B",
    "Mg",
    "Y",
    "Gd",
)
EPIT_MAGPIE_MATERIAL_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"Composition, wt.% {element}" for element in EPIT_MAGPIE_ELEMENTS
)
EPIT_MAGPIE_DESCRIPTOR_NAMES: Final[tuple[str, ...]] = (
    "magpie_mean_electronegativity",
    "magpie_range_electronegativity",
    "magpie_avg_dev_electronegativity",
    "magpie_mean_covalent_radius",
    "magpie_avg_dev_covalent_radius",
    "magpie_mean_melting_temperature",
    "magpie_range_melting_temperature",
    "magpie_mean_d_valence_electrons",
    "magpie_mean_total_valence_electrons",
    "magpie_mean_unfilled_valence_states",
)
EPIT_BASE_FEATURE_COUNT: Final[int] = 21
EPIT_MATERIAL_FEATURE_COUNT: Final[int] = len(EPIT_MAGPIE_ELEMENTS)
EPIT_MAGPIE_FEATURE_COUNT: Final[int] = len(EPIT_MAGPIE_DESCRIPTOR_NAMES)
EPIT_MAGPIE_TOTAL_FEATURE_COUNT: Final[int] = EPIT_BASE_FEATURE_COUNT + EPIT_MAGPIE_FEATURE_COUNT
EPIT_MAGPIE_VERSION: Final[str] = "epit_magpie_v1_matminer_0_10_0"
# Dense softmax compositions give every element a nonzero numerical trace.
# Without a presence cutoff, both range descriptors would therefore be fixed
# global constants. Means and average deviations still use every component.
EPIT_MAGPIE_RANGE_MIN_ATOMIC_FRACTION: Final[float] = 0.01

# Rows follow EPIT_MAGPIE_ELEMENTS. Values are from Matminer 0.10.0 MagpieData.
_ATOMIC_WEIGHTS: Final[tuple[float, ...]] = (
    55.845,
    51.9961,
    58.6934,
    95.96,
    183.84,
    92.90638,
    26.9815386,
    50.9415,
    180.94788,
    186.207,
    140.116,
    47.867,
    58.933195,
    10.811,
    24.305,
    88.90585,
    157.25,
)
_ELEMENTAL_PROPERTIES: Final[tuple[tuple[float, ...], ...]] = (
    # Electronegativity
    (1.83, 1.66, 1.91, 2.16, 2.36, 1.60, 1.61, 1.63, 1.50, 1.90, 1.12, 1.54, 1.88, 2.04, 1.31, 1.22, 1.20),
    # CovalentRadius (the radius property used by the standard Magpie preset)
    (
        132.0,
        139.0,
        124.0,
        154.0,
        162.0,
        164.0,
        121.0,
        153.0,
        170.0,
        151.0,
        204.0,
        160.0,
        126.0,
        84.0,
        141.0,
        190.0,
        196.0,
    ),
    # MeltingT
    (
        1811.0,
        2180.0,
        1728.0,
        2896.0,
        3695.0,
        2750.0,
        933.47,
        2183.0,
        3290.0,
        3459.0,
        1071.0,
        1941.0,
        1768.0,
        2348.0,
        923.0,
        1799.0,
        1586.0,
    ),
    # NdValence
    (6.0, 5.0, 8.0, 5.0, 4.0, 4.0, 0.0, 3.0, 3.0, 5.0, 1.0, 2.0, 7.0, 0.0, 0.0, 1.0, 1.0),
    # NValence
    (8.0, 6.0, 10.0, 6.0, 20.0, 5.0, 3.0, 5.0, 19.0, 21.0, 4.0, 4.0, 9.0, 3.0, 2.0, 3.0, 10.0),
    # NUnfilled
    (4.0, 6.0, 2.0, 6.0, 6.0, 7.0, 5.0, 7.0, 7.0, 5.0, 22.0, 8.0, 3.0, 5.0, 0.0, 9.0, 16.0),
)


def _validate_numpy_compositions(compositions: np.ndarray) -> np.ndarray:
    values = np.asarray(compositions, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != EPIT_MATERIAL_FEATURE_COUNT:
        raise ValueError(
            f"Expected compositions with {EPIT_MATERIAL_FEATURE_COUNT} material columns, "
            f"got shape {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Magpie composition inputs must be finite.")
    if np.any(values < 0.0):
        raise ValueError("Magpie composition inputs must be non-negative.")
    if np.any(values.sum(axis=-1) <= 0.0):
        raise ValueError("Every Magpie composition row must contain positive material mass.")
    return values


def _atomic_fractions_numpy(compositions: np.ndarray) -> np.ndarray:
    values = _validate_numpy_compositions(compositions)
    mole_amounts = values / np.asarray(_ATOMIC_WEIGHTS, dtype=np.float64)
    return mole_amounts / mole_amounts.sum(axis=-1, keepdims=True)


def magpie_descriptors_numpy(compositions: np.ndarray) -> np.ndarray:
    """Calculate the fixed ten descriptors from 17 EPIT weight percentages."""

    fractions = _atomic_fractions_numpy(compositions)
    properties = np.asarray(_ELEMENTAL_PROPERTIES, dtype=np.float64)
    present_for_range = fractions >= EPIT_MAGPIE_RANGE_MIN_ATOMIC_FRACTION

    means = np.einsum("...e,pe->...p", fractions, properties)
    avg_devs = np.sum(
        fractions[..., None, :] * np.abs(properties - means[..., :, None]),
        axis=-1,
    )
    minima = np.min(np.where(present_for_range[..., None, :], properties, np.inf), axis=-1)
    maxima = np.max(np.where(present_for_range[..., None, :], properties, -np.inf), axis=-1)
    ranges = maxima - minima

    return np.stack(
        (
            means[..., 0],
            ranges[..., 0],
            avg_devs[..., 0],
            means[..., 1],
            avg_devs[..., 1],
            means[..., 2],
            ranges[..., 2],
            means[..., 3],
            means[..., 4],
            means[..., 5],
        ),
        axis=-1,
    )


def _validate_torch_compositions(compositions: Tensor) -> None:
    if compositions.ndim < 1 or compositions.shape[-1] != EPIT_MATERIAL_FEATURE_COUNT:
        raise ValueError(
            f"Expected compositions with {EPIT_MATERIAL_FEATURE_COUNT} material columns, "
            f"got shape {tuple(compositions.shape)}."
        )
    if not torch.isfinite(compositions).all():
        raise ValueError("Magpie composition inputs must be finite.")
    if torch.any(compositions < 0.0):
        raise ValueError("Magpie composition inputs must be non-negative.")
    if torch.any(compositions.sum(dim=-1) <= 0.0):
        raise ValueError("Every Magpie composition row must contain positive material mass.")


def magpie_descriptors_torch(compositions: Tensor) -> Tensor:
    """Torch equivalent of :func:`magpie_descriptors_numpy`."""

    if not torch.is_floating_point(compositions):
        compositions = compositions.float()
    _validate_torch_compositions(compositions)
    weights = compositions.new_tensor(_ATOMIC_WEIGHTS)
    properties = compositions.new_tensor(_ELEMENTAL_PROPERTIES)
    mole_amounts = compositions / weights
    fractions = mole_amounts / mole_amounts.sum(dim=-1, keepdim=True)
    present_for_range = fractions >= EPIT_MAGPIE_RANGE_MIN_ATOMIC_FRACTION

    means = torch.einsum("...e,pe->...p", fractions, properties)
    avg_devs = torch.sum(
        fractions.unsqueeze(-2) * torch.abs(properties - means.unsqueeze(-1)),
        dim=-1,
    )
    minima = torch.where(
        present_for_range.unsqueeze(-2),
        properties,
        torch.full_like(properties, torch.inf),
    ).amin(dim=-1)
    maxima = torch.where(
        present_for_range.unsqueeze(-2),
        properties,
        torch.full_like(properties, -torch.inf),
    ).amax(dim=-1)
    ranges = maxima - minima

    return torch.stack(
        (
            means[..., 0],
            ranges[..., 0],
            avg_devs[..., 0],
            means[..., 1],
            avg_devs[..., 1],
            means[..., 2],
            ranges[..., 2],
            means[..., 3],
            means[..., 4],
            means[..., 5],
        ),
        dim=-1,
    )


def append_magpie_descriptors_torch(
    X: Tensor,
    *,
    material_slice: slice = slice(0, EPIT_MATERIAL_FEATURE_COUNT),
) -> Tensor:
    """Append descriptors calculated from a copy of the selected material columns."""

    material = X[..., material_slice]
    descriptors = magpie_descriptors_torch(material)
    return torch.cat((X, descriptors), dim=-1)
