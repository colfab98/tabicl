from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np


EPIT_COMPOSITION_PROFILE = "epit_dataset_v1"
EPIT_COMPOSITION_FAMILIES = (
    "fe_alloy",
    "al_alloy",
    "hea",
    "nicrmo_alloy",
    "other",
)
EPIT_COMPOSITION_FAMILY_COUNTS = (298, 56, 19, 17, 13)
EPIT_COMPOSITION_FAMILY_PROBS = tuple(
    count / sum(EPIT_COMPOSITION_FAMILY_COUNTS) for count in EPIT_COMPOSITION_FAMILY_COUNTS
)
EPIT_COMPOSITION_DEFAULT_PERTURB_STRENGTH = 0.05
EPIT_COMPOSITION_DEFAULT_LATENT_COUNT = 2


@dataclass(frozen=True)
class EpitCompositionProfile:
    name: str
    elements: tuple[str, ...]
    observed_elements: tuple[str, ...]
    omitted_elements: tuple[str, ...]
    families: tuple[str, ...]
    family_probabilities: tuple[float, ...]
    template_ids: tuple[str, ...]
    template_families: tuple[str, ...]
    template_family_indices: np.ndarray
    template_values: np.ndarray
    template_missing_mask: np.ndarray
    observed_element_indices: np.ndarray
    metadata: dict[str, object]

    @property
    def n_templates(self) -> int:
        return int(self.template_values.shape[0])

    @property
    def observed_template_values(self) -> np.ndarray:
        return self.template_values[:, self.observed_element_indices]


@dataclass(frozen=True)
class EpitCompositionBatch:
    full_compositions: np.ndarray
    observed_compositions: np.ndarray
    family_indices: np.ndarray
    family_names: tuple[str, ...]
    template_indices: np.ndarray
    template_ids: tuple[str, ...]


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _asset_files(profile_name: str, asset_dir: str | None):
    if profile_name != EPIT_COMPOSITION_PROFILE:
        raise ValueError(f"Unknown EPIT composition profile: {profile_name!r}")
    if asset_dir is None:
        root = resources.files("tabicl.prior.assets")
    else:
        root = Path(asset_dir)
    return root / f"{profile_name}.json", root / f"{profile_name}.csv"


@lru_cache(maxsize=None)
def _load_epit_composition_profile(profile_name: str, asset_dir: str | None) -> EpitCompositionProfile:
    metadata_path, csv_path = _asset_files(profile_name, asset_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    csv_bytes = csv_path.read_bytes()
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    if csv_sha256 != metadata.get("csv_sha256"):
        raise ValueError(f"Checksum mismatch for EPIT composition asset: {csv_path}")

    elements = tuple(str(value) for value in metadata["elements"])
    observed_elements = tuple(str(value) for value in metadata["observed_elements"])
    omitted_elements = tuple(str(value) for value in metadata["omitted_elements"])
    families = tuple(str(value) for value in metadata["families"])
    if families != EPIT_COMPOSITION_FAMILIES:
        raise ValueError(f"Unexpected EPIT composition families: {families}")
    if set(observed_elements).intersection(omitted_elements):
        raise ValueError("Observed and omitted EPIT elements must be disjoint.")
    if set(observed_elements).union(omitted_elements) != set(elements):
        raise ValueError("Observed and omitted EPIT elements must cover the complete element schema.")

    template_ids: list[str] = []
    template_families: list[str] = []
    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ["template_id", "family", *elements]
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected EPIT composition CSV columns: {reader.fieldnames}")
        for row in reader:
            template_ids.append(str(row["template_id"]))
            template_families.append(str(row["family"]))
            rows.append([float(row[element]) if row[element] != "" else np.nan for element in elements])

    template_values = np.asarray(rows, dtype=np.float64)
    if template_values.shape != (int(metadata["distinct_templates"]), len(elements)):
        raise ValueError(f"Unexpected EPIT composition template shape: {template_values.shape}")
    if np.any(np.nan_to_num(template_values, nan=0.0) < 0.0):
        raise ValueError("EPIT composition templates must be non-negative.")
    if len(set(template_ids)) != len(template_ids):
        raise ValueError("EPIT composition template identifiers must be unique.")

    family_to_index = {family: index for index, family in enumerate(families)}
    try:
        family_indices = np.asarray([family_to_index[family] for family in template_families], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown family in EPIT composition templates: {error.args[0]!r}") from error
    family_counts = tuple(int(np.sum(family_indices == index)) for index in range(len(families)))
    if family_counts != EPIT_COMPOSITION_FAMILY_COUNTS:
        raise ValueError(f"Unexpected EPIT composition family counts: {family_counts}")

    filled_sums = np.nansum(template_values, axis=1)
    if np.any(filled_sums < 99.0) or np.any(filled_sums > 101.0):
        raise ValueError("EPIT composition templates do not close near 100 wt.% across all 24 elements.")

    observed_indices = np.asarray([elements.index(element) for element in observed_elements], dtype=np.int64)
    return EpitCompositionProfile(
        name=profile_name,
        elements=elements,
        observed_elements=observed_elements,
        omitted_elements=omitted_elements,
        families=families,
        family_probabilities=EPIT_COMPOSITION_FAMILY_PROBS,
        template_ids=tuple(template_ids),
        template_families=tuple(template_families),
        template_family_indices=_readonly(family_indices),
        template_values=_readonly(template_values),
        template_missing_mask=_readonly(np.isnan(template_values)),
        observed_element_indices=_readonly(observed_indices),
        metadata=metadata,
    )


def load_epit_composition_profile(
    profile_name: str = EPIT_COMPOSITION_PROFILE,
    *,
    asset_dir: str | Path | None = None,
) -> EpitCompositionProfile:
    """Load and validate a static EPIT composition profile.

    The source workbook is never opened here. ``asset_dir`` exists only for
    testing or explicitly supplied alternative assets.
    """

    normalized_asset_dir = None if asset_dir is None else str(Path(asset_dir).resolve())
    return _load_epit_composition_profile(str(profile_name), normalized_asset_dir)


def normalize_epit_family_probabilities(
    probabilities: Sequence[float] | None,
    *,
    n_families: int = len(EPIT_COMPOSITION_FAMILIES),
) -> np.ndarray:
    values = np.asarray(
        EPIT_COMPOSITION_FAMILY_PROBS if probabilities is None else probabilities,
        dtype=np.float64,
    )
    if values.shape != (n_families,):
        raise ValueError(f"Expected {n_families} EPIT family probabilities, got shape {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("EPIT family probabilities must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("At least one EPIT family probability must be positive.")
    return values / total


def _rank_uniform(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size <= 1:
        return np.full(values.shape, 0.5, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    return ranks / float(values.size + 1)


def _composition_batch_from_indices(
    profile: EpitCompositionProfile,
    family_indices: np.ndarray,
    template_indices: np.ndarray,
    *,
    perturb_strength: float,
    rng: np.random.Generator,
) -> EpitCompositionBatch:
    if not np.isfinite(perturb_strength) or float(perturb_strength) < 0.0:
        raise ValueError("perturb_strength must be finite and non-negative.")

    compositions = np.nan_to_num(profile.template_values[template_indices], nan=0.0).copy()
    if float(perturb_strength) > 0.0:
        noise = rng.normal(0.0, float(perturb_strength), size=compositions.shape)
        compositions *= np.where(compositions > 0.0, np.exp(noise), 1.0)
    totals = compositions.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0) or not np.isfinite(totals).all():
        raise ValueError("Sampled EPIT compositions have invalid total mass.")
    compositions = 100.0 * compositions / totals
    observed = compositions[:, profile.observed_element_indices]

    family_names = tuple(profile.families[index] for index in family_indices)
    selected_template_ids = tuple(profile.template_ids[index] for index in template_indices)
    return EpitCompositionBatch(
        full_compositions=_readonly(compositions),
        observed_compositions=_readonly(observed),
        family_indices=_readonly(family_indices),
        family_names=family_names,
        template_indices=_readonly(template_indices),
        template_ids=selected_template_ids,
    )


def map_epit_latents_to_compositions(
    latents: np.ndarray,
    *,
    profile: EpitCompositionProfile | None = None,
    family_probabilities: Sequence[float] | None = None,
    perturb_strength: float = EPIT_COMPOSITION_DEFAULT_PERTURB_STRENGTH,
    random_state: int | np.random.Generator | None = None,
) -> EpitCompositionBatch:
    """Map SCM material latents to chemistry-coherent empirical compositions.

    The first latent selects the alloy family according to the configured
    probabilities. The remaining latents jointly select a real template within
    that family. With one latent, its position inside the selected family band
    selects the template as well. This preserves relationships learned by the
    SCM while replacing abstract material columns with the 17 observed elements.
    """

    latent_values = np.asarray(latents, dtype=np.float64)
    if latent_values.ndim != 2 or latent_values.shape[0] <= 0 or latent_values.shape[1] <= 0:
        raise ValueError("latents must have shape (n_samples, n_latents) with both dimensions positive.")
    profile = load_epit_composition_profile() if profile is None else profile
    probabilities = normalize_epit_family_probabilities(
        family_probabilities,
        n_families=len(profile.families),
    )
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    family_u = _rank_uniform(latent_values[:, 0])
    cumulative = np.cumsum(probabilities)
    family_indices = np.searchsorted(cumulative, family_u, side="right").astype(np.int64)
    family_indices = np.clip(family_indices, 0, len(profile.families) - 1)

    if latent_values.shape[1] == 1:
        lower = np.concatenate(([0.0], cumulative[:-1]))
        selected_probabilities = probabilities[family_indices]
        template_u = (family_u - lower[family_indices]) / selected_probabilities
        template_u = np.clip(template_u, 0.0, 1.0 - np.finfo(np.float64).eps)
    else:
        remaining_ranks = np.column_stack(
            [_rank_uniform(latent_values[:, index]) for index in range(1, latent_values.shape[1])]
        )
        template_score = remaining_ranks.mean(axis=1)
        template_u = np.empty(latent_values.shape[0], dtype=np.float64)
        for family_index in range(len(profile.families)):
            positions = np.flatnonzero(family_indices == family_index)
            if positions.size:
                template_u[positions] = _rank_uniform(template_score[positions])

    template_indices = np.empty(latent_values.shape[0], dtype=np.int64)
    for family_index in range(len(profile.families)):
        positions = np.flatnonzero(family_indices == family_index)
        if positions.size == 0:
            continue
        candidates = np.flatnonzero(profile.template_family_indices == family_index)
        candidate_offsets = np.floor(template_u[positions] * candidates.size).astype(np.int64)
        candidate_offsets = np.clip(candidate_offsets, 0, candidates.size - 1)
        template_indices[positions] = candidates[candidate_offsets]

    return _composition_batch_from_indices(
        profile,
        family_indices,
        template_indices,
        perturb_strength=perturb_strength,
        rng=rng,
    )


def sample_epit_compositions(
    n_samples: int,
    *,
    profile: EpitCompositionProfile | None = None,
    family_probabilities: Sequence[float] | None = None,
    perturb_strength: float = EPIT_COMPOSITION_DEFAULT_PERTURB_STRENGTH,
    random_state: int | np.random.Generator | None = None,
) -> EpitCompositionBatch:
    """Sample a standalone batch from empirical family templates.

    Positive template entries receive multiplicative log-normal perturbations;
    zero and unreported entries remain zero. Full 24-element compositions are
    then closed to 100 wt.% before the evaluated 17 columns are selected.
    """

    if int(n_samples) != n_samples or int(n_samples) <= 0:
        raise ValueError("n_samples must be a positive integer.")
    profile = load_epit_composition_profile() if profile is None else profile
    probabilities = normalize_epit_family_probabilities(
        family_probabilities,
        n_families=len(profile.families),
    )
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    family_indices = rng.choice(len(profile.families), size=int(n_samples), p=probabilities)
    template_indices = np.empty(int(n_samples), dtype=np.int64)
    for family_index in range(len(profile.families)):
        positions = np.flatnonzero(family_indices == family_index)
        if positions.size == 0:
            continue
        candidates = np.flatnonzero(profile.template_family_indices == family_index)
        template_indices[positions] = rng.choice(candidates, size=positions.size, replace=True)

    return _composition_batch_from_indices(
        profile,
        family_indices,
        template_indices,
        perturb_strength=perturb_strength,
        rng=rng,
    )
