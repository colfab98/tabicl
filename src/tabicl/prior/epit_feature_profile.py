from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from tabicl.prior.epit_composition_profile import (
    EpitCompositionProfile,
    load_epit_composition_profile,
)


EPIT_FEATURE_PROFILE = "epit_fe_nicrmo_features_v1"
EPIT_FEATURE_FAMILIES = ("fe_alloy", "nicrmo_alloy")
EPIT_FEATURE_FAMILY_ENVIRONMENT_COUNTS = (514, 51)
EPIT_FEATURE_FAMILY_TEMPLATE_COUNTS = (298, 17)
EPIT_ENVIRONMENT_COLUMNS = (
    "temperature_celsius",
    "chloride_molar",
    "pH",
)
EPIT_METHOD_FAMILIES = (
    "potentiodynamic",
    "potentiostatic",
    "mixed_dynamic_static",
    "scratch",
    "other",
    "missing",
)


@dataclass(frozen=True)
class EpitFeatureProfile:
    """Target-free empirical inputs for the Fe/Ni--Cr--Mo EPIT prior.

    Environment rows retain their observed joint tuples. The composition bank
    remains the separately versioned 24-element profile and is referenced by
    checksum instead of being duplicated here.
    """

    name: str
    families: tuple[str, ...]
    composition_profile: EpitCompositionProfile
    composition_template_indices: np.ndarray
    composition_family_indices: np.ndarray
    environment_ids: tuple[str, ...]
    source_task_row_indices: np.ndarray
    source_numbers: tuple[str, ...]
    environment_family_indices: np.ndarray
    environment_values: np.ndarray
    environment_missing_mask: np.ndarray
    environment_imputed_values: np.ndarray
    test_methods: tuple[str, ...]
    test_method_families: tuple[str, ...]
    test_method_codes: np.ndarray
    test_method_categories: tuple[str, ...]
    test_method_mapping: dict[str, int]
    metadata: dict[str, object]

    @property
    def n_environment_rows(self) -> int:
        return int(self.environment_values.shape[0])

    @property
    def n_composition_templates(self) -> int:
        return int(self.composition_template_indices.size)


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _asset_files(profile_name: str, asset_dir: str | None):
    if profile_name != EPIT_FEATURE_PROFILE:
        raise ValueError(f"Unknown EPIT feature profile: {profile_name!r}")
    if asset_dir is None:
        root = resources.files("tabicl.prior.assets")
    else:
        root = Path(asset_dir)
    return root / f"{profile_name}.json", root / f"{profile_name}.csv"


def _normalized_method(value: str) -> str:
    stripped = str(value).strip()
    return "<NA>" if stripped in {"", "NA", "N/A", "nan", "None"} else stripped


@lru_cache(maxsize=None)
def _load_epit_feature_profile(profile_name: str, asset_dir: str | None) -> EpitFeatureProfile:
    metadata_path, csv_path = _asset_files(profile_name, asset_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("profile_name") != profile_name:
        raise ValueError(f"Unexpected EPIT feature profile name in {metadata_path}")

    csv_bytes = csv_path.read_bytes()
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    if csv_sha256 != metadata.get("csv_sha256"):
        raise ValueError(f"Checksum mismatch for EPIT feature asset: {csv_path}")

    families = tuple(str(value) for value in metadata["families"])
    if families != EPIT_FEATURE_FAMILIES:
        raise ValueError(f"Unexpected EPIT feature families: {families}")

    composition_metadata = metadata["composition_profile"]
    composition_name = str(composition_metadata["name"])
    composition_root = (
        resources.files("tabicl.prior.assets")
        if asset_dir is None
        else Path(asset_dir)
    )
    composition_metadata_path = composition_root / f"{composition_name}.json"
    composition_metadata_sha256 = hashlib.sha256(
        composition_metadata_path.read_bytes()
    ).hexdigest()
    if composition_metadata_sha256 != composition_metadata.get("json_sha256"):
        raise ValueError("Referenced EPIT composition-profile metadata checksum changed.")

    composition = load_epit_composition_profile(
        composition_name,
        asset_dir=asset_dir,
    )
    if composition.metadata.get("csv_sha256") != composition_metadata.get("csv_sha256"):
        raise ValueError("Referenced EPIT composition-profile checksum changed.")

    family_to_index = {family: index for index, family in enumerate(families)}
    composition_template_indices = np.asarray(
        [
            index
            for index, family in enumerate(composition.template_families)
            if family in family_to_index
        ],
        dtype=np.int64,
    )
    composition_family_indices = np.asarray(
        [family_to_index[composition.template_families[index]] for index in composition_template_indices],
        dtype=np.int64,
    )
    composition_counts = tuple(
        int(np.sum(composition_family_indices == index)) for index in range(len(families))
    )
    if composition_counts != EPIT_FEATURE_FAMILY_TEMPLATE_COUNTS:
        raise ValueError(f"Unexpected restricted composition-template counts: {composition_counts}")

    environment_ids: list[str] = []
    source_task_row_indices: list[int] = []
    source_numbers: list[str] = []
    environment_family_indices: list[int] = []
    environment_rows: list[list[float]] = []
    test_methods: list[str] = []
    test_method_families: list[str] = []
    test_method_codes: list[int] = []
    expected_columns = [
        "environment_id",
        "task_row_index",
        "source_no",
        "family",
        *EPIT_ENVIRONMENT_COLUMNS,
        "test_method",
        "test_method_family",
        "test_method_category",
    ]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected EPIT feature CSV columns: {reader.fieldnames}")
        for row in reader:
            environment_ids.append(str(row["environment_id"]))
            source_task_row_indices.append(int(row["task_row_index"]))
            source_numbers.append(str(row["source_no"]))
            try:
                environment_family_indices.append(family_to_index[str(row["family"])])
            except KeyError as error:
                raise ValueError(f"Unknown family in EPIT feature rows: {error.args[0]!r}") from error
            environment_rows.append(
                [float(row[column]) if row[column] != "" else np.nan for column in EPIT_ENVIRONMENT_COLUMNS]
            )
            test_methods.append(str(row["test_method"]))
            method_family = str(row["test_method_family"])
            if method_family not in EPIT_METHOD_FAMILIES:
                raise ValueError(f"Unknown EPIT test-method family: {method_family!r}")
            test_method_families.append(method_family)
            test_method_codes.append(int(row["test_method_category"]))

    environment_values = np.asarray(environment_rows, dtype=np.float64)
    expected_rows = int(metadata["environment_rows"])
    if environment_values.shape != (expected_rows, len(EPIT_ENVIRONMENT_COLUMNS)):
        raise ValueError(f"Unexpected EPIT environment shape: {environment_values.shape}")
    if len(set(environment_ids)) != len(environment_ids):
        raise ValueError("EPIT environment identifiers must be unique.")
    if len(set(source_task_row_indices)) != len(source_task_row_indices):
        raise ValueError("EPIT source task-row indices must be unique.")
    if np.any(environment_values[:, 1][np.isfinite(environment_values[:, 1])] < 0.0):
        raise ValueError("EPIT chloride concentrations must be non-negative.")

    environment_family_array = np.asarray(environment_family_indices, dtype=np.int64)
    environment_counts = tuple(
        int(np.sum(environment_family_array == index)) for index in range(len(families))
    )
    if environment_counts != EPIT_FEATURE_FAMILY_ENVIRONMENT_COUNTS:
        raise ValueError(f"Unexpected EPIT environment family counts: {environment_counts}")

    preprocessing = metadata["evaluator_preprocessing"]
    raw_mapping = preprocessing["category_mapping"]
    method_mapping = {str(category): int(code) for category, code in raw_mapping.items()}
    method_categories = tuple(
        category for category, _ in sorted(method_mapping.items(), key=lambda item: item[1])
    )
    if tuple(sorted(method_categories)) != method_categories:
        raise ValueError("EPIT test-method category mapping must use lexicographically sorted categories.")
    if tuple(method_mapping.values()) != tuple(range(len(method_mapping))):
        raise ValueError("EPIT test-method category codes must be contiguous and zero based.")

    method_code_array = np.asarray(test_method_codes, dtype=np.int64)
    expected_codes = np.asarray(
        [method_mapping[_normalized_method(method)] for method in test_methods],
        dtype=np.int64,
    )
    if not np.array_equal(method_code_array, expected_codes):
        raise ValueError("EPIT test-method codes do not match the recorded evaluator mapping.")

    imputation_means = preprocessing["numeric_imputation_means"]
    environment_means = np.asarray(
        [float(imputation_means[source]) for source in metadata["source_environment_columns"]],
        dtype=np.float64,
    )
    if not np.isfinite(environment_means).all():
        raise ValueError("EPIT environment imputation means must be finite.")
    environment_missing_mask = np.isnan(environment_values)
    environment_imputed = np.where(environment_missing_mask, environment_means, environment_values)
    if not np.isfinite(environment_imputed).all():
        raise ValueError("Imputed EPIT environment profile contains non-finite values.")

    return EpitFeatureProfile(
        name=profile_name,
        families=families,
        composition_profile=composition,
        composition_template_indices=_readonly(composition_template_indices),
        composition_family_indices=_readonly(composition_family_indices),
        environment_ids=tuple(environment_ids),
        source_task_row_indices=_readonly(np.asarray(source_task_row_indices, dtype=np.int64)),
        source_numbers=tuple(source_numbers),
        environment_family_indices=_readonly(environment_family_array),
        environment_values=_readonly(environment_values),
        environment_missing_mask=_readonly(environment_missing_mask),
        environment_imputed_values=_readonly(environment_imputed),
        test_methods=tuple(test_methods),
        test_method_families=tuple(test_method_families),
        test_method_codes=_readonly(method_code_array),
        test_method_categories=method_categories,
        test_method_mapping=method_mapping,
        metadata=metadata,
    )


def load_epit_feature_profile(
    profile_name: str = EPIT_FEATURE_PROFILE,
    *,
    asset_dir: str | Path | None = None,
) -> EpitFeatureProfile:
    """Load and validate the static target-free Fe/Ni--Cr--Mo feature profile.

    Runtime loading never opens the source workbook. ``asset_dir`` is intended
    for tests or explicitly supplied alternative versioned assets.
    """

    normalized_asset_dir = None if asset_dir is None else str(Path(asset_dir).resolve())
    return _load_epit_feature_profile(str(profile_name), normalized_asset_dir)
