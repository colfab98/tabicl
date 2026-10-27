"""Reviewed v3 Soccol profile by default; historical profiles require a name."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from .soccol_schema import (
    SOCCOL_FEATURE_COLUMNS,
    SOCCOL_FEATURE_PROFILE as LEGACY_SOCCOL_FEATURE_PROFILE,
)
from .soccol_reviewed_schema import (
    SOCCOL_REVIEWED_FEATURE_PROFILE, SOCCOL_REVIEWED_SCHEMA_VERSION,
    SOCCOL_REVIEWED_FEATURE_PROFILES, SOCCOL_COMPOSITION_SAMPLING_POLICY,
    SOCCOL_REVIEWED_FEATURE_COLUMNS, SOCCOL_REVIEWED_COMPOSITION_COLUMNS,
    SOCCOL_REVIEWED_CONTEXT_COLUMNS, SOCCOL_REVIEWED_CATEGORICAL_COLUMNS,
)


SOCCOL_FEATURE_PROFILE = SOCCOL_REVIEWED_FEATURE_PROFILE


@dataclass(frozen=True)
class SoccolFeatureProfile:
    name: str
    row_ids: tuple[str, ...]
    source_task_row_indices: np.ndarray
    feature_values: np.ndarray
    metadata: dict[str, object]
    latent_feature_values: np.ndarray | None = None
    composition_template_indices: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        return int(self.feature_values.shape[0])

    @property
    def composition_count(self) -> int:
        from .soccol_schema import SOCCOL_COMPOSITION_COLUMNS

        return len(self.metadata.get("composition_columns", SOCCOL_COMPOSITION_COLUMNS))


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _validate_reviewed_profile(metadata, row_ids, task_indices, values) -> np.ndarray:
    """Validate the frozen bank without requiring the original dataset at runtime."""
    if (
        metadata.get("schema_version") != SOCCOL_REVIEWED_SCHEMA_VERSION
        or tuple(metadata.get("feature_columns", ())) != SOCCOL_REVIEWED_FEATURE_COLUMNS
        or tuple(metadata.get("composition_columns", ())) != SOCCOL_REVIEWED_COMPOSITION_COLUMNS
        or tuple(metadata.get("context_columns", ())) != SOCCOL_REVIEWED_CONTEXT_COLUMNS
        or tuple(metadata.get("categorical_columns", ())) != SOCCOL_REVIEWED_CATEGORICAL_COLUMNS
    ):
        raise ValueError("Reviewed Soccol bank schema is inconsistent.")
    selection = metadata["development_selection"]
    scope = metadata["sampling_scope"]
    preprocessing = metadata["preprocessing"]
    payload = {key: selection[key] for key in (
        "policy", "source_row_ids", "source_task_row_indices", "validation_folds",
    )}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    folds, counts = np.unique(selection["validation_folds"], return_counts=True)
    if (
        selection["policy"] != "accepted_composition_and_frozen_outer_development"
        or digest != selection["selection_sha256"]
        or preprocessing["selection_sha256"] != digest
        or selection["source_row_ids"] != row_ids
        or selection["source_task_row_indices"] != task_indices
        or selection["reviewed_data"]["accepted_development_task_row_indices"] != task_indices
        or task_indices != sorted(set(task_indices)) or len(set(row_ids)) != len(row_ids)
        or len(row_ids) != selection["source_rows"] or preprocessing["source_rows"] != len(row_ids)
        or len(selection["validation_folds"]) != len(row_ids)
        or folds.tolist() != [1, 2, 3, 4, 5]
        or selection["fold_counts"] != {str(int(f)): int(n) for f, n in zip(folds, counts)}
        or scope["outer_partition"] != "development" or scope["composition_status"] != "accepted"
        or scope["final_query_rows_used"] != 0 or scope["withheld_development_rows_used"] != 0
        or metadata["target_columns_used"] != []
    ):
        raise ValueError("Reviewed Soccol bank must use only the frozen accepted development selection.")
    material = values[:, :len(SOCCOL_REVIEWED_COMPOSITION_COLUMNS)]
    if not np.isfinite(material).all() or np.any(material < 0) or np.any(np.abs(material.sum(axis=1) - 100) > 1 + 1e-12):
        raise ValueError("Reviewed Soccol bank compositions differ from accepted complete amounts.")
    if len(np.unique(material, axis=0)) != metadata["sampling"]["distinct_compositions"]:
        raise ValueError("Reviewed Soccol bank composition count changed.")
    latent = values.copy()
    for index, column in enumerate(SOCCOL_REVIEWED_FEATURE_COLUMNS):
        array = values[:, index]
        missing = np.isnan(array)
        if int(missing.sum()) != metadata["missing_counts"][column]:
            raise ValueError(f"Reviewed Soccol bank missingness changed: {column}")
        if column not in SOCCOL_REVIEWED_CONTEXT_COLUMNS:
            continue
        observed = array[~missing]
        if column in SOCCOL_REVIEWED_CATEGORICAL_COLUMNS:
            mapping = preprocessing["category_mappings"][column]
            if list(mapping) != sorted(mapping) or list(mapping.values()) != list(range(len(mapping))) or set(observed) != set(mapping.values()):
                raise ValueError(f"Invalid reviewed Soccol category mapping: {column}")
            codes, counts = np.unique(observed, return_counts=True)
            expected_fill = float(codes[np.argmax(counts)]) if len(codes) else 0.0
        else:
            expected_fill = float(observed.mean()) if len(observed) else 0.0
        fill = float(preprocessing["latent_fill_values"][column])
        if not np.isfinite(fill) or not np.isclose(fill, expected_fill, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Soccol filling statistic was not fitted on bank rows: {column}")
        latent[missing, index] = fill
    return _readonly(latent)


@lru_cache(maxsize=None)
def load_soccol_feature_profile(
    profile_name: str = SOCCOL_FEATURE_PROFILE,
    *,
    asset_dir: str | None = None,
) -> SoccolFeatureProfile:
    reviewed = profile_name in SOCCOL_REVIEWED_FEATURE_PROFILES
    if profile_name not in (LEGACY_SOCCOL_FEATURE_PROFILE, *SOCCOL_REVIEWED_FEATURE_PROFILES):
        raise ValueError(f"Unknown Soccol feature profile: {profile_name!r}")
    root = resources.files("tabicl.prior.assets") if asset_dir is None else Path(asset_dir)
    metadata_path = root / f"{profile_name}.json"
    csv_path = root / f"{profile_name}.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("profile_name") != profile_name:
        raise ValueError(f"Unexpected Soccol profile name in {metadata_path}")
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != metadata.get("csv_sha256"):
        raise ValueError(f"Checksum mismatch for Soccol feature asset: {csv_path}")

    columns = SOCCOL_REVIEWED_FEATURE_COLUMNS if reviewed else SOCCOL_FEATURE_COLUMNS
    expected_columns = ["profile_row_id", "task_row_index", *columns]
    row_ids: list[str] = []
    task_indices: list[int] = []
    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected Soccol feature columns: {reader.fieldnames}")
        for row in reader:
            row_ids.append(str(row["profile_row_id"]))
            task_indices.append(int(row["task_row_index"]))
            rows.append([float(row[column]) if row[column] != "" else np.nan for column in columns])
    values = np.asarray(rows, dtype=np.float64)
    expected_shape = (int(metadata["rows"]), len(columns))
    if values.shape != expected_shape or np.isinf(values).any() or (not reviewed and np.isnan(values).any()):
        raise ValueError(f"Soccol feature profile has invalid shape or values: {values.shape}")
    if not reviewed and task_indices != list(range(values.shape[0])):
        raise ValueError("Soccol profile task-row indices must be complete and ordered.")
    latent = _validate_reviewed_profile(metadata, row_ids, task_indices, values) if reviewed else values
    template_indices = np.arange(len(values), dtype=np.int64)
    if profile_name == SOCCOL_REVIEWED_FEATURE_PROFILE:
        # Retain the first accepted source row for each exact saved composition.
        # Environment sampling still uses every observation in the unchanged CSV.
        _, first = np.unique(values[:, :len(SOCCOL_REVIEWED_COMPOSITION_COLUMNS)], axis=0, return_index=True)
        template_indices = np.sort(first).astype(np.int64)
        sampling = metadata["sampling"]
        if (
            sampling.get("composition_sampling_policy") != SOCCOL_COMPOSITION_SAMPLING_POLICY
            or sampling.get("composition_template_source_task_row_indices") != [task_indices[i] for i in template_indices]
            or len(template_indices) != sampling["distinct_compositions"]
        ):
            raise ValueError("Reviewed Soccol distinct-composition sampling metadata is inconsistent.")
    return SoccolFeatureProfile(
        name=profile_name,
        row_ids=tuple(row_ids),
        source_task_row_indices=_readonly(np.asarray(task_indices, dtype=np.int64)),
        feature_values=_readonly(values),
        metadata=metadata,
        latent_feature_values=_readonly(latent),
        composition_template_indices=_readonly(template_indices),
    )


__all__ = ["SOCCOL_FEATURE_PROFILE", "SoccolFeatureProfile", "load_soccol_feature_profile"]
