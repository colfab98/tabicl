"""Versioned, target-free empirical feature profile for Soccol pretraining."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from .soccol_schema import SOCCOL_FEATURE_COLUMNS, SOCCOL_FEATURE_PROFILE


@dataclass(frozen=True)
class SoccolFeatureProfile:
    name: str
    row_ids: tuple[str, ...]
    source_task_row_indices: np.ndarray
    feature_values: np.ndarray
    metadata: dict[str, object]

    @property
    def n_rows(self) -> int:
        return int(self.feature_values.shape[0])


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


@lru_cache(maxsize=None)
def load_soccol_feature_profile(
    profile_name: str = SOCCOL_FEATURE_PROFILE,
    *,
    asset_dir: str | None = None,
) -> SoccolFeatureProfile:
    if profile_name != SOCCOL_FEATURE_PROFILE:
        raise ValueError(f"Unknown Soccol feature profile: {profile_name!r}")
    root = resources.files("tabicl.prior.assets") if asset_dir is None else Path(asset_dir)
    metadata_path = root / f"{profile_name}.json"
    csv_path = root / f"{profile_name}.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("profile_name") != profile_name:
        raise ValueError(f"Unexpected Soccol profile name in {metadata_path}")
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != metadata.get("csv_sha256"):
        raise ValueError(f"Checksum mismatch for Soccol feature asset: {csv_path}")

    expected_columns = ["profile_row_id", "task_row_index", *SOCCOL_FEATURE_COLUMNS]
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
            rows.append([float(row[column]) for column in SOCCOL_FEATURE_COLUMNS])
    values = np.asarray(rows, dtype=np.float64)
    expected_shape = (int(metadata["rows"]), len(SOCCOL_FEATURE_COLUMNS))
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"Soccol feature profile has invalid shape or values: {values.shape}")
    if task_indices != list(range(values.shape[0])):
        raise ValueError("Soccol profile task-row indices must be complete and ordered.")
    return SoccolFeatureProfile(
        name=profile_name,
        row_ids=tuple(row_ids),
        source_task_row_indices=_readonly(np.asarray(task_indices, dtype=np.int64)),
        feature_values=_readonly(values),
        metadata=metadata,
    )


__all__ = ["SOCCOL_FEATURE_PROFILE", "SoccolFeatureProfile", "load_soccol_feature_profile"]
