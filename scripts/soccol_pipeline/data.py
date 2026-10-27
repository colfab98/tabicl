"""Load reviewed Soccol inputs without repeating dataset preprocessing.

The reference loader retains every frozen row. The shared development selector
supplies accepted rows to formula fitting, sampling and development evaluation.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from tabicl.prior.epit_schema import EPIT_COMPOSITION_ELEMENTS
from tabicl.prior.soccol_reviewed_schema import (
    SOCCOL_REVIEWED_FEATURE_COLUMNS, SOCCOL_REVIEWED_CATEGORICAL_COLUMNS,
    SOCCOL_REVIEWED_FEATURE_PROFILE,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPO_ROOT / "corrosion_datasets/datasets/soccol_pitting_potential/processed"
PROCESSED_DIR = PROCESSED_ROOT / "soccol_reference_v1"
SPLIT_MANIFEST = PROCESSED_ROOT / "splits_v1/split_manifest.json"
REVIEWED_DATA_VERSION = "soccol_reference_v1"
TARGET_COLUMN = "E_pit"
INPUT_FILES = (
    "soccol_processed.csv", "soccol_features.csv", "composition_row_audit.csv",
    "split_assignments.csv", "feature_registry.json",
)
PARTITION_COLUMNS = ("outer_split", "development_validation_fold")


def encode_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse all nine bank mappings without fitting on context or query rows.

    Missing and unseen labels stay NaN; zero is a valid known category code.
    Numeric values and their missingness pass through without filling.
    """
    from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile

    if tuple(frame.columns) != SOCCOL_REVIEWED_FEATURE_COLUMNS:
        raise ValueError("Reviewed Soccol model inputs must match the exact 58-column order.")
    profile = load_soccol_feature_profile(SOCCOL_REVIEWED_FEATURE_PROFILE)
    identity = frame.attrs.get("soccol_development_selection")
    if identity is not None and identity != profile.metadata["development_selection"]:
        raise RuntimeError("Reviewed Soccol model inputs and sampling bank use different development data.")
    output = frame.copy()
    for column in SOCCOL_REVIEWED_CATEGORICAL_COLUMNS:
        output[column] = output[column].map(profile.metadata["preprocessing"]["category_mappings"][column])
    output = output.astype(float)
    if np.isinf(output.to_numpy()).any():
        raise ValueError("Reviewed Soccol model inputs contain infinite values.")
    return output


@dataclass(frozen=True)
class SoccolDataset:
    rows: pd.DataFrame
    features: pd.DataFrame
    targets: pd.Series
    audit: pd.DataFrame
    assignments: pd.DataFrame
    registry: dict[str, Any]
    identity: dict[str, Any]


@dataclass(frozen=True)
class SoccolDevelopmentData:
    rows: pd.DataFrame
    features: pd.DataFrame
    targets: pd.Series
    assignments: pd.DataFrame
    task_row_indices: np.ndarray
    identity: dict[str, Any]


def select_accepted_development(dataset: SoccolDataset | None = None) -> SoccolDevelopmentData:
    """Select before fitting statistics or building templates; keep frozen IDs."""
    dataset = load_soccol_dataset() if dataset is None else dataset
    mask = dataset.assignments.outer_split.eq("development") & dataset.rows.composition_status.eq("accepted")
    if not dataset.rows.use_as_development_context.equals(mask):
        raise RuntimeError("Soccol accepted-development selection disagrees with the saved eligibility flags.")
    indices = np.flatnonzero(mask.to_numpy())
    if indices.tolist() != dataset.identity["accepted_development_task_row_indices"]:
        raise RuntimeError("Soccol accepted-development rows changed after loading.")
    rows = dataset.rows.iloc[indices].copy().reset_index(drop=True)
    features = dataset.features.iloc[indices].copy().reset_index(drop=True)
    targets = dataset.targets.iloc[indices].copy().reset_index(drop=True)
    assignments = dataset.assignments.iloc[indices].copy().reset_index(drop=True)
    if assignments.task_row_index.tolist() != indices.tolist():
        raise RuntimeError("Soccol development selection lost the original task indices.")
    fold_counts = assignments.development_validation_fold.value_counts().sort_index()
    if fold_counts.index.astype(int).tolist() != [1, 2, 3, 4, 5]:
        raise RuntimeError("Accepted Soccol rows must retain all five frozen development folds.")
    if not np.isfinite(features[dataset.registry["composition_columns"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Accepted Soccol development compositions must remain complete.")
    selection = {
        "policy": "accepted_composition_and_frozen_outer_development",
        "source_row_ids": rows.row_id.tolist(),
        "source_task_row_indices": indices.tolist(),
        "validation_folds": assignments.development_validation_fold.astype(int).tolist(),
    }
    identity = {
        **selection,
        "selection_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_rows": len(indices),
        "fold_counts": {str(int(fold)): int(count) for fold, count in fold_counts.items()},
        "withheld_development_rows": int(dataset.assignments.outer_split.eq("development").sum() - len(indices)),
        "final_query_rows_excluded": int(dataset.assignments.outer_split.eq("final_test").sum()),
        "reviewed_data": dataset.identity,
    }
    indices.setflags(write=False)
    return SoccolDevelopmentData(rows, features, targets, assignments, indices, identity)


def reviewed_final_split_metadata(split_manifest: Path = SPLIT_MANIFEST) -> dict[str, Any]:
    """Record accepted context and every original final query without resplitting."""
    dataset = load_soccol_dataset(split_manifest=split_manifest)
    development = select_accepted_development(dataset)
    final_mask = dataset.assignments.outer_split.eq("final_test")
    return {
        "source_sha256": dataset.identity["files_sha256"]["soccol_processed.csv"],
        "original_development_rows": int(dataset.assignments.outer_split.eq("development").sum()),
        "development_rows": len(development.rows),
        "final_test_rows": int(final_mask.sum()),
        "soccol_development_selection": development.identity,
        "final_query_task_row_indices": np.flatnonzero(final_mask.to_numpy()).tolist(),
        "final_query_row_ids": dataset.rows.loc[final_mask, "row_id"].tolist(),
    }


def _align_rows(frame: pd.DataFrame, expected_ids: list[str], name: str) -> pd.DataFrame:
    if "row_id" not in frame or frame.row_id.duplicated().any() or set(frame.row_id) != set(expected_ids):
        raise RuntimeError(f"Reviewed Soccol {name} has missing, duplicate or unexpected row IDs.")
    return frame.set_index("row_id").loc[expected_ids].reset_index()


def _assert_equal(left: pd.DataFrame, right: pd.DataFrame, message: str) -> None:
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise RuntimeError(message) from error


def load_soccol_dataset(
    processed_dir: Path = PROCESSED_DIR, *, split_manifest: Path = SPLIT_MANIFEST,
) -> SoccolDataset:
    """Read the reviewed 58 inputs and unchanged targets in frozen task order.

    Reported values, assumed zeros and unresolved NaNs are read as saved. No
    element is filled, balanced, scaled, or removed here. In particular, the
    unresolved development rows and ten unresolved final queries survive.
    """
    processed_dir = Path(processed_dir)
    split_manifest = Path(split_manifest)
    manifest_path = processed_dir / "processing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != REVIEWED_DATA_VERSION:
        raise RuntimeError("Unsupported reviewed Soccol export version.")
    hashes = {name: sha256_file(processed_dir / name) for name in INPUT_FILES}
    for name, digest in hashes.items():
        if manifest.get("output_sha256", {}).get(name) != digest:
            raise RuntimeError(f"Reviewed Soccol export checksum changed: {name}")

    registry = json.loads((processed_dir / "feature_registry.json").read_text())
    columns = registry["feature_columns"]
    elements = list(EPIT_COMPOSITION_ELEMENTS)
    if registry.get("schema_version") != REVIEWED_DATA_VERSION:
        raise RuntimeError("Reviewed Soccol feature registry has a different version.")
    if registry.get("composition_columns") != elements or columns[:24] != elements:
        raise RuntimeError("Reviewed Soccol composition columns differ from the shared 24-element order.")
    if tuple(columns) != SOCCOL_REVIEWED_FEATURE_COLUMNS or tuple(
        c for c in columns if registry["columns"][c]["dtype"] == "categorical"
    ) != SOCCOL_REVIEWED_CATEGORICAL_COLUMNS:
        raise RuntimeError("Reviewed Soccol requires its registered 58 unique features.")
    excluded = {TARGET_COLUMN, "E_corr", "event", "source", "row_id", "CP_ion_other", "Prep_medium", "CP_anions_info"}
    if excluded.intersection(columns) or any(column.startswith(("raw_", "legacy_")) for column in columns):
        raise RuntimeError("Reviewed Soccol registry contains excluded predictors.")
    if any(registry["columns"][element].get("unit") != "wt.%" for element in elements):
        raise RuntimeError("Reviewed Soccol composition units must remain wt.%.")

    frozen = load_frozen_split(split_manifest, expected_manifest_schema="soccol_composition_split_manifest_v1")
    records = pd.DataFrame(frozen.manifest["rows"]).sort_values("task_row_index").reset_index(drop=True)
    expected_count = int(frozen.manifest["dataset"]["usable_rows"])
    if records.task_row_index.tolist() != list(range(expected_count)) or records.raw_workbook_row.duplicated().any():
        raise RuntimeError("Frozen Soccol row identities are incomplete or duplicated.")
    expected_ids = [f"soccol_r{int(index):06d}" for index in records.raw_workbook_row]

    # round_trip preserves the saved float values; no coercion or imputation.
    read_options = {"float_precision": "round_trip"}
    rows = _align_rows(pd.read_csv(processed_dir / "soccol_processed.csv", **read_options), expected_ids, "rows")
    feature_table = _align_rows(pd.read_csv(processed_dir / "soccol_features.csv", **read_options), expected_ids, "features")
    audit = _align_rows(pd.read_csv(processed_dir / "composition_row_audit.csv", **read_options), expected_ids, "audit")
    assignments = pd.read_csv(processed_dir / "split_assignments.csv", **read_options)
    if hashes["split_assignments.csv"] != frozen.lock["assignments"]["sha256"]:
        raise RuntimeError("Reviewed Soccol assignments differ from the frozen split.")
    assignments = assignments.sort_values("task_row_index").reset_index(drop=True)
    assignment_fields = ["task_row_index", "raw_workbook_row", "source", "composition_group", *PARTITION_COLUMNS]
    _assert_equal(assignments[assignment_fields], records[assignment_fields], "Reviewed Soccol assignments do not match the frozen row records.")
    _assert_equal(rows[["raw_workbook_row", "source", *PARTITION_COLUMNS]], records[["raw_workbook_row", "source", *PARTITION_COLUMNS]], "Reviewed Soccol source IDs or partitions changed.")
    _assert_equal(audit[list(PARTITION_COLUMNS)], records[list(PARTITION_COLUMNS)], "Reviewed Soccol audit partitions changed.")
    if feature_table.columns.tolist() != ["row_id", *columns]:
        raise RuntimeError("Reviewed Soccol feature table differs from its registry.")
    _assert_equal(feature_table, rows[["row_id", *columns]], "Reviewed Soccol feature values disagree with the reference table.")

    audit_columns = ["composition_status", "composition_sum_available_wt_percent", "missing_composition_cells",
                     "known_unresolved_elements", "use_as_development_context", "frozen_final_query"]
    _assert_equal(rows[audit_columns], audit[audit_columns], "Reviewed Soccol rows disagree with their composition audit.")
    if not rows.composition_status.isin(["accepted", "unresolved"]).all():
        raise RuntimeError("Reviewed Soccol has an unknown composition status.")
    values = rows[elements].to_numpy(dtype=float)
    if np.isinf(values).any() or np.any((values < 0) | (values > 100)):
        raise RuntimeError("Reviewed Soccol contains invalid element amounts.")
    accepted = rows.composition_status.eq("accepted").to_numpy()
    totals = np.nansum(values, axis=1)
    missing = np.isnan(values).sum(axis=1)
    if np.any(missing[accepted]) or np.any(np.abs(totals[accepted] - 100) > 1 + 1e-12):
        raise RuntimeError("Accepted Soccol compositions must be complete and within 99–101 wt.%.")
    if np.any(missing[~accepted] == 0):
        raise RuntimeError("Unresolved Soccol compositions were unexpectedly filled.")
    if not np.array_equal(missing, audit.missing_composition_cells.to_numpy()) or not np.allclose(
        totals, audit.composition_sum_available_wt_percent, rtol=0, atol=1e-10,
    ):
        raise RuntimeError("Reviewed Soccol amounts disagree with the saved composition totals or missing counts.")
    for index, text in audit.known_unresolved_elements.items():
        if pd.isna(text):
            continue
        for element in str(text).split(";"):
            if element and (element not in elements or pd.notna(rows.at[index, element])):
                raise RuntimeError(f"Reviewed Soccol unknown element was filled: {expected_ids[index]} / {element}")

    development = records.outer_split.eq("development")
    final = records.outer_split.eq("final_test")
    if not (development | final).all():
        raise RuntimeError("Reviewed Soccol has an unknown outer partition.")
    if not rows.use_as_development_context.equals(development & accepted) or not rows.frozen_final_query.equals(final):
        raise RuntimeError("Reviewed Soccol row eligibility disagrees with the saved partitions and acceptance.")
    design = frozen.manifest["split_design"]
    if int(development.sum()) != design["development_rows"] or int(final.sum()) != design["final_test_rows"]:
        raise RuntimeError("Reviewed Soccol partition counts differ from the frozen split.")
    targets = pd.to_numeric(rows[TARGET_COLUMN], errors="raise")
    if not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise RuntimeError("Reviewed Soccol targets must remain finite.")
    features = feature_table[columns].copy()
    numeric = [column for column in columns if registry["columns"][column]["dtype"] == "numeric"]
    if np.isinf(features[numeric].to_numpy(dtype=float)).any():
        raise RuntimeError("Reviewed Soccol inputs contain infinite values.")
    identity = {
        "version": REVIEWED_DATA_VERSION,
        "files_sha256": hashes,
        "processing_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": frozen.manifest_sha256,
        "split_lock_sha256": frozen.lock_sha256,
        "feature_columns": list(columns),
        "composition_columns": elements,
        "target_column": TARGET_COLUMN,
        "target_conversion_applied": False,
        "rows": len(rows),
        "accepted_development_rows": int((development & accepted).sum()),
        "accepted_development_task_row_indices": np.flatnonzero((development & accepted).to_numpy()).tolist(),
        "frozen_final_query_rows": int(final.sum()),
        "unresolved_final_query_rows": int((final & ~accepted).sum()),
        "composition_policy": "Saved values, zeros and NaNs preserved; no filling, balance calculation or normalization.",
    }
    return SoccolDataset(rows, features, targets.copy(), audit, assignments, registry, identity)
