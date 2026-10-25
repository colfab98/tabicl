#!/usr/bin/env python3
"""Freeze reviewed EPIT environment tuples using outer development rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_epit_development_profiles import ASSET_DIR, build_assets as build_legacy_assets, _csv_bytes, _json_bytes
from scripts.epit_pipeline.composition_data import PROCESSED_DIR, load_reviewed_compositions
from tabicl.prior.epit_expanded_schema import (
    EPIT_EXPANDED_FEATURE_PROFILE as NAME,
    EPIT_EXPANDED_FEATURE_COLUMNS as COLUMNS,
    EPIT_EXPANDED_ENVIRONMENT_COLUMNS as ENVIRONMENT,
    EPIT_EXPANDED_CATEGORICAL_COLUMNS as CATEGORICAL,
)


def build_assets(reviewed=None) -> dict[str, bytes]:
    reviewed = load_reviewed_compositions() if reviewed is None else reviewed
    registry = json.loads((PROCESSED_DIR / "feature_registry.json").read_text())
    if tuple(registry["feature_columns"]) != COLUMNS:
        raise RuntimeError("Reviewed EPIT registry differs from the synthetic schema.")
    legacy = build_legacy_assets(reviewed)
    metadata = json.loads(legacy["epit_fe_nicrmo_features_v3.json"])
    rows = reviewed.rows.set_index("row_id")
    development = rows.loc[metadata["evaluator_preprocessing"]["source_row_ids"]]
    selected = rows.loc[metadata["sampling_scope"]["source_row_ids"]]
    encoded = pd.DataFrame(index=development.index)
    mappings, fill_values = {}, {}
    for column in ENVIRONMENT:
        values = development[column]
        if column in CATEGORICAL:
            categories = sorted(values.dropna().astype(str).unique())
            mapping = {category: i for i, category in enumerate(categories)}
            mappings[column] = mapping
            encoded[column] = values.map(mapping).astype(float)
            modes = encoded[column].mode()
            fill_values[column] = float(modes.iloc[0]) if len(modes) else 0.0
        else:
            encoded[column] = pd.to_numeric(values, errors="raise")
            observed = encoded[column].dropna()
            fill_values[column] = float(observed.mean()) if len(observed) else 0.0
    if np.isinf(encoded.to_numpy()).any():
        raise RuntimeError("Reviewed EPIT environment contains infinite values.")

    # Keep the auxiliary method codes for the existing target formulas only.
    import io
    legacy_rows = pd.read_csv(io.BytesIO(legacy["epit_fe_nicrmo_features_v3.csv"]), keep_default_na=False)
    formula_columns = {"temperature_celsius": "CP_temp", "chloride_molar": "CP_Cl", "pH": "CP_pH"}
    records = []
    for record, (row_id, _) in zip(legacy_rows.to_dict("records"), selected.iterrows(), strict=True):
        for old, new in formula_columns.items():
            record[old] = encoded.loc[row_id, new]
        record.update(encoded.loc[row_id].to_dict())
        records.append({key: "" if pd.isna(value) else value for key, value in record.items()})
    csv_bytes = _csv_bytes([*legacy_rows.columns, *ENVIRONMENT], records)
    preprocessing = metadata["evaluator_preprocessing"]
    for old, new in zip(metadata["source_environment_columns"], formula_columns.values(), strict=True):
        preprocessing["numeric_imputation_means"][old] = fill_values[new]
    metadata.update({
        "profile_name": NAME, "profile_version": 4,
        "description": "Reviewed 70-column EPIT inputs sampled from frozen Fe/Ni-Cr-Mo development rows.",
        "feature_columns": list(COLUMNS), "categorical_columns": list(CATEGORICAL),
        "expanded_environment_columns": list(ENVIRONMENT),
        "category_mappings": mappings, "latent_fill_values": fill_values,
        "latent_fill_policy": "Development mean for numeric columns, mode for categories, zero only if entirely missing; used internally for target calculation only.",
        "model_missing_policy": "Original missing entries retained as NaN; no extra independent mask.",
        "expanded_missing_counts": encoded.loc[selected.index].isna().sum().astype(int).to_dict(),
        "environment_missing_counts": {old: int(encoded.loc[selected.index, new].isna().sum()) for old, new in formula_columns.items()},
        "auxiliary_test_method": "Used by existing formulas only; excluded from the 70 model inputs.",
        "environment_sampling": "All 46 environment/history fields and their missingness drawn from one development row; composition drawn independently as before.",
        "csv_file": f"{NAME}.csv", "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "builder": "scripts/build_epit_expanded_feature_profile.py",
        "scope": "70 reviewed model inputs; frozen development rows only; existing families and target formulas retained.",
    })
    return {f"{NAME}.csv": csv_bytes, f"{NAME}.json": _json_bytes(metadata)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assets = build_assets()
    if args.check:
        for name, content in assets.items():
            if (args.output_dir / name).read_bytes() != content:
                raise RuntimeError(f"Expanded EPIT asset differs from frozen inputs: {name}")
        print("Expanded EPIT profile matches its reviewed development inputs.")
        return
    if any((args.output_dir / name).exists() for name in assets):
        raise FileExistsError("Refusing to overwrite an existing feature asset.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in assets.items():
        (args.output_dir / name).write_bytes(content)
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
