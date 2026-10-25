#!/usr/bin/env python3
"""Build composition-only successors to the frozen EPIT sampling assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from scripts.epit_pipeline.composition_data import load_reviewed_compositions
from tabicl.prior.epit_schema import EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS


ASSET_DIR = Path(__file__).resolve().parents[1] / "src/tabicl/prior/assets"
COMPOSITION_NAME = "epit_dataset_v3"
FEATURE_NAME = "epit_fe_nicrmo_features_v2"


def build_assets() -> dict[str, bytes]:
    reviewed = load_reviewed_compositions()
    data = reviewed.rows
    elements = list(EPIT_COMPOSITION_ELEMENTS)
    legacy_composition = json.loads((ASSET_DIR / "epit_dataset_v2.json").read_text())
    families = legacy_composition["families"]
    family_lookup = {display: name for name, display in legacy_composition["family_display_names"].items()}
    unique = {}
    for row in data.to_dict("records"):
        values = tuple(float(row[element]) for element in elements)
        key = tuple(round(value, 6) for value in values)
        family = family_lookup[str(row["Material class"])]
        if key not in unique:
            unique[key] = {"template_id": f"epit_reviewed_{len(unique) + 1:04d}", "family": family,
                           **dict(zip(elements, values, strict=True)), "row_ids": []}
        if unique[key]["family"] != family:
            raise RuntimeError("Identical reviewed EPIT compositions have conflicting alloy families.")
        unique[key]["row_ids"].append(row["row_id"])
    records = list(unique.values())
    counts = {family: sum(row["family"] == family for row in records) for family in families}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["template_id", "family", *elements], extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    csv_bytes = buffer.getvalue().encode()
    composition = {
        "profile_name": COMPOSITION_NAME, "profile_version": 3,
        "description": "Reviewed EPIT compositions kept exactly as saved, before synthetic perturbation.",
        "source_dataset": "electrochemical_metrics_alloys", "source_sheet": "Pitting Potential",
        "reviewed_composition": reviewed.identity,
        "source_rows": 810, "usable_rows": len(data), "distinct_templates": len(records),
        "row_filter": "All 760 reviewed usable EPIT rows, retaining the historical feature-only bank scope; no targets stored or used.",
        "deduplication": "First exact saved vector per 6-decimal key; missing and explicit zero no longer create duplicate templates.",
        "elements": elements, "observed_elements": elements, "omitted_elements": [],
        "families": families, "family_display_names": legacy_composition["family_display_names"],
        "family_template_counts": counts,
        "pretraining_template_policy": {
            "closure_target_wt_percent": 100.0, "closure_tolerance_wt_percent": 1.0,
            "missing_values_for_eligible_templates": "None; all reviewed template amounts were already finite.",
            "evaluation_rows_removed": 0, "eligible_templates": len(records), "excluded_template_ids": [],
            "eligible_family_template_counts": counts,
        },
        "default_family_probabilities": {family: count / len(records) for family, count in counts.items()},
        "template_source_row_ids": {row["template_id"]: row["row_ids"] for row in records},
        "csv_file": f"{COMPOSITION_NAME}.csv", "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "builder": "scripts/build_epit_reviewed_composition_profile.py",
        "normalization": "Real rows and stored templates were not normalized. Synthetic perturbation retained its existing closure to 100 wt.%.",
    }
    composition_bytes = (json.dumps(composition, indent=2) + "\n").encode()

    # This step changes compositions only; keep every environment tuple, code
    # and environment imputation value from the previous feature profile.
    features = json.loads((ASSET_DIR / "epit_fe_nicrmo_features_v1.json").read_text())
    feature_csv = (ASSET_DIR / "epit_fe_nicrmo_features_v1.csv").read_bytes()
    features.update(profile_name=FEATURE_NAME, profile_version=2, csv_file=f"{FEATURE_NAME}.csv",
                    default_composition_profile=COMPOSITION_NAME,
                    description="Historical EPIT environment tuples paired with the reviewed composition bank.",
                    builder="scripts/build_epit_reviewed_composition_profile.py",
                    reviewed_composition=reviewed.identity)
    allowed = features["families"]
    features["composition_profile"] = {
        "name": COMPOSITION_NAME, "json_sha256": hashlib.sha256(composition_bytes).hexdigest(),
        "csv_sha256": composition["csv_sha256"], "reuse": "Reference reviewed 24-element templates.",
        "allowed_families": allowed, "allowed_family_template_counts": {family: counts[family] for family in allowed},
    }
    preprocessing = features["evaluator_preprocessing"]
    preprocessing["numeric_columns"] = [*EPIT_COMPOSITION_COLUMNS, *features["source_environment_columns"]]
    for column, element in zip(EPIT_COMPOSITION_COLUMNS, elements, strict=True):
        preprocessing["numeric_imputation_means"][column] = float(np.mean(data[element]))
    preprocessing["composition_handling"] = "Use reviewed numeric compositions verbatim; composition means are unused because no amounts are missing."
    features["scope"] = "Composition-only update; expanded environment features and development-only sampling remain separate migrations."
    return {
        f"{COMPOSITION_NAME}.csv": csv_bytes,
        f"{COMPOSITION_NAME}.json": composition_bytes,
        f"{FEATURE_NAME}.csv": feature_csv,
        f"{FEATURE_NAME}.json": (json.dumps(features, indent=2) + "\n").encode(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assets = build_assets()
    if args.check:
        for name, content in assets.items():
            if (args.output_dir / name).read_bytes() != content:
                raise RuntimeError(f"Reviewed EPIT asset differs from its inputs: {name}")
        print("Reviewed EPIT composition assets match their source export.")
        return
    if any((args.output_dir / name).exists() for name in assets):
        raise FileExistsError("Refusing to overwrite existing reviewed EPIT assets.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in assets.items():
        (args.output_dir / name).write_bytes(content)
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
