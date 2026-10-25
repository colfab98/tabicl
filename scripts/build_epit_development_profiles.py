#!/usr/bin/env python3
"""Build EPIT sampling banks using only the frozen outer development rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from scripts.build_epit_feature_profile import (
    ENVIRONMENT_OUTPUT_COLUMNS,
    ENVIRONMENT_SOURCE_COLUMNS,
    FAMILY_DISPLAY_NAMES,
    METHOD_COLUMN,
    _normalized_method,
    _number,
)
from scripts.epit_pipeline.composition_data import ReviewedCompositions, load_reviewed_compositions
from scripts.epit_pipeline.split_data import _method_family
from scripts.eval_corrosion_datasets import eval_to_float
from tabicl.prior.epit_schema import EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS


ASSET_DIR = Path(__file__).resolve().parents[1] / "src/tabicl/prior/assets"
COMPOSITION_NAME = "epit_dataset_v4"
FEATURE_NAME = "epit_fe_nicrmo_features_v3"
FAMILY_DISPLAY = {
    "fe_alloy": "Fe Alloy", "al_alloy": "Al Alloy", "hea": "HEA",
    "nicrmo_alloy": "NiCrMo Alloy", "other": "Other",
}


def _csv_bytes(fields: list[str], records: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode()


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode()


def build_assets(reviewed: ReviewedCompositions | None = None) -> dict[str, bytes]:
    reviewed = load_reviewed_compositions() if reviewed is None else reviewed
    assignments = reviewed.assignments.sort_values("task_row_index")
    development = assignments.loc[assignments.outer_split.eq("development")]
    final = assignments.loc[assignments.outer_split.eq("final_test")]
    if len(development) != 608 or len(final) != 152:
        raise RuntimeError("Unexpected frozen EPIT partition sizes.")
    if set(development.row_id) & set(final.row_id):
        raise RuntimeError("EPIT development and final-query rows overlap.")

    # Restrict before computing templates, distributions, means or categories.
    data = reviewed.rows.set_index("row_id").loc[development.row_id].reset_index()
    elements = list(EPIT_COMPOSITION_ELEMENTS)
    families = list(FAMILY_DISPLAY)
    family_lookup = {display: name for name, display in FAMILY_DISPLAY.items()}
    unique = {}
    for row in data.to_dict("records"):
        values = tuple(float(row[element]) for element in elements)
        key = tuple(round(value, 6) for value in values)
        family = family_lookup[str(row["Material class"])]
        if key not in unique:
            unique[key] = {
                "template_id": f"epit_development_{len(unique) + 1:04d}",
                "family": family, **dict(zip(elements, values, strict=True)), "row_ids": [],
            }
        if unique[key]["family"] != family:
            raise RuntimeError("Identical development compositions have conflicting families.")
        unique[key]["row_ids"].append(row["row_id"])
    templates = list(unique.values())
    counts = {family: sum(row["family"] == family for row in templates) for family in families}
    composition_csv = _csv_bytes(["template_id", "family", *elements], templates)
    scope = {
        "outer_partition": "development",
        "source_row_ids": development.row_id.tolist(),
        "source_task_row_indices": development.task_row_index.tolist(),
        "source_rows": len(development),
        "final_query_rows_used": 0,
        "targets_used": False,
        "split_manifest_sha256": reviewed.identity["split_manifest_sha256"],
    }
    composition = {
        "profile_name": COMPOSITION_NAME, "profile_version": 4,
        "description": "Reviewed compositions from frozen EPIT development rows only.",
        "source_dataset": "electrochemical_metrics_alloys", "source_sheet": "Pitting Potential",
        "reviewed_composition": reviewed.identity, "sampling_scope": scope,
        "usable_rows": len(data), "distinct_templates": len(templates),
        "deduplication": "First saved vector per 6-decimal key within development; saved amounts unchanged.",
        "elements": elements, "observed_elements": elements, "omitted_elements": [],
        "families": families, "family_display_names": FAMILY_DISPLAY,
        "family_template_counts": counts,
        "pretraining_template_policy": {
            "closure_target_wt_percent": 100.0, "closure_tolerance_wt_percent": 1.0,
            "missing_values_for_eligible_templates": "None; reviewed amounts were already finite.",
            "evaluation_rows_removed": 0, "eligible_templates": len(templates),
            "excluded_template_ids": [], "eligible_family_template_counts": counts,
        },
        "default_family_probabilities": {family: count / len(templates) for family, count in counts.items()},
        "template_source_row_ids": {row["template_id"]: row["row_ids"] for row in templates},
        "csv_file": f"{COMPOSITION_NAME}.csv",
        "csv_sha256": hashlib.sha256(composition_csv).hexdigest(),
        "builder": "scripts/build_epit_development_profiles.py",
        "normalization": "Stored amounts unchanged; synthetic perturbation retained its existing closure to 100 wt.%.",
    }
    composition_json = _json_bytes(composition)

    # Keep the previous Fe/Ni family selection and row-wise environment tuples.
    selected = data.loc[data["Material class"].isin(FAMILY_DISPLAY_NAMES.values())]
    selected_counts = selected["Material class"].value_counts().to_dict()
    if selected_counts != {"Fe Alloy": 411, "NiCrMo Alloy": 41}:
        raise RuntimeError("Unexpected Fe/Ni-Cr-Mo development sampling counts.")
    numeric_columns = [*EPIT_COMPOSITION_COLUMNS, *ENVIRONMENT_SOURCE_COLUMNS]
    means = {
        column: float(data[element].mean())
        for column, element in zip(EPIT_COMPOSITION_COLUMNS, elements, strict=True)
    }
    for column in ENVIRONMENT_SOURCE_COLUMNS:
        values = np.asarray([eval_to_float(v, column=column, group="environment") for v in data[column]])
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise RuntimeError(f"Development rows have no finite values for {column}.")
        means[column] = float(finite.mean())
    categories = sorted({_normalized_method(v) for v in data[METHOD_COLUMN]})
    mapping = {category: index for index, category in enumerate(categories)}
    records = []
    missing_counts = {column: 0 for column in ENVIRONMENT_OUTPUT_COLUMNS}
    for row in selected.to_dict("records"):
        values = [eval_to_float(row[c], column=c, group="environment") for c in ENVIRONMENT_SOURCE_COLUMNS]
        method = _normalized_method(row[METHOD_COLUMN])
        records.append({
            "environment_id": f"epit_development_environment_{int(row['task_row_index']) + 1:06d}",
            "task_row_index": int(row["task_row_index"]), "source_no": row["No."],
            "family": family_lookup[row["Material class"]],
            **{c: _number(v) for c, v in zip(ENVIRONMENT_OUTPUT_COLUMNS, values, strict=True)},
            "test_method": "" if method == "<NA>" else method,
            "test_method_family": _method_family(None if method == "<NA>" else method),
            "test_method_category": mapping[method],
        })
        for column, value in zip(ENVIRONMENT_OUTPUT_COLUMNS, values, strict=True):
            missing_counts[column] += int(not np.isfinite(value))
    feature_csv = _csv_bytes([
        "environment_id", "task_row_index", "source_no", "family", *ENVIRONMENT_OUTPUT_COLUMNS,
        "test_method", "test_method_family", "test_method_category",
    ], records)
    features = {
        "profile_name": FEATURE_NAME, "profile_version": 3,
        "description": "Fe/Ni-Cr-Mo compositions and environment tuples from frozen development rows only.",
        "source_dataset": "electrochemical_metrics_alloys", "source_sheet": "Pitting Potential",
        "reviewed_composition": reviewed.identity, "sampling_scope": {
            **scope, "source_row_ids": selected.row_id.tolist(),
            "source_task_row_indices": selected.task_row_index.tolist(), "source_rows": len(selected),
        },
        "default_composition_profile": COMPOSITION_NAME,
        "usable_task_rows": len(data), "environment_rows": len(selected),
        "families": list(FAMILY_DISPLAY_NAMES), "family_display_names": FAMILY_DISPLAY_NAMES,
        "family_environment_counts": {family: selected_counts[display] for family, display in FAMILY_DISPLAY_NAMES.items()},
        "environment_sampling": "Sample complete observed tuples from the pooled eligible development rows, independently of composition as before.",
        "source_environment_columns": list(ENVIRONMENT_SOURCE_COLUMNS),
        "environment_columns": list(ENVIRONMENT_OUTPUT_COLUMNS),
        "environment_missing_counts": missing_counts, "missing_value_encoding": "empty CSV field",
        "composition_profile": {
            "name": COMPOSITION_NAME, "json_sha256": hashlib.sha256(composition_json).hexdigest(),
            "csv_sha256": composition["csv_sha256"], "allowed_families": list(FAMILY_DISPLAY_NAMES),
            "allowed_family_template_counts": {family: counts[family] for family in FAMILY_DISPLAY_NAMES},
        },
        "evaluator_preprocessing": {
            "numeric_columns": numeric_columns,
            "numeric_missing": "Means fitted on the 608 outer development rows only.",
            "numeric_imputation_means": means,
            "categorical_column": METHOD_COLUMN, "categorical_missing_token": "<NA>",
            "categorical_encoding": "Sorted categories from the 608 development rows only; no final-query categories.",
            "category_count": len(mapping), "category_mapping": mapping,
            "profile_category_count": len({r['test_method_category'] for r in records}),
            "source_row_ids": development.row_id.tolist(),
            "composition_handling": "Reviewed numeric compositions were used verbatim.",
        },
        "csv_file": f"{FEATURE_NAME}.csv", "csv_sha256": hashlib.sha256(feature_csv).hexdigest(),
        "builder": "scripts/build_epit_development_profiles.py",
        "scope": "Outer-development sampling only; 28-column feature layout, family restrictions and formulas unchanged.",
    }
    return {
        f"{COMPOSITION_NAME}.csv": composition_csv, f"{COMPOSITION_NAME}.json": composition_json,
        f"{FEATURE_NAME}.csv": feature_csv, f"{FEATURE_NAME}.json": _json_bytes(features),
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
                raise RuntimeError(f"Development-only EPIT asset differs from its inputs: {name}")
        print("Development-only EPIT sampling assets match their frozen inputs.")
        return
    if any((args.output_dir / name).exists() for name in assets):
        raise FileExistsError("Refusing to overwrite an existing sampling asset.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in assets.items():
        (args.output_dir / name).write_bytes(content)
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
