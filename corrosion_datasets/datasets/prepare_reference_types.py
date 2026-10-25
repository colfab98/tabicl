"""Add the reviewed reference_type feature to standalone EPIT and Soccol exports.

Reported targets, existing predictors, accepted row subsets and frozen split
assignments are preserved. Unknown reference categories remain missing.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import openpyxl
import pandas as pd

from scripts.epit_pipeline.artifact_hashes import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RULES_PATH = HERE / "reference_type_rules_v1.json"
EVIDENCE_DIR = HERE.parent / "analysis/epit_soccol_reference_review_v1"
EPIT_WORKBOOK = HERE / "electrochemical_metrics_alloys/raw/CRA_database_Scientific_Data_Publication_12102020.xlsx"
LEGACY_SHARED = HERE / "combined_epit_soccol_mpea/processed/epit_soccol_context.csv"
EXPORTS = {
    "epit": {
        "directory": "electrochemical_metrics_alloys",
        "input_version": "epit_composition_v2",
        "output_version": "epit_reference_v1",
        "reference_table": "epit_processed.csv",
        "updated_tables": ["epit_processed.csv", "epit_features.csv"],
    },
    "soccol": {
        "directory": "soccol_pitting_potential",
        "input_version": "soccol_composition_v1",
        "output_version": "soccol_reference_v1",
        "reference_table": "soccol_processed.csv",
        "updated_tables": [
            "soccol_processed.csv", "soccol_features.csv",
            "soccol_development_accepted.csv", "soccol_development_features.csv",
            "soccol_frozen_final_query.csv", "soccol_frozen_final_query_features.csv",
        ],
    },
}


def assign_reference_types(dataset: str, frame: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Return one categorical feature plus separate evidence for each row."""
    if not frame.row_id.is_unique:
        raise ValueError("Duplicate row IDs.")
    records = []
    for row in frame.to_dict("records"):
        if dataset == "epit":
            item = rules["epit"]
            source = row["Reference"]
            ref, status, evidence = item["reference_type"], item["status"], item["evidence"]
            url, location, filling = str(EPIT_WORKBOOK.relative_to(ROOT)), "Pitting Potential!Z1", "saturated KCl"
        elif dataset == "soccol":
            source = row["source"]
            item = rules["soccol_sources"].get(source)
            if item is None:
                ref, status = None, "source_reference_not_verified"
                evidence = "No reviewed source-specific assignment; the reference category was left missing."
                url = location = filling = ""
            else:
                ref, status, evidence = item["reference_type"], item["verification_status"], item["evidence"]
                url, location = item["evidence_url"], item["location"]
                filling = item.get("reference_filling_solution") or ""
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if ref is not None and ref not in rules["categories"]:
            raise ValueError(f"Unsupported reference category: {ref}")
        records.append({"row_id": row["row_id"], "source": source, "reference_type": ref,
                        "verification_status": status, "evidence": evidence,
                        "evidence_url": url, "evidence_location": location,
                        "reference_filling_solution": filling, "target_conversion_applied": False})
    return pd.DataFrame(records)


def prepare(dataset: str, output_dir: Path | None = None) -> dict:
    config = EXPORTS[dataset]
    input_dir = HERE / config["directory"] / "processed" / config["input_version"]
    output_dir = (output_dir or input_dir.parent / config["output_version"]).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")
    previous = json.loads((input_dir / "processing_manifest.json").read_text())
    for name, digest in previous["output_sha256"].items():
        if sha256_file(input_dir / name) != digest:
            raise RuntimeError(f"Input export checksum mismatch: {name}")
    rules = json.loads(RULES_PATH.read_text())
    review = json.loads((EVIDENCE_DIR / "summary.json").read_text())
    if sha256_file(RULES_PATH) != review["rules_sha256"]:
        raise RuntimeError("Reviewed reference rules changed.")
    for name, digest in review["evidence_file_sha256"].items():
        if sha256_file(EVIDENCE_DIR / name) != digest:
            raise RuntimeError(f"Reviewed reference evidence changed: {name}")
    manuscript = ROOT / rules["collection_conflict"]["local_file"]
    if sha256_file(manuscript) != review["local_collection_manuscript_sha256"]:
        raise RuntimeError("Collection manuscript changed.")
    paths = [*sorted(p for p in input_dir.iterdir() if p.is_file()), RULES_PATH,
             *sorted(p for p in EVIDENCE_DIR.iterdir() if p.is_file()),
             EPIT_WORKBOOK, manuscript, LEGACY_SHARED]
    hashes = {str(p.resolve()): sha256_file(p) for p in paths}
    workbook = openpyxl.load_workbook(EPIT_WORKBOOK, read_only=True, data_only=True)
    try:
        if workbook["Pitting Potential"]["Z1"].value != "Epit, mV (SCE)":
            raise ValueError("The original EPIT target-reference declaration changed.")
    finally:
        workbook.close()

    frame = pd.read_csv(input_dir / config["reference_table"], dtype=str, keep_default_na=False)
    if "reference_type" in frame:
        raise ValueError("Input already contained reference_type.")
    audit = assign_reference_types(dataset, frame, rules)
    values = audit.set_index("row_id").reference_type
    # Compare by workbook row ID; several source labels are duplicated.
    legacy = pd.read_csv(LEGACY_SHARED, dtype=str, keep_default_na=False)
    if dataset == "soccol":
        legacy = legacy.loc[legacy.source_dataset.eq("Soccol")].copy()
        keys = frame.raw_workbook_row.map(lambda value: f"soccol_r{int(value):06d}")
        if legacy.row_id.duplicated().any() or keys.duplicated().any() or set(keys) != set(legacy.row_id):
            raise ValueError("Cannot uniquely align Soccol with the previous reference assignments.")
        legacy = legacy.set_index("row_id")
        if (keys.map(legacy.source_row_id).str.strip().tolist() != frame.label.str.strip().tolist()
                or keys.map(legacy.source_reference).tolist() != frame.source.tolist()):
            raise ValueError("Soccol source identity changed.")
        expected = keys.map(legacy.reference_type).replace("NaN", "")
        if expected.tolist() != audit.reference_type.fillna("").tolist():
            raise ValueError("Reviewed assignments differed from the agreed previous approach.")
    elif not audit.reference_type.eq("SCE").all():
        raise ValueError("EPIT reference assignments disagreed with its target header.")

    registry = json.loads((input_dir / "feature_registry.json").read_text())
    registry["schema_version"] = config["output_version"]
    registry["feature_columns"].append("reference_type")
    registry["measurement_columns"] = ["reference_type"]
    registry["columns"]["reference_type"] = {
        "group": "measurement", "dtype": "categorical", "unit": None,
        "categories": rules["categories"],
        "meaning": "Adopted reference-electrode category for the unchanged reported pitting-potential target.",
        "qualification": rules["limits"],
        "missing": "Unverified source; empty CSV field read as NaN.",
    }
    registry["reference_type_policy"] = rules["policy"]
    counts = dict(previous["counts"])
    counts["reference_type_counts"] = audit.reference_type.fillna("missing").value_counts().to_dict()
    counts["reference_type_feature_count"] = 1
    count_key = "total_features_with_composition" if dataset == "epit" else "total_features"
    counts[count_key] = len(registry["feature_columns"])
    if "feature_coverage" in counts:
        counts["feature_coverage"]["reference_type"] = int(audit.reference_type.notna().sum())

    updated = {}
    subset_counts = {}
    for name in config["updated_tables"]:
        original = pd.read_csv(input_dir / name, dtype=str, keep_default_na=False)
        if not original.row_id.isin(values.index).all() or "reference_type" in original:
            raise ValueError(f"Unexpected feature-view rows or columns: {name}")
        table = original.assign(reference_type=original.row_id.map(values))
        pd.testing.assert_frame_equal(original, table.drop(columns="reference_type"))
        updated[name] = table
        subset_counts[name] = table.reference_type.fillna("missing").value_counts().to_dict()
    counts["reference_type_by_table"] = subset_counts
    shutil.copytree(input_dir, output_dir)
    for name, table in updated.items():
        table.to_csv(output_dir / name, index=False)
    audit.to_csv(output_dir / "reference_type_audit.csv", index=False)
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Input changed during processing: {path}")
    manifest = {
        "schema_version": config["output_version"], "previous_version": config["input_version"],
        "counts": counts, "input_sha256": hashes,
        "implementation_sha256": {str(Path(__file__).resolve()): sha256_file(Path(__file__))},
        "output_sha256": {p.name: sha256_file(p) for p in sorted(output_dir.iterdir())
                          if p.is_file() and p.name != "processing_manifest.json"},
        "policy": {"inherited_policy": previous["policy"], "reference_type": rules["policy"],
                   "verification_limits": rules["limits"]},
        "scope": "Added reference_type and a separate evidence audit. Existing values, targets, row subsets and frozen assignments were preserved; no target conversion or new exclusion was applied.",
        "historical_records": "Inherited logs describe previous composition/environment stages. Environment-only views retain their original scope; reference_type appears in the full model feature views.",
        "existing_pipeline_paths_changed": False,
    }
    (output_dir / "processing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=EXPORTS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.dataset, args.output_dir)
    print(json.dumps({"schema_version": manifest["schema_version"],
                      "reference_type_counts": manifest["counts"]["reference_type_counts"]}, indent=2))


if __name__ == "__main__":
    main()
