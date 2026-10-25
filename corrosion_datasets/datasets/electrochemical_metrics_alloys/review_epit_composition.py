"""Apply the agreed Ti correction and retain the recorded Re and Nb/Ta amounts.

Build a standalone export from environment v3; preserve earlier exports,
all rows, frozen assignments, environment values and existing model paths.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.epit_pipeline.artifact_hashes import sha256_file

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "processed/epit_environment_v3"
DEFAULT_OUTPUT_DIR = HERE / "processed/epit_composition_v2"
EVIDENCE_DIR = HERE.parents[1] / "analysis/epit_composition_source_review_v1"
VERSION = "epit_composition_v2"
TI_ROW = "epit_r000359"
POLICY = {
    "Ti": "Corrected the numeric Ti amount to 1.86 wt.% using the source-table text and alloy description.",
    "Fe": "Updated the corrected row's Fe remainder to 77.612 wt.%.",
    "Re": "Retained all 12 rows and their original zero Re values; exact amounts remained unverified.",
    "Nb_Ta": "Retained all 10 rows and their recorded equal Nb/Ta amounts; separate amounts and combined totals remained unverified.",
    "acceptance": "The user accepted these documented uncertainties for comparison on a common dataset; unresolved evidence did not exclude rows or block this export.",
    "splits": "Preserved all 608 development rows, 152 frozen final queries and development-fold assignments.",
}


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")
    previous = json.loads((INPUT_DIR / "processing_manifest.json").read_text())
    if previous["schema_version"] != "epit_environment_v3":
        raise ValueError("Unexpected input export version.")
    for name, digest in previous["output_sha256"].items():
        if sha256_file(INPUT_DIR / name) != digest:
            raise RuntimeError(f"Input export checksum mismatch: {name}")
    evidence_summary = json.loads((EVIDENCE_DIR / "summary.json").read_text())
    if sha256_file(EVIDENCE_DIR / "row_findings.csv") != evidence_summary["row_findings_sha256"]:
        raise RuntimeError("Source-review evidence changed.")
    for path, digest in evidence_summary["inputs_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Source-review input changed: {path}")
    input_paths = [*sorted(p for p in INPUT_DIR.iterdir() if p.is_file()),
                   EVIDENCE_DIR / "row_findings.csv", EVIDENCE_DIR / "summary.json"]
    hashes = {str(p): sha256_file(p) for p in input_paths}

    # String loading preserves existing notation and missing values exactly.
    source = pd.read_csv(INPUT_DIR / "epit_processed.csv", dtype=str, keep_default_na=False)
    frame = source.copy()
    splits = pd.read_csv(INPUT_DIR / "split_assignments.csv", dtype=str, keep_default_na=False)
    registry = json.loads((INPUT_DIR / "feature_registry.json").read_text())
    findings = pd.read_csv(EVIDENCE_DIR / "row_findings.csv", dtype=str, keep_default_na=False)
    if (len(frame) != 760 or not frame.row_id.is_unique
            or frame.row_id.tolist() != splits.row_id.tolist()
            or splits.outer_split.value_counts().to_dict() != {"development": 608, "final_test": 152}):
        raise ValueError("Unexpected EPIT row membership or frozen split.")
    correction = findings.loc[findings.row_id.eq(TI_ROW)]
    if (len(correction) != 1 or correction.iloc[0].status != "source_text_supports_correction"
            or float(correction.iloc[0].proposed_Ti_wt_percent) != 1.86
            or float(correction.iloc[0].proposed_Fe_wt_percent) != 77.612):
        raise ValueError("Unexpected Ti correction evidence.")
    indices = frame.index[frame.row_id.eq(TI_ROW)]
    if len(indices) != 1:
        raise ValueError("Expected exactly one Ti correction row.")
    index = indices[0]
    changes = []
    for column, old, new in [("Ti", 1.88, "1.86"), ("Fe", 77.592, "77.612")]:
        if float(frame.at[index, column]) != old:
            raise ValueError(f"Unexpected original {column} amount.")
        changes.append({"row_id": TI_ROW, "column": column,
                        "previous_value": frame.at[index, column], "reviewed_value": new,
                        "reason": POLICY[column], "evidence_doi": correction.iloc[0].doi})
        frame.at[index, column] = new
    if abs(frame.loc[index, registry["composition_columns"]].astype(float).sum() - 100) > 1e-9:
        raise ValueError("Corrected composition did not sum to 100 wt.%.")

    retained = findings.loc[findings.status.str.startswith("unresolved")].copy()
    expected_ids = {f"epit_r{n:06d}" for n in [*range(51, 56), *range(61, 66), *range(211, 223)]}
    if (len(retained) != 22 or set(retained.row_id) != expected_ids
            or retained.outer_split.value_counts().to_dict() != {"final_test": 17, "development": 5}):
        raise ValueError("Unexpected retained composition uncertainties.")
    retained = retained.rename(columns={"status": "evidence_status"})
    retained["processing_decision"] = "retained_as_recorded_with_user_accepted_uncertainty"
    retained["included_in_export"] = True
    for record in retained.itertuples():
        actual = frame.loc[frame.row_id.eq(record.row_id)].iloc[0]
        if any(float(actual[c]) != float(getattr(record, c)) for c in ["Re", "Nb", "Ta", "Fe"]):
            raise ValueError(f"Retained composition differed from source review: {record.row_id}")
    changed = source.ne(frame)
    if int(changed.to_numpy().sum()) != 2 or int(changed.any(axis=1).sum()) != 1:
        raise RuntimeError("Changes exceeded the agreed two cells in one row.")

    registry["schema_version"] = VERSION
    registry["composition_review_policy"] = POLICY
    counts = dict(previous["counts"])
    counts.update({"composition_review_changed_cells": 2, "composition_review_changed_rows": 1,
                   "composition_review_excluded_rows": 0, "composition_review_retained_uncertain_rows": 22,
                   "composition_review_retained_uncertain_development_rows": 5,
                   "composition_review_retained_uncertain_final_test_rows": 17})
    shutil.copytree(INPUT_DIR, output_dir)
    frame.to_csv(output_dir / "epit_processed.csv", index=False)
    frame[["row_id", *registry["feature_columns"]]].to_csv(output_dir / "epit_features.csv", index=False)
    pd.DataFrame(changes).to_csv(output_dir / "composition_review_changes.csv", index=False)
    retained.to_csv(output_dir / "composition_retained_uncertainties.csv", index=False)
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Input changed during processing: {path}")
    manifest = {
        "schema_version": VERSION,
        "previous_version": previous["schema_version"],
        "counts": counts,
        "policy": {"inherited_policy": previous["policy"], "composition_review": POLICY},
        "input_sha256": hashes,
        "implementation_sha256": {str(Path(__file__).resolve()): sha256_file(Path(__file__))},
        "output_sha256": {p.name: sha256_file(p) for p in sorted(output_dir.iterdir())
                          if p.is_file() and p.name != "processing_manifest.json"},
        "historical_records": "Inherited logs and composition_processing_manifest.json described earlier processing stages; composition_review_changes.csv recorded this step's two additional corrections.",
        "scope": "Standalone EPIT export with two corrected cells. All rows, frozen splits, targets, environment values and model paths were preserved. Re and Nb/Ta uncertainties were retained by explicit user decision.",
    }
    (output_dir / "processing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "schema_version": VERSION,
                      "changed_cells": manifest["counts"]["composition_review_changed_cells"],
                      "retained_rows": manifest["counts"]["retained_rows"]}, indent=2))


if __name__ == "__main__":
    main()
