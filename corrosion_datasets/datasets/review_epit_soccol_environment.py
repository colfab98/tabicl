"""Apply the source-consistency review to the standalone EPIT/Soccol exports.

Prior exports are immutable inputs. This review neither changes model loaders
nor fits anything, selects rows, changes composition, or recomputes splits.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.epit_pipeline.artifact_hashes import sha256_file

HERE = Path(__file__).resolve().parent
EXPORTS = {
    "epit": ("electrochemical_metrics_alloys", "epit_environment_v2", "epit_environment_v3",
             "epit_processed.csv", "epit_features.csv"),
    "soccol": ("soccol_pitting_potential", "soccol_environment_v1", "soccol_environment_v2",
               "soccol_processed.csv", "soccol_environment_features.csv"),
}
LOG_COLUMNS = ["row_id", "source_column", "source_text", "status", "extracted_features", "notes"]


def review(dataset: str, source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Check source fields only; the input uses strings to preserve source notation."""
    frame = source.copy()
    changes, log = [], []

    def change(index, column, value, source_columns, status, reason):
        previous = frame.at[index, column]
        frame.at[index, column] = value
        source_text = json.dumps({name: source.at[index, name] for name in source_columns})
        changes.append(dict(row_id=source.at[index, "row_id"], column=column,
                            previous_value=previous, reviewed_value=value,
                            reason=reason, source_text=source_text))
        log.append(dict(row_id=source.at[index, "row_id"], source_column=" / ".join(source_columns),
                        source_text=source_text, status=status,
                        extracted_features=json.dumps({column: None if value == "" else value}), notes=reason))

    if dataset == "epit":
        # All other explicit molar chloride recipes agreed within source precision.
        # Preserve both contradictory source reports rather than choose one.
        mask = (frame["Test Solution"].eq("deareated (by H2) 1 M HCl")
                & pd.to_numeric(frame["[Cl-] M"], errors="coerce").eq(0))
        if frame.loc[mask, "row_id"].tolist() != ["epit_r000148"]:
            raise ValueError("The reviewed EPIT chloride discrepancy changed.")
        for i in frame.index[mask]:
            change(i, "CP_Cl", "", ["Test Solution", "[Cl-] M"], "conflict",
                   "The source named 1 M HCl but recorded zero chloride. CP_Cl was left missing; "
                   "the original solution description and numeric report were preserved.")
    elif dataset == "soccol":
        # 'air' can coexist with a separate electrochemical treatment: 36 rows
        # also report a preparation potential and pH. Do not erase their time.
        mask = (frame.raw_Prep_medium.eq("air") & frame.raw_Prep_time.ne("")
                & frame.raw_Prep_redox.eq("") & frame.raw_Prep_pH.eq(""))
        if mask.sum() != 217:
            raise ValueError("The reviewed Soccol air-only timing cases changed.")
        for i in frame.index[mask]:
            change(i, "Prep_time", "", ["raw_Prep_medium", "raw_Prep_time", "raw_Prep_redox", "raw_Prep_pH"],
                   "unresolved", "Only air was named, without a preparation potential or pH identifying "
                   "a separate solution treatment. The duration could not be assigned to the shared "
                   "chemical/electrochemical Prep_time field and was left missing. It was not moved to CP_time.")
        for i, value in frame.surface_preparation_method.items():
            canonical = " + ".join(sorted(set(value.split(" + "))))
            if canonical != value:
                change(i, "surface_preparation_method", canonical, ["raw_Prep_medium"], "normalized",
                       "Multiple preparation operations were sorted as in EPIT; order did not represent chronology.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return frame, pd.DataFrame(changes), pd.DataFrame(log, columns=LOG_COLUMNS)


def prepare(dataset: str, output_dir: Path | None = None) -> dict:
    directory, previous_version, version, reference_name, features_name = EXPORTS[dataset]
    input_dir = HERE / directory / "processed" / previous_version
    output_dir = (output_dir or input_dir.parent / version).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")
    manifest = json.loads((input_dir / "processing_manifest.json").read_text())
    if manifest["schema_version"] != previous_version:
        raise ValueError("Unexpected input export version.")
    for name, digest in manifest["output_sha256"].items():
        if sha256_file(input_dir / name) != digest:
            raise RuntimeError(f"Input export checksum mismatch: {name}")
    paths = [p for p in sorted(input_dir.iterdir()) if p.is_file()]
    hashes = {str(p): sha256_file(p) for p in paths}
    source = pd.read_csv(input_dir / reference_name, dtype=str, keep_default_na=False)
    frame, changes, added_log = review(dataset, source)
    registry = json.loads((input_dir / "feature_registry.json").read_text())
    registry["schema_version"] = version
    registry["columns"]["Prep_time"]["meaning"] = (
        "Chemical/electrochemical preparation duration in hours, separate from heat treatment, "
        "air exposure and test-bath equilibration. Air-only timing with no separate treatment "
        "evidence remained missing."
    )
    registry["review_policy"] = "Source contradictions and uncertain timing associations remained missing; original reports were preserved."
    if dataset == "soccol":
        registry["columns"]["surface_preparation_method"]["meaning"] = (
            "Explicit operations, alphabetically sorted and joined with + as in EPIT; no inferred chronology."
        )
    audit = pd.concat([pd.read_csv(input_dir / "environment_processing_log.csv", keep_default_na=False), added_log], ignore_index=True)
    coverage = pd.read_csv(input_dir / "environment_column_summary.csv", keep_default_na=False)
    for i, name in coverage.column.items():
        coverage.at[i, "populated_rows"] = int(frame[name].ne("").sum())
        coverage.at[i, "distinct_values"] = int(frame.loc[frame[name].ne(""), name].nunique())
        for key in ["meaning", "qualification"]:
            if key in coverage and key in registry["columns"][name]:
                coverage.at[i, key] = registry["columns"][name][key]
    counts = manifest["counts"]
    counts["feature_coverage"] = dict(zip(coverage.column, coverage.populated_rows.astype(int)))
    counts["conflicting_rows"] = int(audit.loc[audit.status.eq("conflict"), "row_id"].nunique())
    counts["source_status_rows"] = {
        field: group.status.value_counts().to_dict() for field, group in audit.groupby("source_column", sort=False)
    }
    counts["review_changed_cells"] = len(changes)
    counts["review_changed_rows"] = int(changes.row_id.nunique())
    counts["review_changes_by_column"] = changes.column.value_counts().to_dict()
    counts["review_newly_missing_cells"] = int(changes.reviewed_value.eq("").sum())
    # Raw fields, targets, composition, row ordering and frozen assignments are
    # inherited; each changed processed cell is enumerated in review_changes.csv.
    shutil.copytree(input_dir, output_dir)
    frame.to_csv(output_dir / reference_name, index=False)
    frame[["row_id", *registry["feature_columns"]]].to_csv(output_dir / features_name, index=False)
    audit.to_csv(output_dir / "environment_processing_log.csv", index=False)
    audit.loc[audit.status.isin(["partial", "unresolved", "conflict"])].to_csv(output_dir / "environment_review_notes.csv", index=False)
    changes.to_csv(output_dir / "review_changes.csv", index=False)
    coverage.to_csv(output_dir / "environment_column_summary.csv", index=False)
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    for p in paths:
        if sha256_file(p) != hashes[str(p)]:
            raise RuntimeError(f"Input changed during review: {p}")
    reviewed = {
        "schema_version": version, "previous_version": previous_version, "counts": counts,
        "input_sha256": hashes, "implementation_sha256": {str(Path(__file__).resolve()): sha256_file(Path(__file__))},
        "output_sha256": {p.name: sha256_file(p) for p in sorted(output_dir.iterdir()) if p.name != "processing_manifest.json"},
        "policy": {"inherited_policy": manifest["policy"], "source_review": registry["review_policy"]},
        "scope": "Reviewed standalone export; source files, prior exports, targets, composition, frozen splits and existing model paths were preserved.",
    }
    (output_dir / "processing_manifest.json").write_text(json.dumps(reviewed, indent=2) + "\n")
    return reviewed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=list(EXPORTS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.dataset, args.output_dir)
    print(json.dumps({"schema_version": manifest["schema_version"],
                      "changes": manifest["counts"]["review_changes_by_column"]}, indent=2))


if __name__ == "__main__":
    main()
