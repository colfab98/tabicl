"""Extend the reviewed EPIT export with preparation and explicit ion information.

This writes epit_environment_v2 and preserves the previous export. Extraction
uses reviewed procedure clauses, never targets, composition or test outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from .prepare_epit_environment import (
    DEFAULT_OUTPUT_DIR as ENVIRONMENT_V1,
    HERE,
    MISSING,
    REPO_ROOT,
    SPLIT_MANIFEST,
    load_frozen_split,
    sha256_file,
)

DEFAULT_OUTPUT_DIR = HERE / "processed/epit_environment_v2"
RULES_PATH = HERE / "epit_preparation_rules_v1.json"
NEW_COLUMNS = [
    "Prep_grinding_grit", "surface_preparation_method", "preparation_solution_type",
    "Prep_time", "CP_time", "CP_agitation", "CP_Br", "CP_SO4",
]
TEXT_COLUMNS = ["Heat treatment", "[Cl-] Test Method", "Comments"]
LOG_COLUMNS = ["row_id", "source_column", "source_text", "status", "extracted_features", "notes"]


def column_definition(name: str) -> dict:
    definitions = {
        "Prep_grinding_grit": dict(group="preparation", dtype="numeric", unit="reported grit grade",
            meaning="Reported numeric abrasive grade; no common grading standard was assumed. Mesh and slash/emery notations were not converted."),
        "surface_preparation_method": dict(group="preparation", dtype="categorical", unit=None,
            meaning="Explicit preparation operations. A + separated list records multiple operations, not an inferred chronology or complete history."),
        "preparation_solution_type": dict(group="preparation", dtype="categorical", unit=None,
            meaning="Named chemical/electrochemical treatment or cleaning liquid. Rinse water and abrasive carriers were not added to an active treatment mixture."),
        "Prep_time": dict(group="preparation", dtype="numeric", unit="h",
            meaning="Explicit chemical/electrochemical conditioning duration; distinct from heating, air exposure, test duration and bath equilibration."),
        "CP_time": dict(group="test_environment", dtype="numeric", unit="h",
            meaning="Time in the test solution at open circuit or before the scan. Approximate reported times remained nominal estimates."),
        "CP_agitation": dict(group="test_environment", dtype="numeric", unit="binary",
            meaning="1 explicitly stirred or rotating electrode, 0 explicitly quiescent; missing otherwise. Stationary current and gas bubbling did not establish agitation state."),
        "CP_Br": dict(group="test_environment", dtype="numeric", unit="mol/L",
            meaning="Nominal bromide amount from explicit NaBr molarity, one Br per NaBr. Normality was not converted."),
        "CP_SO4": dict(group="test_environment", dtype="numeric", unit="mol/L",
            meaning="Nominal sulfate amount from explicit Na2SO4 molarity in a recipe without a named acid. No free-ion speciation calculation or acid contribution was inferred."),
    }
    return definitions[name]


def load_rules(path: Path = RULES_PATH) -> dict:
    doc = json.loads(path.read_text())
    if doc["schema_version"] != "epit_preparation_rules_v1":
        raise ValueError("Unsupported EPIT preparation rules.")
    result = {}
    for field, entries in doc["fields"].items():
        by_text = {entry["source_text"]: entry for entry in entries}
        if len(entries) != len(by_text):
            raise ValueError(f"Duplicate descriptions in {field}")
        for entry in entries:
            if set(entry["features"]) - set(NEW_COLUMNS[:6]):
                raise ValueError("Procedure rules can only populate the six approved preparation columns.")
            for clause in entry.get("evidence_clauses", []):
                if clause not in entry["source_text"]:
                    raise ValueError(f"Evidence clause absent from {field}: {clause!r}")
            if entry["features"] and not entry.get("evidence_clauses"):
                raise ValueError("A nonempty extraction requires explicit procedure evidence.")
            if set(entry.get("ambiguous_columns", [])) - set(NEW_COLUMNS[:6]):
                raise ValueError("Unexpected ambiguous column.")
        result[field] = by_text
    if set(result) != set(TEXT_COLUMNS):
        raise ValueError("Unexpected preparation source fields.")
    return result


def reviewed_rule(rules: dict, field: str, text: str) -> dict:
    if text in MISSING:
        return dict(features={}, status="missing", notes=["Missing source information."], ambiguous_columns=[])
    try:
        return rules[field][text]
    except KeyError as exc:
        raise ValueError(f"Unreviewed EPIT preparation description in {field}: {text!r}") from exc


def optional_number(value: str) -> float | None:
    if value in MISSING:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid reported concentration: {value!r}")
    return result


def ion_features(row: dict) -> tuple[dict, list[dict]]:
    """Only unambiguous stoichiometric mappings from already reviewed recipes."""
    values = {"CP_Br": None, "CP_SO4": None}
    audit = []
    bromide = optional_number(row["NaBr_mol_L"])
    values["CP_Br"] = bromide
    audit.append(dict(row_id=row["row_id"], source_column="NaBr_mol_L / Test Solution",
                      source_text=row["Test Solution"], status="extracted" if bromide is not None else "missing",
                      extracted_features=json.dumps({"CP_Br": bromide}),
                      notes="One mole of explicitly reported NaBr supplied one mole of nominal bromide. Normality, preparation baths and unreported amounts were not converted."))
    sulfate = optional_number(row["Na2SO4_mol_L"])
    components = set(row["test_solution_type"].split(" + "))
    allowed = sulfate is not None and "Na2SO4" in components and components <= {"Na2SO4", "NaCl"}
    if allowed:
        values["CP_SO4"] = sulfate
    audit.append(dict(row_id=row["row_id"], source_column="Na2SO4_mol_L / test_solution_type",
                      source_text=row["Test Solution"], status="extracted" if allowed else "partial" if sulfate is not None else "missing",
                      extracted_features=json.dumps({"CP_SO4": values["CP_SO4"]}),
                      notes="Nominal sulfate was mapped 1:1 from reported Na2SO4 only for the reviewed Na2SO4/NaCl recipe. Acid mixtures stayed unknown; H2SO4 amount was not equated to free sulfate or added to the salt amount."))
    return values, audit


def transform(frame: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(NEW_COLUMNS) & set(frame):
        raise ValueError("Input already contains the added preparation columns.")
    records, audit = [], []
    for row in frame.to_dict("records"):
        values = dict.fromkeys(NEW_COLUMNS)
        conflicts, ambiguous, methods = set(), set(), set()
        for field in TEXT_COLUMNS:
            rule = reviewed_rule(rules, field, row[field])
            status, notes = rule["status"], list(rule["notes"])
            accepted = dict(rule["features"])
            ambiguous.update(rule.get("ambiguous_columns", []))
            for name, value in rule["features"].items():
                if name == "surface_preparation_method":
                    methods.update(value.split(" + "))
                    continue
                previous = values[name]
                if name in conflicts:
                    accepted[name] = None
                    status = "conflict"
                elif previous is None or previous == value or (
                    isinstance(value, (int, float)) and isinstance(previous, (int, float))
                    and math.isclose(previous, value, rel_tol=1e-9, abs_tol=1e-12)
                ):
                    values[name] = value
                else:
                    values[name] = accepted[name] = None
                    conflicts.add(name)
                    status = "conflict"
                    notes.append(f"Different reports for {name}: {previous!r} and {value!r}. Kept missing rather than choosing, summing or averaging stages.")
            audit.append(dict(row_id=row["row_id"], source_column=field + " / preparation",
                              source_text=row[field], status=status,
                              extracted_features=json.dumps(accepted), notes=" ".join(notes)))
        if methods:
            values["surface_preparation_method"] = " + ".join(sorted(methods))
        if ambiguous:
            for name in ambiguous:
                values[name] = None
            audit.append(dict(row_id=row["row_id"], source_column="preparation duration qualification",
                              source_text="", status="unresolved",
                              extracted_features=json.dumps(dict.fromkeys(sorted(ambiguous))),
                              notes="A bound or multiple preparation stages prevented a single exact duration; numeric values from another preparation step were not substituted."))
        ions, ion_audit = ion_features(row)
        values.update(ions)
        audit.extend(ion_audit)
        records.append(values)
    result = pd.DataFrame(records, columns=NEW_COLUMNS, index=frame.index)
    for name in NEW_COLUMNS:
        if column_definition(name)["dtype"] == "numeric":
            result[name] = pd.to_numeric(result[name], errors="raise")
    return pd.concat([frame.copy(), result], axis=1), pd.DataFrame(audit, columns=LOG_COLUMNS)


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR, input_dir: Path = ENVIRONMENT_V1) -> dict:
    output_dir, input_dir = output_dir.expanduser().resolve(), input_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")
    old_manifest = json.loads((input_dir / "processing_manifest.json").read_text())
    if old_manifest["schema_version"] != "epit_environment_v1":
        raise ValueError("Expected the reviewed EPIT environment v1 export.")
    for name, digest in old_manifest["output_sha256"].items():
        if sha256_file(input_dir / name) != digest:
            raise RuntimeError(f"Previous EPIT export changed: {name}")
    frozen = load_frozen_split(SPLIT_MANIFEST)
    split_path = SPLIT_MANIFEST.parent / frozen.lock["assignments"]["file"]
    assignments = pd.read_csv(input_dir / "split_assignments.csv", dtype=str, keep_default_na=False)
    original_split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(assignments.drop(columns="row_id"), original_split[assignments.columns.drop("row_id")])
    frame = pd.read_csv(input_dir / "epit_processed.csv", dtype=str, keep_default_na=False)
    if len(frame) != 760 or frame.row_id.tolist() != assignments.row_id.tolist():
        raise ValueError("EPIT rows no longer match the frozen assignments.")
    paths = [input_dir / name for name in old_manifest["output_sha256"]]
    paths += [input_dir / "processing_manifest.json", RULES_PATH, SPLIT_MANIFEST, frozen.lock_path, split_path]
    hashes = {str(path): sha256_file(path) for path in paths}
    processed, added_log = transform(frame, load_rules())
    registry = json.loads((input_dir / "feature_registry.json").read_text())
    registry["schema_version"] = "epit_environment_v2"
    registry["feature_columns"] += NEW_COLUMNS
    registry["environment_history_columns"] += NEW_COLUMNS
    registry["columns"].update({name: column_definition(name) for name in NEW_COLUMNS})
    registry["preparation_policy"] = "Only explicit procedure information was extracted. Prep_redox, Prep_pH and Prep_Ra_micron remained deferred. Unknown agitation was not set to zero."
    registry["preparation_source_fields"] = TEXT_COLUMNS
    registry["comments_policy"] = "Only reviewed preparation clauses supplied added features. Targets, composition notes and comments on corrosion outcomes did not supply feature values."
    registry["ion_policy"] = "NaBr molarity mapped to nominal bromide; only the Na2SO4/NaCl recipe without a named acid mapped to nominal sulfate. No normality conversion or acid speciation inference."
    audit = pd.concat([pd.read_csv(input_dir / "environment_processing_log.csv", keep_default_na=False), added_log], ignore_index=True)
    derived = registry["environment_history_columns"] + registry["audit_only_derived_columns"]
    coverage = pd.DataFrame([
        dict(column=name, **registry["columns"][name],
             populated_rows=int((~processed[name].isin(MISSING) & processed[name].notna()).sum()),
             distinct_values=int(processed.loc[~processed[name].isin(MISSING), name].nunique()),
             model_feature=name in registry["feature_columns"]) for name in derived
    ])
    counts = dict(old_manifest["counts"])
    counts.update({
        "derived_columns": len(derived),
        "environment_history_features": len(registry["environment_history_columns"]),
        "total_features_with_composition": len(registry["feature_columns"]),
        "feature_coverage": dict(zip(coverage.column, coverage.populated_rows)),
        "added_columns": NEW_COLUMNS,
        "added_feature_coverage": {name: int(processed[name].notna().sum()) for name in NEW_COLUMNS},
        "conflicting_rows": int(audit.loc[audit.status.eq("conflict"), "row_id"].nunique()),
        "preparation_conflicting_rows": int(added_log.loc[added_log.status.eq("conflict"), "row_id"].nunique()),
        "rows_with_preparation_method": int(processed.surface_preparation_method.notna().sum()),
        "preparation_source_fields": {
            field: {"populated_rows": int((~frame[field].isin(MISSING)).sum()),
                    "distinct_descriptions": int(frame.loc[~frame[field].isin(MISSING), field].nunique())}
            for field in TEXT_COLUMNS
        },
        "source_status_rows": {field: audit.loc[audit.source_column.eq(field), "status"].value_counts().to_dict() for field in audit.source_column.unique()},
    })
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Input changed while processing: {path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rebuilt = {"epit_processed.csv", "epit_features.csv", "feature_registry.json", "environment_processing_log.csv", "environment_review_notes.csv", "environment_column_summary.csv"}
    for name in old_manifest["output_sha256"]:
        if name not in rebuilt:
            shutil.copyfile(input_dir / name, output_dir / name)
    processed.to_csv(output_dir / "epit_processed.csv", index=False)
    processed[["row_id", *registry["feature_columns"]]].to_csv(output_dir / "epit_features.csv", index=False)
    audit.to_csv(output_dir / "environment_processing_log.csv", index=False)
    added_log.to_csv(output_dir / "preparation_processing_log.csv", index=False)
    audit.loc[audit.status.isin(["partial", "unresolved", "conflict"])].to_csv(output_dir / "environment_review_notes.csv", index=False)
    coverage.to_csv(output_dir / "environment_column_summary.csv", index=False)
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    manifest = {
        "schema_version": "epit_environment_v2", "counts": counts,
        "input_sha256": hashes,
        "implementation_sha256": {str(Path(__file__).resolve().relative_to(REPO_ROOT)): sha256_file(Path(__file__))},
        "output_sha256": {p.name: sha256_file(p) for p in sorted(output_dir.iterdir())},
        "policy": {**old_manifest["policy"], "previous_environment_columns": "Preserved verbatim.",
                   "preparation": registry["preparation_policy"], "ions": registry["ion_policy"]},
        "scope": "Standalone EPIT v2 export. Existing v1, Soccol, shared tables and model/training/evaluation paths were preserved.",
    }
    (output_dir / "processing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-dir", type=Path, default=ENVIRONMENT_V1)
    args = parser.parse_args()
    result = prepare(args.output_dir, args.input_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "added_feature_coverage": result["counts"]["added_feature_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
