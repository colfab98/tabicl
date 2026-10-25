"""Add reviewed EPIT environment/history features to the corrected composition export.

This creates a standalone reference export, without rewiring training or evaluation.
Exact source descriptions are reviewed in epit_environment_rules_v1.json. Unseen
descriptions fail instead of silently acquiring an approximate interpretation.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .prepare_epit_composition import (
    DEFAULT_OUTPUT_DIR as COMPOSITION_DIR,
    ELEMENTS,
    HERE,
    REPO_ROOT,
    SPLIT_MANIFEST,
    TARGET_COLUMN,
    load_frozen_split,
    sha256_file,
)

DEFAULT_OUTPUT_DIR = HERE / "processed/epit_environment_v1"
RULES_PATH = HERE / "epit_environment_rules_v1.json"
DIRECT_FIELDS = {
    "Test Temp. oC": ("CP_temp", "degC"),
    "[Cl-] pH": ("CP_pH", "dimensionless"),
    "[Cl-] M": ("CP_Cl", "mol/L"),
    "Scan Rate mV/s": ("scan_rate", "mV/s"),
}
MISSING = {"", "NA", "None", "nan"}
SCAN_PROCEDURES = {"steady state", "20mV steps by manual increament"}
PHASE_COLUMNS = ["phase_FCC", "phase_BCC", "phase_HCP", "phase_IM"]
HEAT_COLUMNS = [
    f"{prefix}{suffix}"
    for prefix in ("heat_treatment_", "heat_treatment_2_")
    for suffix in ("type", "temperature_C", "duration_h", "cooling")
]


def load_rules(path: Path = RULES_PATH) -> dict[str, dict[str, dict[str, Any]]]:
    document = json.loads(path.read_text())
    result = {}
    for field, entries in document["fields"].items():
        index = {entry["source_text"]: entry for entry in entries}
        if len(index) != len(entries):
            raise ValueError(f"Duplicate source descriptions in {field}")
        result[field] = index
    return result


def reviewed_rule(rules: dict, field: str, text: str) -> dict:
    if text in MISSING:
        return dict(source_text=text, features={}, status="missing", notes=["Source missing marker; absence was not inferred."])
    try:
        return rules[field][text]
    except KeyError as exc:
        raise ValueError(f"Unreviewed EPIT description in {field}: {text!r}") from exc


def numeric_source(value: str, field: str) -> float | None:
    if value in MISSING or (field == "Scan Rate mV/s" and value in SCAN_PROCEDURES):
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Unreviewed numeric notation in {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value in {field}: {value!r}")
    if field in {"[Cl-] M", "Scan Rate mV/s"} and result < 0:
        raise ValueError(f"Negative amount/rate in {field}: {value!r}")
    return result


def column_definition(name: str) -> dict:
    if name in ELEMENTS:
        return dict(group="composition", dtype="numeric", unit="wt.%")
    if name in {v[0] for v in DIRECT_FIELDS.values()}:
        unit = next(unit for target, unit in DIRECT_FIELDS.values() if target == name)
        return dict(group="environment", dtype="numeric", unit=unit)
    if name == "Test_area_cm2":
        return dict(group="procedure", dtype="numeric", unit="cm2")
    if name in PHASE_COLUMNS:
        return dict(group="microstructure", dtype="numeric", unit="binary", meaning="1 present, 0 supported absence, missing unknown")
    if name.startswith("heat_treatment_"):
        if name.endswith("temperature_C"):
            return dict(group="heat_treatment", dtype="numeric", unit="degC")
        if name.endswith("duration_h"):
            return dict(group="heat_treatment", dtype="numeric", unit="h")
        return dict(group="heat_treatment", dtype="categorical", unit=None)
    if name in {"test_solution_type", "CP_aeration"}:
        return dict(group="solution", dtype="categorical", unit=None)
    for suffix, unit in {
        "_mol_L": "mol/L", "_eq_L": "eq/L", "_wt_percent": "wt.%",
        "_partial_pressure_atm": "atm", "_ppm_unspecified": "ppm; basis unspecified",
        "_percent_unspecified": "%; basis unspecified",
    }.items():
        if name.endswith(suffix):
            return dict(group="solution", dtype="numeric", unit=unit)
    raise ValueError(f"Missing column definition: {name}")


def transform(frame: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    derived = set(PHASE_COLUMNS + HEAT_COLUMNS + [value[0] for value in DIRECT_FIELDS.values()])
    for field_rules in rules.values():
        for rule in field_rules.values():
            derived.update(rule["features"])
    ordered = [value[0] for value in DIRECT_FIELDS.values()]
    ordered += ["test_solution_type", "CP_aeration"]
    ordered += sorted(derived - set(ordered) - set(PHASE_COLUMNS + HEAT_COLUMNS))
    ordered += HEAT_COLUMNS + PHASE_COLUMNS
    if set(ordered) & set(frame.columns):
        raise ValueError("Source already contains derived environment columns.")

    extracted, audit = [], []
    for source in frame.to_dict("records"):
        features = dict.fromkeys(ordered)
        for field, (target, _) in DIRECT_FIELDS.items():
            value = numeric_source(source[field], field)
            features[target] = value
            if value is None:
                status = "missing" if source[field] in MISSING else "not_numeric"
                note = "No numeric value was inferred from a missing value or scan procedure."
            else:
                status, note = "copied_numeric", "Source numeric value and its stated unit were retained."
            audit.append(dict(row_id=source["row_id"], source_column=field, source_text=source[field],
                              status=status, extracted_features=json.dumps({target: value}), notes=note))
        for field in rules:
            rule = reviewed_rule(rules, field, source[field])
            accepted = dict(rule["features"])
            notes = list(rule["notes"])
            status = rule["status"]
            for name, value in rule["features"].items():
                existing = features[name]
                if existing is None or existing == value:
                    features[name] = value
                elif name == "CP_aeration" and value == "deaerated" and existing in {
                    "hydrogen_saturated", "nitrogen_saturated", "argon_saturated"
                }:
                    accepted[name] = existing
                    notes.append("Kept the explicitly named gas condition alongside the compatible deaeration report.")
                elif name == "CP_aeration" and existing == "deaerated" and value == "nitrogen_atmosphere":
                    accepted[name] = existing
                    notes.append("Kept the explicit solution deaeration; method also reported nitrogen atmosphere.")
                else:
                    features[name] = None
                    accepted[name] = None
                    notes.append(f"Conflicting reports {existing!r} and {value!r}; {name} remained missing.")
                    status = "conflict"
            audit.append(dict(row_id=source["row_id"], source_column=field, source_text=source[field],
                              status=status, extracted_features=json.dumps(accepted), notes=" ".join(notes)))
        # The workbook labels a perchlorate solution as having 0.1 M chloride.
        # Do not replace that ambiguous report with a guessed zero.
        if source["Test Solution"] == "0.1 M NaCIO4" and features["CP_Cl"] != 0:
            original = features["CP_Cl"]
            features["CP_Cl"] = None
            audit.append(dict(row_id=source["row_id"], source_column="Test Solution / [Cl-] M",
                              source_text=source["Test Solution"] + " / " + source["[Cl-] M"],
                              status="conflict", extracted_features=json.dumps({"CP_Cl": None}),
                              notes=f"The solution identified perchlorate, but the chloride column reported {original}. CP_Cl was left missing pending source verification; zero was not assumed."))
        extracted.append(features)
    values = pd.DataFrame(extracted, columns=ordered, index=frame.index)
    for name in ordered:
        if column_definition(name)["dtype"] == "numeric":
            values[name] = pd.to_numeric(values[name], errors="raise")
    return pd.concat([frame.copy(), values], axis=1), pd.DataFrame(audit)


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR, input_dir: Path = COMPOSITION_DIR) -> dict:
    output_dir, input_dir = output_dir.expanduser().resolve(), input_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")
    composition_manifest = json.loads((input_dir / "processing_manifest.json").read_text())
    if composition_manifest["schema_version"] != "epit_processed_composition_v1":
        raise ValueError("Expected the reviewed EPIT composition export.")
    for name, digest in composition_manifest["output_sha256"].items():
        if sha256_file(input_dir / name) != digest:
            raise RuntimeError(f"Composition export checksum mismatch: {name}")
    frozen = load_frozen_split(SPLIT_MANIFEST)
    split_path = SPLIT_MANIFEST.parent / frozen.lock["assignments"]["file"]
    frame = pd.read_csv(input_dir / "epit_processed.csv", dtype=str, keep_default_na=False)
    assignments = pd.read_csv(input_dir / "split_assignments.csv", dtype=str, keep_default_na=False)
    frozen_assignments = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(assignments.drop(columns="row_id"), frozen_assignments[assignments.columns.drop("row_id")])
    if len(frame) != 760 or frame.row_id.tolist() != assignments.row_id.tolist():
        raise ValueError("EPIT rows no longer match their frozen assignments.")
    input_paths = [input_dir / name for name in composition_manifest["output_sha256"]]
    input_paths += [input_dir / "processing_manifest.json", RULES_PATH, SPLIT_MANIFEST, frozen.lock_path, split_path]
    hashes = {str(path): sha256_file(path) for path in input_paths}
    rules = load_rules()
    processed, audit = transform(frame, rules)
    derived = list(processed.columns[len(frame.columns):])
    # A numeric value with an unknown concentration basis is retained for audit,
    # but is not comparable across datasets and is not a registered predictor.
    audit_only = [name for name in derived if name.endswith("_unspecified")]
    environment = [name for name in derived if name not in audit_only]
    feature_columns = ELEMENTS + environment
    registry = {
        "schema_version": "epit_environment_v1",
        "feature_columns": feature_columns,
        "composition_columns": ELEMENTS,
        "environment_history_columns": environment,
        "audit_only_derived_columns": audit_only,
        "columns": {name: column_definition(name) for name in ELEMENTS + derived},
        "missing_policy": "Unknown stays missing, including phase absence and unused later heat-treatment stages; no imputation or scaling was fitted.",
        "excluded_source_columns": [name for name in frame if name not in ELEMENTS],
        "target_column": TARGET_COLUMN,
        "heat_stage_semantics": "Stages are ordered explicitly reported thermal treatments. Missing stage fields do not establish absence. Alternative conditions are not separate stages.",
    }
    coverage = pd.DataFrame([
        dict(column=name, **column_definition(name), populated_rows=int(processed[name].notna().sum()),
             distinct_values=int(processed[name].nunique()), model_feature=name in feature_columns)
        for name in derived
    ])
    source_counts = {}
    for field in rules:
        present = frame.loc[~frame[field].isin(MISSING), field]
        source_counts[field] = dict(populated_rows=len(present), distinct_descriptions=int(present.nunique()))
    counts = {
        "retained_rows": len(frame), "development_rows": int(assignments.outer_split.eq("development").sum()),
        "final_test_rows": int(assignments.outer_split.eq("final_test").sum()),
        "derived_columns": len(derived), "environment_history_features": len(environment),
        "audit_only_derived_columns": len(audit_only), "total_features_with_composition": len(feature_columns),
        "source_fields": source_counts,
        "feature_coverage": dict(zip(coverage.column, coverage.populated_rows)),
        "source_status_rows": {field: audit.loc[audit.source_column.eq(field), "status"].value_counts().to_dict() for field in audit.source_column.unique()},
        "conflicting_rows": int(audit.loc[audit.status.eq("conflict"), "row_id"].nunique()),
        "rows_with_heat_information": int(processed[HEAT_COLUMNS].notna().any(axis=1).sum()),
        "rows_with_second_heat_stage": int(processed[[c for c in HEAT_COLUMNS if c.startswith("heat_treatment_2_")]].notna().any(axis=1).sum()),
        "rows_with_phase_information": int(processed[PHASE_COLUMNS].notna().any(axis=1).sum()),
    }
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Input changed while processing: {path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    processed.to_csv(output_dir / "epit_processed.csv", index=False)
    processed[["row_id", *feature_columns]].to_csv(output_dir / "epit_features.csv", index=False)
    audit.to_csv(output_dir / "environment_processing_log.csv", index=False)
    audit.loc[audit.status.isin(["partial", "unresolved", "conflict"])].to_csv(output_dir / "environment_review_notes.csv", index=False)
    coverage.to_csv(output_dir / "environment_column_summary.csv", index=False)
    for name in composition_manifest["output_sha256"]:
        if name != "epit_processed.csv":
            shutil.copyfile(input_dir / name, output_dir / name)
    shutil.copyfile(input_dir / "processing_manifest.json", output_dir / "composition_processing_manifest.json")
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    manifest = {
        "schema_version": "epit_environment_v1", "counts": counts,
        "input_sha256": hashes,
        "implementation_sha256": {str(Path(__file__).resolve().relative_to(REPO_ROOT)): sha256_file(Path(__file__))},
        "output_sha256": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir())},
        "policy": {
            "original_columns": "Preserved verbatim, including composition, targets, and raw environment/history descriptions.",
            "concentrations": "Each chemical and stated unit had a separate column. Normality, molarity, and weight percent were not conflated. Unknown-basis amounts were audit-only.",
            "heat_treatment": "Temperature in degrees C, time in hours; distinct reported stages retained. Unresolved ranges and alternatives were not averaged.",
            "phase_flags": "1 present, 0 supported absence, missing unknown; general discussion and unspecified secondary phases were not phase assignments.",
            "targets_used_for_extraction": False,
            "row_exclusions": 0,
        },
        "scope": "Standalone EPIT export. Shared datasets, model loaders, synthetic priors, training and evaluation paths were not changed.",
    }
    (output_dir / "processing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-dir", type=Path, default=COMPOSITION_DIR)
    args = parser.parse_args()
    manifest = prepare(args.output_dir, args.input_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
