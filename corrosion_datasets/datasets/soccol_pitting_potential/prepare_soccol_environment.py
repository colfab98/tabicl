"""Export reviewed Soccol environment features without changing existing pipelines.

The frozen regression rows and existing composition are preserved. Environment
values are read from the original workbook; text interpretations are explicit
rules, independent of outcomes. CP_ion_other is never used for extraction.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from .prepare_benchmark import EXPECTED_RAW_SHA256, RAW_WORKBOOK

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY_TABLE = HERE / "processed/soccol_regression_event1.csv"
SPLIT_MANIFEST = HERE / "processed/splits_v1/split_manifest.json"
RULES_PATH = HERE / "soccol_environment_rules_v1.json"
DEFAULT_OUTPUT_DIR = HERE / "processed/soccol_environment_v1"
ION_COLUMNS = ["CP_Cl", "CP_Br", "CP_OH", "CP_SO4", "CP_CO3", "CP_NO3", "CP_PO4", "CP_MoO4", "CP_CrO4"]
DIRECT_UNITS = {
    "Prep_grinding_grit": "reported grit grade",
    "Prep_Ra_micron": "um",
    "Prep_pH": "dimensionless",
    "Prep_redox": "V; reference not established for this column",
    "Prep_time": "h",
    "CP_time": "h",
    "CP_temp": "degC",
    "CP_pH": "dimensionless",
    "Test_area_cm2": "cm2",
    "scan_rate": "mV/s",
    **dict.fromkeys(ION_COLUMNS, "mol/L"),
}
TEXT_FIELDS = ["CP_aeration", "CP_agitation", "CP_anions_info", "Prep_medium"]
SOURCE_ENVIRONMENT = [*DIRECT_UNITS, *TEXT_FIELDS, "CP_ion_other"]
HEAT_COLUMNS = [
    f"{prefix}{suffix}"
    for prefix in ("heat_treatment_", "heat_treatment_2_")
    for suffix in ("type", "temperature_C", "duration_h", "cooling")
]
FEATURE_COLUMNS = [
    *DIRECT_UNITS, "CP_aeration", "CP_agitation", "test_solution_type",
    "H2SO4_mol_L", "surface_preparation_method", "preparation_solution_type",
    *HEAT_COLUMNS,
]
LEGACY_COMPOSITION = ["Fe", "C", "N", "Si", "P", "S", "Ti", "V", "Cr", "Mn", "Ni", "Nb", "Mo"]
MISSING = {"", "NA", "N/A", "NaN", "nan", "None"}


def source_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def numeric_source(value: Any, name: str) -> float | None:
    text = source_text(value)
    if text in MISSING:
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"Unreviewed numeric notation for {name}: {text!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {name}: {text!r}")
    if name not in {"Prep_pH", "CP_pH", "Prep_redox", "CP_temp"} and result < 0:
        raise ValueError(f"Negative amount, duration or rate for {name}: {text!r}")
    return result


def load_rules(path: Path = RULES_PATH) -> dict:
    document = json.loads(path.read_text())
    if document["schema_version"] != "soccol_environment_rules_v1":
        raise ValueError("Unsupported Soccol environment rules.")
    result = {}
    for field, entries in document["fields"].items():
        index = {entry["source_text"]: entry for entry in entries}
        if len(index) != len(entries):
            raise ValueError(f"Duplicate rules for {field}")
        for rule in entries:
            if set(rule["features"]) - set(FEATURE_COLUMNS):
                raise ValueError(f"Unregistered features in {field}: {rule['features']}")
        result[field] = index
    if set(result) != set(TEXT_FIELDS):
        raise ValueError("Unexpected text fields in environment rules.")
    return result


def reviewed_rule(rules: dict, field: str, value: Any, source: str = "") -> dict:
    text = source_text(value)
    if text in MISSING:
        return dict(source_text=text, features={}, status="missing", notes=["Missing information; absence was not inferred."])
    try:
        rule = rules[field][text]
    except KeyError as exc:
        raise ValueError(f"Unreviewed Soccol description in {field}: {text!r}") from exc
    if rule.get("source") and rule["source"] != source:
        raise ValueError(f"Source-specific rule for {text!r} cannot be applied to {source!r}")
    return rule


def column_definition(name: str) -> dict:
    if name in DIRECT_UNITS:
        definition = dict(dtype="numeric", unit=DIRECT_UNITS[name],
                          group="preparation" if name.startswith("Prep_") else "test_environment")
        if name == "Prep_grinding_grit":
            definition["qualification"] = "Reported workbook grade; standards were not assumed equivalent. The explicit 1200/600-US note already had numeric grade 600."
        elif name == "Prep_redox":
            definition["qualification"] = "Original Soccol voltage retained. A common reference for E_pit/E_corr did not establish the reference for Prep_redox. Verify before cross-dataset harmonization."
        elif name == "Prep_time":
            definition["meaning"] = "Reported preparation-treatment duration, separate from heat treatment and test-bath equilibration."
        elif name == "CP_time":
            definition["meaning"] = "Equilibration in the test medium before the scan; not time until pitting."
        elif name in ION_COLUMNS:
            definition["qualification"] = "Workbook concentration retained unless explicit text contradicted it. No universal valence conversion or free-ion calculation was applied."
        return definition
    if name == "CP_agitation":
        return dict(group="test_environment", dtype="numeric", unit="binary", meaning="1 yes, 0 no, missing unknown")
    if name == "H2SO4_mol_L":
        return dict(group="test_solution", dtype="numeric", unit="mol/L", meaning="Reported acid component molarity; not calculated free sulfate.")
    if name.startswith("heat_treatment_"):
        if name.endswith("temperature_C"):
            return dict(group="heat_treatment", dtype="numeric", unit="degC")
        if name.endswith("duration_h"):
            return dict(group="heat_treatment", dtype="numeric", unit="h")
        return dict(group="heat_treatment", dtype="categorical", unit=None)
    if name in {"surface_preparation_method", "preparation_solution_type"}:
        return dict(group="preparation", dtype="categorical", unit=None)
    if name in {"test_solution_type", "CP_aeration"}:
        return dict(group="test_environment", dtype="categorical", unit=None)
    raise ValueError(f"Unknown environment feature: {name}")


def same_value(first: Any, second: Any) -> bool:
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(first, second, rel_tol=1e-8, abs_tol=1e-12)
    return first == second


def transform(frame: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only named procedure/environment fields, never targets or other-ion data."""
    records, audit = [], []
    for row in frame.to_dict("records"):
        values = dict.fromkeys(FEATURE_COLUMNS)
        conflicts = set()
        for name in DIRECT_UNITS:
            value = numeric_source(row[name], name)
            values[name] = value
            audit.append(dict(row_id=row["row_id"], source_column=name,
                              source_text=source_text(row[name]), status="missing" if value is None else "copied_numeric",
                              extracted_features=json.dumps({name: value}),
                              notes="Unknown stayed missing." if value is None else "Retained the original numeric amount and documented unit."))
        for name in TEXT_FIELDS:
            rule = reviewed_rule(rules, name, row[name], source_text(row.get("source")))
            accepted = dict(rule["features"])
            notes, status = list(rule["notes"]), rule["status"]
            for target, value in rule["features"].items():
                current = values[target]
                if target in conflicts:
                    accepted[target] = None
                    notes.append(f"{target} remained missing after an earlier conflict.")
                    status = "conflict"
                elif current is None or same_value(current, value):
                    values[target] = value
                else:
                    values[target] = accepted[target] = None
                    conflicts.add(target)
                    notes.append(f"Conflicting values for {target}: numeric/earlier report {current!r}, text {value!r}. Left missing; both reports were retained in the audit.")
                    status = "conflict"
            audit.append(dict(row_id=row["row_id"], source_column=name,
                              source_text=source_text(row[name]), status=status,
                              extracted_features=json.dumps(accepted), notes=" ".join(notes)))
        records.append(values)
    features = pd.DataFrame(records, columns=FEATURE_COLUMNS, index=frame.index)
    for name in FEATURE_COLUMNS:
        if column_definition(name)["dtype"] == "numeric":
            features[name] = pd.to_numeric(features[name], errors="raise")
    features.insert(0, "row_id", frame.row_id)
    return features, pd.DataFrame(audit)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    if sha256_file(RAW_WORKBOOK) != EXPECTED_RAW_SHA256:
        raise RuntimeError("Raw Soccol workbook changed; review the source version.")
    frozen = load_frozen_split(SPLIT_MANIFEST, expected_manifest_schema="soccol_composition_split_manifest_v1")
    if sha256_file(LEGACY_TABLE) != frozen.lock["source_sha256"]:
        raise RuntimeError("Soccol regression table differs from the frozen split source.")
    split_path = SPLIT_MANIFEST.parent / frozen.lock["assignments"]["file"]
    assignments = pd.read_csv(split_path)
    legacy = pd.read_csv(LEGACY_TABLE, dtype=str, keep_default_na=False)
    workbook = pd.read_excel(RAW_WORKBOOK, sheet_name="pitting_potentials", keep_default_na=False)
    indices = assignments.raw_workbook_row.to_numpy(dtype=int) - 2
    raw = workbook.iloc[indices].reset_index(drop=True)
    if len(raw) != 4027 or assignments.task_row_index.tolist() != list(range(4027)):
        raise ValueError("Unexpected frozen Soccol rows or order.")
    if assignments.raw_workbook_row.duplicated().any():
        raise ValueError("Duplicated raw workbook row in frozen assignments.")
    for name in ["source", "label"]:
        if raw[name].map(source_text).tolist() != legacy[name].tolist():
            raise ValueError(f"Raw {name} values do not match the frozen regression rows.")
        if legacy[name].tolist() != assignments[name].astype(str).tolist():
            raise ValueError(f"Frozen assignment {name} values do not match the regression rows.")
    if legacy.raw_workbook_row.astype(int).tolist() != assignments.raw_workbook_row.tolist():
        raise ValueError("Regression row identity mismatch.")
    if not pd.to_numeric(raw.event).eq(1).all():
        raise ValueError("Non-regression event encountered.")
    pd.testing.assert_series_equal(pd.to_numeric(raw.E_pit), pd.to_numeric(legacy.E_pit), check_dtype=False)
    row_ids = assignments.raw_workbook_row.map(lambda n: f"soccol_r{int(n):06d}")
    raw.insert(0, "row_id", row_ids)
    legacy.insert(0, "row_id", row_ids)
    paths = [RAW_WORKBOOK, LEGACY_TABLE, RULES_PATH, SPLIT_MANIFEST, frozen.lock_path, split_path,
             SPLIT_MANIFEST.parent / frozen.lock["report"]["file"]]
    return raw, legacy, assignments, paths


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")
    raw, legacy, assignments, input_paths = load_inputs()
    hashes = {str(path): sha256_file(path) for path in input_paths}
    features, audit = transform(raw, load_rules())
    # Non-environment legacy fields, including composition and targets, are
    # preserved verbatim. This step does not re-certify legacy composition.
    reference = legacy.drop(columns=SOURCE_ENVIRONMENT)
    reference = pd.concat([reference, raw[SOURCE_ENVIRONMENT].add_prefix("raw_"),
                           features.drop(columns="row_id")], axis=1)
    definitions = {name: column_definition(name) for name in FEATURE_COLUMNS}
    coverage = pd.DataFrame([
        dict(column=name, **definitions[name], populated_rows=int(features[name].notna().sum()),
             distinct_values=int(features[name].nunique())) for name in FEATURE_COLUMNS
    ])
    registry = {
        "schema_version": "soccol_environment_v1", "feature_columns": FEATURE_COLUMNS,
        "columns": definitions, "composition_columns": [],
        "legacy_composition_columns_in_reference_only": LEGACY_COMPOSITION,
        "missing_policy": "Unknown stays NaN. Reported zeros remain zero. Contradictory reports become NaN with an audit; no fitted imputation or scaling.",
        "excluded_columns": ["CP_ion_other", "Prep_medium", "CP_anions_info", "E_corr", "E_pit", "event", "source", "label", "ID", "alloy_designation"],
        "excluded_reference_columns": [name for name in reference if name not in FEATURE_COLUMNS and name != "row_id"],
        "original_environment_columns": {name: f"raw_{name}" for name in SOURCE_ENVIRONMENT},
        "scope": "Environment-only feature table. Existing composition was preserved in the reference table and was not reprocessed or certified by this export.",
        "test_solution_type_semantics": "Only explicitly named whole media (seawater) were assigned. Additive notes were not treated as complete solution recipes.",
        "other_ions": "CP_ion_other was never read during extraction. Borate/citrate amounts were not added in this export.",
    }
    counts = {
        "retained_rows": len(reference), "development_rows": int(assignments.outer_split.eq("development").sum()),
        "final_test_rows": int(assignments.outer_split.eq("final_test").sum()),
        "environment_features": len(FEATURE_COLUMNS), "feature_coverage": dict(zip(coverage.column, coverage.populated_rows)),
        "conflicting_rows": int(audit.loc[audit.status.eq("conflict"), "row_id"].nunique()),
        "rows_with_heat_information": int(features[HEAT_COLUMNS].notna().any(axis=1).sum()),
        "source_status_rows": {field: audit.loc[audit.source_column.eq(field), "status"].value_counts().to_dict() for field in audit.source_column.unique()},
    }
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Input changed while processing: {path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    reference.to_csv(output_dir / "soccol_processed.csv", index=False)
    features.to_csv(output_dir / "soccol_environment_features.csv", index=False)
    raw.to_csv(output_dir / "source_rows.csv", index=False)
    audit.to_csv(output_dir / "environment_processing_log.csv", index=False)
    audit.loc[audit.status.isin(["partial", "unresolved", "conflict"])].to_csv(output_dir / "environment_review_notes.csv", index=False)
    coverage.to_csv(output_dir / "environment_column_summary.csv", index=False)
    shutil.copyfile(SPLIT_MANIFEST.parent / "split_assignments.csv", output_dir / "split_assignments.csv")
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    manifest = {
        "schema_version": "soccol_environment_v1", "counts": counts, "input_sha256": hashes,
        "implementation_sha256": {str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__))},
        "output_sha256": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir())},
        "policy": {"targets_used_for_extraction": False, "row_exclusions": 0,
                   "CP_ion_other_used": False, "imputation_fitted": False,
                   "composition_reprocessed": False,
                   "frozen_assignments": "Copied byte-for-byte; same final-test rows and development folds."},
        "scope": "Standalone Soccol environment export. EPIT, shared tables, model loaders, synthetic priors and evaluation paths were not modified.",
    }
    (output_dir / "processing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
