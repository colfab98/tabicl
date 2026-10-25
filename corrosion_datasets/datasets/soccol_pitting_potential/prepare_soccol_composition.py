"""Review Soccol composition while preserving the frozen outer query partition.

Exact raw compositions, explicit alloy descriptions and reviewed same-paper
recoveries supply values. Targets never select a donor or a processing rule.
Unresolved development rows are withheld; all frozen query rows are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from corrosion_datasets.datasets.electrochemical_metrics_alloys.prepare_epit_composition import (
    ELEMENTS, RENAME, SOURCE_FILE as EPIT_WORKBOOK, TABLE_NAME,
    load_electrochemical_metrics, parse_amount,
)
from .prepare_benchmark import ELEMENTS as NATIVE_ELEMENTS, RAW_WORKBOOK, EXPECTED_RAW_SHA256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT_DIR = HERE / "processed/soccol_environment_v2"
OUTPUT_DIR = HERE / "processed/soccol_composition_v1"
RULES_PATH = HERE / "soccol_composition_rules_v1.json"
SOURCE_REGISTRY = ROOT / "corrosion_datasets/analysis/soccol_source_conventions/source_conventions.csv"
SPLIT_MANIFEST = HERE / "processed/splits_v1/split_manifest.json"
LEGACY_AUDIT = ["Fe_is_approximate", "Fe_balance_basis", "Fe_is_largest_component",
                "composition_reported_count", "composition_sum_without_Fe_wt_pct",
                *[f"{c}_missing" for c in ["Fe", *NATIVE_ELEMENTS]]]


def numeric(value) -> float:
    if pd.isna(value) or str(value).strip() in {"", "NA", "nan", "None"}:
        return math.nan
    amount = float(value)
    if not math.isfinite(amount) or not 0 <= amount <= 100:
        raise ValueError(f"Invalid composition amount: {value!r}")
    return amount


def normalize_doi(value) -> str:
    return str(value).replace("\ufeff", "").strip().lower()


def load_evidence(rules: dict) -> tuple[list[dict], dict]:
    """Read raw EPIT cells; do not use processed zero-fills or target eligibility."""
    table = next(t for t in load_electrochemical_metrics() if t.table == TABLE_NAME)
    refs = {r["epit_reference"] for r in rules["same_paper_recoveries"]}
    evidence = []
    for row in table.rows:
        try:
            ref = int(float(row["Reference"]))
        except (ValueError, TypeError):
            continue
        if ref not in refs:
            continue
        entry = {"reference": ref, "source_no": str(row["No."])}
        entry.update({element: parse_amount(row[column])[0] for column, element in RENAME.items()})
        evidence.append(entry)
    bibliography = pd.read_excel(EPIT_WORKBOOK, sheet_name="References").set_index("No")
    registry = pd.read_csv(SOURCE_REGISTRY).set_index("source")
    dois = {}
    for rule in rules["same_paper_recoveries"]:
        doi = normalize_doi(registry.loc[rule["source"], "doi"])
        if doi != normalize_doi(bibliography.loc[rule["epit_reference"], "DOI"]):
            raise ValueError(f"Paper identity mismatch for {rule['source']}")
        dois[rule["source"]] = doi
    return evidence, dois


def transform(raw: pd.DataFrame, legacy: pd.DataFrame, rules: dict,
              evidence: list[dict], dois: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not raw.row_id.equals(legacy.row_id):
        raise ValueError("Raw and reference row identities differ.")
    designations = {rule["row_id"]: rule for rule in rules["designation_recoveries"]}
    transfers = {rule["source"]: rule for rule in rules["same_paper_recoveries"]}
    for src, rule in transfers.items():
        if raw.source.eq(src).sum() != rule["expected_rows"]:
            raise ValueError(f"Source row count changed for {src}")
    compositions, logs, audits = [], [], []
    for index, row in raw.iterrows():
        old = legacy.loc[index]
        source = row.source
        designation = "" if pd.isna(row.alloy_designation) else str(row.alloy_designation).strip()
        values = dict.fromkeys(ELEMENTS, math.nan)
        actions = dict.fromkeys(ELEMENTS, "unreported")
        evidence_text = dict.fromkeys(ELEMENTS, "")
        reasons = []
        for element in NATIVE_ELEMENTS:
            amount = numeric(row[element])
            if source == "2000Russell" and element == "Ti":
                actions[element] = "discarded_PRE_misfiled_as_Ti"
            elif math.isfinite(amount):
                values[element] = amount
                actions[element] = "reported_numeric"
                evidence_text[element] = f"Soccol raw workbook row {old.raw_workbook_row}"
        if row.row_id in designations:
            rule = designations[row.row_id]
            if source != rule["source"] or designation != rule["designation"]:
                raise ValueError(f"Changed designation for {row.row_id}")
            for element, amount in rule["values"].items():
                if math.isfinite(values[element]):
                    raise ValueError(f"Would overwrite a reported amount: {row.row_id}/{element}")
                values[element] = numeric(amount)
                actions[element] = "recovered_nominal_from_designation"
                evidence_text[element] = designation + ("; " + rule["note"] if rule.get("note") else "")
        combined = source == rules["combined_amount_rule"]["source"] and rules["combined_amount_rule"]["designation_contains"] in designation
        if source in transfers:
            rule = transfers[source]
            # Every numeric composition originally reported by Soccol must
            # agree; outcomes, temperature and label order never break a tie.
            known = {e: numeric(row[e]) for e in NATIVE_ELEMENTS if math.isfinite(numeric(row[e]))}
            candidates = [d for d in evidence if d["reference"] == rule["epit_reference"] and
                          all(math.isfinite(d[e]) and math.isclose(d[e], v, abs_tol=1e-8, rel_tol=0)
                              for e, v in known.items())]
            if not candidates:
                raise ValueError(f"No compatible same-paper composition for {row.row_id}")
            for element in rule["elements"]:
                if math.isfinite(values[element]) or (combined and element in {"Nb", "Ta"}):
                    continue
                amounts = sorted({d[element] for d in candidates if math.isfinite(d[element])})
                if len(amounts) > 1:
                    reasons.append(f"Ambiguous same-paper amounts for {element}")
                    continue
                if amounts:
                    values[element] = amounts[0]
                    actions[element] = "recovered_from_same_paper_in_raw_EPIT"
                    donor_ids = sorted({d["source_no"] for d in candidates if math.isfinite(d[element])})
                    evidence_text[element] = json.dumps({"doi": dois[source], "epit_reference": rule["epit_reference"],
                                                        "epit_source_nos": donor_ids, "matching_fields": known})
        unresolved = set()
        if source in rules["unresolved_study_elements"]:
            rule = rules["unresolved_study_elements"][source]
            unresolved.update(rule["elements"])
            reasons.append(rule["reason"])
        if combined:
            unresolved.update(rules["combined_amount_rule"]["elements"])
            reasons.append(rules["combined_amount_rule"]["reason"])
        missing_named = rules["missing_named_element"]
        if source == missing_named["source"] and designation == missing_named["designation"]:
            unresolved.update(missing_named["elements"])
            reasons.append(missing_named["reason"])
        if source == "1994Malik" and designation in {"17-4PH", "20Cb3"}:
            unresolved.add("Nb")
        for element in unresolved:
            if math.isfinite(values[element]):
                raise ValueError(f"Known omission unexpectedly populated: {row.row_id}/{element}")
            actions[element] = "known_omission_unresolved"
        explicit_non_fe = str(old.Fe_balance_basis).startswith("explicit_")
        can_balance = str(old.Fe_is_approximate) in {"1", "1.0"}
        if not can_balance and not explicit_non_fe:
            reasons.append("Bulk composition or balance element remained unsupported; no Fe or grade composition was invented.")
        if not reasons:
            for element in ELEMENTS:
                if element != "Fe" and not math.isfinite(values[element]):
                    values[element] = 0.0
                    actions[element] = ("assumed_zero_after_PRE_removal" if source == "2000Russell" and element == "Ti"
                                        else "assumed_zero_after_completeness_review")
            subtotal = math.fsum(values[e] for e in ELEMENTS if e != "Fe")
            values["Fe"] = 0.0 if explicit_non_fe else round(100 - subtotal, 12)
            actions["Fe"] = "explicit_non_Fe_alloy" if explicit_non_fe else "recalculated_supported_Fe_balance"
            if values["Fe"] < 0 or not 99 <= math.fsum(values.values()) <= 101:
                raise ValueError(f"Invalid completed composition: {row.row_id}")
            status = "accepted"
        else:
            # Known omitted mass must not be silently included in iron.
            actions["Fe"] = "balance_withheld_pending_composition_review"
            status = "unresolved"
        compositions.append(values)
        present_sum = math.fsum(v for v in values.values() if math.isfinite(v))
        audits.append(dict(row_id=row.row_id, source=source, alloy_designation=designation,
                           composition_status=status, composition_sum_available_wt_percent=present_sum,
                           missing_composition_cells=sum(not math.isfinite(v) for v in values.values()),
                           recovered_element_cells=sum(a.startswith("recovered_") for a in actions.values()),
                           known_unresolved_elements=";".join(sorted(unresolved)),
                           reason=" | ".join(reasons)))
        for element in ELEMENTS:
            original = row[element] if element in NATIVE_ELEMENTS else ""
            previous = numeric(old[element]) if element in old else math.nan
            changed = not (math.isnan(previous) and math.isnan(values[element]) or
                           math.isclose(previous, values[element], rel_tol=0, abs_tol=1e-10))
            logs.append(dict(row_id=row.row_id, element=element, source_value=original,
                             previous_value=previous, processed_value=values[element],
                             action=actions[element], evidence=evidence_text[element], changed=changed))
    return pd.DataFrame(compositions, columns=ELEMENTS), pd.DataFrame(logs), pd.DataFrame(audits)


def prepare(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")
    manifest = json.loads((INPUT_DIR / "processing_manifest.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        if sha256_file(INPUT_DIR / name) != digest:
            raise RuntimeError(f"Environment export checksum mismatch: {name}")
    if sha256_file(RAW_WORKBOOK) != EXPECTED_RAW_SHA256:
        raise RuntimeError("Original Soccol workbook changed.")
    frozen = load_frozen_split(SPLIT_MANIFEST, expected_manifest_schema="soccol_composition_split_manifest_v1")
    split_path = SPLIT_MANIFEST.parent / frozen.lock["assignments"]["file"]
    assignments = pd.read_csv(split_path)
    legacy = pd.read_csv(INPUT_DIR / "soccol_processed.csv", dtype=str, keep_default_na=False)
    raw = pd.read_excel(RAW_WORKBOOK, sheet_name="pitting_potentials").iloc[assignments.raw_workbook_row.to_numpy()-2].reset_index(drop=True)
    raw.insert(0, "row_id", assignments.raw_workbook_row.map(lambda n: f"soccol_r{int(n):06d}"))
    if len(raw) != 4027 or not raw.row_id.equals(legacy.row_id):
        raise ValueError("Unexpected frozen row identities.")
    if not legacy.raw_workbook_row.astype(int).equals(assignments.raw_workbook_row):
        raise ValueError("Frozen assignment row order changed.")
    pd.testing.assert_series_equal(pd.to_numeric(legacy.E_pit), pd.to_numeric(raw.E_pit), check_dtype=False)
    rules = json.loads(RULES_PATH.read_text())
    if rules["schema_version"] != "soccol_composition_rules_v1":
        raise ValueError("Unexpected rule version.")
    paths = [p for p in INPUT_DIR.iterdir() if p.is_file()] + [RAW_WORKBOOK, EPIT_WORKBOOK, RULES_PATH,
             SOURCE_REGISTRY, SPLIT_MANIFEST, frozen.lock_path, split_path]
    hashes = {str(p): sha256_file(p) for p in paths}
    evidence, dois = load_evidence(rules)
    composition, log, audit = transform(raw, legacy, rules, evidence, dois)
    audit["outer_split"] = assignments.outer_split
    audit["development_validation_fold"] = assignments.development_validation_fold
    audit["use_as_development_context"] = audit.composition_status.eq("accepted") & audit.outer_split.eq("development")
    audit["frozen_final_query"] = audit.outer_split.eq("final_test")
    # Retain old accounting under explicit legacy names so the corrected
    # table never claims that a recovered element is still zero or missing.
    processed = legacy.rename(columns={c: f"legacy_{c}" for c in LEGACY_AUDIT})
    for element in ELEMENTS:
        if element in processed:
            processed[f"legacy_{element}"] = processed[element]
        processed[element] = composition[element]
    processed = pd.concat([processed, audit.drop(columns=["row_id", "source", "alloy_designation"])], axis=1)
    registry = json.loads((INPUT_DIR / "feature_registry.json").read_text())
    environment = registry["feature_columns"]
    registry["schema_version"] = "soccol_composition_v1"
    registry["feature_columns"] = ELEMENTS + environment
    registry["composition_columns"] = ELEMENTS
    registry["environment_history_columns"] = environment
    registry["columns"].update({e: dict(group="composition", dtype="numeric", unit="wt.%") for e in ELEMENTS})
    registry.pop("legacy_composition_columns_in_reference_only", None)
    registry["scope"] = "Reviewed composition and inherited environment features. Only accepted development rows are eligible context. Every frozen query row is retained, including unresolved cases; this is not a ready final-evaluation input while any query composition is unresolved."
    registry["composition_policy"] = "Reported amounts retained; explicit nominal and same-paper raw-data recoveries logged; supported Fe balance computed after recovery; unknown elements zero-filled only for accepted compositions. Known omissions remained NaN in review rows."
    registry["excluded_reference_columns"] = [c for c in processed if c not in registry["feature_columns"] and c != "row_id"]
    features = processed[["row_id", *registry["feature_columns"]]]
    dev = audit.use_as_development_context
    query = audit.frozen_final_query
    assert composition.loc[dev].notna().all().all()
    assert int(query.sum()) == 805
    counts = dict(rows=len(processed), accepted_compositions=int(audit.composition_status.eq("accepted").sum()),
                  unresolved_compositions=int(audit.composition_status.eq("unresolved").sum()),
                  original_development_rows=int(assignments.outer_split.eq("development").sum()),
                  accepted_development_rows=int(dev.sum()), withheld_development_rows=int((~dev & ~query).sum()),
                  frozen_final_query_rows=int(query.sum()), unresolved_final_query_rows=int((query & audit.composition_status.ne("accepted")).sum()),
                  recovered_element_cells=int(log.action.str.startswith("recovered_").sum()),
                  rows_with_recovered_elements=int(log.loc[log.action.str.startswith("recovered_"), "row_id"].nunique()),
                  composition_columns=len(ELEMENTS), total_features=len(registry["feature_columns"]))
    counts["final_evaluation_ready"] = counts["unresolved_final_query_rows"] == 0
    registry["final_evaluation_ready"] = counts["final_evaluation_ready"]
    output_dir.mkdir(parents=True)
    for name in ["source_rows.csv", "soccol_environment_features.csv", "environment_processing_log.csv",
                 "environment_review_notes.csv", "environment_column_summary.csv", "review_changes.csv"]:
        shutil.copy2(INPUT_DIR / name, output_dir / name)
    shutil.copy2(split_path, output_dir / "split_assignments.csv")
    for name, frame in {
        "soccol_processed.csv": processed, "soccol_features.csv": features,
        "composition_processing_log.csv": log, "composition_changes.csv": log.loc[log.changed],
        "composition_row_audit.csv": audit, "composition_review_required.csv": audit.loc[audit.composition_status.ne("accepted")],
        "soccol_development_accepted.csv": processed.loc[dev], "soccol_development_features.csv": features.loc[dev],
        "soccol_frozen_final_query.csv": processed.loc[query], "soccol_frozen_final_query_features.csv": features.loc[query],
    }.items():
        frame.to_csv(output_dir / name, index=False)
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    for p in paths:
        if sha256_file(p) != hashes[str(p)]:
            raise RuntimeError(f"Input changed during export: {p}")
    result = {"schema_version": "soccol_composition_v1", "counts": counts,
              "input_sha256": hashes, "implementation_sha256": {str(Path(__file__).resolve()): sha256_file(Path(__file__))},
              "output_sha256": {p.name: sha256_file(p) for p in sorted(output_dir.iterdir())},
              "scope": registry["scope"], "policy": registry["composition_policy"],
              "split_policy": "Frozen assignments were preserved verbatim. No query was dropped, reassigned or replaced by development rows.",
              "existing_pipeline_paths_changed": False}
    (output_dir / "processing_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir)["counts"], indent=2))


if __name__ == "__main__":
    main()
