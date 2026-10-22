#!/usr/bin/env python3
"""Freeze Zhang 2023 external queries in the existing 50-feature corrosion schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_datasets.datasets.combined_epit_soccol_mpea.prepare_shared_tables import (
    ELEMENT_FEATURES, FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN, file_sha256,
)
from corrosion_datasets.datasets.combined_epit_soccol_mpea.schema_alignment import align_feature_frame

CONTEXT_DIR = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
QUERY_NAME = "zhang2023_queries.csv"
SCORING_NAME = "zhang2023_scoring_targets.csv"
TASK_ID = "epit_soccol_context__zhang2023_pitting_potential_mv_sce"
DOI = "10.1038/s41467-023-43752-8"


def electrolyte_features(recipe: dict) -> dict[str, float]:
    result = {}
    for feature, ion in (("CP_Cl", "chloride"), ("CP_Br", "bromide"), ("CP_SO4", "sulfate")):
        result[feature] = sum(
            salt["g_per_L"] / salt["molar_mass"] * salt.get(f"{ion}_stoichiometry", 0)
            for salt in recipe["salts"].values()
        )
    return result


def build_tables(raw_dir: Path = HERE / "raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compositions = pd.read_csv(raw_dir / "experimental_compositions_wt_percent.csv")
    if compositions.alloy_id.duplicated().any() or len(compositions) != 7:
        raise ValueError("Expected seven unique experimental compositions.")
    if not compositions.composition_unit.eq("wt%").all() or not compositions.Fe_reported.eq("Balance").all():
        raise ValueError("Only explicitly reported wt% compositions and Fe balance are supported.")
    compositions = compositions.set_index("alloy_id")
    recipe = json.loads((raw_dir / "seawater_recipe.json").read_text())
    environment = {
        "CP_temp": 72., "CP_pH": 8.2, "Test_area_cm2": 1., "scan_rate": .333,
        "Prep_grinding_grit": 2000., "Prep_medium": "reduced -1.0V", "Prep_pH": 8.2,
        "CP_anions_info": "sea water", "test_method": "Potentiodynamic Polarization",
        "reference_type": "SCE", **electrolyte_features(recipe),
    }
    records, scores, audit = [], [], []
    # 6d is industrial material without a separately verified composition table.
    for sheet in ("Supplementary Fig.6a", "Supplementary Fig.6b", "Supplementary Fig.6c"):
        table = pd.read_excel(raw_dir / "Supplementary Fig. 6.xlsx", sheet_name=sheet)
        table.columns = [str(c).strip() for c in table.columns]
        target_col = next(c for c in table if c.startswith("Pitting potential"))
        for source in table.dropna(subset=["Steels"]).to_dict("records"):
            label = str(source["Steels"]).strip()
            alloy_id, order = label.rsplit("-", 1)
            composition = compositions.loc[alloy_id]
            row = {name: np.nan for name in FEATURE_COLUMNS}
            for element in ELEMENT_FEATURES:
                if element in composition and pd.notna(composition[element]):
                    row[element] = float(composition[element])
            reported_sum = sum(row[e] for e in ELEMENT_FEATURES if e != "Fe" and pd.notna(row[e]))
            row["Fe"] = 100. - reported_sum
            if not 0 < row["Fe"] <= 100:
                raise ValueError(f"Invalid Fe balance for {alloy_id}.")
            row.update(environment)
            row_id = f"zhang2023_{label}"
            row.update(dict(row_id=row_id, source_dataset="Zhang2023", source_row_id=label,
                            source_reference=DOI, reference_type_evidence="primary_paper_and_source_workbook_SCE"))
            records.append(row)
            scores.append({"row_id": row_id, TARGET_COLUMN: float(source[target_col]), "reference_type": "SCE"})
            audit.append({"row_id": row_id, "alloy_id": alloy_id, "source_sheet": sheet,
                          "source_order_sorted_by_target": int(order),
                          "composition_table": int(composition.composition_table),
                          "Fe_balance_from_reported_wt_percent": row["Fe"],
                          "Fe_caveat": "balance estimate; unreported trace elements not individually set to zero",
                          "cathodic_pretreatment_V_SCE": -1., "cathodic_pretreatment_seconds": 300,
                          "OCP_wait_seconds": 1800, "pitting_threshold_microA_per_cm2": 100})
    queries = pd.DataFrame(records, columns=[*METADATA_COLUMNS, *FEATURE_COLUMNS])
    targets = pd.DataFrame(scores)
    registry = json.loads((CONTEXT_DIR / "shared_feature_registry.json").read_text())
    aligned, _ = align_feature_frame(queries.loc[:, FEATURE_COLUMNS], registry=registry)
    queries.loc[:, FEATURE_COLUMNS] = aligned
    if len(queries) != 56 or queries.row_id.duplicated().any():
        raise ValueError("Expected 56 unique row IDs.")
    if queries.row_id.tolist() != targets.row_id.tolist() or not np.isfinite(targets[TARGET_COLUMN]).all():
        raise ValueError("Targets must be finite and aligned with query row IDs.")
    if queries.loc[:, FEATURE_COLUMNS].drop_duplicates().shape[0] != 7:
        raise ValueError("Expected seven distinct feature vectors.")
    return queries, targets, pd.DataFrame(audit)


def prepare(output_dir: Path = HERE / "processed", *, overwrite: bool = False) -> dict:
    names = [QUERY_NAME, SCORING_NAME, "row_audit.csv", "epit_soccol_context.csv",
             "shared_feature_registry.json", "preparation_manifest.json"]
    if not overwrite and any((output_dir / n).exists() for n in names):
        raise FileExistsError("Frozen artifacts already exist; use --overwrite only before evaluation.")
    original = json.loads((CONTEXT_DIR / "preparation_manifest.json").read_text())
    for name in ("epit_soccol_context.csv", "shared_feature_registry.json"):
        if file_sha256(CONTEXT_DIR / name) != original["outputs"][name]["sha256"]:
            raise ValueError(f"Original frozen artifact changed: {name}")
    queries, scoring, audit = build_tables()
    context = pd.read_csv(CONTEXT_DIR / "epit_soccol_context.csv", low_memory=False)
    if len(context) != 4787 or set(context.row_id) & set(queries.row_id):
        raise ValueError("Context must be the original 4,787 rows, disjoint from external queries.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("epit_soccol_context.csv", "shared_feature_registry.json"):
        shutil.copyfile(CONTEXT_DIR / name, output_dir / name)
    for name, frame in ((QUERY_NAME, queries), (SCORING_NAME, scoring), ("row_audit.csv", audit)):
        frame.to_csv(output_dir / name, index=False, na_rep="NaN", float_format="%.12g")
    manifest = {
        "preparation_version": "zhang2023_external_v1", "task_id": TASK_ID,
        "dataset": "zhang2023_duplex_pitting", "doi": DOI,
        "rows": {"context": 4787, "queries": 56, "composition_groups": 7,
                 "replicates_per_composition": 8, "excluded_industrial_targets": 16},
        "query_partition": "all 56 rows frozen as external queries; no query labels used as context",
        "features": list(FEATURE_COLUMNS), "target_column": TARGET_COLUMN,
        "composition_policy": "Reported wt%; Fe = 100 minus reported non-Fe elements because Fe is explicitly balance. Unreported elements/dashes stay NaN; explicit Nb zeros remain zero.",
        "electrolyte": electrolyte_features(json.loads((HERE / "raw/seawater_recipe.json").read_text())),
        "electrolyte_policy": "Nominal mol/L from cited double-concentration recipe; sum all chloride salts. Bicarbonate not mapped to carbonate. No pH-derived hydroxide.",
        "procedure_policy": "Numeric CP_time, Prep_time and Prep_redox left NaN because shared registry units are unverified; original seconds/volts retained in row_audit. Aeration/agitation unreported.",
        "target_policy": "Workbook mV vs SCE retained without conversion; separate scoring file; no re-estimation from curves.",
        "metadata_policy": "Alloy identifiers and sorted replicate rank excluded from predictors. No row aggregation; 56 replicate targets but seven distinct inputs.",
        "limitations": ["One study/environment", "Nb-induced microstructure absent from feature schema"],
        "raw_inputs": {p.name: file_sha256(p) for p in sorted((HERE / "raw").iterdir()) if p.is_file()},
        "processor_sha256": file_sha256(Path(__file__)),
        "original_context_manifest_sha256": file_sha256(CONTEXT_DIR / "preparation_manifest.json"),
        "outputs": {n: {"sha256": file_sha256(output_dir / n)} for n in names if n != "preparation_manifest.json"},
    }
    (output_dir / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "processed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, overwrite=args.overwrite)["rows"], indent=2))
