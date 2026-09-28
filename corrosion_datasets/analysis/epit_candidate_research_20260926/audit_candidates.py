"""Reproduce workbook counts and conservative DOI-overlap screening; no model fitting.

Run with pandas and openpyxl installed. Raw data are preserved unchanged.
This is a research audit, not a finalized data loader or evaluation split.
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATASETS = REPO / "corrosion_datasets/datasets"
COMPOSITION = "C N Si P S Ti V Cr Mn Ni Nb Mo".split()
# Matched on author/year/venue/page (or thesis institution), not target values.
MANUAL_SOURCE_MATCHES = {"1991Azuma": 16, "1986Jargelius": 42, "2009Wong": 44}


def doi(value):
    if pd.isna(value):
        return None
    value = re.sub(r"\s|\ufeff", "", str(value)).lower()
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", value)
    return value.rstrip(".") or None


def describe_subset(frame):
    return {
        "rows": len(frame),
        "source_identifiers": int(frame.source.nunique()),
        "distinct_labels": int(frame.label.nunique()),
        "distinct_reported_compositions_including_missingness": len(
            frame[COMPOSITION].drop_duplicates()
        ),
        "exact_duplicate_rows_beyond_first": int(frame.duplicated().sum()),
    }


def main():
    filename = HERE / "raw/Pitting_potential_dataset_2024.xlsx"
    data = pd.read_excel(filename, sheet_name="pitting_potentials")
    refs = pd.read_excel(filename, sheet_name="references")
    refs["doi_normalized"] = refs.DOI.map(doi)
    nyby_path = DATASETS / "electrochemical_metrics_alloys/raw/CRA_database_Scientific_Data_Publication_12102020.xlsx"
    nyby_refs = pd.read_excel(nyby_path, sheet_name="References")
    nyby_refs["doi_normalized"] = nyby_refs.DOI.map(doi)
    nyby_pit = pd.read_excel(nyby_path, sheet_name="Pitting Potential", header=1)
    # The source workbook has a two-row header: column 37 is Reference.
    pit_ref_ids = pd.to_numeric(nyby_pit.iloc[:, 37], errors="coerce")
    doi_sets = {
        "all_nyby": set(nyby_refs.doi_normalized.dropna()),
        "pitting_nyby": set(
            nyby_refs.loc[nyby_refs.No.isin(pit_ref_ids), "doi_normalized"].dropna()
        ),
    }
    numeric_target = pd.to_numeric(data.E_pit, errors="coerce").notna()
    observed = data.event.eq(1) & numeric_target
    flags = data[["label", "source", "ID", "event"]].copy()
    flags.insert(0, "excel_row", range(2, len(data) + 2))
    flags["numeric_target"] = numeric_target
    flags["observed_numeric_target"] = observed
    mapped_doi = data.source.map(refs.set_index("source").doi_normalized)
    flags["doi_normalized"] = mapped_doi
    summary = {
        "github_commit": "50f2df0918116e12026f8ea18c32704a5f6822e5",
        "soccol": {
            "all": describe_subset(data),
            "columns": list(data.columns),
            "numeric_E_pit": int(numeric_target.sum()),
            "observed_numeric": describe_subset(data[observed]),
            "censored_event_0": int(data.event.eq(0).sum()),
            "censored_numeric": int((data.event.eq(0) & numeric_target).sum()),
            "unknown_event": int(data.event.isna().sum()),
            "unknown_event_numeric": int((data.event.isna() & numeric_target).sum()),
            "observed_missing_target": int((data.event.eq(1) & ~numeric_target).sum()),
            "reference_rows": len(refs),
            "reference_rows_missing_doi": int(refs.doi_normalized.isna().sum()),
            "overlap": {},
        },
    }
    for name, reference_dois in doi_sets.items():
        shared_refs = refs[refs.doi_normalized.isin(reference_dois)]
        shared = data.source.isin(shared_refs.source)
        flags[f"shared_{name}_source_doi"] = shared
        summary["soccol"]["overlap"][name] = {
            "shared_source_identifiers": len(shared_refs),
            "shared_distinct_dois": int(shared_refs.doi_normalized.nunique()),
            "shared_all_rows": int(shared.sum()),
            "shared_observed_numeric_rows": int((shared & observed).sum()),
            "remaining_observed_numeric": describe_subset(data[observed & ~shared]),
            "remaining_observed_numeric_with_doi": describe_subset(
                data[observed & ~shared & mapped_doi.notna()]
            ),
        }
        shared_refs.to_csv(HERE / f"overlap_{name}.csv", index=False)
    flags["shared_nyby_source_bibliographic_match"] = data.source.isin(MANUAL_SOURCE_MATCHES)
    remaining = observed & ~flags.shared_all_nyby_source_doi & ~flags.shared_nyby_source_bibliographic_match
    summary["soccol"]["manual_bibliographic_matches"] = MANUAL_SOURCE_MATCHES
    summary["soccol"]["additional_observed_rows_excluded_by_bibliographic_match"] = int(
        (observed & ~flags.shared_all_nyby_source_doi & flags.shared_nyby_source_bibliographic_match).sum()
    )
    summary["soccol"]["after_all_known_source_exclusions"] = describe_subset(data[remaining])
    summary["soccol"]["remaining_without_doi"] = describe_subset(data[remaining & mapped_doi.isna()])
    flags["provisional_regression_after_known_source_exclusion"] = remaining
    flags.to_csv(HERE / "soccol_row_audit.csv", index=False)
    refs[refs.doi_normalized.isna()].to_csv(HERE / "soccol_references_missing_doi.csv", index=False)
    nyby_refs[nyby_refs.doi_normalized.isna()].to_csv(HERE / "nyby_references_missing_doi.csv", index=False)
    # Numeric coverage: never interpret missing composition as zero.
    numeric_features = COMPOSITION + [
        "CP_temp", "CP_pH", "CP_Cl", "CP_Br", "CP_SO4", "CP_NO3",
        "Prep_grinding_grit", "Prep_Ra_micron", "CP_time", "scan_rate", "Test_area_cm2",
    ]
    coverage = []
    for name, mask in [("observed_numeric", observed), ("after_known_source_exclusion", remaining)]:
        frame = data.loc[mask, numeric_features].apply(pd.to_numeric, errors="coerce")
        for column in frame:
            coverage.append({"subset": name, "column": column, "rows": len(frame),
                             "numeric_count": int(frame[column].notna().sum()),
                             "coverage_fraction": float(frame[column].notna().mean())})
    pd.DataFrame(coverage).to_csv(HERE / "soccol_feature_coverage.csv", index=False)
    mpea = pd.read_excel(DATASETS / "mpea_corrosion/raw/Mendeley_MPEA_corrosion_database.xlsx")
    mpea_valid = pd.to_numeric(mpea["Pitting potential (mV vs SCE)"], errors="coerce").notna()
    mpea_shared = mpea.Reference.map(doi).isin(doi_sets["all_nyby"])
    summary["mpea"] = {
        "raw_rows": len(mpea), "numeric_E_pit": int(mpea_valid.sum()),
        "numeric_E_pit_in_NaCl": int((mpea_valid & mpea.Electrolyte.eq("NaCl")).sum()),
        "source_strings_among_numeric": int(mpea.loc[mpea_valid, "Reference"].nunique()),
        "numeric_E_pit_matching_nyby_doi": int((mpea_valid & mpea_shared).sum()),
        "numeric_E_pit_after_known_source_exclusion": int((mpea_valid & ~mpea_shared).sum()),
        "paper_reports_E_pit_rows": 306,
        "note": "Public workbook count differs from the ML paper; preprocessing/version difference unresolved.",
    }
    summary["rebar"] = []
    for sheet, header in [("Carbon steel 1", 3), ("Carbon steel 2", 2),
                          ("Stainless steel 1", 3), ("Stainless steel 2", 2)]:
        frame = pd.read_excel(HERE / "raw/rebar_electrochemical_parameters.xlsx", sheet_name=sheet, header=header)
        target = next(c for c in frame if "Epit/" in str(c) or "Eb/" in str(c))
        valid = pd.to_numeric(frame[target], errors="coerce").notna()
        summary["rebar"].append({"sheet": sheet, "parsed_rows": len(frame),
                                 "target": target, "numeric_target_rows": int(valid.sum())})
    manifest = []
    for path in sorted((HERE / "raw").iterdir()):
        manifest.append({"file": str(path.relative_to(HERE)), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (HERE / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
