"""Regression checks for MPEA candidate preparation, not model evaluation."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "corrosion_datasets/datasets/mpea_corrosion/prepare_evaluation.py"
SPEC = importlib.util.spec_from_file_location("mpea_preparation", MODULE_PATH)
mpea = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mpea)


def environment(text, electrolyte="NaCl", molarity=0.6):
    return pd.Series({"Test environment": text, "Electrolyte": electrolyte, "Concentration in M": molarity})


def test_atomic_to_weight_percent_uses_excluded_elements_in_denominator():
    fractions = pd.DataFrame([dict.fromkeys(mpea.ELEMENTS, 0.0)])
    fractions.loc[0, ["Fe", "Zr"]] = [0.5, 0.5]
    weights, total, valid = mpea.convert_composition(fractions)
    expected_fe = 100 * mpea.ATOMIC_WEIGHTS["Fe"] / (mpea.ATOMIC_WEIGHTS["Fe"] + mpea.ATOMIC_WEIGHTS["Zr"])
    assert valid.iloc[0] and total.iloc[0] == 1
    assert weights.loc[0, "Fe"] == pytest.approx(expected_fe)
    assert weights[mpea.RETAINED_ELEMENTS].sum(axis=1).iloc[0] < 100
    assert weights.sum(axis=1).iloc[0] == pytest.approx(100)


@pytest.mark.parametrize("fe", [0.9375, np.nan, -1.0, np.inf])
def test_invalid_composition_is_not_repaired_or_zero_filled(fe):
    fractions = pd.DataFrame([dict.fromkeys(mpea.ELEMENTS, 0.0)])
    fractions.loc[0, "Fe"] = fe
    weights, _, valid = mpea.convert_composition(fractions)
    assert not valid.iloc[0]
    assert weights.isna().all().all()


def test_missing_conditions_stay_missing_and_zero_is_not_missing():
    values, flags, _ = mpea.parse_environment(environment("0 M NaCl", molarity=0))
    assert values["CP_Cl"] == 0
    assert not flags
    assert np.isnan(values["CP_temp"]) and np.isnan(values["CP_pH"])
    assert np.isnan(values["CP_OH"])


@pytest.mark.parametrize("text,electrolyte,molarity,flag", [
    ("0.6 M NaCl", "H2SO4", 0.5, "electrolyte_text_conflict"),
    ("0.5 M NaCl", "NaCl", 0.6, "molarity_text_conflict"),
    ("3.5wt% NaCl", "NaCl", 0.6, "mass_percent_to_molarity_unverified"),
    ("5% NaOH", "NaOH", 5, "mass_percent_to_molarity_unverified"),
    ("Seawater", "Seawater", 0.55, "mixed_or_unspecified_electrolyte_recipe"),
    ("0.5 M H2SO4", "H2SO4", 0.5, "acid_to_ion_mapping_unverified"),
])
def test_ambiguous_electrolytes_do_not_become_ion_features(text, electrolyte, molarity, flag):
    values, flags, _ = mpea.parse_environment(environment(text, electrolyte, molarity))
    assert flag in flags
    assert all(np.isnan(values[name]) for name in ("CP_Cl", "CP_OH", "CP_SO4", "CP_NO3"))


def test_temperature_and_scan_rate_are_extracted_independently_of_ion_ambiguity():
    values, flags, _ = mpea.parse_environment(environment("3.5% NaCl solution at 353K"))
    assert values["CP_temp"] == pytest.approx(79.85)
    assert "mass_percent_to_molarity_unverified" in flags
    values, _, _ = mpea.parse_environment(environment("Seawater (scan rate of 0.1mV/s)", "Seawater", 0.55))
    assert values["scan_rate"] == 0.1
    assert np.isnan(values["CP_Cl"])
    values, _, _ = mpea.parse_environment(environment("Phosphate buffer solution (37 ◦C)", "PBS", np.nan))
    assert values["CP_temp"] == 37
    assert np.isnan(values["CP_pH"])


def test_simple_hydroxide_mapping():
    values, flags, _ = mpea.parse_environment(environment("1 M KOH", "KOH", 1))
    assert values["CP_OH"] == 1 and not flags


@pytest.mark.parametrize("text,expected", [
    ("\ufeffhttps://doi.org/10.1016/0010-938X(69)80007-7", ["10.1016/0010-938x(69)80007-7"]),
    ("150.203.68.65 (https://iopscience.iop.org/article/10.1149/07711.0741ecst/pdf)", ["10.1149/07711.0741ecst"]),
    ("https://www.tandfonline.com/action/showCitFormats?doi=10.1179%2F1743284715Y.0000000026", ["10.1179/1743284715y.0000000026"]),
    ("www.sciencedirect.com", []),
    (np.nan, []),
])
def test_doi_extraction(text, expected):
    assert mpea.extract_dois(text) == expected


def test_candidate_selection_keeps_zero_but_discards_invalid_compositions_to_audit():
    rows = []
    for target, fe, reference in [(0, 1, "10.1234/shared"), (50, 0.9375, "unknown"), (np.nan, 1, "10.1234/other")]:
        rows.append({**dict.fromkeys(mpea.ELEMENTS, 0.0), "Fe": fe,
                     "Alloy name": "test", "Reference": reference, mpea.RAW_TARGET: target,
                     **environment("NaCl").to_dict()})
    candidates, audit, sources = mpea.prepare_tables(pd.DataFrame(rows), {"10.1234/shared"}, set())
    assert candidates[mpea.TARGET].tolist() == [0]
    assert candidates["row_id"].tolist() == ["mpea_v1_r000002"]
    assert audit["source_overlap_status"].tolist() == ["shared_source_doi", "unresolved_reference", "no_shared_doi_found"]
    assert audit["has_finite_target"].tolist() == [True, True, False]
    assert audit["included_in_candidate_table"].tolist() == [True, False, False]
    assert audit["exclusion_reason"].tolist() == ["", "composition_not_convertible", "missing_or_nonfinite_target"]
    assert not audit["included_in_source_screened_subset"].any()
    assert sources["candidate_rows"].sum() == 1
    assert candidates.columns.tolist() == ["row_id", *mpea.FEATURES, mpea.TARGET]
    assert not set(candidates.columns) & {"Reference", "Calculated passive window", "Corrosion potential (mV vs SCE)"}


def test_local_workbook_contract():
    if not mpea.RAW_PATH.exists():
        pytest.skip("Local raw MPEA workbook not present")
    assert mpea.file_sha256(mpea.RAW_PATH) == mpea.EXPECTED_RAW_SHA256
    raw = pd.read_excel(mpea.RAW_PATH)
    candidates, audit, _ = mpea.prepare_tables(raw, set(), set())
    selected = audit.loc[audit["included_in_candidate_table"]]
    assert len(raw) == 619 and len(candidates) == 332
    assert audit["has_finite_target"].sum() == 335
    assert audit.loc[audit["exclusion_reason"].eq("composition_not_convertible"), "raw_workbook_row"].tolist() == [466, 467, 468]
    assert not candidates["row_id"].isin(["mpea_v1_r000466", "mpea_v1_r000467", "mpea_v1_r000468"]).any()
    assert (candidates[mpea.TARGET] == 0).sum() == 1
    assert selected["composition_conversion_valid"].sum() == 332
    assert selected["unsupported_elements_present"].ne("").sum() == 67
    assert candidates["CP_temp"].notna().sum() == 5
    assert candidates["scan_rate"].notna().sum() == 18
    assert candidates["CP_pH"].isna().all()


def test_local_cli_reproduction_and_provenance(tmp_path):
    if not all(path.exists() for path in (mpea.RAW_PATH, mpea.EPIT_PATH, mpea.SOCCOL_PATH)):
        pytest.skip("Local source workbooks not present")
    command = [sys.executable, str(MODULE_PATH), "--output-dir", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    hashes = {path.name: mpea.file_sha256(path) for path in tmp_path.iterdir()}
    manifest = json.loads((tmp_path / "preprocessing_manifest.json").read_text())
    features = json.loads((tmp_path / "feature_manifest.json").read_text())
    for record in manifest["inputs"]:
        assert mpea.file_sha256(REPO_ROOT / record["path"]) == record["sha256"]
    for name, record in manifest["outputs"].items():
        assert hashes[name] == record["sha256"]
    assert features["predictor_columns_in_order"] == mpea.FEATURES
    assert manifest["summary"]["candidate_source_overlap_status"] == {
        "no_shared_doi_found": 267, "unresolved_reference": 43, "shared_source_doi": 22,
    }
    assert manifest["summary"]["excluded_missing_or_nonfinite_target"] == 284
    assert manifest["summary"]["excluded_incomplete_composition_with_target"] == 3
    assert manifest["summary"]["paper_count_discrepancy"]["workbook_numeric_epit_entries"] == 335
    screened = pd.read_csv(tmp_path / "mpea_source_screened_candidates.csv")
    audit = pd.read_csv(tmp_path / "mpea_row_audit.csv")
    selected = audit[audit["included_in_source_screened_subset"]]
    assert len(screened) == 267
    assert set(screened["row_id"]) == set(selected["row_id"])
    assert selected["source_overlap_status"].eq("no_shared_doi_found").all()
    assert screened.columns.tolist() == ["row_id", *mpea.FEATURES, mpea.TARGET]
    assert not manifest["frozen_partitions_created"]
    assert not manifest["models_trained_or_evaluated"]
    assert not manifest["exact_cross_dataset_row_overlap_assessed"]
    assert (tmp_path / "mpea_evaluation_candidates.csv").read_text().find("NaN") != -1
    subprocess.run([*command, "--overwrite"], check=True, capture_output=True, text=True)
    assert hashes == {path.name: mpea.file_sha256(path) for path in tmp_path.iterdir()}
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0 and "FileExistsError" in refused.stderr
