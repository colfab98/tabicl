from __future__ import annotations

import json

import pandas as pd
import pytest

from corrosion_datasets.datasets.soccol_pitting_potential.prepare_soccol_environment import (
    FEATURE_COLUMNS,
    HEAT_COLUMNS,
    LEGACY_COMPOSITION,
    LEGACY_TABLE,
    SOURCE_ENVIRONMENT,
    SPLIT_MANIFEST,
    column_definition,
    load_inputs,
    load_rules,
    numeric_source,
    prepare,
    reviewed_rule,
    sha256_file,
    transform,
)


@pytest.fixture(scope="module")
def rules():
    return load_rules()


@pytest.fixture(scope="module")
def inputs():
    return load_inputs()


def test_preparation_never_becomes_test_solution_or_heat_duration(rules):
    prep = reviewed_rule(rules, "Prep_medium", "pickling HNO3 + HF")["features"]
    assert prep == {"surface_preparation_method": "pickling", "preparation_solution_type": "HF + HNO3"}
    hot_acid = reviewed_rule(rules, "Prep_medium", "+ nitric (20% - 50degC)")["features"]
    assert hot_acid == {"preparation_solution_type": "HNO3"}
    exposure = reviewed_rule(rules, "Prep_medium", "(24h air)")["features"]
    assert exposure == {"surface_preparation_method": "air_exposure"}
    assert not set(exposure) & {"CP_time", "Prep_time", "CP_aeration"}
    timed = reviewed_rule(rules, "Prep_medium", "0.5 V - 30'")["features"]
    assert timed["Prep_time"] == .5
    assert timed["Prep_redox"] == .5
    assert not set(timed) & set(HEAT_COLUMNS)
    assert column_definition("CP_time")["unit"] == "h"
    assert "reference not established" in column_definition("Prep_redox")["unit"]


def test_solution_notes_do_not_invent_counterions_or_complete_recipes(rules):
    acid = reviewed_rule(rules, "CP_anions_info", "0.5% MgSO4 + 0.1M H2SO4")["features"]
    assert acid == {"H2SO4_mol_L": .1}
    assert reviewed_rule(rules, "CP_anions_info", "0.6mM SO4")["features"] == {"CP_SO4": .0006}
    assert reviewed_rule(rules, "CP_anions_info", "2x25mM HCO3+CO3")["features"] == {"CP_CO3": .025}
    for note in ["none", "pure chloride", "dichromate", "0.5% MgSO4", "borate buffer", "citrate"]:
        assert not reviewed_rule(rules, "CP_anions_info", note)["features"]
    assert reviewed_rule(rules, "CP_anions_info", "sea water")["features"] == {"test_solution_type": "seawater"}
    assert reviewed_rule(rules, "CP_anions_info", "artificial sea water")["features"] == {"test_solution_type": "synthetic_seawater"}


def test_heat_evidence_is_source_specific_and_grit_is_not_converted_twice(rules):
    heat = reviewed_rule(rules, "Prep_medium", "(annealed 1040)", "2019Ha")["features"]
    assert heat == {"heat_treatment_type": "annealing", "heat_treatment_temperature_C": 1040,
                    "heat_treatment_duration_h": 1, "heat_treatment_cooling": "water_quench"}
    with pytest.raises(ValueError, match="Source-specific"):
        reviewed_rule(rules, "Prep_medium", "(annealed 1040)", "different_paper")
    generic = reviewed_rule(rules, "Prep_medium", "(anneal+quench)")["features"]
    assert generic == {"heat_treatment_type": "annealing", "heat_treatment_cooling": "quench_unspecified"}
    grit = reviewed_rule(rules, "Prep_medium", "(1200 = 600 US grit)")["features"]
    assert grit["Prep_grinding_grit"] == 600


def test_missingness_explicit_zeros_conflicts_and_unit_conversions(inputs, rules):
    raw = inputs[0]
    indices = [raw.index[raw.CP_anions_info.eq("0.06M SO4")][0],
               raw.index[raw.Prep_medium.eq("0.436mm2 wire")][0],
               raw.index[raw.Prep_medium.eq("(1200 = 600 US grit)")][0]]
    sample = raw.loc[indices].astype(object).copy()
    sample.loc[indices[0], "CP_agitation"] = " "
    sample.loc[indices[0], "CP_Br"] = ""
    sample.loc[indices[1], "CP_agitation"] = "no"
    sample.loc[indices[1], "CP_Br"] = 0
    sample.loc[indices[2], "CP_agitation"] = "yes"
    features, audit = transform(sample, rules)
    assert pd.isna(features.loc[indices[0], "CP_SO4"])
    assert pd.isna(features.loc[indices[1], "Test_area_cm2"])
    assert features.loc[indices[2], "Prep_grinding_grit"] == 600
    assert pd.isna(features.loc[indices[0], "CP_agitation"])
    assert features.loc[indices[1], "CP_agitation"] == 0
    assert features.loc[indices[2], "CP_agitation"] == 1
    assert pd.isna(features.loc[indices[0], "CP_Br"])
    assert features.loc[indices[1], "CP_Br"] == 0
    assert audit.loc[audit.status.eq("conflict"), "row_id"].nunique() == 2
    # A missing numeric field can be filled from explicit text; an existing
    # disagreement must instead stay missing, as checked above.
    sample.loc[indices[0], "CP_SO4"] = ""
    sample.loc[indices[1], "Test_area_cm2"] = ""
    filled, _ = transform(sample, rules)
    assert filled.loc[indices[0], "CP_SO4"] == .06
    assert filled.loc[indices[1], "Test_area_cm2"] == .00436


def test_targets_other_ion_and_composition_are_not_read_for_extraction(inputs, rules):
    raw = inputs[0]
    sample = raw.loc[pd.to_numeric(raw.CP_ion_other, errors="coerce").gt(0)].copy()
    original, _ = transform(sample, rules)
    for col in ["E_pit", "E_corr", "event", "CP_ion_other", "Cr", "label"]:
        sample[col] = "arbitrary text; 1 M HNO3; annealed 9999 C"
    changed, _ = transform(sample, rules)
    pd.testing.assert_frame_equal(original, changed)
    assert "CP_ion_other" not in original
    assert not any("borate" in c or "citrate" in c for c in original)
    assert numeric_source(-1, "CP_pH") == -1
    with pytest.raises(ValueError, match="Unreviewed"):
        numeric_source("5 minutes", "CP_time")
    with pytest.raises(ValueError, match="Unreviewed"):
        reviewed_rule(rules, "CP_anions_info", "new solution")


def test_export_keeps_frozen_rows_legacy_composition_and_feature_boundary(tmp_path, inputs):
    raw, legacy, assignments, paths = inputs
    hashes = {path: sha256_file(path) for path in paths}
    output = tmp_path / "soccol_environment"
    manifest = prepare(output)
    features = pd.read_csv(output / "soccol_environment_features.csv")
    reference = pd.read_csv(output / "soccol_processed.csv", dtype=str, keep_default_na=False)
    original = pd.read_csv(LEGACY_TABLE, dtype=str, keep_default_na=False)
    preserved = original.columns.drop(SOURCE_ENVIRONMENT)
    pd.testing.assert_frame_equal(reference[preserved], original[preserved])
    assert all(name in reference for name in LEGACY_COMPOSITION)
    assert all(f"raw_{name}" in reference for name in SOURCE_ENVIRONMENT)
    assert "CP_ion_other" not in reference
    assert features.columns.tolist() == ["row_id", *FEATURE_COLUMNS]
    assert features.row_id.tolist() == raw.row_id.tolist() == legacy.row_id.tolist()
    assert len(features) == 4027
    assert sha256_file(output / "split_assignments.csv") == sha256_file(SPLIT_MANIFEST.parent / "split_assignments.csv")
    assert manifest["counts"]["development_rows"] == 3222
    assert manifest["counts"]["final_test_rows"] == 805
    assert all(sha256_file(p) == digest for p, digest in hashes.items())
    for name, digest in manifest["output_sha256"].items():
        assert sha256_file(output / name) == digest
    registry = json.loads((output / "feature_registry.json").read_text())
    assert registry["composition_columns"] == []
    assert registry["feature_columns"] == FEATURE_COLUMNS
    for col in ["E_pit", "E_corr", "event", "CP_ion_other", "Prep_medium", "CP_anions_info", *LEGACY_COMPOSITION]:
        assert col not in features
    # Raw saved source cells, including excluded other-ion values, remain
    # recoverable in the separate audit table rather than entering features.
    saved_raw = pd.read_csv(output / "source_rows.csv", dtype=str, keep_default_na=False)
    expected_raw = raw.copy()
    for col in saved_raw:
        assert saved_raw[col].map(str).tolist() == expected_raw[col].map(str).tolist()
    with pytest.raises(FileExistsError):
        prepare(output)
