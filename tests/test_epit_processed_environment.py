from __future__ import annotations

import json

import pandas as pd
import pytest

from corrosion_datasets.datasets.electrochemical_metrics_alloys.prepare_epit_environment import (
    COMPOSITION_DIR,
    DIRECT_FIELDS,
    ELEMENTS,
    PHASE_COLUMNS,
    RULES_PATH,
    TARGET_COLUMN,
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


def test_solution_components_and_units_are_not_conflated(rules):
    mix = reviewed_rule(rules, "Test Solution", "1M H2SO4 + 0.5M NaCl, H2 deaerated")["features"]
    assert mix["H2SO4_mol_L"] == 1
    assert mix["NaCl_mol_L"] == .5
    assert mix["CP_aeration"] == "deaerated"
    normal = reviewed_rule(rules, "Test Solution", "Deaerated (nitrogen purge) 1 N H2SO4 + 0.5 M NaCl")["features"]
    assert normal["H2SO4_eq_L"] == 1
    assert "H2SO4_mol_L" not in normal
    assert normal["NaCl_mol_L"] == .5
    mass = reviewed_rule(rules, "Test Solution", "3.5 wt.% NaCl")["features"]
    unknown = reviewed_rule(rules, "Test Solution", "3.5% NaCl")["features"]
    assert mass["NaCl_wt_percent"] == 3.5
    assert unknown["NaCl_percent_unspecified"] == 3.5
    assert "NaCl_mol_L" not in mass
    assert "NaCl_wt_percent" not in unknown
    assert reviewed_rule(rules, "Test Solution", "0.1 M NaCIO4")["features"] == {
        "test_solution_type": "NaClO4", "NaClO4_mol_L": .1,
    }
    chloride = reviewed_rule(rules, "Test Solution", "0.1 M ChlorideDeaerated")["features"]
    assert chloride["test_solution_type"] == "chloride_unspecified"
    assert "NaCl_mol_L" not in chloride


def test_heat_sequences_are_separate_but_alternatives_are_not_stages(rules):
    text = "Melted in the laboratory. The ingots were forged, hot rolled to plates of 5 mm in thickness, and then solution heat treated at 1050 C for 30 min, followed by water quenching, then 500 C heated for 1 h"
    values = reviewed_rule(rules, "Heat treatment", text)["features"]
    assert values["heat_treatment_temperature_C"] == 1050
    assert values["heat_treatment_duration_h"] == .5
    assert values["heat_treatment_cooling"] == "water_quench"
    assert values["heat_treatment_2_temperature_C"] == 500
    assert values["heat_treatment_2_duration_h"] == 1
    assert "heat_treatment_2_cooling" not in values
    alternatives = reviewed_rule(rules, "Heat treatment", "solution-annealed for 2 h at 1,050°C and then aged at 850°C for 30 min, 1 h, and 10 h.")["features"]
    assert alternatives["heat_treatment_duration_h"] == 2
    assert alternatives["heat_treatment_2_temperature_C"] == 850
    assert "heat_treatment_2_duration_h" not in alternatives
    fahrenheit = reviewed_rule(rules, "Heat treatment", "heat-treated for 1 hr at 1500F followed by water quenching")["features"]
    assert fahrenheit["heat_treatment_temperature_C"] == pytest.approx(815.5555555556)
    ordering = next(r for t, r in rules["Heat treatment"].items() if "7 days at 500" in t)["features"]
    assert ordering["heat_treatment_2_duration_h"] == 168
    for text in ["NA", "None", "", "as cast"]:
        assert not reviewed_rule(rules, "Heat treatment", text)["features"]


def test_phase_presence_absence_and_general_discussion_are_distinct(rules):
    mix = reviewed_rule(rules, "Microstructures", "FCC and BCC")["features"]
    assert mix == {"phase_FCC": 1, "phase_BCC": 1}
    assert reviewed_rule(rules, "Microstructures", "austenitic")["features"] == {"phase_FCC": 1}
    assert reviewed_rule(rules, "Microstructures", "𝛿-ferrite")["features"] == {"phase_BCC": 1}
    none = reviewed_rule(rules, "Microstructures", "Metallographic examinations did not show the presence of intermetallic phases in any of the materials.")["features"]
    assert none == {"phase_IM": 0}
    single = reviewed_rule(rules, "Microstructures", "Single austenitic phase. No second phase. Two types of inclusions, MnS and alumina, were found")["features"]
    assert single == {"phase_FCC": 1, "phase_BCC": 0, "phase_HCP": 0, "phase_IM": 0}
    general = next(t for t in rules["Microstructures"] if t.startswith("Nitrogen, which"))
    for text in [general, "secondary phases present", "Nanocrystalline", "NA"]:
        assert reviewed_rule(rules, "Microstructures", text)["features"] == {}


def test_procedure_text_and_unseen_notation_are_never_silently_numeric(rules):
    for text in ["NA", "steady state", "20mV steps by manual increament"]:
        assert numeric_source(text, "Scan Rate mV/s") is None
    assert numeric_source("0.0028", "Scan Rate mV/s") == .0028
    assert numeric_source("-1", "[Cl-] pH") == -1
    with pytest.raises(ValueError, match="Unreviewed"):
        numeric_source("2 V/s", "Scan Rate mV/s")
    with pytest.raises(ValueError, match="Unreviewed"):
        reviewed_rule(rules, "Test Solution", "unreviewed new solution")


def test_target_values_and_unrelated_text_do_not_change_extraction(rules):
    source = pd.read_csv(COMPOSITION_DIR / "epit_processed.csv", dtype=str, keep_default_na=False).head(12)
    original, _ = transform(source, rules)
    modified = source.copy()
    modified[TARGET_COLUMN] = "999999"
    modified["Comments"] = "arbitrary outcome; annealed at 9999C in HNO3"
    changed, _ = transform(modified, rules)
    derived = original.columns[len(source.columns):]
    pd.testing.assert_frame_equal(original[derived], changed[derived])


def test_export_preserves_original_data_splits_and_feature_boundary(tmp_path):
    output = tmp_path / "environment"
    old_paths = list(COMPOSITION_DIR.iterdir()) + [RULES_PATH]
    old_hashes = {p: sha256_file(p) for p in old_paths if p.is_file()}
    manifest = prepare(output)
    original = pd.read_csv(COMPOSITION_DIR / "epit_processed.csv", dtype=str, keep_default_na=False)
    strings = pd.read_csv(output / "epit_processed.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(strings[original.columns], original)
    assert sha256_file(output / "split_assignments.csv") == sha256_file(COMPOSITION_DIR / "split_assignments.csv")
    assert all(sha256_file(path) == digest for path, digest in old_hashes.items())
    data = pd.read_csv(output / "epit_processed.csv")
    registry = json.loads((output / "feature_registry.json").read_text())
    features = pd.read_csv(output / "epit_features.csv")
    assert features.columns.tolist() == ["row_id", *registry["feature_columns"]]
    assert data.row_id.tolist() == original.row_id.tolist() == features.row_id.tolist()
    assert len(data) == 760
    assert registry["composition_columns"] == ELEMENTS
    for name in [TARGET_COLUMN, "Test Solution", "Heat treatment", "Microstructures", "[Cl-] Test Method", "Reference", "Comments", "NaCl_percent_unspecified", "NaCl_ppm_unspecified"]:
        assert name not in features
    assert set(registry["audit_only_derived_columns"]) == {"NaCl_percent_unspecified", "NaCl_ppm_unspecified"}
    for name in PHASE_COLUMNS:
        assert set(data[name].dropna()) <= {0, 1}
    assert data.loc[data.Microstructures.eq("FCC and BCC"), "phase_HCP"].isna().all()
    for source, (target, _) in DIRECT_FIELDS.items():
        expected = pd.to_numeric(original[source], errors="coerce")
        if target == "CP_Cl":
            expected.loc[original["Test Solution"].eq("0.1 M NaCIO4")] = float("nan")
        pd.testing.assert_series_equal(data[target], expected, check_names=False, check_dtype=False)
    contradiction = data.loc[data["Test Solution"].eq("0.1 M NaCIO4")]
    assert len(contradiction) == 1
    assert contradiction.CP_Cl.isna().all()
    assert contradiction["[Cl-] M"].eq(.1).all()
    assert manifest["counts"]["development_rows"] == 608
    assert manifest["counts"]["final_test_rows"] == 152
    for name, digest in manifest["output_sha256"].items():
        assert sha256_file(output / name) == digest
    with pytest.raises(FileExistsError):
        prepare(output)
