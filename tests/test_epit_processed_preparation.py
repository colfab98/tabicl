from __future__ import annotations

import json

import pandas as pd
import pytest

from corrosion_datasets.datasets.electrochemical_metrics_alloys.prepare_epit_preparation import (
    ENVIRONMENT_V1, NEW_COLUMNS, RULES_PATH, column_definition,
    load_rules, prepare, reviewed_rule, sha256_file, transform,
)


@pytest.fixture(scope="module")
def source():
    return pd.read_csv(ENVIRONMENT_V1 / "epit_processed.csv", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def rules():
    return load_rules()


@pytest.fixture(scope="module")
def extracted(source, rules):
    return transform(source, rules)


def test_grit_is_not_mesh_particle_size_or_another_specimens_finish(source, extracted):
    values = extracted[0]
    assert values.loc[source.Comments.eq("Ground to 600"), "Prep_grinding_grit"].eq(600).all()
    assert values.loc[source.Comments.str.contains("600 mesh", regex=False), "Prep_grinding_grit"].isna().all()
    assert values.loc[source.Comments.str.contains("abraded through 2.0 emery", regex=False), "Prep_grinding_grit"].isna().all()
    assert values.loc[source.Comments.eq("Surface finish: 4/0"), "Prep_grinding_grit"].isna().all()
    other_coupons = source["[Cl-] Test Method"].str.contains("The coupons for the exposure tests", regex=False)
    assert other_coupons.sum() == 3
    assert values.loc[other_coupons, "Prep_grinding_grit"].isna().all()
    assert values.loc[other_coupons, "surface_preparation_method"].eq("polishing").all()
    assert "Prep_Ra_micron" not in values


def test_waiting_preparation_and_measurement_times_are_separate(source, extracted):
    values, audit = extracted
    method = source["[Cl-] Test Method"]
    first = method.str.contains("50 minutes after the immersion", regex=False)
    assert values.loc[first, "CP_time"].eq(50 / 60).all()
    assert values.loc[first, "Prep_time"].isna().all()
    seconds = method.str.contains("cathodically polarizes for 300s", regex=False)
    assert values.loc[seconds, "Prep_time"].eq(300 / 3600).all()
    assert values.loc[seconds, "CP_time"].isna().all()
    lower_bound = method.str.contains("for at least 15 min", regex=False)
    assert lower_bound.sum() == 18
    assert values.loc[lower_bound, "Prep_time"].isna().all()
    assert values.loc[lower_bound, "CP_time"].isna().all()
    stepped = method.str.contains("kept immersed in the experimental solution for 5 min", regex=False)
    assert values.loc[stepped, "CP_time"].eq(5 / 60).all()
    assert audit.loc[audit.source_column.eq("preparation duration qualification"), "row_id"].nunique() == 18


def test_preparation_chemistry_stays_separate_and_stationary_current_is_not_agitation(source, extracted):
    values = extracted[0]
    pickle = source["Heat treatment"].str.contains("pickled again in the same HNO3-HF", regex=False)
    assert values.loc[pickle, "preparation_solution_type"].eq("HF + HNO3").all()
    acid = source["[Cl-] Test Method"].str.contains("0.1 M hydrochloric acid (HCl) for 15 min", regex=False)
    assert values.loc[acid, "preparation_solution_type"].eq("HCl").all()
    rotating = source["[Cl-] Test Method"].str.contains("rotating disk", regex=False)
    assert rotating.sum() == 6
    assert values.loc[rotating, "CP_agitation"].eq(1).all()
    assert values.loc[~rotating, "CP_agitation"].isna().all()
    for name in ["test_solution_type", "HCl_mol_L", "H2SO4_mol_L", "CP_pH", "CP_temp", "CP_aeration"]:
        pd.testing.assert_series_equal(values[name], source[name])
    assert not {"Prep_redox", "Prep_pH"} & set(values)


def test_ions_are_mapped_only_when_the_reported_recipe_supports_it(source, extracted):
    values = extracted[0]
    bromide = source.NaBr_mol_L.ne("")
    assert bromide.sum() == 25
    pd.testing.assert_series_equal(values.loc[bromide, "CP_Br"], pd.to_numeric(source.loc[bromide, "NaBr_mol_L"]), check_names=False)
    normality = source.NaBr_eq_L.ne("")
    assert normality.sum() == 12
    assert values.loc[normality, "CP_Br"].isna().all()
    salt = source["Test Solution"].eq("0.1M Na2SO4 + 0.6M NaCl")
    assert salt.sum() == 2
    assert values.loc[salt, "CP_SO4"].eq(.1).all()
    acid = source.test_solution_type.str.contains("H2SO4", regex=False)
    assert values.loc[acid, "CP_SO4"].isna().all()
    assert values.loc[~salt, "CP_SO4"].isna().all()


def test_targets_and_outcome_commentary_do_not_supply_features(source, rules):
    sample = source.head(20).copy()
    original, _ = transform(sample, rules)
    for name in ["Epit, mV (SCE) Avg.", "Fe", "Reference"]:
        sample[name] = "arbitrary outcome value"
    changed, _ = transform(sample, rules)
    pd.testing.assert_frame_equal(original[NEW_COLUMNS], changed[NEW_COLUMNS])
    # Two original comments differ by an outcome interpretation, but share
    # the same preparation clause and must yield identical extracted features.
    with_outcome = next(t for t in rules["Comments"] if t.startswith("Since no chloride ion was present."))
    without_outcome = next(t for t in rules["Comments"] if t.startswith("The alloys were mecanically polished"))
    assert reviewed_rule(rules, "Comments", with_outcome)["features"] == reviewed_rule(rules, "Comments", without_outcome)["features"]
    with pytest.raises(ValueError, match="Unreviewed"):
        reviewed_rule(rules, "Comments", "unreviewed preparation description")


def test_export_preserves_previous_columns_split_and_shared_units(tmp_path, source):
    output = tmp_path / "epit_environment_v2"
    old_paths = [p for p in ENVIRONMENT_V1.iterdir() if p.is_file()] + [RULES_PATH]
    old_hashes = {p: sha256_file(p) for p in old_paths}
    manifest = prepare(output)
    actual = pd.read_csv(output / "epit_processed.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(actual[source.columns], source)
    assert actual.row_id.tolist() == source.row_id.tolist()
    assert sha256_file(output / "split_assignments.csv") == sha256_file(ENVIRONMENT_V1 / "split_assignments.csv")
    assert all(sha256_file(p) == h for p, h in old_hashes.items())
    registry = json.loads((output / "feature_registry.json").read_text())
    features = pd.read_csv(output / "epit_features.csv")
    assert len(features) == 760
    assert features.columns.tolist() == ["row_id", *registry["feature_columns"]]
    assert len(registry["feature_columns"]) == 69
    assert len(registry["environment_history_columns"]) == 45
    assert registry["schema_version"] == manifest["schema_version"] == "epit_environment_v2"
    assert manifest["counts"]["development_rows"] == 608
    assert manifest["counts"]["final_test_rows"] == 152
    assert all(c not in features for c in ["Comments", "Reference", "Heat treatment", "Epit, mV (SCE) Avg.", "CP_ion_other", "Prep_redox", "Prep_Ra_micron", "Prep_pH"])
    for name, digest in manifest["output_sha256"].items():
        assert sha256_file(output / name) == digest
    # The fields added for EPIT must retain the existing Soccol feature types
    # and units, even when their coverage or source descriptions differ.
    from corrosion_datasets.datasets.soccol_pitting_potential.prepare_soccol_environment import column_definition as soccol_definition
    for name in NEW_COLUMNS:
        for key in ["dtype", "unit"]:
            assert column_definition(name)[key] == soccol_definition(name)[key]
    with pytest.raises(FileExistsError):
        prepare(output)
