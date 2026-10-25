from __future__ import annotations

import json

import pandas as pd
import pytest

from corrosion_datasets.datasets.review_epit_soccol_environment import (
    EXPORTS, HERE, prepare, review, sha256_file,
)


def read(dataset):
    directory, version, _, reference, _ = EXPORTS[dataset]
    path = HERE / directory / "processed" / version
    return path, pd.read_csv(path / reference, dtype=str, keep_default_na=False)


def test_chloride_contradiction_is_missing_without_rewriting_either_source_report():
    _, source = read("epit")
    result, changes, log = review("epit", source)
    affected = result.row_id.eq("epit_r000148")
    assert len(changes) == 1
    assert result.loc[affected, "CP_Cl"].eq("").all()
    assert pd.to_numeric(result.loc[affected, "[Cl-] M"]).eq(0).all()
    assert pd.to_numeric(result.loc[affected, "HCl_mol_L"]).eq(1).all()
    assert log.status.eq("conflict").all()
    pd.testing.assert_frame_equal(result.drop(columns="CP_Cl"), source.drop(columns="CP_Cl"))
    pd.testing.assert_series_equal(result.loc[~affected, "CP_Cl"], source.loc[~affected, "CP_Cl"])


def test_air_only_timing_is_not_solution_treatment_but_separate_conditioning_survives():
    _, source = read("soccol")
    result, changes, _ = review("soccol", source)
    timing = changes.loc[changes.column.eq("Prep_time"), "row_id"]
    assert len(timing) == 217
    assert result.loc[result.row_id.isin(timing), "Prep_time"].eq("").all()
    assert source.loc[source.row_id.isin(timing), "raw_Prep_medium"].eq("air").all()
    separate = source.raw_Prep_medium.eq("air") & source.raw_Prep_redox.ne("") & source.raw_Prep_time.ne("")
    assert separate.sum() == 36
    pd.testing.assert_series_equal(result.loc[separate, "Prep_time"], source.loc[separate, "Prep_time"])
    pd.testing.assert_series_equal(result.CP_time, source.CP_time)
    for before, after in zip(source.surface_preparation_method, result.surface_preparation_method):
        assert set(before.split(" + ")) == set(after.split(" + "))
        assert after == " + ".join(sorted(set(after.split(" + "))))
    # Target and alloy values cannot influence which environment cells change.
    altered = source.copy()
    for column in ["E_pit", "E_corr", "event", "Cr", "Fe"]:
        altered[column] = "unrelated outcome"
    assert review("soccol", altered)[1].equals(changes)


@pytest.mark.parametrize("dataset, expected_changes", [("epit", 1), ("soccol", 249)])
def test_export_preserves_originals_splits_and_every_unlisted_cell(tmp_path, dataset, expected_changes):
    path, source = read(dataset)
    _, _, version, reference, feature_file = EXPORTS[dataset]
    original_hashes = {p: sha256_file(p) for p in path.iterdir() if p.is_file()}
    output = tmp_path / version
    manifest = prepare(dataset, output)
    registry = json.loads((output / "feature_registry.json").read_text())
    result = pd.read_csv(output / reference, dtype=str, keep_default_na=False)
    features = pd.read_csv(output / feature_file, dtype=str, keep_default_na=False)
    changes = pd.read_csv(output / "review_changes.csv", dtype=str, keep_default_na=False)
    assert len(changes) == expected_changes
    assert source.shape == result.shape
    assert source.row_id.equals(result.row_id)
    actual_changes = {(source.at[i, "row_id"], column) for i, column_index in zip(*(source != result).to_numpy().nonzero())
                      for column in [source.columns[column_index]]}
    assert actual_changes == set(zip(changes.row_id, changes.column))
    pd.testing.assert_frame_equal(features, result[["row_id", *registry["feature_columns"]]])
    assert all(sha256_file(p) == h for p, h in original_hashes.items())
    assert sha256_file(output / "split_assignments.csv") == sha256_file(path / "split_assignments.csv")
    assert all(sha256_file(output / name) == h for name, h in manifest["output_sha256"].items())
    assert "CP_ion_other" not in features
    assert registry["schema_version"] == manifest["schema_version"] == version
    coverage = pd.read_csv(output / "environment_column_summary.csv")
    for _, item in coverage.iterrows():
        assert item.populated_rows == result[item.column].ne("").sum()
    # Existing sulfate/carbonate/area contradictions must not be restored.
    if dataset == "soccol":
        for column in ["CP_SO4", "CP_CO3", "CP_OH", "Test_area_cm2"]:
            pd.testing.assert_series_equal(result[column], source[column])
        assert result.Prep_time.ne("").sum() == 789
    else:
        assert result.CP_Cl.ne("").sum() == 758
    with pytest.raises(FileExistsError):
        prepare(dataset, output)
