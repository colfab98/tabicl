from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from corrosion_datasets.datasets.soccol_pitting_potential import prepare_soccol_composition as m


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    output = tmp_path_factory.mktemp("soccol_composition") / "export"
    manifest = m.prepare(output)
    return output, manifest


def test_reported_amounts_survive_and_recoveries_precede_fe_balance(export):
    output, _ = export
    frame = pd.read_csv(output / "soccol_processed.csv").set_index("row_id")
    raw = pd.read_csv(output / "source_rows.csv").set_index("row_id")
    for element in m.NATIVE_ELEMENTS:
        present = raw[element].notna()
        if element == "Ti":
            present &= raw.source.ne("2000Russell")
        np.testing.assert_allclose(frame.loc[present, element], raw.loc[present, element], rtol=0, atol=1e-12)
    expected = {
        "soccol_r000112": {"Ti": 1.86, "Fe": 77.612},
        "soccol_r000117": {"Ti": 1.75, "Fe": 75.47},
        "soccol_r000149": {"Nb": 1, "Fe": 67.93},
        "soccol_r000152": {"Nb": 1, "Cu": 3.5, "Fe": 43.44},
    }
    for row_id, values in expected.items():
        for element, amount in values.items():
            assert frame.loc[row_id, element] == pytest.approx(amount)
    # The EPIT numeric Ti of 1.88 did not override the explicit nominal 1.86.
    log = pd.read_csv(output / "composition_processing_log.csv")
    ti = log[log.row_id.eq("soccol_r000112") & log.element.eq("Ti")].iloc[0]
    assert ti.action == "recovered_nominal_from_designation"
    assert "1.88" in ti.evidence
    ahn = frame[frame.source.eq("1998Ahn")]
    assert len(ahn) == 8 and ahn.W.gt(0).all() and ahn.Nb.gt(0).all()
    assert frame.loc[frame.source.eq("1983Bui"), "W"].gt(0).sum() == 12
    ferralium = frame.source.eq("1974Machin") & frame.Cr.eq(25) & frame.Ni.eq(5.5)
    assert ferralium.sum() == 5 and frame.loc[ferralium, "Cu"].eq(3).all()
    russell = frame.source.eq("2000Russell")
    assert russell.sum() == 11 and frame.loc[russell, "Ti"].eq(0).all()
    accepted = frame[frame.composition_status.eq("accepted")]
    assert accepted[m.ELEMENTS].notna().all().all()
    assert accepted[m.ELEMENTS].sum(axis=1).between(99, 101).all()
    # Complete, explicitly non-Fe alloys were not forced to have an Fe balance.
    non_fe = frame.legacy_Fe_balance_basis.str.startswith("explicit_")
    assert non_fe.sum() == 61 and frame.loc[non_fe, "Fe"].eq(0).all()


def test_incomplete_rows_are_withheld_without_changing_frozen_queries(export):
    output, manifest = export
    audit = pd.read_csv(output / "composition_row_audit.csv").set_index("row_id")
    frame = pd.read_csv(output / "soccol_processed.csv").set_index("row_id")
    unresolved = audit.composition_status.eq("unresolved")
    assert unresolved.sum() == 282
    assert frame.loc[unresolved, "Fe"].isna().all()
    for source, element in [("1969Boehni", "Re"), ("2018Shi", "Y"), ("1994Malik", "Cu")]:
        rows = frame.source.eq(source)
        assert unresolved[rows].all() and frame.loc[rows, element].isna().all()
    combined = frame.source.eq("1995Malik") & frame.alloy_designation.str.contains("Nb \\+ Ta", na=False)
    assert combined.sum() == 10 and frame.loc[combined, ["Nb", "Ta"]].isna().all().all()
    empty = frame.legacy_composition_reported_count.eq(0)
    sulfur_only = frame.source.eq("2014Zhou")
    assert empty.sum() == 95 and sulfur_only.sum() == 87
    assert unresolved[empty | sulfur_only].all()
    assert audit.loc[empty | sulfur_only, "outer_split"].eq("development").all()
    dev = pd.read_csv(output / "soccol_development_features.csv")
    query = pd.read_csv(output / "soccol_frozen_final_query_features.csv")
    frozen = pd.read_csv(m.SPLIT_MANIFEST.parent / "split_assignments.csv")
    query_ids = frozen.loc[frozen.outer_split.eq("final_test"), "raw_workbook_row"].map(lambda n: f"soccol_r{n:06d}")
    assert query.row_id.tolist() == query_ids.tolist()
    assert len(query) == 805 and len(dev) == 2950
    assert dev[m.ELEMENTS].notna().all().all()
    assert not set(dev.row_id) & set(query.row_id)
    pending_final = audit[audit.outer_split.eq("final_test") & unresolved]
    assert len(pending_final) == 10 and pending_final.source.eq("1994Malik").all()
    assert manifest["counts"]["withheld_development_rows"] == 272
    assert not manifest["counts"]["final_evaluation_ready"]
    assert m.sha256_file(output / "split_assignments.csv") == m.sha256_file(m.SPLIT_MANIFEST.parent / "split_assignments.csv")


def test_recovery_ignores_targets_and_refuses_conflicting_donor_amounts():
    rules = json.loads(m.RULES_PATH.read_text())
    rules["same_paper_recoveries"] = [r for r in rules["same_paper_recoveries"] if r["source"] == "1998Ahn"]
    evidence, dois = m.load_evidence(rules)
    raw = pd.read_csv(m.INPUT_DIR / "source_rows.csv")
    legacy = pd.read_csv(m.INPUT_DIR / "soccol_processed.csv", dtype=str, keep_default_na=False)
    selected = raw.source.eq("1998Ahn")
    raw = raw.loc[selected].reset_index(drop=True)
    legacy = legacy.loc[selected].reset_index(drop=True)
    original = m.transform(raw, legacy, rules, evidence, dois)
    changed = raw.copy()
    for column in ["E_pit", "E_corr", "event", "CP_temp", "label"]:
        changed[column] = "unusable outcome or environment text"
    reprocessed = m.transform(changed, legacy, rules, list(reversed(evidence)), dois)
    for before, after in zip(original, reprocessed):
        pd.testing.assert_frame_equal(before, after)
    # An otherwise matching donor with a different W amount cannot be
    # resolved using target similarity, donor order or zero filling.
    donors = copy.deepcopy(evidence)
    conflict = next(d for d in donors if d["W"] == pytest.approx(8.17)).copy()
    conflict["W"] = 9.17
    donors.append(conflict)
    composition, _, audit = m.transform(raw, legacy, rules, donors, dois)
    ambiguous = raw.Cr.eq(28.78)
    assert ambiguous.sum() == 1
    assert audit.loc[ambiguous, "composition_status"].eq("unresolved").all()
    assert composition.loc[ambiguous, ["W", "Fe"]].isna().all().all()


def test_export_preserves_environment_targets_and_input_files(export):
    output, manifest = export
    frame = pd.read_csv(output / "soccol_processed.csv", dtype=str, keep_default_na=False)
    legacy = pd.read_csv(m.INPUT_DIR / "soccol_processed.csv", dtype=str, keep_default_na=False)
    retained = legacy.columns.difference([*m.ELEMENTS, *m.LEGACY_AUDIT])
    pd.testing.assert_frame_equal(frame[retained], legacy[retained])
    assert frame.columns.is_unique
    for name in ["source_rows.csv", "soccol_environment_features.csv", "environment_processing_log.csv",
                 "environment_review_notes.csv", "environment_column_summary.csv", "review_changes.csv"]:
        assert m.sha256_file(output / name) == m.sha256_file(m.INPUT_DIR / name)
    for name, digest in manifest["input_sha256"].items():
        assert m.sha256_file(m.Path(name)) == digest
    for name, digest in manifest["output_sha256"].items():
        assert m.sha256_file(output / name) == digest
    registry = json.loads((output / "feature_registry.json").read_text())
    features = pd.read_csv(output / "soccol_features.csv")
    assert features.columns.tolist() == ["row_id", *registry["feature_columns"]]
    assert registry["composition_columns"] == m.ELEMENTS
    assert len(registry["feature_columns"]) == 57
    assert not registry["final_evaluation_ready"]
    for column in ["E_pit", "E_corr", "event", "composition_status", "CP_ion_other", "source", "outer_split"]:
        assert column not in features
    with pytest.raises(FileExistsError):
        m.prepare(output)
