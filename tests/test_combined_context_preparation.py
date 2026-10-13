from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corrosion_datasets.datasets.combined_epit_soccol_mpea.prepare_shared_tables import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_tables,
    feature_registry,
    missingness_profile,
)
from corrosion_datasets.datasets.combined_epit_soccol_mpea.schema_alignment import (
    align_feature_frame,
)


def test_combined_context_and_query_boundaries() -> None:
    context, queries, scoring = build_tables()

    assert len(context) == 4787
    assert context["source_dataset"].value_counts().to_dict() == {
        "Soccol": 4027,
        "EPIT": 760,
    }
    assert context["reference_type"].value_counts().to_dict() == {
        "SCE": 1325,
        "AgAgCl": 112,
    }
    assert context["reference_type"].isna().sum() == 3350
    unresolved = context["reference_type"].isna()
    assert context.loc[unresolved, "reference_type_evidence"].eq(
        "source_reference_not_verified"
    ).all()

    assert len(queries) == len(scoring) == 267
    assert TARGET_COLUMN not in queries.columns
    assert queries["row_id"].tolist() == scoring["row_id"].tolist()
    assert queries["reference_type"].eq("SCE").all()
    assert np.isfinite(scoring[TARGET_COLUMN]).all()


def test_shared_schema_preserves_missingness_and_measured_zero() -> None:
    context, queries, _ = build_tables()

    assert list(context.loc[:, FEATURE_COLUMNS].columns) == list(FEATURE_COLUMNS)
    assert list(queries.loc[:, FEATURE_COLUMNS].columns) == list(FEATURE_COLUMNS)

    soccol = context.loc[context["source_dataset"].eq("Soccol")]
    assert soccol["W"].isna().all()
    assert soccol["N"].isna().any()
    assert (soccol["N"] == 0).any()

    epit = context.loc[context["source_dataset"].eq("EPIT")]
    assert epit["reference_type"].eq("SCE").all()
    assert epit["scan_rate"].notna().any()

    assert queries["Prep_pH"].isna().all()


def test_missingness_profile_uses_context_only_and_preserves_joint_patterns() -> None:
    context, _, _ = build_tables()
    profile = missingness_profile(context)

    assert profile["composition_feature_count"] == 24
    assert len(profile["feature_columns_in_order"]) == 50
    assert "MPEA queries and scoring targets excluded" in profile["source"]
    assert set(profile["patterns_by_source"]) == {"EPIT", "Soccol"}
    assert {
        source: source_profile["row_count"]
        for source, source_profile in profile["patterns_by_source"].items()
    } == {"EPIT": 760, "Soccol": 4027}
    assert {
        source: len(source_profile["patterns"])
        for source, source_profile in profile["patterns_by_source"].items()
    } == {"EPIT": 9, "Soccol": 265}
    for source_profile in profile["patterns_by_source"].values():
        assert sum(pattern["count"] for pattern in source_profile["patterns"]) == source_profile["row_count"]
        assert all(len(pattern["missing_mask"]) == 50 for pattern in source_profile["patterns"])


def test_name_alignment_reorders_aliases_and_fills_absent_features() -> None:
    raw = pd.DataFrame(
        {
            "Reference electrode": ["SCE", "AgAgCl"],
            "Test Temp. oC": [25.0, 50.0],
            "Composition, wt.% Cr": [18.0, 20.0],
        },
        index=[10, 20],
    )

    aligned, report = align_feature_frame(raw, registry=feature_registry())

    assert list(aligned.columns) == list(FEATURE_COLUMNS)
    assert aligned.index.tolist() == [10, 20]
    assert aligned["Cr"].tolist() == [18.0, 20.0]
    assert aligned["CP_temp"].tolist() == [25.0, 50.0]
    assert aligned["reference_type"].tolist() == ["SCE", "AgAgCl"]
    assert aligned["Ni"].isna().all()
    assert report.aliases_applied == {
        "Reference electrode": "reference_type",
        "Test Temp. oC": "CP_temp",
        "Composition, wt.% Cr": "Cr",
    }
    assert "Ni" in report.missing_columns_filled_with_nan


def test_name_alignment_rejects_unknown_and_ambiguous_columns() -> None:
    registry = feature_registry()
    with pytest.raises(ValueError, match="Unrecognized columns"):
        align_feature_frame(pd.DataFrame({"Cr": [18.0], "mystery": [1.0]}), registry=registry)

    aligned, report = align_feature_frame(
        pd.DataFrame({"Cr": [18.0], "row_id": ["fourth-1"]}),
        registry=registry,
        unknown_columns="ignore",
    )
    assert aligned["Cr"].iloc[0] == 18.0
    assert report.ignored_columns == ("row_id",)

    with pytest.raises(ValueError, match="both map to canonical feature 'Cr'"):
        align_feature_frame(
            pd.DataFrame({"Cr": [18.0], "Composition, wt.% Cr": [19.0]}),
            registry=registry,
        )
