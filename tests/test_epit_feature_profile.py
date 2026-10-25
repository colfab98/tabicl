import csv
from importlib import resources

import numpy as np
import pytest

from tabicl.prior.epit_feature_profile import (
    EPIT_FEATURE_FAMILIES,
    EPIT_FEATURE_FAMILY_ENVIRONMENT_COUNTS,
    EPIT_FEATURE_FAMILY_TEMPLATE_COUNTS,
    EPIT_FEATURE_PROFILE,
    load_epit_feature_profile,
)
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_ENVIRONMENT_COLUMNS


def test_epit_fe_nicrmo_feature_profile_loads_and_reuses_compositions():
    profile = load_epit_feature_profile()

    assert profile.name == EPIT_FEATURE_PROFILE
    assert profile.families == EPIT_FEATURE_FAMILIES
    assert profile.environment_values.shape == (452, 3)
    assert profile.environment_missing_mask.shape == (452, 3)
    assert profile.environment_imputed_values.shape == (452, 3)
    assert np.isfinite(profile.environment_imputed_values).all()
    assert tuple(np.bincount(profile.environment_family_indices)) == EPIT_FEATURE_FAMILY_ENVIRONMENT_COUNTS
    assert tuple(np.bincount(profile.composition_family_indices)) == EPIT_FEATURE_FAMILY_TEMPLATE_COUNTS
    assert profile.n_composition_templates == 255
    assert profile.composition_profile.template_values.shape == (323, 24)
    assert profile.composition_profile.observed_template_values.shape == (323, 24)


def test_historical_feature_profile_still_uses_its_old_composition_bank():
    profile = load_epit_feature_profile("epit_fe_nicrmo_features_v1")
    assert profile.composition_profile.name == "epit_dataset_v2"
    assert profile.n_composition_templates == 315
    reviewed_full = load_epit_feature_profile("epit_fe_nicrmo_features_v2")
    assert reviewed_full.composition_profile.name == "epit_dataset_v3"
    assert reviewed_full.n_composition_templates == 313
    assert np.array_equal(profile.environment_imputed_values, reviewed_full.environment_imputed_values)
    assert np.array_equal(profile.test_method_codes, reviewed_full.test_method_codes)


def test_epit_feature_profile_preserves_missingness_and_evaluator_categories():
    profile = load_epit_feature_profile()

    assert tuple(profile.environment_missing_mask.sum(axis=0)) == (37, 1, 38)
    assert len(profile.test_method_categories) == 50
    assert len(set(profile.test_methods)) == 39
    assert profile.metadata["evaluator_preprocessing"]["profile_category_count"] == 39
    expected_codes = np.asarray(
        [
            profile.test_method_mapping[
                "<NA>" if method in {"", "NA", "N/A", "nan", "None"} else method
            ]
            for method in profile.test_methods
        ],
        dtype=np.int64,
    )
    assert np.array_equal(profile.test_method_codes, expected_codes)


def test_epit_feature_csv_contains_features_and_provenance_only():
    csv_path = resources.files("tabicl.prior.assets") / f"{EPIT_FEATURE_PROFILE}.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        fields = csv.DictReader(handle).fieldnames

    assert fields == [
        "environment_id",
        "task_row_index",
        "source_no",
        "family",
        "temperature_celsius",
        "chloride_molar",
        "pH",
        "test_method",
        "test_method_family",
        "test_method_category",
        *EPIT_EXPANDED_ENVIRONMENT_COLUMNS,
    ]
    assert not any("target" in field.lower() for field in fields)
    assert not any("split" in field.lower() or "fold" in field.lower() for field in fields)
    assert "reference" not in fields


def test_epit_feature_profile_rejects_modified_csv(tmp_path):
    asset_root = resources.files("tabicl.prior.assets")
    for filename in (
        "epit_dataset_v1.csv",
        "epit_dataset_v1.json",
        "epit_dataset_v2.json",
        "epit_dataset_v3.csv",
        "epit_dataset_v3.json",
        "epit_dataset_v4.csv",
        "epit_dataset_v4.json",
        f"{EPIT_FEATURE_PROFILE}.csv",
        f"{EPIT_FEATURE_PROFILE}.json",
    ):
        (tmp_path / filename).write_bytes((asset_root / filename).read_bytes())
    csv_path = tmp_path / f"{EPIT_FEATURE_PROFILE}.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_epit_feature_profile(asset_dir=tmp_path)


def test_epit_feature_profile_rejects_modified_composition_metadata(tmp_path):
    asset_root = resources.files("tabicl.prior.assets")
    for filename in (
        "epit_dataset_v1.csv",
        "epit_dataset_v1.json",
        "epit_dataset_v2.json",
        "epit_dataset_v3.csv",
        "epit_dataset_v3.json",
        "epit_dataset_v4.csv",
        "epit_dataset_v4.json",
        f"{EPIT_FEATURE_PROFILE}.csv",
        f"{EPIT_FEATURE_PROFILE}.json",
    ):
        (tmp_path / filename).write_bytes((asset_root / filename).read_bytes())
    metadata_path = tmp_path / "epit_dataset_v4.json"
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="metadata checksum"):
        load_epit_feature_profile(asset_dir=tmp_path)
