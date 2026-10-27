from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from corrosion_datasets.datasets.soccol_pitting_potential.prepare_pipeline_assets import (
    ASSET_DIR, build_reviewed_feature_assets,
)
from corrosion_datasets.datasets.soccol_pitting_potential import prepare_pipeline_assets as builder
from scripts.soccol_pipeline.data import load_soccol_dataset, select_accepted_development
from tabicl.prior.soccol_feature_generator import sample_soccol_feature_rows
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile
from tabicl.prior.soccol_reviewed_schema import (
    SOCCOL_REVIEWED_FEATURE_PROFILE as NAME,
    SOCCOL_REVIEWED_FEATURE_COLUMNS as COLUMNS,
    SOCCOL_REVIEWED_COMPOSITION_COLUMNS as COMPOSITION,
    SOCCOL_REVIEWED_CONTEXT_COLUMNS as CONTEXT,
    SOCCOL_REVIEWED_CATEGORICAL_COLUMNS as CATEGORICAL,
)


@pytest.fixture(scope="module")
def reviewed():
    return load_soccol_dataset()


@pytest.fixture(scope="module")
def profile():
    return load_soccol_feature_profile(NAME)


def test_default_profile_and_sampler_use_reviewed_development():
    profile = load_soccol_feature_profile()
    assert profile.name == NAME
    assert profile.feature_values.shape == (2950, 58)
    batch = sample_soccol_feature_rows(128, random_state=42)
    assert batch.profile_name == NAME
    assert batch.features.shape == (128, 58)
    assert np.isnan(batch.features[:, 24:]).any()
    assert np.isfinite(batch.features[:, :24]).all()
    assert set(batch.compositions.template_ids).issubset(profile.row_ids)
    np.testing.assert_array_equal(batch.features[:, 24:], profile.feature_values[batch.context_indices, 24:])


def test_missing_reviewed_assets_never_fall_back_to_legacy(tmp_path):
    for suffix in ("csv", "json"):
        filename = f"soccol_pitting_features_v1.{suffix}"
        (tmp_path / filename).write_bytes((ASSET_DIR / filename).read_bytes())
    assert load_soccol_feature_profile("soccol_pitting_features_v1", asset_dir=str(tmp_path)).n_rows == 4027
    with pytest.raises(FileNotFoundError, match=NAME):
        load_soccol_feature_profile(asset_dir=str(tmp_path))


def test_builder_defaults_to_reviewed_bank_and_keeps_frozen_bytes(tmp_path, capsys):
    builder.main(["--output-dir", str(tmp_path)])
    assert {p.name for p in tmp_path.iterdir()} == {f"{NAME}.csv", f"{NAME}.json"}
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (ASSET_DIR / path.name).read_bytes()
    builder.main(["--output-dir", str(tmp_path), "--check"])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.main(["--output-dir", str(tmp_path)])
    capsys.readouterr()
    builder.main(["--inspect-inputs"])
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["sampling_source_rows"] == 2950
    assert inspected["final_query_rows_used"] == 0


def test_builder_legacy_route_is_explicit():
    assert builder.parse_args([]).reviewed_data
    assert builder.parse_args(["--reviewed-data"]).reviewed_data
    assert not builder.parse_args(["--legacy-data"]).reviewed_data
    with pytest.raises(SystemExit):
        builder.parse_args(["--reviewed-data", "--legacy-data"])
    with pytest.raises(SystemExit):
        builder.parse_args(["--legacy-data", "--check"])


def test_frozen_assets_rebuild_identically_from_reviewed_development(reviewed):
    for filename, content in build_reviewed_feature_assets(reviewed).items():
        assert (ASSET_DIR / filename).read_bytes() == content


def test_bank_keeps_all_accepted_observations_and_distinct_composition_templates(reviewed, profile):
    selected = select_accepted_development(reviewed)
    assert profile.feature_values.shape == (2950, 58)
    assert profile.row_ids == tuple(selected.rows.row_id)
    np.testing.assert_array_equal(profile.source_task_row_indices, selected.task_row_indices)
    np.testing.assert_array_equal(profile.feature_values[:, :24], selected.features[list(COMPOSITION)])
    assert profile.metadata["sampling"]["distinct_compositions"] == 349
    first = np.flatnonzero(~selected.features[list(COMPOSITION)].duplicated())
    np.testing.assert_array_equal(profile.composition_template_indices, first)
    assert profile.metadata["sampling"]["composition_template_source_task_row_indices"] == selected.task_row_indices[first].tolist()
    # Includes accepted materials beyond EPIT's restricted families, at source frequency.
    assert (profile.feature_values[:, 0] == 0).sum() == selected.features.Fe.eq(0).sum() > 0
    assert np.any(np.abs(profile.feature_values[:, :24].sum(axis=1) - 100) > 0.001)
    assert profile.metadata["target_columns_used"] == []
    assert not {"E_pit", "E_corr", "CP_ion_other", "source"}.intersection(COLUMNS)
    final = set(reviewed.assignments.loc[reviewed.rows.frozen_final_query, "task_row_index"])
    withheld = set(reviewed.assignments.loc[
        reviewed.rows.outer_split.eq("development") & ~reviewed.rows.use_as_development_context,
        "task_row_index",
    ])
    assert not set(profile.source_task_row_indices) & (final | withheld)


def test_uniform_templates_remove_repeated_observation_bias_but_keep_context_weights(profile):
    old = load_soccol_feature_profile("soccol_pitting_features_v2")
    np.testing.assert_array_equal(old.feature_values, profile.feature_values)
    np.testing.assert_array_equal(old.latent_feature_values, profile.latent_feature_values)
    assert old.row_ids == profile.row_ids
    assert old.metadata["development_selection"] == profile.metadata["development_selection"]
    _, groups, counts = np.unique(old.feature_values[:, :24], axis=0, return_inverse=True, return_counts=True)
    frequent = np.argsort(counts)[-20:]
    old_batch = sample_soccol_feature_rows(50000, profile=old, perturb_strength=0., random_state=42)
    new_batch = sample_soccol_feature_rows(50000, profile=profile, perturb_strength=0., random_state=42)
    assert len(old.composition_template_indices) == 2950
    assert len(profile.composition_template_indices) == 349
    assert np.isin(groups[old_batch.compositions.template_indices], frequent).mean() > .50
    assert np.isin(groups[new_batch.compositions.template_indices], frequent).mean() == pytest.approx(20/349, abs=.005)
    assert len(np.unique(new_batch.compositions.template_indices)) == 349
    # All 2,950 observations remain available for independent joint contexts.
    assert len(np.unique(new_batch.context_indices)) == 2950
    np.testing.assert_array_equal(new_batch.features[:, 24:], profile.feature_values[new_batch.context_indices, 24:])
    assert not profile.composition_template_indices.flags.writeable


def test_statistics_and_encodings_match_only_accepted_development(reviewed, profile):
    source = select_accepted_development(reviewed).features
    prep = profile.metadata["preprocessing"]
    for column in COLUMNS:
        actual = profile.feature_values[:, COLUMNS.index(column)]
        expected = source[column]
        if column in CATEGORICAL:
            categories = sorted(expected.dropna().astype(str).unique())
            mapping = {value: i for i, value in enumerate(categories)}
            assert prep["category_mappings"][column] == mapping
            expected = expected.map(mapping).astype(float)
        np.testing.assert_array_equal(actual, expected.to_numpy(dtype=float))
        if column in CONTEXT:
            observed = expected.dropna()
            fill = (float(observed.mode().iloc[0]) if column in CATEGORICAL else float(observed.mean())) if len(observed) else 0.0
            assert prep["latent_fill_values"][column] == pytest.approx(fill)
    assert np.isnan(profile.feature_values).any()
    assert np.isfinite(profile.latent_feature_values).all()


@pytest.mark.parametrize("partition", ["final_test", "withheld_development", "all_targets"])
def test_excluded_features_and_all_targets_cannot_change_bank_or_statistics(reviewed, partition):
    rows, features = reviewed.rows.copy(), reviewed.features.copy()
    targets = reviewed.targets.copy()
    if partition == "all_targets":
        rows["E_pit"] = -123456.0
        rows["E_corr"] = 123456.0
        targets[:] = -123456.0
    else:
        mask = rows.frozen_final_query if partition == "final_test" else rows.outer_split.eq("development") & ~rows.use_as_development_context
        for column in COLUMNS:
            value = "excluded-only-category" if column in CATEGORICAL else -123456.0
            if column in CATEGORICAL:
                rows[column] = rows[column].astype(object)
                features[column] = features[column].astype(object)
            rows.loc[mask, column] = value
            features.loc[mask, column] = value
    altered = replace(reviewed, rows=rows, features=features, targets=targets)
    assert build_reviewed_feature_assets(altered) == build_reviewed_feature_assets(reviewed)


@pytest.mark.parametrize("strength", [0.0, 0.3])
def test_sampling_uses_new_bank_and_keeps_joint_context_and_zeros(profile, strength):
    batch = sample_soccol_feature_rows(2048, profile=profile, perturb_strength=strength, random_state=42)
    assert batch.profile_name == NAME
    assert batch.features.shape == (2048, 58)
    np.testing.assert_array_equal(batch.features[:, 24:], profile.feature_values[batch.context_indices, 24:])
    np.testing.assert_array_equal(batch.latent_features[:, 24:], profile.latent_feature_values[batch.context_indices, 24:])
    original = profile.feature_values[batch.compositions.template_indices, :24]
    sampled = batch.features[:, :24]
    np.testing.assert_array_equal(sampled == 0, original == 0)
    np.testing.assert_allclose(sampled.sum(axis=1), 100)
    if strength == 0:
        np.testing.assert_allclose(sampled, original / original.sum(axis=1, keepdims=True) * 100)
    assert batch.compositions.template_ids == tuple(profile.row_ids[i] for i in batch.compositions.template_indices)
    assert np.isfinite(batch.latent_features).all()
    assert np.isnan(batch.features[:, 24:]).any()
    assert np.any(batch.context_indices != batch.compositions.template_indices)
    assert not profile.feature_values.flags.writeable and not batch.features.flags.writeable
    assert not profile.latent_feature_values.flags.writeable and not batch.latent_features.flags.writeable


@pytest.mark.parametrize("corruption", ["scope", "source_index", "statistic", "category", "missing_count", "template_indices", "sampling_policy"])
def test_loader_rejects_inconsistent_provenance_or_statistics(tmp_path, corruption):
    (tmp_path / f"{NAME}.csv").write_bytes((ASSET_DIR / f"{NAME}.csv").read_bytes())
    metadata = json.loads((ASSET_DIR / f"{NAME}.json").read_text())
    if corruption == "scope":
        metadata["sampling_scope"]["final_query_rows_used"] = 1
    elif corruption == "source_index":
        metadata["development_selection"]["source_task_row_indices"][0] = 20
    elif corruption == "statistic":
        metadata["preprocessing"]["latent_fill_values"]["CP_temp"] += 10
    elif corruption == "category":
        metadata["preprocessing"]["category_mappings"]["CP_aeration"]["final-only"] = 99
    elif corruption == "missing_count":
        metadata["missing_counts"]["CP_SO4"] = 0
    elif corruption == "template_indices":
        metadata["sampling"]["composition_template_source_task_row_indices"][0] = -1
    else:
        metadata["sampling"]["composition_sampling_policy"] = "uniform_observations"
    (tmp_path / f"{NAME}.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_soccol_feature_profile(NAME, asset_dir=str(tmp_path))


def test_historical_profile_remains_separate_and_unchanged():
    old = load_soccol_feature_profile("soccol_pitting_features_v1")
    assert old.feature_values.shape == (4027, 37)
    assert np.isfinite(old.feature_values).all()
    batch = sample_soccol_feature_rows(100, profile=old, random_state=42)
    assert batch.profile_name == "soccol_pitting_features_v1"
    assert batch.features.shape == (100, 37)
    np.testing.assert_array_equal(batch.features, batch.latent_features)
