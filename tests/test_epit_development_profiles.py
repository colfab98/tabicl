from dataclasses import replace

import numpy as np

from scripts.build_epit_development_profiles import build_assets
from scripts.build_epit_feature_profile import ENVIRONMENT_SOURCE_COLUMNS, METHOD_COLUMN, _normalized_method
from scripts.epit_pipeline.composition_data import load_reviewed_compositions
from scripts.eval_corrosion_datasets import eval_to_float
from tabicl.prior.epit_feature_generator import sample_epit_feature_rows
from tabicl.prior.epit_feature_profile import load_epit_feature_profile
from tabicl.prior.epit_schema import EPIT_COMPOSITION_ELEMENTS


def test_sampling_sources_and_statistics_exclude_every_final_query():
    reviewed = load_reviewed_compositions()
    assignments = reviewed.assignments
    development = assignments.loc[assignments.outer_split.eq("development")]
    final = assignments.loc[assignments.outer_split.eq("final_test")]
    profile = load_epit_feature_profile("epit_fe_nicrmo_features_v3")
    composition = profile.composition_profile
    sources = composition.metadata["template_source_row_ids"]
    all_composition_sources = {row_id for ids in sources.values() for row_id in ids}
    assert all_composition_sources == set(development.row_id)
    assert not all_composition_sources.intersection(final.row_id)

    active_sources = {
        row_id
        for index in profile.composition_template_indices
        for row_id in sources[composition.template_ids[index]]
    }
    expected = reviewed.rows.loc[
        reviewed.rows.row_id.isin(development.row_id)
        & reviewed.rows["Material class"].isin(["Fe Alloy", "NiCrMo Alloy"])
    ]
    assert len(expected) == 452
    assert active_sources == set(expected.row_id)
    assert set(profile.source_task_row_indices) == set(expected.task_row_index)
    assert not set(profile.source_task_row_indices).intersection(final.task_row_index)

    preprocessing = profile.metadata["evaluator_preprocessing"]
    assert set(preprocessing["source_row_ids"]) == set(development.row_id)
    data = reviewed.rows.loc[reviewed.rows.row_id.isin(development.row_id)]
    for column in ENVIRONMENT_SOURCE_COLUMNS:
        values = np.asarray([eval_to_float(v, column=column, group="environment") for v in data[column]])
        assert np.isclose(preprocessing["numeric_imputation_means"][column], np.nanmean(values))
    assert set(profile.test_method_categories) == {_normalized_method(v) for v in data[METHOD_COLUMN]}

    batch = sample_epit_feature_rows(512, profile=profile, random_state=42)
    assert set(batch.compositions.family_names) == {"fe_alloy", "nicrmo_alloy"}
    assert all(set(sources[template]).issubset(active_sources) for template in batch.compositions.template_ids)
    assert set(profile.source_task_row_indices[batch.environment_indices]).issubset(set(expected.task_row_index))


def test_final_query_values_cannot_change_any_sampling_asset():
    reviewed = load_reviewed_compositions()
    before = build_assets(reviewed)
    changed = reviewed.rows.copy(deep=True)
    final_ids = reviewed.assignments.loc[reviewed.assignments.outer_split.eq("final_test"), "row_id"]
    final = changed.row_id.isin(final_ids)
    changed.loc[final, list(EPIT_COMPOSITION_ELEMENTS)] = 12345.0
    changed.loc[final, list(ENVIRONMENT_SOURCE_COLUMNS)] = 99999.0
    changed.loc[final, METHOD_COLUMN] = "query_only_method_must_never_enter_the_bank"
    changed.loc[final, "Material class"] = "query_only_family"
    # Targets are unused for sampling in both partitions.
    changed["Epit, mV (SCE) Avg."] = -987654.0
    assert build_assets(replace(reviewed, rows=changed)) == before
