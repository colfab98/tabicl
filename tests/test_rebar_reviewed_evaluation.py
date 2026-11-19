"""Reviewed selection follows Soccol CV while preserving every outer query."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.rebar_pipeline import reviewed_evaluation_data as data
from scripts.rebar_pipeline import evaluate_checkpoint_comparison as comparison


def test_reviewed_cv_coverage_and_unchanged_outer_queries():
    task, final_membership, identity = data.load_task()
    original = json.loads(data.ORIGINAL_SPLIT.read_text())
    expected_final = [r['task_row_index'] for r in original['rows'] if r['cv_fold'] == 1]
    assert final_membership['query_task_row_indices'] == expected_final
    assert len(final_membership['context_task_row_indices']) == 152
    assert task.X.shape == (278, 65)
    assert task.feature_group_counts['material'] == 31
    assert task.X.reference_type.eq('SCE').all()
    np.testing.assert_array_equal(task.fixed_split.preprocessing_index, task.fixed_split.train_index)
    assert task.split_strategy == 'rebar_pipeline_final_test'

    development = final_membership['context_task_row_indices']
    expected_queries = [expected_final] + [sorted(x.tolist()) for x in np.array_split(
        np.random.default_rng(42).permutation(development), 4)]
    covered = []
    for fold, query in enumerate(expected_queries, 1):
        membership = data.fold_membership(data.DEFAULT_SPLIT, fold)
        context = membership['context_task_row_indices']
        assert membership['query_task_row_indices'] == query
        assert not set(context) & set(query)
        assert sorted(context + query) == identity['active_task_row_indices']
        assert len(query) == (56 if fold == 1 else 38)
        covered.extend(query)
    assert sorted(covered) == identity['active_task_row_indices']
    assert len(set(covered)) == 208
    audit = pd.read_csv(data.DEFAULT_SPLIT.parent / 'composition_row_audit.csv').set_index('row_id')
    active_ids = [identity['row_ids'][i] for i in development]
    assert audit.loc[active_ids, 'eligible_candidate'].all()
    final_ids = [identity['row_ids'][i] for i in expected_final]
    assert (~audit.loc[final_ids, 'composition_eligible']).sum() == 13
    with pytest.raises(ValueError, match='frozen outer'):
        data.load_task(fold=2, exploratory=False)


def test_comparison_defaults_to_reviewed_native_features(tmp_path):
    args = comparison.parse_args(['--candidate-run', str(tmp_path / 'candidate'),
                                  '--output-dir', str(tmp_path / 'out'), '--dry-run'])
    command = comparison.evaluation_command(args)
    assert command[command.index('--rebar-split-manifest') + 1] == str(data.DEFAULT_SPLIT)
    assert command[command.index('--exploratory-cv-folds') + 1] == '5'
    assert 'reference_type' in command
    assert '--native-tabicl-baselines' in command
    assert '--fixed-schema-model' not in command
    assert '--missing-aware-model' not in command
