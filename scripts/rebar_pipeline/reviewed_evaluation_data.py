"""Reviewed rebar inputs for the shared native checkpoint comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.epit_pipeline.exploratory_cv import CV_SEED
from scripts.rebar_pipeline.evaluation_data import (
    DEFAULT_SPLIT as ORIGINAL_SPLIT, ROOT, TASK_ID, MEMBERSHIP_KEY,
)
from scripts.rebar_pipeline.prepare_evaluation import sha
from scripts.rebar_pipeline.prepare_reviewed import DEFAULT_OUTPUT, validate

DEFAULT_SPLIT = DEFAULT_OUTPUT / 'processing_manifest.json'
CATEGORICAL_COLUMNS = ('reference_type',)


def frozen_split(path=DEFAULT_SPLIT):
    path = Path(path)
    if path.name != 'processing_manifest.json':
        raise ValueError('Reviewed rebar evaluation requires processing_manifest.json.')
    manifest = validate(path.parent)
    registry = json.loads((path.parent / 'feature_registry.json').read_text())
    rows = pd.read_csv(path.parent / 'split_assignments.csv')
    development = pd.read_csv(path.parent / 'development_context_features.csv').row_id.tolist()
    final = manifest['frozen_final_query_row_ids']
    indices = dict(zip(rows.row_id, rows.task_row_index))
    if (not rows.row_id.is_unique or rows.task_row_index.tolist() != list(range(len(rows)))
            or set(development) & set(final) or not set(development + final) <= set(indices)
            or rows.loc[rows.outer_split == 'final_test', 'row_id'].tolist() != final):
        raise RuntimeError('Reviewed rebar row identities do not match the frozen partition.')
    return dict(manifest, dataset=registry, rows=rows.to_dict('records'),
                development_indices=[int(indices[r]) for r in development],
                final_indices=[int(indices[r]) for r in final])


def _membership(manifest, fold):
    if fold not in range(1, 6):
        raise ValueError('CV fold must be between 1 and 5.')
    development, final = manifest['development_indices'], manifest['final_indices']
    queries = [final] + [sorted(chunk.tolist()) for chunk in np.array_split(
        np.random.default_rng(CV_SEED).permutation(development), 4)]
    query = queries[fold - 1]
    return {'context_task_row_indices': sorted((set(development) | set(final)) - set(query)),
            'query_task_row_indices': query}


def fold_membership(manifest_path, fold):
    return _membership(frozen_split(manifest_path), fold)


def load_task(manifest_path=DEFAULT_SPLIT, *, fold=1, exploratory=False):
    from scripts import eval_corrosion_datasets as ev

    if not exploratory and fold != 1:
        raise ValueError('Final evaluation must keep frozen outer queries.')
    manifest_path = Path(manifest_path)
    manifest = frozen_split(manifest_path)
    data = pd.read_csv(manifest_path.parent / 'features_all_frozen_rows.csv')
    # This original target export is an input covered by the reviewed hash lock.
    targets = pd.read_csv(ORIGINAL_SPLIT.parent / 'targets.csv')
    row_ids = [r['row_id'] for r in manifest['rows']]
    if data.row_id.tolist() != row_ids or targets.row_id.tolist() != row_ids:
        raise RuntimeError('Reviewed rebar features and targets have different row identities.')
    registry = manifest['dataset']
    columns = registry['feature_columns_in_order']
    X, y = data[columns].copy(), targets[registry['target_column']]
    if not np.isfinite(y).all() or np.isinf(X.select_dtypes(include='number').to_numpy()).any():
        raise RuntimeError('Nonfinite rebar target or infinite feature.')
    membership = _membership(manifest, fold)
    context = np.asarray(membership['context_task_row_indices'], dtype=int)
    strategy = f'rebar_pipeline_exploratory_cv_fold_{fold}' if exploratory else 'rebar_pipeline_final_test'
    split = ev.EvalSplit(train_index=context,
        test_index=np.asarray(membership['query_task_row_indices'], dtype=int),
        preprocessing_index=context.copy(), split_strategy=strategy)
    n_material = len(registry['composition_features'])
    task = ev.EvalTask(task_id=TASK_ID, dataset='rebar_pitting', table='rebar',
        target=registry['target_column'], threshold=float('nan'), target_binning='continuous',
        target_bins=0, bin_edges=[], class_labels=[], class_counts={}, X=X, y=y,
        y_ordinal=pd.Series(np.zeros(len(y), dtype=int)), task_family='normal_corrosion',
        feature_groups_used=['material', 'environment_history'],
        feature_group_counts={'material': n_material, 'environment_history': len(columns)-n_material},
        dropped_feature_columns=list(registry['excluded_predictors']),
        quality_flags=['reviewed_v2', 'source_verification_incomplete',
                       'frozen_queries_include_unresolved_compositions'],
        split_groups=None, split_strategy=strategy, include_in_summary=True,
        summary_exclusion_reason='', fixed_split=split)
    return task, membership, dict(manifest_sha256=sha(manifest_path),
        inputs=manifest['inputs'], exports=manifest['exports'], feature_columns=columns,
        row_ids=row_ids, target_unit=registry['target_unit'], cv_seed=CV_SEED,
        active_task_row_indices=sorted(manifest['development_indices'] + manifest['final_indices']),
        split_policy='Frozen outer queries plus four shuffled accepted-development row folds; '
                     'development queries are only for exploratory checkpoint comparison.')
