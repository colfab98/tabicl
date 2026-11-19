"""Frozen, evaluation-only rebar inputs; shared native model preprocessing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rebar_pipeline.prepare_evaluation import ROOT, BASE, sha

DEFAULT_SPLIT = BASE / 'processed/splits_v1/split_manifest.json'
TASK_ID = 'rebar_pitting__pitting_potential_mv_sce'
MEMBERSHIP_KEY = 'rebar_exploratory_cv_split'
CATEGORICAL_COLUMNS = ()


def frozen_split(path=DEFAULT_SPLIT):
    path = Path(path)
    lock = json.loads(path.with_name('split_lock.json').read_text())
    if sha(path) != lock['manifest_sha256']:
        raise RuntimeError('Frozen rebar manifest changed.')
    manifest = json.loads(path.read_text())
    if manifest['schema_version'] != 'rebar_evaluation_split_v1':
        raise RuntimeError('Unsupported rebar split schema.')
    if sha(ROOT/manifest['raw']['path']) != manifest['raw']['sha256']:
        raise RuntimeError('Frozen rebar raw workbook changed.')
    for name, digest in manifest['exports'].items():
        if sha(path.parent/name) != digest:
            raise RuntimeError(f'Frozen rebar export changed: {name}')
    return manifest


def fold_membership(manifest_path, fold):
    if fold not in range(1, 6):
        raise ValueError('CV fold must be between 1 and 5.')
    rows = frozen_split(manifest_path)['rows']
    return {'context_task_row_indices': [r['task_row_index'] for r in rows if r['cv_fold'] != fold],
            'query_task_row_indices': [r['task_row_index'] for r in rows if r['cv_fold'] == fold]}


def load_task(manifest_path=DEFAULT_SPLIT, *, fold=1, exploratory=True):
    from scripts import eval_corrosion_datasets as ev

    manifest_path = Path(manifest_path)
    manifest = frozen_split(manifest_path)
    data = pd.read_csv(manifest_path.parent/'features.csv')
    targets = pd.read_csv(manifest_path.parent/'targets.csv')
    row_ids = [r['row_id'] for r in manifest['rows']]
    if (data.row_id.tolist() != row_ids or targets.row_id.tolist() != row_ids
            or not data.row_id.is_unique):
        raise RuntimeError('Rebar row identities do not match frozen split.')
    columns = manifest['dataset']['feature_columns_in_order']
    X = data[columns].copy()
    target = manifest['dataset']['target_column']
    y = targets[target]
    if not np.isfinite(y).all() or np.isinf(X.to_numpy()).any():
        raise RuntimeError('Nonfinite rebar target or infinite feature.')
    if not exploratory and fold != 1:
        raise ValueError('Final evaluation must keep frozen outer queries.')
    membership = fold_membership(manifest_path, fold)
    context = np.asarray(membership['context_task_row_indices'], dtype=int)
    strategy = f'rebar_pipeline_exploratory_cv_fold_{fold}' if exploratory else 'rebar_pipeline_final_test'
    split = ev.EvalSplit(train_index=context, test_index=np.asarray(membership['query_task_row_indices'], dtype=int),
                         preprocessing_index=context.copy(), split_strategy=strategy)
    n_material = sum(c.endswith('_wtpct') and not c.startswith('cement_') for c in columns)
    task = ev.EvalTask(task_id=TASK_ID, dataset='rebar_pitting', table='rebar', target=target,
        threshold=float('nan'), target_binning='continuous', target_bins=0, bin_edges=[],
        class_labels=[], class_counts={}, X=X, y=y, y_ordinal=pd.Series(np.zeros(len(y), dtype=int)),
        task_family='normal_corrosion', feature_groups_used=['material', 'environment_history'],
        feature_group_counts={'material': n_material, 'environment_history': len(columns)-n_material},
        dropped_feature_columns=[], quality_flags=[manifest['status'], 'publication_grouped_cv'],
        split_groups=None, split_strategy=strategy, include_in_summary=True,
        summary_exclusion_reason='', fixed_split=split)
    return task, membership, dict(manifest_sha256=sha(manifest_path), raw=manifest['raw'],
        exports=manifest['exports'], feature_columns=columns, row_ids=row_ids,
        status=manifest['status'], target_unit='mV SCE', split_group='publication')
