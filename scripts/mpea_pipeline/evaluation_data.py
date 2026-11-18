"""Native MPEA inputs and exploratory folds, anchored to the frozen outer split.

No EPIT/Soccol context rows, schema padding, missing-value filling or refitting.
The all-row CV is exploratory; only fold 1 is the frozen final-query evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from scripts.epit_pipeline.exploratory_cv import CV_SEED, FOLDS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / 'corrosion_datasets/datasets/mpea_corrosion/processed/splits_v1/split_manifest.json'
TASK_ID = 'mpea_corrosion__pitting_potential_mv_sce'
MEMBERSHIP_KEY = 'mpea_exploratory_cv_split'
CATEGORICAL_COLUMNS = ('Prep_medium', 'CP_aeration', 'CP_agitation', 'CP_anions_info',
                       'test_method', 'reference_type')


def frozen_split(path=DEFAULT_SPLIT):
    return load_frozen_split(Path(path), expected_manifest_schema='mpea_composition_split_manifest_v1')


def fold_membership(manifest_path, fold):
    if fold not in FOLDS:
        raise ValueError('Exploratory CV fold must be between 1 and 5.')
    frozen = frozen_split(manifest_path)
    rows = sorted(frozen.manifest['rows'], key=lambda row: row['task_row_index'])
    if [row['task_row_index'] for row in rows] != list(range(len(rows))):
        raise RuntimeError('MPEA task-row identities are not contiguous.')
    development = [row['task_row_index'] for row in rows if row['outer_split'] == 'development']
    final = [row['task_row_index'] for row in rows if row['outer_split'] == 'final_test']
    if len(development) != 214 or len(final) != 53:
        raise RuntimeError('MPEA evaluation requires the frozen 214/53 split.')
    queries = [final] + [sorted(chunk.tolist()) for chunk in np.array_split(
        np.random.default_rng(CV_SEED).permutation(development), 4)]
    query = queries[fold - 1]
    return {'context_task_row_indices': sorted(set(development + final) - set(query)),
            'query_task_row_indices': query}


def load_task(manifest_path=DEFAULT_SPLIT, *, fold=1, exploratory=True):
    from scripts import eval_corrosion_datasets as ev

    frozen = frozen_split(manifest_path)
    manifest = frozen.manifest
    inputs = {}
    for name, record in manifest['inputs'].items():
        path = ROOT / record['path']
        if sha256_file(path) != record['sha256']:
            raise RuntimeError(f'Frozen MPEA input changed: {path}')
        inputs[name] = path
    for record in frozen.lock['exported_tables'].values():
        path = frozen.manifest_path.parent / record['file']
        if sha256_file(path) != record['sha256']:
            raise RuntimeError(f'Frozen MPEA export changed: {path}')
    queries = pd.read_csv(inputs['queries'])
    targets = pd.read_csv(inputs['targets'])
    rows = sorted(manifest['rows'], key=lambda row: row['task_row_index'])
    row_ids = [row['row_id'] for row in rows]
    if (queries['row_id'].tolist() != row_ids or not queries['row_id'].is_unique
            or not targets['row_id'].is_unique or set(targets['row_id']) != set(row_ids)):
        raise RuntimeError('MPEA inputs do not match frozen row identities.')
    columns = manifest['dataset']['feature_columns_in_order']
    target = manifest['dataset']['target_column']
    X = queries.loc[:, columns].copy()
    y = targets.set_index('row_id').loc[row_ids, target].reset_index(drop=True)
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise RuntimeError('MPEA targets must be finite.')
    if not exploratory and fold != 1:
        raise ValueError('Final evaluation must keep the frozen outer queries.')
    membership = fold_membership(manifest_path, fold)
    context = np.asarray(membership['context_task_row_indices'], dtype=int)
    strategy = f'mpea_pipeline_exploratory_cv_fold_{fold}' if exploratory else 'mpea_pipeline_final_test'
    split = ev.EvalSplit(train_index=context,
        test_index=np.asarray(membership['query_task_row_indices'], dtype=int),
        preprocessing_index=context.copy(), split_strategy=strategy)
    task = ev.EvalTask(task_id=TASK_ID, dataset='mpea_corrosion', table='MPEA', target=target,
        threshold=float('nan'), target_binning='continuous', target_bins=0,
        bin_edges=[], class_labels=[], class_counts={}, X=X, y=y,
        y_ordinal=pd.Series(np.zeros(len(y), dtype=int)), task_family='normal_corrosion',
        feature_groups_used=['material', 'environment_history'],
        feature_group_counts={'material': 24, 'environment_history': len(columns)-24},
        dropped_feature_columns=[], quality_flags=['mpea_frozen_native_inputs_v1'],
        split_groups=None, split_strategy=strategy, include_in_summary=True,
        summary_exclusion_reason='', fixed_split=split)
    provenance = {'manifest_sha256': frozen.manifest_sha256, 'lock_sha256': frozen.lock_sha256,
                  'inputs': manifest['inputs'], 'feature_columns': columns, 'row_ids': row_ids,
                  'target_unit': manifest['dataset']['target_unit']}
    return task, membership, provenance
