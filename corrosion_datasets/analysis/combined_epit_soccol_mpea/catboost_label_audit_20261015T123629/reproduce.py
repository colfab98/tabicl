from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import shutil
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path('/home/fcolanto/projects/tabicl')
base = root / 'corrosion_datasets/analysis/combined_epit_soccol_mpea'
out = base / ('catboost_label_audit_' + datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, out / 'reproduce.py')
data = root / 'corrosion_datasets/datasets/combined_epit_soccol_mpea/processed'
registry = json.loads((data / 'shared_feature_registry.json').read_text())
features = registry['feature_columns_in_order']
categories = registry['categorical_feature_columns']
target = registry['target_column']
context = pd.read_csv(data / 'epit_soccol_context.csv')
query = pd.read_csv(data / 'mpea_queries.csv')
assert target not in query.columns and target not in features
assert len(context) == 4787 and len(query) == 267 and len(features) == 50
assert not set(context.row_id) & set(query.row_id)
Xtr = context[features].copy()
Xte = query[features].copy()
for frame in (Xtr, Xte):
    for col in categories:
        frame[col] = frame[col].astype('string').fillna('__CATBOOST_MISSING__').astype(str)
ytr = context[target].to_numpy(float)
params = dict(iterations=1000, depth=6, learning_rate=.03, l2_leaf_reg=3.,
              random_seed=42, thread_count=8, loss_function='RMSE', eval_metric='RMSE',
              task_type='CPU', verbose=False, allow_writing_files=False)
print('OUTPUT', out, flush=True)
model = CatBoostRegressor(**params)
start = time.time()
# Independent direct fit: no shared evaluator, no eval_set, no query labels.
model.fit(Xtr, ytr, cat_features=categories)
pred = model.predict(Xte)
elapsed = time.time() - start
pd.DataFrame({'row_id':query.row_id, 'prediction':pred}).to_csv(out / 'direct_predictions.csv', index=False)
# Scoring targets are first loaded after the model has already predicted.
scoring = pd.read_csv(data / 'mpea_scoring_targets.csv')
assert query.row_id.tolist() == scoring.row_id.tolist()
yte = scoring[target].to_numpy(float)
def metrics(name, prediction, seconds):
    return dict(case=name, spearman=float(spearmanr(yte, prediction).statistic),
                rmse=float(np.sqrt(mean_squared_error(yte, prediction))),
                mae=float(mean_absolute_error(yte, prediction)), r2=float(r2_score(yte, prediction)),
                elapsed_seconds=seconds)
baseline = metrics('direct_context_only_fit', pred, elapsed)
print(json.dumps(baseline), flush=True)
saved = pd.read_csv(base / 'checkpoint_comparison_v2_cpu/rows.csv')
expected = saved.loc[saved.model.eq('catboost')].iloc[0]
assert np.isclose(baseline['spearman'], expected.test_spearman, atol=1e-12)
assert np.isclose(baseline['rmse'], expected.test_rmse, atol=1e-9)
importance = sorted(zip(features, model.feature_importances_.tolist()), key=lambda x:x[1], reverse=True)
hashes = pd.util.hash_pandas_object(pd.concat([Xtr, Xte], ignore_index=True), index=False).to_numpy()
overlap = np.isin(hashes[len(Xtr):], hashes[:len(Xtr)])
old = pd.read_csv(base / 'checkpoint_comparison_cpu/rows.csv')
joined = old.merge(saved, on=['model','checkpoint_step','task_id'], suffixes=('_old','_v2'))
comparison = joined[['model','checkpoint_step','model_source_old','model_source_v2','test_spearman_old','test_spearman_v2']]
comparison.to_csv(out / 'same_step_comparison.csv', index=False)
generic = comparison.loc[comparison.model.eq('generic_tabicl')]
assert generic.model_source_old.eq(generic.model_source_v2).all()
assert generic.test_spearman_old.eq(generic.test_spearman_v2).all()
report = dict(diagnostic_only=True, frozen_queries_unchanged=True, context_rows=len(Xtr), query_rows=len(Xte),
              independent_fit_loaded_scoring_targets_only_after_prediction=True, params=params,
              direct_catboost_matches_saved_metrics=True, generic_shared_steps=generic.checkpoint_step.tolist(),
              generic_scores_and_checkpoint_paths_unchanged=True,
              exact_feature_duplicate_queries_in_context=int(overlap.sum()),
              duplicate_check_limitation='Exact 50-feature equality only; different missingness or source encodings can conceal duplicate measurements.',
              unique_query_feature_rows=int(np.unique(hashes[len(Xtr):]).size),
              top_training_feature_importances=[{'feature':a,'importance':b} for a,b in importance[:10]],
              rows=[baseline])
(out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
for seed in [17,29,43]:
    shuffled = np.random.default_rng(seed).permutation(ytr)
    control = CatBoostRegressor(**params)
    start = time.time()
    control.fit(Xtr, shuffled, cat_features=categories)
    control_pred = control.predict(Xte)
    row = metrics('shuffled_context_labels_'+str(seed), control_pred, time.time()-start)
    report['rows'].append(row)
    pd.DataFrame({'row_id':query.row_id, 'prediction':control_pred}).to_csv(out / (row['case']+'_predictions.csv'), index=False)
    (out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(row), flush=True)
print('Exact feature overlaps:',report['exact_feature_duplicate_queries_in_context'],flush=True)
print('Generic unchanged at steps:',report['generic_shared_steps'],flush=True)
print('Top features:',report['top_training_feature_importances'],flush=True)
