from pathlib import Path
import json, shutil, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root=Path('/home/fcolanto/projects/tabicl')
out=root/'corrosion_datasets/analysis/combined_epit_soccol_mpea/catboost_label_audit_20261015T123629'
shutil.copyfile(__file__,out/'permutation_followup.py')
report=json.loads((out/'results.json').read_text())
data=root/'corrosion_datasets/datasets/combined_epit_soccol_mpea/processed'
registry=json.loads((data/'shared_feature_registry.json').read_text())
features=registry['feature_columns_in_order']; categories=registry['categorical_feature_columns']
context=pd.read_csv(data/'epit_soccol_context.csv');query=pd.read_csv(data/'mpea_queries.csv')
Xtr=context[features].copy();Xte=query[features].copy()
for frame in (Xtr,Xte):
 for c in categories: frame[c]=frame[c].astype('string').fillna('__CATBOOST_MISSING__').astype(str)
ytr=context[registry['target_column']].to_numpy(float)
yte=pd.read_csv(data/'mpea_scoring_targets.csv')[registry['target_column']].to_numpy(float)
rows=[]
for seed in range(100,120):
 start=time.time()
 model=CatBoostRegressor(**report['params'])
 model.fit(Xtr,np.random.default_rng(seed).permutation(ytr),cat_features=categories)
 pred=model.predict(Xte)
 row=dict(case='shuffled_context_labels_'+str(seed),spearman=float(spearmanr(yte,pred).statistic),
          rmse=float(np.sqrt(mean_squared_error(yte,pred))),mae=float(mean_absolute_error(yte,pred)),
          r2=float(r2_score(yte,pred)),elapsed_seconds=time.time()-start)
 rows.append(row)
 pd.DataFrame({'row_id':query.row_id,'prediction':pred}).to_csv(out/(row['case']+'_predictions.csv'),index=False)
 (out/'permutation_followup.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(row),flush=True)
controls=[r for r in report['rows'] if r['case'].startswith('shuffled_')]+rows
values=np.array([r['spearman'] for r in controls]); actual=report['rows'][0]['spearman']
summary=dict(n_permutations=len(values),observed_spearman=actual,null_mean=float(values.mean()),
             null_median=float(np.median(values)),null_min=float(values.min()),null_max=float(values.max()),
             permutations_at_least_observed=int((values>=actual).sum()),
             note='Small descriptive permutation audit; all context labels randomly permuted before fitting; no checkpoint or input-policy selection. Same frozen MPEA queries throughout.')
(out/'permutation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('SUMMARY',json.dumps(summary),flush=True)
