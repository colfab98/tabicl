from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import gc,json,time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from scripts import eval_corrosion_datasets as ev
from tabicl.sklearn.preprocessing import TransformToNumerical

root=Path('/home/fcolanto/projects/tabicl')
out=root/'corrosion_datasets/analysis/combined_epit_soccol_mpea'/('missingness_audit_'+datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True,exist_ok=False)
data=root/'corrosion_datasets/datasets/combined_epit_soccol_mpea/processed'
task=ev._make_combined_pipeline_task(data)
tr,te=task.fixed_split.train_index,task.fixed_split.test_index
rawtr=task.X.iloc[tr].reset_index(drop=True); rawte=task.X.iloc[te].reset_index(drop=True)
ytr=task.y.iloc[tr].to_numpy(dtype=float);yte=task.y.iloc[te].to_numpy(dtype=float)
encoder=TransformToNumerical(preserve_missing_values=True)
Xtr=encoder.fit_transform(rawtr);Xte=encoder.transform(rawte)
fill=np.nanmean(Xtr,axis=0);cat=encoder._categorical_input_positions_;fill[cat]=-1
assert np.isfinite(fill).all()
filltr=np.where(np.isnan(Xtr),fill,Xtr);fillte=np.where(np.isnan(Xte),fill,Xte)
for original,filled in [(Xtr,filltr),(Xte,fillte)]: np.testing.assert_array_equal(original[np.isfinite(original)],filled[np.isfinite(original)])
legacy=TransformToNumerical();legtr=legacy.fit_transform(rawtr);legte=legacy.transform(rawte)
legacycodes_tr=np.where(np.isnan(Xtr),np.nan,legtr);legacycodes_te=np.where(np.isnan(Xte),np.nan,legte)
modelpath=root/'checkpoints/combined_epit_soccol_mpea/final_seed42/step-5500.ckpt'
cases=[('encoded_nan_control',Xtr,Xte),('query_only_imputed',Xtr,fillte),('context_only_imputed',filltr,Xte),('both_imputed_same_category_codes',filltr,fillte),('legacy_category_codes_keep_nan',legacycodes_tr,legacycodes_te)]
payload={'diagnostic_only':True,'created_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(),'checkpoint':str(modelpath),'query_rows':len(te),'context_rows':len(tr),'query_partition':'unchanged frozen outer MPEA','imputation':'numeric means fitted only on context; categorical -1; original observed values and category codes unchanged except named category-code control','rows':[]}
print('OUTPUT',out,flush=True)
for label,train,test in cases:
 started=time.time()
 est=ev.make_tabicl_regressor(model_path=str(modelpath),checkpoint_version='',device='cpu',n_estimators=8,random_state=42,allow_auto_download=False,feat_shuffle_method='none',norm_methods=['none'],remove_constant_features=False)
 est.fit(pd.DataFrame(train,columns=rawtr.columns),ytr)
 pred=est.predict(pd.DataFrame(test,columns=rawtr.columns),output_type='median')
 row={'case':label,'spearman':float(spearmanr(yte,pred).statistic),'rmse':float(np.sqrt(np.mean((yte-pred)**2))),'context_nan_fraction':float(np.isnan(train).mean()),'query_nan_fraction':float(np.isnan(test).mean()),'n_model_features':est.n_features_in_prediction_,'elapsed_seconds':time.time()-started}
 payload['rows'].append(row)
 pd.DataFrame({'query_index':te,'prediction':pred}).to_csv(out/(label+'_predictions.csv'),index=False)
 (out/'results.json').write_text(json.dumps(payload,indent=2))
 print(json.dumps(row),flush=True)
 del est; gc.collect()
print('COMPLETE',out,flush=True)
