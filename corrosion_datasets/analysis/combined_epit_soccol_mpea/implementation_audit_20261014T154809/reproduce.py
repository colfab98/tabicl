from pathlib import Path
from datetime import datetime
import gc, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from huggingface_hub import hf_hub_download
from scripts import eval_corrosion_datasets as ev
from tabicl.sklearn.preprocessing import TransformToNumerical

root=Path('/home/fcolanto/projects/tabicl')
out=root/'corrosion_datasets/analysis/combined_epit_soccol_mpea'/('implementation_audit_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True,exist_ok=False)
data=root/'corrosion_datasets/datasets/combined_epit_soccol_mpea/processed'
task=ev._make_combined_pipeline_task(data)
tr,te=task.fixed_split.train_index,task.fixed_split.test_index
Xtr=task.X.iloc[tr].reset_index(drop=True)
Xte=task.X.iloc[te].reset_index(drop=True)
ytr=task.y.iloc[tr].to_numpy(dtype=float)
yte=task.y.iloc[te].to_numpy(dtype=float)
pretrained=hf_hub_download('jingang/TabICL','tabicl-regressor-v2-20260212.ckpt',local_files_only=True)
combined=root/'checkpoints/combined_epit_soccol_mpea/final_seed42/step-5500.ckpt'
generic=root/'checkpoints/tabicl_s1_regression_baseline/step-5500.ckpt'
encoder=TransformToNumerical()
filltr=pd.DataFrame(encoder.fit_transform(Xtr),columns=Xtr.columns)
fillte=pd.DataFrame(encoder.transform(Xte),columns=Xte.columns)
payload={'created_at':datetime.now().isoformat(),'diagnostic_only':True,'checkpoint_policy':'step 5500, the latest point in the user-referenced plot; no checkpoint selection','context_rows':len(tr),'query_rows':len(te),'query_policy':'unchanged frozen outer MPEA queries','data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.csv')},'torch_threads':torch.get_num_threads(),'rows':[]}
print('OUTPUT',out,flush=True)
for label,path,train,test in [('combined_original',combined,Xtr,Xte),('generic_matched',generic,Xtr,Xte),('pretrained_matched',pretrained,Xtr,Xte),('combined_imputed',combined,filltr,fillte)]:
 started=time.time()
 est=ev.make_tabicl_regressor(model_path=str(path),checkpoint_version='',device='cpu',n_estimators=8,random_state=42,allow_auto_download=False,feat_shuffle_method='none',norm_methods=['none'],remove_constant_features=False)
 est.fit(train,ytr)
 trace=[]
 original_forward=est._batch_forward
 def traced_forward(Xs,ys,*args,**kwargs):
  trace.append({'X_shape':list(Xs.shape),'y_shape':list(ys.shape),'context_nan_fraction':float(np.isnan(Xs[:,:len(tr)]).mean()),'query_nan_fraction':float(np.isnan(Xs[:,len(tr):]).mean()),'unique_ensemble_inputs':len({hashlib.sha256(np.nan_to_num(x,nan=9876543.0).tobytes()).hexdigest() for x in Xs})})
  return original_forward(Xs,ys,*args,**kwargs)
 est._batch_forward=traced_forward
 print('START',label,flush=True)
 pred=est.predict(test,output_type='median')
 row={'model':label,'model_path':str(path),'feat_shuffle_method':est.feat_shuffle_method,'norm_methods':est.norm_methods,'remove_constant_features':est.remove_constant_features,'requested_n_estimators':est.n_estimators,'ensemble_counts':{k:len(v) for k,v in est.ensemble_generator_.ensemble_configs_.items()},'preserve_missing_values':est.preserve_missing_values_,'n_model_features':est.n_features_in_prediction_,'y_mean':est.y_scaler_.mean_.tolist(),'y_scale':est.y_scaler_.scale_.tolist(),'test_spearman':float(spearmanr(yte,pred).statistic),'test_rmse':float(np.sqrt(np.mean((yte-pred)**2))),'trace':trace,'elapsed_seconds':time.time()-started}
 payload['rows'].append(row)
 (out/'results.json').write_text(json.dumps(payload,indent=2))
 pd.DataFrame({'query_index':te,'prediction':pred}).to_csv(out/(label+'_predictions.csv'),index=False)
 print(json.dumps(row),flush=True)
 del est,original_forward
 gc.collect()
print('COMPLETE',out,flush=True)
