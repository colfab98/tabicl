"""Compare saved formula draws with saved model errors; no inference."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / 'corrosion_datasets/analysis/epit_pipeline'
OUT = Path(__file__).resolve().parent
RUN = BASE / 'formula_mixture_real_features/run_20261113T122634_467179Z'
f = pd.read_csv(RUN / 'assigned_predictions.csv')
f = f[f.outer_split.eq('final_test')].copy()
s = pd.read_csv(BASE / 'targetv5_cv5_sample_diagnostic/sample_consistency.csv')
s = s[s.cv_fold.eq(1)]
p = pd.read_csv(BASE / 'v14_trial27_28_vs_70_columns_step5500_est8_20261106/predictions.csv')
r = pd.read_csv(BASE / 'targetv5_four_ratios_through_8000_cv5_cpu/cv_fold_1/rows.csv')
assert set(f.task_row_index) == set(s.task_row_index) == set(p.task_row_index)
assert len(s) == len(p) == 152
assert not f.duplicated(['repeat','task_row_index']).any()
assert f.groupby('task_row_index').size().eq(1000).all()
p['pretrained_error'] = np.abs(p.target - p.pretrained_tabicl_v2_70cols)
expected = r[r.model.eq('pretrained_tabicl_v2')].iloc[0]
assert abs(p.pretrained_error.mean() - expected.test_mae) < 1e-4
assert abs(spearmanr(p.target, p.pretrained_tabicl_v2_70cols).statistic - expected.test_spearman) < 1e-10
m = f.groupby('task_row_index').agg(formula_error=('absolute_error_mV','mean'), observed_mV=('observed_mV','first')).reset_index()
m = m.merge(s[['task_row_index','No.','Material class','generic_abs_error_mV','informed_abs_error_mV','observed_mV']], on='task_row_index',validate='one_to_one',suffixes=('','_model'))
m = m.merge(p[['task_row_index','target','pretrained_error','generic_baseline_70cols']],on='task_row_index',validate='one_to_one')
np.testing.assert_allclose(m.observed_mV,m.observed_mV_model,rtol=0,atol=1e-8)
np.testing.assert_allclose(m.observed_mV,m.target,rtol=0,atol=1e-8)
m['generic_5500_error']=np.abs(m.generic_baseline_70cols-m.target)
for name,col in [('generic','generic_abs_error_mV'),('pretrained','pretrained_error'),('generic_5500','generic_5500_error')]:
    m[f'gain_vs_{name}']=m[col]-m.formula_error
    draws=f.merge(m[['task_row_index',col]],on='task_row_index',validate='many_to_one')
    draws['win']=draws.absolute_error_mV < draws[col]-1
    fraction=draws.groupby('task_row_index').win.mean()
    m[f'win_fraction_vs_{name}']=m.task_row_index.map(fraction)
m['gain_vs_both']=m[['gain_vs_generic','gain_vs_pretrained']].min(axis=1)
m['informed_gain_vs_generic']=m.generic_abs_error_mV-m.informed_abs_error_mV
m.sort_values('gain_vs_both',ascending=False).to_csv(OUT/'per_sample.csv',index=False)
summary=[]
for group, frame in [('all',m),*list(m.groupby('Material class'))]:
    summary.append({'group':group,'n':len(frame),'formula_mae':frame.formula_error.mean(),
       'generic_mae':frame.generic_abs_error_mV.mean(),'pretrained_mae':frame.pretrained_error.mean(),
       'beats_generic_1mV':int(frame.gain_vs_generic.gt(1).sum()),
       'beats_pretrained_1mV':int(frame.gain_vs_pretrained.gt(1).sum()),
       'beats_both_1mV':int(frame.gain_vs_both.gt(1).sum()),
       'beats_both_25mV':int(frame.gain_vs_both.gt(25).sum()),
       'wins_75pct_draws_both':int(((frame.win_fraction_vs_generic>=.75)&(frame.win_fraction_vs_pretrained>=.75)).sum())})
summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
print(summary.round(2).to_string(index=False))
for baseline in ['generic','pretrained','generic_5500']:
 print(baseline,'strict wins',int(m[f'gain_vs_{baseline}'].gt(0).sum()),'wins>1mV',int(m[f'gain_vs_{baseline}'].gt(1).sum()))
both=m[m.gain_vs_both.gt(1)]
print('Both better: informed also improves generic >1mV',int(both.informed_gain_vs_generic.gt(1).sum()),'out of',len(both))
print('Examples:');print(m.nlargest(10,'gain_vs_both')[['No.','Material class','formula_error','generic_abs_error_mV','pretrained_error','informed_abs_error_mV','gain_vs_both']].round(1).to_string(index=False))
(OUT/'README.md').write_text('Comparison on 152 frozen outer-test samples only. Formula error is mean absolute error over 1000 assignments, not error of the mean prediction or best-formula selection. Generic/informed errors average checkpoints 500–5500 (11 checkpoints); this is a descriptive trajectory comparison, not a selected final model. Pretrained uses the saved 70-feature, 8-estimator, median, seed-42 evaluation from November 6; aggregate MAE agrees with current fold 1 within 0.0001 mV and Spearman within 1e-10. Main counts require improvement >1 mV; >25 mV and repeat consistency are supplementary. Repeats are not independent experiments. Inputs: '+str(RUN)+'; targetv5_cv5_sample_diagnostic/sample_consistency.csv; v14_trial27_28_vs_70_columns_step5500_est8_20261106/predictions.csv.\n')
