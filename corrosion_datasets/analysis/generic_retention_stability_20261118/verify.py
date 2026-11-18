"""Verify regenerated metrics against pre-existing independent evaluation exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
records=[]
for dataset,folder in [('epit','targetv5_four_ratios_through_8000_cv5_cpu'),('soccol','soccol_four_ratios_cv5_updated')]:
    reference=Path('corrosion_datasets/analysis')/(dataset+'_pipeline')/folder/'rows.csv'
    old=pd.read_csv(reference)
    for step in [8000,8500,9000,9500,10000]:
        path=ROOT/dataset/f'step_{step}'/'metrics.csv'
        fresh=pd.read_csv(path)
        ref=old[(old.model=='generic_baseline')&(old.checkpoint_step==step)]
        if len(ref)!=5:
            continue
        joined=fresh.merge(ref,left_on='fold',right_on='cv_fold',suffixes=('_new','_old'),validate='one_to_one')
        assert len(joined)==5
        for metric in ['test_mae','test_rmse','test_spearman','test_r2']:
            np.testing.assert_allclose(joined[metric+'_new'],joined[metric+'_old'],rtol=1e-5,atol=1e-5)
            records.append({'dataset':dataset,'step':step,'metric':metric,
                'max_absolute_delta':float(abs(joined[metric+'_new']-joined[metric+'_old']).max()),'source':str(reference)})
pd.DataFrame(records).to_csv(ROOT/'metric_reproduction.csv',index=False)
for dataset in ['epit','soccol']:
    for step in [8000,8500,9000,9500,10000]:
        frame=pd.read_csv(ROOT/dataset/f'step_{step}'/'predictions.csv')
        metrics=pd.read_csv(ROOT/dataset/f'step_{step}'/'metrics.csv')
        from scipy.stats import spearmanr
        for fold,part in frame.groupby('cv_fold'):
            ref=metrics[metrics.fold==fold].iloc[0]
            np.testing.assert_allclose(abs(part.prediction_mV-part.observed_mV).mean(),ref.test_mae,atol=1e-5,rtol=1e-5)
            np.testing.assert_allclose(spearmanr(part.prediction_mV,part.observed_mV).statistic,ref.test_spearman,atol=1e-5,rtol=1e-5)
print('Passed: 50 fold prediction exports; 30 fold comparisons with previously saved evaluations.')
