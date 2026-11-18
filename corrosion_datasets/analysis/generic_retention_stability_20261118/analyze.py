"""Analyze checkpoint sensitivity with frozen recipe formulas and query/context rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.epit_pipeline.artifact_hashes import sha256_file

ROOT = Path(__file__).resolve().parent
STEPS = [8000,8500,9000,9500,10000]
summary_rows, overall_metrics, flip_rows, examples = [], [], [], []
fig, axes = plt.subplots(2,2,figsize=(11,7.2), constrained_layout=True)
for di, (dataset,recipe_name) in enumerate([('epit','selected_pairs_all760_generic10k'),('soccol','selected_pairs_allaccepted_generic10k')]):
    recipe=Path('corrosion_datasets/analysis')/f'{dataset}_pipeline'/recipe_name
    pairs=pd.read_csv(recipe/'all_pairs.csv').sort_values(['task_row_index','formula']).reset_index(drop=True)
    original=pd.read_csv(recipe/'generic_cv_predictions.csv').sort_values('task_row_index').reset_index(drop=True)
    assert not pairs.duplicated(['task_row_index','formula']).any()
    sample=original[['task_row_index','cv_fold','observed_mV']].copy()
    hashes={}
    for step in STEPS:
        path=ROOT/dataset/f'step_{step}'/'predictions.csv'
        pred=pd.read_csv(path).sort_values('task_row_index').reset_index(drop=True)
        assert pred.task_row_index.tolist()==original.task_row_index.tolist()
        assert pred.cv_fold.tolist()==original.cv_fold.tolist()
        np.testing.assert_allclose(pred.observed_mV, original.observed_mV, atol=1e-8, rtol=0)
        hashes[str(path.relative_to(ROOT))]=sha256_file(path)
        sample[f'prediction_{step}']=pred.prediction_mV
        sample[f'error_{step}']=abs(pred.prediction_mV-pred.observed_mV)
        errors=sample.set_index('task_row_index')[f'error_{step}']
        pairs[f'gain_{step}']=pairs.task_row_index.map(errors)-pairs.formula_error_mV
        pairs[f'keep_{step}']=pairs[f'gain_{step}']>0
        metrics=pd.read_csv(ROOT/dataset/f'step_{step}'/'metrics.csv')
        overall_metrics.append({'dataset':dataset,'step':step,'n':len(pred),'pooled_mae':sample[f'error_{step}'].mean(),
            'mean_fold_mae':metrics.test_mae.mean(),'mean_fold_spearman':metrics.test_spearman.mean()})
    np.testing.assert_allclose(sample.prediction_10000,original.prediction_mV,atol=1e-8,rtol=0)
    assert (pairs.keep_10000==pairs.retained).all()
    np.testing.assert_allclose(pairs.gain_10000,pairs.gain_mV,atol=1e-8,rtol=0)
    keepcols=[f'keep_{s}' for s in STEPS]
    gaincols=[f'gain_{s}' for s in STEPS]
    pairs['wins_out_of_5']=pairs[keepcols].sum(axis=1)
    pairs['mean_gain']=pairs[gaincols].mean(axis=1)
    pairs['min_gain']=pairs[gaincols].min(axis=1)
    pairs['max_gain']=pairs[gaincols].max(axis=1)
    pairs['gain_9k_minus_10k']=pairs.gain_9000-pairs.gain_10000
    pairs['mean_gain_before_10k']=pairs[[f'gain_{s}' for s in STEPS[:-1]]].mean(axis=1)
    pairs['wins_before_10k']=pairs[[f'keep_{s}' for s in STEPS[:-1]]].sum(axis=1)
    counts=pairs.groupby('task_row_index')[keepcols].sum()
    for step in STEPS:
        sample[f'retained_formulas_{step}']=sample.task_row_index.map(counts[f'keep_{step}']).fillna(0).astype(int)
    sample['eligible']=sample.task_row_index.isin(pairs.task_row_index)
    sample['abs_change_9k_10k']=abs(sample.prediction_9000-sample.prediction_10000)
    sample['prediction_range']=sample[[f'prediction_{s}' for s in STEPS]].max(axis=1)-sample[[f'prediction_{s}' for s in STEPS]].min(axis=1)
    sample['prediction_sd']=sample[[f'prediction_{s}' for s in STEPS]].std(axis=1,ddof=0)
    for fold in ['all',1,2,3,4,5]:
        pa=pairs if fold=='all' else pairs[pairs.cv_fold==fold]
        sa=sample if fold=='all' else sample[sample.cv_fold==fold]
        r=pa[pa.keep_10000]
        summary_rows.append({'dataset':dataset,'fold':fold,'samples':len(sa),'eligible_samples':int(sa.eligible.sum()),
            'pairs':len(pa),'retained_pairs_10k':len(r),'retained_parents_10k':r.task_row_index.nunique(),
            'retained_pairs_win_all5':int((r.wins_out_of_5==5).sum()),
            'retained_pairs_win_atleast4':int((r.wins_out_of_5>=4).sum()),
            'retained_pairs_win_atmost2':int((r.wins_out_of_5<=2).sum()),
            'retained_pairs_win_only10k':int((r.wins_out_of_5==1).sum()),
            'retained_pairs_nonpositive_mean_gain':int((r.mean_gain<=0).sum()),
            'retained_pairs_nonpositive_mean_gain_before10k':int((r.mean_gain_before_10k<=0).sum()),
            'original_retained_parents_with_all5_pair':r.loc[r.wins_out_of_5==5,'task_row_index'].nunique(),
            'all_pairs_win_all5':int((pa.wins_out_of_5==5).sum()),
            'all_pairs_win_atleast4':int((pa.wins_out_of_5>=4).sum()),
            'new_pairs_majority4_not10k':int(((pa.wins_out_of_5>=4)&~pa.keep_10000).sum()),
            'changed_pair_decision_any':int(((pa.wins_out_of_5>0)&(pa.wins_out_of_5<5)).sum()),
            'prediction_change_9k_10k_mean':sa.abs_change_9k_10k.mean(),
            'prediction_change_9k_10k_median':sa.abs_change_9k_10k.median(),
            'prediction_change_9k_10k_p90':sa.abs_change_9k_10k.quantile(.9),
            'prediction_change_9k_10k_gt25_pct':100*(sa.abs_change_9k_10k>25).mean(),
            'prediction_change_9k_10k_gt50_pct':100*(sa.abs_change_9k_10k>50).mean(),
            'prediction_change_9k_10k_gt100_pct':100*(sa.abs_change_9k_10k>100).mean(),
            'prediction_spearman_9k_10k':spearmanr(sa.prediction_9000,sa.prediction_10000).statistic,
            'prediction_range_median':sa.prediction_range.median(),'prediction_sd_median':sa.prediction_sd.median()})
        for step in STEPS:
            a=pa[f'keep_{step}']; b=pa.keep_10000
            oldparents=set(pa.loc[b,'task_row_index']);newparents=set(pa.loc[a,'task_row_index'])
            flip_rows.append({'dataset':dataset,'fold':fold,'step':step,'retained_pairs':int(a.sum()),
                'informed_ratio':a.mean(),'shared_pairs':int((a&b).sum()),'lost_from10k':int((b&~a).sum()),
                'added_vs10k':int((a&~b).sum()),'flipped_pairs':int((a!=b).sum()),
                'fraction_10k_pairs_still_retained':(a&b).sum()/b.sum(),'jaccard':(a&b).sum()/(a|b).sum(),
                'retained_parents':len(newparents),'lost_10k_parents':len(oldparents-newparents),
                'added_parents':len(newparents-oldparents),
                'prediction_change_mean':abs(sa[f'prediction_{step}']-sa.prediction_10000).mean(),
                'signed_change_toward10k':(sa.prediction_10000-sa[f'prediction_{step}']).mean(),
                'fraction_shifted_down_toward10k':(sa.prediction_10000<sa[f'prediction_{step}']).mean(),
                'prediction_spearman_with10k':spearmanr(sa.prediction_10000,sa[f'prediction_{step}']).statistic})
    for bins,view in [('all',pairs),('retained10k',pairs[pairs.keep_10000])]:
        view.groupby('wins_out_of_5').size().rename('pairs').to_csv(ROOT/dataset/f'win_frequency_{bins}.csv')
    fsummary=pairs.groupby('formula').agg(pairs=('formula','size'),retained_10k=('keep_10000','sum'),
        stable_all5=('wins_out_of_5',lambda x:(x==5).sum()),stable_atleast4=('wins_out_of_5',lambda x:(x>=4).sum()))
    fsummary.to_csv(ROOT/dataset/'per_formula.csv')
    pairs.to_csv(ROOT/dataset/'pair_stability.csv',index=False)
    sample.to_csv(ROOT/dataset/'sample_stability.csv',index=False)
    ex=pairs[pairs.keep_10000 & ~pairs.keep_9000].sort_values('gain_9k_minus_10k').drop_duplicates('task_row_index').head(6)
    ex=ex.merge(sample[['task_row_index','prediction_9000','prediction_10000']],on='task_row_index',validate='one_to_one')
    ex.insert(0,'dataset',dataset);examples.append(ex)
    pa=pd.DataFrame(flip_rows); pa=pa[(pa.dataset==dataset)&(pa.fold=='all')]
    ax=axes[0,di];ax.plot(pa.step,100*pa.fraction_10k_pairs_still_retained,'o-',color='#007c91')
    ax.set(title=dataset.upper()+': original retained pairs still winning',ylabel='% of 10k-retained pairs',ylim=(0,105),xlabel='Generic checkpoint')
    ax.grid(alpha=.2)
    ax=axes[1,di]; ax.hist(sample.abs_change_9k_10k,bins=np.arange(0,55,2.5),color='#007c91',alpha=.85)
    ax.axvline(sample.abs_change_9k_10k.median(),color='#ca5a29',label=f'Median {sample.abs_change_9k_10k.median():.1f} mV')
    ax.set(title=dataset.upper()+': change in predictions, 9k → 10k',xlabel='Absolute prediction change (mV)',ylabel='Samples')
    ax.legend();ax.grid(alpha=.2)
    (ROOT/dataset/'analysis_identity.json').write_text(json.dumps({'scope':'Exploratory sensitivity of all-data recipe construction; not independent model evaluation',
        'steps':STEPS,'prediction_sha256':hashes,'all_pairs_sha256':sha256_file(recipe/'all_pairs.csv'),
        'analysis_script_sha256':sha256_file(Path(__file__))},indent=2))
pd.DataFrame(summary_rows).to_csv(ROOT/'summary.csv',index=False)
pd.DataFrame(overall_metrics).to_csv(ROOT/'checkpoint_metrics.csv',index=False)
pd.DataFrame(flip_rows).to_csv(ROOT/'selection_changes.csv',index=False)
pd.concat(examples,ignore_index=True).to_csv(ROOT/'example_flips.csv',index=False)
fig.savefig(ROOT/'checkpoint_stability.png',dpi=180)
fig.savefig(ROOT/'checkpoint_stability.pdf')
print(pd.DataFrame(summary_rows).query('fold == "all"').to_string(index=False))
print(pd.DataFrame(overall_metrics).to_string(index=False))
print(pd.DataFrame(flip_rows).query('fold == "all"').to_string(index=False))
