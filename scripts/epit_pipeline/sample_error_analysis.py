"""Paired sample/category diagnostics; never infer formula causation from aggregates."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def pair_predictions(predictions):
    keys=['checkpoint_step','cv_fold','task_row_index']
    if predictions.duplicated(['model',*keys]).any():
        raise ValueError('Duplicate model/sample predictions.')
    if predictions.duplicated(['model','checkpoint_step','task_row_index']).any():
        raise ValueError('A sample appears in multiple query folds at the same checkpoint.')
    if predictions.groupby('task_row_index').observed_mV.nunique().gt(1).any():
        raise ValueError('Observed targets differ across models or checkpoints.')
    required=['observed_mV','prediction_mV']
    if not np.isfinite(predictions[required].to_numpy()).all():
        raise ValueError('Nonfinite predictions/targets.')
    frame=predictions.copy()
    frame['prediction_rank']=frame.groupby(['model','checkpoint_step','cv_fold']).prediction_mV.rank(pct=True)
    frame['observed_rank']=frame.groupby(['model','checkpoint_step','cv_fold']).observed_mV.rank(pct=True)
    frame['absolute_rank_error']=(frame.prediction_rank-frame.observed_rank).abs()
    generic=frame[frame.model.eq('generic_baseline')].drop(columns='model')
    informed=frame[~frame.model.eq('generic_baseline')]
    if generic.empty or informed.empty:
        raise ValueError('Requires generic and at least one informed model.')
    for label,group in informed.groupby('model'):
        if set(map(tuple,group[keys].to_numpy()))!=set(map(tuple,generic[keys].to_numpy())):
            raise ValueError(f'Unpaired query membership for {label}.')
    paired=informed.merge(generic,on=keys,suffixes=('','_generic'),validate='many_to_one')
    if not np.allclose(paired.observed_mV,paired.observed_mV_generic,rtol=0,atol=1e-10):
        raise ValueError('Informed and generic targets differ.')
    paired['informed_abs_error_mV']=(paired.prediction_mV-paired.observed_mV).abs()
    paired['generic_abs_error_mV']=(paired.prediction_mV_generic-paired.observed_mV).abs()
    paired['error_penalty_mV']=paired.informed_abs_error_mV-paired.generic_abs_error_mV
    paired['informed_bias_mV']=paired.prediction_mV-paired.observed_mV
    paired['generic_bias_mV']=paired.prediction_mV_generic-paired.observed_mV
    paired['rank_error_penalty']=paired.absolute_rank_error-paired.absolute_rank_error_generic
    paired['scope']=np.where(paired.cv_fold.eq(1),'outer_fold_1','exploratory_development_folds')
    return paired


def cluster_interval(group,seed=42,repeats=2000):
    """Publication-cluster percentile interval; samples already averaged over steps."""
    stats=group.groupby('Reference',dropna=False).error_penalty_mV.agg(['sum','count'])
    if len(stats)<5:
        return np.nan,np.nan
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(stats),size=(repeats,len(stats)))
    means=stats['sum'].to_numpy()[indices].sum(axis=1)/stats['count'].to_numpy()[indices].sum(axis=1)
    return tuple(np.quantile(means,[.025,.975]))


def categorize(metadata):
    frame=metadata.copy()
    if frame.task_row_index.duplicated().any():
        raise ValueError('Duplicate sample metadata.')
    for column,edges,labels in [
        ('Cr',[-np.inf,12,20,np.inf],['Cr <=12','Cr 12–20','Cr >20']),
        ('Mo',[-np.inf,0,3,np.inf],['Mo =0','Mo 0–3','Mo >3']),
        ('N',[-np.inf,0,.1,np.inf],['N =0','N 0–0.1','N >0.1']),
        ('CP_pH',[-np.inf,3,6,9,np.inf],['pH <=3','pH 3–6','pH 6–9','pH >9']),
        ('CP_temp',[-np.inf,30,60,np.inf],['T <=30','T 30–60','T >60']),
        ('CP_Cl',[-np.inf,.1,1,np.inf],['Cl <=0.1','Cl 0.1–1','Cl >1']),
    ]:
        frame[column+'_bin']=pd.cut(frame[column],edges,labels=labels).astype(object).fillna('missing')
    for column in ['phase_FCC','phase_BCC','phase_HCP','phase_IM']:
        frame[column+'_known']=np.where(frame[column].isna(),'missing',np.where(frame[column]>0,'present','reported_zero'))
    frame['censored_target']=frame['Epit, mV (SCE) Avg.'].astype(str).str.contains(r'[<>]',regex=True)
    frame['family_and_heat']=frame['Material class'].fillna('missing')+' | '+frame.heat_treatment_type.fillna('missing')
    return frame


def summarize_categories(samples,group_columns,tie_mv):
    records=[]
    for (model,scope),partition in samples.groupby(['model','scope']):
        for column in group_columns:
            for label,group in partition.groupby(column,dropna=False):
                lower,upper=cluster_interval(group)
                per_source=group.groupby('Reference',dropna=False).error_penalty_mV.mean()
                records.append({'model':model,'scope':scope,'category_type':column,'category':str(label),
                    'n_samples':len(group),'n_publications':len(per_source),
                    'informed_mae_mV':group.informed_abs_error_mV.mean(),
                    'generic_mae_mV':group.generic_abs_error_mV.mean(),
                    'mean_error_penalty_mV':group.error_penalty_mV.mean(),
                    'median_error_penalty_mV':group.error_penalty_mV.median(),
                    'fraction_samples_worse':(group.error_penalty_mV>tie_mv).mean(),
                    'fraction_samples_not_better':(group.error_penalty_mV>=-tie_mv).mean(),
                    'fraction_sources_worse':(per_source>tie_mv).mean(),
                    'mean_rank_error_penalty':group.rank_error_penalty.mean(),
                    'informed_bias_mV':group.informed_bias_mV.mean(),
                    'generic_bias_mV':group.generic_bias_mV.mean(),
                    'cluster_ci_low_mV':lower,'cluster_ci_high_mV':upper,
                    'support':('limited' if len(group)<10 or len(per_source)<3 else 'multi_source'),
                    'checkpoint_count_min':int(group.n_checkpoints.min()),
                    'checkpoint_count_max':int(group.n_checkpoints.max())})
    return pd.DataFrame(records).sort_values('mean_error_penalty_mV',ascending=False)


def analyze(predictions,metadata,output,tie_mv=1.):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    paired=pair_predictions(predictions)
    meta=categorize(metadata)
    if not set(paired.task_row_index)<=set(meta.task_row_index):
        raise ValueError('Missing sample metadata.')
    # Reduce repeated checkpoints to one record per sample before category statistics.
    samples=paired.groupby(['model','scope','cv_fold','task_row_index']).agg(
        n_checkpoints=('checkpoint_step','nunique'),
        observed_mV=('observed_mV','first'),
        informed_abs_error_mV=('informed_abs_error_mV','mean'),
        generic_abs_error_mV=('generic_abs_error_mV','mean'),
        error_penalty_mV=('error_penalty_mV','mean'),
        median_penalty_mV=('error_penalty_mV','median'),
        min_penalty_mV=('error_penalty_mV','min'),max_penalty_mV=('error_penalty_mV','max'),
        informed_bias_mV=('informed_bias_mV','mean'),generic_bias_mV=('generic_bias_mV','mean'),
        rank_error_penalty=('rank_error_penalty','mean'),
        fraction_checkpoints_worse=('error_penalty_mV',lambda x:(x>tie_mv).mean()),
        fraction_checkpoints_not_better=('error_penalty_mV',lambda x:(x>=-tie_mv).mean()),
    ).reset_index().merge(meta,on='task_row_index',validate='many_to_one')
    paired=paired.merge(meta,on='task_row_index',validate='many_to_one')
    paired.to_csv(output/'paired_sample_errors.csv',index=False)
    samples=samples.sort_values('error_penalty_mV',ascending=False)
    samples.to_csv(output/'sample_consistency.csv',index=False)
    persistent=samples[(samples.fraction_checkpoints_worse>=.75)&(samples.error_penalty_mV>tie_mv)]
    persistent.to_csv(output/'persistently_worse_samples.csv',index=False)
    samples[samples.fraction_checkpoints_not_better>=.75].to_csv(output/'persistently_not_better_samples.csv',index=False)
    columns=['Material class','heat_treatment_type','heat_treatment_cooling','surface_preparation_method',
             'test_solution_type','Reference','family_and_heat','censored_target',
             'Cr_bin','Mo_bin','N_bin','CP_pH_bin','CP_temp_bin','CP_Cl_bin',
             'phase_FCC_known','phase_BCC_known','phase_HCP_known','phase_IM_known']
    categories=summarize_categories(samples,columns,tie_mv)
    categories.to_csv(output/'category_errors.csv',index=False)
    by_step=[]
    for step,group in paired.groupby('checkpoint_step'):
        group=group.copy();group['n_checkpoints']=1
        table=summarize_categories(group,columns,tie_mv);table.insert(0,'checkpoint_step',step)
        by_step.append(table)
    pd.concat(by_step,ignore_index=True).to_csv(output/'category_errors_by_step.csv',index=False)
    details={
        'positive_penalty_means':'informed absolute error exceeds generic absolute error',
        'practical_tie_mV':tie_mv,
        'scope':'Fold 1 kept separate; folds 2–5 are exploratory and influenced prior fitting.',
        'checkpoint_aggregation':'Equal weight per selected checkpoint within each sample, then equal weight per sample.',
        'uncertainty':'2000 publication-cluster bootstrap draws, descriptive pointwise 95% intervals; omitted below 5 sources. No multiple-testing correction; not confirmatory.',
        'rank_metric':'Difference in per-sample absolute percentile-rank errors within each query fold. Not a per-sample decomposition of Spearman.',
        'persistent':'Worse or not better in at least 75% of selected checkpoints; stability is conditional on the chosen checkpoint set.',
        'causation':'Category losses are diagnostic associations. Formula changes require controlled development-only ablations.',
    }
    (output/'analysis_notes.json').write_text(json.dumps(details,indent=2)+'\n')
    lines=['# Informed versus generic: paired sample diagnostic','',
           'Positive penalty means the informed model is worse. Rows below average checkpoints within each sample before aggregating samples.',
           'Outer fold 1 is reported separately from exploratory development folds. Category confidence intervals are descriptive, not evidence of causation.','']
    for (model,scope),group in categories.groupby(['model','scope']):
        shortlist=group[(group.support=='multi_source')&(~group.category_type.eq('Reference'))].head(12)
        lines += [f'## {model}: {scope}','', '| Category | n | Sources | MAE penalty mV | Samples worse |',
                  '|---|---:|---:|---:|---:|']
        for row in shortlist.itertuples():
            category=f'{row.category_type}: {row.category}'.replace('|','/')
            lines.append(f'| {category} | {row.n_samples} | {row.n_publications} | {row.mean_error_penalty_mV:+.2f} | {row.fraction_samples_worse:.1%} |')
        lines.append('')
    lines += ['Inspect `persistently_worse_samples.csv` for exact sample IDs, source references, compositions and conditions.',
              'Inspect `category_errors_by_step.csv` before attributing an average loss to a persistent category problem.',
              'Groups overlap and small/source-specific groups are not independent discoveries.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model,group in samples.groupby('model'):
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
        maximum=max(group.generic_abs_error_mV.max(),group.informed_abs_error_mV.max())*1.05
        colors=dict(zip(sorted(group['Material class'].unique()),plt.get_cmap('tab10').colors))
        for ax,scope in zip(axes,['outer_fold_1','exploratory_development_folds']):
            for family,part in group[group.scope.eq(scope)].groupby('Material class'):
                ax.scatter(part.generic_abs_error_mV,part.informed_abs_error_mV,
                           color=colors[family],s=18,alpha=.65,label=family)
            ax.plot([0,maximum],[0,maximum],color='black',lw=1)
            ax.set_title(scope.replace('_',' '));ax.set_xlabel('Generic absolute error (mV)')
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Informed absolute error (mV)')
        axes[1].legend(fontsize=8)
        fig.suptitle(f'{model}: above diagonal = informed worse; errors averaged over selected checkpoints')
        fig.tight_layout();fig.savefig(output/f'{model}_paired_errors.png',dpi=180);plt.close(fig)
    return paired,samples,categories
