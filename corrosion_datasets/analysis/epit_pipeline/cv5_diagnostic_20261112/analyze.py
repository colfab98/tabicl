"""Audit saved EPIT scores and formula OOF predictions; no model inference or fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'corrosion_datasets/analysis/epit_pipeline'
SOURCE = ROOT / 'targetv5_four_ratios_through_8000_cv5'

def save(frame, name):
    frame.to_csv(OUT / (name + '.csv'), index=False)

rows = pd.read_csv(SOURCE / 'rows.csv')
assert not rows.duplicated(['model', 'checkpoint_step', 'cv_fold']).any()
assert rows.groupby(['model', 'checkpoint_step']).cv_fold.apply(set).eq({1,2,3,4,5}).all()
payload = json.loads((SOURCE / 'results.json').read_text())
assert not payload['errors']
metrics = ['test_mae', 'test_spearman', 'test_rmse', 'test_r2']
base = rows[rows.model.eq('generic_baseline')][['checkpoint_step', 'cv_fold', *metrics]]
paired = rows[rows.model.str.startswith('epit_')].merge(base, on=['checkpoint_step','cv_fold'], suffixes=('','_generic'), validate='many_to_one')
paired['mae_gain_mV'] = paired.test_mae_generic - paired.test_mae
paired['rho_gain'] = paired.test_spearman - paired.test_spearman_generic
paired['mae_reduction_percent'] = 100 * paired.mae_gain_mV / paired.test_mae_generic
save(paired, 'paired_model_fold_scores')
summary = paired.groupby(['model','checkpoint_step']).agg(
    mae=('test_mae','mean'), generic_mae=('test_mae_generic','mean'),
    rho=('test_spearman','mean'), generic_rho=('test_spearman_generic','mean'),
    mae_gain_mV=('mae_gain_mV','mean'), rho_gain=('rho_gain','mean'),
    mae_gain_sd=('mae_gain_mV','std'),rho_gain_sd=('rho_gain','std'),
    mae_wins=('mae_gain_mV',lambda x:int((x>0).sum())),
    rho_wins=('rho_gain',lambda x:int((x>0).sum()))).reset_index()
save(summary,'paired_checkpoint_summary')
common = paired[paired.checkpoint_step.le(4500)].copy()
save(common.groupby('model').agg(mean_mae_gain_mV=('mae_gain_mV','mean'),mean_rho_gain=('rho_gain','mean'),
    mae_wins=('mae_gain_mV',lambda x:int((x>0).sum())),rho_wins=('rho_gain',lambda x:int((x>0).sum())),n=('cv_fold','size')).reset_index(),'common_window_descriptive')
save(paired[paired.cv_fold.eq(1)],'outer_fold_1_comparison')
fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True)
for label,g in summary.groupby('model'):
    axes[0].plot(g.checkpoint_step,g.mae_gain_mV,marker='o',label=label)
    axes[1].plot(g.checkpoint_step,g.rho_gain,marker='o',label=label)
for ax in axes:
    ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2)
axes[0].set_ylabel('MAE reduction versus generic (mV)')
axes[1].set_ylabel('Spearman gain versus generic')
axes[1].set_xlabel('Matched training checkpoint step')
axes[0].legend(ncol=2);fig.suptitle('EPIT exploratory CV: five-fold means; positive = informed better')
fig.tight_layout();fig.savefig(OUT/'matched_checkpoint_gains.png',dpi=180);plt.close(fig)

data = pd.read_csv(ROOT/'corrosion_datasets/datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/epit_processed.csv')
assign = pd.read_csv(ROOT/'corrosion_datasets/datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/split_assignments.csv')
data = data.merge(assign[['task_row_index','outer_split','composition_group']],on='task_row_index',validate='one_to_one')
recipe = json.loads((ROOT/'src/tabicl/prior/assets/epit_target_v5.json').read_text())
fit_rows = set(recipe['formula']['fitting_task_row_indices'])
data['in_formula_fit'] = data.task_row_index.isin(fit_rows)
assert not set(data.loc[data.outer_split.eq('final_test'),'task_row_index']) & fit_rows
save(data.groupby(['Material class','outer_split']).agg(n=('task_row_index','size'),formula_fit_rows=('in_formula_fit','sum'),publications=('Reference','nunique')).reset_index(),'family_coverage')
folds=[]
overlap=[]
for fold, membership in payload['fold_membership'].items():
    query=data.set_index('task_row_index').loc[membership['query_task_row_indices']]
    context=data.set_index('task_row_index').loc[membership['context_task_row_indices']]
    assert not set(query.index)&set(context.index)
    for family,g in query.groupby('Material class'):
        folds.append({'cv_fold':int(fold),'family':family,'n_query':len(g),'in_formula_fit':int(g.in_formula_fit.sum())})
    overlap.append({'cv_fold':int(fold),'queries':len(query),
                    'query_publication_seen_in_context':int(query.Reference.isin(context.Reference).sum()),
                    'query_composition_group_seen_in_context':int(query.composition_group.isin(context.composition_group).sum()),
                    'query_used_in_formula_fit':int(query.in_formula_fit.sum())})
save(pd.DataFrame(folds),'cv_family_counts');save(pd.DataFrame(overlap),'cv_overlap')
eligible=data[data.in_formula_fit].copy()
heat=[]
for label,g in eligible.groupby('heat_treatment_type',dropna=False):
    heat.append({'heat_type':str(label),'n':len(g),'publications':g.Reference.nunique(),
                 'dominant_publication_share':g.Reference.value_counts(normalize=True).iloc[0],
                 'dominant_publication':g.Reference.value_counts().index[0]})
save(pd.DataFrame(heat),'heat_publication_concentration')
save(pd.DataFrame([{'feature':c,'n_missing':int(eligible[c].isna().sum()),'missing_percent':100*eligible[c].isna().mean()}
    for c in ['CP_temp','CP_Cl','CP_pH','scan_rate','heat_treatment_type','heat_treatment_temperature_C','heat_treatment_duration_h','heat_treatment_cooling','surface_preparation_method','phase_FCC','phase_BCC','phase_HCP','phase_IM']]),'formula_input_missingness')

names=['existing_method_aware_pren_n','extend_heat_categories','extend_heat_without_method','extend_surface','extend_solution','existing_mns_inclusion_penalty','extend_material']
comparisons=[];groups=[];contrasts=[]
for protocol in ['frozen_development_folds','source_publication_folds']:
    folder=ANALYSIS/'formula_extensions_v2'/protocol
    s=pd.read_csv(folder/'summary.csv');s.insert(0,'protocol',protocol);comparisons.append(s)
    pred=pd.read_csv(folder/'predictions.csv')
    assert set(pred.task_row_index)==fit_rows
    assert not pred.duplicated(['name','task_row_index']).any()
    pred=pred[pred.name.isin(names)].merge(data,on='task_row_index',validate='many_to_one')
    observed_numeric = pd.to_numeric(pred['Epit, mV (SCE) Avg.'].astype(str).str.lstrip('>'))
    assert np.allclose(pred.observed_epit_mV, observed_numeric)
    pred['residual_mV']=pred.prediction_mV-pred.observed_epit_mV
    pred['abs_error_mV']=pred.residual_mV.abs()
    pred['pH_group']=pd.cut(pred.CP_pH,[-np.inf,3,6,9,np.inf],labels=['<=3','3–6','6–9','>9']).astype(object).fillna('missing')
    pred['temperature_group']=pd.cut(pred.CP_temp,[-np.inf,30,60,np.inf],labels=['<=30 C','30–60 C','>60 C']).astype(object).fillna('missing')
    for column in ['Material class','heat_treatment_type','surface_preparation_method','pH_group','temperature_group','Reference']:
        for (model,label),g in pred.groupby(['name',column],dropna=False):
            groups.append({'protocol':protocol,'formula':model,'group_type':column,'group':str(label),
                'n':len(g),'publications':g.Reference.nunique(),'mae_mV':g.abs_error_mV.mean(),
                'bias_mV':g.residual_mV.mean(),'median_abs_error_mV':g.abs_error_mV.median()})
    heatp=pred[pred.name.eq('extend_heat_categories')].copy()
    basep=pred[pred.name.eq('existing_method_aware_pren_n')][['task_row_index','abs_error_mV','prediction_mV']]
    heatp=heatp.merge(basep,on='task_row_index',suffixes=('','_base'),validate='one_to_one')
    heatp['heat_mae_gain_mV']=heatp.abs_error_mV_base-heatp.abs_error_mV
    heatp.insert(0,'protocol',protocol);contrasts.append(heatp)
save(pd.concat(comparisons),'formula_protocol_scores')
save(pd.DataFrame(groups),'formula_subgroup_errors')
contrast=pd.concat(contrasts)
save(contrast,'formula_heat_vs_base_per_sample')
save(contrast.sort_values('abs_error_mV',ascending=False).groupby('protocol').head(30),'formula_largest_errors')
for column in ['Material class','heat_treatment_type']:
    save(contrast.groupby(['protocol',column],dropna=False).agg(n=('task_row_index','size'),
         heat_mae=('abs_error_mV','mean'),base_mae=('abs_error_mV_base','mean'),
         heat_gain=('heat_mae_gain_mV','mean'),publications=('Reference','nunique')).reset_index(),
         'heat_gain_'+column.replace(' ','_'))
print('Wrote diagnostic tables and figure to',OUT)
print(summary[summary.checkpoint_step.isin([1000,2000,4500])].round(4).to_string(index=False))
print(pd.DataFrame(overlap).to_string(index=False))
