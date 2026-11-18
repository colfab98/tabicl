"""Summarize diagnostics and compare with saved matched-step model errors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.epit_pipeline.artifact_hashes import sha256_file
ROOT=Path(__file__).resolve().parent
comparison=[];correlations=[];replicate=[];coverage=[]
fig,axs=plt.subplots(1,2,figsize=(10,4.3),constrained_layout=True)
for di,dataset in enumerate(['epit','soccol']):
    folder=ROOT/dataset
    n=pd.read_csv(folder/'measured_neighbour_comparisons.csv')
    primary=n[(n.clr_rms_distance<=.3)&(n.neighbour_rank<=3)]
    agg=primary.groupby('parent').agg(local_delta_mae=('delta_error_mV','mean'),
        local_no_change_mae=('no_change_error_mV','mean'),local_improvement=('improvement_over_no_change_mV','mean'),
        unique_neighbours=('neighbour','nunique'),source=('source','first'))
    source=Path('corrosion_datasets/analysis')/(dataset+'_pipeline')/'single_rule_retention_delay_20261118/step_3000/samples.csv'
    evaluated=pd.read_csv(source)
    paired=agg.merge(evaluated,left_index=True,right_on='task_row_index',validate='one_to_one')
    assert paired.group.eq('retained').all()
    paired['model_change']=np.where(paired.delta_error_mV>1,'worsened',np.where(paired.delta_error_mV < -1,'improved','within_1mV'))
    paired.to_csv(folder/'local_diagnostic_vs_model_step3000.csv',index=False)
    for group,part in paired.groupby('model_change'):
        comparison.append({'dataset':dataset,'model_change':group,'n_parents':len(part),
            'local_delta_mae':part.local_delta_mae.mean(),'local_no_change_mae':part.local_no_change_mae.mean(),
            'local_improvement':part.local_improvement.mean(),'model_delta_mae':part.delta_error_mV.mean()})
    for var in ['local_delta_mae','local_improvement']:
        correlations.append({'dataset':dataset,'n_parents':len(paired),'variable':var,
            'spearman_with_informed_minus_generic_error':spearmanr(paired[var],paired.delta_error_mV).statistic})
    identical=pd.read_csv(folder/'identical_recorded_input_pairs.csv')
    replicate.append({'dataset':dataset,'pairs':len(identical),'rows':len(set(identical.parent)|set(identical.neighbour)),
        'studies':identical.source.nunique(),'median_abs_target_difference':identical.absolute_target_difference_mV.median(),
        'p90_abs_target_difference':identical.absolute_target_difference_mV.quantile(.9),
        'pct_difference_gt25':100*(identical.absolute_target_difference_mV>25).mean()})
    for radius in [.15,.3,.6]:
        view=n[n.clr_rms_distance<=radius]
        coverage.append({'dataset':dataset,'radius':radius,'parents':view.parent.nunique(),
            'source_studies':view.source.nunique()})
    summary=pd.read_csv(folder/'perturbation_summary.csv').query('formula=="ALL" and strength>0')
    ax=axs[di]
    ax.bar(['0.1','0.3 (current)'],summary.target_change_gt100_pct,color=['#007c91','#cf6939'])
    for i,value in enumerate(summary.target_change_gt100_pct):ax.text(i,value+.6,f'{value:.1f}%',ha='center')
    ax.set(title=dataset.upper(),xlabel='Composition perturbation strength',ylabel='Synthetic targets changing by >100 mV (%)',ylim=(0,36))
    ax.grid(axis='y',alpha=.2)
    (folder/'comparison_identity.json').write_text(json.dumps({'evaluation_step':3000,'evaluation_predictions':str(source),
        'evaluation_sha256':sha256_file(source),'scope':'Associations only; same exploratory sample predictions used previously. No new model evaluation or training.'},indent=2))
pd.DataFrame(comparison).to_csv(ROOT/'model_error_association.csv',index=False)
pd.DataFrame(correlations).to_csv(ROOT/'model_error_correlations.csv',index=False)
pd.DataFrame(replicate).to_csv(ROOT/'recorded_replicate_summary.csv',index=False)
pd.DataFrame(coverage).to_csv(ROOT/'neighbour_coverage.csv',index=False)
fig.savefig(ROOT/'perturbation_strength.png',dpi=180)
fig.savefig(ROOT/'perturbation_strength.pdf')
print('ASSOCIATIONS\n',pd.DataFrame(comparison).to_string(index=False))
print('CORRELATIONS\n',pd.DataFrame(correlations).to_string(index=False))
print('RECORDED REPLICATES\n',pd.DataFrame(replicate).to_string(index=False))
print('COVERAGE\n',pd.DataFrame(coverage).to_string(index=False))
