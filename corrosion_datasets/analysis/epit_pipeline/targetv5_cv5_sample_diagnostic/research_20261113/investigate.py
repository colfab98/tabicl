"""Interpret saved predictions; no inference, training, or formula refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ROOT=OUT.parents[4]
s=pd.read_csv(RUN/'sample_consistency.csv')
p=pd.read_csv(RUN/'paired_sample_errors.csv')
v=pd.read_csv(RUN/'metric_reproduction_deltas.csv')
assert len(s)==760 and len(p)==8360 and len(v)==110
assert not p.duplicated(['model','checkpoint_step','task_row_index']).any()
assert set(p.checkpoint_step)==set(range(500,5501,500))
assert p.groupby('task_row_index').size().eq(11).all()
assert v.test_spearman.abs().max()==0

def save(x,name):x.to_csv(OUT/f'{name}.csv',index=False)

families=s.groupby(['scope','Material class']).agg(n=('row_id','size'),sources=('Reference','nunique'),
    informed_mae=('informed_abs_error_mV','mean'),generic_mae=('generic_abs_error_mV','mean'),
    penalty_mV=('error_penalty_mV','mean'),persistent_losses=('fraction_checkpoints_worse',lambda x:int((x>=.75).sum()))).reset_index()
save(families,'families')
trajectory=p.groupby(['scope','Material class','checkpoint_step']).error_penalty_mV.mean().reset_index()
save(trajectory,'family_trajectory')
save(s[s['No.'].isin([187,188,189,602,625,648,671,703,779,780,781,782,783,784,752,753,282,288,331])], 'priority_samples')
save(s[s.Reference.eq(57)].sort_values('No.'),'reference57_chemistry_series')
save(s[s['Material class'].eq('NiCrMo Alloy')].sort_values(['Cr','Mo','CP_temp']),'nicrmo_chemistry_series')

data=pd.read_csv(ROOT/'corrosion_datasets/datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/epit_processed.csv')
data['input_group']=pd.util.hash_pandas_object(data[list(EPIT_EXPANDED_FEATURE_COLUMNS)],index=False)
annotated=s.merge(data[['task_row_index','input_group']],on='task_row_index',validate='one_to_one')
duplicates=[]
for key,g in annotated.groupby('input_group'):
    if len(g)<2:continue
    source=data[data.input_group.eq(key)]
    assert len(source[list(EPIT_EXPANDED_FEATURE_COLUMNS)].drop_duplicates())==1
    duplicates.append({'input_group':str(key),'n':len(g),'source_numbers':','.join(map(str,sorted(g['No.']))),
        'y_min':g.observed_mV.min(),'y_max':g.observed_mV.max(),'target_range':g.observed_mV.max()-g.observed_mV.min(),
        'best_common_constant_mae_mV':np.abs(g.observed_mV-g.observed_mV.median()).mean(),
        'informed_mae':g.informed_abs_error_mV.mean(),'generic_mae':g.generic_abs_error_mV.mean(),
        'penalty_mV':g.error_penalty_mV.mean(),'scopes':','.join(sorted(g.scope.unique()))})
save(pd.DataFrame(duplicates).sort_values('target_range',ascending=False),'identical_inputs')
al=annotated[(annotated.scope=='outer_fold_1')&(annotated['Material class']=='Al Alloy')]
sensitivity=[]
for name,g in [('all_Al',al),('exclude_reference32',al[al.Reference.ne(32)]),
               ('exclude_Al69p5_Fe7p99',al[~(al.Al.eq(69.5)&al.Fe.eq(7.99))])]:
    sensitivity.append({'subset':name,'n':len(g),'sources':g.Reference.nunique(),'penalty_mV':g.error_penalty_mV.mean()})
sensitivity.append({'subset':'equal_weight_unique_inputs','n':al.input_group.nunique(),'sources':al.Reference.nunique(),
                    'penalty_mV':al.groupby('input_group').error_penalty_mV.mean().mean()})
save(pd.DataFrame(sensitivity),'al_sensitivity')

assignment=pd.read_csv(ROOT/'corrosion_datasets/datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/split_assignments.csv').set_index('task_row_index')
original=ROOT/'corrosion_datasets/analysis/epit_pipeline/targetv5_four_ratios_through_8000_cv5_cpu/results.json'
memberships=json.loads(original.read_text())['fold_membership']
seen={}
for m in memberships.values():
    context_groups=set(assignment.loc[m['context_task_row_indices'],'composition_group'])
    for row in m['query_task_row_indices']:
        seen[row]=assignment.loc[row,'composition_group'] in context_groups
s['same_composition_in_context']=s.task_row_index.map(seen)
save(s.groupby(['scope','Material class','same_composition_in_context']).agg(
    n=('row_id','size'),penalty_mV=('error_penalty_mV','mean')).reset_index(),'context_support')

fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
colors=dict(zip(['Fe Alloy','Al Alloy','HEA','NiCrMo Alloy','Other'],plt.get_cmap('tab10').colors))
for ax,scope,title in zip(axes,['outer_fold_1','exploratory_development_folds'],['Original outer fold 1','Exploratory folds 2–5']):
    for family,g in trajectory[trajectory.scope.eq(scope)].groupby('Material class'):
        ax.plot(g.checkpoint_step,g.error_penalty_mV,marker='o',markersize=3,color=colors[family],label=family)
    ax.axhline(0,color='black',lw=.8);ax.set_title(title);ax.set_xlabel('Checkpoint step');ax.grid(alpha=.2)
axes[0].set_ylabel('100% informed MAE minus generic MAE (mV)')
axes[1].legend(fontsize=8)
fig.suptitle('Positive = informed worse; family trajectories reveal effects hidden by the overall mean')
fig.tight_layout();fig.savefig(OUT/'family_trajectory.png',dpi=180);plt.close(fig)
print(f'Validated 760 samples × 11 checkpoints; wrote evidence to {OUT}')
