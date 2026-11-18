"""Explore local formula behavior; never refit, train, or change a recipe.
Formulas were fit on all rows: measured-neighbour checks are descriptive, not
held-out evaluation. Identical missing values do not establish identical conditions.
"""
import argparse
from copy import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scripts.epit_pipeline.artifact_hashes import sha256_file
from tabicl.prior.epit_schema import EPIT_COMPOSITION_ELEMENTS
ROOT=Path(__file__).resolve().parent
ELEMENTS=list(EPIT_COMPOSITION_ELEMENTS)

def run(dataset):
    if dataset=='epit':
        from tabicl.prior.epit_selected_pairs import SelectedPairSampler
        from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_ENVIRONMENT_COLUMNS as CONTEXT
        recipe=Path('corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k')
        sampler=SelectedPairSampler(recipe)
        frame=sampler.reviewed.rows.copy()
        source='Reference'
        extra=['Heat treatment','Microstructures','[Cl-] Test Method']
    else:
        from tabicl.prior.soccol_selected_pairs import SelectedPairSampler
        from tabicl.prior.soccol_reviewed_schema import SOCCOL_REVIEWED_CONTEXT_COLUMNS as CONTEXT
        recipe=Path('corrosion_datasets/analysis/soccol_pipeline/selected_pairs_allaccepted_generic10k')
        sampler=SelectedPairSampler(recipe)
        frame=sampler.rows.copy()
        source='source'
        extra=[c for c in frame if c.startswith('raw_') and c!='raw_workbook_row']
    out=ROOT/dataset;out.mkdir(parents=True,exist_ok=True)
    frame=frame.reset_index(drop=True)
    pairs=pd.read_csv(recipe/'all_pairs.csv')
    retained=pairs[pairs.retained].copy()
    assert len(retained)==len(sampler.pairs)
    y=pairs.groupby('task_row_index').observed_mV.first()
    predictions=pairs.pivot(index='task_row_index',columns='formula',values='prediction_mV')
    keeps=pairs.pivot(index='task_row_index',columns='formula',values='retained')
    rowids=frame.task_row_index.to_numpy()
    amounts=frame[ELEMENTS].to_numpy(dtype=float)
    normalized=amounts*100/amounts.sum(axis=1,keepdims=True)
    positive=amounts>0
    # Exact model context + source + original treatment/method details, where present.
    groupcols=list(dict.fromkeys([source,*CONTEXT,*extra]))
    keys=frame[groupcols].astype(object).where(frame[groupcols].notna(),'<MISSING>')
    groupids=pd.util.hash_pandas_object(keys,index=False).to_numpy()
    core=frame[['CP_temp','CP_pH','CP_Cl']].notna().all(axis=1).to_numpy() & frame[source].notna().to_numpy()
    corepositions=np.flatnonzero(core)
    group_series=pd.Series(groupids[corepositions],index=corepositions)
    directed=[];replicates=[]
    # Collapse recorded replicates for the primary change comparison, while keeping
    # unaggregated results as a sensitivity check. No outcome enters neighbour choice.
    composition_keys=[tuple(np.round(v,8)) for v in normalized]
    aggregate_groups={}
    for pos in corepositions:
        aggregate_groups.setdefault((int(groupids[pos]),composition_keys[pos]),[]).append(pos)
    aggregate={}
    for positions in aggregate_groups.values():
        ids=rowids[positions]
        values=(float(y.loc[ids].median()),predictions.loc[ids].median(axis=0),len(ids))
        for i in ids:aggregate[int(i)]=values
    for gid,positions in group_series.groupby(group_series,sort=False):
        pos=positions.index.to_numpy()
        if len(pos)<2:continue
        for p in pos:
            candidates=pos[pos!=p]
            half_l1=np.abs(normalized[candidates]-normalized[p]).sum(axis=1)/2
            same_support=np.all(positive[candidates]==positive[p],axis=1)
            # Record target variability in rows indistinguishable in recorded inputs.
            for q in candidates[half_l1<1e-7]:
                if p<q:
                    replicates.append({'parent':int(rowids[p]),'neighbour':int(rowids[q]),'source':str(frame.iloc[p][source]),
                        'absolute_target_difference_mV':abs(y.loc[rowids[q]]-y.loc[rowids[p]])})
            valid=(half_l1>1e-7)&same_support
            if not valid.any():continue
            candidates=candidates[valid]; half_l1=half_l1[valid]
            support=positive[p]
            logratio=np.log(normalized[candidates][:,support])-np.log(normalized[p,support])
            logratio-=logratio.mean(axis=1,keepdims=True)
            distance=np.sqrt(np.mean(logratio**2,axis=1))
            order=np.argsort(distance,kind='stable')
            # Use nearest three distinct measured compositions, avoiding replicated-row weighting.
            seen=set();rank=0
            for idx in order:
                q=candidates[idx]
                composition=tuple(np.round(normalized[q],8))
                if composition in seen:continue
                seen.add(composition);rank+=1
                if rank>3:break
                parent,neighbour=int(rowids[p]),int(rowids[q])
                for name in predictions.columns[keeps.loc[parent].to_numpy(dtype=bool)]:
                    observed=float(y.loc[neighbour]-y.loc[parent])
                    predicted=float(predictions.loc[neighbour,name]-predictions.loc[parent,name])
                    directed.append({'parent':parent,'neighbour':neighbour,'formula':name,'source':str(frame.iloc[p][source]),
                        'context_group':str(gid),'neighbour_rank':rank,'clr_rms_distance':float(distance[idx]),
                        'composition_exchange_wt_pct':float(half_l1[idx]),
                        'observed_delta_mV':observed,'formula_delta_mV':predicted,
                        'delta_error_mV':abs(predicted-observed),'no_change_error_mV':abs(observed),
                        'improvement_over_no_change_mV':abs(observed)-abs(predicted-observed),
                        'neighbour_retained_same_formula':bool(keeps.loc[neighbour,name]),
                        'observed_parent_mV':float(y.loc[parent]),'observed_neighbour_mV':float(y.loc[neighbour]),
                        'formula_parent_mV':float(predictions.loc[parent,name]),'formula_neighbour_mV':float(predictions.loc[neighbour,name])})
    neighbours=pd.DataFrame(directed)
    neighbours['observed_delta_raw_mV']=neighbours.observed_delta_mV
    neighbours['formula_delta_raw_mV']=neighbours.formula_delta_mV
    neighbours['delta_error_raw_mV']=neighbours.delta_error_mV
    neighbours['no_change_error_raw_mV']=neighbours.no_change_error_mV
    neighbours['observed_delta_mV']=[aggregate[q][0]-aggregate[p][0] for p,q in neighbours[['parent','neighbour']].to_numpy()]
    neighbours['formula_delta_mV']=[aggregate[q][1][f]-aggregate[p][1][f] for p,q,f in neighbours[['parent','neighbour','formula']].to_numpy()]
    neighbours['parent_replicates']=[aggregate[p][2] for p in neighbours.parent]
    neighbours['neighbour_replicates']=[aggregate[q][2] for q in neighbours.neighbour]
    neighbours['delta_error_mV']=abs(neighbours.formula_delta_mV-neighbours.observed_delta_mV)
    neighbours['no_change_error_mV']=abs(neighbours.observed_delta_mV)
    neighbours['improvement_over_no_change_mV']=neighbours.no_change_error_mV-neighbours.delta_error_mV
    neighbours.to_csv(out/'measured_neighbour_comparisons.csv',index=False)
    pd.DataFrame(replicates).to_csv(out/'identical_recorded_input_pairs.csv',index=False)
    summaries=[]
    for radius in [.15,.3,.6]:
        for k in [1,3]:
            selected=neighbours[(neighbours.clr_rms_distance<=radius)&(neighbours.neighbour_rank<=k)]
            for scope,view in [('all_neighbours',selected),('both_retained',selected[selected.neighbour_retained_same_formula])]:
                for name,part in [('ALL',view),*view.groupby('formula')]:
                    if part.empty:continue
                    # Pair/formula observations are dependent; report counts and descriptive metrics only.
                    clear=part[part.observed_delta_mV.abs()>=25]
                    summaries.append({'dataset':dataset,'radius':radius,'k':k,'scope':scope,'formula':name,'comparisons':len(part),
                        'parents':part.parent.nunique(),'parent_formula_pairs':len(part[['parent','formula']].drop_duplicates()),
                        'unique_measured_pairs':len({tuple(sorted(z)) for z in part[['parent','neighbour']].to_numpy()}),
                        'studies':part.source.nunique(),'context_groups':part.context_group.nunique(),
                        'mean_delta_error_raw_mV':part.delta_error_raw_mV.mean(),'no_change_error_raw_mV':part.no_change_error_raw_mV.mean(),
                        'mean_delta_error_mV':part.delta_error_mV.mean(),'no_change_mae_mV':part.no_change_error_mV.mean(),
                        'mean_improvement_over_no_change_mV':part.improvement_over_no_change_mV.mean(),
                        'source_macro_improvement_mV':part.groupby('source').improvement_over_no_change_mV.mean().mean(),
                        'clear_comparisons':len(clear),'direction_correct_pct':100*(clear.formula_delta_mV*clear.observed_delta_mV>0).mean(),
                        'direction_opposite_pct':100*(clear.formula_delta_mV*clear.observed_delta_mV<0).mean(),
                        'predicted_delta_abs_median':part.formula_delta_mV.abs().median(),
                        'observed_delta_abs_median':part.observed_delta_mV.abs().median()})
    pd.DataFrame(summaries).to_csv(out/'local_relationship_summary.csv',index=False)
    # All strengths use identical parent/formula draws and common Gaussian variates.
    # Marginal sampling over retained pairs matches single-rule tables exactly.
    n=50000;seed=20261118
    zero=sampler.sample(n,perturb_strength=0,random_state=seed)
    expected=sampler.pairs.iloc[zero.pair_indices]
    np.testing.assert_array_equal(zero.parent_indices,expected.task_row_index)
    np.testing.assert_allclose(zero.target_mV,expected.prediction_mV,atol=1e-6,rtol=1e-8)
    position=pd.Series(np.arange(len(frame)),index=rowids)
    original=amounts[position.loc[zero.parent_indices].to_numpy()]
    original_normalized=original*100/original.sum(axis=1,keepdims=True)
    sensitivity=[];details=[]
    neighbourdist=neighbours[neighbours.neighbour_rank==1].groupby('parent').clr_rms_distance.first()
    for strength in [0,.1,.3]:
        batch=zero if strength==0 else sampler.sample(n,perturb_strength=strength,random_state=seed)
        np.testing.assert_array_equal(batch.pair_indices,zero.pair_indices)
        np.testing.assert_array_equal(batch.formulas,zero.formulas)
        composition=batch.physical[ELEMENTS].to_numpy(dtype=float)
        np.testing.assert_array_equal(composition==0,original==0)
        pd.testing.assert_frame_equal(batch.physical[list(CONTEXT)],zero.physical[list(CONTEXT)])
        if strength:np.testing.assert_allclose(composition.sum(axis=1),100,atol=1e-10)
        diff=composition-original
        logs=np.zeros_like(composition)
        np.log(np.divide(composition,original,out=np.ones_like(composition),where=original>0),out=logs)
        active=original>0
        meanlog=logs.sum(axis=1)/active.sum(axis=1)
        clr=np.where(active,logs-meanlog[:,None],0)
        rms=np.sqrt((clr**2).sum(axis=1)/active.sum(axis=1))
        part=pd.DataFrame({'draw':np.arange(n),'strength':strength,'parent':batch.parent_indices,'formula':batch.formulas,
            'pair_index':batch.pair_indices,'parent_formula_target_mV':zero.target_mV,'synthetic_target_mV':batch.target_mV,
            'target_change_mV':batch.target_mV-zero.target_mV,'abs_target_change_mV':abs(batch.target_mV-zero.target_mV),
            'composition_exchange_wt_pct':abs(diff).sum(axis=1)/2,'max_element_change_wt_pct':abs(diff).max(axis=1),
            'clr_rms_distance':rms,'largest_element_changed':np.argmax(original,axis=1)!=np.argmax(composition,axis=1)})
        part['nearest_measured_distance']=part.parent.map(neighbourdist)
        details.append(part)
        for name,view in [('ALL',part),*part.groupby('formula')]:
            sensitivity.append({'dataset':dataset,'strength':strength,'formula':name,'draws':len(view),
                'exchange_median_wt_pct':view.composition_exchange_wt_pct.median(),
                'exchange_p90_wt_pct':view.composition_exchange_wt_pct.quantile(.9),
                'max_element_change_median_wt_pct':view.max_element_change_wt_pct.median(),
                'target_change_abs_median_mV':view.abs_target_change_mV.median(),
                'target_change_abs_p90_mV':view.abs_target_change_mV.quantile(.9),
                'target_change_abs_p99_mV':view.abs_target_change_mV.quantile(.99),
                'target_change_gt100_pct':100*(view.abs_target_change_mV>100).mean(),
                'target_change_gt250_pct':100*(view.abs_target_change_mV>250).mean(),
                'largest_element_changed_pct':100*view.largest_element_changed.mean(),
                'clr_rms_median':view.clr_rms_distance.median()})
    pd.DataFrame(sensitivity).to_csv(out/'perturbation_summary.csv',index=False)
    pd.concat(details).to_csv(out/'synthetic_draws.csv.gz',index=False,compression='gzip')
    primary=neighbours[(neighbours.clr_rms_distance<=.3)&(neighbours.neighbour_rank<=3)]
    primary.groupby('source').agg(comparisons=('parent','size'),parents=('parent','nunique'),
        delta_error=('delta_error_mV','mean'),no_change_error=('no_change_error_mV','mean'),
        improvement=('improvement_over_no_change_mV','mean')).to_csv(out/'per_study.csv')
    examples=primary[(primary.observed_delta_mV*primary.formula_delta_mV<0)&(primary.observed_delta_mV.abs()>=25)]
    examples.sort_values('delta_error_mV',ascending=False).drop_duplicates(['parent','neighbour']).head(20).to_csv(out/'opposite_direction_examples.csv',index=False)
    info={'dataset':dataset,'scope':'Descriptive all-data recipe diagnostic; formulas fitted on these measurements, not independent evaluation',
        'data_identity':sampler.identity,'script_sha256':sha256_file(Path(__file__)),
        'matching_columns':groupcols,'require_nonmissing':['CP_temp','CP_pH','CP_Cl',source],
        'same_nonzero_elements':True,'distance':'RMS centered log-ratio difference over common positive elements',
        'neighbours':'Nearest three distinct normalized compositions under exact recorded context and same source. Primary changes use median targets/formula predictions across exact normalized-composition replicates; raw-row sensitivity retained.',
        'n_eligible':len(frame),'n_core_known':int(core.sum()),'n_retained_parents':retained.task_row_index.nunique(),
        'n_retained_parent_pairs':len(retained),'n_parents_with_local_matches':primary.parent.nunique(),
        'replicate_pair_count':len(replicates),'replicate_abs_difference_median_mV':float(pd.DataFrame(replicates).absolute_target_difference_mV.median()) if replicates else None,
        'synthetic_draws_per_strength':n,'strengths':[0,.1,.3],'seed':seed,
        'training_runs_launched':False,'limitations':['Missing context may hide differences.','Real-neighbour changes are observational.','Zero perturbation changes synthetic diversity as well as target locality.','Matching known physical conditions does not establish causal composition effects.']}
    (out/'identity.json').write_text(json.dumps(info,indent=2))
    print(dataset, 'DONE', 'matched retained parents',primary.parent.nunique(),'of',retained.task_row_index.nunique(),flush=True)
    print(pd.DataFrame(summaries).query('radius==.3 and k==3 and scope=="all_neighbours" and formula=="ALL"').to_string(index=False),flush=True)
    print(pd.DataFrame(sensitivity).query('formula=="ALL"').to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['epit','soccol'],required=True)
    run(p.parse_args().dataset)
