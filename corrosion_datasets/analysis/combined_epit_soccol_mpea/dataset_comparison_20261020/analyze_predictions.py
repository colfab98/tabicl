"""Post-hoc study-aware error analysis of fixed-fold predictions."""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reproduce import OUT, TARGET, rho, corr, save

LABELS=["epit_shared_v2","soccol_shared_v2","combined_corrpfn_v2"]
NAMES={"epit_shared_v2":"EPIT","soccol_shared_v2":"Soccol","combined_corrpfn_v2":"COMB v2"}
RNG=np.random.default_rng(20261020)


def prediction_plot(d):
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained")
    for ax,label in zip(axes,LABELS):
        scatter=ax.scatter(d[TARGET],d[label],s=19,alpha=.65,c=d.soccol_unsupported_mass_wt_pct,cmap="viridis",vmin=0,vmax=65)
        ax.plot([-1000,2400],[-1000,2400],ls="--",color="#555555",lw=1)
        ax.set_xlim(-1000,2400);ax.set_ylim(-1000,2400)
        ax.set_xlabel("Observed potential (mV)");ax.set_ylabel("Predicted potential (mV)")
        ax.set_title(f"{NAMES[label]}: pooled rho={rho(d[TARGET],d[label]):.3f}")
    fig.colorbar(scatter,ax=axes,shrink=.8,label="Soccol-unsupported mass (wt%)")
    fig.suptitle("Step 5000, original frozen MPEA folds")
    fig.savefig(OUT/"prediction_scatter.png",dpi=170);plt.close(fig)


def main():
    descriptors=pd.read_csv(OUT/"mpea_row_descriptors.csv")
    predictions=pd.read_csv(OUT/"model_diagnostics/predictions.csv")
    original=predictions[predictions.variant.eq("original")]
    assert len(original)==267*3 and original.groupby("model").row_id.nunique().eq(267).all()
    wide=original.pivot(index="row_id",columns="model",values="y_pred")
    d=descriptors.merge(wide,on="row_id",validate="one_to_one")
    folds=original[original.model.eq(LABELS[0])][["row_id","outer_fold"]]
    d=d.merge(folds,on="row_id",validate="one_to_one")
    for label in LABELS:
        d[f"{label}_absolute_error"]=np.abs(d[label]-d[TARGET])
    d["Cr_3p3Mo_common"]=d.Cr+3.3*d.Mo
    subsets={
        "all":np.ones(len(d),bool),
        "NaCl_seawater":d.environment_family.eq("NaCl/seawater"),
        "chloride_0p6M":d.CP_Cl.eq(.6),
        "no_elements_omitted_from_shared_schema":d.both_unsupported_mass_wt_pct.lt(.01),
        "Soccol_unsupported_elements_present":d.soccol_unsupported_mass_wt_pct.ge(.01),
        "Soccol_unsupported_elements_absent":d.soccol_unsupported_mass_wt_pct.lt(.01),
        "target_le1000mV":d[TARGET].le(1000),
        "target_below_zero":d[TARGET].lt(0),
    }
    records,bootstrap=[],[]
    for name,mask in subsets.items():
        g=d[mask].reset_index(drop=True)
        studies=g.groupby("source_reference").indices
        keys=list(studies)
        for label in LABELS:
            records.append({
                "subset":name,"model":label,"rows":len(g),"studies":len(studies),
                "pooled_prediction_spearman":rho(g[TARGET],g[label]),
                "mae":np.mean(np.abs(g[TARGET]-g[label])),
                "rmse":np.sqrt(np.mean((g[TARGET]-g[label])**2)),
                "mean_residual_pred_minus_true":np.mean(g[label]-g[TARGET]),
            })
        boot_rho,boot_mae=[],[]
        for _ in range(1499):
            ix=np.concatenate([studies[keys[i]] for i in RNG.integers(0,len(keys),len(keys))])
            h=g.iloc[ix]
            boot_rho.append(rho(h[TARGET],h[LABELS[0]])-rho(h[TARGET],h[LABELS[1]]))
            boot_mae.append(np.mean(h[f"{LABELS[1]}_absolute_error"]-h[f"{LABELS[0]}_absolute_error"]))
        valid=np.asarray(boot_rho)[np.isfinite(boot_rho)]
        bootstrap.append({
            "subset":name,"rows":len(g),"studies":len(keys),
            "epit_minus_soccol_pooled_rho":rho(g[TARGET],g[LABELS[0]])-rho(g[TARGET],g[LABELS[1]]),
            "rho_difference_q025":np.quantile(valid,.025),"rho_difference_q975":np.quantile(valid,.975),
            "soccol_minus_epit_mae":np.mean(g[f"{LABELS[1]}_absolute_error"]-g[f"{LABELS[0]}_absolute_error"]),
            "mae_difference_q025":np.quantile(boot_mae,.025),"mae_difference_q975":np.quantile(boot_mae,.975),
        })
    save("model_subgroup_metrics",records);save("paired_study_bootstrap",bootstrap)
    study=[]
    for reference,g in d.groupby("source_reference"):
        row={"source_reference":reference,"rows":len(g)}
        for label in LABELS:
            row[label+"_mae"]=g[label+"_absolute_error"].mean()
            row[label+"_within_study_rho"]=rho(g[TARGET],g[label])
        row["soccol_minus_epit_mae"]=row[LABELS[1]+"_mae"]-row[LABELS[0]+"_mae"]
        study.append(row)
    save("model_per_study",study)
    cues=[]
    for feature in ["Cr","Mo","Ti","Nb","Cu","Co","Ni","Al","Fe","Cr_3p3Mo_common"]:
        row={"feature":feature,"observed_target_spearman":rho(d[feature],d[TARGET])}
        for label in LABELS:
            row[label+"_prediction_spearman"]=rho(d[feature],d[label])
        cues.append(row)
    save("model_composition_associations",cues)
    within=[]
    for label in LABELS:
        yr,pr=rankdata(d[TARGET]),rankdata(d[label])
        tmp=pd.DataFrame({"yr":yr,"pr":pr,"source":d.source_reference})
        residual=tmp[["yr","pr"]]-tmp.groupby("source")[["yr","pr"]].transform("mean")
        record={"model":label,"pooled_rho":rho(d[TARGET],d[label]),
                "within_study_centered_rank_correlation":corr(residual.yr,residual.pr)}
        tmp["fold"]=d.outer_fold
        for key,name in [(["source"],"study"),(["source","fold"],"study_and_fold")]:
            residual=tmp[["yr","pr"]]-tmp.groupby(key)[["yr","pr"]].transform("mean")
            record[f"{name}_centered_global_rank_correlation"]=corr(residual.yr,residual.pr)
            ranked=tmp.groupby(key)[["yr","pr"]].rank(pct=True)
            ranked[key]=tmp[key]
            residual=ranked[["yr","pr"]]-ranked.groupby(key)[["yr","pr"]].transform("mean")
            record[f"{name}_centered_within_group_rank_correlation"]=corr(residual.yr,residual.pr)
        within.append(record)
    save("model_within_study_ranks",within)
    # Leave one entire study out of the metric, not a refit or a new evaluation split.
    leave=[]
    for reference in d.source_reference.unique():
        g=d[d.source_reference.ne(reference)]
        leave.append({"excluded_study":reference,"rows_removed":len(d)-len(g),
                      "epit_minus_soccol_rho":rho(g[TARGET],g[LABELS[0]])-rho(g[TARGET],g[LABELS[1]])})
    save("leave_one_study_out_metric_sensitivity",leave)
    save("mpea_predictions_with_metadata",d)
    prediction_plot(d)
    metrics=pd.read_csv(OUT/"model_diagnostics/fold_metrics.csv")
    fig,ax=plt.subplots(figsize=(7.5,4.5),layout="constrained")
    for i,label in enumerate(LABELS):
        sub=metrics[metrics.model.eq(label)].pivot(index="outer_fold",columns="variant",values="spearman")
        ax.scatter(np.full(5,i-.14),sub.original,color="#2879b5",alpha=.65,s=30)
        ax.scatter(np.full(5,i+.14),sub.hide_soccol_unsupported_composition,color="#d76c2b",alpha=.65,s=30)
        for _,row in sub.iterrows():
            ax.plot([i-.14,i+.14],[row.original,row.hide_soccol_unsupported_composition],color="#999999",alpha=.6)
        ax.plot(i-.14,sub.original.mean(),marker="_",markersize=24,markeredgewidth=3,color="#2879b5",
                label="Original inputs" if i==0 else None)
        ax.plot(i+.14,sub.hide_soccol_unsupported_composition.mean(),marker="_",markersize=24,
                markeredgewidth=3,color="#d76c2b",label="11 unsupported composition columns hidden" if i==0 else None)
    ax.set_xticks(range(3),[NAMES[n] for n in LABELS]);ax.set_ylabel("Outer-fold Spearman");ax.set_ylim(0,1)
    ax.legend(fontsize=9);ax.set_title("Identical query rows and labels; columns hidden in context and query")
    fig.savefig(OUT/"composition_ablation.png",dpi=170);plt.close(fig)
    (OUT/"prediction_analysis_metadata.json").write_text(json.dumps({
        "bootstrap_replicates":1499,"bootstrap_unit":"source_reference",
        "bootstrap_limit":"Conditional on existing fitted predictions; no refitting; exploratory pointwise intervals.",
        "subgroup_limit":"Pooled cross-fold correlations; not equal-fold means; subsets are descriptive, not matched causal controls.",
        "checkpoint_choice":"Step 5000 was fixed as latest checkpoint shared by all trained models, not chosen for best score.",
    },indent=2))
    print(pd.DataFrame(bootstrap).round(3).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
