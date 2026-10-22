"""Descriptive, post-hoc comparison of EPIT, Soccol, and frozen MPEA rows.

Does not train/select a predictor or modify any existing dataset or partition.
Unknowns remain missing except in explicitly labelled diagnostic projections.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, rankdata, spearmanr, wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabicl.prior.combined_corrosion import COMBINED_COMPOSITION_COLUMNS, EPIT_FEATURE_COLUMNS
from tabicl.prior.soccol_schema import SOCCOL_COMPOSITION_COLUMNS, SOCCOL_FEATURE_COLUMNS

OUT = Path(__file__).resolve().parent
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
TARGET = "reported_pitting_potential_mV"
ELEMENTS = list(COMBINED_COMPOSITION_COLUMNS)
SOURCES = ["EPIT", "Soccol", "MPEA"]
COLORS = {"EPIT": "#2879b5", "Soccol": "#d76c2b", "MPEA": "#23856c"}
RNG = np.random.default_rng(20261020)


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.unique(x[good]).size < 2 or np.unique(y[good]).size < 2:
        return np.nan
    return float(spearmanr(x[good], y[good]).statistic)


def corr(x, y, weights=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    den = np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))
    return float(np.sum(w * x * y) / den) if den > 1e-12 else np.nan


def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return dict(n=len(x), mean=x.mean(), std=x.std(), minimum=x.min(),
                q05=np.quantile(x, .05), q25=np.quantile(x, .25), median=np.median(x),
                q75=np.quantile(x, .75), q95=np.quantile(x, .95), maximum=x.max())


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    return frame


def load():
    registry = json.loads((DATA / "shared_feature_registry.json").read_text())
    c = pd.read_csv(DATA / "epit_soccol_context.csv")
    q = pd.read_csv(DATA / "mpea_queries.csv")
    y = pd.read_csv(DATA / "mpea_scoring_targets.csv")
    q = q.merge(y[["row_id", TARGET]], on="row_id", validate="one_to_one")
    d = pd.concat([c, q], ignore_index=True)
    assert len(d) == 5054 and d.row_id.is_unique and d[TARGET].notna().all()
    audit = pd.read_csv(ROOT / "corrosion_datasets/datasets/mpea_corrosion/processed/mpea_row_audit.csv")
    audit = audit.set_index("row_id").loc[q.row_id].reset_index()
    return d, audit, registry


def summarize_data(d, audit, registry):
    summaries, features, categories, repeats, studies = [], [], [], [], []
    columns = registry["feature_columns_in_order"]
    numeric = registry["numeric_feature_columns"]
    for name, g in d.groupby("source_dataset", sort=False):
        studies_count = g.source_reference.value_counts()
        vals = g[ELEMENTS].fillna(0)
        count = len(g)
        summaries.append({
            "dataset": name, "rows": count, "studies": len(studies_count),
            "effective_studies_inverse_concentration": 1 / np.sum((studies_count / count) ** 2),
            "largest_study_rows": studies_count.max(),
            "top5_studies_row_fraction": studies_count.head(5).sum() / count,
            "unique_feature_rows": len(g[columns].drop_duplicates()),
            "unique_composition_projections": len(vals.round(2).drop_duplicates()),
            "mean_observed_features": g[columns].notna().sum(axis=1).mean(),
            "mean_observed_noncomposition": g[columns[24:]].notna().sum(axis=1).mean(),
            "Fe_ge50_fraction": g.Fe.ge(50).mean(),
            "max_reported_element_median_wt_pct": vals.max(axis=1).median(),
            "n_elements_ge5wt_median": vals.ge(5).sum(axis=1).median(),
            **{f"target_{k}": v for k, v in describe(g[TARGET]).items()},
        })
        for feature in numeric:
            features.append({
                "dataset": name, "feature": feature, "unit": registry["feature_units"][feature],
                "missing_fraction": g[feature].isna().mean(),
                "positive_fraction_all_rows": g[feature].gt(0).mean(),
                "n_unique_observed": g[feature].nunique(),
                **describe(g[feature]),
            })
        for feature in registry["categorical_feature_columns"]:
            for value, n in g[feature].fillna("<MISSING>").value_counts().items():
                categories.append({"dataset": name, "feature": feature, "value": value, "rows": n, "fraction": n/count})
        keys = pd.util.hash_pandas_object(g[columns], index=False)
        for _, h in g.assign(feature_key=keys.to_numpy()).groupby("feature_key"):
            if len(h) > 1:
                repeats.append({"dataset": name, "rows": len(h), "studies": h.source_reference.nunique(),
                                "target_range": h[TARGET].max()-h[TARGET].min(),
                                "target_std": h[TARGET].std(), "row_ids": ";".join(h.row_id)})
        for study, h in g.groupby("source_reference"):
            studies.append({"dataset": name, "source_reference": study, "rows": len(h),
                            **{f"target_{k}": v for k, v in describe(h[TARGET]).items()}})
    save("dataset_summary", summaries)
    save("feature_distributions", features)
    save("categorical_distributions", categories)
    save("identical_input_groups", repeats)
    save("study_summary", studies)
    q = d[d.source_dataset.eq("MPEA")].copy()
    absent = sorted(set(ELEMENTS) - set(SOCCOL_COMPOSITION_COLUMNS))
    q["soccol_unsupported_mass_wt_pct"] = q[absent].fillna(0).sum(axis=1)
    q["both_unsupported_mass_wt_pct"] = audit.unsupported_elements_wt_percent.to_numpy()
    q["both_unsupported_elements"] = audit.unsupported_elements_present.to_numpy()
    q["alloy_name_raw"] = audit.alloy_name_raw.to_numpy()
    q["environment_raw"] = audit.test_environment_raw.to_numpy()
    text = q.environment_raw.fillna("").str.lower()
    q["environment_family"] = np.select(
        [text.str.contains("nacl|seawater|sea water"), text.str.contains("h2so4"),
         text.str.contains("hno3"), text.str.contains("naoh"), text.str.contains("pbs|phosphate")],
        ["NaCl/seawater", "sulfuric acid", "nitric acid", "sodium hydroxide", "PBS"],
        default="other/unresolved",
    )
    save("mpea_row_descriptors", q)
    save("mpea_environment_targets", [
        {"environment_family": k, "rows": len(g), "studies": g.source_reference.nunique(),
         "chloride_reported_fraction": g.CP_Cl.notna().mean(),
         **{f"target_{a}": b for a,b in describe(g[TARGET]).items()}}
        for k, g in q.groupby("environment_family")
    ])
    return q


def compare_distributions(d, registry):
    records = []
    for first, second in [("EPIT", "Soccol"), ("EPIT", "MPEA"), ("Soccol", "MPEA")]:
        a, b = d[d.source_dataset.eq(first)], d[d.source_dataset.eq(second)]
        for feature in registry["numeric_feature_columns"]:
            x, y = a[feature].dropna().to_numpy(), b[feature].dropna().to_numpy()
            if len(x) < 5 or len(y) < 5:
                continue
            spread = np.quantile(x,.95) - np.quantile(x,.05)
            records.append({
                "source": first, "comparison": second, "feature": feature,
                "source_observed_n": len(x), "comparison_observed_n": len(y),
                "ks_statistic": ks_2samp(x,y).statistic,
                "wasserstein_source_q95_q05_scaled": wasserstein_distance(x,y)/spread if spread else np.nan,
                "comparison_outside_source_minmax_fraction": np.mean((y<x.min()) | (y>x.max())),
                "comparison_outside_source_q05q95_fraction": np.mean((y<np.quantile(x,.05)) | (y>np.quantile(x,.95))),
            })
    save("distribution_distances", records)


def relation_summary(d, registry):
    records, contrasts = [], []
    selected = ["Cr", "Mo", "Ni", "Fe", "Co", "Al", "Ti", "Nb", "Cu", "Mn", "CP_Cl", "CP_temp", "CP_pH"]
    for source in SOURCES:
        g = d[d.source_dataset.eq(source)].copy()
        g["Cr_3p3Mo_1p65W_noN"] = g.Cr.fillna(0)+3.3*g.Mo.fillna(0)+1.65*g.W.fillna(0)
        g["Cr_3p3Mo_common"] = g.Cr.fillna(0)+3.3*g.Mo.fillna(0)
        for feature in selected + ["Cr_3p3Mo_1p65W_noN", "Cr_3p3Mo_common"]:
            h = g[[feature,TARGET,"source_reference"]].dropna()
            if len(h) < 10 or h[feature].nunique() < 2:
                continue
            x,y = h[feature].to_numpy(),h[TARGET].to_numpy()
            xr,yr = rankdata(x),rankdata(y)
            labels,inverse = np.unique(h.source_reference.to_numpy(),return_inverse=True)
            sizes = np.bincount(inverse)
            wx = xr - (np.bincount(inverse,weights=xr)/sizes)[inverse]
            wy = yr - (np.bincount(inverse,weights=yr)/sizes)[inverse]
            boot = []
            indices = [np.flatnonzero(inverse == i) for i in range(len(labels))]
            for _ in range(399):
                ix=np.concatenate([indices[i] for i in RNG.integers(0,len(labels),len(labels))])
                boot.append(rho(x[ix],y[ix]))
            finite = np.asarray(boot)[np.isfinite(boot)]
            records.append({
                "dataset": source, "feature": feature, "rows": len(h), "studies": len(labels),
                "spearman": rho(x,y), "equal_study_weighted_rank_correlation": corr(xr,yr,1/sizes[inverse]),
                "within_study_centered_rank_correlation": corr(wx,wy),
                "study_bootstrap_rho_q025": np.quantile(finite,.025) if len(finite) else np.nan,
                "study_bootstrap_rho_q975": np.quantile(finite,.975) if len(finite) else np.nan,
            })
            # Within-study monotone contrast: only studies with >=3 rows and >=2 feature values.
            for study, sub in h.groupby("source_reference"):
                if len(sub) >= 3 and sub[feature].nunique() >= 2:
                    contrasts.append({"dataset": source,"feature":feature,"source_reference":study,
                                      "rows":len(sub),"spearman":rho(sub[feature],sub[TARGET])})
    save("target_relationships", records)
    save("within_study_relationships", contrasts)


def prior_and_overlap(d, q):
    """Compare actual informed sampling; no invented sampling of excluded EPIT families."""
    from tabicl.prior.epit_feature_generator import sample_epit_feature_rows
    from tabicl.prior.epit_feature_profile import load_epit_feature_profile
    from tabicl.prior.soccol_feature_generator import sample_soccol_feature_rows
    from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile
    config = json.loads((DATA/"combined_prior_config.json").read_text())
    profiles, details = {}, []
    sampled = {}
    for source in ["EPIT","Soccol"]:
        hp=config["regimes"][source]["fixed_hp"]
        if source=="EPIT":
            profile=load_epit_feature_profile()
            full=hp["pitting_composition_family_probs"]
            probs=[full[profile.composition_profile.families.index(f)] for f in profile.families]
            batch=sample_epit_feature_rows(12000,profile=profile,composition_family_probabilities=probs,
                                          perturb_strength=hp["pitting_composition_perturb_strength"],random_state=42)
            x=pd.DataFrame(batch.features,columns=EPIT_FEATURE_COLUMNS)
            profiles[source]={"effective_families":dict(zip(profile.families,np.asarray(probs)/sum(probs))),
                              "composition_template_count":profile.n_composition_templates,
                              "environment_row_count":profile.n_environment_rows}
        else:
            profile=load_soccol_feature_profile()
            batch=sample_soccol_feature_rows(12000,profile=profile,
                                            perturb_strength=hp["pitting_composition_perturb_strength"],random_state=42)
            x=pd.DataFrame(batch.features,columns=SOCCOL_FEATURE_COLUMNS)
            profiles[source]={"profile_rows":profile.n_rows,
                              "unique_profile_compositions":len(pd.DataFrame(
                                  profile.feature_values,columns=SOCCOL_FEATURE_COLUMNS
                              )[list(SOCCOL_COMPOSITION_COLUMNS)].drop_duplicates()),
                              "unique_sampled_perturbed_compositions":len(x[list(SOCCOL_COMPOSITION_COLUMNS)].drop_duplicates())}
        sampled[source]=x
        profiles[source].update(formula_weight=hp["informed_target_mix_weight"],
                                perturb_strength=hp["pitting_composition_perturb_strength"],
                                informed_mlp_tree_probabilities=hp["informed_mix_probs"],
                                numeric_native_columns=x.columns.tolist())
        for f in x:
            vals=q[f].dropna() if f in q else pd.Series(dtype=float)
            if not pd.api.types.is_numeric_dtype(vals):
                continue
            v=x[f].to_numpy()
            details.append({"source":source,"feature":f,**describe(v),
                            "nonzero_fraction":float(np.mean(v!=0)),
                            "mpea_observed_n":len(vals),
                            "mpea_outside_sample_q01q99_fraction":
                                np.mean((vals<np.quantile(v,.01))|(vals>np.quantile(v,.99))) if len(vals) else np.nan,
                            "mpea_outside_sample_minmax_fraction":
                                np.mean((vals<v.min())|(vals>v.max())) if len(vals) else np.nan})
    save("sampled_prior_feature_coverage",details)
    (OUT/"prior_profiles.json").write_text(json.dumps(profiles,indent=2))
    # Chemical projection distances. Missing/unsupported elements are ZERO ONLY HERE.
    # Therefore these are coverage diagnostics, not claims about true unreported chemistry.
    distances=[]
    projection=list(q[ELEMENTS].columns[q[ELEMENTS].notna().any()])
    for source in ["EPIT","Soccol"]:
        reference=d[d.source_dataset.eq(source)][projection].fillna(0).to_numpy()
        target=q[projection].fillna(0).to_numpy()
        pair=cdist(target,reference,metric="cityblock")
        prior=sampled[source].reindex(columns=projection).fillna(0).to_numpy()
        prior_pair=cdist(target,prior,metric="cityblock")
        equal_n=[]
        for _ in range(100):
            ix=RNG.choice(len(reference),size=500,replace=False)
            equal_n.append(pair[:,ix].min(axis=1))
        equal_n=np.mean(equal_n,axis=0)
        for i,row_id in enumerate(q.row_id):
            distances.append({"row_id":row_id,"source":source,
                              "nearest_real_L1_wt_pct":pair[i].min(),
                              "nearest_real_500row_subsample_mean_L1":equal_n[i],
                              "nearest_sampled_prior_L1_wt_pct":prior_pair[i].min()})
    save("composition_projection_distances",distances)
    common=["Fe","Cr","Ni","Mo","Ti","Nb","Mn","Si","V"]
    mpea_corr=q[common].corr(method="spearman")
    covariance=[]
    for source,x in sampled.items():
        prior_corr=x[common].corr(method="spearman")
        for i,a in enumerate(common):
            for b in common[:i]:
                covariance.append({
                    "source":source,"feature_a":a,"feature_b":b,
                    "prior_rho":prior_corr.loc[a,b],"mpea_rho":mpea_corr.loc[a,b],
                    "absolute_difference":abs(prior_corr.loc[a,b]-mpea_corr.loc[a,b]),
                })
    save("prior_composition_correlation_shift",covariance)
    return sampled


def plots(d,q,registry):
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout="constrained")
    for ax,feat in zip(axes.flat,["Fe","Co","Ni","Mo","CP_Cl",TARGET]):
        for source in SOURCES:
            values=d.loc[d.source_dataset.eq(source),feat].dropna().sort_values().to_numpy()
            if len(values):
                ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",color=COLORS[source],label=source)
        ax.set_title(feat if feat!=TARGET else "Reported pitting potential (mV)")
        ax.set_ylabel("Fraction of observed values")
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle("Observed distributions: absent measurements are excluded, not set to zero")
    fig.savefig(OUT/"distributions.png",dpi=170);plt.close(fig)
    key=["Fe","Cr","Ni","Mo","Co","Al","Cu","Ti","Nb","N","S","CP_temp","CP_pH","CP_Cl","scan_rate",
         "Prep_grinding_grit","CP_aeration","test_method","reference_type"]
    values=np.array([[d.loc[d.source_dataset.eq(n),f].notna().mean() for f in key] for n in SOURCES])
    fig,ax=plt.subplots(figsize=(13,3.3),layout="constrained")
    im=ax.imshow(values,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    ax.set_xticks(range(len(key)),key,rotation=55,ha="right");ax.set_yticks(range(3),SOURCES)
    for i in range(3):
        for j in range(len(key)):
            ax.text(j,i,f"{values[i,j]*100:.0f}",ha="center",va="center",
                    color="white" if values[i,j]>.6 else "black",fontsize=8)
    ax.set_title("Percent of rows with an observed value in the model's actual input table")
    fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(OUT/"coverage.png",dpi=170);plt.close(fig)
    rel=pd.read_csv(OUT/"target_relationships.csv")
    key=["Fe","Cr","Ni","Mo","Co","Al","Ti","Nb","CP_Cl","CP_temp","CP_pH","Cr_3p3Mo_common"]
    matrix=rel.pivot(index="dataset",columns="feature",values="spearman").reindex(index=SOURCES,columns=key)
    fig,ax=plt.subplots(figsize=(11,3.2),layout="constrained")
    cmap=plt.get_cmap("RdBu_r").copy();cmap.set_bad("#dddddd")
    im=ax.imshow(matrix,vmin=-1,vmax=1,cmap=cmap,aspect="auto")
    ax.set_xticks(range(len(key)),key,rotation=40,ha="right");ax.set_yticks(range(3),SOURCES)
    for i in range(3):
        for j in range(len(key)):
            v=matrix.iloc[i,j]
            ax.text(j,i,f"{v:.2f}" if np.isfinite(v) else "NA",ha="center",va="center",fontsize=9)
    ax.set_title("Pooled feature–target Spearman correlations (associations, not causal effects)")
    fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(OUT/"relationships.png",dpi=170);plt.close(fig)
    prior=pd.read_csv(OUT/"sampled_prior_feature_coverage.csv")
    key=["Fe","Cr","Ni","Mo","Co","Al","Ti","Nb"]
    fig,axes=plt.subplots(2,4,figsize=(13,6.5),layout="constrained")
    for ax,f in zip(axes.flat,key):
        for i,source in enumerate(["EPIT","Soccol"]):
            row=prior[(prior.source==source)&(prior.feature==f)]
            if len(row):
                row=row.iloc[0]
                ax.plot([row.q05,row.q95],[i,i],lw=5,color=COLORS[source],alpha=.6)
                ax.plot(row["median"],i,"o",color=COLORS[source])
            else:
                ax.text(.02,i,"unsupported",transform=ax.get_yaxis_transform(),fontsize=8)
        v=q[f].dropna()
        ax.plot(v.quantile([.05,.95]),[2,2],lw=5,color=COLORS["MPEA"],alpha=.6)
        ax.plot(v.median(),2,"o",color=COLORS["MPEA"])
        ax.set_yticks(range(3),["EPIT prior","Soccol prior","MPEA"])
        ax.set_title(f);ax.set_xlabel("wt%");ax.grid(axis="x",alpha=.2)
    fig.suptitle("Actual informed sampling versus MPEA: median and 5–95% interval")
    fig.savefig(OUT/"prior_coverage.png",dpi=170);plt.close(fig)


def main():
    d,audit,registry=load()
    q=summarize_data(d,audit,registry)
    print("Descriptive summaries complete",flush=True)
    compare_distributions(d,registry)
    relation_summary(d,registry)
    print("Target relationships and study bootstrap complete",flush=True)
    prior_and_overlap(d,q)
    plots(d,q,registry)
    names=["epit_soccol_context.csv","mpea_queries.csv","mpea_scoring_targets.csv",
           "combined_prior_config.json","shared_feature_registry.json"]
    (OUT/"provenance.json").write_text(json.dumps({
        "scope":"Post-hoc descriptive analysis, no fitting to MPEA labels or checkpoint selection.",
        "seed":20261020,"study_bootstrap_replicates":399,
        "bootstrap_limit":"Pointwise exploratory intervals; not corrected for feature screening.",
        "projection_limit":"Composition distance fills unknowns with zero only for projection diagnostics.",
        "source_reference_limit":"Source IDs approximate study groups; separate IDs may share provenance.",
        "input_sha256":{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in names},
    },indent=2))
    print("Saved",OUT,flush=True)


if __name__=="__main__":
    warnings.filterwarnings("ignore",message="An input array is constant")
    main()
