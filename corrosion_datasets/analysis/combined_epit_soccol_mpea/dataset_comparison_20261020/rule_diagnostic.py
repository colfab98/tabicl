"""Compatibility of actual frozen synthetic rule families with observed targets.

This is NOT a trained-model evaluation. Formula terms are standardized within
each diagnostic table, exactly as in synthetic target construction. Missing
inputs receive source-profile means (or a fixed, explicitly labelled environment
sensitivity); no coefficients or imputation values are fitted to MPEA labels.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from reproduce import DATA, OUT, TARGET, load, rho
from tabicl.prior.combined_corrosion import CombinedCorrosionPriorDataset, EPIT_FEATURE_COLUMNS
from tabicl.prior.dataset import PittingProfileInfo
from tabicl.prior.epit_schema import epit_feature_slices
from tabicl.prior.epit_feature_generator import sample_epit_feature_rows
from tabicl.prior.epit_feature_profile import load_epit_feature_profile
from tabicl.prior.soccol_feature_generator import sample_soccol_feature_rows
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS, soccol_feature_slices


def main():
    torch.set_num_threads(2)
    data,_,registry=load()
    wrapper=CombinedCorrosionPriorDataset(config_path=DATA/"combined_prior_config.json",
        batch_size=4,batch_size_per_gp=4,max_classes=0,min_seq_len=None,max_seq_len=1024,
        min_train_size=.1,max_train_size=.9,n_jobs=1,device="cpu")
    q=data[data.source_dataset.eq("MPEA")].copy()
    output, rows, coefficients=[],[],[]
    prior_config=json.loads((DATA/"combined_prior_config.json").read_text())
    for source in ["EPIT","Soccol"]:
        prior=wrapper.priors[source]
        hp=prior_config["regimes"][source]["fixed_hp"]
        columns=list(EPIT_FEATURE_COLUMNS if source=="EPIT" else SOCCOL_FEATURE_COLUMNS)
        blocks=epit_feature_slices(24) if source=="EPIT" else soccol_feature_slices()
        if source=="EPIT":
            profile=load_epit_feature_profile()
            full=hp["pitting_composition_family_probs"]
            probs=[full[profile.composition_profile.families.index(f)] for f in profile.families]
            batch=sample_epit_feature_rows(4096,profile=profile,composition_family_probabilities=probs,
                perturb_strength=hp["pitting_composition_perturb_strength"],random_state=42)
        else:
            batch=sample_soccol_feature_rows(4096,perturb_strength=hp["pitting_composition_perturb_strength"],random_state=42)
        sample=pd.DataFrame(batch.features,columns=columns)
        means=sample.mean()
        source_data=data[data.source_dataset.eq(source)]
        tables={}
        for name,frame in [("own_source",source_data),("MPEA_source_mean_imputation",q),
                           ("MPEA_fixed_environment_sensitivity",q)]:
            X=frame.reindex(columns=columns).apply(pd.to_numeric,errors="coerce").fillna(means)
            # Method/procedure strings are not meaningful continuous inputs.
            for c in registry["categorical_feature_columns"]:
                if c in columns:
                    X[c]=means[c]
            if name=="MPEA_fixed_environment_sensitivity":
                for c,v in {"CP_temp":25.0,"CP_pH":7.0,"CP_Cl":.6,"CP_Br":0.0,
                            "CP_SO4":0.0,"CP_NO3":0.0,"N":0.0,"S":0.0}.items():
                    if c in X:
                        X[c]=v
            tables[name]=(torch.tensor(X.to_numpy(),dtype=torch.float64),frame)
        scores={name:[] for name in tables}
        material_scores=[]
        environment_scores=[]
        sample_tensor=torch.tensor(sample.to_numpy(),dtype=torch.float64)
        for draw in range(128):
            np.random.seed(20261020+draw)
            torch.manual_seed(20261020+draw)
            info=PittingProfileInfo(applied=True,physical_profile_applied=True)
            rule=prior._sample_fixed_epit_target_rule(sample_tensor,blocks,1.0,info)
            coefficient=rule["target_rule_coefficients"]
            coefficients.append({"source":source,"draw":draw,"family":rule["target_rule_family"],**coefficient})
            for name,(X,frame) in tables.items():
                _,drive=prior._evaluate_fixed_epit_target_rule_tensor(X,rule)
                scores[name].append(drive.numpy())
                output.append({"source":source,"table":name,"draw":draw,
                               "family":rule["target_rule_family"],"spearman":rho(drive.numpy(),frame[TARGET])})
            # How closely the source rule tracks composition-only vs environment-only variation.
            _,full=prior._evaluate_fixed_epit_target_rule_tensor(sample_tensor,rule)
            material_only=sample_tensor.clone()
            material_only[:,blocks["material"].stop:]=material_only[:,blocks["material"].stop:].mean(dim=0)
            environment_only=sample_tensor.clone()
            environment_only[:,blocks["material"]]=environment_only[:,blocks["material"]].mean(dim=0)
            _,md=prior._evaluate_fixed_epit_target_rule_tensor(material_only,rule)
            _,ed=prior._evaluate_fixed_epit_target_rule_tensor(environment_only,rule)
            material_scores.append(rho(full.numpy(),md.numpy()))
            environment_scores.append(rho(full.numpy(),ed.numpy()))
        for name,(X,frame) in tables.items():
            # Equal average over actual probability-weighted sampled draws.
            standardized=[]
            for drive in scores[name]:
                standardized.append((drive-drive.mean())/max(drive.std(),1e-12))
            mean=np.mean(standardized,axis=0)
            for rid,y,v in zip(frame.row_id,frame[TARGET],mean,strict=True):
                rows.append({"source":source,"table":name,"row_id":rid,"y_true":y,"mean_standardized_rule_score":v})
            print(source,name,rho(frame[TARGET],mean),flush=True)
        pd.DataFrame({"source":source,"draw":range(128),
                      "full_vs_material_only_spearman":material_scores,
                      "full_vs_environment_only_spearman":environment_scores}).to_csv(
                          OUT/f"{source.lower()}_rule_dependence.csv",index=False)
    pd.DataFrame(output).to_csv(OUT/"rule_draw_correlations.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"rule_scores.csv",index=False)
    pd.DataFrame(coefficients).to_csv(OUT/"rule_sampled_coefficients.csv",index=False)
    (OUT/"rule_diagnostic_metadata.json").write_text(json.dumps({
        "draws_per_source":128,"source":str(DATA/"combined_prior_config.json"),
        "label_use":"Scores only; no target-dependent coefficient, family, or imputation selection.",
        "missing_input_policy":"Source-generated feature means; categorical procedure fixed at mean code.",
        "sensitivity":"All temperature/pH/chloride held at 25C/7/0.6M, N/S/Br/sulfate/nitrate zero.",
        "limit":"Diagnostic surrogates for unreported inputs, not measured chemistry or model predictions.",
        "standardization":"Per diagnostic table, following synthetic generator; no fitting to labels.",
    },indent=2))


if __name__=="__main__":
    main()
