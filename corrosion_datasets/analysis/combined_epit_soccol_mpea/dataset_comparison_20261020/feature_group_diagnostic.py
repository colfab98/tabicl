"""Fixed-hyperparameter CatBoost feature diagnostics on frozen outer queries.

EPIT/Soccol use their original development/final-test partitions.
MPEA uses the previously frozen five outer folds. No development fold is
substituted for an original source final-test partition. No model selection.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from reproduce import DATA, OUT, TARGET, ELEMENTS, load, rho
from scripts.eval_corrosion_datasets import make_catboost_regressor


def main():
    d,_,registry=load()
    plans=[]
    assignments={
        "EPIT":ROOT/"corrosion_datasets/analysis/epit_pipeline/splits_v2/split_assignments.csv",
        "Soccol":ROOT/"corrosion_datasets/datasets/soccol_pitting_potential/processed/splits_v1/split_assignments.csv",
    }
    for source,path in assignments.items():
        g=d[d.source_dataset.eq(source)].reset_index(drop=True)
        a=pd.read_csv(path).sort_values("task_row_index").reset_index(drop=True)
        np.testing.assert_array_equal(a.task_row_index,np.arange(len(g)))
        if source=="EPIT":
            np.testing.assert_array_equal(pd.to_numeric(g.source_row_id),a.source_no)
            ycol="target_mV_SCE_development_only"
        else:
            np.testing.assert_array_equal(g.source_row_id.astype(str).str.strip(),a.label.astype(str).str.strip())
            ycol="E_pit_development_only"
        train=np.flatnonzero(a.outer_split.eq("development"))
        test=np.flatnonzero(a.outer_split.eq("final_test"))
        assert (len(train),len(test))==((608,152) if source=="EPIT" else (3222,805))
        # Frozen split metadata stores some EPIT means rounded to two decimals.
        np.testing.assert_allclose(g[TARGET].iloc[train],a[ycol].iloc[train],atol=.005,rtol=0)
        assert not (set(a.composition_group.iloc[train]) & set(a.composition_group.iloc[test]))
        plans.append((source,0,g,train,test))
    g=d[d.source_dataset.eq("MPEA")].reset_index(drop=True)
    rows=pd.DataFrame(json.loads((OUT.parent/"mpea_only_fivefold_comparison_cpu/split_manifest.json").read_text())["rows"])
    np.testing.assert_array_equal(rows.row_id,g.row_id)
    for fold in range(1,6):
        plans.append(("MPEA",fold,g,np.flatnonzero(rows.outer_fold.ne(fold)),np.flatnonzero(rows.outer_fold.eq(fold))))
    groups={"all_features":registry["feature_columns_in_order"],
            "composition_only":ELEMENTS,
            "composition_and_chloride":[*ELEMENTS,"CP_Cl"],
            "noncomposition_only":registry["feature_columns_in_order"][24:]}
    records,predictions=[],[]
    for source,fold,g,train,test in plans:
        for name,columns in groups.items():
            model=make_catboost_regressor(iterations=1000,depth=6,learning_rate=.03,l2_leaf_reg=3,
                random_state=42,thread_count=4,
                categorical_columns=tuple(c for c in columns if c in registry["categorical_feature_columns"]))
            model.fit(g[columns].iloc[train],g[TARGET].iloc[train])
            pred=model.predict(g[columns].iloc[test])
            truth=g[TARGET].iloc[test].to_numpy()
            record={"dataset":source,"outer_fold":fold,"feature_group":name,"n_context":len(train),
                    "n_query":len(test),"spearman":rho(truth,pred),"rmse":np.sqrt(mean_squared_error(truth,pred)),
                    "mae":mean_absolute_error(truth,pred),"r2":r2_score(truth,pred)}
            records.append(record)
            for rid,y,p in zip(g.row_id.iloc[test],truth,pred,strict=True):
                predictions.append({"dataset":source,"outer_fold":fold,"feature_group":name,
                                    "row_id":rid,"y_true":y,"y_pred":p})
            pd.DataFrame(records).to_csv(OUT/"catboost_feature_group_metrics.csv",index=False)
            print(record,flush=True)
    pd.DataFrame(predictions).to_csv(OUT/"catboost_feature_group_predictions.csv",index=False)
    summary=pd.DataFrame(records).groupby(["dataset","feature_group"])[["spearman","rmse","mae","r2"]].agg(["mean","std"])
    summary.to_csv(OUT/"catboost_feature_group_summary.csv")
    previous=pd.read_csv(OUT.parent/"mpea_only_fivefold_comparison_cpu/rows.csv")
    old=previous[previous.model.eq("catboost") & previous.checkpoint_step.eq(5000)].sort_values("outer_fold")
    new=pd.DataFrame(records).query("dataset == 'MPEA' and feature_group == 'all_features'").sort_values("outer_fold")
    np.testing.assert_allclose(new.spearman,old.test_spearman,atol=1e-7)
    np.testing.assert_allclose(new.rmse,old.test_rmse,atol=1e-4)


if __name__=="__main__":
    main()
