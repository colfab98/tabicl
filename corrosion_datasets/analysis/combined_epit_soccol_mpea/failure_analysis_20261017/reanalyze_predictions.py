"""Reanalyse existing frozen-MPEA predictions; no training or new model selection."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
PREDICTIONS = OUT.parent / "context_label_diagnostic_20261016/predictions.csv"


def rho(y, prediction):
    if np.ptp(y) == 0 or np.ptp(prediction) == 0:
        return None
    return float(spearmanr(y, prediction).statistic)


def main():
    queries = pd.read_csv(DATA / "mpea_queries.csv")
    targets = pd.read_csv(DATA / "mpea_scoring_targets.csv")
    context = pd.read_csv(DATA / "epit_soccol_context.csv")
    predictions = pd.read_csv(PREDICTIONS)
    assert queries.row_id.equals(targets.row_id)
    np.testing.assert_array_equal(predictions.query_index, np.arange(4787, 5054))
    y = targets.reported_pitting_potential_mV.to_numpy()
    np.testing.assert_allclose(predictions.target, y, rtol=0, atol=0)
    features = list(context.columns[5:-1])
    assert len(features) == 50
    assert len(queries) == 267
    codes, names = pd.factorize(queries.source_reference.fillna("missing"))
    groups = [np.flatnonzero(codes == index) for index in range(len(names))]
    rows = []
    per_study = []
    for model in ["combined_v2_1000", "combined_v2_3000", "soccol_shared_1000", "epit_shared_1000"]:
        original = predictions[f"{model}__original_labels"].to_numpy()
        shuffled = predictions[[f"{model}__shuffled_labels_seed{i}" for i in range(3)]].mean(axis=1).to_numpy()
        for case, prediction in [("original_labels", original), ("mean_of_three_shuffled_predictions", shuffled)]:
            yr, pr = rankdata(y), rankdata(prediction)
            yr = (yr - yr.mean()) / yr.std()
            pr = (pr - pr.mean()) / pr.std()
            group_y = np.array([yr[ix].mean() for ix in groups])[codes]
            group_p = np.array([pr[ix].mean() for ix in groups])[codes]
            between = float(np.mean(group_y * group_p))
            within = float(np.mean((yr - group_y) * (pr - group_p)))
            np.testing.assert_allclose(between + within, rho(y, prediction), atol=1e-12)
            rows.append({
                "model": model, "case": case, "spearman": rho(y, prediction),
                "between_study_rank_covariance": between,
                "within_study_rank_covariance": within,
                "prediction_mean_mV": float(prediction.mean()),
                "prediction_std_mV": float(prediction.std()),
                "rmse_mV": float(np.sqrt(np.mean((prediction - y) ** 2))),
            })
        for name, ix in zip(names, groups):
            per_study.append({
                "model": model, "source_reference": name, "n": len(ix),
                "original_spearman": rho(y[ix], original[ix]),
                "shuffled_mean_spearman": rho(y[ix], shuffled[ix]),
                "target_mean_mV": float(y[ix].mean()),
                "original_mean_mV": float(original[ix].mean()),
                "shuffled_mean_mV": float(shuffled[ix].mean()),
            })

    original = predictions.combined_v2_3000__original_labels.to_numpy()
    shuffled = predictions[[f"combined_v2_3000__shuffled_labels_seed{i}" for i in range(3)]].mean(axis=1).to_numpy()
    rng = np.random.default_rng(20261017)
    bootstrap = []
    for _ in range(2000):
        ix = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        bootstrap.append(rho(y[ix], shuffled[ix]) - rho(y[ix], original[ix]))
    duplicate_groups = queries[features].assign(target=y, original=original, shuffled=shuffled).groupby(
        features, dropna=False
    )[["target", "original", "shuffled"]].mean()
    report = {
        "scope": "Exploratory reanalysis of existing predictions on all 267 frozen outer MPEA queries; no model changes or checkpoint selection.",
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [PREDICTIONS, DATA / "mpea_queries.csv", DATA / "mpea_scoring_targets.csv"]
        },
        "query_rows": len(queries), "study_groups": len(groups),
        "unique_model_visible_query_rows": len(queries[features].drop_duplicates()),
        "observed_query_feature_counts": queries[features].notna().sum().to_dict(),
        "mean_observed_composition_count": {
            **{source: float(frame[features[:24]].notna().sum(axis=1).mean())
               for source, frame in context.groupby("source_dataset")},
            "MPEA": float(queries[features[:24]].notna().sum(axis=1).mean()),
        },
        "rank_decomposition_definition": "Standardize global target/prediction ranks to mean zero and population variance one. Between = mean(product of study mean ranks); within = mean(product of study-centered ranks). These covariance contributions sum exactly to global Spearman. They are not independent within/between correlations or causal effects.",
        "rank_decomposition": rows,
        "cluster_bootstrap_combined_3000": {
            "contrast": "Mean of three saved shuffled-label predictions minus original-label prediction",
            "resampling_unit": "source_reference; keep all rows of each sampled study",
            "replicates": 2000, "seed": 20261017,
            "spearman_difference": rho(y, shuffled) - rho(y, original),
            "percentile_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "limitations": "Conditional on these checkpoints and three shuffle draws; no training-seed or shuffle-population uncertainty. Source references are available grouping proxies, not proof of independent studies.",
        },
        "duplicate_input_group_sensitivity_combined_3000": {
            "description": "Average targets and predictions within identical 50-feature query rows, then give each unique input equal weight. Sensitivity analysis only, not a replacement evaluation set.",
            "groups": len(duplicate_groups),
            "original_spearman": rho(duplicate_groups.target, duplicate_groups.original),
            "shuffled_mean_spearman": rho(duplicate_groups.target, duplicate_groups.shuffled),
        },
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "rank_decomposition.csv", index=False)
    pd.DataFrame(per_study).to_csv(OUT / "per_study.csv", index=False)
    print(json.dumps({key: report[key] for key in ["query_rows", "study_groups", "unique_model_visible_query_rows", "cluster_bootstrap_combined_3000"]}, indent=2))


if __name__ == "__main__":
    main()
