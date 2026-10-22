"""Audit frozen synthetic priors without training or reading MPEA targets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, rankdata

from tabicl.prior.combined_corrosion import (
    COMBINED_FEATURE_COLUMNS, EPIT_FEATURE_COLUMNS, CombinedCorrosionPriorDataset,
)
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS
from tabicl.train.missingness import MissingnessAugmenter

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    denominator = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
    return np.divide((x * y).sum(axis=0), denominator,
                     out=np.full(np.shape(denominator), np.nan), where=denominator > 1e-12)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            "q01": float(np.quantile(values, .01)), "q05": float(np.quantile(values, .05)),
            "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
            "q99": float(np.quantile(values, .99)), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-per-seed", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261018, 20261019])
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.tasks_per_seed % 4:
        parser.error("--tasks-per-seed must be divisible by four (training HP group size)")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = DATA / "combined_prior_config.json"
    config = json.loads(config_path.read_text())
    augmenter = MissingnessAugmenter(DATA / "context_missingness_profile.json",
                                    query_column_probability=.8, preserve_observed_composition=True)
    tasks, features, target_arrays, raw_arrays, input_arrays = [], [], {}, {}, {}
    digest = hashlib.sha256()
    checks = Counter()

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        wrapper = CombinedCorrosionPriorDataset(
            config_path=config_path, batch_size=4, batch_size_per_gp=4, max_classes=0,
            min_seq_len=None, max_seq_len=args.rows, min_train_size=.1, max_train_size=.9,
            n_jobs=1, device="cpu", seed_offset=seed,
        )
        for source_id, source in enumerate(wrapper.REGIME_ORDER):
            prior = wrapper.priors[source]
            captures = []
            original_generate = prior.generate_dataset
            original_scm = prior._generate_scm_target_from_physical_features
            scm_capture = {}

            def capture_scm(*a, **kw):
                result = original_scm(*a, **kw)
                scm_capture["y"] = result.clone()
                return result

            def capture_generate(params):
                result = original_generate(params)
                row = {"scm_type": params["prior_type"]}
                if source != "generic":
                    row.update(raw=torch.tensor(prior.last_pitting_feature_batch.features,
                                                dtype=result[0].dtype),
                               rule=copy.deepcopy(prior.last_pitting_target_rule),
                               drive=prior.last_pitting_target_drive.clone(),
                               component=prior.last_pitting_target_component.clone(),
                               scm=scm_capture["y"].clone())
                captures.append(row)
                return result

            prior._generate_scm_target_from_physical_features = capture_scm
            prior.generate_dataset = capture_generate
            for group in range(args.tasks_per_seed // 4):
                captures.clear()
                native, y, _, _, _ = prior.get_batch(4)
                X = wrapper._map_regime(source, native)
                # The combined wrapper draws a new, common boundary after generation.
                boundary = int(prior.sample_train_size(.1, .9, args.rows))
                masked, _ = augmenter.augment(
                    X, torch.full((4,), 50, dtype=torch.long), train_size=boundary,
                    source_ids=torch.full((4,), source_id, dtype=torch.long),
                    generator=torch.Generator().manual_seed(seed + source_id * 1000 + group),
                )
                observed = torch.isfinite(masked)
                torch.testing.assert_close(masked[observed], X[observed], rtol=0, atol=0)
                assert torch.isfinite(y).all() and not torch.isinf(masked).any()
                checks["tables_with_finite_targets_and_position_preserving_masks"] += 4
                digest.update(X.numpy().tobytes())
                digest.update(y.numpy().tobytes())
                digest.update(masked.numpy().tobytes())
                for i, capture in enumerate(captures):
                    task = group * 4 + i
                    yi, xi, mi = y[i].numpy(), X[i].numpy(), masked[i].numpy()
                    rec = {"source": source, "seed": seed, "task": task, "context_rows": boundary,
                           "scm_type": capture["scm_type"], "target_mean": float(yi.mean()),
                           "target_std": float(yi.std()), "target_max_abs": float(np.abs(yi).max()),
                           "context_target_mean": float(yi[:boundary].mean()),
                           "context_target_std": float(yi[:boundary].std()),
                           "query_target_mean": float(yi[boundary:].mean()),
                           "query_target_std": float(yi[boundary:].std()),
                           "context_observed_features": float(np.isfinite(mi[:boundary]).sum(axis=1).mean()),
                           "query_observed_features": float(np.isfinite(mi[boundary:]).sum(axis=1).mean()),
                           "target_abs_above_5_fraction": float((np.abs(yi) > 5).mean())}
                    if source != "generic":
                        raw = capture["raw"]
                        expected = prior._standardize_features_for_scm_target(raw)
                        torch.testing.assert_close(native[i], expected, rtol=0, atol=0)
                        columns = EPIT_FEATURE_COLUMNS if source == "EPIT" else SOCCOL_FEATURE_COLUMNS
                        for j, column in enumerate(columns):
                            k = COMBINED_FEATURE_COLUMNS.index(column)
                            torch.testing.assert_close(X[i, :, k], expected[:, j], rtol=0, atol=0)
                        for k, column in enumerate(COMBINED_FEATURE_COLUMNS):
                            if column not in columns:
                                assert torch.isnan(X[i, :, k]).all()
                        weight = float(prior.fixed_hp["informed_target_mix_weight"])
                        reconstructed = prior._standardize_signal(
                            (1-weight)*prior._standardize_signal(capture["scm"]) + capture["component"])
                        torch.testing.assert_close(y[i], reconstructed, rtol=0, atol=0)
                        checks["informed_tables_with_exact_raw_to_shared_mapping_and_target_reconstruction"] += 1
                        drive, scm = capture["drive"].numpy(), capture["scm"].numpy()
                        rec.update(target_rule_family=capture["rule"]["target_rule_family"],
                                   target_rule_mix_weight=weight,
                                   target_vs_rule_pearson=float(correlation(yi, drive)),
                                   target_vs_scm_pearson=float(correlation(yi, scm)),
                                   rule_vs_scm_pearson=float(correlation(drive, scm)),
                                   target_vs_rule_spearman=float(correlation(rankdata(yi), rankdata(drive))))
                        mapped_raw = wrapper._map_regime(source, raw.unsqueeze(0))[0].numpy()
                        raw_arrays.setdefault(source, []).append(mapped_raw)
                    tasks.append(rec)
                    target_arrays.setdefault(source, []).append(yi)
                    input_arrays.setdefault(source, []).append(xi)
                    ranks = rankdata(np.nan_to_num(xi), axis=0)
                    rhos = correlation(ranks, rankdata(yi)[:, None])
                    for k, column in enumerate(COMBINED_FEATURE_COLUMNS):
                        vals = xi[:, k]
                        finite = vals[np.isfinite(vals)]
                        features.append({"source": source, "seed": seed, "task": task,
                                         "feature": column, "target_spearman": rhos[k],
                                         "input_std": float(finite.std()) if finite.size else np.nan,
                                         "input_mean": float(finite.mean()) if finite.size else np.nan,
                                         "context_observed_fraction": float(np.isfinite(mi[:boundary, k]).mean()),
                                         "query_observed_fraction": float(np.isfinite(mi[boundary:, k]).mean())})
                if (group + 1) % 8 == 0:
                    print(f"seed={seed} source={source} tables={(group+1)*4}", flush=True)
            prior.generate_dataset = original_generate
            prior._generate_scm_target_from_physical_features = original_scm

    frame = pd.DataFrame(tasks)
    feature_frame = pd.DataFrame(features)
    frame.to_csv(args.output_dir / "tasks.csv", index=False)
    feature_frame.to_csv(args.output_dir / "features_by_task.csv", index=False)
    summaries, raw_rows, input_rows = {}, [], []
    for source in wrapper.REGIME_ORDER:
        source_tasks = frame[frame.source == source]
        summaries[source] = {col: describe(source_tasks[col]) for col in source_tasks.select_dtypes("number").columns
                             if col not in ("seed", "task")}
        summaries[source]["scm_types"] = source_tasks.scm_type.value_counts().to_dict()
        summaries[source]["pooled_target"] = describe(np.concatenate(target_arrays[source]))
        for representation, arrays, rows_out in [("raw", raw_arrays, raw_rows), ("input", input_arrays, input_rows)]:
            if source not in arrays:
                continue
            pooled = np.concatenate(arrays[source])
            for k, column in enumerate(COMBINED_FEATURE_COLUMNS):
                vals = pooled[:, k]
                row = {"source": source, "feature": column, **describe(vals)}
                row["zero_fraction_of_observed"] = float((vals[np.isfinite(vals)] == 0).mean()) if np.isfinite(vals).any() else np.nan
                rows_out.append(row)
    pd.DataFrame(raw_rows).to_csv(args.output_dir / "raw_feature_summary.csv", index=False)
    pd.DataFrame(input_rows).to_csv(args.output_dir / "model_input_summary.csv", index=False)
    feature_summary = feature_frame.groupby(["source", "feature"]).agg(
        median_target_spearman=("target_spearman", "median"),
        mean_target_spearman=("target_spearman", "mean"),
        mean_input_std=("input_std", "mean"),
        constant_table_fraction=("input_std", lambda s: float((s.dropna() < 1e-6).mean()) if s.notna().any() else np.nan),
        context_observed_fraction=("context_observed_fraction", "mean"),
        query_observed_fraction=("query_observed_fraction", "mean"),
    ).reset_index()
    feature_summary.to_csv(args.output_dir / "feature_summary.csv", index=False)
    pooled_raw = {source: np.concatenate(arrays) for source, arrays in raw_arrays.items()}
    pooled_input = {source: np.concatenate(arrays) for source, arrays in input_arrays.items()}
    shifts = []
    for k, column in enumerate(COMBINED_FEATURE_COLUMNS):
        if column in EPIT_FEATURE_COLUMNS and column in SOCCOL_FEATURE_COLUMNS:
            shifts.append({"feature": column,
                           "raw_distribution_KS_distance": float(ks_2samp(pooled_raw["EPIT"][:, k], pooled_raw["Soccol"][:, k]).statistic),
                           "standardized_distribution_KS_distance": float(ks_2samp(pooled_input["EPIT"][:, k], pooled_input["Soccol"][:, k]).statistic)})
    pd.DataFrame(shifts).to_csv(args.output_dir / "common_feature_distribution_differences.csv", index=False)
    specialist_equivalence = {}
    for run in ["epit_shared_v2_seed42", "soccol_shared_v2_seed42"]:
        specialist = json.loads((ROOT / "checkpoints/combined_epit_soccol_mpea" / run / "prior_config.json").read_text())
        assert specialist["feature_columns_in_order"] == config["feature_columns_in_order"]
        for source in wrapper.REGIME_ORDER:
            left = {k: v for k, v in config["regimes"][source].items() if k != "probability"}
            right = {k: v for k, v in specialist["regimes"][source].items() if k != "probability"}
            assert left == right
        specialist_equivalence[run] = "Same feature order and all generator settings; only regime sampling probabilities differ."
    context = pd.read_csv(DATA / "epit_soccol_context.csv")
    context_scale = context.groupby("source_dataset")["reported_pitting_potential_mV"].agg(
        ["count", "mean", "std", "min", "median", "max"]).to_dict(orient="index")
    registry = json.loads((DATA / "shared_feature_registry.json").read_text())
    result = {"scope": "Synthetic data audit only; no MPEA targets, training, or checkpoint selection.",
              "seeds": args.seeds, "tasks_per_source": args.tasks_per_seed * len(args.seeds),
              "rows_per_task": args.rows, "hyperparameter_group_size": 4,
              "prior_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "sample_sha256": digest.hexdigest(), "checks": dict(checks), "summary": summaries,
              "specialist_generator_equivalence": specialist_equivalence,
              "shared_feature_units": registry["feature_units"],
              "real_context_target_scale_mV": context_scale,
              "limitations": ["Finite synthetic sample, not evidence of the cause of MPEA performance.",
                              "Four tables share a hyperparameter draw, as in training; rows are not independent tasks.",
                              "Source correlations describe observational generated relationships, not feature effects."]}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for source, summary in summaries.items():
        keys = ["target_std", "target_vs_rule_pearson", "target_vs_scm_pearson", "context_observed_features", "query_observed_features"]
        print(source, {key: summary[key]["median"] for key in keys if key in summary and summary[key]["count"]}, flush=True)


if __name__ == "__main__":
    main()
