"""Diagnostic label controls; all frozen outer MPEA queries stay unchanged."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from scripts import eval_corrosion_datasets as ev

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
RUNS = ROOT / "checkpoints/combined_epit_soccol_mpea"


def main():
    torch.set_num_threads(4)
    task = ev._make_combined_pipeline_task(DATA)
    train, query = task.fixed_split.train_index, task.fixed_split.test_index
    assert len(train) == 4787 and len(query) == 267
    raw_context = task.X.iloc[train].reset_index(drop=True)
    raw_queries = task.X.iloc[query].reset_index(drop=True)
    targets = task.y.iloc[train].to_numpy(dtype=float)
    truth = task.y.iloc[query].to_numpy(dtype=float)
    sources = pd.read_csv(DATA / "epit_soccol_context.csv")["source_dataset"].to_numpy()
    rows = []
    predictions = pd.DataFrame({"query_index": query, "target": truth})
    metadata = {
        "diagnostic_only": True,
        "frozen_query_indices": query.tolist(),
        "protocol": "Fit preprocessing once on unchanged context; alter only scaled context labels. Query features, query rows, feature preprocessing and inverse target scaling stay fixed. Shuffles destroy feature-label pairing, not the label distribution. Within-source shuffle preserves source-specific target distributions. Constant labels are deliberately outside normal training conditions.",
        "rows": rows,
    }
    runs = [
        ("combined_v2_1000", "final_v2_seed42", 1000),
        ("soccol_shared_1000", "soccol_shared_v2_seed42", 1000),
        ("epit_shared_1000", "epit_shared_v2_seed42", 1000),
        ("combined_v2_3000", "final_v2_seed42", 3000),
    ]
    for model, run, step in runs:
        path = RUNS / run / f"step-{step}.ckpt"
        estimator = ev.make_tabicl_regressor(
            model_path=str(path), checkpoint_version="", device="cpu",
            n_estimators=8, random_state=42, allow_auto_download=False,
            feat_shuffle_method="none", norm_methods=["none"], remove_constant_features=False,
        )
        estimator.fit(raw_context, targets)
        views = estimator.ensemble_generator_.transform(estimator.X_encoder_.transform(raw_queries), mode="both")
        assert len(views) == 1
        X, original_y = next(iter(views.values()))
        assert X.shape == (1, 5054, 50) and original_y.shape == (1, 4787)
        cases = {"original_labels": original_y}
        for seed in range(3):
            cases[f"shuffled_labels_seed{seed}"] = original_y[:, np.random.default_rng(seed).permutation(len(train))]
        within = original_y.copy()
        rng = np.random.default_rng(0)
        for source in np.unique(sources):
            ix = np.flatnonzero(sources == source)
            within[:, ix] = original_y[:, rng.permutation(ix)]
        cases["shuffled_within_source_seed0"] = within
        cases["constant_context_mean"] = np.zeros_like(original_y)
        original_prediction = None
        for case, y in cases.items():
            started = time.monotonic()
            normalized = estimator._batch_forward(X, y, output_type="median")
            prediction = estimator.y_scaler_.inverse_transform(normalized.reshape(-1, 1)).reshape(-1)
            assert np.isfinite(prediction).all()
            if original_prediction is None:
                original_prediction = prediction.copy()
            row = {
                "model": model, "checkpoint": str(path), "case": case,
                "context_rows": len(train), "query_rows": len(query), "features": X.shape[-1],
                "spearman": float(spearmanr(truth, prediction).statistic),
                "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
                "prediction_std_mV": float(np.std(prediction)),
                "spearman_vs_original_predictions": float(spearmanr(original_prediction, prediction).statistic),
                "elapsed_seconds": time.monotonic() - started,
            }
            rows.append(row)
            predictions[f"{model}__{case}"] = prediction
            pd.DataFrame(rows).to_csv(OUT / "rows.csv", index=False)
            predictions.to_csv(OUT / "predictions.csv", index=False)
            (OUT / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        del estimator
        gc.collect()


if __name__ == "__main__":
    main()
