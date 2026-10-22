"""Change labeled context sources while retaining frozen MPEA queries and preprocessing."""
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
    context_targets = task.y.iloc[train].to_numpy(dtype=float)
    query_targets = task.y.iloc[query].to_numpy(dtype=float)
    sources = pd.read_csv(DATA / "epit_soccol_context.csv")["source_dataset"].to_numpy()
    epit = np.flatnonzero(sources == "EPIT")
    soccol = np.flatnonzero(sources == "Soccol")
    rng = np.random.default_rng(42)
    cases = {
        "full_context": np.arange(len(train)),
        "epit_only_760": epit,
        "soccol_only_4027": soccol,
        "soccol_only_760": np.sort(rng.choice(soccol, 760, replace=False)),
        "mixed_760": np.sort(rng.choice(len(train), 760, replace=False)),
        "mixed_1024": np.sort(rng.choice(len(train), 1024, replace=False)),
    }
    old = pd.read_csv(OUT.parent / "shared_specialist_comparison_cpu/rows.csv")
    rows = []
    predictions = pd.DataFrame({"query_index": query, "target": query_targets})
    metadata = {
        "diagnostic_only": True,
        "frozen_query_indices": query.tolist(),
        "preprocessing": "Feature encoding/scaling and target scaling fitted once on all 4787 context rows, unchanged across cases; no query fitting.",
        "context_selection_seed": 42,
        "context_indices": {key: value.tolist() for key, value in cases.items()},
        "rows": rows,
    }
    for model, run in (("combined_corrpfn_v2", "final_v2_seed42"), ("soccol_shared_v2", "soccol_shared_v2_seed42")):
        path = RUNS / run / "step-1000.ckpt"
        estimator = ev.make_tabicl_regressor(
            model_path=str(path), checkpoint_version="", device="cpu",
            n_estimators=8, random_state=42, allow_auto_download=False,
            feat_shuffle_method="none", norm_methods=["none"], remove_constant_features=False,
        )
        estimator.fit(raw_context, context_targets)
        encoded_queries = estimator.X_encoder_.transform(raw_queries)
        views = estimator.ensemble_generator_.transform(encoded_queries, mode="both")
        assert len(views) == 1
        X_all, y_context = next(iter(views.values()))
        assert X_all.shape == (1, 5054, 50)
        assert y_context.shape == (1, 4787)
        for case, indices in cases.items():
            started = time.monotonic()
            chosen = np.concatenate([indices, np.arange(len(train), len(train) + len(query))])
            X = X_all[:, chosen]
            y = y_context[:, indices]
            normalized = estimator._batch_forward(X, y, output_type="median")
            prediction = estimator.y_scaler_.inverse_transform(normalized.reshape(-1, 1)).reshape(1, len(query)).mean(axis=0)
            assert np.isfinite(prediction).all()
            row = {
                "model": model, "checkpoint": str(path), "case": case,
                "context_rows": len(indices), "epit_context_rows": int((sources[indices] == "EPIT").sum()),
                "soccol_context_rows": int((sources[indices] == "Soccol").sum()),
                "query_rows": len(query), "features": X.shape[-1],
                "spearman": float(spearmanr(query_targets, prediction).statistic),
                "rmse": float(np.sqrt(np.mean((query_targets - prediction) ** 2))),
                "mae": float(np.mean(np.abs(query_targets - prediction))),
                "elapsed_seconds": time.monotonic() - started,
            }
            if case == "full_context":
                reference = old[(old.model == model) & (old.checkpoint_step == 1000)].iloc[0]
                assert abs(row["spearman"] - reference.test_spearman) < 1e-5
                assert abs(row["rmse"] - reference.test_rmse) < 0.01
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
