"""Probe composition NaNs without changing frozen query rows or observed values."""
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
from tabicl.prior.combined_corrosion import COMBINED_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
RUNS = ROOT / "checkpoints/combined_epit_soccol_mpea"


def main():
    torch.set_num_threads(4)
    task = ev._make_combined_pipeline_task(DATA)
    train, query = task.fixed_split.train_index, task.fixed_split.test_index
    assert len(train) == 4787 and len(query) == 267
    context = task.X.iloc[train].reset_index(drop=True)
    queries = task.X.iloc[query].reset_index(drop=True)
    assert tuple(context.columns) == COMBINED_FEATURE_COLUMNS
    targets = task.y.iloc[train].to_numpy(dtype=float)
    truth = task.y.iloc[query].to_numpy(dtype=float)
    rows = []
    predictions = pd.DataFrame({"query_index": query, "target": truth})
    metadata = {
        "diagnostic_only": True,
        "frozen_query_indices": query.tolist(),
        "protocol": "Fixed full context, true context labels, frozen outer queries, encoding, scaling and observed feature values. Replace only missing composition values by the corresponding context mean in the final transformed model-input space. This is a diagnostic intervention, not a claim that missing elemental content equals the mean or zero.",
        "composition_columns": list(COMBINED_FEATURE_COLUMNS[:24]),
        "rows": rows,
    }
    for model, run, step in [
        ("combined_v2_1000", "final_v2_seed42", 1000),
        ("soccol_shared_1000", "soccol_shared_v2_seed42", 1000),
        ("combined_v2_3000", "final_v2_seed42", 3000),
    ]:
        path = RUNS / run / f"step-{step}.ckpt"
        estimator = ev.make_tabicl_regressor(
            model_path=str(path), checkpoint_version="", device="cpu",
            n_estimators=8, random_state=42, allow_auto_download=False,
            feat_shuffle_method="none", norm_methods=["none"], remove_constant_features=False,
        )
        estimator.fit(context, targets)
        views = estimator.ensemble_generator_.transform(estimator.X_encoder_.transform(queries), mode="both")
        assert len(views) == 1
        original_X, y = next(iter(views.values()))
        assert original_X.shape == (1, 5054, 50) and y.shape == (1, 4787)
        fill = np.nanmean(original_X[:, :len(train), :24], axis=1, keepdims=True)
        assert np.isfinite(fill).all()
        cases = {
            "original": (False, False),
            "context_composition_filled": (True, False),
            "query_composition_filled": (False, True),
            "both_composition_filled": (True, True),
        }
        for case, (fill_context, fill_query) in cases.items():
            started = time.monotonic()
            X = original_X.copy()
            for selected, do_fill in [(slice(0, len(train)), fill_context), (slice(len(train), None), fill_query)]:
                if do_fill:
                    values = X[:, selected, :24]
                    X[:, selected, :24] = np.where(np.isnan(values), fill, values)
            observed = np.isfinite(original_X)
            np.testing.assert_array_equal(X[observed], original_X[observed])
            np.testing.assert_array_equal(X[:, :, 24:], original_X[:, :, 24:])
            normalized = estimator._batch_forward(X, y, output_type="median")
            prediction = estimator.y_scaler_.inverse_transform(normalized.reshape(-1, 1)).reshape(-1)
            assert np.isfinite(prediction).all()
            row = {
                "model": model, "checkpoint": str(path), "case": case,
                "context_rows": len(train), "query_rows": len(query), "features": X.shape[-1],
                "spearman": float(spearmanr(truth, prediction).statistic),
                "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
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
