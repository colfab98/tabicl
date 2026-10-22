"""Post-hoc feature diagnostics on the already frozen MPEA outer folds.

No training, checkpoint selection, repartitioning, or label-based preprocessing.
Step 5000 is the latest checkpoint shared by all models in the existing run.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts import eval_corrosion_datasets as ev
from scripts.combined_corrosion import evaluate_mpea_folds as mf
from tabicl.prior.combined_corrosion import COMBINED_COMPOSITION_COLUMNS
from tabicl.prior.soccol_schema import SOCCOL_COMPOSITION_COLUMNS

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "mpea_only_fivefold_comparison_cpu"
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    metadata = json.loads((PREVIOUS / "results.json").read_text())
    old_rows = pd.read_csv(PREVIOUS / "rows.csv")
    checkpoints = next(c["models"] for c in metadata["checkpoint_evaluations"] if c["step"] == 5000)
    options = mf.parse_args(["--device", "cpu", "--checkpoint", "5000"])
    shared = mf.shared_options(options)
    task, row_ids, hashes = mf.load_mpea(DATA)
    manifest = json.loads((PREVIOUS / "split_manifest.json").read_text())
    assert hashes == manifest["input_sha256"]
    tasks = mf.fold_tasks(task, manifest)
    unsupported = sorted(set(COMBINED_COMPOSITION_COLUMNS) - set(SOCCOL_COMPOSITION_COLUMNS))
    records, predictions = [], []
    destination = OUT / "model_diagnostics"
    destination.mkdir(exist_ok=True)
    for label in ["soccol_shared_v2", "epit_shared_v2", "combined_corrpfn_v2"]:
        shuffle, norms = ev.tabicl_inference_settings(shared, label, pretrained=False)
        factory = ev.make_cached_tabicl_factory(
            partial(ev.make_tabicl_regressor, model_path=checkpoints[label],
                    checkpoint_version=shared.pretrained_checkpoint_version,
                    device="cpu", n_estimators=shared.n_estimators, random_state=42,
                    allow_auto_download=False, feat_shuffle_method=shuffle, norm_methods=norms),
            cache_key=(label, checkpoints[label]), model_cache={},
        )
        for variant in ["original", "hide_soccol_unsupported_composition"]:
            for fold, original in enumerate(tasks, 1):
                X = original.X.copy()
                if variant != "original":
                    X.loc[:, unsupported] = np.nan
                modified = replace(original, X=X)
                captured = {}

                def capturing_factory():
                    estimator = factory()
                    original_predict = estimator.predict

                    def predict(*a, **kw):
                        output = original_predict(*a, **kw)
                        captured["prediction"] = np.asarray(output, dtype=float).copy()
                        return output

                    estimator.predict = predict
                    return estimator

                record = ev.evaluate_estimator(
                    model_label=label, model_kind="local_tabicl_regressor",
                    estimator_factory=capturing_factory, task=modified,
                    task_split=modified.fixed_split, test_size=shared.test_size,
                    random_state=42, regression_output="median",
                    regression_uncertainty=False, preserve_feature_schema=True,
                    expected_preserve_missing_values=True,
                )
                if variant == "original":
                    expected = old_rows.loc[
                        old_rows.checkpoint_step.eq(5000) & old_rows.model.eq(label)
                        & old_rows.outer_fold.eq(fold)
                    ].iloc[0]
                    for metric in ["test_spearman", "test_rmse"]:
                        if not np.isclose(record[metric], expected[metric], atol=1e-4, rtol=1e-5):
                            raise RuntimeError(f"Baseline mismatch: {label}, {fold}, {metric}")
                records.append({"model": label, "variant": variant, "outer_fold": fold,
                                "spearman": record["test_spearman"], "rmse": record["test_rmse"],
                                "mae": record["test_mae"], "r2": record["test_r2"]})
                for index, prediction in zip(modified.fixed_split.test_index, captured["prediction"], strict=True):
                    predictions.append({
                        "row_id": row_ids[index], "outer_fold": fold, "model": label, "variant": variant,
                        "y_true": float(task.y.iloc[index]), "y_pred": float(prediction),
                    })
                pd.DataFrame(records).to_csv(destination / "fold_metrics.csv", index=False)
                pd.DataFrame(predictions).to_csv(destination / "predictions.csv", index=False)
                print(records[-1], flush=True)
    pd.DataFrame(records).groupby(["model", "variant"])[["spearman", "rmse", "mae", "r2"]].agg(
        ["mean", "std"]
    ).to_csv(destination / "summary.csv")
    (destination / "metadata.json").write_text(json.dumps({
        "scope": "Post-hoc diagnostic, existing frozen MPEA outer folds; no checkpoint selection.",
        "checkpoint_step": 5000, "checkpoints": checkpoints, "input_hashes": hashes,
        "masked_columns": unsupported, "mask_scope": "context and query, same fixed 50-column order",
        "baseline_verified_against_previous_run": True, "n_predictions": len(predictions),
    }, indent=2))


if __name__ == "__main__":
    main()
