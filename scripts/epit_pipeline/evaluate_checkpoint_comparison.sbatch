#!/usr/bin/env bash
#SBATCH --job-name=epit_trial36_compare
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

SPLIT_MANIFEST="${SPLIT_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/fcolanto/projects/tabicl/checkpoints}"
GENERIC_RUN="${GENERIC_RUN:-/home/fcolanto/projects/tabicl/checkpoints/tabicl_s1_regression_baseline}"
TRIAL_RUN="${TRIAL_RUN:-/home/fcolanto/projects/tabicl/checkpoints/epit_pipeline_final_v1/final_epit_pipeline_optuna_v3_trial_0036}"
OUTPUT_ROOT="${OUTPUT_ROOT:-corrosion_datasets/analysis/epit_pipeline/trial36_checkpoint_folds_${SLURM_JOB_ID}}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-${SLURM_CPUS_PER_TASK:-8}}"

for FOLD in 1 2 3 4 5; do
  FOLD_OUTPUT="${OUTPUT_ROOT}/fold_${FOLD}"
  python scripts/eval_corrosion_datasets.py \
    --run "$GENERIC_RUN" \
    --run "$TRIAL_RUN" \
    --local-model-label generic_baseline \
    --local-model-label trial_0036 \
    --checkpoint all \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --run-prefix "/" \
    --min-checkpoint-step 500 \
    --checkpoint-step-interval 500 \
    --include-latest-common-checkpoint \
    --task electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg \
    --target-mode primary \
    --target-binning continuous \
    --epit-split-manifest "$SPLIT_MANIFEST" \
    --epit-validation-fold "$FOLD" \
    --device cuda \
    --n-estimators "$N_ESTIMATORS" \
    --tabicl-feat-shuffle-method none \
    --tabicl-norm-methods none \
    --regression-output median \
    --no-regression-uncertainty \
    --max-samples-per-task 0 \
    --pitting-magpie-model trial_0036 \
    --compare-pretrained-tabicl \
    --pretrained-checkpoint-version tabicl-regressor-v2-20260212.ckpt \
    --compare-catboost \
    --catboost-thread-count "$CATBOOST_THREAD_COUNT" \
    --output-json "${FOLD_OUTPUT}/results.json" \
    --output-csv "${FOLD_OUTPUT}/rows.csv" \
    --output-wide-csv "${FOLD_OUTPUT}/wide.csv" \
    --output-summary-csv "${FOLD_OUTPUT}/summary.csv" \
    --output-plot-dir "${FOLD_OUTPUT}/plots"
done

python - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
from scripts.eval_corrosion_datasets import (
    DEFAULT_REGRESSION_PLOT_METRICS,
    write_checkpoint_trend_plots,
)

from scripts.epit_pipeline.evaluate_optuna_folds import summarize_rows

root = Path(sys.argv[1]).resolve()
frames = []
for fold in range(1, 6):
    fold_dir = root / f"fold_{fold}"
    result = json.loads((fold_dir / "results.json").read_text(encoding="utf-8"))
    if result.get("errors"):
        raise RuntimeError(f"Fold {fold} reported errors: {result['errors']}")
    rows = pd.read_csv(fold_dir / "rows.csv")
    rows.insert(0, "optuna_validation_fold", fold)
    frames.append(rows)
combined = pd.concat(frames, ignore_index=True)
summaries = []
for (name, step), rows in combined.groupby(["checkpoint_name", "checkpoint_step"]):
    summary = summarize_rows(rows)
    summary.insert(0, "checkpoint_step", int(step))
    summary.insert(0, "checkpoint_name", str(name))
    summaries.append(summary)
summary = pd.concat(summaries, ignore_index=True)
root.mkdir(parents=True, exist_ok=True)
combined.to_csv(root / "rows.csv", index=False)
summary.to_csv(root / "summary.csv", index=False)
plot_paths = write_checkpoint_trend_plots(
    summary,
    root / "plots",
    list(DEFAULT_REGRESSION_PLOT_METRICS),
)
print(summary.to_string(index=False))
print(f"Combined results: {root}")
if plot_paths:
    print(f"Five-fold average checkpoint trend plots: {root / 'plots'}")
    for path in plot_paths:
        print(f"  {path}")
PY
