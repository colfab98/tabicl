#!/usr/bin/env bash

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

DEVICE="${DEVICE:-cpu}"
MIN_CHECKPOINT_STEP="${MIN_CHECKPOINT_STEP:-500}"
CHECKPOINT_STEP_INTERVAL="${CHECKPOINT_STEP_INTERVAL:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/eval_results/pitting_t21_material_ablation_common}"

python scripts/eval_corrosion_datasets.py \
  --run tabicl_s1_regression_pitting_t21_e1_6k \
  --run tabicl_s1_regression_pitting_t21_e2_6k \
  --run tabicl_s1_regression_pitting_t21_l_6k \
  --local-model-label t21_e1 \
  --local-model-label t21_e2 \
  --local-model-label t21_l \
  --checkpoint all \
  --checkpoint-root /home/fcolanto/projects/tabicl/checkpoints \
  --run-prefix "./" \
  --min-checkpoint-step "$MIN_CHECKPOINT_STEP" \
  --checkpoint-step-interval "$CHECKPOINT_STEP_INTERVAL" \
  --include-latest-common-checkpoint \
  --task electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg \
  --target-mode primary \
  --target-binning continuous \
  --split-seeds 1001 1002 1003 1004 1005 \
  --device "$DEVICE" \
  --n-estimators 8 \
  --tabicl-feat-shuffle-method none \
  --test-size 0.25 \
  --regression-output median \
  --no-regression-uncertainty \
  --no-compare-pretrained-tabicl \
  --output-json "$OUTPUT_ROOT/results.json" \
  --output-csv "$OUTPUT_ROOT/rows.csv" \
  --output-wide-csv "$OUTPUT_ROOT/wide.csv" \
  --output-summary-csv "$OUTPUT_ROOT/summary.csv" \
  --output-plot-dir "$OUTPUT_ROOT/plots"
