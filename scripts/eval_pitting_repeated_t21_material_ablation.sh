#!/usr/bin/env bash

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

DEVICE="${DEVICE:-cpu}"
MIN_CHECKPOINT_STEP="${MIN_CHECKPOINT_STEP:-500}"
CHECKPOINT_STEP_INTERVAL="${CHECKPOINT_STEP_INTERVAL:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/eval_results/pitting_v8_material_family_ablation_common}"

python scripts/eval_corrosion_datasets.py \
  --run tabicl_s1_regression_pitting_direct_prior_v8_8k \
  --run tabicl_s1_regression_pitting_v8_comp_softmax_8k \
  --run tabicl_s1_regression_pitting_v8_comp_dirichlet_8k \
  --run tabicl_s1_regression_pitting_v8_sparse_alloying_8k \
  --local-model-label v8_original_mixture \
  --local-model-label v8_comp_softmax \
  --local-model-label v8_comp_dirichlet \
  --local-model-label v8_sparse_alloying \
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
