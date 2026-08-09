#!/usr/bin/env bash
#SBATCH --job-name=pitting_trial15_compare
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/eval_results/pitting_trial15_all_checkpoints_${SLURM_JOB_ID}}"

echo "Writing evaluation results to ${OUTPUT_ROOT}"

# Evaluate every common checkpoint in one result set. Only trial15_magpie and
# catboost_magpie receive the ten appended Magpie descriptors.
python scripts/eval_corrosion_datasets.py \
  --run tabicl_s1_regression_baseline \
  --run tabicl_s1_regression_pitting_trial15_no_magpie_8k \
  --run tabicl_s1_regression_pitting_magpie_trial15_8k \
  --local-model-label generic_baseline \
  --local-model-label trial15_no_magpie \
  --local-model-label trial15_magpie \
  --checkpoint all \
  --checkpoint-root /home/fcolanto/projects/tabicl/checkpoints \
  --run-prefix "./" \
  --min-checkpoint-step 500 \
  --checkpoint-step-interval 500 \
  --include-latest-common-checkpoint \
  --task electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg \
  --target-mode primary \
  --target-binning continuous \
  --split-seeds 1001 1002 1003 1004 1005 \
  --device cuda \
  --n-estimators 8 \
  --tabicl-feat-shuffle-method none \
  --test-size 0.25 \
  --regression-output median \
  --no-regression-uncertainty \
  --pitting-magpie-model trial15_magpie \
  --compare-pretrained-tabicl \
  --compare-catboost \
  --compare-catboost-magpie \
  --catboost-thread-count "${SLURM_CPUS_PER_TASK}" \
  --plot-metric test_spearman \
  --plot-metric test_mae \
  --plot-metric test_rmse \
  --plot-metric test_r2 \
  --output-json "${OUTPUT_ROOT}/results.json" \
  --output-csv "${OUTPUT_ROOT}/rows.csv" \
  --output-wide-csv "${OUTPUT_ROOT}/wide.csv" \
  --output-summary-csv "${OUTPUT_ROOT}/summary.csv" \
  --output-plot-dir "${OUTPUT_ROOT}/plots"

echo "Evaluation complete: ${OUTPUT_ROOT}"
