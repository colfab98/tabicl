#!/usr/bin/env bash
#SBATCH --job-name=shared_specialist_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

EPIT_RUN="${EPIT_RUN:-checkpoints/combined_epit_soccol_mpea/epit_shared_v2_seed42}"
SOCCOL_RUN="${SOCCOL_RUN:-checkpoints/combined_epit_soccol_mpea/soccol_shared_v2_seed42}"
COMBINED_RUN="${COMBINED_RUN:-checkpoints/combined_epit_soccol_mpea/final_v2_seed42}"
GENERIC_RUN="${GENERIC_RUN:-checkpoints/tabicl_s1_regression_baseline}"
CHECKPOINT="${CHECKPOINT:-all}"
MIN_CHECKPOINT_STEP="${MIN_CHECKPOINT_STEP:-0}"
CHECKPOINT_STEP_INTERVAL="${CHECKPOINT_STEP_INTERVAL:-500}"
RUN_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-corrosion_datasets/analysis/combined_epit_soccol_mpea/shared_specialist_comparison_${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
RANDOM_STATE="${RANDOM_STATE:-42}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

python -m scripts.combined_corrosion.evaluate_shared_specialists \
  --epit-run "$EPIT_RUN" \
  --soccol-run "$SOCCOL_RUN" \
  --combined-run "$COMBINED_RUN" \
  --generic-run "$GENERIC_RUN" \
  --checkpoint "$CHECKPOINT" \
  --min-checkpoint-step "$MIN_CHECKPOINT_STEP" \
  --checkpoint-step-interval "$CHECKPOINT_STEP_INTERVAL" \
  --device "$DEVICE" \
  --n-estimators "$N_ESTIMATORS" \
  --random-state "$RANDOM_STATE" \
  --catboost-thread-count "$CATBOOST_THREAD_COUNT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo "Completed shared specialist comparison: $OUTPUT_DIR"
