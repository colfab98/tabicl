#!/usr/bin/env bash
#SBATCH --job-name=combined_checkpoint_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

CANDIDATE_RUN="${CANDIDATE_RUN:-checkpoints/combined_epit_soccol_mpea/final_seed42}"
GENERIC_RUN="${GENERIC_RUN:-checkpoints/tabicl_s1_regression_baseline}"
CHECKPOINT="${CHECKPOINT:-all}"
RUN_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-corrosion_datasets/analysis/combined_epit_soccol_mpea/checkpoint_comparison_${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
RANDOM_STATE="${RANDOM_STATE:-42}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-8}"

python -m scripts.combined_corrosion.evaluate_final \
  --candidate-run "$CANDIDATE_RUN" \
  --generic-run "$GENERIC_RUN" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --n-estimators "$N_ESTIMATORS" \
  --random-state "$RANDOM_STATE" \
  --catboost-thread-count "$CATBOOST_THREAD_COUNT" \
  --output-dir "$OUTPUT_DIR"

echo "Completed combined final comparison: $OUTPUT_DIR"
