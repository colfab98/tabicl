#!/usr/bin/env bash
#SBATCH --job-name=soccol_checkpoint_compare
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

OUTPUT_DIR="${OUTPUT_DIR:-corrosion_datasets/analysis/soccol_pipeline/baseline_folds_v1}"
GENERIC_RUN="${GENERIC_RUN:-checkpoints/tabicl_s1_regression_baseline}"
CANDIDATE_RUN="${CANDIDATE_RUN:-}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-soccol_epit}"
DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
RANDOM_STATE="${RANDOM_STATE:-42}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-${SLURM_CPUS_PER_TASK:-8}}"

EXTRA_ARGS=()
if [[ -n "$CANDIDATE_RUN" ]]; then
  EXTRA_ARGS+=(
    --candidate-run "$CANDIDATE_RUN"
    --candidate-label "$CANDIDATE_LABEL"
  )
fi

python -m scripts.soccol_pipeline.evaluate_baseline_folds \
  --output-dir "$OUTPUT_DIR" \
  --auto-output-dir \
  --generic-run "$GENERIC_RUN" \
  --device "$DEVICE" \
  --n-estimators "$N_ESTIMATORS" \
  --random-state "$RANDOM_STATE" \
  --min-checkpoint-step 500 \
  --checkpoint-step-interval 500 \
  --catboost-iterations 1000 \
  --catboost-depth 6 \
  --catboost-learning-rate 0.03 \
  --catboost-l2-leaf-reg 3.0 \
  --catboost-thread-count "$CATBOOST_THREAD_COUNT" \
  "${EXTRA_ARGS[@]}"
