#!/usr/bin/env bash
#SBATCH --job-name=datacor_inhibitor_prior_search
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

BACKEND="${BACKEND:-optuna}"
N_TRIALS="${N_TRIALS:-12}"
STUDY_NAME="${STUDY_NAME:-datacor_inhibitor_prior_search}"
MAX_STEPS="${MAX_STEPS:-1000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
MIN_FEATURES="${MIN_FEATURES:-11}"
MAX_FEATURES="${MAX_FEATURES:-32}"
SPLIT_SEEDS="${SPLIT_SEEDS:-11 22 33 44 55}"
STUDY_DIR="/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/datacor_inhibitor_prior_search/${STUDY_NAME}"
STORAGE="${STORAGE:-sqlite:///${STUDY_DIR}/${STUDY_NAME}.db}"
BASE_RUN_NAME="${BASE_RUN_NAME:-tabicl_datacor_inhibitor_prior_${STUDY_NAME}_${SLURM_JOB_ID}}"

mkdir -p "$STUDY_DIR"

EXTRA_ARGS=()
if [[ -n "$SCHEDULER_TOTAL_STEPS" ]]; then
  EXTRA_ARGS+=(--scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
fi

read -r -a SPLIT_SEED_ARGS <<< "$SPLIT_SEEDS"

python scripts/optuna_datacor_inhibitor_prior_search.py \
  --backend "$BACKEND" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --n-trials "$N_TRIALS" \
  --base-run-name "$BASE_RUN_NAME" \
  --split-seeds "${SPLIT_SEED_ARGS[@]}" \
  --device cuda \
  --nproc-per-node 2 \
  --max-steps "$MAX_STEPS" \
  --min-features "$MIN_FEATURES" \
  --max-features "$MAX_FEATURES" \
  "${EXTRA_ARGS[@]}"
