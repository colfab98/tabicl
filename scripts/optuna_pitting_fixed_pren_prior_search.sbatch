#!/usr/bin/env bash
#SBATCH --job-name=pitting_fixed_pren_search
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

BACKEND="${BACKEND:-optuna}"
N_TRIALS="${N_TRIALS:-4}"
STUDY_NAME="${STUDY_NAME:-pitting_fixed_pren_prior_search}"
MAX_STEPS="${MAX_STEPS:-2000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
STUDY_DIR="/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/pitting_fixed_pren_prior_search/${STUDY_NAME}"
STORAGE="${STORAGE:-sqlite:///${STUDY_DIR}/${STUDY_NAME}.db}"
BASE_RUN_NAME="${BASE_RUN_NAME:-tabicl_pitting_fixed_pren_prior_${STUDY_NAME}_${SLURM_JOB_ID}}"

mkdir -p "$STUDY_DIR"

EXTRA_ARGS=()
if [[ -n "$SCHEDULER_TOTAL_STEPS" ]]; then
  EXTRA_ARGS+=(--scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
fi

python scripts/optuna_pitting_fixed_pren_prior_search.py \
  --backend "$BACKEND" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --n-trials "$N_TRIALS" \
  --base-run-name "$BASE_RUN_NAME" \
  --split-seeds 1001 1002 1003 1004 1005 \
  --device cuda \
  --nproc-per-node 2 \
  --max-steps "$MAX_STEPS" \
  "${EXTRA_ARGS[@]}"
