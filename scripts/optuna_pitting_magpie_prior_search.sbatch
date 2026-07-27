#!/usr/bin/env bash
#SBATCH --job-name=pitting_magpie_search
#SBATCH --array=0-59%3
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/tmp/%x.%A_%a.out
#SBATCH --error=/home/%u/tmp/%x.%A_%a.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-pitting_magpie_conditional_v1}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-1}"
MAX_STEPS="${MAX_STEPS:-4000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-8000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RANDOM_SEED="$((42 + SLURM_ARRAY_TASK_ID))"
STUDY_DIR="/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/pitting_magpie_prior_search/${STUDY_NAME}"
STORAGE="${STORAGE:-journal://${STUDY_DIR}/${STUDY_NAME}.log}"
BASE_RUN_NAME="${BASE_RUN_NAME:-tabicl_pitting_magpie_${STUDY_NAME}_${SLURM_ARRAY_JOB_ID}}"

mkdir -p "$STUDY_DIR"

CONTROL_ARGS=()
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  CONTROL_ARGS+=(--enqueue-controls)
fi

python scripts/optuna_pitting_magpie_prior_search.py \
  --backend optuna \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --n-trials "$TRIALS_PER_WORKER" \
  --random-seed "$RANDOM_SEED" \
  --base-run-name "$BASE_RUN_NAME" \
  --split-seeds 2001 2002 2003 \
  --device cuda \
  --nproc-per-node "$NPROC_PER_NODE" \
  --max-steps "$MAX_STEPS" \
  --scheduler-total-steps "$SCHEDULER_TOTAL_STEPS" \
  "${CONTROL_ARGS[@]}"
