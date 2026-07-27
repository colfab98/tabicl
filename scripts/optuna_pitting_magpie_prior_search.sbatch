#!/usr/bin/env bash
#SBATCH --job-name=pitting_magpie_search
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-pitting_magpie_conditional_v1}"
N_TRIALS="${N_TRIALS:-60}"
MAX_STEPS="${MAX_STEPS:-4000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-8000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RANDOM_SEED="${RANDOM_SEED:-42}"
OPTUNA_STORAGE_ROOT="${OPTUNA_STORAGE_ROOT:-/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna}"
STUDY_DIR="${OPTUNA_STORAGE_ROOT}/${STUDY_NAME}"
STORAGE="${STORAGE:-journal://${STUDY_DIR}/study_journal_v1.log}"
BASE_RUN_NAME="${BASE_RUN_NAME:-tabicl_pitting_magpie_${STUDY_NAME}_${SLURM_JOB_ID}}"

mkdir -p "$STUDY_DIR"

python scripts/optuna_pitting_magpie_prior_search.py \
  --backend optuna \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --n-trials "$N_TRIALS" \
  --random-seed "$RANDOM_SEED" \
  --base-run-name "$BASE_RUN_NAME" \
  --split-seeds 2001 2002 2003 \
  --device cuda \
  --nproc-per-node "$NPROC_PER_NODE" \
  --max-steps "$MAX_STEPS" \
  --scheduler-total-steps "$SCHEDULER_TOTAL_STEPS" \
  --enqueue-controls
