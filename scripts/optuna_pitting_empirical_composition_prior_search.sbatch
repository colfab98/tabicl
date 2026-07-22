#!/usr/bin/env bash
#SBATCH --job-name=pitting_emp_comp_search
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

BACKEND="${BACKEND:-optuna}"
# N_TRIALS is per submitted job. The default is one job running all 50 trials.
# For five workers and 50 trials total, submit a 5-task array with N_TRIALS=10.
N_TRIALS="${N_TRIALS:-50}"
STUDY_NAME="${STUDY_NAME:-pitting_fixed_pren_empirical_composition_v1}"
MAX_STEPS="${MAX_STEPS:-1000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
STUDY_DIR="/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/pitting_empirical_composition_prior_search/${STUDY_NAME}"
# Journal storage and its lock file are shared by every node joining this study.
STORAGE="${STORAGE:-journal://${STUDY_DIR}/study_journal_v1.log}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"
BASE_RUN_NAME="${BASE_RUN_NAME:-tabicl_pitting_empirical_composition_${STUDY_NAME}_${SLURM_JOB_ID}_${WORKER_ID}}"

mkdir -p "$STUDY_DIR"

EXTRA_ARGS=()
if [[ -n "$SCHEDULER_TOTAL_STEPS" ]]; then
  EXTRA_ARGS+=(--scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
fi

python scripts/optuna_pitting_empirical_composition_prior_search.py \
  --backend "$BACKEND" \
  --study-name "$STUDY_NAME" \
  --pitting-composition-mode empirical \
  --storage "$STORAGE" \
  --n-trials "$N_TRIALS" \
  --base-run-name "$BASE_RUN_NAME" \
  --split-seeds 1001 1002 1003 1004 1005 \
  --device cuda \
  --nproc-per-node "$NPROC_PER_NODE" \
  --max-steps "$MAX_STEPS" \
  "${EXTRA_ARGS[@]}"
