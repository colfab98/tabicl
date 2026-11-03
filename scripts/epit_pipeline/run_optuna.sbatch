#!/usr/bin/env bash
#SBATCH --job-name=epit_optuna_empirical_scm_v14
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

STUDY_NAME="${STUDY_NAME:-epit_pipeline_optuna_empirical_features_scm_target_v14}"
N_TRIALS="${N_TRIALS:-50}"
N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
PYTHON_SEED="${PYTHON_SEED:-42}"
PRIOR_STREAM_SEED="${PRIOR_STREAM_SEED:-42}"
PRIOR_N_JOBS="${PRIOR_N_JOBS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
EVAL_N_ESTIMATORS="${EVAL_N_ESTIMATORS:-8}"
RANDOM_SEED="${RANDOM_SEED:-$((42 + SLURM_JOB_ID))}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json}"
TARGET_RULE_SUMMARY="${TARGET_RULE_SUMMARY:-corrosion_datasets/analysis/epit_pipeline/target_rules_v4/calibration_summary.json}"
OPTUNA_STORAGE_ROOT="${OPTUNA_STORAGE_ROOT:-/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna}"
STUDY_DIR="${OPTUNA_STORAGE_ROOT}/${STUDY_NAME}"
STORAGE="${STORAGE:-journal://${STUDY_DIR}/study_journal_v1.log}"
BASE_RUN_NAME="${BASE_RUN_NAME:-tabicl_epit_pipeline_${STUDY_NAME}_${SLURM_JOB_ID}}"

mkdir -p "$STUDY_DIR"

python -m scripts.epit_pipeline.run_optuna \
  --backend optuna \
  --study-name "$STUDY_NAME" \
  --pitting-composition-mode empirical_features_scm_target \
  --storage "$STORAGE" \
  --n-trials "$N_TRIALS" \
  --n-startup-trials "$N_STARTUP_TRIALS" \
  --random-seed "$RANDOM_SEED" \
  --base-run-name "$BASE_RUN_NAME" \
  --split-manifest "$SPLIT_MANIFEST" \
  --target-rule-summary "$TARGET_RULE_SUMMARY" \
  --device cuda \
  --nproc-per-node "$NPROC_PER_NODE" \
  --np-seed "$NP_SEED" \
  --torch-seed "$TORCH_SEED" \
  --python-seed "$PYTHON_SEED" \
  --prior-stream-seed "$PRIOR_STREAM_SEED" \
  --prior-n-jobs "$PRIOR_N_JOBS" \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --dataloader-prefetch-factor "$DATALOADER_PREFETCH_FACTOR" \
  --eval-n-estimators "$EVAL_N_ESTIMATORS" \
  --max-steps "$MAX_STEPS" \
  --scheduler-total-steps "$SCHEDULER_TOTAL_STEPS"
