#!/usr/bin/env bash
#SBATCH --job-name=epit_train_final_trial36
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-epit_pipeline_optuna_v3}"
OPTUNA_STORAGE_ROOT="${OPTUNA_STORAGE_ROOT:-/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna}"
STUDY_DIR="${OPTUNA_STORAGE_ROOT}/${STUDY_NAME}"
STORAGE="${STORAGE:-journal://${STUDY_DIR}/study_journal_v1.log}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json}"
TARGET_RULE_SUMMARY="${TARGET_RULE_SUMMARY:-corrosion_datasets/analysis/epit_pipeline/target_rules_v2/calibration_summary.json}"
SELECTED_TRIAL_NUMBER="${SELECTED_TRIAL_NUMBER:-36}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-100}"
SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-500}"
CHECKPOINT_SELECTION_MIN_STEP="${CHECKPOINT_SELECTION_MIN_STEP:-500}"
CHECKPOINT_SELECTION_INTERVAL="${CHECKPOINT_SELECTION_INTERVAL:-500}"
NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
PRIOR_N_JOBS="${PRIOR_N_JOBS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
EVAL_N_ESTIMATORS="${EVAL_N_ESTIMATORS:-8}"

echo "Training EPIT pipeline Trial ${SELECTED_TRIAL_NUMBER} for ${MAX_STEPS} Stage 1 steps"
echo "Scheduler horizon: ${SCHEDULER_TOTAL_STEPS}"
echo "Study: ${STUDY_NAME}"

python -m scripts.epit_pipeline.train_final \
  --study-name "$STUDY_NAME" \
  --selected-trial-number "$SELECTED_TRIAL_NUMBER" \
  --allow-stale-running-trials \
  --allow-missing-selected-trial-artifacts \
  --storage "$STORAGE" \
  --split-manifest "$SPLIT_MANIFEST" \
  --target-rule-summary "$TARGET_RULE_SUMMARY" \
  --device cuda \
  --nproc-per-node "$NPROC_PER_NODE" \
  --max-steps "$MAX_STEPS" \
  --scheduler-total-steps "$SCHEDULER_TOTAL_STEPS" \
  --save-temp-every "$SAVE_TEMP_EVERY" \
  --save-perm-every "$SAVE_PERM_EVERY" \
  --checkpoint-selection-min-step "$CHECKPOINT_SELECTION_MIN_STEP" \
  --checkpoint-selection-interval "$CHECKPOINT_SELECTION_INTERVAL" \
  --np-seed "$NP_SEED" \
  --torch-seed "$TORCH_SEED" \
  --prior-n-jobs "$PRIOR_N_JOBS" \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --dataloader-prefetch-factor "$DATALOADER_PREFETCH_FACTOR" \
  --eval-n-estimators "$EVAL_N_ESTIMATORS"
