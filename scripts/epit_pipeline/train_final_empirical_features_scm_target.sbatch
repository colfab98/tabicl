#!/usr/bin/env bash
#SBATCH --job-name=epit_final_empirical_scm_v7
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-epit_pipeline_optuna_empirical_features_scm_target_v7}"
OPTUNA_STORAGE_ROOT="${OPTUNA_STORAGE_ROOT:-/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna}"
STUDY_DIR="${OPTUNA_STORAGE_ROOT}/${STUDY_NAME}"
STORAGE="${STORAGE:-journal://${STUDY_DIR}/study_journal_v1.log}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json}"
TARGET_RULE_SUMMARY="${TARGET_RULE_SUMMARY:-corrosion_datasets/analysis/epit_pipeline/target_rules_v2/calibration_summary.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-100}"
SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-500}"
CHECKPOINT_SELECTION_MIN_STEP="${CHECKPOINT_SELECTION_MIN_STEP:-500}"
CHECKPOINT_SELECTION_INTERVAL="${CHECKPOINT_SELECTION_INTERVAL:-500}"
NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
PRIOR_N_JOBS="${PRIOR_N_JOBS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
EVAL_N_ESTIMATORS="${EVAL_N_ESTIMATORS:-8}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OVERRIDE_INFORMED_PRIOR_RATIO="${OVERRIDE_INFORMED_PRIOR_RATIO:-}"
OVERRIDE_MLP_PROB="${OVERRIDE_MLP_PROB:-}"
OVERRIDE_PITTING_COMPOSITION_PERTURB_STRENGTH="${OVERRIDE_PITTING_COMPOSITION_PERTURB_STRENGTH:-}"
OVERRIDE_USE_MAGPIE="${OVERRIDE_USE_MAGPIE:-}"
OVERRIDE_PITTING_COMPOSITION_FAMILY_PROBS="${OVERRIDE_PITTING_COMPOSITION_FAMILY_PROBS:-}"

EXTRA_ARGS=()
if [[ -n "$FINAL_RUN_NAME" ]]; then
  EXTRA_ARGS+=(--final-run-name "$FINAL_RUN_NAME")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$OVERRIDE_INFORMED_PRIOR_RATIO" ]]; then
  EXTRA_ARGS+=(--override-informed-prior-ratio "$OVERRIDE_INFORMED_PRIOR_RATIO")
fi
if [[ -n "$OVERRIDE_MLP_PROB" ]]; then
  EXTRA_ARGS+=(--override-mlp-prob "$OVERRIDE_MLP_PROB")
fi
if [[ -n "$OVERRIDE_PITTING_COMPOSITION_PERTURB_STRENGTH" ]]; then
  EXTRA_ARGS+=(
    --override-pitting-composition-perturb-strength
    "$OVERRIDE_PITTING_COMPOSITION_PERTURB_STRENGTH"
  )
fi
if [[ -n "$OVERRIDE_USE_MAGPIE" ]]; then
  EXTRA_ARGS+=(--override-use-magpie "$OVERRIDE_USE_MAGPIE")
fi
if [[ -n "$OVERRIDE_PITTING_COMPOSITION_FAMILY_PROBS" ]]; then
  read -r -a FAMILY_PROBS <<< "$OVERRIDE_PITTING_COMPOSITION_FAMILY_PROBS"
  EXTRA_ARGS+=(
    --override-pitting-composition-family-probs
    "${FAMILY_PROBS[@]}"
  )
fi

python -m scripts.epit_pipeline.train_final \
  --study-name "$STUDY_NAME" \
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
  --eval-n-estimators "$EVAL_N_ESTIMATORS" \
  "${EXTRA_ARGS[@]}"
