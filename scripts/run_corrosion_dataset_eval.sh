#!/usr/bin/env bash
set -euo pipefail

RUN_OR_CKPT="${1:-}"
CHECKPOINT="${2:-}"
MODEL_LABEL="${3:-}"
TARGET_MODE="${4:-continuous}"
RUN_PREFIX_MODE="${5-${RUN_PREFIX:-auto}}"
EXTRA_EVAL_ARGS=("${@:6}")

if [ -z "$RUN_OR_CKPT" ] || [ -z "$CHECKPOINT" ]; then
  echo "Usage: $0 <run_suffix_or_checkpoint_path[,run_suffix...]> <checkpoint_step_or_name> [job_or_model_label] [continuous|target_bins] [run_prefix|auto]"
  echo "Examples:"
  echo "  $0 v19 1000 v19_step1000_reg continuous"
  echo "  $0 tabicl_s1_regression_baseline,tabicl_s1_regression_test4 all compare_reg_all continuous"
  echo "  $0 v15 600 v15_step600_bins3 3"
  echo "  $0 v15 600 v15_step600_bins5 5"
  echo "  $0 v12,v13,v14,v15 2500 compare_bins3 3"
  echo "  $0 v12,v13,v14,v15 all compare_all_bins5 5"
  echo "  $0 /path/to/step-3350.ckpt step-3350 custom_bins3 3"
  echo
  echo "For checkpoint='all', override defaults with MIN_CHECKPOINT_STEP and CHECKPOINT_STEP_INTERVAL."
  echo "The wrapper defaults are stage-aware: s1 every 500 steps, s2 every 100 steps, fallback every 1000 steps."
  echo "The latest common checkpoint is included by default; set INCLUDE_LATEST_COMMON_CHECKPOINT=0 to disable."
  echo "Set CHECKPOINT_STEP_INTERVAL=0 to evaluate every common checkpoint."
  echo "Pass extra eval_corrosion_datasets.py args after the run_prefix argument, e.g. --task <task_id>."
  echo "Set COMPARE_PRETRAINED_TABICL=0 to skip pretrained TabICL in the submitted eval."
  echo "Set DRY_RUN=1 to print resolved arguments without submitting sbatch."
  exit 1
fi

all_run_tokens_are_full_dir_names() {
  local input="$1"
  local token
  local found=0
  local -a tokens
  IFS=',' read -r -a tokens <<< "$input"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    if [ -z "$token" ]; then
      continue
    fi
    found=1
    if [[ "$token" != tabicl_* ]]; then
      return 1
    fi
  done
  [ "$found" = "1" ]
}

short_model_label() {
  local token="$1"
  local label
  label="$(basename "$token")"
  label="${label#tabicl_s1_regression_}"
  label="${label#tabicl_s2_regression_}"
  label="${label#tabicl_s1_}"
  label="${label#tabicl_s2_}"
  label="${label#pitting_optuna_}"
  if [[ "$label" =~ ^best_trial([0-9]+)$ ]]; then
    label="trial${BASH_REMATCH[1]}_optuna_best"
  elif [[ "$label" =~ ^worst_trial([0-9]+)$ ]]; then
    label="trial${BASH_REMATCH[1]}_optuna_worst"
  fi
  printf '%s' "$label"
}

extra_args_have_model_labels() {
  local arg
  for arg in "${EXTRA_EVAL_ARGS[@]}"; do
    if [[ "$arg" == "--local-model-label" || "$arg" == "--model-label" ]]; then
      return 0
    fi
  done
  return 1
}

default_checkpoint_step_interval() {
  local text="$RUN_PREFIX_MODE $RUN_OR_CKPT"
  if [[ "$text" == *tabicl_s2_* ]]; then
    echo 100
  elif [[ "$text" == *tabicl_s1_* || "$text" == *tabicl_s1mini* ]]; then
    echo 500
  else
    echo 1000
  fi
}

if [[ "$TARGET_MODE" != "continuous" && "$TARGET_MODE" != "regression" ]]; then
  if ! [[ "$TARGET_MODE" =~ ^[0-9]+$ ]] || [ "$TARGET_MODE" -lt 2 ]; then
    echo "fourth argument must be 'continuous' or an integer target_bins >= 2." >&2
    exit 1
  fi
fi

REGRESSION_EVAL=0
if [[ "$TARGET_MODE" == "continuous" || "$TARGET_MODE" == "regression" ]]; then
  REGRESSION_EVAL=1
fi

if [ "$REGRESSION_EVAL" = "0" ]; then
  TARGET_BINS="$TARGET_MODE"
else
  TARGET_BINS=""
fi

mkdir -p /home/"$USER"/tmp

PASS_MODEL_LABEL=1
if [[ "$RUN_OR_CKPT" == *.ckpt || "$RUN_OR_CKPT" == /* ]]; then
  if [ "$CHECKPOINT" = "all" ]; then
    echo "Checkpoint 'all' requires run suffixes, not an explicit checkpoint path." >&2
    exit 1
  fi
  CKPT_ARGS=(--local-ckpt-path "$RUN_OR_CKPT")
  CKPT_BASENAME="$(basename "$RUN_OR_CKPT" .ckpt)"
  RUN_LABEL="$(basename "$(dirname "$RUN_OR_CKPT")")"
  if [ "$REGRESSION_EVAL" = "1" ]; then
    LABEL="${MODEL_LABEL:-${RUN_LABEL}_${CKPT_BASENAME}_reg}"
  else
    LABEL="${MODEL_LABEL:-${RUN_LABEL}_${CKPT_BASENAME}_bins${TARGET_BINS}}"
  fi

elif [[ "$RUN_OR_CKPT" == *,* ]]; then
  if [ "$CHECKPOINT" = "latest" ]; then
    echo "Do not use 'latest'. Pass an explicit checkpoint, e.g. 600 or step-600." >&2
    exit 1
  fi

  IFS=',' read -r -a RUNS <<< "$RUN_OR_CKPT"
  CKPT_ARGS=()
  RUN_LABELS=()

  for RUN in "${RUNS[@]}"; do
    RUN="${RUN//[[:space:]]/}"
    if [ -z "$RUN" ]; then
      continue
    fi
    CKPT_ARGS+=(--run "$RUN")
    RUN_LABELS+=("$(short_model_label "$RUN")")
  done

  if ! extra_args_have_model_labels; then
    for LABEL_ITEM in "${RUN_LABELS[@]}"; do
      CKPT_ARGS+=(--local-model-label "$LABEL_ITEM")
    done
  fi

  if [ "${#RUN_LABELS[@]}" -eq 0 ]; then
    echo "No run suffixes found in '$RUN_OR_CKPT'." >&2
    exit 1
  fi

  CKPT_ARGS+=(--checkpoint "$CHECKPOINT")

  if [ "$CHECKPOINT" = "all" ]; then
    CKPT_LABEL="all_common"
  else
    CKPT_LABEL="$CHECKPOINT"
    CKPT_LABEL="${CKPT_LABEL%.ckpt}"
    CKPT_LABEL="${CKPT_LABEL#step-}"
    CKPT_LABEL="step${CKPT_LABEL}"
  fi

  JOINED_RUNS="${RUN_LABELS[*]}"
  JOINED_RUNS="${JOINED_RUNS// /_}"
  if [ "$REGRESSION_EVAL" = "1" ]; then
    LABEL="${MODEL_LABEL:-compare_${JOINED_RUNS}_${CKPT_LABEL}_reg}"
  else
    LABEL="${MODEL_LABEL:-compare_${JOINED_RUNS}_${CKPT_LABEL}_bins${TARGET_BINS}}"
  fi
  PASS_MODEL_LABEL=0

else
  if [ "$CHECKPOINT" = "latest" ]; then
    echo "Do not use 'latest'. Pass an explicit checkpoint, e.g. 600 or step-600." >&2
    exit 1
  fi

  CKPT_ARGS=(--run "$RUN_OR_CKPT" --checkpoint "$CHECKPOINT")

  if [ "$CHECKPOINT" = "all" ]; then
    CKPT_LABEL="all_common"
    PASS_MODEL_LABEL=0
  else
    CKPT_LABEL="$CHECKPOINT"
    CKPT_LABEL="${CKPT_LABEL%.ckpt}"
    CKPT_LABEL="${CKPT_LABEL#step-}"
    CKPT_LABEL="step${CKPT_LABEL}"
  fi

  if [ "$REGRESSION_EVAL" = "1" ]; then
    LABEL="${MODEL_LABEL:-${RUN_OR_CKPT}_${CKPT_LABEL}_reg}"
  else
    LABEL="${MODEL_LABEL:-${RUN_OR_CKPT}_${CKPT_LABEL}_bins${TARGET_BINS}}"
  fi
fi

EVAL_ARGS=("${CKPT_ARGS[@]}")

if [[ "$RUN_OR_CKPT" != *.ckpt && "$RUN_OR_CKPT" != /* ]]; then
  if [ "$RUN_PREFIX_MODE" = "auto" ]; then
    if all_run_tokens_are_full_dir_names "$RUN_OR_CKPT"; then
      EVAL_ARGS+=(--run-prefix "")
    fi
  else
    EVAL_ARGS+=(--run-prefix "$RUN_PREFIX_MODE")
  fi
fi

if [ "$CHECKPOINT" = "all" ]; then
  MIN_CHECKPOINT_STEP="${MIN_CHECKPOINT_STEP:-0}"
  CHECKPOINT_STEP_INTERVAL="${CHECKPOINT_STEP_INTERVAL:-$(default_checkpoint_step_interval)}"
  INCLUDE_LATEST_COMMON_CHECKPOINT="${INCLUDE_LATEST_COMMON_CHECKPOINT:-1}"
  if ! [[ "$MIN_CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
    echo "MIN_CHECKPOINT_STEP must be an integer >= 0." >&2
    exit 1
  fi
  if ! [[ "$CHECKPOINT_STEP_INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "CHECKPOINT_STEP_INTERVAL must be an integer >= 0; use 0 for every common checkpoint." >&2
    exit 1
  fi
  if ! [[ "$INCLUDE_LATEST_COMMON_CHECKPOINT" =~ ^[01]$ ]]; then
    echo "INCLUDE_LATEST_COMMON_CHECKPOINT must be 0 or 1." >&2
    exit 1
  fi
  EVAL_ARGS+=(--min-checkpoint-step "$MIN_CHECKPOINT_STEP")
  EVAL_ARGS+=(--checkpoint-step-interval "$CHECKPOINT_STEP_INTERVAL")
  if [ "$INCLUDE_LATEST_COMMON_CHECKPOINT" = "0" ]; then
    EVAL_ARGS+=(--no-include-latest-common-checkpoint)
  else
    EVAL_ARGS+=(--include-latest-common-checkpoint)
  fi
fi

if [ "$PASS_MODEL_LABEL" = "1" ]; then
  EVAL_ARGS+=(--local-model-label "$LABEL")
fi

if [ "${#EXTRA_EVAL_ARGS[@]}" -gt 0 ]; then
  EVAL_ARGS+=("${EXTRA_EVAL_ARGS[@]}")
fi

if [ "$REGRESSION_EVAL" = "1" ]; then
  EVAL_ARGS+=(--target-binning continuous)
elif [ "$TARGET_BINS" = "2" ]; then
  EVAL_ARGS+=(--target-binning median_binary --target-bins 2)
else
  EVAL_ARGS+=(--target-binning quantile_multiclass --target-bins "$TARGET_BINS")
fi

printf -v EVAL_ARGS_STR "%q " "${EVAL_ARGS[@]}"

COMPARE_PRETRAINED_TABICL="${COMPARE_PRETRAINED_TABICL:-1}"
if [ "$COMPARE_PRETRAINED_TABICL" = "0" ]; then
  PRETRAINED_ARG="--no-compare-pretrained-tabicl"
else
  PRETRAINED_ARG="--compare-pretrained-tabicl"
fi

JOB_NAME="tabicl_corrosion_${LABEL}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "JOB_NAME=${JOB_NAME}"
  echo "EVAL_ARGS=${EVAL_ARGS_STR}"
  exit 0
fi

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/${USER}/tmp/${JOB_NAME}.%j.out
#SBATCH --error=/home/${USER}/tmp/${JOB_NAME}.%j.err

set -euo pipefail

cd /home/${USER}/projects/tabicl
source .venv/bin/activate

python scripts/eval_corrosion_datasets.py \\
  ${EVAL_ARGS_STR}\\
  --target-mode primary \\
  ${PRETRAINED_ARG}
EOF
