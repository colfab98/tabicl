#!/usr/bin/env bash
set -euo pipefail

RUN_OR_CKPT="${1:-}"
CHECKPOINT="${2:-}"
MODEL_LABEL="${3:-}"
TARGET_MODE="${4:-continuous}"
RUN_PREFIX_MODE="${5-${RUN_PREFIX:-auto}}"

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
  echo "The wrapper defaults to MIN_CHECKPOINT_STEP=0 and CHECKPOINT_STEP_INTERVAL=0, so all common checkpoints are evaluated."
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
    RUN_LABELS+=("$RUN")
  done

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
  CHECKPOINT_STEP_INTERVAL="${CHECKPOINT_STEP_INTERVAL:-0}"
  if ! [[ "$MIN_CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
    echo "MIN_CHECKPOINT_STEP must be an integer >= 0." >&2
    exit 1
  fi
  if ! [[ "$CHECKPOINT_STEP_INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "CHECKPOINT_STEP_INTERVAL must be an integer >= 0; use 0 for every common checkpoint." >&2
    exit 1
  fi
  EVAL_ARGS+=(--min-checkpoint-step "$MIN_CHECKPOINT_STEP")
  EVAL_ARGS+=(--checkpoint-step-interval "$CHECKPOINT_STEP_INTERVAL")
fi

if [ "$PASS_MODEL_LABEL" = "1" ]; then
  EVAL_ARGS+=(--local-model-label "$LABEL")
fi

if [ "$REGRESSION_EVAL" = "1" ]; then
  EVAL_ARGS+=(--target-binning continuous)
elif [ "$TARGET_BINS" = "2" ]; then
  EVAL_ARGS+=(--target-binning median_binary --target-bins 2)
else
  EVAL_ARGS+=(--target-binning quantile_multiclass --target-bins "$TARGET_BINS")
fi

printf -v EVAL_ARGS_STR "%q " "${EVAL_ARGS[@]}"

JOB_NAME="tabicl_corrosion_${LABEL}"

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
  --compare-pretrained-tabicl
EOF
