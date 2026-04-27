#!/usr/bin/env bash
set -euo pipefail

RUN_OR_CKPT="${1:-}"
CHECKPOINT="${2:-}"
MODEL_LABEL="${3:-}"

if [ -z "$RUN_OR_CKPT" ] || [ -z "$CHECKPOINT" ]; then
  echo "Usage: $0 <run_suffix_or_checkpoint_path[,run_suffix...]> <checkpoint_step_or_name> [job_or_model_label]"
  echo "Examples:"
  echo "  $0 v15 600"
  echo "  $0 v12,v13,v14,v15 2500"
  echo "  $0 v15 step-3350"
  echo "  $0 /path/to/step-3350.ckpt step-3350 custom_label"
  exit 1
fi

mkdir -p /home/"$USER"/tmp

PASS_MODEL_LABEL=1
if [[ "$RUN_OR_CKPT" == *.ckpt || "$RUN_OR_CKPT" == /* ]]; then
  CKPT_ARGS=(--local-ckpt-path "$RUN_OR_CKPT")
  CKPT_BASENAME="$(basename "$RUN_OR_CKPT" .ckpt)"
  RUN_LABEL="$(basename "$(dirname "$RUN_OR_CKPT")")"
  LABEL="${MODEL_LABEL:-${RUN_LABEL}_${CKPT_BASENAME}}"
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
  CKPT_LABEL="$CHECKPOINT"
  CKPT_LABEL="${CKPT_LABEL%.ckpt}"
  CKPT_LABEL="${CKPT_LABEL#step-}"
  JOINED_RUNS="${RUN_LABELS[*]}"
  JOINED_RUNS="${JOINED_RUNS// /_}"
  LABEL="${MODEL_LABEL:-compare_${JOINED_RUNS}_step${CKPT_LABEL}}"
  PASS_MODEL_LABEL=0
else
  if [ "$CHECKPOINT" = "latest" ]; then
    echo "Do not use 'latest'. Pass an explicit checkpoint, e.g. 600 or step-600." >&2
    exit 1
  fi
  CKPT_ARGS=(--run "$RUN_OR_CKPT" --checkpoint "$CHECKPOINT")
  CKPT_LABEL="$CHECKPOINT"
  CKPT_LABEL="${CKPT_LABEL%.ckpt}"
  CKPT_LABEL="${CKPT_LABEL#step-}"
  LABEL="${MODEL_LABEL:-${RUN_OR_CKPT}_step${CKPT_LABEL}}"
fi

EVAL_ARGS=("${CKPT_ARGS[@]}")
if [ "$PASS_MODEL_LABEL" = "1" ]; then
  EVAL_ARGS+=(--local-model-label "$LABEL")
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

python scripts/eval_corrosion_datasets.py \
  ${EVAL_ARGS_STR}\
  --target-mode primary \
  --compare-pretrained-tabicl
EOF
