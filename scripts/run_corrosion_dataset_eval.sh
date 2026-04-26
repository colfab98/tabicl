#!/usr/bin/env bash
set -euo pipefail

RUN_OR_CKPT="${1:-}"
CHECKPOINT="${2:-}"
MODEL_LABEL="${3:-}"

if [ -z "$RUN_OR_CKPT" ] || [ -z "$CHECKPOINT" ]; then
  echo "Usage: $0 <run_suffix_or_checkpoint_path> <checkpoint_step_or_name> [model_label]"
  echo "Examples:"
  echo "  $0 v15 600"
  echo "  $0 v15 step-3350"
  echo "  $0 /path/to/step-3350.ckpt step-3350 custom_label"
  exit 1
fi

mkdir -p /home/"$USER"/tmp

if [[ "$RUN_OR_CKPT" == *.ckpt || "$RUN_OR_CKPT" == /* ]]; then
  CKPT_ARGS=(--local-ckpt-path "$RUN_OR_CKPT")
  CKPT_BASENAME="$(basename "$RUN_OR_CKPT" .ckpt)"
  RUN_LABEL="$(basename "$(dirname "$RUN_OR_CKPT")")"
  LABEL="${MODEL_LABEL:-${RUN_LABEL}_${CKPT_BASENAME}}"
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

JOB_NAME="tabicl_corrosion_${LABEL}"

printf -v CKPT_ARGS_STR "%q " "${CKPT_ARGS[@]}"

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
  ${CKPT_ARGS_STR} \
  --local-model-label "${LABEL}" \
  --target-mode primary \
  --compare-pretrained-tabicl
EOF
