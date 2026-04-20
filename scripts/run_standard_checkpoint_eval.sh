#!/usr/bin/env bash
set -euo pipefail

CKPT_PATH="${1:-}"
MODEL_LABEL="${2:-checkpoint_eval}"
JOB_NAME="${3:-tabicl_standard_eval}"

if [ -z "$CKPT_PATH" ]; then
  echo "Usage: $0 <checkpoint_path> [model_label] [job_name]"
  exit 1
fi

mkdir -p /home/"$USER"/tmp

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/${USER}/tmp/${JOB_NAME}.%j.out
#SBATCH --error=/home/${USER}/tmp/${JOB_NAME}.%j.err

set -euo pipefail

cd /home/${USER}/projects/tabicl
source .venv/bin/activate

python scripts/eval_checkpoint_standard.py \
  --local-ckpt-path "${CKPT_PATH}" \
  --local-model-label "${MODEL_LABEL}" \
  --compare-pretrained
EOF
