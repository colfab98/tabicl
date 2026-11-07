#!/usr/bin/env bash
#SBATCH --job-name=epit_finetune
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail
cd /home/fcolanto/projects/tabicl
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

EXTRA_ARGS=()
if [[ -n "${PARENT_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--finetune-from "$PARENT_CHECKPOINT")
fi
if [[ -n "${FINAL_RUN_NAME:-}" ]]; then
  EXTRA_ARGS+=(--final-run-name "$FINAL_RUN_NAME")
fi

python -m scripts.epit_pipeline.finetune \
  --nproc-per-node "${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}" \
  --override-informed-prior-ratio "${INFORMED_RATIO:-1}" \
  "${EXTRA_ARGS[@]}" "$@"
