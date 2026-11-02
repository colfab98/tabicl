#!/usr/bin/env bash
#SBATCH --job-name=epit_eval_final
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

# Let the Python entry point supply the current study default.
EXTRA_ARGS=()
if [[ -n "${FINAL_MODEL_MANIFEST:-}" ]]; then
  EXTRA_ARGS+=(--final-model-manifest "$FINAL_MODEL_MANIFEST")
fi

python -m scripts.epit_pipeline.evaluate_final "${EXTRA_ARGS[@]}" "$@"
