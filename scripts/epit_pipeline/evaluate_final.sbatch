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

FINAL_MODEL_MANIFEST="${FINAL_MODEL_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/final_v1/epit_pipeline_optuna_v2/final_model_manifest.json}"

python -m scripts.epit_pipeline.evaluate_final \
  --final-model-manifest "$FINAL_MODEL_MANIFEST"
