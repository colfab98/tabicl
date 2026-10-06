#!/usr/bin/env bash
#SBATCH --job-name=soccol_eval_final
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

FINAL_MODEL_MANIFEST="${FINAL_MODEL_MANIFEST:-corrosion_datasets/analysis/soccol_pipeline/final_v1/soccol_pipeline_optuna_empirical_features_scm_target_v1/final_model_manifest.json}"

python -m scripts.soccol_pipeline.evaluate_final \
  --final-model-manifest "$FINAL_MODEL_MANIFEST"
