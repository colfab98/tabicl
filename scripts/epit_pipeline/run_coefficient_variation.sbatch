#!/usr/bin/env bash
#SBATCH --job-name=epit_coefficient_variation_2k
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-epit_coefficient_variation_2k_v1}"
OPTUNA_STORAGE_ROOT="${OPTUNA_STORAGE_ROOT:-/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna}"
STORAGE="${STORAGE:-journal://${OPTUNA_STORAGE_ROOT}/${STUDY_NAME}/study_journal.log}"

python -m scripts.epit_pipeline.run_coefficient_variation \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  "$@"
