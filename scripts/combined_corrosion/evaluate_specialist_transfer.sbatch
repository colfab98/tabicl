#!/usr/bin/env bash
#SBATCH --job-name=specialist_transfer_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

EPIT_RUN="${EPIT_RUN:-checkpoints/epit_pipeline_final_v2/final_epit_pipeline_optuna_empirical_features_scm_target_v8_trial_0017}"
SOCCOL_RUN="${SOCCOL_RUN:-checkpoints/soccol_pipeline_final_v1/final_soccol_pipeline_optuna_empirical_features_scm_target_v1_trial_0019}"
CHECKPOINT="${CHECKPOINT:-all}"
RUN_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-corrosion_datasets/analysis/combined_epit_soccol_mpea/specialist_transfer_${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
RANDOM_STATE="${RANDOM_STATE:-42}"

python -m scripts.combined_corrosion.evaluate_specialist_transfer \
  --epit-run "$EPIT_RUN" \
  --soccol-run "$SOCCOL_RUN" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --n-estimators "$N_ESTIMATORS" \
  --random-state "$RANDOM_STATE" \
  --output-dir "$OUTPUT_DIR"

echo "Completed specialist-transfer evaluation: $OUTPUT_DIR"
