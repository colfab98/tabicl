#!/usr/bin/env bash
#SBATCH --job-name=mpea_fivefold_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

RUN_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-corrosion_datasets/analysis/combined_epit_soccol_mpea/mpea_only_fivefold_comparison_${RUN_TAG}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

python -m scripts.combined_corrosion.evaluate_mpea_folds \
  --combined-run "${COMBINED_RUN:-checkpoints/combined_epit_soccol_mpea/final_v2_seed42}" \
  --epit-run "${EPIT_RUN:-checkpoints/combined_epit_soccol_mpea/epit_shared_v2_seed42}" \
  --soccol-run "${SOCCOL_RUN:-checkpoints/combined_epit_soccol_mpea/soccol_shared_v2_seed42}" \
  --g50-run "${G50_RUN:-checkpoints/combined_epit_soccol_mpea/final_v2_g50_epit25_soccol25_seed42}" \
  --generic-run "${GENERIC_RUN:-checkpoints/tabicl_s1_regression_baseline}" \
  --checkpoint "${CHECKPOINT:-all}" \
  --min-checkpoint-step "${MIN_CHECKPOINT_STEP:-0}" \
  --checkpoint-step-interval "${CHECKPOINT_STEP_INTERVAL:-500}" \
  --device "${DEVICE:-cuda}" \
  --n-estimators "${N_ESTIMATORS:-8}" \
  --random-state "${RANDOM_STATE:-42}" \
  --split-seed "${SPLIT_SEED:-42}" \
  --catboost-thread-count "${CATBOOST_THREAD_COUNT:-8}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo "Completed MPEA-only five-fold comparison: $OUTPUT_DIR"
