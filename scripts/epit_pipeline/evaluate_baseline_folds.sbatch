#!/usr/bin/env bash
#SBATCH --job-name=epit_baselines
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

SPLIT_MANIFEST="${SPLIT_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json}"
GENERIC_CHECKPOINT="${GENERIC_CHECKPOINT:-checkpoints/tabicl_s1_regression_baseline/step-1000.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-corrosion_datasets/analysis/epit_pipeline/baseline_folds_v2}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-${SLURM_CPUS_PER_TASK:-8}}"

python -m scripts.epit_pipeline.evaluate_baseline_folds \
  --split-manifest "$SPLIT_MANIFEST" \
  --generic-checkpoint "$GENERIC_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --auto-output-dir \
  --device cuda \
  --n-estimators "$N_ESTIMATORS" \
  --catboost-thread-count "$CATBOOST_THREAD_COUNT"
