#!/usr/bin/env bash
#SBATCH --job-name=soccol_baseline_folds
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

OUTPUT_DIR="${OUTPUT_DIR:-corrosion_datasets/analysis/soccol_pipeline/baseline_folds_v1}"
GENERIC_CHECKPOINT="${GENERIC_CHECKPOINT:-checkpoints/tabicl_s1_regression_baseline/step-1000.ckpt}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
RANDOM_STATE="${RANDOM_STATE:-42}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-8}"

python -m scripts.soccol_pipeline.evaluate_baseline_folds \
  --output-dir "$OUTPUT_DIR" \
  --auto-output-dir \
  --generic-checkpoint "$GENERIC_CHECKPOINT" \
  --device cuda \
  --n-estimators "$N_ESTIMATORS" \
  --random-state "$RANDOM_STATE" \
  --catboost-iterations 1000 \
  --catboost-depth 6 \
  --catboost-learning-rate 0.03 \
  --catboost-l2-leaf-reg 3.0 \
  --catboost-thread-count "$CATBOOST_THREAD_COUNT"
