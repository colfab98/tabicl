#!/usr/bin/env bash
#SBATCH --job-name=pitting_baseline_1000
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

python scripts/eval_pitting_repeated_splits.py \
  --local-ckpt-path checkpoints/tabicl_s1_regression_baseline/step-1000.ckpt \
  --model-label regression_baseline_1000 \
  --split-seeds 1001 1002 1003 1004 1005 \
  --device cuda