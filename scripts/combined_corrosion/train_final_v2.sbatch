#!/usr/bin/env bash
#SBATCH --job-name=combined_corr_final_v2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl

export QUERY_COLUMN_MASK_PROBABILITY="${QUERY_COLUMN_MASK_PROBABILITY:-0.8}"
export PRESERVE_OBSERVED_COMPOSITION=True
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/combined_epit_soccol_mpea/final_v2_seed42}"
export WANDB_NAME="${WANDB_NAME:-combined_epit_soccol_mpea_final_v2_seed42}"

# A new augmentation policy starts a fresh run, without overwriting checkpoints.
if compgen -G "$CHECKPOINT_DIR/step-*.ckpt" > /dev/null; then
  echo "Checkpoint directory already contains a run: $CHECKPOINT_DIR" >&2
  exit 1
fi

exec bash scripts/combined_corrosion/train_final.sbatch
