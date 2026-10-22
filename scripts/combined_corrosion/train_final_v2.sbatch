#!/usr/bin/env bash
#SBATCH --job-name=combined_corr_final_v2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl

export COMBINED_V2_RUN_MODE=fresh
export COMBINED_DRY_RUN=False
for argument in "$@"; do
  case "$argument" in
    --resume) export COMBINED_V2_RUN_MODE=resume ;;
    --dry-run) export COMBINED_DRY_RUN=True ;;
    *) echo "Usage: $0 [--resume] [--dry-run]" >&2; exit 2 ;;
  esac
done

export QUERY_COLUMN_MASK_PROBABILITY="${QUERY_COLUMN_MASK_PROBABILITY:-0.8}"
export PRESERVE_OBSERVED_COMPOSITION=True
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/combined_epit_soccol_mpea/final_v2_seed42}"
export WANDB_NAME="${WANDB_NAME:-combined_epit_soccol_mpea_final_v2_seed42}"

# The shared launcher validates fresh/resume mode while holding the run lock.
exec bash scripts/combined_corrosion/train_final.sbatch
