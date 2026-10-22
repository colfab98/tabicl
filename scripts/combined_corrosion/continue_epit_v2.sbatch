#!/usr/bin/env bash
#SBATCH --job-name=epit_v2_continuation
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail
cd /home/fcolanto/projects/tabicl

export EPIT_CONTINUATION_ARM="${1:-}"
case "$EPIT_CONTINUATION_ARM" in
  control) PRIOR_RUN=epit_shared_v2_seed42; RUN_NAME=epit_v2_5000_control_seed42 ;;
  soccol) PRIOR_RUN=soccol_shared_v2_seed42; RUN_NAME=epit_v2_5000_to_soccol_seed42 ;;
  *) echo "Usage: $0 {control|soccol} [--resume] [--dry-run]" >&2; exit 2 ;;
esac
shift

export EPIT_CONTINUATION_PARENT="${EPIT_CONTINUATION_PARENT:-checkpoints/combined_epit_soccol_mpea/epit_shared_v2_seed42/step-5000.ckpt}"
export PRIOR_CONFIG="checkpoints/combined_epit_soccol_mpea/$PRIOR_RUN/prior_config.json"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/combined_epit_soccol_mpea/$RUN_NAME}"
export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
export MAX_STEPS="${MAX_STEPS:-10000}"
exec bash scripts/combined_corrosion/train_final_v2.sbatch "$@"
