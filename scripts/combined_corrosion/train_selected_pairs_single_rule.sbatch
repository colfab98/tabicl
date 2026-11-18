#!/usr/bin/env bash
#SBATCH --job-name=combined_pairs_single_rule
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=47:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail
cd /home/fcolanto/projects/tabicl
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
python -m scripts.combined_corrosion.train_selected_pairs_single_rule \
  --device cuda --nproc-per-node "$NPROC_PER_NODE" "$@"
