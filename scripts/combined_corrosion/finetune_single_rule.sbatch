#!/usr/bin/env bash
#SBATCH --job-name=combined_rule_s2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail
cd /home/fcolanto/projects/tabicl
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m scripts.combined_corrosion.finetune_single_rule \
  --nproc-per-node "${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}" "$@"
