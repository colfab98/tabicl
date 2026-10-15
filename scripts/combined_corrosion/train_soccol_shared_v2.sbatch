#!/usr/bin/env bash
#SBATCH --job-name=soccol_shared_v2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl

# Original Soccol recipe: 100% Soccol-informed; shared V2 representation.
exec .venv/bin/python -m scripts.combined_corrosion.train_specialist_shared_v2 --source Soccol "$@"
