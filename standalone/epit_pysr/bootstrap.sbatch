#!/usr/bin/env bash
#SBATCH --job-name=epit_pysr_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

BUNDLE_DIR="${SLURM_SUBMIT_DIR}"
export PYTHON_JULIACALL_THREADS="${SLURM_CPUS_PER_TASK}"
bash "${BUNDLE_DIR}/bootstrap_offline.sh"
