#!/usr/bin/env bash
#SBATCH --job-name=epit_pysr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

BUNDLE_DIR="${SLURM_SUBMIT_DIR}"
export PYTHON_JULIACALL_THREADS="${SLURM_CPUS_PER_TASK}"

args=()
if [[ -n "${NITERATIONS:-}" ]]; then
  args+=(--niterations "${NITERATIONS}")
fi
if [[ -n "${POPULATIONS:-}" ]]; then
  args+=(--populations "${POPULATIONS}")
fi
if [[ -n "${SELECT_K_FEATURES:-}" ]]; then
  args+=(--select-k-features "${SELECT_K_FEATURES}")
fi
if [[ -n "${SEEDS:-}" ]]; then
  IFS=':' read -ra seed_values <<< "${SEEDS}"
  for seed in "${seed_values[@]}"; do
    args+=(--seed "${seed}")
  done
fi

case "${MODE:-dry}" in
  dry)
    exec bash "${BUNDLE_DIR}/run.sh" --dry-run "${args[@]}"
    ;;
  full)
    exec bash "${BUNDLE_DIR}/run.sh" "${args[@]}"
    ;;
  *)
    echo "MODE must be 'dry' or 'full'" >&2
    exit 2
    ;;
esac
