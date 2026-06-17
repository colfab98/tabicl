#!/usr/bin/env bash
#SBATCH --job-name=epit_direct_prior_phase1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

PHASE="${PHASE:-phase1}"
N_SYNTH="${N_SYNTH:-32}"
TEMPERATURE="${TEMPERATURE:-0.10}"
RANDOM_STATE="${RANDOM_STATE:-42}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
MAX_ETAS="${MAX_ETAS:-0}"
PHASE2_SOURCE_DIR="${PHASE2_SOURCE_DIR:-}"
PHASE2_TOP_REGIMES="${PHASE2_TOP_REGIMES:-2}"
N_CORE_SAMPLES="${N_CORE_SAMPLES:-64}"
RUN_NAME="${RUN_NAME:-epit_direct_prior_${PHASE}_${SLURM_JOB_ID}}"
OUT_DIR="${OUT_DIR:-/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/epit_direct_prior/${RUN_NAME}}"

mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

ARGS=(
  --phase "$PHASE"
  --n-synth "$N_SYNTH"
  --temperature "$TEMPERATURE"
  --random-state "$RANDOM_STATE"
  --synthetic-seed "$SYNTHETIC_SEED"
  --output-dir "$OUT_DIR"
)

if [[ "$PHASE" == "phase1" ]]; then
  ARGS+=(--max-etas "$MAX_ETAS")
elif [[ "$PHASE" == "phase2" ]]; then
  if [[ -z "$PHASE2_SOURCE_DIR" ]]; then
    echo "PHASE2_SOURCE_DIR is required when PHASE=phase2." >&2
    exit 1
  fi
  ARGS+=(
    --phase2-source-dir "$PHASE2_SOURCE_DIR"
    --phase2-top-regimes "$PHASE2_TOP_REGIMES"
    --n-core-samples "$N_CORE_SAMPLES"
  )
else
  echo "PHASE must be phase1 or phase2 for this sbatch script, got '$PHASE'." >&2
  exit 1
fi

python scripts/eval_epit_direct_prior.py "${ARGS[@]}"
