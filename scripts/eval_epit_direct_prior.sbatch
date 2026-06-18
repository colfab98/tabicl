#!/usr/bin/env bash
#SBATCH --job-name=epit_direct_prior_phase1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

REPO_ROOT="/home/fcolanto/projects/tabicl"
cd "$REPO_ROOT"

# Prefer a venv built on the CPU cluster. A copied venv can contain host-specific
# Python symlinks, e.g. .venv/bin/python3.12 -> /usr/bin/python3.12.
pick_python() {
  if [[ -n "${EPIT_PYTHON:-}" ]]; then
    printf '%s\n' "$EPIT_PYTHON"
    return 0
  fi

  local candidate
  for candidate in \
    "$REPO_ROOT/.venv-cpu/bin/python" \
    "$REPO_ROOT/.venv/bin/python" \
    python3 \
    python
  do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
  echo "No usable Python >=3.10 found." >&2
  echo "Create .venv-cpu on hpc03/sv3000, or submit with EPIT_PYTHON=/path/to/python." >&2
  exit 127
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts:${PYTHONPATH:-}"

if ! "$PYTHON_BIN" -c 'import joblib, numpy, pandas, scipy, sklearn, torch; from tabicl.prior.dataset import PriorDataset' >/dev/null 2>&1; then
  echo "Python exists but required packages are missing or unusable: $PYTHON_BIN" >&2
  "$PYTHON_BIN" -c 'import sys; print("executable:", sys.executable); print("version:", sys.version)' >&2 || true
  echo "Required runtime packages: torch numpy pandas scipy scikit-learn joblib." >&2
  echo "On offline sv3000, build/install these into .venv-cpu from a local wheelhouse or packed conda env." >&2
  exit 127
fi

echo "Host: $(hostname)"
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

PHASE="${PHASE:-phase1}"
N_SYNTH="${N_SYNTH:-32}"
TEMPERATURE="${TEMPERATURE:-0.10}"
RANDOM_STATE="${RANDOM_STATE:-42}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
MAX_ETAS="${MAX_ETAS:-0}"
PHASE2_SOURCE_DIR="${PHASE2_SOURCE_DIR:-}"
PHASE2_TOP_REGIMES="${PHASE2_TOP_REGIMES:-2}"
N_CORE_SAMPLES="${N_CORE_SAMPLES:-64}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
RESUME="${RESUME:-1}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_NAME:-epit_direct_prior_${PHASE}_${JOB_ID}}"
OUT_DIR="${OUT_DIR:-/home/fcolanto/projects/tabicl/corrosion_datasets/analysis/epit_direct_prior/${RUN_NAME}}"

mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

ARGS=(
  --phase "$PHASE"
  --n-synth "$N_SYNTH"
  --temperature "$TEMPERATURE"
  --random-state "$RANDOM_STATE"
  --synthetic-seed "$SYNTHETIC_SEED"
  --output-dir "$OUT_DIR"
  --n-workers "$N_WORKERS"
  --chunk-size "$CHUNK_SIZE"
  --progress-interval "$PROGRESS_INTERVAL"
)

if [[ "$RESUME" == "0" || "$RESUME" == "false" || "$RESUME" == "False" ]]; then
  ARGS+=(--no-resume)
fi


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

"$PYTHON_BIN" scripts/eval_epit_direct_prior.py "${ARGS[@]}"
