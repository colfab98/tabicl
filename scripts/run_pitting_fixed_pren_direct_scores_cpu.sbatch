#!/usr/bin/env bash
#SBATCH --job-name=pitting_fixed_pren_direct
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

REPO_ROOT="${TABICL_ROOT:-/home/fcolanto/projects/tabicl}"
LOCAL_TRIAL_RESULTS="${LOCAL_TRIAL_RESULTS:-${REPO_ROOT}/corrosion_datasets/analysis/pitting_fixed_pren_direct_trial_results}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${REPO_ROOT}/corrosion_datasets/analysis/pitting_fixed_pren_direct_prior_scores}"
STUDY_RUN_DIR="${STUDY_RUN_DIR:-${LOCAL_TRIAL_RESULTS}}"
N_SYNTH="${N_SYNTH:-256}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
MODEL="${MODEL:-extra_trees}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-200000}"
SYNTHETIC_SEQ_LEN="${SYNTHETIC_SEQ_LEN:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
MAX_TRIALS="${MAX_TRIALS:-0}"
TRIAL_NUMBERS="${TRIAL_NUMBERS:-}"
MAX_SCHEMA_ATTEMPTS="${MAX_SCHEMA_ATTEMPTS:-100}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
RESUME="${RESUME:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_OUTPUT_ROOT}/direct_scores_${MODEL}_n${N_SYNTH}}"
PYTHON_BIN="${PYTHON_BIN:-}"

cd "$REPO_ROOT"

if [[ ! -d "$STUDY_RUN_DIR" ]]; then
  echo "Missing local trial results directory: $STUDY_RUN_DIR" >&2
  echo "Run this first from the handoff directory:" >&2
  echo "  bash install_repo_updates.sh" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

SITE_PACKAGES="$REPO_ROOT/.venv/lib/python3.12/site-packages"
if [[ -d "$SITE_PACKAGES" ]]; then
  export PYTHONPATH="$SITE_PACKAGES:$REPO_ROOT/src:$REPO_ROOT/scripts:${PYTHONPATH:-}"
else
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts:${PYTHONPATH:-}"
fi

pick_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  local candidate
  for candidate in \
    "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/.offline_python312/bin/python3.12" \
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
  echo "Run install_offline_venv.sh from the fixed_pren_direct_handoff directory first." >&2
  exit 127
fi

if ! "$PYTHON_BIN" -c 'import joblib, numpy, pandas, scipy, sklearn, torch; from tabicl.prior.dataset import PriorDataset' >/dev/null 2>&1; then
  echo "Python exists but required packages are missing or unusable: $PYTHON_BIN" >&2
  "$PYTHON_BIN" -c 'import sys; print("executable:", sys.executable); print("version:", sys.version)' >&2 || true
  echo "PYTHONPATH: ${PYTHONPATH:-}" >&2
  exit 127
fi

mkdir -p "$OUTPUT_DIR"

echo "Host: $(hostname)"
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Study run dir: $STUDY_RUN_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "MODEL=$MODEL N_SYNTH=$N_SYNTH N_WORKERS=$N_WORKERS SYNTHETIC_SEED=$SYNTHETIC_SEED"
echo "Analysis: one independent score row per trial and synthetic mechanism; no mechanism averaging."

ARGS=(
  --study-run-dir "$STUDY_RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --n-synth "$N_SYNTH"
  --n-workers "$N_WORKERS"
  --model "$MODEL"
  --synthetic-seed "$SYNTHETIC_SEED"
  --synthetic-seq-len "$SYNTHETIC_SEQ_LEN"
  --random-state "$RANDOM_STATE"
  --max-schema-attempts "$MAX_SCHEMA_ATTEMPTS"
  --progress-interval "$PROGRESS_INTERVAL"
)

if [[ "$MAX_TRIALS" != "0" ]]; then
  ARGS+=(--max-trials "$MAX_TRIALS")
fi

if [[ -n "$TRIAL_NUMBERS" ]]; then
  read -r -a TRIAL_NUMBER_ARGS <<< "$TRIAL_NUMBERS"
  ARGS+=(--trial-number "${TRIAL_NUMBER_ARGS[@]}")
fi

if [[ "$RESUME" == "0" || "$RESUME" == "false" || "$RESUME" == "False" ]]; then
  ARGS+=(--no-resume)
fi

"$PYTHON_BIN" scripts/eval_pitting_fixed_pren_trial_direct_prior.py "${ARGS[@]}"
