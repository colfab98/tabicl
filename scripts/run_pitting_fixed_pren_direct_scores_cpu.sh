#!/usr/bin/env bash
set -euo pipefail

TABICL_ROOT="${TABICL_ROOT:-/home/fcolanto/projects/tabicl}"
LOCAL_TRIAL_RESULTS="${LOCAL_TRIAL_RESULTS:-${TABICL_ROOT}/corrosion_datasets/analysis/pitting_fixed_pren_direct_trial_results}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${TABICL_ROOT}/corrosion_datasets/analysis/pitting_fixed_pren_direct_prior_scores}"
STUDY_RUN_DIR="${STUDY_RUN_DIR:-${LOCAL_TRIAL_RESULTS}}"
N_SYNTH="${N_SYNTH:-256}"
N_WORKERS="${N_WORKERS:-32}"
MODEL="${MODEL:-extra_trees}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-200000}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_OUTPUT_ROOT}/direct_scores_${MODEL}_n${N_SYNTH}}"

cd "$TABICL_ROOT"

if [[ ! -d "$STUDY_RUN_DIR" ]]; then
  echo "Missing local trial results directory: $STUDY_RUN_DIR" >&2
  echo "Run install_repo_updates.sh from the fixed_pren_direct_handoff directory first." >&2
  exit 1
fi

source .venv/bin/activate

python scripts/eval_pitting_fixed_pren_trial_direct_prior.py \
  --study-run-dir "$STUDY_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --n-synth "$N_SYNTH" \
  --n-workers "$N_WORKERS" \
  --model "$MODEL" \
  --synthetic-seed "$SYNTHETIC_SEED" \
  "$@"
