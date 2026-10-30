#!/usr/bin/env bash
#SBATCH --job-name=epit_reviewed_compare
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${VENV_PATH:-.venv}/bin/activate"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

OPTIONS=(
  --split-manifest "${SPLIT_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json}"
  --generic-run "${GENERIC_RUN:-$PWD/checkpoints/tabicl_s1_regression_baseline}"
  --device "${DEVICE:-cuda}"
  --n-estimators "${N_ESTIMATORS:-8}"
  --random-state "${RANDOM_STATE:-42}"
  --catboost-thread-count "${CATBOOST_THREAD_COUNT:-${SLURM_CPUS_PER_TASK:-8}}"
  --checkpoint-step-interval "${CHECKPOINT_STEP_INTERVAL:-500}"
  --checkpoint "${CHECKPOINT:-all}"
  --min-checkpoint-step "${MIN_CHECKPOINT_STEP:-0}"
  --output-dir "${OUTPUT_ROOT:-corrosion_datasets/analysis/epit_pipeline/reviewed_final_checkpoint_comparison_${SLURM_JOB_ID:-local}}"
)
if [[ -n "${CANDIDATE_RUN:-}" && -n "${FINAL_MODEL_MANIFEST:-}" ]]; then
  echo "Set only one of CANDIDATE_RUN and FINAL_MODEL_MANIFEST." >&2
  exit 1
fi
if [[ -n "${CANDIDATE_RUN:-}" ]]; then OPTIONS+=(--candidate-run "$CANDIDATE_RUN"); fi
if [[ -n "${FINAL_MODEL_MANIFEST:-}" ]]; then OPTIONS+=(--final-model-manifest "$FINAL_MODEL_MANIFEST"); fi
if [[ -n "${EPIT25_RUN:-}${EPIT50_RUN:-}${EPIT75_RUN:-}${EPIT100_RUN:-}" ]]; then
  OPTIONS+=(--ratio-runs
    "${EPIT25_RUN:?Set all four EPIT ratio run paths}"
    "${EPIT50_RUN:?Set all four EPIT ratio run paths}"
    "${EPIT75_RUN:?Set all four EPIT ratio run paths}"
    "${EPIT100_RUN:?Set all four EPIT ratio run paths}")
fi
if [[ -n "${CANDIDATE_LABEL:-}" ]]; then OPTIONS+=(--candidate-label "$CANDIDATE_LABEL"); fi
if [[ -n "${CONTEXT_FOLDS:-}" ]]; then OPTIONS+=(--context-folds "$CONTEXT_FOLDS"); fi
if [[ "${LEGACY_DATA:-false}" == "true" ]]; then OPTIONS+=(--legacy-data); fi
if [[ "${DRY_RUN:-false}" == "true" ]]; then OPTIONS+=(--dry-run); fi

# The Python entry point selects reviewed data, fixed final queries and all
# checkpoints, and rejects incomplete evaluations or missing required plots.
python -m scripts.epit_pipeline.evaluate_checkpoint_comparison "${OPTIONS[@]}"
