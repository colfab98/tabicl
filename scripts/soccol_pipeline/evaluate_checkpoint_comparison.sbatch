#!/usr/bin/env bash
#SBATCH --job-name=soccol_final_compare
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

SPLIT_MANIFEST="${SPLIT_MANIFEST:-corrosion_datasets/datasets/soccol_pitting_potential/processed/splits_v1/split_manifest.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/fcolanto/projects/tabicl/checkpoints}"
GENERIC_RUN="${GENERIC_RUN:-/home/fcolanto/projects/tabicl/checkpoints/tabicl_s1_regression_baseline}"
CANDIDATE_RUN="${CANDIDATE_RUN:-/home/fcolanto/projects/tabicl/checkpoints/soccol_pipeline_final_v1/final_soccol_pipeline_optuna_empirical_features_scm_target_v1_trial_0019}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-soccol_epit_trial19}"
RUN_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_ROOT="${OUTPUT_ROOT:-corrosion_datasets/analysis/soccol_pipeline/final_checkpoint_comparison_${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
RANDOM_STATE="${RANDOM_STATE:-42}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-${SLURM_CPUS_PER_TASK:-8}}"

if [[ ! -d "$CANDIDATE_RUN" ]]; then
  echo "Soccol checkpoint run not found: $CANDIDATE_RUN" >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output path already exists: $OUTPUT_ROOT" >&2
  exit 1
fi

python scripts/eval_corrosion_datasets.py \
  --run "$GENERIC_RUN" \
  --run "$CANDIDATE_RUN" \
  --local-model-label generic_baseline \
  --local-model-label "$CANDIDATE_LABEL" \
  --checkpoint all \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --run-prefix / \
  --min-checkpoint-step 0 \
  --checkpoint-step-interval 0 \
  --include-latest-common-checkpoint \
  --task soccol_pitting_potential__event1__epit_mv_agagcl_3m_kcl \
  --target-mode primary \
  --target-binning continuous \
  --epit-split-manifest "$SPLIT_MANIFEST" \
  --epit-final-test \
  --device "$DEVICE" \
  --n-estimators "$N_ESTIMATORS" \
  --random-state "$RANDOM_STATE" \
  --tabicl-feat-shuffle-method none \
  --tabicl-norm-methods none \
  --regression-output median \
  --no-regression-uncertainty \
  --max-samples-per-task 0 \
  --compare-pretrained-tabicl \
  --pretrained-checkpoint-version tabicl-regressor-v2-20260212.ckpt \
  --compare-catboost \
  --catboost-iterations 1000 \
  --catboost-depth 6 \
  --catboost-learning-rate 0.03 \
  --catboost-l2-leaf-reg 3.0 \
  --catboost-thread-count "$CATBOOST_THREAD_COUNT" \
  --catboost-categorical-column Prep_medium \
  --catboost-categorical-column CP_aeration \
  --catboost-categorical-column CP_agitation \
  --catboost-categorical-column CP_anions_info \
  --output-json "${OUTPUT_ROOT}/results.json" \
  --output-csv "${OUTPUT_ROOT}/rows.csv" \
  --output-wide-csv "${OUTPUT_ROOT}/wide.csv" \
  --output-summary-csv "${OUTPUT_ROOT}/summary.csv" \
  --output-plot-dir "${OUTPUT_ROOT}/plots"

EXPECTED_PLOTS=(
  trend_mean_test_spearman.svg
  trend_mean_test_mae.svg
  trend_mean_test_nmae_iqr.svg
  trend_mean_test_nrmse_iqr.svg
  trend_mean_test_pearson.svg
  trend_mean_test_r2.svg
)
for plot in "${EXPECTED_PLOTS[@]}"; do
  if [[ ! -s "${OUTPUT_ROOT}/plots/${plot}" ]]; then
    echo "Required plot missing: ${OUTPUT_ROOT}/plots/${plot}" >&2
    exit 1
  fi
done

echo "Completed Soccol final-test checkpoint comparison: $OUTPUT_ROOT"
