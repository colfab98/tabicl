#!/usr/bin/env bash
#SBATCH --job-name=epit_trial56_final_compare
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

SPLIT_MANIFEST="${SPLIT_MANIFEST:-corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/fcolanto/projects/tabicl/checkpoints}"
GENERIC_RUN="${GENERIC_RUN:-/home/fcolanto/projects/tabicl/checkpoints/tabicl_s1_regression_baseline}"
TRIAL_RUN="${TRIAL_RUN:-/home/fcolanto/projects/tabicl/checkpoints/epit_pipeline_final_v1/final_epit_pipeline_optuna_empirical_features_scm_target_v7_trial_0056}"
OUTPUT_ROOT="${OUTPUT_ROOT:-corrosion_datasets/analysis/epit_pipeline/trial56_final_test_all_checkpoints_${SLURM_JOB_ID}}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-${SLURM_CPUS_PER_TASK:-8}}"

python scripts/eval_corrosion_datasets.py \
    --run "$GENERIC_RUN" \
    --run "$TRIAL_RUN" \
    --local-model-label generic_baseline \
    --local-model-label trial_0056 \
    --checkpoint all \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --run-prefix "/" \
    --min-checkpoint-step 0 \
    --checkpoint-step-interval 0 \
    --include-latest-common-checkpoint \
    --task electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg \
    --target-mode primary \
    --target-binning continuous \
    --epit-split-manifest "$SPLIT_MANIFEST" \
    --epit-final-test \
    --device cuda \
    --n-estimators "$N_ESTIMATORS" \
    --tabicl-feat-shuffle-method none \
    --tabicl-norm-methods none \
    --regression-output median \
    --no-regression-uncertainty \
    --max-samples-per-task 0 \
    --compare-pretrained-tabicl \
    --pretrained-checkpoint-version tabicl-regressor-v2-20260212.ckpt \
    --compare-catboost \
    --catboost-thread-count "$CATBOOST_THREAD_COUNT" \
    --output-json "${OUTPUT_ROOT}/results.json" \
    --output-csv "${OUTPUT_ROOT}/rows.csv" \
    --output-wide-csv "${OUTPUT_ROOT}/wide.csv" \
    --output-summary-csv "${OUTPUT_ROOT}/summary.csv" \
    --output-plot-dir "${OUTPUT_ROOT}/plots"
