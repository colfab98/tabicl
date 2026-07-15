#!/usr/bin/env bash
#SBATCH --job-name=tabicl_s1_regression_fixed_pren_trial28
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

RUN_NAME="${RUN_NAME:-${SLURM_JOB_NAME}}"
MAX_STEPS="${MAX_STEPS:-10000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-100}"
SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-500}"
PRIOR_N_JOBS="${PRIOR_N_JOBS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

CHECKPOINT_DIR="/home/fcolanto/projects/tabicl/checkpoints/${RUN_NAME}"
if [[ -e "$CHECKPOINT_DIR" ]] && [[ -n "$(find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Checkpoint directory already exists and is non-empty: $CHECKPOINT_DIR" >&2
  echo "Set RUN_NAME to a new value if you want another run." >&2
  exit 1
fi

echo "Training fixed-PREN Optuna best trial 28"
echo "RUN_NAME=$RUN_NAME"
echo "MAX_STEPS=$MAX_STEPS"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"

TRAIN_CMD=(
  torchrun
  --standalone
  --nproc_per_node=2
  /home/fcolanto/projects/tabicl/src/tabicl/train/run.py
  --wandb_log False
  --wandb_project TabICL
  --wandb_name FixedPrenTrial28
  --wandb_dir /home/fcolanto/wandb
  --wandb_mode disabled
  --device cuda
  --dtype float32
  --np_seed 42
  --torch_seed 42
  --max_steps "${MAX_STEPS}"
  --batch_size 512
  --micro_batch_size 4
  --lr 1e-4
  --scheduler cosine_warmup
  --warmup_proportion 0.02
  --gradient_clipping 1.0
  --prior_type hybrid_scm
  --informed_prior_ratio 0.75
  --mix_probs 1 0
  --informed_mix_probs 1 0
  --informed_task_family_probs 1.0 0.0
  --informed_normal_block_allocation 17 3 1 0 0 0 0 0 0
  --informed_normal_block_allocation_min_counts 17 3 1 0 0 0 0 0 0
  --informed_inhibitor_block_allocation 0.05 0.08 0.02 0.02 0 0.05 0.78 0 0
  --informed_feature_block_strength 0.4104161732
  --informed_interaction_strength 0.4079950508
  --informed_history_strength 0.0
  --informed_intervention_strength 0.0
  --informed_target_family pitting_potential
  --informed_physical_marginal_prob 0.25
  --informed_physical_marginal_profile pitting_potential_v1
  --pitting_material_dirichlet_prob 0.5
  --pitting_material_dirichlet_concentration 0.5
  --pitting_material_dirichlet_active_prob 0.9
  --pitting_process_role test_method_category
  --pitting_process_category_count 52
  --pitting_fixed_epit_schema True
  --cat_prob 0.0
  --permute_features False
  --epit_material_coef 0.5926227247
  --epit_environment_coef 0.4458409316
  --epit_interaction_coef 0.8968342005
  --prior_device cpu
  --prior_n_jobs "${PRIOR_N_JOBS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --batch_size_per_gp 4
  --min_features 21
  --max_features 21
  --max_classes 0
  --num_quantiles 999
  --max_seq_len 1024
  --min_train_size 0.1
  --max_train_size 0.9
  --embed_dim 128
  --col_num_blocks 3
  --col_nhead 4
  --col_num_inds 128
  --row_num_blocks 3
  --row_nhead 8
  --row_num_cls 4
  --row_rope_base 100000
  --icl_num_blocks 12
  --icl_nhead 4
  --ff_factor 2
  --norm_first True
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --save_temp_every "${SAVE_TEMP_EVERY}"
  --save_perm_every "${SAVE_PERM_EVERY}"
  --scheduler_total_steps "${SCHEDULER_TOTAL_STEPS}"
)

python scripts/write_train_params.py \
  --run-name "${RUN_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --script-path "$0" \
  --label fixed_pren_trial28 \
  -- "${TRAIN_CMD[@]}"

"${TRAIN_CMD[@]}"
