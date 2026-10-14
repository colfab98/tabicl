#!/usr/bin/env bash
#SBATCH --job-name=combined_corr_final
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

cd /home/fcolanto/projects/tabicl
source .venv/bin/activate

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
PYTHON_SEED="${PYTHON_SEED:-42}"
PRIOR_N_JOBS="${PRIOR_N_JOBS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-100}"
SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-500}"
WANDB_LOG="${WANDB_LOG:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-TabICL}"
WANDB_NAME="${WANDB_NAME:-combined_epit_soccol_mpea_final_seed42}"
WANDB_DIR="${WANDB_DIR:-/home/fcolanto/wandb}"
WANDB_MODE="${WANDB_MODE:-disabled}"

PRIOR_CONFIG="${PRIOR_CONFIG:-corrosion_datasets/datasets/combined_epit_soccol_mpea/processed/combined_prior_config.json}"
MISSINGNESS_PROFILE="${MISSINGNESS_PROFILE:-corrosion_datasets/datasets/combined_epit_soccol_mpea/processed/context_missingness_profile.json}"
QUERY_COLUMN_MASK_PROBABILITY="${QUERY_COLUMN_MASK_PROBABILITY:-0.0}"
PRESERVE_OBSERVED_COMPOSITION="${PRESERVE_OBSERVED_COMPOSITION:-False}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/combined_epit_soccol_mpea/final_seed42}"

[[ -f "$PRIOR_CONFIG" ]] || { echo "Missing prior config: $PRIOR_CONFIG" >&2; exit 1; }
[[ -f "$MISSINGNESS_PROFILE" ]] || { echo "Missing missingness profile: $MISSINGNESS_PROFILE" >&2; exit 1; }

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" src/tabicl/train/run.py \
  --device cuda \
  --dtype float32 \
  --amp True \
  --wandb_log "$WANDB_LOG" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_NAME" \
  --wandb_dir "$WANDB_DIR" \
  --wandb_mode "$WANDB_MODE" \
  --np_seed "$NP_SEED" \
  --torch_seed "$TORCH_SEED" \
  --python_seed "$PYTHON_SEED" \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --lr 1e-4 \
  --scheduler cosine_warmup \
  --scheduler_total_steps "$SCHEDULER_TOTAL_STEPS" \
  --warmup_proportion 0.02 \
  --gradient_clipping 1.0 \
  --combined_corrosion_prior_config "$PRIOR_CONFIG" \
  --prior_device cpu \
  --prior_n_jobs "$PRIOR_N_JOBS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR" \
  --batch_size_per_gp 4 \
  --min_features 50 \
  --max_features 50 \
  --max_classes 0 \
  --num_quantiles 999 \
  --max_seq_len 1024 \
  --min_train_size 0.1 \
  --max_train_size 0.9 \
  --col_missing_value_embedding True \
  --missingness_profile_path "$MISSINGNESS_PROFILE" \
  --missingness_augmentation_probability 0.8 \
  --missingness_min_observed_composition 2 \
  --missingness_query_column_probability "$QUERY_COLUMN_MASK_PROBABILITY" \
  --missingness_preserve_observed_composition "$PRESERVE_OBSERVED_COMPOSITION" \
  --embed_dim 128 \
  --col_num_blocks 3 \
  --col_nhead 4 \
  --col_num_inds 128 \
  --row_num_blocks 3 \
  --row_nhead 8 \
  --row_num_cls 4 \
  --row_rope_base 100000 \
  --icl_num_blocks 12 \
  --icl_nhead 4 \
  --ff_factor 2 \
  --norm_first True \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --save_temp_every "$SAVE_TEMP_EVERY" \
  --save_perm_every "$SAVE_PERM_EVERY"
