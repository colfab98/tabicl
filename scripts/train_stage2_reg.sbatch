#!/usr/bin/env bash
#SBATCH --job-name=tabicl_s2_regression_baseline
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

RUN_NAME="${SLURM_JOB_NAME}"
VERSION="${SLURM_JOB_NAME##*_}"

PRIOR_N_JOBS=4
DATALOADER_NUM_WORKERS=1
DATALOADER_PREFETCH_FACTOR=1

STAGE1_CKPT_DIR="${STAGE1_CKPT_DIR:-/home/fcolanto/projects/tabicl/checkpoints/tabicl_s1_regression_baseline}"
STAGE2_CKPT_DIR="${STAGE2_CKPT_DIR:-/home/fcolanto/projects/tabicl/checkpoints/${RUN_NAME}}"
RESUME_STAGE2="${RESUME_STAGE2:-1}"

cd ~/projects/tabicl
source .venv/bin/activate

mkdir -p "$STAGE2_CKPT_DIR"

if [ -z "${STAGE2_CKPT:-}" ] && [ "$RESUME_STAGE2" != "0" ]; then
  STAGE2_CKPT="$(find "$STAGE2_CKPT_DIR" -maxdepth 1 -type f -name 'step-*.ckpt' | sort -V | tail -n 1)"
fi

if [ -n "${STAGE2_CKPT:-}" ]; then
  if [ ! -f "$STAGE2_CKPT" ]; then
    echo "Stage 2 checkpoint not found: $STAGE2_CKPT" >&2
    exit 1
  fi
  CHECKPOINT_PATH="$STAGE2_CKPT"
  ONLY_LOAD_MODEL=False
  echo "Resuming stage 2 checkpoint: $CHECKPOINT_PATH"
else
  if [ -z "${STAGE1_CKPT:-}" ]; then
    STAGE1_CKPT="$(find "$STAGE1_CKPT_DIR" -maxdepth 1 -type f -name 'step-*.ckpt' | sort -V | tail -n 1)"
  fi

  if [ -z "${STAGE1_CKPT:-}" ] || [ ! -f "$STAGE1_CKPT" ]; then
    echo "Stage 1 checkpoint not found. Set STAGE1_CKPT or STAGE1_CKPT_DIR." >&2
    exit 1
  fi
  CHECKPOINT_PATH="$STAGE1_CKPT"
  ONLY_LOAD_MODEL=True
  echo "Starting stage 2 from stage 1 checkpoint: $CHECKPOINT_PATH"
fi

echo "Writing stage 2 checkpoints to: $STAGE2_CKPT_DIR"

torchrun --standalone --nproc_per_node=2 /home/fcolanto/projects/tabicl/src/tabicl/train/run.py \
            --wandb_log False \
            --wandb_project TabICL \
            --wandb_name Stage2 \
            --wandb_dir /home/fcolanto/wandb \
            --wandb_mode disabled \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 2000 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --lr 2e-5 \
            --scheduler polynomial_decay_warmup \
            --warmup_proportion 0 \
            --poly_decay_lr_end 5e-6 \
            --poly_decay_power 2.0 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --mix_probs 0.7 0.3 \
            --prior_device cpu \
            --prior_n_jobs "${PRIOR_N_JOBS}" \
            --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
            --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
            --batch_size_per_gp 2 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 0 \
            --num_quantiles 999 \
            --min_seq_len 1000 \
            --max_seq_len 20000 \
            --log_seq_len True \
            --seq_len_per_gp True \
            --min_train_size 0.5 \
            --max_train_size 0.9 \
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
            --checkpoint_dir "$STAGE2_CKPT_DIR" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --save_temp_every 10 \
            --save_perm_every 100 \
            --only_load_model "$ONLY_LOAD_MODEL"
