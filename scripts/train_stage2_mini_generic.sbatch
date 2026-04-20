#!/usr/bin/env bash
#SBATCH --job-name=tabicl_s2mini_generic
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/tabicl_s2mini_generic.%j.out
#SBATCH --error=/home/%u/tmp/tabicl_s2mini_generic.%j.err

set -euo pipefail

cd ~/projects/tabicl
source .venv/bin/activate

STAGE1_CKPT_DIR="${STAGE1_CKPT_DIR:-$HOME/projects/tabicl/checkpoints/tabicl_s1mini_generic}"
STAGE2_CKPT_DIR="${STAGE2_CKPT_DIR:-$HOME/projects/tabicl/checkpoints/tabicl_s2mini_generic}"

if [ -z "${STAGE1_CKPT:-}" ]; then
  STAGE1_CKPT="$(find "$STAGE1_CKPT_DIR" -maxdepth 1 -type f -name 'step-*.ckpt' | sort -V | tail -n 1)"
fi

if [ -z "${STAGE1_CKPT:-}" ] || [ ! -f "$STAGE1_CKPT" ]; then
  echo "Stage 1 checkpoint not found. Set STAGE1_CKPT or STAGE1_CKPT_DIR." >&2
  exit 1
fi

mkdir -p "$STAGE2_CKPT_DIR"

echo "Using stage 1 checkpoint: $STAGE1_CKPT"
echo "Writing stage 2 checkpoints to: $STAGE2_CKPT_DIR"

torchrun --standalone --nproc_per_node=2 src/tabicl/train/run.py \
  --wandb_log False \
  --wandb_mode disabled \
  --device cuda \
  --dtype float32 \
  --amp False \
  --np_seed 42 \
  --torch_seed 42 \
  --max_steps 300 \
  --batch_size 512 \
  --micro_batch_size 1 \
  --lr 2e-5 \
  --scheduler polynomial_decay_warmup \
  --warmup_proportion 0 \
  --poly_decay_lr_end 5e-6 \
  --poly_decay_power 2.0 \
  --gradient_clipping 1.0 \
  --prior_type mix_scm \
  --prior_device cpu \
  --batch_size_per_gp 2 \
  --min_features 2 \
  --max_features 100 \
  --max_classes 10 \
  --min_seq_len 1000 \
  --max_seq_len 40000 \
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
  --checkpoint_path "$STAGE1_CKPT" \
  --save_temp_every 10 \
  --save_perm_every 100 \
  --only_load_model True
