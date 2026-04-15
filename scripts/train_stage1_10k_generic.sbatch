#!/usr/bin/env bash
#SBATCH --job-name=tabicl_s1_10k_generic
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/tmp/tabicl_s1_10k_generic.%j.out
#SBATCH --error=/home/%u/tmp/tabicl_s1_10k_generic.%j.err

set -euo pipefail

cd ~/projects/tabicl
source .venv/bin/activate

python -m tabicl.train.run \
  --wandb_log False \
  --wandb_mode disabled \
  --device cuda \
  --dtype float32 \
  --amp False \
  --np_seed 42 \
  --torch_seed 42 \
  --max_steps 10000 \
  --batch_size 512 \
  --micro_batch_size 4 \
  --lr 1e-4 \
  --scheduler cosine_warmup \
  --warmup_proportion 0.02 \
  --gradient_clipping 1.0 \
  --prior_type mix_scm \
  --prior_device cpu \
  --batch_size_per_gp 4 \
  --min_features 2 \
  --max_features 100 \
  --max_classes 10 \
  --max_seq_len 1024 \
  --min_train_size 0.1 \
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
  --checkpoint_dir ~/projects/tabicl/checkpoints/stage1_10k_generic \
  --save_temp_every 50 \
  --save_perm_every 250
