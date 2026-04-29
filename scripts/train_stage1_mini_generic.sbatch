#!/usr/bin/env bash
#SBATCH --job-name=tabicl_s1mini_generic_v19
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/tmp/%x.%j.out
#SBATCH --error=/home/%u/tmp/%x.%j.err

set -euo pipefail

RUN_NAME="${SLURM_JOB_NAME}"
VERSION="${SLURM_JOB_NAME##*_}"

PRIOR_N_JOBS=8
DATALOADER_NUM_WORKERS=4
DATALOADER_PREFETCH_FACTOR=4


cd ~/projects/tabicl
source .venv/bin/activate


torchrun --standalone --nproc_per_node=1 /home/fcolanto/projects/tabicl/src/tabicl/train/run.py \
            --wandb_log False \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir /home/fcolanto/wandb \
            --wandb_mode disabled \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 10000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_type hybrid_scm \
            --informed_prior_ratio 0.5 \
            --mix_probs 0.7 0.3 \
            --informed_mix_probs 0.7 0.3 \
            --informed_block_allocation 0.70 0.18 0.10 0.02 0.0 \
            --informed_feature_block_strength 0.2 \
            --informed_interaction_strength 0.25 \
            --informed_history_strength 0.20 \
            --informed_intervention_strength 0.05 \
            --informed_physical_marginal_prob 0.30 \
            --informed_physical_marginal_profile corrosion_broad \
            --prior_device cpu \
            --prior_n_jobs "${PRIOR_N_JOBS}" \
            --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
            --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
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
            --checkpoint_dir "/home/fcolanto/projects/tabicl/checkpoints/${RUN_NAME}" \
            --save_temp_every 50 \
            --save_perm_every 250
