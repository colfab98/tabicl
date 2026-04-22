# Training efficiency report: tabicl_s1mini_generic_6536

- PID: `29790`
- Command: `/home/fcolanto/projects/tabicl/.venv/bin/python -u /home/fcolanto/projects/tabicl/src/tabicl/train/run.py --wandb_log False --wandb_project TabICL --wandb_name Stage1 --wandb_dir /home/fcolanto/wandb --wandb_mode disabled --device cuda --dtype float32 --np_seed 42 --torch_seed 42 --max_steps 10000 --batch_size 512 --micro_batch_size 4 --lr 1e-4 --scheduler cosine_warmup --warmup_proportion 0.02 --gradient_clipping 1.0 --prior_type mlp_scm --prior_device cpu --batch_size_per_gp 4 --min_features 2 --max_features 100 --max_classes 10 --max_seq_len 1024 --min_train_size 0.1 --max_train_size 0.9 --embed_dim 128 --col_num_blocks 3 --col_nhead 4 --col_num_inds 128 --row_num_blocks 3 --row_nhead 8 --row_num_cls 4 --row_rope_base 100000 --icl_num_blocks 12 --icl_nhead 4 --ff_factor 2 --norm_first True --checkpoint_dir /home/fcolanto/projects/tabicl/checkpoints/tabicl_s1mini_generic_v4 --save_temp_every 50 --save_perm_every 250`

## Summary

- GPU avg utilization: `4.86%`
- GPU p95 utilization: `86.45%`
- GPU avg memory used: `1853.52 MB`
- GPU peak memory used: `36467.00 MB`
- Process avg CPU: `35.56%`
- Process peak GPU memory: `36458.00 MB`
- Avg prior_time: `93.78 s`
- Avg train_time: `22.96 s`
- Prior share of step time: `80.33%`
- Train share of step time: `19.67%`

## Findings

- Average GPU utilization is low at 4.86%, which usually means the accelerator is underfed.
- Prior/data generation accounts for 80.33% of observed step time, which points to a CPU/input bottleneck.
