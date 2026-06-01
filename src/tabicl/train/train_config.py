"""Define argument parser for TabICL training."""

import argparse


def str2bool(value):
    return value.lower() == "true"


def false_or_float(value):
    if isinstance(value, str) and value.lower() == "false":
        return 0.0
    return float(value)


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def build_parser():
    """Build an argument parser with all TabICL training arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all training, model, and
        checkpoint arguments.
    """
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument("--wandb_log", default=False, type=str2bool, help="Log results using wandb")
    parser.add_argument("--wandb_project", type=str, default="TabICL", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb logging directory")
    parser.add_argument(
        "--wandb_mode", default="offline", type=str, help="Wandb logging mode: online, offline, or disabled"
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument("--device", default="cuda", type=str, help="Device for training: cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="Data type (supported for float16, float32) used for training"
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--max_steps", type=int, default=60000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=8, help="Size of micro-batches for gradient accumulation"
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="cosine_warmup", help="Learning rate scheduler: see optim.py for options."
    )
    parser.add_argument(
        "--scheduler_total_steps",
        type=int,
        default=None,
        help=(
            "Optional LR scheduler horizon. Defaults to --max_steps. "
            "Use this for short proxy runs that should follow the early part of a longer training schedule."
        ),
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay / L2 regularization penalty")
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=1.0,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument("--cosine_lr_end", type=float, default=0, help="Final learning rate for cosine_with_restarts")
    parser.add_argument(
        "--poly_decay_lr_end", type=float, default=1e-7, help="Final learning rate for polynomial decay scheduler"
    )
    parser.add_argument(
        "--poly_decay_power", type=float, default=1.0, help="Power factor for polynomial decay scheduler"
    )

    # Prior Dataset Config
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=5, help="The minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="The maximum number of features")
    parser.add_argument(
        "--max_classes",
        type=int,
        default=10,
        help="The maximum number of classes. Use 0 for regression.",
    )
    parser.add_argument(
        "--num_quantiles",
        type=int,
        default=999,
        help="Number of quantiles predicted when --max_classes 0 enables regression.",
    )
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum samples per dataset")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum samples per dataset")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Starting position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Ending position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type",
        default="mix_scm",
        type=str,
        help="Prior type: dummy, mlp_scm, tree_scm, mix_scm, informed_scm, hybrid_scm",
    )
    parser.add_argument(
        "--informed_prior_ratio",
        type=false_or_float,
        default=0.5,
        help="For prior_type=hybrid_scm, probability of sampling informed subgroups (0 to 1).",
    )
    parser.add_argument(
        "--mix_probs",
        type=float,
        nargs=2,
        default=None,
        metavar=("MLP_PROB", "TREE_PROB"),
        help="Optional override for mix_scm probabilities (mlp_scm, tree_scm).",
    )
    parser.add_argument(
        "--informed_mix_probs",
        type=float,
        nargs=2,
        default=None,
        metavar=("MLP_PROB", "TREE_PROB"),
        help="Optional override for informed_scm/hybrid_scm probabilities (mlp_scm, tree_scm).",
    )
    parser.add_argument(
        "--informed_block_allocation",
        type=float,
        nargs=5,
        default=None,
        metavar=("MATERIAL", "ENVIRONMENT", "ELECTROCHEM", "HISTORY", "INTERVENTION"),
        help=(
            "Deprecated five-weight allocation from the old coarse schema. Prefer "
            "--informed_normal_block_allocation and --informed_inhibitor_block_allocation."
        ),
    )
    parser.add_argument(
        "--informed_task_family_probs",
        type=float,
        nargs=2,
        default=None,
        metavar=("NORMAL_CORROSION", "INHIBITOR_AGENT"),
        help="Optional informed task-family mixture weights for audit-v2 grouping.",
    )
    parser.add_argument(
        "--informed_normal_block_allocation",
        type=float,
        nargs=9,
        default=None,
        metavar=(
            "MATERIAL",
            "ENVIRONMENT",
            "PROCESS_HISTORY",
            "EXPOSURE_DURATION",
            "TEMPORAL_HISTORY",
            "DIRECT_INTERVENTION",
            "MOLECULAR_DESCRIPTOR",
            "ELECTROCHEM_CONTROL",
            "ELECTROCHEM_DOWNSTREAM",
        ),
        help="Optional audit-v2 block allocation for normal corrosion synthetic tasks.",
    )
    parser.add_argument(
        "--informed_inhibitor_block_allocation",
        type=float,
        nargs=9,
        default=None,
        metavar=(
            "MATERIAL",
            "ENVIRONMENT",
            "PROCESS_HISTORY",
            "EXPOSURE_DURATION",
            "TEMPORAL_HISTORY",
            "DIRECT_INTERVENTION",
            "MOLECULAR_DESCRIPTOR",
            "ELECTROCHEM_CONTROL",
            "ELECTROCHEM_DOWNSTREAM",
        ),
        help="Optional audit-v2 block allocation for inhibitor-agent synthetic tasks.",
    )
    parser.add_argument(
        "--informed_normal_block_allocation_ranges",
        type=float,
        nargs=18,
        default=None,
        help="Optional low/high pairs for per-dataset normal-corrosion block allocation sampling.",
    )
    parser.add_argument(
        "--informed_inhibitor_block_allocation_ranges",
        type=float,
        nargs=18,
        default=None,
        help="Optional low/high pairs for per-dataset inhibitor-agent block allocation sampling.",
    )
    parser.add_argument(
        "--informed_normal_block_allocation_min_counts",
        type=int,
        nargs=9,
        default=None,
        help="Optional minimum column counts for normal-corrosion allocation blocks.",
    )
    parser.add_argument(
        "--informed_inhibitor_block_allocation_min_counts",
        type=int,
        nargs=9,
        default=None,
        help="Optional minimum column counts for inhibitor-agent allocation blocks.",
    )
    parser.add_argument(
        "--informed_feature_block_strength",
        type=false_or_float,
        default=None,
        help="Optional override for informed block-level feature coupling strength.",
    )
    parser.add_argument(
        "--informed_interaction_strength",
        type=false_or_float,
        default=None,
        help="Optional override for informed material-environment interaction strength.",
    )
    parser.add_argument(
        "--informed_history_strength",
        type=false_or_float,
        default=None,
        help="Optional override for informed autoregressive history coupling strength.",
    )
    parser.add_argument(
        "--informed_intervention_strength",
        type=false_or_float,
        default=None,
        help="Optional override for informed intervention damping strength.",
    )
    parser.add_argument(
        "--informed_target_family",
        type=str,
        default=None,
        choices=("generic_corrosion", "pitting_potential", "inhibitor_efficiency"),
        help=(
            "Optional target semantics for informed SCM target generation. "
            "Use pitting_potential for Epit/passivity-breakdown threshold targets; "
            "use inhibitor_efficiency when higher targets mean stronger inhibitor protection."
        ),
    )
    parser.add_argument(
        "--informed_physical_marginal_prob",
        type=false_or_float,
        default=None,
        help=(
            "Optional probability that an informed synthetic dataset receives corrosion-like "
            "physical feature marginal transforms."
        ),
    )
    parser.add_argument(
        "--informed_physical_marginal_profile",
        type=str,
        default=None,
        help="Optional physical marginal profile for informed SCM features, e.g. corrosion_broad, pitting_potential_v1, or inhibitor_efficiency_v1.",
    )
    parser.add_argument(
        "--epit_material_coef_scale",
        type=float,
        default=None,
        help="Optional multiplier for the material/passivity term in the Epit-like target drive.",
    )
    parser.add_argument(
        "--epit_environment_coef_scale",
        type=float,
        default=None,
        help="Optional multiplier for the environment-aggressiveness term in the Epit-like target drive.",
    )
    parser.add_argument(
        "--epit_interaction_coef_scale",
        type=float,
        default=None,
        help="Optional multiplier for the material-environment term in the Epit-like target drive.",
    )
    parser.add_argument("--prior_device", default="cpu", type=str, help="Device for prior data generation")
    parser.add_argument(
        "--prior_n_jobs",
        type=int,
        default=1,
        help="Number of CPU jobs used inside the prior generator when creating priors on the fly.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of DataLoader workers used to fetch/generated training batches.",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=4,
        help="Number of batches prefetched by each DataLoader worker.",
    )

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )

    # Column Embedding Config
    parser.add_argument("--embed_dim", type=int, default=128, help="Base embedding dimension")
    parser.add_argument("--col_num_blocks", type=int, default=3, help="Number of blocks in column embedder")
    parser.add_argument("--col_nhead", type=int, default=4, help="Number of attention heads in column embedder")
    parser.add_argument("--col_num_inds", type=int, default=128, help="Number of inducing points in column embedder")
    parser.add_argument("--freeze_col", default=False, type=str2bool, help="Whether to freeze the column embedder")

    # Row Interaction Config
    parser.add_argument("--row_num_blocks", type=int, default=3, help="Number of blocks in row interactor")
    parser.add_argument("--row_nhead", type=int, default=8, help="Number of attention heads in row interactor")
    parser.add_argument("--row_num_cls", type=int, default=4, help="Number of CLS tokens in row interactor")
    parser.add_argument("--row_rope_base", type=float, default=100000, help="RoPE base value for row interactor")
    parser.add_argument("--freeze_row", default=False, type=str2bool, help="Whether to freeze the row interactor")

    # ICL Config
    parser.add_argument("--icl_num_blocks", type=int, default=12, help="Number of transformer blocks in ICL predictor")
    parser.add_argument("--icl_nhead", type=int, default=4, help="Number of attention heads in ICL predictor")
    parser.add_argument("--freeze_icl", default=False, type=str2bool, help="Whether to freeze the ICL predictor")

    # Shared Architecture Config
    parser.add_argument("--ff_factor", type=int, default=2, help="Expansion factor for feedforward dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function type")
    parser.add_argument(
        "--norm_first", default=True, type=str2bool, help="If True, use pre-norm transformer architecture"
    )

    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory for checkpoint saving and loading")
    parser.add_argument("--save_temp_every", default=50, type=int, help="Steps between temporary checkpoints")
    parser.add_argument("--save_perm_every", default=5000, type=int, help="Steps between permanent checkpoints")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of temporary checkpoints to keep. Permanent checkpoints are not counted.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to specific checkpoint file to load")
    parser.add_argument("--only_load_model", default=False, type=str2bool, help="Whether to only load model weights")

    return parser
