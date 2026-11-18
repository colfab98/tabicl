"""Opt-in trainer reusing the exact standalone EPIT/Soccol prior constructors."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from torch.multiprocessing import set_start_method

from tabicl.prior.combined_selected_pairs import CombinedSelectedPairPrior
from tabicl.prior.selected_pairs_single_rule import install_single_rule_sampler
from tabicl.train.run import Trainer
from tabicl.train.train_config import build_parser as original_parser

# Dataset-specific differences observed in the two validated standalone plans.
# Architecture, optimization, generic generation and stream seeds cannot differ.
SOCCOL_OVERRIDE_KEYS = frozenset({
    'epit_selected_pair_dir', 'soccol_selected_pair_dir', 'min_features', 'max_features',
    'informed_prior_ratio', 'informed_normal_block_allocation',
    'informed_normal_block_allocation_min_counts', 'pitting_feature_profile',
    'pitting_composition_family_probs', 'pitting_process_role', 'pitting_process_category_count',
    'pitting_target_rule_scores', 'pitting_target_rule_coefficients', 'pitting_coefficient_upper_bounds',
})


def build_parser():
    parser = original_parser()
    parser.add_argument('--combined_soccol_overrides', type=json.loads, required=True,
                        help='Dataset-only overrides generated from the standalone Soccol launch plan.')
    return parser


class CombinedSelectedPairTrainer(Trainer):
    def configure_prior(self):
        overrides = self.config.combined_soccol_overrides
        if not isinstance(overrides, dict) or set(overrides)-SOCCOL_OVERRIDE_KEYS:
            raise ValueError('Soccol overrides may change only dataset-specific prior settings.')
        if (not self.config.epit_selected_pair_dir or self.config.soccol_selected_pair_dir
                or overrides.get('epit_selected_pair_dir', 'missing') is not None
                or not overrides.get('soccol_selected_pair_dir')
                or self.config.epit_selected_pair_informed_only):
            raise ValueError('Expected separate EPIT and Soccol selected-pair recipes.')
        super().configure_prior()
        epit = install_single_rule_sampler(self.dataloader.dataset)
        # Configure only a second prior/loader; never initialize another model or optimizer.
        builder = Trainer.__new__(Trainer)
        builder.config = deepcopy(self.config)
        for key, value in overrides.items():
            setattr(builder.config, key, deepcopy(value))
        builder.ddp_rank, builder.ddp_world_size = self.ddp_rank, self.ddp_world_size
        builder.master_process = False
        builder.missingness_augmenter = None
        Trainer.configure_prior(builder)
        soccol = install_single_rule_sampler(builder.dataloader.dataset)
        combined = CombinedSelectedPairPrior(epit, soccol)
        combined._config['trainer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.dataloader.dataset = combined  # Workers have not started; the loader is still empty.
        if self.master_process:
            print(f'Combined single-rule tables: generic={1-combined.selected_ratio:.6%}, '
                  f'EPIT={combined.epit_ratio:.6%}, Soccol={combined.soccol_ratio:.6%}.', flush=True)


def main():
    config = build_parser().parse_args()
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    CombinedSelectedPairTrainer(config).train()


if __name__ == '__main__':
    main()
