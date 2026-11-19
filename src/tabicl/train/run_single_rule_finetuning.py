"""Opt-in S2 allocation; existing trainers, generators and checkpoint identities stay intact."""
import hashlib
from pathlib import Path
import json

from torch.multiprocessing import set_start_method

from tabicl.prior.selected_pairs_single_rule import install_single_rule_sampler
from tabicl.train.run import Trainer
from tabicl.train.run_combined_selected_pairs import CombinedSelectedPairTrainer
from tabicl.train.train_config import build_parser as base_parser


def build_parser():
    parser = base_parser()
    parser.add_argument('--single_rule_s2_prior', choices=['epit', 'soccol', 'combined', 'generic'], required=True)
    parser.add_argument('--combined_soccol_overrides', type=json.loads)
    return parser


class SingleRuleFinetuningTrainer(CombinedSelectedPairTrainer):
    def configure_prior(self):
        mode = self.config.single_rule_s2_prior
        if self.config.informed_prior_ratio != (0. if mode == 'generic' else 1.):
            raise ValueError('S2 requires 100% informed, or a 100% generic control.')
        if mode == 'combined':
            master = self.master_process
            self.master_process = False
            try:
                CombinedSelectedPairTrainer.configure_prior(self)
            finally:
                self.master_process = master
            dataset = self.dataloader.dataset
            groups = dataset.batch_size * dataset.world_size // dataset.group_size
            if groups % 2:
                raise ValueError('Equal EPIT/Soccol S2 requires an even number of global groups.')
            dataset.epit_ratio = dataset.soccol_ratio = .5
            dataset.selected_ratio = 1.
            dataset._config.update(source_probabilities={'generic': 0., 'epit': .5, 'soccol': .5},
                                   allocation='half EPIT / half Soccol; no generic groups')
        else:
            expected = (bool(self.config.epit_selected_pair_dir), bool(self.config.soccol_selected_pair_dir))
            if expected != {'epit': (True, False), 'soccol': (False, True), 'generic': (False, False)}[mode]:
                raise ValueError('S2 source and selected-pair recipe do not match.')
            if self.config.combined_soccol_overrides is not None:
                raise ValueError('Only combined S2 accepts Soccol overrides.')
            Trainer.configure_prior(self)
            dataset = self.dataloader.dataset
            if mode != 'generic':
                install_single_rule_sampler(dataset)
                dataset.selected_ratio = 1.
                dataset._config.update(selected_pair_effective_informed_ratio=1.,
                                       selected_pair_allocation='all groups informed; one rule per table')
        dataset._config['single_rule_s2'] = {
            'version': 'single_rule_finetuning_v1', 'prior': mode,
            'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if self.master_process:
            allocation = '50% EPIT / 50% Soccol / 0% generic' if mode == 'combined' else f'100% {mode}'
            print(f'S2 effective allocation: {allocation}.', flush=True)


def main():
    config = build_parser().parse_args()
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    SingleRuleFinetuningTrainer(config).train()


if __name__ == '__main__':
    main()
