"""Separate trainer entry point; existing selected-pair runs stay unchanged."""
import hashlib
from pathlib import Path

from torch.multiprocessing import set_start_method

from tabicl.prior.selected_pairs_single_rule import install_single_rule_sampler
from tabicl.train.run import Trainer
from tabicl.train.train_config import build_parser


class SingleRuleTrainer(Trainer):
    def configure_prior(self):
        super().configure_prior()
        dataset = install_single_rule_sampler(self.dataloader.dataset)
        dataset._config["single_rule_trainer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if self.master_process:
            print("Selected pairs: one formula per table; multiple retained parents; original mixture ratio.", flush=True)


def main():
    config = build_parser().parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    SingleRuleTrainer(config).train()


if __name__ == "__main__":
    main()
