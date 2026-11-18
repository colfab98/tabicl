"""Train single-rule EPIT tables with the existing retained pairs and settings."""
import hashlib

from scripts.epit_pipeline import train_selected_pairs as original
from tabicl.prior.selected_pairs_single_rule import SingleRuleSelectedPairSampler

DEFAULT_NAME = "epit_selected_pairs_single_rule_all760_generic10k_v1"
TRAINER = original.ROOT / "src/tabicl/train/run_selected_pairs_single_rule.py"


def adapt_plan(plan, sampler):
    """Change only the table sampler/entry point; reuse the original planner."""
    command = list(plan["command"])
    command[command.index(str(original.ROOT / "src/tabicl/train/run.py"))] = str(TRAINER)
    return {**plan, "command": command, "recipe_identity": sampler.identity,
            "table_sampling_policy": sampler.policy,
            "single_rule_trainer_sha256": hashlib.sha256(TRAINER.read_bytes()).hexdigest()}


def build_plan(args):
    plan = original.build_plan(args)
    sampler = SingleRuleSelectedPairSampler(original.SelectedPairSampler(args.recipe_dir))
    return adapt_plan(plan, sampler)


def main(argv=None):
    original.main(argv, default_name=DEFAULT_NAME, plan_builder=build_plan)


if __name__ == "__main__":
    main()
