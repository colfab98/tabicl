"""Soccol adapter for the same single-rule table experiment as EPIT."""
from scripts.epit_pipeline.train_selected_pairs_single_rule import adapt_plan
from scripts.soccol_pipeline import train_selected_pairs as original
from tabicl.prior.selected_pairs_single_rule import SingleRuleSelectedPairSampler

DEFAULT_NAME = "soccol_selected_pairs_single_rule_allaccepted_generic10k_v1"


def build_plan(args):
    plan = original.build_plan(args)
    sampler = SingleRuleSelectedPairSampler(original.SelectedPairSampler(args.recipe_dir))
    return adapt_plan(plan, sampler)


def main(argv=None):
    original.shared.main(argv, pipeline="soccol", default_name=DEFAULT_NAME,
                         default_recipe=original.DEFAULT_RECIPE, plan_builder=build_plan)


if __name__ == "__main__":
    main()
