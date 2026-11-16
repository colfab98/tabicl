"""Train the exploratory Soccol selected-pair recipe using EPIT's launcher."""
from scripts.epit_pipeline import train_selected_pairs as shared
from scripts.epit_pipeline import train_final as base_final, run_optuna as base_search
from scripts.soccol_pipeline import train_final
from tabicl.prior.soccol_selected_pairs import SelectedPairSampler

DEFAULT_NAME = 'soccol_selected_pairs_allaccepted_generic10k_v1'
DEFAULT_RECIPE = shared.ROOT / 'corrosion_datasets/analysis/soccol_pipeline/selected_pairs_allaccepted_generic10k'


def build_plan(args):
    # The established Soccol adapter configures EPIT module globals. Restore
    # them even on failure, so planning Soccol cannot contaminate an EPIT run.
    modules = (base_final, base_search)
    snapshots = {module: vars(module).copy() for module in modules}
    try:
        return shared.build_plan(args, pipeline='soccol', sampler_class=SelectedPairSampler,
                                 template_parser=train_final.parse_args)
    finally:
        for module, snapshot in snapshots.items():
            for name in set(vars(module)) - set(snapshot):
                delattr(module, name)
            for name, value in snapshot.items():
                setattr(module, name, value)


def main(argv=None):
    shared.main(argv, pipeline='soccol', default_name=DEFAULT_NAME,
                default_recipe=DEFAULT_RECIPE, plan_builder=build_plan)


if __name__ == '__main__':
    main()
