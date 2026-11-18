"""MPEA adapter for the same native evaluator and CV engine as EPIT/Soccol."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline.exploratory_cv import run_cv
from scripts.mpea_pipeline.evaluation_data import DEFAULT_SPLIT, TASK_ID, MEMBERSHIP_KEY, fold_membership, load_task


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mpea-split-manifest', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--exploratory-cv-folds', type=int, choices=[5])
    parser.add_argument('--epit-exploratory-cv-fold', '--mpea-cv-fold', dest='cv_fold', type=int, choices=range(1, 6))
    options, remaining = parser.parse_known_args(raw)
    if options.exploratory_cv_folds and options.cv_fold:
        parser.error('Use either the CV driver or a single fold.')
    args = ev.parse_args(remaining)
    ev.validate_args(args)
    if (args.task != [TASK_ID] or args.target_binning != 'continuous'
            or not args.native_tabicl_baselines or args.fixed_schema_model or args.missing_aware_model
            or args.epit_split_manifest or args.epit_expanded_features or args.soccol_reviewed_data
            or args.split_seeds or args.epit_final_context_folds or args.pitting_magpie_features
            or args.pitting_magpie_model or args.compare_catboost_magpie):
        parser.error('MPEA requires its own native-input continuous task and split.')
    # Validate every frozen input before starting model evaluation or spawning folds.
    task, membership, provenance = load_task(options.mpea_split_manifest,
        fold=options.cv_fold or 1, exploratory=bool(options.cv_fold or options.exploratory_cv_folds))
    if options.exploratory_cv_folds:
        # The shared driver uses this attribute to pass a manifest to membership_fn.
        args.epit_split_manifest = options.mpea_split_manifest
        run_cv(args, ev, membership_fn=fold_membership, membership_key=MEMBERSHIP_KEY,
            dataset_label='MPEA', split_strategy='mpea_pipeline_exploratory_cv',
            argv=raw, worker_script=__file__,
            interpretation='Exploratory checkpoint comparison on MPEA-only context/query folds. '
                           'Only fold 1 preserves the frozen final-query partition; '
                           'the five-fold mean is not the frozen final-test score.')
        return
    ev.main(remaining, task_factory=lambda _: [task], result_metadata={
        MEMBERSHIP_KEY if options.cv_fold else 'mpea_final_split': membership,
        'mpea_data_identity': provenance, 'missing_aware_models': [],
    })


if __name__ == '__main__':
    main()
