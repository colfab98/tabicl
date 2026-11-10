"""Exploratory Soccol CV using accepted development rows and frozen queries."""
import numpy as np

from scripts.epit_pipeline.exploratory_cv import CV_SEED, FOLDS, run_cv as run_shared_cv
from scripts.soccol_pipeline.data import reviewed_final_split_metadata


def fold_membership(manifest_path, fold):
    if fold not in FOLDS:
        raise ValueError("Exploratory CV fold must be between 1 and 5.")
    metadata = reviewed_final_split_metadata(manifest_path)
    development = metadata["soccol_development_selection"]["source_task_row_indices"]
    final = metadata["final_query_task_row_indices"]
    # Preserve the existing accepted-row policy and every original final query.
    queries = [final] + [sorted(chunk.tolist()) for chunk in np.array_split(
        np.random.default_rng(CV_SEED).permutation(development), 4
    )]
    if any(not query for query in queries):
        raise ValueError("Soccol exploratory CV requires five nonempty query folds.")
    query = queries[fold - 1]
    context = sorted((set(development) | set(final)) - set(query))
    return {"context_task_row_indices": context, "query_task_row_indices": query}


def apply_fold(tasks, args, ev):
    task, _, _, _ = ev._load_epit_pipeline_indices(tasks, manifest_path=args.epit_split_manifest)
    membership = fold_membership(args.epit_split_manifest, args.epit_exploratory_cv_fold)
    context = np.asarray(membership["context_task_row_indices"], dtype=int)
    task.fixed_split = ev.EvalSplit(
        train_index=context,
        test_index=np.asarray(membership["query_task_row_indices"], dtype=int),
        preprocessing_index=context.copy(),
        split_strategy=f"soccol_pipeline_exploratory_cv_fold_{args.epit_exploratory_cv_fold}",
    )
    task.split_strategy = task.fixed_split.split_strategy


def run_cv(args, ev):
    run_shared_cv(args, ev, membership_fn=fold_membership,
                  membership_key="soccol_exploratory_cv_split", dataset_label="Soccol",
                  split_strategy="soccol_pipeline_exploratory_cv")
