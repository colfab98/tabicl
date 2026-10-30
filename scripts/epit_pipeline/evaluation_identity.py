"""Frozen row membership for reviewed EPIT comparisons."""
from scripts.epit_pipeline.artifact_hashes import load_frozen_split


def frozen_membership(path, *, validation_fold=None, context_fold=None):
    rows = sorted(load_frozen_split(path).manifest["rows"], key=lambda row: row["task_row_index"])
    development = [row for row in rows if row["outer_split"] == "development"]
    if validation_fold is not None:
        context = [row for row in development if row["optuna_validation_fold"] != validation_fold]
        query = [row for row in development if row["optuna_validation_fold"] == validation_fold]
    else:
        context = [row for row in development
                   if context_fold is None or row["optuna_validation_fold"] == context_fold]
        query = [row for row in rows if row["outer_split"] == "final_test"]
    return {"context_task_row_indices": [row["task_row_index"] for row in context],
            "query_task_row_indices": [row["task_row_index"] for row in query]}
