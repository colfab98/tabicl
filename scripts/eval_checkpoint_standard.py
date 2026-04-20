#!/usr/bin/env python
"""Evaluate a local TabICL checkpoint on standard sklearn datasets.

This script intentionally uses the public scikit-learn API of TabICL so the
full built-in preprocessing and inference path is exercised. It is useful to
check whether a local checkpoint is broadly healthy before debugging a
task-specific evaluation wrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from tabicl import TabICLClassifier


DATASET_LOADERS: dict[str, Callable] = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ckpt-path", type=Path, required=True, help="Path to the local .ckpt file to evaluate.")
    parser.add_argument("--local-model-label", type=str, default="local_checkpoint")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_LOADERS),
        help="Dataset(s) to evaluate. Can be passed multiple times. Defaults to breast_cancer.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compare-pretrained", action="store_true")
    parser.add_argument(
        "--pretrained-checkpoint-version",
        type=str,
        default="tabicl-classifier-v2-20260212.ckpt",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_dataset_frame(dataset_name: str, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    loader = DATASET_LOADERS[dataset_name]
    dataset = loader(as_frame=True)

    if dataset.frame is not None and dataset.target.name in dataset.frame.columns:
        X = dataset.frame.drop(columns=[dataset.target.name]).copy()
        y = dataset.frame[dataset.target.name].copy()
    else:
        X = dataset.data.copy()
        y = dataset.target.copy()

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=getattr(dataset, "feature_names", None))
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")

    # Keep row order stable and explicit.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if dataset_name == "iris":
        y = y.astype(str)
    return X, y


def make_classifier(
    *,
    model_path: str | None,
    checkpoint_version: str,
    device: str,
    n_estimators: int,
    random_state: int,
    allow_auto_download: bool,
) -> TabICLClassifier:
    kwargs = dict(
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        allow_auto_download=allow_auto_download,
    )
    if model_path is None:
        kwargs["checkpoint_version"] = checkpoint_version
    else:
        kwargs["model_path"] = model_path
    return TabICLClassifier(**kwargs)


def evaluate_model(
    *,
    model_label: str,
    model_path: str | None,
    checkpoint_version: str,
    dataset_name: str,
    device: str,
    n_estimators: int,
    cv_folds: int,
    test_size: float,
    random_state: int,
    allow_auto_download: bool,
) -> dict:
    X, y = load_dataset_frame(dataset_name, random_state=random_state)
    n_classes = y.nunique()
    scoring = ["accuracy", "balanced_accuracy"]
    if n_classes == 2:
        scoring.append("roc_auc")
    else:
        scoring.append("roc_auc_ovr")

    estimator = make_classifier(
        model_path=model_path,
        checkpoint_version=checkpoint_version,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        allow_auto_download=allow_auto_download,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)

    result = {
        "model": model_label,
        "dataset": dataset_name,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "source": str(estimator.model_path_),
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_scores["test_accuracy"].std(ddof=1)),
        "cv_balanced_accuracy_mean": float(cv_scores["test_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(cv_scores["test_balanced_accuracy"].std(ddof=1)),
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }

    if n_classes == 2:
        positive_scores = y_proba[:, 1]
        result["cv_roc_auc_mean"] = float(cv_scores["test_roc_auc"].mean())
        result["cv_roc_auc_std"] = float(cv_scores["test_roc_auc"].std(ddof=1))
        result["holdout_roc_auc"] = float(roc_auc_score(y_test, positive_scores))
    else:
        result["cv_roc_auc_mean"] = float(cv_scores["test_roc_auc_ovr"].mean())
        result["cv_roc_auc_std"] = float(cv_scores["test_roc_auc_ovr"].std(ddof=1))
        result["holdout_roc_auc"] = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", labels=estimator.classes_)
        )

    return result


def main() -> None:
    args = parse_args()
    datasets = args.dataset or ["breast_cancer"]
    local_ckpt_path = args.local_ckpt_path.expanduser().resolve()
    if not local_ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {local_ckpt_path}")

    jobs = [
        dict(
            model_label=args.local_model_label,
            model_path=str(local_ckpt_path),
            checkpoint_version=args.pretrained_checkpoint_version,
            allow_auto_download=False,
        )
    ]
    if args.compare_pretrained:
        jobs.append(
            dict(
                model_label="pretrained_v2",
                model_path=None,
                checkpoint_version=args.pretrained_checkpoint_version,
                allow_auto_download=True,
            )
        )

    rows: list[dict] = []
    for dataset_name in datasets:
        for job in jobs:
            row = evaluate_model(
                dataset_name=dataset_name,
                device=args.device,
                n_estimators=args.n_estimators,
                cv_folds=args.cv_folds,
                test_size=args.test_size,
                random_state=args.random_state,
                **job,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.output_csv is not None:
        output_path = args.output_csv.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
