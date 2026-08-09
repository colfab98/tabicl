import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as corrosion_eval
from scripts import eval_pitting_repeated_splits as repeated_eval


class RecordingCatBoostRegressor:
    instances: list["RecordingCatBoostRegressor"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_frame = None
        self.predict_frame = None
        self.cat_features = None
        self.__class__.instances.append(self)

    def fit(self, X, y, *, cat_features, verbose):
        self.fit_frame = X.copy()
        self.fit_target = np.asarray(y, dtype=float)
        self.cat_features = list(cat_features)
        self.fit_verbose = verbose
        return self

    def predict(self, X):
        self.predict_frame = X.copy()
        return np.arange(len(X), dtype=float)


def test_catboost_adapter_preserves_numeric_missing_values_and_prepares_categories(monkeypatch):
    RecordingCatBoostRegressor.instances.clear()
    fake_catboost = SimpleNamespace(
        CatBoostRegressor=RecordingCatBoostRegressor,
        __version__="test-version",
    )
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)

    estimator = corrosion_eval.make_catboost_regressor(
        iterations=25,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=2.0,
        random_state=1001,
        thread_count=3,
    )
    train = pd.DataFrame(
        {
            "composition": [1.0, np.nan, 3.0],
            "test_method": pd.Series(["potentiodynamic", pd.NA, "scratch"], dtype="string"),
        }
    )
    estimator.fit(train, pd.Series([10.0, 20.0, 30.0]))

    model = RecordingCatBoostRegressor.instances[-1]
    assert model.cat_features == ["test_method"]
    assert np.isnan(model.fit_frame.loc[1, "composition"])
    assert model.fit_frame.loc[1, "test_method"] == corrosion_eval.CATBOOST_MISSING_CATEGORY
    assert model.kwargs == {
        "allow_writing_files": False,
        "depth": 4,
        "eval_metric": "RMSE",
        "iterations": 25,
        "l2_leaf_reg": 2.0,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": 1001,
        "task_type": "CPU",
        "thread_count": 3,
        "verbose": False,
    }
    assert "catboost==test-version" in estimator.model_source_
    assert "thread_count=3" in estimator.model_source_

    test = pd.DataFrame(
        {
            "composition": [4.0, 5.0],
            "test_method": pd.Series([pd.NA, "potentiostatic"], dtype="string"),
        }
    )
    assert estimator.predict(test).tolist() == [0.0, 1.0]
    assert model.predict_frame.loc[0, "test_method"] == corrosion_eval.CATBOOST_MISSING_CATEGORY


def test_repeated_split_wrapper_forwards_catboost_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_pitting_repeated_splits.py",
            "--pretrained-tabicl-only",
            "--compare-catboost",
            "--catboost-iterations",
            "75",
            "--catboost-depth",
            "5",
            "--catboost-learning-rate",
            "0.04",
            "--catboost-l2-leaf-reg",
            "4.5",
            "--catboost-thread-count",
            "2",
        ],
    )
    args = repeated_eval.parse_args()
    repeated_eval.validate_args(args)
    captured = []
    def fake_run(command, **kwargs):
        captured.extend(command)
        output_csv = command[command.index("--output-csv") + 1]
        pd.DataFrame([{"model": "catboost", "task_id": repeated_eval.PITTING_TASK_ID}]).to_csv(
            output_csv,
            index=False,
        )

    monkeypatch.setattr(repeated_eval.subprocess, "run", fake_run)
    rows = repeated_eval.run_seed_eval(args, 1001, tmp_path / "seed_1001")

    assert rows["model"].tolist() == ["catboost"]
    assert captured[captured.index("--catboost-iterations") + 1] == "75"
    assert captured[captured.index("--catboost-depth") + 1] == "5"
    assert captured[captured.index("--catboost-learning-rate") + 1] == "0.04"
    assert captured[captured.index("--catboost-l2-leaf-reg") + 1] == "4.5"
    assert captured[captured.index("--catboost-thread-count") + 1] == "2"
    assert "--compare-catboost" in captured


def test_model_specific_magpie_routing_and_label_validation():
    args = SimpleNamespace(
        pitting_magpie_features=False,
        pitting_magpie_model=["trial15_magpie"],
    )

    assert corrosion_eval.model_uses_pitting_magpie(args, "trial15_magpie")
    assert not corrosion_eval.model_uses_pitting_magpie(args, "trial15_no_magpie")
    assert corrosion_eval.model_uses_pitting_magpie(args, "catboost_magpie", force=True)

    corrosion_eval.validate_pitting_magpie_model_labels(
        args,
        {"trial15_magpie", "trial15_no_magpie", "catboost", "catboost_magpie"},
    )
    with pytest.raises(ValueError, match="missing_model"):
        corrosion_eval.validate_pitting_magpie_model_labels(
            SimpleNamespace(
                pitting_magpie_features=False,
                pitting_magpie_model=["missing_model"],
            ),
            {"trial15_magpie", "trial15_no_magpie"},
        )


def test_parse_args_accepts_magpie_catboost_and_model_label(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_corrosion_datasets.py",
            "--run",
            "example",
            "--compare-catboost-magpie",
            "--pitting-magpie-model",
            "trial15_magpie",
        ],
    )
    args = corrosion_eval.parse_args()
    assert args.compare_catboost_magpie
    assert args.pitting_magpie_model == ["trial15_magpie"]
