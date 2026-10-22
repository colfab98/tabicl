from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from corrosion_datasets.datasets.zhang2023_duplex_pitting import prepare_evaluation as prep
from scripts.combined_corrosion import evaluate_zhang2023 as evaluation
from scripts import eval_corrosion_datasets as ev


def test_reported_compositions_and_targets_remain_aligned():
    queries, scoring, audit = prep.build_tables()
    assert len(queries) == len(scoring) == 56
    assert prep.TARGET_COLUMN not in queries
    assert queries.row_id.tolist() == scoring.row_id.tolist() == audit.row_id.tolist()
    assert queries.loc[:, prep.FEATURE_COLUMNS].drop_duplicates().shape[0] == 7
    assert audit.alloy_id.value_counts().eq(8).all()
    first = queries.iloc[0]
    assert first.Cr == 22.51 and first.Ni == 5.62 and first.Nb == 0
    assert first.Fe == pytest.approx(66.8644)
    assert scoring.iloc[0][prep.TARGET_COLUMN] == 250.2
    assert scoring[prep.TARGET_COLUMN].min() == -85.97
    assert scoring[prep.TARGET_COLUMN].max() == 734.8
    assert queries.Al.isna().all()
    assert queries.loc[audit.alloy_id.str.startswith("S32750"), "Cu"].isna().all()
    assert queries.loc[audit.alloy_id.str.startswith("S32101"), "P"].isna().all()
    assert not any("Industry" in str(x) for x in queries.row_id)


def test_seawater_conversion_uses_all_chloride_salts_and_no_false_carbonate():
    queries, _, _ = prep.build_tables()
    # Independent stoichiometric calculation from Xin and Li 2014 Table 1.
    chloride = 49.06 / 58.44277 + 2 * 10.4 / 95.211 + 2 * 2.32 / 110.984 + 1.390 / 74.5513 + 2 * .050 / 158.526
    assert queries.CP_Cl.iloc[0] == pytest.approx(chloride)
    assert queries.CP_SO4.iloc[0] == pytest.approx(8.18 / 142.04214)
    assert queries.CP_Br.iloc[0] == pytest.approx(.202 / 119.0023)
    assert queries[["CP_CO3", "CP_OH", "CP_time", "Prep_time", "Prep_redox"]].isna().all().all()


def test_outer_query_boundary_and_original_context_are_fixed(tmp_path):
    prep.prepare(tmp_path)
    task, ids, _ = evaluation.load_task(tmp_path)
    assert task.X.shape == (4843, 50)
    assert list(task.X.columns) == list(prep.FEATURE_COLUMNS)
    assert task.fixed_split.train_index.tolist() == list(range(4787))
    assert task.fixed_split.test_index.tolist() == list(range(4787, 4843))
    assert len(ids) == 56 and np.intersect1d(task.fixed_split.train_index, task.fixed_split.test_index).size == 0
    assert prep.file_sha256(tmp_path / "epit_soccol_context.csv") == prep.file_sha256(prep.CONTEXT_DIR / "epit_soccol_context.csv")
    scoring = tmp_path / prep.SCORING_NAME
    scoring.write_text(scoring.read_text().replace("250.2", "250.3"))
    with pytest.raises(ValueError, match="artifact changed"):
        evaluation.load_task(tmp_path)


def test_query_target_injection_is_rejected_even_with_updated_hash(tmp_path):
    prep.prepare(tmp_path)
    path = tmp_path / prep.QUERY_NAME
    q = pd.read_csv(path)
    q[prep.TARGET_COLUMN] = 0.
    q.to_csv(path, index=False)
    manifest_path = tmp_path / "preparation_manifest.json"
    m = json.loads(manifest_path.read_text())
    m["outputs"][prep.QUERY_NAME]["sha256"] = prep.file_sha256(path)
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="target-free"):
        evaluation.load_task(tmp_path)


def test_evaluation_settings_match_mpea_protocol():
    options = evaluation.parse_args(["--checkpoint", "5000"])
    args = evaluation.shared_options(options)
    assert args.n_estimators == 8 and args.random_state == 42
    assert args.catboost_iterations == 1000 and args.catboost_depth == 6
    for label in evaluation.MISSING_AWARE_MODELS:
        assert ev.tabicl_inference_settings(args, label) == ("none", ["none"])
        assert ev.expected_missing_value_mode(args, label) is True
    assert ev.tabicl_inference_settings(args, "generic_tabicl") == ("latin", None)
    assert ev.expected_missing_value_mode(args, "generic_tabicl") is False
    assert ev.tabicl_inference_settings(args, "pretrained_tabicl_v2", pretrained=True) == ("latin", None)
    assert args.regression_output == "median" and args.regression_uncertainty is False
