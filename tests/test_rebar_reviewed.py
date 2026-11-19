"""Review eligibility must never silently change the frozen final-query set."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.rebar_pipeline.prepare_reviewed import (
    DEFAULT_SPLIT, TARGET, parse_amount, prepare, process_composition, validate,
)


def test_balance_missingness_and_known_omissions():
    raw = dict(Fe='Bal.', C=.2, Mn=1.1, Si=None, Cr=0.)
    values, actions, _, total, reasons = process_composition(raw)
    assert not reasons and total == pytest.approx(100)
    assert values['Fe'] == pytest.approx(98.7)
    assert actions['Cr'] == 'reported_numeric'
    assert actions['Si'] == 'assumed_zero_after_completeness_review'
    values, actions, _, _, reasons = process_composition(raw, known_omissions=['Si'])
    assert reasons and np.isnan(values['Fe']) and np.isnan(values['Si'])
    values, _, _, _, reasons = process_composition({**raw, 'Mn': 'N/A'})
    assert reasons and np.isnan(values['Mn']) and np.isnan(values['Fe'])


def test_incomplete_or_invalid_composition_is_never_normalized():
    values, _, _, total, reasons = process_composition(dict(Fe=80., Cr=5., Ni=None))
    assert 'composition_total_below_90' in reasons
    assert values['Fe'] == 80. and total == 85. and np.isnan(values['Ni'])
    assert process_composition(dict(Fe=None, C=.2))[4]
    assert process_composition(dict(Fe=None, C=None))[4]
    assert process_composition(dict(Fe='Bal.', Cr=80., Ni=30.))[4]
    assert parse_amount(-1)[1] == 'invalid_amount'
    assert parse_amount('5.5-7.5')[1] == 'unresolved_notation'
    assert parse_amount('<0.003') == (.003, 'upper_limit_as_numeric')
    assert parse_amount(True)[1] == 'unresolved_notation'


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    output = tmp_path_factory.mktemp('review') / 'export'
    report = prepare(output)
    return output, report


def test_real_review_preserves_frozen_queries_and_targets(prepared):
    output, report = prepared
    parent = json.loads(DEFAULT_SPLIT.read_text())
    expected = [r['row_id'] for r in parent['rows'] if r['outer_split'] == 'final_test']
    query = pd.read_csv(output / 'final_test_queries.csv').set_index('row_id')
    assert query.index.tolist() == expected
    assert TARGET not in query
    pd.testing.assert_frame_equal(pd.read_csv(output / 'split_assignments.csv'), pd.DataFrame(parent['rows']))
    old_y = pd.read_csv(DEFAULT_SPLIT.parent / 'targets.csv').set_index('row_id')
    y = pd.read_csv(output / 'final_test_scoring_targets.csv').set_index('row_id')
    pd.testing.assert_frame_equal(y, old_y.loc[expected])
    development = pd.read_csv(output / 'development_context_features.csv')
    assert not set(expected) & set(development.row_id)
    assert report['eligible_candidates'] == 195
    assert report['development_context_rows'] == 152
    assert report['final_queries_needing_review'] == 13
    assert report['final_evaluation_ready'] is False
    assert len(query) == 56
    audit = pd.read_csv(output / 'composition_row_audit.csv').set_index('row_id')
    assert audit.loc[expected, 'eligible_candidate'].eq(False).sum() == 13


def test_real_corrections_units_and_predictor_boundary(prepared):
    output, _ = prepared
    frame = pd.read_csv(output / 'features_all_frozen_rows.csv').set_index('row_id')
    eligible = pd.read_csv(output / 'eligible_candidate_features.csv').set_index('row_id')
    audit = pd.read_csv(output / 'composition_row_audit.csv').set_index('row_id')
    registry = json.loads((output / 'feature_registry.json').read_text())
    assert frame.columns.tolist() == registry['feature_columns_in_order']
    assert 'source_id' not in frame and TARGET not in frame
    assert 'solution_(MO4)2-_M' not in frame
    assert frame.loc['Carbon_steel_2:731', 'Fe'] == pytest.approx(98.793)
    for row in [2303, 2306, 2309, 2312]:
        rid = f'Carbon_steel_2:{row}'
        assert frame.loc[rid, ['Ce', 'La', 'Fe']].isna().all()
        assert rid not in eligible.index
    assert not any(f'Carbon_steel_2:{row}' in eligible.index for row in range(2301, 2313))
    assert audit.loc['Stainless_steel_2:95', 'composition_eligible'] == False
    old = pd.read_csv(DEFAULT_SPLIT.parent / 'features.csv').set_index('row_id')
    np.testing.assert_allclose(frame.CP_Cl, old['solution_Cl-_M'], equal_nan=True)
    np.testing.assert_allclose(frame.CP_pH, old.pH, equal_nan=True)
    chemistry = eligible[registry['composition_features']]
    assert np.isfinite(chemistry).all().all()
    assert chemistry.sum(axis=1).between(99 - 1e-9, 101 + 1e-9).all()


def test_hash_lock_and_overwrite_protection(prepared, tmp_path):
    import shutil

    output, _ = prepared
    validate(output)
    with pytest.raises(FileExistsError):
        prepare(output)
    changed = tmp_path / 'changed'
    shutil.copytree(output, changed)
    with (changed / 'final_test_queries.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(RuntimeError, match='exports changed'):
        validate(changed)
