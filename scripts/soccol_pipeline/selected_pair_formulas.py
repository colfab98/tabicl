"""Soccol data/formula adapter for the exploratory all-accepted-row recipe.

Saved preprocessing and composition eligibility are unchanged. Only the fitting
scope changes: accepted former final rows join accepted development rows.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrosion_datasets.datasets.soccol_pitting_potential import compare_target_rules as old
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.evaluate_formula_mixture import calibrate_mv
from scripts.epit_pipeline.formula_extension_study import transform_features
from scripts.soccol_pipeline.data import REPO_ROOT, load_soccol_dataset
from scripts.soccol_pipeline.discover_formula_extensions import (
    SoccolFormulaStudy, candidates, feature_library,
)
from scripts.soccol_pipeline.target_rules import RULE_SELECTION, PROMOTED_RULES
from tabicl.prior.soccol_reviewed_schema import SOCCOL_REVIEWED_FEATURE_COLUMNS

EXTENSION_REPORT = REPO_ROOT / 'corrosion_datasets/analysis/soccol_pipeline/formula_extension_study_v1/results.json'


def load_inputs():
    dataset = load_soccol_dataset()
    indices = np.flatnonzero(dataset.rows.composition_status.eq('accepted'))
    frame = dataset.rows.iloc[indices].reset_index(drop=True).copy()
    frame['task_row_index'] = indices
    frame['composition_group'] = dataset.assignments.iloc[indices].composition_group.to_numpy()
    identity = {'reviewed_data': dataset.identity, 'policy': 'all_composition_accepted_rows',
                'fitting_task_row_indices': indices.tolist(),
                'excluded_task_row_indices': np.flatnonzero(~dataset.rows.composition_status.eq('accepted')).tolist()}
    # Existing formula functions use this historical attribute name to select
    # reviewed input handling (including unknown-ion imputation). Its policy
    # explicitly records all accepted rows; it is not a development selection.
    frame.attrs['soccol_development_selection'] = identity
    return dataset, frame, identity


def extension_spec():
    report = json.loads(EXTENSION_REPORT.read_text())
    if not report.get('completed') or not report.get('all_nine_current_baselines_reproduced'):
        raise ValueError('Soccol formula extension study is incomplete.')
    selected = report['full_development_selected']
    spec = next(s for s in candidates() if s.name == selected['name'])
    if spec.mode != 'extension':
        raise ValueError('Expected nine existing rules plus one previously selected extension.')
    return spec, float(selected['penalty'])


class AllAcceptedFormulaStudy(SoccolFormulaStudy):
    """Use the established fitting methods with explicit all-accepted scope.

    Do not weaken SoccolFormulaStudy's development-only constructor guard.
    """
    def __init__(self, frame):
        if not frame.composition_status.eq('accepted').all() or frame.task_row_index.duplicated().any():
            raise ValueError('Expected unique composition-accepted rows.')
        self.development = frame.copy().reset_index(drop=True)
        self.frame = self.development.loc[:, list(SOCCOL_REVIEWED_FEATURE_COLUMNS)].copy()
        self.rows = frame.task_row_index.to_numpy(dtype=int)
        self.y = frame.E_pit.to_numpy(dtype=float)
        if not np.isfinite(self.y).all():
            raise ValueError('Fitting targets must be finite.')
        self.raw, self.categories, self.definitions = feature_library(self.frame)
        self.specs = {s.name: s for s in candidates()}
        self.old_cache, self.design_cache, self.model_cache = {}, {}, {}
        specs = {s.name: s for s in old.RULES}
        self.old_specs = {family: specs[name] for family, name in RULE_SELECTION}


def fit_all(frame):
    study = AllAcceptedFormulaStudy(frame)
    indices = np.arange(len(frame))
    artifacts, predictions = {}, []
    for name in PROMOTED_RULES:
        print('Fitting all accepted rows:', name, flush=True)
        score, artifact = study.old_model(indices, name)
        prediction, calibration = calibrate_mv(score, study.y, indices)
        artifact = dict(artifact, name=name, fitting_task_row_indices=study.rows.tolist(),
                        mV_calibration=calibration)
        artifacts[name] = artifact
        predictions.append(pd.DataFrame({'task_row_index': study.rows, 'formula': name,
            'observed_mV': study.y, 'raw_score': score, 'prediction_mV': prediction}))
    spec, penalty = extension_spec()
    print('Fitting all accepted rows:', spec.name, flush=True)
    score, artifact = study.fit(indices, spec, penalty=penalty)
    calibration = artifact['mV_calibration']
    artifacts[spec.name] = artifact
    predictions.append(pd.DataFrame({'task_row_index': study.rows, 'formula': spec.name,
        'observed_mV': study.y, 'raw_score': score,
        'prediction_mV': calibration['intercept'] + calibration['slope'] * score}))
    return artifacts, pd.concat(predictions, ignore_index=True)


def predict_formulas(frame, formulas):
    """Replay frozen preprocessing, coefficients and mV calibration; never fit."""
    specs = {family: next(s for s in old.RULES if s.name == name) for family, name in RULE_SELECTION}
    raw, categories, _ = feature_library(frame)
    indices = np.arange(len(frame))

    def score(artifact):
        if artifact['mode'] == 'existing':
            return old._transform_terms(frame, specs[artifact['family']], artifact['feature_state']) @ np.asarray(artifact['coefficients'])
        terms = transform_features(raw, categories, artifact['feature_state'], indices)
        base = (score(artifact['base']) - artifact['base_center']) / artifact['base_scale']
        return np.column_stack([base, terms]) @ np.asarray(artifact['coefficients'])

    return {name: artifact['mV_calibration']['intercept'] + artifact['mV_calibration']['slope'] * score(artifact)
            for name, artifact in formulas.items()}


def source_identity():
    paths = [Path(__file__), Path(old.__file__),
             REPO_ROOT / 'scripts/soccol_pipeline/discover_formula_extensions.py',
             REPO_ROOT / 'scripts/soccol_pipeline/target_rules.py',
             REPO_ROOT / 'scripts/soccol_pipeline/data.py',
             REPO_ROOT / 'scripts/epit_pipeline/formula_extension_study.py',
             REPO_ROOT / 'scripts/epit_pipeline/evaluate_formula_mixture.py']
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}
