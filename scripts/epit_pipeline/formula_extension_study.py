"""Development-only search over explicit EPIT equations (no symbolic search).

All feature eligibility, imputation, scaling, category levels, old-rule fits,
and coefficient fits belong to the supplied fitting rows. The final partition
is masked and never passed to the study. Existing training defaults are unused.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import rankdata, spearmanr

from scripts.epit_pipeline.calibrate_target_rules import _load_manifest, fit_coefficients
from scripts.epit_pipeline.composition_data import load_reviewed_compositions
from scripts.epit_pipeline.formula_data import load_formula_dataset
from scripts.epit_pipeline.target_rules import get_rule_families, PROMOTED_TARGET_RULE_FAMILIES
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS


PENALTIES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
SELECTION_TOLERANCE = 0.005
BASE_RULE = 'method_aware_pren_n'
MIN_OBSERVED = 20
MIN_CATEGORY = 10

SOURCES = [
    {'topic': 'Temperature, pH and chloride interactions', 'url': 'https://doi.org/10.1016/j.corsci.2019.108160'},
    {'topic': 'Scan-rate dependence of measured pitting potential', 'url': 'https://doi.org/10.1016/j.corsci.2013.04.028'},
    {'topic': 'Surface roughness and pitting potential', 'url': 'https://doi.org/10.1016/S0010-938X(96)00066-2'},
    {'topic': 'Exposed specimen area and measured pitting potential', 'url': 'https://doi.org/10.1016/S0010-938X(97)83146-0'},
    {'topic': 'Sulfate effects depend on endpoint and conditions', 'url': 'https://www.sciencedirect.com/science/article/pii/S0010938X06000436'},
    {'topic': 'Solutionizing duration and temperature affect pitting resistance', 'url': 'https://www.mdpi.com/1996-1944/15/24/8780'},
    {'topic': 'Heat-treatment effects depend on microstructure and alloy', 'url': 'https://doi.org/10.1016/j.corsci.2014.12.009'},
]


def rho(y, pred):
    value = float(spearmanr(y, pred).statistic)
    return value if np.isfinite(value) else 0.0


def rank_target(y):
    ranks = rankdata(np.asarray(y, dtype=float))
    scale = ranks.std()
    if not np.isfinite(y).all() or scale <= 0:
        raise ValueError('Fitting targets must be finite and nonconstant.')
    return (ranks - ranks.mean()) / scale


def feature_library(frame):
    """Fixed candidate hypotheses; sparse terms are filtered using fit rows."""
    raw, definitions = {}, {}
    def column(name):
        return pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float)
    def add(name, value, expression):
        raw[name] = np.asarray(value, dtype=float)
        definitions[name] = expression
    for element in ('Cr', 'Mo', 'W', 'N', 'Ni', 'C', 'Mn', 'S', 'Cu', 'Nb', 'Si'):
        add(element, column(element), element + ' (wt.%)')
    cr, mo, w, nitrogen = (column(c) for c in ('Cr', 'Mo', 'W', 'N'))
    pren = cr + 3.3 * (mo + 0.5*w) + 16*nitrogen
    temp, chloride, ph = (column(c) for c in ('CP_temp', 'CP_Cl', 'CP_pH'))
    logcl = np.log10(np.maximum(chloride, 1e-6))
    add('PREN', pren, 'Cr + 3.3*(Mo + 0.5*W) + 16*N')
    add('log_PREN', np.log1p(pren), 'log(1 + PREN)')
    add('log_Mo', np.log1p(mo), 'log(1 + Mo)')
    add('log_W', np.log1p(w), 'log(1 + W)')
    add('Ni_fraction', column('Ni') / np.maximum(column('Fe') + column('Ni'), 1e-8), 'Ni / max(Fe + Ni, 1e-8)')
    add('Cr_MoW_synergy', np.sqrt(cr*(mo+0.5*w)), 'sqrt(Cr*(Mo + 0.5*W))')
    add('MnS', np.sqrt(column('Mn')*column('S')), 'sqrt(Mn*S)')
    add('temperature', temp, 'CP_temp (deg C)')
    add('log_chloride', logcl, 'log10(max(CP_Cl [mol/L], 1e-6))')
    add('pH', ph, 'CP_pH')
    add('acidic_pH', np.maximum(7-ph, 0), 'max(7 - CP_pH, 0)')
    add('temperature_chloride', (temp-25)*logcl, '(CP_temp - 25)*log10(max(CP_Cl, 1e-6))')
    add('PREN_chloride', pren*logcl, 'PREN*log10(max(CP_Cl, 1e-6))')
    add('MoN_acidity', np.sqrt(mo*nitrogen)*np.maximum(7-ph, 0), 'sqrt(Mo*N)*max(7 - CP_pH, 0)')
    for name, col in [('log_scan', 'scan_rate'), ('log_area', 'Test_area_cm2'),
                      ('log_grit', 'Prep_grinding_grit'), ('log_exposure', 'CP_time'),
                      ('log_prep_time', 'Prep_time'), ('log_heat_hours', 'heat_treatment_duration_h')]:
        x = column(col)
        add(name, np.log10(np.where(x > 0, x, np.nan)), f'log10({col}); nonpositive/missing stays unknown')
    add('scan_chloride', raw['log_scan']*logcl, 'log_scan*log_chloride')
    heat = column('heat_treatment_temperature_C')
    hours = column('heat_treatment_duration_h')
    add('heat_temperature', heat, 'heat_treatment_temperature_C')
    dose = np.exp(-((heat-700)/200)**2)*np.log1p(hours)
    add('thermal_window', dose, 'exp(-((heat_temperature-700)/200)^2)*log(1+heat_hours); exploratory basis, not a calibrated kinetic law')
    add('carbon_thermal_window', column('C')*dose, 'C*thermal_window')
    add('phase_BCC', column('phase_BCC'), 'reported phase_BCC; unknown is not absence')
    for col in ('H2SO4_mol_L', 'HCl_mol_L', 'NaCl_mol_L', 'Na2SO4_mol_L', 'CP_Br', 'CP_SO4'):
        x = column(col)
        add('log_'+col, np.log10(np.where(x > 0, x, np.nan)), f'log10({col}); no inferred zeros or unit conversion')
    categories = {col: frame[col].astype('string').fillna('').str.strip().to_numpy(dtype=str)
                  for col in ('test_solution_type', 'CP_aeration', 'heat_treatment_type',
                              'heat_treatment_cooling', 'surface_preparation_method',
                              'preparation_solution_type')}
    return raw, categories, definitions


GROUPS = {
    'scan': (('log_scan',), ()),
    'area': (('log_area',), ()),
    'measurement': (('log_scan', 'log_area', 'log_exposure', 'scan_chloride'), ()),
    'surface': (('log_grit', 'log_prep_time'), ('surface_preparation_method', 'preparation_solution_type')),
    'heat': (('heat_temperature', 'log_heat_hours', 'thermal_window', 'carbon_thermal_window'), ('heat_treatment_type', 'heat_treatment_cooling')),
    'solution': ((), ('test_solution_type', 'CP_aeration')),
    'phase': (('phase_BCC',), ()),
    'material': (('Cr', 'log_Mo', 'log_W', 'N', 'Ni_fraction', 'C', 'Cu', 'Nb', 'Si', 'MnS'), ()),
    'solution_amounts': (('log_H2SO4_mol_L', 'log_HCl_mol_L', 'log_NaCl_mol_L', 'log_Na2SO4_mol_L', 'log_CP_Br', 'log_CP_SO4'), ()),
}
POSITIVE = {'PREN', 'log_PREN', 'Cr', 'log_Mo', 'log_W', 'N', 'Cr_MoW_synergy'}
NEGATIVE = {'temperature', 'log_chloride'}


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    existing_rule: str | None = None


def candidates(include_followup=True):
    specs = [Candidate('existing_'+name, 'existing', existing_rule=name) for name in PROMOTED_TARGET_RULE_FAMILIES]
    def combined(names):
        return tuple(dict.fromkeys(x for name in names for x in GROUPS[name][0])), tuple(dict.fromkeys(x for name in names for x in GROUPS[name][1]))
    for name, groups in [('scan', ('scan',)), ('area', ('area',)), ('measurement', ('measurement',)),
                         ('surface', ('surface',)), ('heat', ('heat',)), ('solution', ('solution',)),
                         ('phase', ('phase',)), ('material', ('material',)),
                         ('solution_amounts', ('solution_amounts',)),
                         ('measurement_surface', ('measurement', 'surface')),
                         ('process', ('measurement', 'surface', 'heat')),
                         ('all', ('measurement', 'surface', 'heat', 'solution', 'phase', 'material', 'solution_amounts'))]:
        numeric, categorical = combined(groups)
        specs.append(Candidate('extend_'+name, 'extension', numeric, categorical))
    core = ('PREN', 'temperature', 'log_chloride', 'pH')
    interaction = core + ('temperature_chloride', 'PREN_chloride', 'MoN_acidity')
    resolved = GROUPS['material'][0] + ('temperature', 'log_chloride', 'pH', 'temperature_chloride', 'MoN_acidity')
    specs.extend([Candidate('new_pren_loglinear', 'new', core),
                  Candidate('new_pren_interactions', 'new', interaction),
                  Candidate('new_resolved_material', 'new', resolved)])
    for name, groups in [('measurement', ('measurement',)), ('process', ('measurement', 'surface', 'heat')),
                         ('all', ('measurement', 'surface', 'heat', 'solution', 'phase', 'solution_amounts'))]:
        numeric, categorical = combined(groups)
        specs.append(Candidate('new_resolved_'+name, 'new', tuple(dict.fromkeys(resolved+numeric)), categorical))
    if include_followup:
        # Follow-up ablations, motivated by the first study's heat-treatment
        # gains and weaker source-publication transfer. Keep this history in
        # artifacts: these are exploratory development comparisons.
        specs.extend([
            Candidate('extend_heat_numeric', 'extension', ('heat_temperature', 'log_heat_hours')),
            Candidate('extend_heat_temperature', 'extension', ('heat_temperature',)),
            Candidate('extend_heat_duration', 'extension', ('log_heat_hours',)),
            Candidate('extend_heat_window', 'extension', ('thermal_window', 'carbon_thermal_window')),
            Candidate('extend_heat_categories', 'extension', (), ('heat_treatment_type', 'heat_treatment_cooling')),
            Candidate('extend_heat_type', 'extension', (), ('heat_treatment_type',)),
            Candidate('extend_heat_cooling', 'extension', (), ('heat_treatment_cooling',)),
            Candidate('extend_surface_numeric', 'extension', ('log_grit', 'log_prep_time')),
            Candidate('extend_surface_categories', 'extension', (), ('surface_preparation_method', 'preparation_solution_type')),
            Candidate('extend_heat_without_method', 'extension', *GROUPS['heat'], existing_rule='mns_inclusion_penalty'),
        ])
    return specs


def fit_feature_state(raw, categories, train, numeric, categorical):
    state, excluded = [], []
    minimum = max(MIN_OBSERVED, int(np.ceil(0.1*len(train))))
    for name in numeric:
        observed = raw[name][train]
        observed = observed[np.isfinite(observed)]
        if len(observed) < minimum or np.unique(observed).size < 2:
            excluded.append({'term': name, 'reason': 'too few observed values or constant observed values', 'observed': len(observed)})
            continue
        if np.unique(observed).size == 2 and min(np.unique(observed, return_counts=True)[1]) < 5:
            excluded.append({'term': name, 'reason': 'rare binary value', 'observed': len(observed)})
            continue
        fill = float(np.median(observed))
        values = np.where(np.isfinite(raw[name][train]), raw[name][train], fill)
        scale = float(values.std())
        if scale < 1e-12:
            continue
        state.append({'term': name, 'kind': 'numeric', 'fill': fill, 'center': float(values.mean()), 'scale': scale,
                      'observed': len(observed), 'observed_min': float(observed.min()), 'observed_max': float(observed.max())})
    for name in categorical:
        values, counts = np.unique(categories[name][train], return_counts=True)
        levels = [str(v) for v, count in zip(values, counts) if v and count >= MIN_CATEGORY]
        if len(levels) < 2:
            excluded.append({'term': name, 'reason': 'fewer than two supported observed categories'})
            continue
        known = np.isin(categories[name][train], levels)
        for level in levels:
            frequency = float(np.mean(categories[name][train][known] == level))
            values = np.where(known, (categories[name][train] == level).astype(float), frequency)
            state.append({'term': name+'='+level, 'source': name, 'kind': 'category', 'level': level,
                          'levels': levels, 'fill': frequency, 'center': float(values.mean()), 'scale': float(values.std())})
    return state, excluded


def transform_features(raw, categories, state, indices):
    columns = []
    for item in state:
        if item['kind'] == 'numeric':
            x = raw[item['term']][indices]
            x = np.where(np.isfinite(x), x, item['fill'])
        else:
            values = categories[item['source']][indices]
            x = np.where(np.isin(values, item['levels']), (values == item['level']).astype(float), item['fill'])
        columns.append((x-item['center'])/item['scale'])
    return np.column_stack(columns) if columns else np.empty((len(indices), 0))


def fit_linear(X, y, penalty, names, extension):
    target = rank_target(y)
    anchor = np.zeros(X.shape[1])
    if extension:
        anchor[0] = 1.
    lower, upper = np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf)
    for i, name in enumerate(names):
        if name == 'base_score' or name in POSITIVE:
            lower[i] = 0.
        elif name in NEGATIVE:
            upper[i] = 0.
    design = np.vstack([X/np.sqrt(len(y)), np.sqrt(penalty)*np.eye(X.shape[1])])
    rhs = np.concatenate([target/np.sqrt(len(y)), np.sqrt(penalty)*anchor])
    result = lsq_linear(design, rhs, bounds=(lower, upper), method='bvls', tol=1e-10, max_iter=1000)
    if not result.success:
        raise RuntimeError('Formula coefficient optimization failed: '+result.message)
    return result.x


class FormulaStudy:
    def __init__(self, dataset, frame, row_indices, fold_ids, composition_groups):
        self.dataset = dataset
        self.frame = frame.reset_index(drop=True).copy()
        self.rows = np.asarray(row_indices, dtype=int)
        self.y = np.asarray(dataset.target[self.rows], dtype=float)
        if not np.isfinite(self.y).all():
            raise ValueError('The study may contain development targets only.')
        self.folds = np.asarray(fold_ids, dtype=int)
        self.groups = np.asarray(composition_groups, dtype=str)
        self.raw, self.categories, self.definitions = feature_library(self.frame)
        self.specs = {spec.name: spec for spec in candidates()}
        self.old_cache, self.design_cache, self.model_cache = {}, {}, {}

    def old_model(self, train, family_name):
        key = (tuple(train), family_name)
        if key not in self.old_cache:
            family = get_rule_families([family_name])[0]
            prepared = family.fit_terms(self.dataset, self.rows[train])
            fit = fit_coefficients(prepared.values, self.y[train], family.coefficient_anchor,
                                   anchor_strength=.1, upper_bounds=family.coefficient_upper_bounds)
            all_terms = family.transform_terms(self.dataset, self.rows, prepared.state)
            self.old_cache[key] = (all_terms @ fit.coefficients, {'mode': 'existing', 'family': family_name,
                'feature_state': prepared.state, 'coefficients': fit.coefficients.tolist(), 'term_names': list(family.term_names)})
        return self.old_cache[key]

    def design(self, train, spec):
        key = (tuple(train), spec.name)
        if key not in self.design_cache:
            state, excluded = fit_feature_state(self.raw, self.categories, train, spec.numeric, spec.categorical)
            X = transform_features(self.raw, self.categories, state, np.arange(len(self.y)))
            names = [item['term'] for item in state]
            metadata = {'mode': spec.mode, 'feature_state': state, 'excluded': excluded}
            if spec.mode == 'extension':
                base, old = self.old_model(train, spec.existing_rule or BASE_RULE)
                center, scale = float(base[train].mean()), max(float(base[train].std()), 1e-12)
                X = np.column_stack([(base-center)/scale, X])
                names.insert(0, 'base_score')
                metadata.update(base=old, base_center=center, base_scale=scale)
            if X.shape[1] == 0:
                raise ValueError('No usable terms in '+spec.name)
            self.design_cache[key] = (X, names, metadata)
        return self.design_cache[key]

    def fit(self, train, spec, penalty=None):
        train = np.asarray(sorted(train), dtype=int)
        key = (tuple(train), spec.name, penalty)
        if key not in self.model_cache:
            if spec.mode == 'existing':
                prediction, artifact = self.old_model(train, spec.existing_rule)
                artifact = dict(artifact)
            else:
                X, names, metadata = self.design(train, spec)
                weights = fit_linear(X[train], self.y[train], penalty, names, spec.mode == 'extension')
                prediction = X @ weights
                artifact = dict(metadata, coefficients=weights.tolist(), term_names=names, penalty=float(penalty))
            score = prediction[train]
            centered = score-score.mean()
            slope = max(0., float(centered @ (self.y[train]-self.y[train].mean()) / max(centered @ centered, 1e-12)))
            artifact.update(name=spec.name, fitting_task_row_indices=self.rows[train].tolist(),
                mV_calibration={'intercept': float(self.y[train].mean()-slope*score.mean()), 'slope': slope},
                coefficient_objective='standardized Epit ranks; reported mV errors use a training-only affine calibration')
            self.model_cache[key] = prediction, artifact
        return self.model_cache[key]

    def predict_artifact(self, artifact):
        def old_prediction(model):
            family = get_rule_families([model['family']])[0]
            return family.transform_terms(self.dataset, self.rows, model['feature_state']) @ np.asarray(model['coefficients'])
        if artifact['mode'] == 'existing':
            return old_prediction(artifact)
        X = transform_features(self.raw, self.categories, artifact['feature_state'], np.arange(len(self.y)))
        if artifact['mode'] == 'extension':
            base = old_prediction(artifact['base'])
            X = np.column_stack([(base-artifact['base_center'])/artifact['base_scale'], X])
        return X @ np.asarray(artifact['coefficients'])


def load_study(split_manifest):
    dataset, identity = load_formula_dataset()
    frozen, rows = _load_manifest(split_manifest, dataset)
    masked = dataset.target.copy()
    masked[rows.final_test] = np.nan
    dataset = replace(dataset, target=masked)
    reviewed = load_reviewed_compositions()
    eligible = get_rule_families([BASE_RULE])[0].eligible_rows(dataset, rows.development)
    assert not np.intersect1d(eligible, rows.final_test).size
    assignments = reviewed.assignments.iloc[eligible]
    # Remove labels and identifying metadata from candidate feature access.
    frame = reviewed.rows.iloc[eligible].loc[:, list(EPIT_EXPANDED_FEATURE_COLUMNS)].copy()
    study = FormulaStudy(dataset, frame, eligible, assignments.optuna_validation_fold.to_numpy(dtype=int),
                         assignments.composition_group.to_numpy(dtype=str))
    study.source_groups = reviewed.rows.iloc[eligible]['Reference'].astype(str).to_numpy()
    return study, frozen, identity


def choose(records):
    """Prefer the simpler model within 0.005 mean inner-fold Spearman."""
    best = max(r['mean_spearman'] for r in records)
    accepted = [r for r in records if r['mean_spearman'] >= best-SELECTION_TOLERANCE]
    return min(accepted, key=lambda r: (r['complexity'], -(r.get('penalty') or 0.), r['name']))


def nested_evaluation(study, folds, *, progress=print):
    folds = np.asarray(folds)
    fold_labels = sorted(np.unique(folds))
    all_ids = np.arange(len(study.y))
    rows, predictions, selections = [], [], []
    fitted = {}
    for outer in fold_labels:
        fitting = all_ids[folds != outer]
        validation = all_ids[folds == outer]
        inner = [fold for fold in fold_labels if fold != outer]
        candidate_records = []
        for spec in study.specs.values():
            trials = []
            for penalty in ((None,) if spec.mode == 'existing' else PENALTIES):
                scores, sizes = [], []
                for fold in inner:
                    train, val = fitting[folds[fitting] != fold], fitting[folds[fitting] == fold]
                    pred, artifact = study.fit(train, spec, penalty)
                    scores.append(rho(study.y[val], pred[val]))
                    sizes.append(len(artifact['coefficients']))
                trials.append({'name': spec.name, 'penalty': penalty, 'mean_spearman': float(np.mean(scores)),
                               'complexity': float(np.mean(sizes)), 'inner_scores': scores})
            selected = choose(trials)
            candidate_records.append(selected)
            pred, artifact = study.fit(fitting, spec, selected['penalty'])
            mv = artifact['mV_calibration']
            row = {'name': spec.name, 'fold': int(outer), 'penalty': selected['penalty'],
                   'inner_mean_spearman': selected['mean_spearman'], 'spearman': rho(study.y[validation], pred[validation]),
                   'mae_mV': float(np.mean(np.abs(study.y[validation]-(mv['intercept']+mv['slope']*pred[validation])))),
                   'terms': len(artifact['coefficients']), 'fitting_rows': len(fitting), 'validation_rows': len(validation)}
            rows.append(row)
            fitted[(spec.name, int(outer))] = artifact
            for i in validation:
                predictions.append({'name': spec.name, 'fold': int(outer), 'task_row_index': int(study.rows[i]),
                                    'observed_epit_mV': float(study.y[i]), 'score': float(pred[i]),
                                    'prediction_mV': float(mv['intercept']+mv['slope']*pred[i])})
        for label, records in [('nested_new_and_existing', candidate_records),
                               ('nested_existing_only', [r for r in candidate_records if study.specs[r['name']].mode == 'existing'])]:
            selected = choose(records)
            selections.append({'selector': label, 'fold': int(outer), **selected})
            match = next(r for r in rows if r['name'] == selected['name'] and r['fold'] == outer)
            rows.append({**match, 'name': label, 'selected_formula': selected['name']})
            predictions.extend([{**r, 'name': label} for r in list(predictions) if r['name'] == selected['name'] and r['fold'] == outer])
        progress(f'fold {outer}: selected {selections[-2]["name"]}; held-out Spearman={rows[-2]["spearman"]:.4f}', flush=True)
    return rows, predictions, selections, fitted


def full_development_fits(study):
    """For future final training only; never evaluate these on development rows."""
    all_ids = np.arange(len(study.y))
    artifacts, selection = {}, []
    for spec in study.specs.values():
        trials = []
        for penalty in ((None,) if spec.mode == 'existing' else PENALTIES):
            scores, sizes = [], []
            for fold in sorted(np.unique(study.folds)):
                train, val = all_ids[study.folds != fold], all_ids[study.folds == fold]
                pred, artifact = study.fit(train, spec, penalty)
                scores.append(rho(study.y[val], pred[val])); sizes.append(len(artifact['coefficients']))
            trials.append({'name': spec.name, 'penalty': penalty, 'mean_spearman': float(np.mean(scores)),
                           'complexity': float(np.mean(sizes)), 'inner_scores': scores})
        selected = choose(trials)
        selection.append(selected)
        pred, artifact = study.fit(all_ids, spec, selected['penalty'])
        artifacts[spec.name] = artifact
        np.testing.assert_allclose(study.predict_artifact(artifact), pred, rtol=1e-12, atol=1e-12)
    return artifacts, selection, choose(selection)
