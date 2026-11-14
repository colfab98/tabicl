"""Apply one saved EPIT v5 formula per real sample, in training proportions.

CPU only. This is the fixed fitted-formula mixture, not a replay of synthetic
training: no coefficient perturbations, SCM targets, or synthetic features.
Nine v4 equations use their saved development preprocessing; the heat equation
uses its production evaluator. Training's per-table formula probabilities are
used here to allocate real rows, as a deliberate sample-level diagnostic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.composition_data import load_reviewed_compositions
from scripts.epit_pipeline.formula_data import load_formula_dataset
from scripts.epit_pipeline.target_rules import get_rule_families
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS
from tabicl.prior.epit_target_v5 import heat_formula_drive, load_recipe

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = search.PIPELINE_ROOT / 'formula_mixture_real_features'


def assign_formulas(n, probabilities, rng):
    """Nearest integer quotas (largest remainder), randomly assigned to rows."""
    p = np.asarray(probabilities, dtype=float)
    if n < 1 or p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError('Need positive row count and valid normalized probabilities.')
    quotas = n * p
    counts = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - counts), kind='stable')
    counts[order[:n - counts.sum()]] += 1
    return rng.permutation(np.repeat(np.arange(len(p)), counts))


def calibrate_mv(scores, targets, fit_indices):
    """Positive affine mV conversion; access labels only at development indices."""
    x = np.asarray(scores)[fit_indices]
    y = np.asarray(targets)[fit_indices]
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid development calibration values.')
    centered = x - x.mean()
    slope = max(0., float(centered @ (y - y.mean()) / max(centered @ centered, 1e-12)))
    intercept = float(y.mean() - slope * x.mean())
    return intercept + slope * scores, {'intercept': intercept, 'slope': slope}


def metrics(y, predicted, score):
    def rho(a):
        return float(spearmanr(y, a).statistic) if len(y) > 1 and np.ptp(y) and np.ptp(a) else np.nan
    return {'n': len(y), 'mae_mV': float(np.abs(predicted - y).mean()),
            'spearman': rho(predicted), 'raw_score_spearman': rho(score)}


def load_inputs():
    rules = search.load_target_rule_config(summary_path=search.DEFAULT_TARGET_RULE_SUMMARY,
                                         split_manifest_path=search.DEFAULT_SPLIT_MANIFEST)
    recipe = load_recipe()
    if recipe['split_manifest_sha256'] != rules.split_manifest_sha256:
        raise ValueError('Heat recipe and v4 rules use different frozen splits.')
    reviewed = load_reviewed_compositions()
    dataset, identity = load_formula_dataset()
    fit = np.asarray(recipe['formula']['fitting_task_row_indices'], dtype=int)
    development = np.flatnonzero(reviewed.assignments.outer_split.eq('development'))
    if len(set(fit)) != len(fit) or not set(fit).issubset(development):
        raise ValueError('Formula fitting rows must be unique development rows.')
    if len(dataset.target) != len(reviewed.rows) or not np.isfinite(dataset.target).all():
        raise ValueError('Targets do not match reviewed rows.')
    scores = {**rules.scores, 'extend_heat_categories': float(recipe['mean_fold_spearman'])}
    probabilities = np.array(list(scores.values())) / sum(scores.values())
    return rules, recipe, reviewed, dataset, identity, fit, list(scores), probabilities


def formula_predictions(rules, recipe, reviewed, dataset, fit, names):
    indices = np.arange(len(dataset.target))
    raw, mv, calibrations = [], [], []
    for name in names:
        if name == 'extend_heat_categories':
            frame = reviewed.rows.loc[:, list(EPIT_EXPANDED_FEATURE_COLUMNS)].copy()
            for column, mapping in recipe['category_mappings'].items():
                frame[column] = frame[column].map(mapping)
            methods = {value: i for i, value in enumerate(recipe['test_method_categories'])}
            codes = []
            for row in dataset.rows:
                label = str(row.get('[Cl-] Test Method') or '<NA>').strip()
                # Unseen methods receive the saved missing-method offset.
                codes.append(methods.get(label, methods.get('<NA>', 0)))
            values = heat_formula_drive(torch.tensor(frame.to_numpy(dtype=float)),
                                        torch.tensor(codes), recipe).numpy()
        else:
            artifact = json.loads(Path(rules.artifacts[name]).read_text())
            fitted = artifact['final_development_calibration']
            family = get_rule_families([name])[0]
            if set(family.eligible_rows(dataset, np.flatnonzero(reviewed.assignments.outer_split.eq('development')))) != set(fit):
                raise ValueError(f'{name}: calibration row set differs from heat recipe.')
            terms = family.transform_terms(dataset, indices, fitted['feature_state'])
            coefficients = np.array([fitted['coefficients'][term] for term in family.term_names])
            values = terms @ coefficients
        if not np.isfinite(values).all():
            raise ValueError(f'{name} returned invalid scores.')
        predictions, calibration = calibrate_mv(values, dataset.target, fit)
        raw.append(values)
        mv.append(predictions)
        calibrations.append({'formula': name, **calibration})
    return np.column_stack(raw), np.column_stack(mv), calibrations


def run(args):
    torch.set_num_threads(1)
    rules, recipe, reviewed, dataset, identity, fit, names, probabilities = load_inputs()
    if args.dry_run:
        print(f'Validated {len(dataset.target)} real rows, {len(names)} formulas, {len(fit)} development calibration rows. CPU only; no evaluation run.')
        print(pd.DataFrame({'formula': names, 'probability': probabilities}).to_string(index=False))
        return
    raw, mv, calibrations = formula_predictions(rules, recipe, reviewed, dataset, fit, names)
    output = args.output_dir or OUTPUT / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    metadata = reviewed.rows[['row_id', 'task_row_index', 'No.', 'Material class', 'Reference']].copy()
    metadata['outer_split'] = reviewed.assignments.outer_split.to_numpy()
    metadata['observed_mV'] = dataset.target
    metadata['used_to_fit_formulas'] = np.isin(np.arange(len(metadata)), fit)
    scopes = {'all_rows_descriptive': np.arange(len(metadata)),
              'development_in_sample': np.flatnonzero(metadata.outer_split.eq('development')),
              'final_test': np.flatnonzero(metadata.outer_split.eq('final_test'))}
    if sum(len(scopes[s]) for s in ('development_in_sample', 'final_test')) != len(metadata):
        raise ValueError('Unknown split labels.')
    counts, results, exports = [], [], []
    for repeat in range(args.repeats):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, repeat]))
        assigned = np.full(len(metadata), -1, dtype=int)
        for scope in ('development_in_sample', 'final_test'):
            rows = scopes[scope]
            assigned[rows] = assign_formulas(len(rows), probabilities, rng)
            for j, name in enumerate(names):
                counts.append({'repeat': repeat, 'scope': scope, 'formula': name,
                               'probability': probabilities[j], 'n': int((assigned[rows] == j).sum())})
        predicted = mv[np.arange(len(metadata)), assigned]
        score = raw[np.arange(len(metadata)), assigned]
        for scope, rows in scopes.items():
            results.append({'repeat': repeat, 'scope': scope, 'group': 'all',
                            **metrics(dataset.target[rows], predicted[rows], score[rows])})
            for family in sorted(metadata['Material class'].dropna().unique()):
                selected = rows[metadata.iloc[rows]['Material class'].eq(family).to_numpy()]
                if len(selected):
                    results.append({'repeat': repeat, 'scope': scope, 'group': family,
                                    **metrics(dataset.target[selected], predicted[selected], score[selected])})
        frame = metadata.copy()
        frame.insert(0, 'repeat', repeat)
        frame['formula'] = np.asarray(names)[assigned]
        frame['raw_score'] = score
        frame['prediction_mV'] = predicted
        frame['absolute_error_mV'] = np.abs(predicted - dataset.target)
        exports.append(frame)
    measurements = pd.DataFrame(results)
    summary = measurements.groupby(['scope', 'group']).agg(
        n=('n', 'first'), repeats=('repeat', 'size'),
        mae_mV_mean=('mae_mV', 'mean'), mae_mV_p05=('mae_mV', lambda x: x.quantile(.05)),
        mae_mV_p95=('mae_mV', lambda x: x.quantile(.95)),
        spearman_mean=('spearman', 'mean'), spearman_p05=('spearman', lambda x: x.quantile(.05)),
        spearman_p95=('spearman', lambda x: x.quantile(.95)),
        raw_score_spearman_mean=('raw_score_spearman', 'mean')).reset_index()
    measurements.to_csv(output / 'repeat_metrics.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.concat(exports, ignore_index=True).to_csv(output / 'assigned_predictions.csv', index=False)
    pd.DataFrame(counts).to_csv(output / 'formula_counts.csv', index=False)
    pd.DataFrame(calibrations).to_csv(output / 'mV_calibrations.csv', index=False)
    config = {'seed': args.seed, 'repeats': args.repeats, 'formula_probabilities': dict(zip(names, probabilities.tolist())),
              'input_identity': identity, 'rule_artifact_sha256s': rules.artifact_sha256s,
              'heat_recipe_sha256': sha256_file(ROOT / 'src/tabicl/prior/assets/epit_target_v5.json'),
              'script_sha256': sha256_file(Path(__file__)), 'calibration_task_row_indices': fit.tolist(),
              'assignment': 'Fixed largest-remainder quotas within each split; shuffled per repeat.',
              'scope': __doc__, 'mV_conversion': 'Per-formula positive affine fit on eligible development rows only; coefficients and feature states stay fixed.'}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = ['# Combined formulas on real EPiT samples', '',
              'Each sample receives ONE formula, with quotas matching training probabilities to integer rounding. No per-sample averaging.',
              'MAE and main Spearman use development-calibrated mV predictions. raw_score_spearman uses the uncalibrated mixed formula scores.',
              'Final-test results are held out. Development and pooled results are descriptive and partly in-sample.',
              'The 5th–95th percentile range measures random assignment variation, not statistical confidence.',
              'Fixed fitted formulas only: saved development preprocessing, no coefficient variation or SCM mixing. This is not a full training-target replay.',
              'All alloy families are retained, including those outside the Fe/Ni formula fitting domain. Threshold targets retain the existing numeric-threshold convention.', '',
              '```', summary[summary.group.eq('all')].to_string(index=False), '```', '',
              'See assigned_predictions.csv for every sample, formula, prediction and error; summary.csv also separates alloy families.']
    (output / 'report.md').write_text('\n'.join(report) + '\n')
    print(summary[summary.group.eq('all')].to_string(index=False))
    print(f'Saved results to {output.resolve()}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.seed < 0:
        parser.error('Repeats must be positive and seed nonnegative.')
    return args


if __name__ == '__main__':
    run(parse_args())
