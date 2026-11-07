#!/usr/bin/env python
"""Fit and assess explicit EPIT formula candidates without PySR."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.calibrate_target_rules import DEFAULT_SPLIT_DIR
from scripts.epit_pipeline.formula_extension_study import (
    BASE_RULE, PENALTIES, SELECTION_TOLERANCE, SOURCES, candidates,
    full_development_fits, load_study, nested_evaluation, rho,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'corrosion_datasets/analysis/epit_pipeline/formula_extensions_v2'


def summarize(rows):
    frame = pd.DataFrame(rows)
    result = frame.groupby('name', sort=False).agg(
        mean_spearman=('spearman', 'mean'), sd_fold_spearman=('spearman', lambda x: float(np.std(x))),
        minimum_fold_spearman=('spearman', 'min'), maximum_fold_spearman=('spearman', 'max'),
        mean_mae_mV=('mae_mV', 'mean'), mean_terms=('terms', 'mean')).reset_index()
    baseline = frame[frame.name.eq('existing_'+BASE_RULE)].set_index('fold').spearman
    result['gain_vs_current_best'] = result.mean_spearman - baseline.mean()
    result['folds_better_than_current'] = [int((group.set_index('fold').spearman > baseline).sum())
                                         for _, group in frame.groupby('name', sort=False)]
    return result.sort_values('mean_spearman', ascending=False).reset_index(drop=True)


def coefficient_stability(fitted):
    rows = []
    names = sorted({name for name, _ in fitted})
    for name in names:
        folds = [a for (n, _), a in fitted.items() if n == name]
        terms = sorted({term for a in folds for term in a['term_names']})
        for term in terms:
            values = [dict(zip(a['term_names'], a['coefficients'])).get(term, 0.) for a in folds]
            present = sum(term in a['term_names'] for a in folds)
            rows.append({'name': name, 'term': term, 'present_in_folds': present,
                         'mean_standardized_coefficient': float(np.mean(values)),
                         'min_coefficient': min(values), 'max_coefficient': max(values),
                         'positive_folds': sum(v > 1e-8 for v in values), 'negative_folds': sum(v < -1e-8 for v in values)})
    return pd.DataFrame(rows)


def bootstrap_difference(predictions, study, groups, *, samples=1000):
    """Paired group resampling within held-out folds; not a training-seed CI."""
    frame = pd.DataFrame(predictions)
    left = frame[frame.name.eq('nested_new_and_existing')].set_index('task_row_index')
    right = frame[frame.name.eq('nested_existing_only')].set_index('task_row_index')
    source = dict(zip(study.rows, groups))
    folds = []
    for fold in sorted(left.fold.unique()):
        ids = left[left.fold.eq(fold)].index.to_numpy()
        by_group = {}
        for index in ids:
            by_group.setdefault(str(source[index]), []).append(index)
        folds.append(list(by_group.values()))
    rng = np.random.default_rng(42)
    values = []
    for _ in range(samples):
        differences = []
        for units in folds:
            ids = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
            a, b = left.loc[ids], right.loc[ids]
            differences.append(rho(a.observed_epit_mV, a.score)-rho(b.observed_epit_mV, b.score))
        values.append(np.mean(differences))
    return {'paired_mean_spearman_gain_ci95': np.quantile(values, [.025, .975]).tolist(),
            'bootstrap_samples': samples,
            'interpretation': 'Uncertainty from evaluation groups conditional on fitted fold models; does not include training/search instability.'}


def coverage_table(study):
    rows = []
    for column in study.frame:
        series = study.frame[column]
        values = series.dropna()
        rows.append({'column': column, 'observed_rows': len(values), 'missing_fraction': float(series.isna().mean()),
                     'distinct_observed_values': int(values.nunique()),
                     'note': 'constant among observed values' if values.nunique() <= 1 else ''})
    return pd.DataFrame(rows)


def write_report(output, report, summaries, artifacts):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    main = summaries['frozen_development_folds']
    selected = report['full_development_selected']['name']
    baseline = 'existing_'+BASE_RULE
    shown = list(dict.fromkeys([baseline, 'nested_existing_only', 'nested_new_and_existing', *main.head(8).name]))
    figure, ax = plt.subplots(figsize=(10, max(4, .38*len(shown))))
    selected_rows = main.set_index('name').loc[shown].sort_values('mean_spearman')
    ax.barh(selected_rows.index, selected_rows.mean_spearman)
    ax.set_xlabel('Mean held-out development-fold Spearman')
    ax.set_xlim(0, 1)
    ax.set_title('Explicit EPIT formulas; 452 Fe/Ni development rows')
    figure.tight_layout()
    figure.savefig(output/'formula_comparison.svg')
    plt.close(figure)
    blocks = []
    for name, summary in summaries.items():
        blocks.append(f'<h2>{html.escape(name)}</h2>'+summary.round(4).to_html(index=False))
    model = artifacts[selected]
    equation = '\n'.join(f'{coefficient:+.6f} * z({term})' for term, coefficient in zip(model['term_names'], model['coefficients']) if abs(coefficient) > 1e-8)
    source_links = ''.join(f'<li><a href="{html.escape(s["url"])}">{html.escape(s["topic"])}</a></li>' for s in SOURCES)
    count = len(report['candidate_specs'])
    content = f'''<!doctype html><html><meta charset="utf-8"><title>EPIT formula extension study</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#20252b}}td,th{{padding:6px;text-align:left}}table{{border-collapse:collapse;font-size:13px}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:16px}}img{{max-width:100%}}</style>
<h1>EPIT formula extension study</h1>
<p>{count} candidates: 9 current rules and {count-9} explicit equations/extensions. No PySR. Only 452 applicable Fe/Ni development rows; the 152 final-test targets were masked and excluded. These are formula scores, not TabICL model scores.</p>
<p>Exploratory follow-up: ten heat/surface ablations were added after the first 27-candidate study. Development data were reused; nested fitting prevents direct held-out-target leakage but these results are not an independent confirmation of the researcher's hypothesis choices.</p>
<p>Each outer development fold is held out while the other four folds choose regularization and formula. Imputation, scaling, supported categories and the old formula are refitted inside every fitting split. Among candidates within 0.005 inner-fold Spearman of the best, prefer fewer terms, then stronger regularization.</p>
<p><b>Interpretation:</b> individual-family scores describe the candidates examined; choosing the largest of them introduces selection optimism. The nested_new_and_existing row assesses the complete selection procedure. nested_existing_only assesses choosing among the original nine rules by the same procedure. Full-development fits below are for future final training only and were never used to compute validation scores.</p>
<img src="formula_comparison.svg" alt="Formula comparison">
{''.join(blocks)}
<h2>Full-development selection: {html.escape(selected)}</h2>
<p>Score equation (standardization constants and missing-value handling are in the corresponding JSON):</p><pre>{html.escape(equation)}</pre>
<p>Numeric terms use fitting-row median imputation and standardization. Missing and unseen categories receive the mean fitted category contribution. No missingness indicator, source ID, reference, target-derived column or final-test data is a predictor. Category effects are empirical experimental-condition associations, not established physical laws.</p>
<p>Basic PREN/material coefficients are constrained nonnegative, and direct temperature/log-chloride coefficients nonpositive. Interactions and category corrections may modify these effects, so the full equation is not globally monotone. The thermal-window shape is a hypothesis, not a measured sensitization law. Coefficient bounds, regularization and eligibility are documented in source and artifacts.</p>
<p>Source-publication-blocked folds are an additional development robustness check. They reuse the same development rows and never replace the frozen final partition. Bootstrap intervals condition on the fitted models and do not represent training-seed uncertainty.</p>
<h2>Research grounding</h2><ul>{source_links}</ul>
<p>Literature motivates candidate effects; it does not validate their fitted coefficients on other alloys or conditions. No current generator, Optuna study, or training configuration was changed.</p>
</html>'''
    (output/'report.html').write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--split-manifest', type=Path, default=DEFAULT_SPLIT_DIR/'split_manifest.json')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('--bootstrap-samples must be positive')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    study, frozen, identity = load_study(args.split_manifest)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'schema': 'epit_explicit_formula_study_v2',
        'research_history': '27 initial candidates; ten heat/surface ablations added after inspecting formula_extensions_v1. Exploratory reuse of development data, not independent confirmation.',
        'scope': 'Fe Alloy and NiCrMo Alloy', 'development_rows': len(study.y),
        'development_task_row_indices': study.rows.tolist(), 'final_test_targets_used': False,
        'final_test_targets_masked': True, 'split_manifest_sha256': frozen.manifest_sha256,
        'split_lock_sha256': frozen.lock_sha256, 'reviewed_formula_inputs': identity,
        'candidate_specs': [asdict(spec) for spec in candidates()], 'penalties': PENALTIES,
        'selection_tolerance': SELECTION_TOLERANCE, 'sources': SOURCES,
        'feature_definitions': study.definitions,
        'code_sha256': {p.name: sha256_file(p) for p in [Path(__file__), Path(__file__).with_name('formula_extension_study.py'), Path(__file__).with_name('target_rules.py'), Path(__file__).with_name('calibrate_target_rules.py')]},
        'experiments': {}}
    coverage_table(study).to_csv(output/'feature_coverage.csv', index=False)
    (output/'study_design.json').write_text(json.dumps(report, indent=2)+'\n')
    source_folds = np.empty(len(study.y), dtype=int)
    for number, (_, validation) in enumerate(GroupKFold(5).split(study.frame, groups=study.source_groups), 1):
        source_folds[validation] = number
    summaries = {}
    for label, folds, groups in [('frozen_development_folds', study.folds, study.groups),
                                  ('source_publication_folds', source_folds, study.source_groups)]:
        print('START '+label, flush=True)
        rows, predictions, selections, fitted = nested_evaluation(study, folds)
        summary = summarize(rows)
        summaries[label] = summary
        subdir = output/label
        subdir.mkdir()
        summary.to_csv(subdir/'summary.csv', index=False)
        pd.DataFrame(rows).to_csv(subdir/'fold_scores.csv', index=False)
        pd.DataFrame(predictions).to_csv(subdir/'predictions.csv', index=False)
        coefficient_stability(fitted).to_csv(subdir/'coefficient_stability.csv', index=False)
        (subdir/'selections.json').write_text(json.dumps(selections, indent=2)+'\n')
        (subdir/'fold_models.json').write_text(json.dumps({f'{name}__fold{fold}': model for (name, fold), model in fitted.items()}, indent=2)+'\n')
        paired = bootstrap_difference(predictions, study, groups, samples=args.bootstrap_samples)
        report['experiments'][label] = {'summary': summary.to_dict(orient='records'), 'selections': selections,
            'fold_by_task_row': dict(zip(map(str, study.rows), map(int, folds))), **paired}
        if label == 'frozen_development_folds':
            historical = json.loads((ROOT/'corrosion_datasets/analysis/epit_pipeline/target_rules_v4/calibration_summary.json').read_text())
            for previous in historical['rules']:
                actual = summary.set_index('name').loc['existing_'+previous['rule_family'], 'mean_spearman']
                if not np.isclose(actual, previous['mean_fold_spearman'], atol=1e-10, rtol=0):
                    raise RuntimeError('Current baseline was not reproduced: '+previous['rule_family'])
            report['all_nine_current_baselines_reproduced'] = True
        (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(summary.head(8).to_string(index=False), flush=True)
        print('PAIRED INTERVAL', paired, flush=True)
    artifacts, selection, selected = full_development_fits(study)
    folder = output/'full_development_formulas'
    folder.mkdir()
    for name, artifact in artifacts.items():
        artifact.update(purpose='Final-training coefficient candidate only; must not be used for Optuna validation on its fitting rows',
                        applicability=report['scope'], feature_definitions=study.definitions,
                        split_manifest_sha256=frozen.manifest_sha256)
        (folder/(name+'.json')).write_text(json.dumps(artifact, indent=2)+'\n')
    report['full_development_selected'] = selected
    report['full_development_selection_scores'] = selection
    report['completed'] = True
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    write_report(output, report, summaries, artifacts)
    print('COMPLETE', output, flush=True)
    print('FINAL DEVELOPMENT SELECTION', selected, flush=True)


if __name__ == '__main__':
    main()
