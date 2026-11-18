"""Compare retained gains and a generic-exposure delay on frozen CV predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3200]
RATIO = 2300 / 7600


def generic_steps(step):
    # Exact cumulative allocation: 128 groups of four tables per update.
    return step - np.floor(step * 128 * RATIO) / 128


def main():
    records = []
    for step in STEPS:
        frame = pd.read_csv(ROOT / f'step_{step}/samples.csv')
        assert len(frame) == 760 and frame.task_row_index.nunique() == 760
        assert frame.group.value_counts().to_dict() == {'retained': 424, 'not_retained': 336}
        for model in ['generic', 'informed']:
            if model == 'generic' and step == 3200:
                continue  # This export reuses the generic 3000 reference.
            for fold, subset in [('all', frame), *list(frame.groupby('cv_fold'))]:
                for group, part in [('all', subset), *list(subset.groupby('group'))]:
                    correlations = [spearmanr(f.observed_mV_generic, f[f'prediction_mV_{model}']).statistic
                                    for _, f in part.groupby('cv_fold')]
                    records.append({'checkpoint_step': step, 'model': model, 'fold': str(fold), 'group': group,
                                    'n': len(part), 'mae_mV': part[f'{model}_error_mV'].mean(),
                                    'spearman': np.mean(correlations)})
    metrics = pd.DataFrame(records)
    metrics.to_csv(ROOT / 'subgroup_metrics.csv', index=False)
    comparisons = []
    for fold in ['all', '1', '2', '3', '4', '5']:
        for group in ['retained', 'not_retained', 'all']:
            subset = metrics[(metrics.fold == fold) & (metrics.group == group)]
            gen = subset[subset.model == 'generic'].set_index('checkpoint_step').sort_index()
            informed = subset[subset.model == 'informed'].set_index('checkpoint_step').sort_index()
            for step, row in informed.iterrows():
                exposure = generic_steps(step)
                result = {'checkpoint_step': step, 'generic_exposure_equivalent_step': exposure,
                          'group': group, 'fold': fold, 'n': row['n']}
                for metric in ['mae_mV', 'spearman']:
                    same = gen.loc[step, metric] if step in gen.index else np.nan
                    lagged = np.interp(exposure, gen.index, gen[metric]) if gen.index.min() <= exposure <= gen.index.max() else np.nan
                    result.update({f'informed_{metric}': row[metric], f'generic_same_step_{metric}': same,
                                   f'generic_exposure_matched_{metric}': lagged,
                                   f'delta_same_step_{metric}': row[metric] - same,
                                   f'delta_exposure_matched_{metric}': row[metric] - lagged})
                lower = gen.index[gen.index <= exposure]
                upper = gen.index[gen.index >= exposure]
                result['interpolation_lower_step'] = int(lower.max()) if len(lower) else None
                result['interpolation_upper_step'] = int(upper.min()) if len(upper) else None
                comparisons.append(result)
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(ROOT / 'checkpoint_comparison.csv', index=False)
    final = comparisons[(comparisons.fold == 'all') & comparisons.group.ne('all')]
    print(final[['checkpoint_step', 'group', 'generic_exposure_equivalent_step',
                 'informed_mae_mV', 'generic_same_step_mae_mV', 'generic_exposure_matched_mae_mV',
                 'delta_same_step_mae_mV', 'delta_exposure_matched_mae_mV',
                 'informed_spearman', 'generic_same_step_spearman', 'generic_exposure_matched_spearman']].round(3).to_string(index=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, group in enumerate(['retained', 'not_retained']):
        subset = metrics[(metrics.fold == 'all') & (metrics.group == group)]
        gen = subset[subset.model == 'generic'].sort_values('checkpoint_step')
        informed = subset[subset.model == 'informed'].sort_values('checkpoint_step')
        for row, metric in enumerate(['mae_mV', 'spearman']):
            ax = axes[row, col]
            ax.plot(gen.checkpoint_step, gen[metric], 'o-', label='Generic: actual steps', color='#777777')
            ax.plot(informed.checkpoint_step, informed[metric], 'o-', label='Single-rule informed mixture', color='#1465ba')
            ax.plot(gen.checkpoint_step / (1 - RATIO), gen[metric], '--', label='Generic: 30.26% exposure delay', color='#d77c12')
            ax.set_xlim(450, 3300)
            ax.set_title(f'{group.replace("_", " ").title()} ({424 if group == "retained" else 336} samples)')
            ax.set_xlabel('Optimizer steps')
            ax.set_ylabel('MAE (mV), lower is better' if row == 0 else 'Mean fold Spearman, higher is better')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('EPIT: exploratory five-fold CV; retained status fixed before this analysis')
    fig.savefig(ROOT / 'retention_delay.png', dpi=170)
    fig.savefig(ROOT / 'retention_delay.pdf')
    metadata = {'source_results': 'corrosion_datasets/analysis/epit_pipeline/selected_pairs_single_rule_through_3500_cv5',
                'steps': STEPS, 'informed_ratio': RATIO,
                'retained_definition': 'parent appears in at least one retained sample/formula pair',
                'mae_aggregation': 'all query samples pooled once; five disjoint folds',
                'spearman_aggregation': 'unweighted mean of five within-fold subgroup correlations',
                'delay_reference': 'linear interpolation of generic metrics at cumulative generic-table exposure',
                'interpretation': 'Exploratory. Exposure matching does not match LR position or exact generic table identities.'}
    (ROOT / 'analysis_identity.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
