"""Soccol adapter for EPIT's one-parent, formula-only selected-pair stream."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .epit_selected_pairs import SelectedPairPriorMixture as EpitMixture, _SelectedPrior
from .original_mixture import OriginalPriorMixture
from .epit_schema import EPIT_COMPOSITION_ELEMENTS
from .soccol_reviewed_schema import (
    SOCCOL_REVIEWED_FEATURE_PROFILE, SOCCOL_REVIEWED_FEATURE_COLUMNS,
    SOCCOL_REVIEWED_CATEGORICAL_COLUMNS,
)


class SelectedPairSampler:
    def __init__(self, recipe_dir):
        from scripts.epit_pipeline.artifact_hashes import sha256_file
        from scripts.soccol_pipeline.selected_pair_formulas import load_inputs, source_identity
        self.path = Path(recipe_dir).expanduser().resolve()
        self.pairs = pd.read_csv(self.path / 'retained_pairs.csv')
        self.formulas = json.loads((self.path / 'fitted_formulas.json').read_text())
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        self.dataset, self.rows, identity = load_inputs()
        if identity != self.manifest['data']:
            raise ValueError('Selected pairs refer to different physical inputs.')
        current_sources = source_identity()
        if any(self.manifest['source_sha256'].get(k) != v for k, v in current_sources.items()):
            raise ValueError('Formula implementation changed since pair preparation.')
        all_pairs = pd.read_csv(self.path / 'all_pairs.csv')
        expected_inventory = {(int(i), name) for i in self.rows.task_row_index for name in self.formulas}
        actual_inventory = set(map(tuple, all_pairs[['task_row_index', 'formula']].to_numpy()))
        if all_pairs.duplicated(['task_row_index', 'formula']).any() or actual_inventory != expected_inventory:
            raise ValueError('Incomplete or duplicate pair inventory.')
        # Validate error/gain columns, rather than trusting a manually edited flag.
        numeric = all_pairs[['observed_mV', 'prediction_mV', 'prediction_mV_generic', 'gain_mV']]
        if not np.isfinite(numeric).all().all():
            raise ValueError('Nonfinite pair predictions.')
        gain = np.abs(all_pairs.prediction_mV_generic - all_pairs.observed_mV) - np.abs(all_pairs.prediction_mV - all_pairs.observed_mV)
        np.testing.assert_allclose(all_pairs.gain_mV, gain, rtol=1e-10, atol=1e-8)
        expected = all_pairs[all_pairs.gain_mV > 0][['task_row_index', 'formula']]
        actual = self.pairs[['task_row_index', 'formula']]
        if actual.duplicated().any() or not len(actual) or set(map(tuple, actual.to_numpy())) != set(map(tuple, expected.to_numpy())):
            raise ValueError('Retained pairs do not match positive gains.')
        for artifact in self.formulas.values():
            if set(artifact['fitting_task_row_indices']) != set(self.rows.task_row_index):
                raise ValueError('Expected all-accepted-row fitted formulas.')
        self.ratio = len(actual) / len(all_pairs)
        self.identity = {'artifacts': {name: sha256_file(self.path / name) for name in
            ('retained_pairs.csv', 'all_pairs.csv', 'fitted_formulas.json', 'manifest.json')},
            'source_sha256': sha256_file(Path(__file__)), 'informed_ratio': self.ratio,
            'input_identity': identity, 'all_data_recipe': True, 'formula_code_sha256': current_sources,
            'shared_selected_pair_sha256': sha256_file(Path(__file__).with_name('epit_selected_pairs.py'))}
        self.categories = {name: {v: i for i, v in enumerate(sorted(self.rows[name].dropna().astype(str).unique()))}
                           for name in SOCCOL_REVIEWED_CATEGORICAL_COLUMNS}
        self.parent_positions = pd.Series(np.arange(len(self.rows)), index=self.rows.task_row_index)

    def sample(self, n, *, perturb_strength, random_state):
        from scripts.soccol_pipeline.selected_pair_formulas import predict_formulas
        if n <= 0 or int(n) != n or not math.isfinite(perturb_strength) or perturb_strength < 0:
            raise ValueError('Invalid sample count or perturbation strength.')
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        pair_indices = rng.integers(len(self.pairs), size=n)
        selected = self.pairs.iloc[pair_indices]
        parents = selected.task_row_index.to_numpy(dtype=int)
        names = selected.formula.to_numpy()
        frame = self.rows.iloc[self.parent_positions.loc[parents].to_numpy()].reset_index(drop=True).copy()
        compositions = frame.loc[:, list(EPIT_COMPOSITION_ELEMENTS)].to_numpy(dtype=float, copy=True)
        if perturb_strength:
            compositions *= np.exp(rng.normal(0, perturb_strength, size=compositions.shape))
            compositions *= 100 / compositions.sum(axis=1, keepdims=True)
        frame.loc[:, list(EPIT_COMPOSITION_ELEMENTS)] = compositions
        targets = np.empty(n)
        values = predict_formulas(frame, {name: self.formulas[name] for name in np.unique(names)})
        for name, prediction in values.items():
            mask = names == name
            targets[mask] = prediction[mask]
        physical = frame.loc[:, list(SOCCOL_REVIEWED_FEATURE_COLUMNS)].copy()
        for column, mapping in self.categories.items():
            physical[column] = physical[column].map(mapping)
        features = physical.to_numpy(dtype=float)
        if not np.isfinite(targets).all() or np.isinf(features).any():
            raise ValueError('Invalid sampled features or targets.')
        return SimpleNamespace(features=features, target_mV=targets, parent_indices=parents,
                               pair_indices=pair_indices, formulas=names, physical=frame)


class SelectedPairPriorMixture(EpitMixture):
    """Same target construction, global group allocation and replay as EPIT.

    Only initialization is dataset-specific. Keep EPIT's existing sampler file
    unchanged so its hash-locked recipes/checkpoints remain resumable.
    """
    def __init__(self, *, selected_pair_dir, **kwargs):
        sampler = SelectedPairSampler(selected_pair_dir)
        fixed = dict(kwargs.get('scm_fixed_hp') or {})
        if fixed.get('pitting_feature_profile') != SOCCOL_REVIEWED_FEATURE_PROFILE or fixed.get('pitting_composition_mode') != 'empirical_features_scm_target':
            raise ValueError('Selected pairs require reviewed Soccol physical features.')
        if fixed.get('pitting_coefficient_variation', 0) != 0:
            raise ValueError('Selected-pair coefficients must stay fixed.')
        fixed['informed_target_mix_weight'] = 1.
        kwargs.update(scm_fixed_hp=fixed, prior_type='hybrid_scm', informed_prior_ratio=1.)
        OriginalPriorMixture.__init__(self, **kwargs)
        old = self.priors[1]
        shape = dict(self._config['shape_settings'])
        shape.update(batch_size=self.batch_size * self.world_size, seq_len_per_gp=False)
        prior = _SelectedPrior(**shape, min_features=old.min_features, max_features=old.max_features,
            prior_type='informed_scm', fixed_hp=old.fixed_hp, sampled_hp=old.sampled_hp)
        prior.sampler = sampler
        self.priors[1] = prior
        self.selected_ratio = sampler.ratio
        self._config.update(selected_pair_recipe=sampler.identity,
            selected_pair_allocation='cumulative rounded global groups; ratio of retained pairs',
            selected_pair_target='calibrated mV then one shared table standardization; no SCM')
        self._config['group_counts'] = 'varies by indexed batch; see selected_pair_allocation'
