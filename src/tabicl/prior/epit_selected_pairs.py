"""One-parent EPIT generation from a frozen list of successful sample/rule pairs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import math

import numpy as np
import pandas as pd
import torch

from .original_mixture import OriginalPriorMixture, _InformedPrior
from .dataset import PittingProfileInfo
from .epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS, EPIT_EXPANDED_CATEGORICAL_COLUMNS, EPIT_EXPANDED_FEATURE_PROFILE
from .epit_schema import EPIT_COMPOSITION_ELEMENTS, EPIT_COMPOSITION_COLUMNS


class SelectedPairSampler:
    """Frozen formulas, sampled uniformly over retained pairs; no labels in generation."""
    def __init__(self, recipe_dir):
        from scripts.epit_pipeline.artifact_hashes import sha256_file
        from scripts.epit_pipeline.composition_data import load_reviewed_compositions
        from scripts.epit_pipeline.formula_data import load_formula_dataset
        self.path=Path(recipe_dir).expanduser().resolve()
        self.pairs=pd.read_csv(self.path/'retained_pairs.csv')
        self.formulas=json.loads((self.path/'fitted_formulas.json').read_text())
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        self.reviewed=load_reviewed_compositions()
        self.dataset,identity=load_formula_dataset()
        if identity!=self.manifest['data']:
            raise ValueError('Selected pairs refer to different physical inputs.')
        all_pairs=pd.read_csv(self.path/'all_pairs.csv')
        if all_pairs.duplicated(['task_row_index','formula']).any() or len(all_pairs)!=len(self.reviewed.rows)*len(self.formulas):
            raise ValueError('Incomplete or duplicate pair inventory.')
        expected=all_pairs[all_pairs.gain_mV>0][['task_row_index','formula']]
        actual=self.pairs[['task_row_index','formula']]
        if actual.duplicated().any() or not len(actual) or set(map(tuple,actual.to_numpy()))!=set(map(tuple,expected.to_numpy())):
            raise ValueError('Retained pairs do not match positive gains.')
        if not set(actual.formula)<=set(self.formulas) or not actual.task_row_index.between(0,len(self.reviewed.rows)-1).all():
            raise ValueError('Invalid sample/rule identities.')
        for artifact in self.formulas.values():
            if set(artifact['fitting_task_row_indices'])!=set(range(len(self.reviewed.rows))):
                raise ValueError('Expected all-row fitted formulas.')
        self.ratio=len(actual)/len(all_pairs)
        self.identity={'artifacts':{name:sha256_file(self.path/name) for name in
            ('retained_pairs.csv','all_pairs.csv','fitted_formulas.json','manifest.json')},
            'source_sha256':sha256_file(Path(__file__)), 'informed_ratio':self.ratio,
            'input_identity':identity,'all_data_recipe':True}
        root=Path(__file__).resolve().parents[3]
        self.identity['formula_code_sha256']={name:sha256_file(root/name) for name in (
            'scripts/epit_pipeline/target_rules.py','scripts/epit_pipeline/formula_extension_study.py')}
        # Stable per-feature category encoding; retained missingness stays NaN.
        self.categories={name:{v:i for i,v in enumerate(sorted(self.reviewed.rows[name].dropna().astype(str).unique()))}
                         for name in EPIT_EXPANDED_CATEGORICAL_COLUMNS}

    def sample(self,n,*,perturb_strength,random_state):
        from scripts.epit_pipeline.target_rules import get_rule_families
        from scripts.epit_pipeline.formula_extension_study import feature_library,transform_features
        if n<=0 or int(n)!=n or not math.isfinite(perturb_strength) or perturb_strength<0:
            raise ValueError('Invalid sample count or perturbation strength.')
        rng=random_state if isinstance(random_state,np.random.Generator) else np.random.default_rng(random_state)
        pair_indices=rng.integers(len(self.pairs),size=n)
        selected=self.pairs.iloc[pair_indices]
        parents=selected.task_row_index.to_numpy(dtype=int)
        names=selected.formula.to_numpy()
        frame=self.reviewed.rows.iloc[parents].reset_index(drop=True).copy()
        compositions=frame.loc[:,list(EPIT_COMPOSITION_ELEMENTS)].to_numpy(dtype=float, copy=True)
        if perturb_strength:
            compositions*=np.exp(rng.normal(0,perturb_strength,size=compositions.shape))
            compositions*=100/compositions.sum(axis=1,keepdims=True)
        frame.loc[:,list(EPIT_COMPOSITION_ELEMENTS)]=compositions
        rows=[]
        for i,parent in enumerate(parents):
            row=dict(self.dataset.rows[parent])
            row.update(dict(zip(EPIT_COMPOSITION_COLUMNS,compositions[i].tolist())))
            rows.append(row)
        data=replace(self.dataset,rows=rows,target=np.full(n,np.nan))
        indices=np.arange(n)
        raw,categories,_=feature_library(frame)
        def score(artifact):
            if artifact['mode']=='existing':
                rule=get_rule_families([artifact['family']])[0]
                return rule.transform_terms(data,indices,artifact['feature_state'])@np.asarray(artifact['coefficients'])
            terms=transform_features(raw,categories,artifact['feature_state'],indices)
            base=(score(artifact['base'])-artifact['base_center'])/artifact['base_scale']
            return np.column_stack([base,terms])@np.asarray(artifact['coefficients'])
        targets=np.empty(n)
        for name in np.unique(names):
            artifact=self.formulas[name]
            values=score(artifact)
            calibration=artifact['mV_calibration']
            values=calibration['intercept']+calibration['slope']*values
            mask=names==name
            targets[mask]=values[mask]
        physical=frame.loc[:,list(EPIT_EXPANDED_FEATURE_COLUMNS)].copy()
        for column,mapping in self.categories.items():
            physical[column]=physical[column].map(mapping)
        features=physical.to_numpy(dtype=float)
        if not np.isfinite(targets).all() or np.isinf(features).any():
            raise ValueError('Invalid sampled features or targets.')
        return SimpleNamespace(features=features,target_mV=targets,parent_indices=parents,
                               pair_indices=pair_indices,formulas=names,physical=frame)


class _SelectedPrior(_InformedPrior):
    def _apply_empirical_pitting_feature_profile(self,X,blocks,family):
        self.selected_batch=self.sampler.sample(len(X),
            perturb_strength=float(self.fixed_hp['pitting_composition_perturb_strength']),
            random_state=int(np.random.randint(0,2**32-1)))
        features=self.selected_batch.features
        self.last_pitting_feature_batch=SimpleNamespace(features=features)
        # Temporary finite values; original-input path restores missingness and
        # imputes from context only. Formula evaluation uses physical features.
        values=torch.as_tensor(np.nan_to_num(features,nan=0.),device=X.device,dtype=X.dtype)
        return values,PittingProfileInfo(applied=True,material_style='selected_pair',physical_profile_applied=True)

    def _generate_scm_epit_target_from_physical_features(self,X,params,prior_cls):
        target=torch.as_tensor(self.selected_batch.target_mV,device=X.device,dtype=X.dtype)
        self.last_pitting_target_drive=target.detach().cpu()
        self.last_pitting_target_rule={'policy':'selected_pair_fixed_formula_only'}
        return self._standardize_signal(target)


def informed_groups(step,groups,ratio):
    """Carry rounding across batches instead of rounding 30.26% permanently."""
    return math.floor((step+1)*groups*ratio)-math.floor(step*groups*ratio)


class SelectedPairPriorMixture(OriginalPriorMixture):
    def __init__(self,*,selected_pair_dir,**kwargs):
        sampler=SelectedPairSampler(selected_pair_dir)
        fixed=dict(kwargs.get('scm_fixed_hp') or {})
        if fixed.get('pitting_feature_profile')!=EPIT_EXPANDED_FEATURE_PROFILE or fixed.get('pitting_composition_mode')!='empirical_features_scm_target':
            raise ValueError('Selected pairs require expanded EPIT physical features.')
        if fixed.get('pitting_coefficient_variation',0)!=0:
            raise ValueError('Selected-pair coefficients must stay fixed.')
        fixed['informed_target_mix_weight']=1.
        kwargs.update(scm_fixed_hp=fixed,prior_type='hybrid_scm',informed_prior_ratio=1.)
        super().__init__(**kwargs)
        old=self.priors[1]
        shape=dict(self._config['shape_settings'])
        shape.update(batch_size=self.batch_size*self.world_size,seq_len_per_gp=False)
        prior=_SelectedPrior(**shape,min_features=old.min_features,max_features=old.max_features,
            prior_type='informed_scm',fixed_hp=old.fixed_hp,sampled_hp=old.sampled_hp)
        prior.sampler=sampler
        self.priors[1]=prior
        self.selected_ratio=sampler.ratio
        self._config.update(selected_pair_recipe=sampler.identity,
            selected_pair_allocation='cumulative rounded global groups; ratio of retained pairs',
            selected_pair_target='calibrated mV then one shared table standardization; no SCM')
        self._config['group_counts']='varies by indexed batch; see selected_pair_allocation'

    def _global_group_counts(self,step):
        groups=self.batch_size*self.world_size//self.group_size
        count=informed_groups(step,groups,self.selected_ratio)
        return groups-count,count
