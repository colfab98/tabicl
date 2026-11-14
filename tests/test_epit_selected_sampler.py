from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from tabicl.prior.epit_selected_pairs import SelectedPairSampler,SelectedPairPriorMixture,informed_groups
from tabicl.prior.epit_schema import EPIT_COMPOSITION_ELEMENTS
from tests.test_paired_prior_streams import make_prior,assert_batch_equal

RECIPE=Path(__file__).resolve().parents[1]/'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k'

@pytest.fixture(scope='module')
def sampler():
    if not (RECIPE/'retained_pairs.csv').exists():
        pytest.skip('Local fitted pair recipe not installed.')
    return SelectedPairSampler(RECIPE)


def test_replays_fitted_predictions_without_perturbation(sampler):
    batch=sampler.sample(1500,perturb_strength=0,random_state=42)
    expected=sampler.pairs.iloc[batch.pair_indices]
    np.testing.assert_allclose(batch.target_mV,expected.prediction_mV,rtol=1e-10,atol=1e-8)
    assert len(set(batch.formulas))==10
    parents=sampler.reviewed.rows.iloc[batch.parent_indices].reset_index(drop=True)
    pd.testing.assert_frame_equal(batch.physical,parents)


def test_perturbation_keeps_conditions_zeros_and_mass(sampler):
    a=sampler.sample(100,perturb_strength=.3,random_state=43)
    b=sampler.sample(100,perturb_strength=.3,random_state=43)
    np.testing.assert_array_equal(a.target_mV,b.target_mV)
    parents=sampler.reviewed.rows.iloc[a.parent_indices].reset_index(drop=True)
    columns=list(EPIT_COMPOSITION_ELEMENTS)
    perturbed=a.physical[columns].to_numpy()
    np.testing.assert_allclose(perturbed.sum(axis=1),100)
    assert (perturbed[parents[columns].to_numpy()==0]==0).all()
    pd.testing.assert_frame_equal(a.physical.drop(columns=columns),parents.drop(columns=columns))
    assert not np.allclose(perturbed,parents[columns])


def test_pair_probabilities_and_ratio_rounding(sampler):
    # Pair choices use the exact production RNG call without expensive formula evaluation.
    choices=np.random.default_rng(42).integers(len(sampler.pairs),size=230000)
    counts=np.bincount(choices,minlength=len(sampler.pairs))
    assert counts.min()>60 and counts.max()<145
    assert sampler.ratio==pytest.approx(2300/7600)
    assert sum(informed_groups(i,4,sampler.ratio) for i in range(190))==230


def test_training_batch_and_resume(sampler):
    torch.set_num_threads(1)
    kwargs=dict(dataset_cls=SelectedPairPriorMixture,selected_pair_dir=RECIPE,
                hp={'pitting_coefficient_variation':0.,'pitting_composition_perturb_strength':.3})
    prior=make_prior(**kwargs)
    first=prior.get_batch()
    assert all(torch.isfinite(x).all() for x in first[0].unbind()) and torch.isfinite(first[1]).all()
    assert (first[5]==1).sum()==2
    state=prior.sampling_state_dict()
    expected=prior.get_batch()
    resumed=make_prior(**kwargs)
    resumed.load_sampling_state_dict(state)
    assert_batch_equal(expected,resumed.get_batch())


def test_generic_stream_unchanged_and_no_scm_target(sampler,monkeypatch):
    from tabicl.prior.epit_selected_pairs import _SelectedPrior
    from tabicl.prior.original_mixture import OriginalPriorMixture
    def forbidden(*args,**kwargs):
        raise AssertionError('Selected informed targets must not call SCM target generation.')
    monkeypatch.setattr(_SelectedPrior,'_generate_scm_target_from_physical_features',forbidden)
    torch.set_num_threads(1)
    selected=make_prior(dataset_cls=SelectedPairPriorMixture,selected_pair_dir=RECIPE,
        hp={'pitting_coefficient_variation':0.,'pitting_composition_perturb_strength':.3})
    reference=make_prior(dataset_cls=OriginalPriorMixture,ratio=.25)
    a,b=selected.get_batch(),reference.get_batch()
    assert torch.equal(a[5],b[5])
    for i in torch.where(a[5]==0)[0].tolist():
        for left,right in zip(a[:5],b[:5]):
            torch.testing.assert_close(left[i],right[i],rtol=0,atol=0)


def test_gpu_partition_preserves_global_tables_and_resume(sampler):
    torch.set_num_threads(1)
    kwargs=dict(dataset_cls=SelectedPairPriorMixture,selected_pair_dir=RECIPE,
                hp={'pitting_coefficient_variation':0.,'pitting_composition_perturb_strength':.3})
    single=make_prior(**kwargs)
    single.batch_size=16
    ranks=[make_prior(**kwargs,rank=rank,world_size=2) for rank in range(2)]
    informed_total=0
    for _ in range(4):
        whole=single.get_batch()
        parts=[prior.get_batch() for prior in ranks]
        assert single.last_batch_ids==[key for prior in ranks for key in prior.last_batch_ids]
        for rank,part in enumerate(parts):
            for row in range(8):
                assert_batch_equal(tuple(t[rank*8+row] for t in whole),tuple(t[row] for t in part))
        informed_total+=int(whole[5].sum())
    assert informed_total==18

    states=[prior.sampling_state_dict() for prior in ranks]
    expected=[prior.get_batch() for prior in ranks]
    for rank in range(2):
        resumed=make_prior(**kwargs,rank=rank,world_size=2)
        resumed.load_sampling_state_dict(states[rank])
        assert_batch_equal(expected[rank],resumed.get_batch())
