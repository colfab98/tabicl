"""Combined selected-pair parity, allocation, distributed slicing and resume."""
from argparse import Namespace
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tabicl.prior.combined_selected_pairs import CombinedSelectedPairPrior, VERSION
from tabicl.prior.original_mixture import OriginalPriorMixture
from tabicl.prior.selected_pairs_single_rule import install_single_rule_sampler
from tabicl.train.run_combined_selected_pairs import CombinedSelectedPairTrainer, SOCCOL_OVERRIDE_KEYS
from tests.test_selected_pairs_single_rule import make_selected, config_for, RECIPES
from tests.test_paired_prior_streams import make_prior, assert_batch_equal

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def mixture():
    return CombinedSelectedPairPrior(install_single_rule_sampler(make_selected('epit')),
                                     install_single_rule_sampler(make_selected('soccol')))


def config(tmp_path,workers=0):
    e=config_for(tmp_path,'epit',workers)
    s=config_for(tmp_path,'soccol',workers)
    e.batch_size=s.batch_size=16
    e.combined_soccol_overrides={k:deepcopy(getattr(s,k)) for k in SOCCOL_OVERRIDE_KEYS
                                if getattr(e,k)!=getattr(s,k)}
    return e


def configured(c,rank=0,world_size=1):
    obj=CombinedSelectedPairTrainer.__new__(CombinedSelectedPairTrainer)
    obj.config=deepcopy(c);obj.config.batch_size//=world_size
    obj.ddp_rank,obj.ddp_world_size,obj.master_process=rank,world_size,False
    obj.missingness_augmenter=None
    obj.configure_prior()
    return obj.dataloader.dataset


def test_actual_source_tables_match_each_standalone_and_pure_generic():
    combined=mixture()
    epit=install_single_rule_sampler(make_selected('epit'))
    soccol=install_single_rule_sampler(make_selected('soccol'))
    generic=make_prior('epit',0.,prior_type='mix_scm',dataset_cls=OriginalPriorMixture)
    seen=set()
    for _ in range(6):
        references={}
        for source,prior in [(0,generic),(1,epit),(2,soccol)]:
            b=prior.get_batch()
            for i,(original_source,table) in enumerate(prior.last_batch_ids):
                if (source==0 and original_source==0) or (source!=0 and original_source==1):
                    references[source,table]=tuple(t[i] for t in b[:5])
        batch=combined.get_batch()
        assert len(combined.last_batch_ids)==combined.batch_size
        for i,key in enumerate(combined.last_batch_ids):
            seen.add(key[0])
            assert_batch_equal(tuple(t[i] for t in batch[:5]),references[key])
        assert batch[4].unique().numel()==1
        assert batch[3].unique().numel()==1
        assert torch.isfinite(batch[1]).all()
        for start in range(0,combined.batch_size,combined.group_size):
            assert len({key[0] for key in combined.last_batch_ids[start:start+combined.group_size]})==1
    assert seen=={0,1,2}
    assert [p.max_features for p in combined.priors]==[100,70,58]
    assert combined.priors[0].min_features==2
    assert combined.priors[0]._get_prior_mix_probs().tolist()==[.7,.3]


def test_allocation_and_rounding_carry_over_10000_steps():
    prior=mixture();prior.batch_size=512;prior.group_size=4
    counts=np.asarray([prior._global_group_counts(i) for i in range(10000)])
    assert np.all(counts.sum(axis=1)==128) and np.all(counts>=0)
    np.testing.assert_array_equal(counts[:,1:].sum(axis=0),
        [math.floor(10000*128*r) for r in (prior.epit_ratio,prior.soccol_ratio)])
    assert prior.epit_ratio==pytest.approx(2300/7600/2)
    assert prior.soccol_ratio==pytest.approx(10897/37450/2)
    assert counts[:,1].min()==19 and counts[:,1].max()==20
    assert counts[:,2].min()==18 and counts[:,2].max()==19


def test_state_replay_and_rejection_of_old_state():
    p=mixture();p.get_batch();state=p.sampling_state_dict();expected=p.get_batch()
    q=mixture();q.load_sampling_state_dict(state);assert_batch_equal(expected,q.get_batch())
    with pytest.raises(ValueError,match='another prior policy'):
        q.load_sampling_state_dict({'version':'original_tabicl_mixture_v1','next_batch':1})
    assert state['version']==VERSION


def test_unmatched_source_settings_rejected():
    e=install_single_rule_sampler(make_selected('epit'))
    s=install_single_rule_sampler(make_selected('soccol'))
    s.seed+=1
    with pytest.raises(ValueError,match='seed'):
        CombinedSelectedPairPrior(e,s)
    with pytest.raises(ValueError,match='one rule per table'):
        CombinedSelectedPairPrior(make_selected('epit'),make_selected('soccol'))


def test_distributed_slices_match_the_same_global_batch(tmp_path):
    c=config(tmp_path)
    single=configured(c)
    ranks=[configured(c,i,2) for i in range(2)]
    assert ranks[0].checkpoint_identity()==ranks[1].checkpoint_identity()
    for _ in range(3):
        batch=single.get_batch()
        expected={key:tuple(t[i] for t in batch) for i,key in enumerate(single.last_batch_ids)}
        seen=set()
        for rank in ranks:
            actual=rank.get_batch()
            for i,key in enumerate(rank.last_batch_ids):
                assert key not in seen;seen.add(key)
                assert_batch_equal(tuple(t[i] for t in actual),expected[key])
        assert seen==set(expected)


@pytest.mark.parametrize('workers',[0,2])
def test_full_cpu_training_resume_restores_batches_weights_and_optimizer(tmp_path,workers):
    c=config(tmp_path,workers)
    first=CombinedSelectedPairTrainer(c)
    if workers:first.dataloader.kwargs['multiprocessing_context']='spawn'
    batch=next(first.dataloader)
    first.run_batch(batch);first.curr_step=1;first.save_checkpoint('step-1.ckpt')
    expected=next(first.dataloader);first.run_batch(expected)
    restored_config=deepcopy(c);restored_config.max_steps+=2
    restored=CombinedSelectedPairTrainer(restored_config)
    if workers:restored.dataloader.kwargs['multiprocessing_context']='spawn'
    assert restored.curr_step==1
    actual=next(restored.dataloader);assert_batch_equal(expected,actual)
    restored.run_batch(actual)
    for key,value in first.raw_model.state_dict().items():
        torch.testing.assert_close(value,restored.raw_model.state_dict()[key],atol=0,rtol=0)
    assert first.scheduler.state_dict()==restored.scheduler.state_dict()
    for key,state in first.optimizer.state_dict()['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,restored.optimizer.state_dict()['state'][key][name],atol=0,rtol=0)
    from tabicl.train.run_selected_pairs_single_rule import SingleRuleTrainer
    with pytest.raises(ValueError,match='matching recipe/profile/seed'):
        SingleRuleTrainer(deepcopy(c))
    changed=deepcopy(c);changed.prior_stream_seed+=1
    with pytest.raises(ValueError,match='matching recipe/profile/seed'):
        CombinedSelectedPairTrainer(changed)


def test_overrides_cannot_change_model_or_generic_configuration(tmp_path):
    c=config(tmp_path);c.combined_soccol_overrides['lr']=.2
    with pytest.raises(ValueError,match='only dataset-specific'):
        configured(c)


@pytest.mark.parametrize('processes',[1,2])
def test_plan_preserves_standalone_training_parameters(processes):
    from scripts.combined_corrosion import train_selected_pairs_single_rule as launcher
    from scripts.epit_pipeline import train_selected_pairs_single_rule as standalone
    from tabicl.train.run_combined_selected_pairs import build_parser
    args=Namespace(recipe_dir=RECIPES['epit'],soccol_recipe_dir=RECIPES['soccol'],
        run_name='combined_plan_test_only',device='cuda',nproc_per_node=processes,max_steps=500,
        scheduler_total_steps=10000,workers=4,resume=False)
    plan=launcher.build_plan(args);old=standalone.build_plan(args)
    command=list(plan['command'])
    extra=command.index('--combined_soccol_overrides');del command[extra:extra+2]
    command[command.index(str(launcher.TRAINER))]=str(standalone.TRAINER)
    from scripts.epit_pipeline.train_final import replace_option
    replace_option(command,'--checkpoint_dir',old['checkpoint_dir'])
    replace_option(command,'--informed_prior_ratio',repr(old['informed_proportion']))
    assert command==old['command']
    assert not plan['automatic_evaluation']
    assert plan['scheduler_total_steps']==10000 and plan['max_steps']==500
    parsed=build_parser().parse_args(plan['command'][plan['command'].index(str(launcher.TRAINER))+1:])
    assert parsed.pitting_composition_perturb_strength==.3
    assert parsed.informed_target_mix_weight==1 and parsed.pitting_coefficient_variation==0
    assert sum(plan['source_proportions'].values())==pytest.approx(1.)


def test_launcher_resume_validates_original_plan(tmp_path,monkeypatch):
    from scripts.combined_corrosion import train_selected_pairs_single_rule as launcher
    monkeypatch.setattr(launcher,'CHECKPOINT_ROOT',tmp_path/'checkpoints')
    monkeypatch.setattr(launcher,'OUTPUT_ROOT',tmp_path/'outputs')
    args=launcher.parse_args(['--run-name','combined_resume_plan_test','--max-steps','500','--dry-run'])
    first=launcher.build_plan(args)
    checkpoint=Path(first['checkpoint_dir']);checkpoint.mkdir(parents=True)
    (checkpoint/'step-500.ckpt').touch()
    output=Path(first['output_dir']);output.mkdir(parents=True)
    (output/'training_plan.json').write_text(json.dumps(first))
    with pytest.raises(FileExistsError,match='Run already exists'):
        launcher.build_plan(args)
    args.resume=True;args.max_steps=1000
    resumed=launcher.build_plan(args)
    assert resumed['max_steps']==1000 and resumed['scheduler_total_steps']==10000
    args.workers+=1
    with pytest.raises(ValueError,match='identical recipes'):
        launcher.build_plan(args)


@pytest.mark.parametrize('ds', ['mpea', 'epit', 'soccol', 'rebar'])
def test_evaluation_delegates_native_inputs_and_cv_without_changing_checkpoint_scope(tmp_path, ds):
    from scripts.combined_corrosion.evaluate_selected_pairs import commands
    result=commands(['--dataset',ds,'--candidate-run','candidate','--output-dir',str(tmp_path),
        '--checkpoint','all','--checkpoint-scope','common','--max-checkpoint-step','3000',
        '--device','cpu'])
    assert len(result)==1
    command=result[0]
    assert command[2]==f'scripts.{ds}_pipeline.evaluate_checkpoint_comparison'
    assert command[command.index('--output-dir')+1]==str(tmp_path.with_name(f'{tmp_path.name}_{ds}'))
    assert command[command.index('--checkpoint-scope')+1]=='common'
    assert command[command.index('--exploratory-cv-folds')+1]=='5'
    assert '--legacy-model-inputs' not in command
    if ds=='mpea':
        assert commands(['--output-dir',str(tmp_path)])[0][2]==command[2]
