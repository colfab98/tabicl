from argparse import Namespace
import sys
import pytest
import torch
from scripts.epit_pipeline import train_selected_pairs as launch
from tabicl.train.train_config import build_parser
from tabicl.train.optim import get_scheduler

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_launch_plan_uses_selected_recipe_without_training(device,tmp_path,monkeypatch):
    recipe=launch.ROOT/'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k'
    if not (recipe/'retained_pairs.csv').exists():
        pytest.skip('Local recipe unavailable.')
    # Planning validates existing artifacts and arguments but never launches a process.
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('Planning launched training'))
    args=Namespace(recipe_dir=recipe,run_name='selected_pair_command_test',device=device,
                   max_steps=10000,scheduler_total_steps=10000,nproc_per_node=1,workers=4)
    plan=launch.build_plan(args)
    assert plan['command'][:2]==[sys.executable,str(launch.ROOT/'src/tabicl/train/run.py')]
    config=build_parser().parse_args(plan['command'][2:])
    assert config.device==device
    assert config.pitting_coefficient_variation==0
    assert config.informed_target_mix_weight==1
    assert config.pitting_composition_perturb_strength==.3
    assert config.informed_prior_ratio==pytest.approx(2300/7600)
    assert config.epit_selected_pair_dir==str(recipe)
    assert config.checkpoint_path is None
    assert not plan['automatic_evaluation']


@pytest.mark.parametrize('horizon',[10000,20000])
def test_training_duration_does_not_change_lr_schedule(horizon,monkeypatch,capsys):
    recipe=launch.ROOT/'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k'
    if not (recipe/'retained_pairs.csv').exists():
        pytest.skip('Local recipe unavailable.')
    plans=[]
    build=launch.build_plan
    def capture(args):
        plan=build(args)
        plans.append(plan)
        return plan
    monkeypatch.setattr(launch,'build_plan',capture)
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('Dry run launched training'))
    for steps in (500,1000):
        argv=['--dry-run','--device','cpu','--run-name','selected_pair_schedule_test',
              '--max-steps',str(steps)]
        if horizon!=10000:
            argv+=['--scheduler-total-steps',str(horizon)]
        launch.main(argv)
    configs=[build_parser().parse_args(plan['command'][2:]) for plan in plans]
    assert [config.max_steps for config in configs]==[500,1000]
    assert all(config.scheduler_total_steps==horizon for config in configs)
    assert all(plan['scheduler_total_steps']==horizon for plan in plans)
    schedules=[get_scheduler(config,torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=config.lr))
               for config in configs]
    for step in (0,100,200,499):
        assert schedules[0].lr_lambdas[0](step)==schedules[1].lr_lambdas[0](step)
