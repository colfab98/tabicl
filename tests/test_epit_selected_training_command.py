from argparse import Namespace
import sys
import pytest
from scripts.epit_pipeline import train_selected_pairs as launch
from tabicl.train.train_config import build_parser

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_launch_plan_uses_selected_recipe_without_training(device,tmp_path,monkeypatch):
    recipe=launch.ROOT/'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k'
    if not (recipe/'retained_pairs.csv').exists():
        pytest.skip('Local recipe unavailable.')
    # Planning validates existing artifacts and arguments but never launches a process.
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('Planning launched training'))
    args=Namespace(recipe_dir=recipe,run_name='selected_pair_command_test',device=device,
                   max_steps=10000,nproc_per_node=1,workers=4)
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
