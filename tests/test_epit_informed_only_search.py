from dataclasses import asdict, replace

import optuna
import pytest

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline import train_final
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_PROFILE


MODE = "empirical_features_scm_target"


@pytest.fixture(scope="module")
def rules():
    args = search.parse_args([])
    return search.load_target_rule_config(
        summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest,
    )


def _selected_params():
    # Optuna stores only the three searched values; rho comes from the study.
    values = {"mlp_prob": 0.37, "informed_target_mix_weight": 0.42,
              "pitting_composition_perturb_strength": 0.27}
    return search.trial_params_from_mapping(
        values, composition_mode=MODE, fixed_informed_prior_ratio=1.,
    )


def test_real_optuna_trial_has_three_continuous_parameters_and_fixed_ratio(rules):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()
    params = search.sample_params(trial, composition_mode=MODE)
    assert params.informed_prior_ratio == 1.
    assert set(trial.params) == {"mlp_prob", "informed_target_mix_weight", "pitting_composition_perturb_strength"}
    for name, bounds in {"mlp_prob": (0., 1.), "informed_target_mix_weight": (0., 1.),
                         "pitting_composition_perturb_strength": (0., 0.30)}.items():
        distribution = trial.distributions[name]
        assert isinstance(distribution, optuna.distributions.FloatDistribution)
        assert (distribution.low, distribution.high) == bounds
        assert distribution.step is None and distribution.log is False
    fingerprint = search.build_pipeline_fingerprint(search.parse_args([]), rules)
    assert "informed_prior_ratio" not in fingerprint["search_space"]
    assert fingerprint["fixed_prior"]["informed_prior_ratio"] == 1.
    assert fingerprint["search_space"]["mlp_prob"] == {"distribution": "float", "low": 0., "high": 1.}
    assert fingerprint["search_space"]["pitting_composition_perturb_strength"] == [0., 0.30]
    reconstructed = search.trial_params_from_mapping(
        trial.params, composition_mode=MODE,
        fixed_informed_prior_ratio=fingerprint["fixed_prior"]["informed_prior_ratio"],
    )
    assert reconstructed == params


@pytest.mark.parametrize("mlp,weight,perturb", [(0., 0., 0.), (0.37, 0.42, 0.27), (1., 1., 0.30)])
def test_broader_parameters_reach_the_training_command(rules, tmp_path, mlp, weight, perturb):
    trial = optuna.trial.FixedTrial({"mlp_prob": mlp, "informed_target_mix_weight": weight,
                                    "pitting_composition_perturb_strength": perturb})
    params = search.sample_params(trial, composition_mode=MODE)
    command = search.training_command(search.parse_args([]), params, tmp_path, rules)
    assert float(command[command.index("--informed_prior_ratio") + 1]) == 1.
    mix_index = command.index("--informed_mix_probs")
    assert list(map(float, command[mix_index + 1:mix_index + 3])) == pytest.approx([mlp, 1. - mlp])
    assert float(command[command.index("--informed_target_mix_weight") + 1]) == weight
    assert float(command[command.index("--pitting_composition_perturb_strength") + 1]) == perturb
    assert command[command.index("--max_features") + 1] == "70"


def test_current_search_cannot_run_a_mixed_ratio_trial():
    with pytest.raises(ValueError, match="fixed 100% informed"):
        search.run_trial(search.parse_args([]), 0, replace(_selected_params(), informed_prior_ratio=.75))


def test_fixed_ratio_requires_provenance_and_rejects_conflicting_saved_values():
    values = {"mlp_prob": .5, "informed_target_mix_weight": .5, "pitting_composition_perturb_strength": .1}
    with pytest.raises(RuntimeError, match="missing parameters"):
        search.trial_params_from_mapping(values, composition_mode=MODE)
    with pytest.raises(RuntimeError, match="conflicts"):
        search.trial_params_from_mapping(
            {**values, "informed_prior_ratio": .5}, composition_mode=MODE, fixed_informed_prior_ratio=1.,
        )


@pytest.mark.parametrize("ratio", [.25, .50, .75, 1.])
def test_four_final_training_ratios_change_only_the_ratio(rules, tmp_path, ratio):
    selected = _selected_params()
    args = train_final.parse_args(["--storage", "journal:///unused.log", "--override-informed-prior-ratio", str(ratio)])
    args.pitting_composition_mode = MODE
    effective, provenance = train_final.apply_parameter_overrides(args, selected, composition_mode=MODE)
    assert asdict(effective) == {**asdict(selected), "informed_prior_ratio": ratio}
    assert selected.informed_prior_ratio == 1.
    assert provenance["trial_parameter_overrides"] == {"informed_prior_ratio": ratio}
    command = search.training_command(args, effective, tmp_path, rules)
    assert float(command[command.index("--informed_prior_ratio") + 1]) == ratio


def test_soccol_keeps_its_previous_search_space(monkeypatch):
    monkeypatch.setattr(search, "EPIT_FEATURE_PROFILE", SOCCOL_FEATURE_PROFILE)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()
    params = search.sample_params(trial, composition_mode=MODE)
    assert isinstance(trial.distributions["informed_prior_ratio"], optuna.distributions.CategoricalDistribution)
    assert isinstance(trial.distributions["mlp_prob"], optuna.distributions.CategoricalDistribution)
    assert trial.distributions["pitting_composition_perturb_strength"].high == .15
    assert params.informed_prior_ratio in (.25, .5, .75, 1.)
    assert search.trial_params_from_mapping(trial.params, composition_mode=MODE) == params


def test_old_search_cannot_be_resumed_as_the_fixed_ratio_search(rules):
    fingerprint = search.build_pipeline_fingerprint(search.parse_args([]), rules)
    previous = {**fingerprint, "fixed_prior": {**fingerprint["fixed_prior"]},
                "search_space": {**fingerprint["search_space"]}}
    previous["fixed_prior"].pop("informed_prior_ratio")
    previous["search_space"]["informed_prior_ratio"] = [.25, .5, .75, 1.]
    previous["search_space"]["mlp_prob"] = list(search.original.MLP_PROB_GRID)
    previous["search_space"]["pitting_composition_perturb_strength"] = [0., .15]
    study = optuna.create_study()
    search.bind_study_pipeline_fingerprint(study, previous)
    with pytest.raises(RuntimeError, match="different split, target-rule set, or"):
        search.bind_study_pipeline_fingerprint(study, fingerprint)
