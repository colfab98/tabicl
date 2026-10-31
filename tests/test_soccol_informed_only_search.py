"""Reviewed Soccol search policy and its separation from historical runs."""

from copy import deepcopy
from dataclasses import replace

import optuna
import pytest

from scripts.epit_pipeline import run_optuna as base
from scripts.soccol_pipeline import run_optuna as search
from tabicl.prior.soccol_reviewed_schema import (
    SOCCOL_REVIEWED_FEATURE_COLUMNS, SOCCOL_REVIEWED_FEATURE_PROFILE,
)
from tabicl.train.train_config import build_parser


@pytest.fixture
def reviewed():
    # Soccol wrappers configure the shared EPIT module; restore it for other tests.
    original = vars(base).copy()
    try:
        args = search.parse_args([])
        rules = base.load_target_rule_config(
            summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest,
        )
        yield args, rules
    finally:
        for name, value in original.items():
            if vars(base).get(name) is not value:
                setattr(base, name, value)


def test_reviewed_defaults_and_real_optuna_distributions(reviewed):
    args, rules = reviewed
    assert args.study_name == "soccol_pipeline_optuna_empirical_features_scm_target_v6"
    assert args.work_dir.parent.name == args.eval_output_root.parent.name == "optuna_v4"
    assert args.target_rule_summary.parent.name == "target_rules_v2"
    assert (args.n_trials, args.max_steps, args.scheduler_total_steps) == (50, 1000, 10000)
    assert args.n_startup_trials == 10
    assert args.eval_n_estimators == 8

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()
    params = base.sample_params(trial, composition_mode=args.pitting_composition_mode)
    expected = {"mlp_prob": (0., 1.), "informed_target_mix_weight": (0., 1.),
                "pitting_composition_perturb_strength": (0., .15)}
    assert set(trial.params) == set(expected)
    assert params.informed_prior_ratio == 1.
    for name, bounds in expected.items():
        distribution = trial.distributions[name]
        assert isinstance(distribution, optuna.distributions.FloatDistribution)
        assert (distribution.low, distribution.high) == bounds
        assert distribution.step is None and not distribution.log

    fingerprint = base.build_pipeline_fingerprint(args, rules)
    fixed = fingerprint["fixed_prior"]
    assert "informed_prior_ratio" not in fingerprint["search_space"]
    assert fingerprint["search_space"]["pitting_composition_perturb_strength"] == [0., .15]
    assert fixed["informed_prior_ratio"] == 1.
    assert fixed["feature_columns"] == list(SOCCOL_REVIEWED_FEATURE_COLUMNS)
    assert fixed["feature_count"] == 58
    assert fixed["development_selection"]["source_rows"] == 2950
    assert fixed["pitting_coefficient_variation"] == .8
    assert fixed["pitting_coefficient_variation_seed"] == 42
    assert fixed["composition_sampling_policy"] == "uniform_distinct_compositions_v1"
    assert fixed["distinct_composition_templates"] == 349
    assert fingerprint["evaluation"]["validation_folds"] == [1, 2, 3, 4, 5]
    reconstructed = base.trial_params_from_mapping(
        trial.params, composition_mode=args.pitting_composition_mode,
        fixed_informed_prior_ratio=fixed["informed_prior_ratio"],
    )
    assert reconstructed == params
    with pytest.raises(ValueError, match="fixed 100% informed"):
        base.run_trial(args, 0, replace(params, informed_prior_ratio=.75))


@pytest.mark.parametrize("mlp,weight,perturb", [(0., 0., 0.), (.37, .42, .12), (1., 1., .15)])
def test_new_ranges_reach_soccol_training_and_matching_evaluation(reviewed, tmp_path, mlp, weight, perturb):
    args, rules = reviewed
    trial = optuna.trial.FixedTrial({"mlp_prob": mlp, "informed_target_mix_weight": weight,
                                    "pitting_composition_perturb_strength": perturb})
    params = base.sample_params(trial, composition_mode=args.pitting_composition_mode)
    command = base.training_command(args, params, tmp_path, rules)
    entry = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
    config = build_parser().parse_args(command[entry + 1:])
    assert config.informed_prior_ratio == 1.
    assert config.informed_mix_probs == pytest.approx([mlp, 1. - mlp])
    assert config.informed_target_mix_weight == weight
    assert config.pitting_composition_perturb_strength == perturb
    assert config.min_features == config.max_features == 58
    assert tuple(config.informed_normal_block_allocation) == (24, 34, 0, 0, 0, 0, 0, 0, 0)
    assert config.pitting_feature_profile == SOCCOL_REVIEWED_FEATURE_PROFILE
    assert config.col_missing_value_embedding
    assert config.pitting_coefficient_variation == .8
    assert config.pitting_coefficient_variation_seed == 42
    assert not base.uses_expanded_schema(args)
    evaluation = base.eval_command(args, params, tmp_path / "step-1000.ckpt", "trial", tmp_path / "eval")
    assert "--soccol-reviewed-data" in evaluation
    assert "--epit-expanded-features" not in evaluation
    assert "--epit-final-test" not in evaluation
    start = evaluation.index("--validation-folds") + 1
    assert evaluation[start:start + 5] == ["1", "2", "3", "4", "5"]


def test_previous_policy_cannot_resume_in_new_study(reviewed):
    args, rules = reviewed
    fingerprint = base.build_pipeline_fingerprint(args, rules)
    previous = deepcopy(fingerprint)
    previous["fixed_prior"].pop("informed_prior_ratio")
    previous["search_space"]["informed_prior_ratio"] = [.25, .5, .75, 1.]
    previous["search_space"]["mlp_prob"] = list(base.original.MLP_PROB_GRID)
    previous["search_space"]["pitting_composition_perturb_strength"] = [0., .15]
    study = optuna.create_study()
    base.bind_study_pipeline_fingerprint(study, previous)
    with pytest.raises(RuntimeError, match="different split, target-rule set, or"):
        base.bind_study_pipeline_fingerprint(study, fingerprint)


def test_observation_weighted_profile_cannot_resume_as_uniform_templates(reviewed):
    args, rules = reviewed
    fingerprint = base.build_pipeline_fingerprint(args, rules)
    previous = deepcopy(fingerprint)
    previous["fixed_prior"]["feature_profile"] = "soccol_pitting_features_v2"
    previous["fixed_prior"].pop("composition_sampling_policy")
    previous["fixed_prior"].pop("distinct_composition_templates")
    study = optuna.create_study()
    base.bind_study_pipeline_fingerprint(study, previous)
    with pytest.raises(RuntimeError, match="different split, target-rule set, or"):
        base.bind_study_pipeline_fingerprint(study, fingerprint)


def test_previous_broad_composition_range_cannot_resume_as_reduced_range(reviewed):
    args, rules = reviewed
    fingerprint = base.build_pipeline_fingerprint(args, rules)
    previous = deepcopy(fingerprint)
    previous["search_space"]["pitting_composition_perturb_strength"] = [0., .30]
    study = optuna.create_study()
    base.bind_study_pipeline_fingerprint(study, previous)
    with pytest.raises(RuntimeError, match="different split, target-rule set, or"):
        base.bind_study_pipeline_fingerprint(study, fingerprint)


def test_legacy_cli_retains_its_original_policy_and_reviewed_flag_is_explicit(reviewed):
    expected, _ = reviewed
    explicit = search.parse_args(["--soccol-reviewed-data"])
    assert explicit.study_name == expected.study_name
    legacy = search.parse_args(["--legacy-data"])
    assert legacy.study_name.endswith("_v1")
    assert legacy.target_rule_summary.parent.name == "target_rules_v1"
    assert base.model_feature_count(legacy) == 37
    study = optuna.create_study()
    trial = study.ask()
    base.sample_params(trial, composition_mode=legacy.pitting_composition_mode)
    assert isinstance(trial.distributions["informed_prior_ratio"], optuna.distributions.CategoricalDistribution)
    assert isinstance(trial.distributions["mlp_prob"], optuna.distributions.CategoricalDistribution)
    assert trial.distributions["pitting_composition_perturb_strength"].high == .15
    with pytest.raises(SystemExit):
        search.parse_args(["--legacy-data", "--soccol-reviewed-data"])
