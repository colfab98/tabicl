from pathlib import Path

import scripts.optuna_pitting_magpie_prior_search as search


class RecordingTrial:
    def __init__(
        self,
        *,
        use_magpie: bool,
        dirichlet_prob: float,
    ):
        self.use_magpie = use_magpie
        self.dirichlet_prob = dirichlet_prob
        self.suggested_names: list[str] = []

    def suggest_categorical(self, name, choices):
        self.suggested_names.append(name)
        if name == "use_magpie":
            return self.use_magpie
        if name == "pitting_material_dirichlet_prob":
            return self.dirichlet_prob
        return choices[0]

    def suggest_float(self, name, low, high):
        self.suggested_names.append(name)
        return (low + high) / 2.0


def _value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_search_defaults_match_proxy_training_and_historical_evaluation():
    args = search.parse_args([])

    assert args.study_name == "pitting_magpie_full_v1"
    assert args.n_trials == 50
    assert args.n_startup_trials == 10
    assert args.max_steps == 1000
    assert args.scheduler_total_steps == 10000
    assert args.split_seeds == [1001, 1002, 1003, 1004, 1005]


def test_softmax_branch_does_not_suggest_inactive_dirichlet_parameters():
    trial = RecordingTrial(use_magpie=False, dirichlet_prob=0.0)
    params = search.sample_params(trial)

    assert params.pitting_material_dirichlet_prob == 0.0
    assert params.pitting_material_dirichlet_concentration is None
    assert params.pitting_material_dirichlet_active_prob is None
    assert "pitting_material_dirichlet_prob" in trial.suggested_names
    assert "pitting_material_dirichlet_concentration" not in trial.suggested_names
    assert "pitting_material_dirichlet_active_prob" not in trial.suggested_names


def test_dirichlet_branch_suggests_all_conditional_parameters():
    trial = RecordingTrial(use_magpie=True, dirichlet_prob=0.5)
    params = search.sample_params(trial)

    assert params.pitting_material_dirichlet_prob in search.DIRICHLET_PROB_GRID
    assert params.pitting_material_dirichlet_concentration in search.DIRICHLET_CONCENTRATION_GRID
    assert params.pitting_material_dirichlet_active_prob in search.DIRICHLET_ACTIVE_PROB_GRID
    assert "pitting_material_dirichlet_prob" in trial.suggested_names
    assert "pitting_material_dirichlet_concentration" in trial.suggested_names
    assert "pitting_material_dirichlet_active_prob" in trial.suggested_names


def test_physical_marginal_probability_is_not_an_optuna_parameter():
    trial = RecordingTrial(use_magpie=False, dirichlet_prob=0.0)

    search.sample_params(trial)

    assert "informed_physical_marginal_prob" not in trial.suggested_names


def test_training_and_eval_commands_switch_only_the_optional_feature_branch(tmp_path: Path):
    args = search.parse_args(["--device", "cpu", "--nproc-per-node", "1"])
    no_magpie = search.sample_params(RecordingTrial(use_magpie=False, dirichlet_prob=0.0))
    with_magpie = search.sample_params(RecordingTrial(use_magpie=True, dirichlet_prob=1.0))
    checkpoint_dir = tmp_path / "checkpoint"

    base_command = search.training_command(args, no_magpie, checkpoint_dir)
    magpie_command = search.training_command(args, with_magpie, checkpoint_dir)
    base_eval = search.eval_command(args, no_magpie, checkpoint_dir / "step.ckpt", "base", tmp_path / "base")
    magpie_eval = search.eval_command(
        args,
        with_magpie,
        checkpoint_dir / "step.ckpt",
        "magpie",
        tmp_path / "magpie",
    )

    assert _value_after(base_command, "--min_features") == "21"
    assert _value_after(base_command, "--max_features") == "21"
    assert _value_after(base_command, "--pitting_magpie_features") == "False"
    assert "--pitting_material_dirichlet_concentration" not in base_command
    assert "--pitting_material_dirichlet_active_prob" not in base_command
    assert "--pitting-magpie-features" not in base_eval

    assert _value_after(magpie_command, "--min_features") == "31"
    assert _value_after(magpie_command, "--max_features") == "31"
    assert _value_after(magpie_command, "--pitting_magpie_features") == "True"
    assert "--pitting_material_dirichlet_concentration" in magpie_command
    assert "--pitting_material_dirichlet_active_prob" in magpie_command
    assert "--pitting-magpie-features" in magpie_eval

    for command in (base_command, magpie_command):
        assert _value_after(command, "--prior_type") == "hybrid_scm"
        assert _value_after(command, "--informed_target_family") == "pitting_potential"
        assert _value_after(command, "--informed_physical_marginal_prob") == "1.0"
        assert _value_after(command, "--pitting_composition_mode") == "legacy"
        assert "--informed_interaction_strength" not in command
