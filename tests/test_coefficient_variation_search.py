from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pytest
import torch

import tabicl.prior.dataset as dataset_module
from scripts.epit_pipeline import diagnose_target_variation as diagnostic
from scripts.epit_pipeline import run_coefficient_variation as experiment
from scripts.epit_pipeline import resume_coefficient_variation as recovery
from tabicl.prior.dataset import SCMPrior
from tabicl.train.train_config import build_parser


@pytest.fixture
def args(tmp_path):
    return experiment.parse_args([
        "--work-root", str(tmp_path / "work"),
        "--checkpoint-root", str(tmp_path / "checkpoints"),
    ])


@pytest.fixture(scope="module")
def settings():
    return diagnostic.load_settings(
        diagnostic.DEFAULT_MANIFEST, diagnostic.search.LEGACY_TARGET_RULE_SUMMARY,
        diagnostic.search.DEFAULT_SPLIT_MANIFEST,
    )


def test_commands_change_only_declared_experiment_settings(args):
    loaded = experiment.load_experiment(args)
    plan = experiment.build_trial_plan(args, loaded, 0, .5, 43)
    original = loaded.template
    index = next(i for i, token in enumerate(original) if token.endswith("/train/run.py"))
    baseline = vars(build_parser().parse_args(original[index + 1:]))
    command = plan["train_command"]
    index = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
    current = vars(build_parser().parse_args(command[index + 1:]))
    allowed = {
        "max_steps", "scheduler_total_steps", "np_seed", "torch_seed", "python_seed",
        "device", "prior_n_jobs", "checkpoint_dir", "save_perm_every", "save_temp_every",
        "max_checkpoints", "wandb_log", "wandb_mode", "wandb_name",
        "pitting_coefficient_variation", "pitting_coefficient_variation_seed",
        "pitting_coefficient_upper_bounds",
    }
    assert {key for key in baseline if baseline[key] != current[key]} <= allowed
    assert current["max_steps"] == 2000
    assert current["scheduler_total_steps"] == 10000
    assert current["dataloader_num_workers"] == baseline["dataloader_num_workers"] == 4
    assert current["pitting_coefficient_variation"] == .5
    assert current["np_seed"] == current["torch_seed"] == current["python_seed"] == 43
    assert current["pitting_coefficient_upper_bounds"] == loaded.bounds
    assert current["checkpoint_path"] is None
    assert current["informed_target_mix_weight"] == baseline["informed_target_mix_weight"]
    assert current["save_perm_every"] == 1000
    assert [row["step"] for row in plan["evaluations"]] == [1000, 2000]
    for evaluation in plan["evaluations"]:
        assert "evaluate_optuna_folds.py" in " ".join(evaluation["command"])
        assert "evaluate_final.py" not in " ".join(evaluation["command"])


def test_dry_run_does_not_create_trials_or_checkpoints(args):
    args.dry_run = True
    path = experiment.run(args)
    plans = json.loads(path.read_text())
    assert len(plans) == 12
    assert {(p["coefficient_variation"], p["training_seed"]) for p in plans} == {
        (v, seed) for v in (0, .3, .5, .8) for seed in (42, 43, 44)
    }
    assert not (path.parent / "study_journal.log").exists()
    assert not args.checkpoint_root.exists()
    assert experiment.run(args) == path


def test_partial_grid_resume_and_paired_summary(args, monkeypatch):
    args.variations = [0.0, .5]
    args.seeds = [42, 43]
    args.n_trials = 2
    calls = []

    def fake_trial(plan, loaded):
        calls.append((plan["coefficient_variation"], plan["training_seed"]))
        score = .6 + .1 * plan["coefficient_variation"] + .01 * (plan["training_seed"] - 42)
        result = {**plan, "objective": score, "results": [
            {"step": 1000, "mean_test_spearman": .99},
            {"step": 2000, "mean_test_spearman": score},
        ]}
        experiment.search.write_json(Path(plan["trial_dir"]) / "trial_result.json", result)
        return result

    monkeypatch.setattr(experiment, "run_trial", fake_trial)
    output = experiment.run(args)
    assert len(calls) == 2
    assert not json.loads((output / "summary.json").read_text())["grid_complete"]
    args.n_trials = None
    experiment.run(args)
    assert len(calls) == len(set(calls)) == 4
    summary = json.loads((output / "summary.json").read_text())
    assert summary["grid_complete"]
    assert summary["best_variation_by_mean"] == .5
    assert summary["results"][1]["mean_paired_gain"] == pytest.approx(.05)
    assert summary["results"][1]["std_paired_gain"] == pytest.approx(0)
    assert summary["results"][1]["std_across_seeds"] > 0
    experiment.run(args)
    assert len(calls) == 4


def test_recovery_retries_failed_cell_before_unstarted_cells(args, monkeypatch):
    args.variations = [0.0, .5]
    args.seeds = [42, 43]
    args.n_trials = 1
    calls = []

    def fake_trial(plan, loaded):
        key = (plan["coefficient_variation"], plan["training_seed"])
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("simulated interrupted training")
        score = .6 + .1 * key[0] + .01 * (key[1] - 42)
        result = {**plan, "objective": score, "results": [
            {"step": 1000, "mean_test_spearman": .59},
            {"step": 2000, "mean_test_spearman": score},
        ]}
        experiment.search.write_json(
            Path(plan["trial_dir"]) / "trial_result.json", result
        )
        return result

    monkeypatch.setattr(experiment, "run_trial", fake_trial)
    with pytest.raises(RuntimeError, match="simulated interrupted training"):
        experiment.run(args)
    failed_cell = calls[0]

    args.n_trials = None
    output = recovery.run(args)
    assert calls[1] == failed_cell
    assert len(calls) == 5
    summary = json.loads((output / "summary.json").read_text())
    assert summary["grid_complete"]
    assert summary["trial_states"]["COMPLETE"] == 4
    assert summary["trial_states"]["FAIL"] == 1

    storage = experiment.search.original.search_utils.build_optuna_storage(
        f"journal://{output / 'study_journal.log'}"
    )
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    assert study.trials[1].params == study.trials[0].params
    assert study.trials[1].user_attrs["recovery_fixed_grid_cell"] is True


def test_changed_experiment_refused(args):
    args.dry_run = True
    path = experiment.run(args)
    original = path.read_bytes()
    args.variations = [0, .4]
    with pytest.raises(RuntimeError, match="configuration or source code changed"):
        experiment.run(args)
    assert path.read_bytes() == original


def test_experiment_lock(tmp_path):
    with experiment.experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another runner"):
            with experiment.experiment_lock(tmp_path):
                pass


@pytest.mark.parametrize("options", [
    ["--variations", ".3"], ["--variations", "0", "1"],
    ["--seeds", "42", "42"], ["--study-name", "../bad"],
    ["--study-name", experiment.search.DEFAULT_STUDY_NAME],
])
def test_invalid_search_settings(options):
    with pytest.raises(SystemExit):
        experiment.parse_args(options)


def generate_sequence(settings, monkeypatch, variation, worker_seed=None):
    fixed, bounds, _, _ = settings
    hp = copy.deepcopy(fixed)
    if variation is not None:
        hp.update({
            "pitting_coefficient_variation": variation,
            "pitting_coefficient_variation_seed": 42,
            "pitting_coefficient_upper_bounds": bounds,
        })
    worker = None if worker_seed is None else SimpleNamespace(seed=worker_seed)
    monkeypatch.setattr(dataset_module, "get_worker_info", lambda: worker)
    np.random.seed(12)
    torch.manual_seed(12)
    random.seed(12)
    prior = diagnostic.DiagnosticPrior(batch_size=1, fixed_hp=hp, sampled_hp={}, n_jobs=1)

    def scm(self, X, params, prior_cls):
        return torch.randn(len(X)) + torch.tensor(np.random.normal(size=len(X)), dtype=torch.float32) + random.random()

    monkeypatch.setattr(SCMPrior, "_generate_scm_target_from_physical_features", scm)
    results = []
    for _ in range(3):
        params = diagnostic.make_params(prior, "mlp_scm", 64)
        X, y, _ = prior.generate_dataset(params)
        results.append((X.clone(), y.clone(), prior.recorded_scm.clone(), copy.deepcopy(prior.last_pitting_target_rule)))
    return results, np.random.random(), torch.rand(1), random.random()


def test_default_and_explicit_zero_preserve_targets_and_rng(settings, monkeypatch):
    original = generate_sequence(settings, monkeypatch, None)
    zero = generate_sequence(settings, monkeypatch, 0)
    for a, b in zip(original[0], zero[0]):
        for i in range(3):
            assert torch.equal(a[i], b[i])
        assert a[3]["target_rule_coefficients"] == b[3]["target_rule_coefficients"]
        assert "coefficient_variation" not in b[3]
    assert original[1] == zero[1] and torch.equal(original[2], zero[2]) and original[3] == zero[3]


def test_variation_changes_only_coefficients_and_target(settings, monkeypatch):
    baseline = generate_sequence(settings, monkeypatch, 0)
    varied = generate_sequence(settings, monkeypatch, .5)
    repeated = generate_sequence(settings, monkeypatch, .5)
    for a, b, c in zip(baseline[0], varied[0], repeated[0]):
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
        assert not torch.equal(a[1], b[1])
        assert torch.equal(b[1], c[1])
        assert b[3]["coefficient_variation"]["reference"] == a[3]["target_rule_coefficients"]
        assert b[3]["target_rule_family"] == a[3]["target_rule_family"]
        np.testing.assert_array_equal(b[3]["process_offsets"], a[3]["process_offsets"])
    assert baseline[1] == varied[1] and torch.equal(baseline[2], varied[2]) and baseline[3] == varied[3]


def test_dataloader_workers_get_distinct_reproducible_coefficient_streams(settings, monkeypatch):
    worker_a = generate_sequence(settings, monkeypatch, .5, worker_seed=100)
    worker_b = generate_sequence(settings, monkeypatch, .5, worker_seed=101)
    worker_a_again = generate_sequence(settings, monkeypatch, .5, worker_seed=100)
    assert worker_a[0][0][3]["target_rule_coefficients"] != worker_b[0][0][3]["target_rule_coefficients"]
    assert worker_a[0][0][3]["target_rule_coefficients"] == worker_a_again[0][0][3]["target_rule_coefficients"]


def test_nonzero_variation_requires_calibrated_bounds(settings):
    hp = copy.deepcopy(settings[0])
    hp["pitting_coefficient_variation"] = .3
    with pytest.raises(ValueError, match="matching upper bounds"):
        SCMPrior(fixed_hp=hp, n_jobs=1)


def test_trainer_passes_opt_in_settings(args):
    from tabicl.train.run import Trainer

    loaded = experiment.load_experiment(args)
    command = experiment.build_trial_plan(args, loaded, 0, .5, 42)["train_command"]
    start = next(i for i, token in enumerate(command) if token.endswith("/train/run.py")) + 1
    config = build_parser().parse_args(command[start:])
    config.device = "cpu"
    config.dataloader_num_workers = 0
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.ddp_rank = 0
    trainer.master_process = False
    trainer.configure_prior()
    prior = trainer.dataloader.dataset.prior
    assert prior.coefficient_variation == .5
    assert prior.fixed_hp["pitting_coefficient_upper_bounds"] == loaded.bounds


@pytest.mark.parametrize("wrong_folds", [False, True])
def test_trial_evaluates_both_steps_and_rejects_wrong_folds(args, monkeypatch, wrong_folds):
    loaded = experiment.load_experiment(args)
    plan = experiment.build_trial_plan(args, loaded, 0, .3, 42)
    calls = []

    def fake_command(command, *, cwd, log_path):
        calls.append(command)
        if command == plan["train_command"]:
            checkpoint_dir = Path(plan["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True)
            for step in (1000, 2000):
                (checkpoint_dir / f"step-{step}.ckpt").write_bytes(f"checkpoint-{step}".encode())
            return
        evaluation = next(e for e in plan["evaluations"] if e["command"] == command)
        output = Path(evaluation["output_dir"])
        output.mkdir(parents=True)
        score = .99 if evaluation["step"] == 1000 else .65
        (output / "summary.csv").write_text(
            f"model,mean_test_spearman\n{evaluation['model_label']},{score}\n"
        )
        (output / "rows.csv").write_text("model,test_spearman\n")
        experiment.search.write_json(output / "summary.json", {
            "split_manifest_sha256": loaded.rules.split_manifest_sha256,
            "split_lock_sha256": loaded.rules.split_lock_sha256,
            "source_sha256": loaded.rules.source_sha256,
            "validation_folds": [1] if wrong_folds else [1, 2, 3, 4, 5],
            "development_rows_only": True,
            "source": {
                "local_ckpt_path": evaluation["checkpoint"],
                "local_ckpt_sha256": experiment.sha256_file(Path(evaluation["checkpoint"])),
            },
            "settings": {
                "tabicl_norm_methods": ["none"], "tabicl_feat_shuffle_method": "none",
                "pitting_magpie_features": False, "n_estimators": loaded.eval_args.eval_n_estimators,
            },
        })

    monkeypatch.setattr(experiment.search.original.search_utils, "run_command", fake_command)
    if wrong_folds:
        with pytest.raises(RuntimeError, match="five development folds"):
            experiment.run_trial(plan, loaded)
    else:
        result = experiment.run_trial(plan, loaded)
        assert result["objective"] == .65
        assert len(calls) == 3
        assert [r["step"] for r in result["results"]] == [1000, 2000]
        before = (Path(plan["trial_dir"]) / "trial_result.json").read_bytes()
        with pytest.raises(FileExistsError):
            experiment.run_trial(plan, loaded)
        assert (Path(plan["trial_dir"]) / "trial_result.json").read_bytes() == before
