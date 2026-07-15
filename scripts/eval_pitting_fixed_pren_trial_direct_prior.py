#!/usr/bin/env python
"""Direct-prior scores for fixed-PREN Optuna pitting trials.

This script scores completed trials from ``optuna_pitting_fixed_pren_prior_search.py``.
For each trial eta it samples the same fixed-schema hybrid prior used during
training and compares direct-prior diagnostics with the recorded transformer
Spearman score.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_epit_direct_prior import (  # noqa: E402
    DIRECT_EPIT_BLOCK_ALLOCATION,
    EXPECTED_EPIT_FEATURE_COUNT,
    SCHEMA_RETRY_SEED_STRIDE,
    SyntheticEpitSample,
    build_fixed_hp_for_eta,
    fit_and_score_theta,
    load_epit_task,
    preprocess_real_epit,
    score_target_rule_theta,
    summarize_eta,
    summarize_target_rule_eta,
)
from tabicl.prior.dataset import PriorDataset, SCMPrior  # noqa: E402
from tabicl.prior.prior_config import DEFAULT_SAMPLED_HP  # noqa: E402


DEFAULT_STUDY_RUN_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "pitting_fixed_pren_prior_search"
    / "pitting_fixed_pren_prior_search"
    / "tabicl_pitting_fixed_pren_prior_pitting_fixed_pren_prior_search_10588"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "pitting_fixed_pren_direct_prior_scores"
DEFAULT_N_SYNTH = 256
DEFAULT_SCHEMA_ATTEMPTS = 100
DEFAULT_SYNTHETIC_SEED = 200_000
DEFAULT_CHUNK_LOG_INTERVAL = 30.0


@dataclass(frozen=True)
class TrialSpec:
    trial_number: int
    trial_dir: Path
    result_path: Path
    result: dict[str, Any]
    params: dict[str, Any]

    @property
    def trial_id(self) -> str:
        return f"trial_{self.trial_number:04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-run-dir", type=Path, default=DEFAULT_STUDY_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-synth", type=int, default=DEFAULT_N_SYNTH)
    parser.add_argument("--synthetic-seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--synthetic-seq-len", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--trial-number", type=int, nargs="*", default=None)
    parser.add_argument("--max-schema-attempts", type=int, default=DEFAULT_SCHEMA_ATTEMPTS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--write-theta-scores", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=DEFAULT_CHUNK_LOG_INTERVAL)
    return parser.parse_args()


def default_output_dir(study_run_dir: Path) -> Path:
    return DEFAULT_OUTPUT_ROOT / study_run_dir.name


def load_trial_specs(study_run_dir: Path, trial_numbers: set[int] | None = None) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for result_path in sorted(study_run_dir.glob("trial_*/trial_result.json")):
        trial_dir = result_path.parent
        try:
            trial_number = int(trial_dir.name.split("_")[-1])
        except ValueError:
            continue
        if trial_numbers is not None and trial_number not in trial_numbers:
            continue
        result = json.loads(result_path.read_text())
        if str(result.get("status", "")) != "completed":
            continue
        params = dict(result.get("params") or {})
        if not params:
            config_path = trial_dir / "trial_config.json"
            if config_path.is_file():
                params = dict(json.loads(config_path.read_text()).get("params") or {})
        if not params:
            raise RuntimeError(f"Could not find params for {result_path}")
        specs.append(
            TrialSpec(
                trial_number=trial_number,
                trial_dir=trial_dir,
                result_path=result_path,
                result=result,
                params=params,
            )
        )
    return specs


def synthetic_train_size_bounds(seq_len: int) -> tuple[int, int]:
    train_size = max(1, min(int(seq_len) - 1, int(int(seq_len) * 0.5)))
    return train_size, train_size + 1


def schema_retry_seed(seed: int, attempt: int) -> int:
    return int((int(seed) + int(attempt) * SCHEMA_RETRY_SEED_STRIDE) % (2**32 - 1))


def eta_params_from_trial(params: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    mlp_prob = float(params.get("mlp_prob", params.get("informed_mlp_prob", 0.7)))
    informed_ratio = float(params.get("informed_prior_ratio", 1.0))
    eta_params = {
        "informed_mlp_prob": mlp_prob,
        "informed_feature_block_strength": float(params["informed_feature_block_strength"]),
        "informed_interaction_strength": float(params["informed_interaction_strength"]),
        "informed_physical_marginal_prob": float(params["informed_physical_marginal_prob"]),
        "pitting_material_dirichlet_prob": float(params["pitting_material_dirichlet_prob"]),
        "pitting_material_dirichlet_concentration": float(params["pitting_material_dirichlet_concentration"]),
        "pitting_material_dirichlet_active_prob": float(params["pitting_material_dirichlet_active_prob"]),
        "epit_material_coef": float(params["epit_material_coef"]),
        "epit_environment_coef": float(params["epit_environment_coef"]),
        "epit_interaction_coef": float(params["epit_interaction_coef"]),
    }
    return informed_ratio, eta_params


def sample_hybrid_trial_dataset(
    *,
    category_count: int,
    synthetic_seed: int,
    seq_len: int,
    params: dict[str, Any],
    max_schema_attempts: int,
) -> SyntheticEpitSample:
    informed_ratio, eta_params = eta_params_from_trial(params)
    min_train_size, max_train_size = synthetic_train_size_bounds(seq_len)
    last_error = ""

    for attempt in range(int(max_schema_attempts)):
        sampling_seed = schema_retry_seed(int(synthetic_seed), attempt)
        np.random.seed(sampling_seed)
        random.seed(sampling_seed)
        torch.manual_seed(sampling_seed)

        dataset = PriorDataset(
            batch_size=1,
            batch_size_per_gp=1,
            min_features=EXPECTED_EPIT_FEATURE_COUNT,
            max_features=EXPECTED_EPIT_FEATURE_COUNT,
            max_classes=0,
            min_seq_len=None,
            max_seq_len=int(seq_len),
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            prior_type="hybrid_scm",
            informed_prior_ratio=informed_ratio,
            scm_fixed_hp=build_fixed_hp_for_eta(category_count, eta_params=eta_params),
            scm_sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=1,
            device="cpu",
        )
        X, y, d, seq_lens, train_sizes = dataset.get_batch()
        X_np = X[0].detach().cpu().numpy().astype(float)
        y_np = y[0].detach().cpu().numpy().astype(float)
        d_value = int(d[0].item())

        if X_np.shape != (int(seq_len), EXPECTED_EPIT_FEATURE_COUNT):
            last_error = f"shape {X_np.shape}"
            continue
        if d_value != EXPECTED_EPIT_FEATURE_COUNT:
            last_error = f"d={d_value}"
            continue
        if not np.isfinite(X_np).all() or not np.isfinite(y_np).all():
            last_error = "non-finite values"
            continue
        if float(np.std(y_np, ddof=0)) <= 0.0:
            last_error = "zero target variance"
            continue

        target_rule = getattr(dataset.prior, "last_pitting_target_rule", None)
        if not isinstance(target_rule, dict):
            target_rule = {}
        process_unique = np.unique(X_np[:, -1])
        return SyntheticEpitSample(
            X=X_np,
            y=y_np,
            d=d_value,
            seq_len=int(seq_lens[0].item()),
            train_size=int(train_sizes[0].item()),
            synthetic_seed=int(synthetic_seed),
            sampling_seed=int(sampling_seed),
            schema_attempts=int(attempt) + 1,
            process_unique_count=int(process_unique.size),
            process_unique_values=[float(value) for value in process_unique.tolist()],
            target_rule=target_rule,
        )

    raise RuntimeError(
        f"Could not sample accepted hybrid fixed-PREN dataset for seed {synthetic_seed} "
        f"after {max_schema_attempts} attempts. Last error: {last_error}"
    )


def is_informed_sample(sample: SyntheticEpitSample) -> bool:
    return sample.target_rule.get("rule_type") == "fixed_epit_target_rule_v2_pren_anchor"


def fake_pren_rule_for_generic_sample(sample: SyntheticEpitSample, params: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(sample.sampling_seed + 17_171)
    material_width = DIRECT_EPIT_BLOCK_ALLOCATION[0]
    anchor = SCMPrior._fixed_epit_material_anchor(material_width)
    material_weights = anchor + 0.15 * rng.standard_normal(material_width)
    for idx in (1, 2, 3, 4):
        if idx < material_width and anchor[idx] > 0.0:
            material_weights[idx] = max(material_weights[idx], 0.05 * anchor[idx])
    norm = float(np.linalg.norm(material_weights))
    if not np.isfinite(norm) or norm <= 1e-12:
        material_weights = anchor
        norm = float(np.linalg.norm(material_weights))
    material_weights = material_weights / max(norm, 1e-12)
    category_count = int(params.get("pitting_process_category_count", 52))
    return {
        "rule_type": "fake_positional_pren_control",
        "material_anchor_type": "fake_generic_positional_pren_like_cr_mo_w_weak_ni_v1",
        "environment_rule_type": "fake_generic_positional_chloride_dominant_weak_temp_ph_v1",
        "material_cols": list(range(0, 17)),
        "temperature_col": 17,
        "chloride_col": 18,
        "ph_col": 19,
        "process_col": 20,
        "material_anchor_weights": anchor,
        "material_weights": material_weights.astype(float),
        "environment_weights": np.asarray(
            [
                rng.uniform(0.03, 0.12),
                rng.uniform(0.75, 1.10),
                rng.uniform(0.05, 0.18),
            ],
            dtype=float,
        ),
        "ph_neutral": float(rng.uniform(6.3, 8.2)),
        "process_offsets": rng.standard_normal(category_count).astype(float),
        "process_category_count": category_count,
        "material_coef": float(params["epit_material_coef"]),
        "environment_coef": float(params["epit_environment_coef"]),
        "interaction_coef": float(params["epit_interaction_coef"]),
        "process_coef": float(rng.uniform(0.03, 0.10)),
        "interaction_strength": float(params["informed_interaction_strength"]),
        "synthetic_environment_mode": "raw",
    }


def nan_metric_dict(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_n": 0,
        f"{prefix}_spearman": math.nan,
        f"{prefix}_mae": math.nan,
        f"{prefix}_rmse": math.nan,
        f"{prefix}_median_theta_spearman": math.nan,
        f"{prefix}_max_theta_spearman": math.nan,
    }


def summary_metric_dict(prefix: str, summary: Any) -> dict[str, Any]:
    return {
        f"{prefix}_n": int(summary.n_synth),
        f"{prefix}_spearman": float(summary.ensemble_spearman),
        f"{prefix}_mae": float(summary.standardized_mae),
        f"{prefix}_rmse": float(summary.standardized_rmse),
        f"{prefix}_median_theta_spearman": float(summary.median_theta_spearman),
        f"{prefix}_max_theta_spearman": float(summary.max_theta_spearman),
    }


def score_trial(spec: TrialSpec, args_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    task = load_epit_task(random_state=int(args_dict["random_state"]))
    processed = preprocess_real_epit(task)
    seq_len = processed.n_rows if int(args_dict["synthetic_seq_len"]) == 0 else int(args_dict["synthetic_seq_len"])
    n_synth = int(args_dict["n_synth"])
    synthetic_seed_start = int(args_dict["synthetic_seed"]) + spec.trial_number * 1_000_000

    samples: list[SyntheticEpitSample] = []
    all_scores = []
    informed_scores = []
    generic_scores = []
    pren_scores = []
    fake_pren_scores = []
    theta_rows = []
    start = time.time()

    for offset in range(n_synth):
        sample = sample_hybrid_trial_dataset(
            category_count=processed.category_count,
            synthetic_seed=synthetic_seed_start + offset,
            seq_len=seq_len,
            params=spec.params,
            max_schema_attempts=int(args_dict["max_schema_attempts"]),
        )
        samples.append(sample)
        ridge_score = fit_and_score_theta(processed, sample)
        informed = is_informed_sample(sample)
        all_scores.append(ridge_score)
        if informed:
            informed_scores.append(ridge_score)
            pren_scores.append(score_target_rule_theta(processed, sample))
        else:
            generic_scores.append(ridge_score)
            fake_sample = SyntheticEpitSample(
                X=sample.X,
                y=sample.y,
                d=sample.d,
                seq_len=sample.seq_len,
                train_size=sample.train_size,
                synthetic_seed=sample.synthetic_seed,
                sampling_seed=sample.sampling_seed,
                schema_attempts=sample.schema_attempts,
                process_unique_count=sample.process_unique_count,
                process_unique_values=sample.process_unique_values,
                target_rule=fake_pren_rule_for_generic_sample(sample, spec.params),
            )
            fake_pren_scores.append(score_target_rule_theta(processed, fake_sample))

        if bool(args_dict["write_theta_scores"]):
            theta_rows.append(
                {
                    "trial_number": spec.trial_number,
                    "trial_id": spec.trial_id,
                    "synthetic_seed": sample.synthetic_seed,
                    "sampling_seed": sample.sampling_seed,
                    "schema_attempts": sample.schema_attempts,
                    "sample_type": "informed" if informed else "generic",
                    "ridge_spearman": ridge_score.spearman,
                    "ridge_mae": ridge_score.standardized_mae,
                    "ridge_rmse": ridge_score.standardized_rmse,
                }
            )

    row: dict[str, Any] = {
        "trial_number": spec.trial_number,
        "trial_id": spec.trial_id,
        "trial_result_path": str(spec.result_path),
        "mean_test_spearman": float(spec.result.get("mean_test_spearman", math.nan)),
        "median_test_spearman": float(spec.result.get("median_test_spearman", math.nan)),
        "std_test_spearman": float(spec.result.get("std_test_spearman", math.nan)),
        "mean_test_mae": float(spec.result.get("mean_test_mae", math.nan)),
        "mean_test_rmse": float(spec.result.get("mean_test_rmse", math.nan)),
        "mean_test_r2": float(spec.result.get("mean_test_r2", math.nan)),
        "n_synth_requested": n_synth,
        "n_informed_sampled": len(informed_scores),
        "n_generic_sampled": len(generic_scores),
        "informed_sample_fraction": len(informed_scores) / float(n_synth),
        "elapsed_seconds": time.time() - start,
        **spec.params,
    }

    row.update(summary_metric_dict("all_score", summarize_eta(processed, all_scores, eta_id=spec.trial_id)))
    row.update(
        summary_metric_dict("informed_score", summarize_eta(processed, informed_scores, eta_id=spec.trial_id))
        if informed_scores
        else nan_metric_dict("informed_score")
    )
    row.update(
        summary_metric_dict("generic_score", summarize_eta(processed, generic_scores, eta_id=spec.trial_id))
        if generic_scores
        else nan_metric_dict("generic_score")
    )
    row.update(
        summary_metric_dict("pren_score", summarize_target_rule_eta(processed, pren_scores, eta_id=spec.trial_id))
        if pren_scores
        else nan_metric_dict("pren_score")
    )
    row.update(
        summary_metric_dict(
            "fake_pren_score",
            summarize_target_rule_eta(processed, fake_pren_scores, eta_id=spec.trial_id),
        )
        if fake_pren_scores
        else nan_metric_dict("fake_pren_score")
    )

    trial_output_dir = Path(args_dict["output_dir"]) / "trials" / spec.trial_id
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    (trial_output_dir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    if bool(args_dict["write_theta_scores"]):
        pd.DataFrame(theta_rows).to_csv(trial_output_dir / "theta_scores.csv", index=False)
    return row


def correlation_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    target = summary["mean_test_spearman"].to_numpy(dtype=float)
    columns = [
        "all_score_spearman",
        "informed_score_spearman",
        "generic_score_spearman",
        "pren_score_spearman",
        "fake_pren_score_spearman",
        "all_score_mae",
        "all_score_rmse",
    ]
    rows = []
    for column in columns:
        if column not in summary.columns:
            continue
        values = summary[column].to_numpy(dtype=float)
        mask = np.isfinite(target) & np.isfinite(values)
        if int(mask.sum()) < 3:
            rows.append({"metric": column, "n": int(mask.sum()), "spearman": math.nan, "kendall_tau": math.nan})
            continue
        rows.append(
            {
                "metric": column,
                "n": int(mask.sum()),
                "spearman": float(spearmanr(target[mask], values[mask]).correlation),
                "kendall_tau": float(kendalltau(target[mask], values[mask]).correlation),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.n_synth <= 0:
        raise ValueError("--n-synth must be positive.")
    if args.max_schema_attempts <= 0:
        raise ValueError("--max-schema-attempts must be positive.")

    output_dir = (args.output_dir or default_output_dir(args.study_run_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.trial_number) if args.trial_number else None
    specs = load_trial_specs(args.study_run_dir.expanduser().resolve(), selected)
    if args.max_trials > 0:
        specs = specs[: int(args.max_trials)]
    if not specs:
        raise SystemExit(f"No completed trial_result.json files found under {args.study_run_dir}")

    args_dict = {
        "output_dir": str(output_dir),
        "n_synth": int(args.n_synth),
        "synthetic_seed": int(args.synthetic_seed),
        "synthetic_seq_len": int(args.synthetic_seq_len),
        "random_state": int(args.random_state),
        "max_schema_attempts": int(args.max_schema_attempts),
        "write_theta_scores": bool(args.write_theta_scores),
    }
    metadata = {
        "study_run_dir": str(args.study_run_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "n_trials": len(specs),
        **args_dict,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    pending: list[TrialSpec] = []
    resume = not bool(args.no_resume)
    for spec in specs:
        summary_path = output_dir / "trials" / spec.trial_id / "summary.json"
        if resume and summary_path.is_file():
            rows.append(json.loads(summary_path.read_text()))
        else:
            pending.append(spec)

    print(
        f"Scoring fixed-PREN trial direct priors: total={len(specs)}, "
        f"pending={len(pending)}, output={output_dir}"
    )
    last_progress = time.time()
    if pending:
        if int(args.n_workers) <= 1:
            for idx, spec in enumerate(pending, start=1):
                row = score_trial(spec, args_dict)
                rows.append(row)
                print(
                    f"{idx}/{len(pending)} {spec.trial_id}: "
                    f"test={row['mean_test_spearman']:.4f}, all={row['all_score_spearman']:.4f}, "
                    f"informed_n={row['n_informed_sampled']}, generic_n={row['n_generic_sampled']}"
                )
        else:
            with ProcessPoolExecutor(max_workers=int(args.n_workers)) as executor:
                futures = {executor.submit(score_trial, spec, args_dict): spec for spec in pending}
                completed = 0
                for future in as_completed(futures):
                    spec = futures[future]
                    row = future.result()
                    rows.append(row)
                    completed += 1
                    now = time.time()
                    if now - last_progress >= float(args.progress_interval) or completed == len(pending):
                        print(
                            f"Progress {completed}/{len(pending)} latest={spec.trial_id}: "
                            f"test={row['mean_test_spearman']:.4f}, all={row['all_score_spearman']:.4f}"
                        )
                        last_progress = now

    summary = pd.DataFrame(rows).sort_values("trial_number").reset_index(drop=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    corr = pd.DataFrame(correlation_rows(summary))
    corr.to_csv(output_dir / "correlations.csv", index=False)
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'correlations.csv'}")
    if not corr.empty:
        print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
