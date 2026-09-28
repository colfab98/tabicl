#!/usr/bin/env python
"""Paired synthetic-target diagnostics, without training or changing v7 artifacts.

Run with: python -m scripts.epit_pipeline.diagnose_target_variation --help
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import platform
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import rankdata

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline.artifact_hashes import sha256_file
from tabicl.prior.dataset import SCMPrior
from tabicl.prior.mlp_scm import MLPSCM
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.prior.target_variation import sample_coefficients
from tabicl.prior.tree_scm import TreeSCM
from tabicl.train.train_config import build_parser

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = search.PIPELINE_ROOT / "target_variation"
DEFAULT_MANIFEST = (
    search.PIPELINE_ROOT / "final_v1" / search.DEFAULT_STUDY_NAME / "final_model_manifest.json"
)
SCM_TYPES = {"mlp_scm": MLPSCM, "tree_scm": TreeSCM}


class DiagnosticPrior(SCMPrior):
    """Record original components and replay the unchanged production mixer."""

    replay_rule: dict | None = None
    recorded_features: torch.Tensor
    recorded_scm: torch.Tensor

    def _generate_scm_target_from_physical_features(self, X, params, prior_cls):
        if self.replay_rule is not None:
            return self.recorded_scm.clone()
        target = super()._generate_scm_target_from_physical_features(X, params, prior_cls)
        self.recorded_features = X.detach().clone()
        self.recorded_scm = target.detach().clone()
        return target

    def _sample_fixed_epit_target_rule(self, *args, **kwargs):
        if self.replay_rule is not None:
            return copy.deepcopy(self.replay_rule)
        return super()._sample_fixed_epit_target_rule(*args, **kwargs)

    def replay(self, rule, params):
        self.replay_rule = copy.deepcopy(rule)
        try:
            target = self._generate_scm_epit_target_from_physical_features(
                self.recorded_features, params, SCM_TYPES[params["prior_type"]]
            )
            return target.clone(), self.last_pitting_target_drive.clone()
        finally:
            self.replay_rule = None

    def rule_terms(self, rule):
        # Unit coefficient vectors expose signed terms via the real evaluator,
        # without a second implementation of any physical formula.
        terms = {}
        for name in rule["target_rule_coefficients"]:
            probe = copy.deepcopy(rule)
            probe["target_rule_coefficients"] = {
                key: float(key == name) for key in rule["target_rule_coefficients"]
            }
            _, drive = self._evaluate_fixed_epit_target_rule_tensor(self.recorded_features, probe)
            terms[name] = drive.clone()
        return terms


def correlation(a, b, *, ranks=False):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    if np.array_equal(a, b):
        return 1.0
    if ranks:
        a, b = rankdata(a), rankdata(b)
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))


def compare_targets(original, candidate):
    a, b = np.asarray(original), np.asarray(candidate)
    delta = b.astype(float) - a.astype(float)
    return {
        "pearson": correlation(a, b),
        "spearman": correlation(a, b, ranks=True),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "max_abs_change": float(np.max(np.abs(delta))),
    }


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def variation_rng(seed, table_id, variation, draw_id):
    # Stable identities make results independent of variation order and draw count.
    token = f"{seed}:{table_id}:{float(variation).hex()}:{draw_id}".encode("ascii")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:16], "big"))


def load_settings(manifest_path, summary_path, split_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rules = search.load_target_rule_config(summary_path=summary_path, split_manifest_path=split_path)
    if manifest["target_rules"]["summary_sha256"] != rules.summary_sha256:
        raise ValueError("Rule calibration does not match the saved model manifest.")
    if manifest["target_rules"]["artifact_sha256s"] != rules.artifact_sha256s:
        raise ValueError("Rule artifacts do not match the saved model manifest.")
    if manifest["split"]["manifest_sha256"] != rules.split_manifest_sha256:
        raise ValueError("Split does not match the saved model manifest.")
    fixed_prior = manifest["study"]["pipeline_fingerprint"]["fixed_prior"]
    for key, value in search.empirical_feature_profile_identity().items():
        if fixed_prior.get(key) != value:
            raise ValueError(f"Current empirical feature profile differs: {key}")
    command = manifest["training"]["train_command"]
    start = next(i for i, arg in enumerate(command) if arg.endswith("/train/run.py")) + 1
    config = build_parser().parse_args(command[start:])
    fixed = copy.deepcopy(DEFAULT_FIXED_HP)
    for key in fixed:
        value = getattr(config, key, None)
        if value is not None:
            fixed[key] = value
    if fixed["pitting_composition_mode"] != "empirical_features_scm_target":
        raise ValueError("This diagnostic requires empirical_features_scm_target.")
    if not fixed["pitting_fixed_epit_schema"] or fixed["pitting_magpie_features"]:
        raise ValueError("This diagnostic requires the fixed 21-feature schema without Magpie.")
    bounds = {}
    for family, path in rules.artifacts.items():
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        bounds[family] = artifact["coefficient_upper_bounds"] or {
            name: 1.0 for name in artifact["term_names"]
        }
    return fixed, bounds, config, rules


def make_params(prior, kind, seq_len):
    sampled = {key: value() if callable(value) else value for key, value in prior.hp_sampling().items()}
    return {
        **prior.fixed_hp, **sampled, "seq_len": seq_len, "train_size": seq_len // 2,
        "max_features": 21, "num_features": 21, "num_classes": 0,
        "prior_type": kind, "informed_mode": True, "device": "cpu",
    }


def inspect_table(prior, params, table_id, bounds, variations, draws, seed, max_attempts):
    _, baseline, _ = prior.generate_dataset(params)
    rule = copy.deepcopy(prior.last_pitting_target_rule)
    original_drive = prior.last_pitting_target_drive.clone()
    original_coefficients = rule["target_rule_coefficients"]
    family = rule["target_rule_family"]
    replayed, _ = prior.replay(rule, params)
    if not torch.equal(baseline, replayed):
        raise RuntimeError("Recorded target replay differs from the original generator output.")
    rule_z = prior._standardize_signal(original_drive)
    scm = prior.recorded_scm
    scm_std = float(scm.std(unbiased=False))
    if not torch.isfinite(scm).all():
        scm_status = "nonfinite"
    elif torch.all(scm == -100):
        scm_status = "failure_sentinel"
    elif scm_std <= 1e-6:
        scm_status = "degenerate"
    else:
        scm_status = "ok"
    common = {"table_id": table_id, "family": family, "scm_type": params["prior_type"], "scm_status": scm_status}
    table = {
        **common, "rows": params["seq_len"], "raw_scm_std": scm_std,
        "raw_rule_std": float(original_drive.std(unbiased=False)),
        "scm_rule_pearson": correlation(scm, rule_z),
        "scm_rule_spearman": correlation(scm, rule_z, ranks=True),
        "scm_nonfinite_count": int((~torch.isfinite(scm)).sum()),
        "scm_max_abs_z": float(prior._standardize_signal(scm).abs().max()),
        "rule_max_abs_z": float(rule_z.abs().max()),
        "features_sha256": tensor_hash(prior.recorded_features),
        "scm_sha256": tensor_hash(scm), "baseline_sha256": tensor_hash(baseline),
        "rule_metadata_sha256": search.canonical_json_sha256(json_safe(rule)),
    }
    terms = prior.rule_terms(rule)
    reconstructed = sum(original_coefficients[name] * values for name, values in terms.items())
    if not torch.allclose(reconstructed, original_drive, atol=2e-6, rtol=2e-6):
        raise RuntimeError("Signed terms do not reconstruct the production rule drive.")
    term_rows = []
    names = list(terms)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            term_rows.append({
                **common, "term_a": left, "term_b": right,
                "pearson": correlation(terms[left], terms[right]),
                "spearman": correlation(terms[left], terms[right], ranks=True),
            })
    measurements, coefficient_rows = [], []
    for variation in variations:
        for draw_id in range(draws if variation else 1):
            draw = sample_coefficients(
                original_coefficients, variation, variation_rng(seed, table_id, variation, draw_id),
                upper_bounds=bounds, max_attempts=max_attempts,
            )
            candidate_rule = copy.deepcopy(rule)
            candidate_rule["target_rule_coefficients"] = draw.coefficients
            candidate, drive = prior.replay(candidate_rule, params)
            candidate_z = prior._standardize_signal(drive)
            if variation == 0 and not torch.equal(candidate, baseline):
                raise RuntimeError("Zero variation changed the generated target.")
            record = {
                **common, "variation": variation, "draw": draw_id,
                "attempts": draw.attempts, "rejected_draws": draw.rejected_draws,
                "used_fallback": draw.used_fallback,
                "candidate_rule_std": float(drive.std(unbiased=False)),
                "candidate_mixed_std": float(candidate.std(unbiased=False)),
                "candidate_scm_rule_pearson": correlation(scm, candidate_z),
            }
            record.update({f"rule_{key}": value for key, value in compare_targets(rule_z, candidate_z).items()})
            record.update({f"mixed_{key}": value for key, value in compare_targets(baseline, candidate).items()})
            measurements.append(record)
            for name, value in draw.coefficients.items():
                coefficient_rows.append({
                    **common, "variation": variation, "draw": draw_id, "coefficient": name,
                    "reference": original_coefficients[name], "sampled": value,
                    "delta": value - original_coefficients[name], "upper_bound": bounds[name],
                })
    return table, term_rows, measurements, coefficient_rows


def summarize(measurements):
    groups = defaultdict(list)
    for row in measurements:
        groups[(row["family"], row["scm_type"], row["scm_status"], row["variation"])].append(row)
    summaries = []
    for (family, kind, status, variation), rows in sorted(groups.items()):
        attempts = sum(row["attempts"] for row in rows)
        summary = {
            "family": family, "scm_type": kind, "scm_status": status, "variation": variation,
            "tables": len({row["table_id"] for row in rows}), "draws": len(rows),
            "rejection_rate": sum(row["rejected_draws"] for row in rows) / attempts if attempts else 0.0,
            "fallback_rate": sum(row["used_fallback"] for row in rows) / len(rows),
        }
        for name in ("rule_pearson", "rule_spearman", "rule_mae", "rule_rmse", "mixed_pearson", "mixed_spearman", "mixed_mae", "mixed_rmse"):
            values = [row[name] for row in rows if row[name] is not None and np.isfinite(row[name])]
            summary[f"{name}_count"] = len(values)
            for label, quantile in (("p10", .1), ("median", .5), ("p90", .9)):
                summary[f"{name}_{label}"] = float(np.quantile(values, quantile)) if values else None
        summaries.append(summary)
    return summaries


def summarize_coefficients(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["variation"], row["coefficient"])].append(row)
    return [{
        "family": family, "variation": variation, "coefficient": name,
        "reference": values[0]["reference"], "sampled_mean": float(np.mean([r["sampled"] for r in values])),
        "sampled_std": float(np.std([r["sampled"] for r in values])),
        "mean_shift": float(np.mean([r["delta"] for r in values])),
        "draws": len(values),
    } for (family, variation, name), values in sorted(groups.items())]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, summaries, tables, coefficient_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, kind in enumerate(SCM_TYPES):
        for family in sorted({row["family"] for row in summaries}):
            selected = [r for r in summaries if r["family"] == family and r["scm_type"] == kind and r["scm_status"] == "ok"]
            selected.sort(key=lambda r: r["variation"])
            for row_index, prefix in enumerate(("rule", "mixed")):
                axes[row_index, column].plot(
                    [100 * r["variation"] for r in selected],
                    [r[f"{prefix}_mae_median"] for r in selected], marker="o", label=family,
                )
                axes[row_index, column].set_ylabel(f"{prefix.title()} target mean absolute change")
                axes[row_index, column].grid(alpha=.2)
        axes[0, column].set_title(kind)
        axes[1, column].set_xlabel("Coefficient variation (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle("Median across paired draws; valid SCM tables only")
    figure.tight_layout(rect=(0, .12, 1, .95))
    figure.savefig(output / "target_changes.png", dpi=150)
    plt.close(figure)

    def table_html(rows, columns):
        head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
        body = []
        for row in rows:
            cells = []
            for column in columns:
                value = row[column]
                label = "undefined" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
                cells.append(f"<td>{html.escape(label)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "<div class='table'><table><tr>" + head + "</tr>" + "".join(body) + "</table></div>"

    failures = [row for row in tables if row["scm_status"] != "ok"]
    report = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Target coefficient variation diagnostic</title>
<style>body{font:15px system-ui,sans-serif;margin:32px auto;padding:0 20px;max-width:1200px;color:#222}
img{max-width:100%;height:auto}.table{overflow-x:auto}table{border-collapse:collapse;font-size:12px}
th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}th{background:#eee}</style>
<h1>Target coefficient variation diagnostic</h1>"""
    report += f"<p>{len(tables)} base tables; {len(failures)} flagged SCM outputs. No model training or real-label evaluation.</p>"
    report += "<p>Balanced coverage of seven formula families and two SCM types, not the production family frequencies. "
    report += "Each comparison holds features, SCM, method offsets, and lambda fixed. Changes are measured after standardization. "
    report += "Draws sharing a table are not independent evidence. Undefined correlations are not replaced with zero. "
    report += "Flagged SCM cases are retained separately, not repaired.</p><img src='target_changes.png' alt='Rule and mixed target changes by variation'>"
    report += "<h2>Paired target comparisons</h2>"
    report += table_html(summaries, ["family", "scm_type", "scm_status", "variation", "tables", "rule_spearman_median", "rule_mae_median", "mixed_spearman_median", "mixed_mae_median", "rejection_rate", "fallback_rate"])
    report += "<h2>Coefficient spread and drift</h2>" + table_html(coefficient_summary, ["family", "variation", "coefficient", "reference", "sampled_mean", "sampled_std", "mean_shift"])
    report += "<h2>SCM versus rule</h2>" + table_html(tables, ["table_id", "family", "scm_type", "scm_status", "scm_rule_pearson", "scm_rule_spearman", "raw_scm_std"])
    report += "<p>Signed within-formula term correlations are in <a href='term_correlations.csv'>term_correlations.csv</a>. "
    report += "Exact run settings and source hashes are in <a href='config.json'>config.json</a>. "
    report += "Predictive benefit requires a separate training experiment.</p></html>"
    (output / "report.html").write_text(report, encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-rule-summary", type=Path, default=search.DEFAULT_TARGET_RULE_SUMMARY)
    parser.add_argument("--split-manifest", type=Path, default=search.DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tables-per-cell", type=int, default=8, help="Tables per formula family / SCM type (14 cells).")
    parser.add_argument("--draws", type=int, default=10, help="Coefficient draws per nonzero variation per table.")
    parser.add_argument("--variations", type=float, nargs="+", default=[0, .1, .2, .3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, help="Default: saved training maximum sequence length.")
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args(argv)
    if args.tables_per_cell < 1 or args.draws < 1 or args.max_attempts < 1:
        parser.error("Table, draw, and attempt counts must be positive.")
    if args.seq_len is not None and args.seq_len < 4:
        parser.error("--seq-len must be at least 4.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32).")
    if any(not np.isfinite(v) or not 0 <= v < 1 for v in args.variations):
        parser.error("Variations must be finite and in [0, 1).")
    args.variations = sorted(set([0.0, *args.variations]))
    return args


def run(args):
    fixed, bounds, training_config, rules = load_settings(
        args.manifest, args.target_rule_summary, args.split_manifest
    )
    seq_len = args.seq_len or training_config.max_seq_len
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or OUTPUT_ROOT / f"run_{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running"})
    try:
        source_paths = [*sorted((ROOT / "src/tabicl/prior").glob("*.py")), Path(__file__),
                        ROOT / "src/tabicl/train/train_config.py", ROOT / "scripts/epit_pipeline/run_optuna.py",
                        ROOT / "scripts/epit_pipeline/artifact_hashes.py"]
        config = {
            "schema": "target_variation_diagnostic_v1", "created_utc": stamp,
            "arguments": vars(args), "output_dir": output, "seq_len": seq_len,
            "fixed_hp": fixed, "coefficient_upper_bounds": bounds,
            "manifest_sha256": sha256_file(args.manifest),
            "rule_summary_sha256": rules.summary_sha256, "rule_artifact_sha256s": rules.artifact_sha256s,
            "source_sha256s": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "scipy": scipy.__version__},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
            "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout,
            "design": "balanced family/SCM cells; independent base-task seeds; separate coefficient RNG; context size = half the table",
            "scope": "generator diagnostic only; no Optuna or model training; no real target labels read",
        }
        write_json(output / "config.json", config)
        torch.set_num_threads(1)
        tables, terms, measurements, coefficients = [], [], [], []
        total = len(bounds) * len(SCM_TYPES) * args.tables_per_cell
        for family in sorted(bounds):
            for kind in SCM_TYPES:
                for _ in range(args.tables_per_cell):
                    table_id = len(tables)
                    task_seed = int(np.random.SeedSequence([args.seed, table_id]).generate_state(1)[0])
                    random.seed(task_seed)
                    np.random.seed(task_seed)
                    torch.manual_seed(task_seed)
                    hp = copy.deepcopy(fixed)
                    hp["pitting_target_rule_scores"] = {family: rules.scores[family]}
                    prior = DiagnosticPrior(batch_size=1, fixed_hp=hp, max_classes=0, n_jobs=1, device="cpu")
                    params = make_params(prior, kind, seq_len)
                    table, term_rows, rows, coefficient_rows = inspect_table(
                        prior, params, table_id, bounds[family], args.variations,
                        args.draws, args.seed, args.max_attempts,
                    )
                    table["task_seed"] = task_seed
                    tables.append(table)
                    terms.extend(term_rows)
                    measurements.extend(rows)
                    coefficients.extend(coefficient_rows)
                    print(f"[{len(tables)}/{total}] {family} {kind}: {table['scm_status']}", flush=True)
        summaries = summarize(measurements)
        coefficient_summary = summarize_coefficients(coefficients)
        for name, rows in (("tables", tables), ("term_correlations", terms), ("measurements", measurements),
                           ("coefficients", coefficients), ("summary", summaries), ("coefficient_summary", coefficient_summary)):
            write_csv(output / f"{name}.csv", rows)
        write_report(output, summaries, tables, coefficient_summary)
        write_json(output / "status.json", {"status": "complete", "tables": len(tables), "comparisons": len(measurements)})
        print(f"Report: {output / 'report.html'}", flush=True)
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    return output


if __name__ == "__main__":
    run(parse_args())
