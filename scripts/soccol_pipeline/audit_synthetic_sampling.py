"""Reproduce the observation-weighted Soccol v2 versus EPIT sampling audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.soccol_pipeline.data import REPO_ROOT, encode_model_features, select_accepted_development
from tabicl.prior.epit_composition_profile import EPIT_COMPOSITION_FAMILY_PROBS
from tabicl.prior.epit_feature_generator import sample_epit_feature_rows
from tabicl.prior.epit_feature_profile import load_epit_feature_profile
from tabicl.prior.soccol_feature_generator import sample_soccol_feature_rows
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile


DEFAULT_OUTPUT = REPO_ROOT / "corrosion_datasets/analysis/soccol_pipeline/synthetic_sampling_audit_v1"
STRENGTHS = (0.0, 0.05, 0.30)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare(n_samples=20000, seed=606):
    require(n_samples > 0, "Sample count must be positive.")
    soccol = load_soccol_feature_profile("soccol_pitting_features_v2")
    epit = load_epit_feature_profile()
    development = select_accepted_development()
    require(np.array_equal(encode_model_features(development.features).to_numpy(), soccol.feature_values, equal_nan=True),
            "Soccol bank differs from encoded real development inputs.")
    require(list(soccol.row_ids) == development.rows.row_id.tolist(), "Soccol bank has different source rows.")
    composition = development.features.iloc[:, :24]
    group_ids = composition.groupby(list(composition), sort=False, dropna=False).ngroup()
    weights = []
    for group, indices in group_ids.groupby(group_ids).groups.items():
        source = development.rows.loc[indices]
        weights.append({
            "composition_group": int(group), "observations": len(indices),
            "current_probability": len(indices) / len(development.rows),
            "uniform_template_probability": 1 / group_ids.nunique(),
            "source_count": source.source.nunique(),
            "source_row_ids": ";".join(source.row_id),
            "stored_families": ";".join(sorted(source.row_material_family.unique())),
            **composition.loc[indices[0]].to_dict(),
        })
    weights = pd.DataFrame(weights).sort_values("observations", ascending=False).reset_index(drop=True)
    p = weights.current_probability.to_numpy()
    active = np.array([EPIT_COMPOSITION_FAMILY_PROBS[epit.composition_profile.families.index(f)] for f in epit.families])
    active /= active.sum()
    template_indices = epit.composition_template_indices
    epit_weights = np.asarray([
        active[i] / np.sum(epit.composition_family_indices == i) for i in epit.composition_family_indices
    ])
    require(np.isclose(epit_weights.sum(), 1.), "EPIT active template probabilities do not sum to one.")
    banks = {
        "soccol": {
            "profile": soccol, "sampler": sample_soccol_feature_rows,
            "composition": soccol.feature_values[:, :24], "context": soccol.feature_values[:, 24:],
            "latent_context": soccol.latent_feature_values[:, 24:],
            "columns": soccol.metadata["feature_columns"],
            "categories": soccol.metadata["categorical_columns"],
            "mappings": soccol.metadata["preprocessing"]["category_mappings"],
        },
        "epit": {
            "profile": epit, "sampler": sample_epit_feature_rows,
            "composition": epit.composition_profile.observed_template_values[template_indices],
            "context": epit.expanded_environment_values, "latent_context": epit.expanded_environment_latent_values,
            "columns": epit.metadata["feature_columns"], "categories": epit.metadata["categorical_columns"],
            "mappings": epit.metadata["category_mappings"],
        },
    }
    diagnostics, features, perturbations, sampled_columns, checks = [], [], [], [], {}
    for name, bank in banks.items():
        for index, column in enumerate(bank["columns"]):
            values = bank["composition"][:, index] if index < 24 else bank["context"][:, index - 24]
            latent = values if index < 24 else bank["latent_context"][:, index - 24]
            observed = values[np.isfinite(values)]
            features.append({
                "dataset": name, "column": column, "block": "composition" if index < 24 else "context",
                "source_entries": len(values), "missing_fraction": float(np.isnan(values).mean()),
                "zero_fraction": float(np.equal(values, 0).mean()), "observed_unique_values": len(np.unique(observed)),
                "all_missing": len(observed) == 0, "constant_after_internal_filling": len(np.unique(latent)) == 1,
                "observed_minimum": float(observed.min()) if len(observed) else np.nan,
                "observed_maximum": float(observed.max()) if len(observed) else np.nan,
            })
        for strength in STRENGTHS:
            kwargs = {"profile": bank["profile"], "perturb_strength": strength, "random_state": seed}
            if name == "epit":
                kwargs["composition_family_probabilities"] = active
            batch = bank["sampler"](n_samples, **kwargs)
            indices = batch.context_indices if name == "soccol" else batch.environment_indices
            original = (soccol.feature_values[batch.compositions.template_indices, :24] if name == "soccol"
                        else epit.composition_profile.observed_template_values[batch.compositions.template_indices])
            repeated = bank["sampler"](n_samples, **kwargs)
            checks[f"{name}_{strength}_same_seed_reproduces_features"] = np.array_equal(batch.features, repeated.features, equal_nan=True)
            del repeated
            checks[f"{name}_{strength}_joint_context_and_nan_pattern"] = np.array_equal(batch.features[:, 24:], bank["context"][indices], equal_nan=True)
            checks[f"{name}_{strength}_matching_internal_context"] = np.array_equal(batch.latent_features[:, 24:], bank["latent_context"][indices])
            checks[f"{name}_{strength}_finite_internal_values"] = bool(np.isfinite(batch.latent_features).all())
            checks[f"{name}_{strength}_valid_public_values"] = bool(not np.isinf(batch.features).any() and np.isfinite(batch.features[:, :24]).all())
            checks[f"{name}_{strength}_zero_elements_stay_zero"] = np.array_equal(batch.features[:, :24] == 0, original == 0)
            checks[f"{name}_{strength}_composition_closes"] = bool(np.allclose(batch.features[:, :24].sum(axis=1), 100., atol=1e-10, rtol=0))
            checks[f"{name}_{strength}_categories_use_recorded_codes"] = all(
                set(batch.features[:, bank["columns"].index(c)][np.isfinite(batch.features[:, bank["columns"].index(c)])])
                <= set(bank["mappings"][c].values()) for c in bank["categories"])
            if strength == 0:
                checks[f"{name}_zero_noise_only_closes_synthetic_compositions"] = bool(np.allclose(
                    batch.features[:, :24], original * (100. / original.sum(axis=1, keepdims=True)), atol=1e-12, rtol=1e-12))
            shifts = np.abs(batch.features[:, :24] - original)
            perturbations.append({
                "dataset": name, "strength": strength, "samples": n_samples,
                "max_abs_element_change_per_row_median_wt_percent": float(np.median(shifts.max(axis=1))),
                "max_abs_element_change_per_row_p95_wt_percent": float(np.quantile(shifts.max(axis=1), .95)),
                "max_abs_element_change_per_row_max_wt_percent": float(shifts.max()),
                "max_composition_closure_error": float(np.max(np.abs(batch.features[:, :24].sum(axis=1) - 100))),
            })
            for index, column in enumerate(bank["columns"][:24]):
                x, support = batch.features[:, index], bank["composition"][:, index]
                sampled_columns.append({"dataset": name, "strength": strength, "column": column,
                                        "minimum": x.min(), "maximum": x.max(),
                                        "outside_original_column_range_fraction": float(((x < support.min()-1e-10) | (x > support.max()+1e-10)).mean())})
            observed_context_missing = np.isnan(batch.features[:, 24:]).mean(axis=0)
            diagnostics.append({
                "dataset": name, "strength": strength, "sampled_context_rows": len(np.unique(indices)),
                "sampled_composition_source_entries": len(np.unique(batch.compositions.template_indices)),
                "sampled_distinct_source_compositions": len(np.unique(original, axis=0)),
                "context_missing_fraction": float(np.isnan(batch.features[:, 24:]).mean()),
                "maximum_column_missing_rate_deviation_from_bank": float(np.max(np.abs(observed_context_missing - np.isnan(bank["context"]).mean(axis=0)))),
                "feature_count": batch.features.shape[1],
            })
            if name == "soccol" and strength == 0:
                draw_groups = group_ids.to_numpy()[batch.compositions.template_indices]
                sampled = pd.Series(draw_groups).value_counts() / n_samples
                weights["sampled_probability"] = weights.composition_group.map(sampled).fillna(0.)
                varying_indices = np.std(batch.compositions.template_indices) > 0 and np.std(indices) > 0
                summary_draw = {
                    "sampled_top20_composition_share": float(np.isin(draw_groups, weights.composition_group.head(20)).mean()),
                    "source_composition_environment_same_row_fraction": float(np.mean(batch.compositions.template_indices == indices)),
                    "source_index_pearson_correlation": float(np.corrcoef(batch.compositions.template_indices, indices)[0, 1]) if varying_indices else None,
                }
    checks["epit_formula_and_scm_fill_the_same_three_environment_inputs"] = all(np.array_equal(
        epit.environment_imputed_values[:, i], epit.expanded_environment_latent_values[:, banks["epit"]["columns"][24:].index(c)])
        for i, c in enumerate(("CP_temp", "CP_Cl", "CP_pH")))
    require(all(checks.values()), f"Sampling checks failed: {[k for k, v in checks.items() if not v]}")
    shared = sorted(set(banks["soccol"]["columns"][24:]) & set(banks["epit"]["columns"][24:]))
    sources = [Path(__file__),
               *(REPO_ROOT / "src/tabicl/prior" / f for f in (
                   "soccol_feature_generator.py", "soccol_feature_profile.py", "epit_feature_generator.py", "epit_feature_profile.py",
                   "epit_composition_profile.py", "dataset.py", "assets/soccol_pitting_features_v2.csv", "assets/soccol_pitting_features_v2.json",
                   "assets/epit_fe_nicrmo_features_v4.csv", "assets/epit_fe_nicrmo_features_v4.json", "assets/epit_dataset_v4.csv", "assets/epit_dataset_v4.json"))]
    summary = {
        "schema_version": "soccol_synthetic_sampling_audit_v1", "seed": seed, "samples_per_dataset_per_strength": n_samples,
        "strengths": list(STRENGTHS), "final_query_rows_used": 0, "targets_used_for_bank_sampling": False,
        "soccol_source_rows": len(development.rows), "soccol_distinct_compositions": len(weights),
        "soccol_top1_composition_share": float(p[0]), "soccol_top10_composition_share": float(p[:10].sum()),
        "soccol_top20_composition_share": float(p[:20].sum()),
        "soccol_effective_template_count": float(1 / np.sum(p*p)),
        "epit_distinct_compositions": len(epit_weights), "epit_top20_composition_share": float(np.sort(epit_weights)[-20:].sum()),
        "epit_effective_template_count": float(1 / np.sum(epit_weights*epit_weights)),
        "effective_count_definition": "1 / sum(template_probability squared); concentration of draw weights, not number of physically possible perturbed alloys.",
        "soccol_filling_scope": "2,950 accepted development rows",
        "epit_filling_scope": "608 outer development rows; 452 Fe/Ni environment rows sampled",
        "shared_context_columns": shared,
        "shared_context_missing_fraction": {
            name: float(np.isnan(bank["context"][:, [bank["columns"][24:].index(c) for c in shared]]).mean()) for name, bank in banks.items()
        },
        "soccol_draw_diagnostic": summary_draw, "checks": checks,
        "scope": "Feature sampling diagnostic only. Sampling policy, datasets, splits, formulas and training settings were unchanged.",
        "interpretation": "Repeated Soccol observations concentrate draws on a few compositions. This is a measured difference from EPIT; impact on trained-model scores requires a controlled comparison.",
        "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in sources},
    }
    return summary, {
        "soccol_composition_weights.csv": weights,
        "bank_feature_summary.csv": pd.DataFrame(features),
        "sampling_summary.csv": pd.DataFrame(diagnostics),
        "composition_perturbation_summary.csv": pd.DataFrame(perturbations),
        "sampled_composition_ranges.csv": pd.DataFrame(sampled_columns),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=606)
    args = parser.parse_args()
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Choose a new --output-dir; existing audits are not overwritten.")
    summary, tables = compare(args.samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / name, index=False)
    summary["output_sha256"] = {name: sha256_file(args.output_dir / name) for name in tables}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Saved sampling audit to {args.output_dir}; {len(summary['checks'])} checks passed.")
    print(f"Top 20 composition shares: Soccol {summary['soccol_top20_composition_share']:.4%}; EPIT {summary['epit_top20_composition_share']:.4%}")


if __name__ == "__main__":
    main()
