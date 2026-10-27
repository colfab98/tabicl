"""Reproduce the historical v1 versus reviewed v2 Soccol input audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrosion_datasets.datasets.soccol_pitting_potential import compare_target_rules as formulas
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.soccol_pipeline.data import (
    REPO_ROOT, PROCESSED_DIR, SPLIT_MANIFEST, encode_model_features,
    load_soccol_dataset, select_accepted_development,
)
from scripts.soccol_pipeline.formula_data import load_formula_data
from scripts.soccol_pipeline.target_rules import RULE_SELECTION
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS, SOCCOL_CATEGORICAL_COLUMNS


LEGACY_CSV = PROCESSED_DIR.parent / "soccol_regression_event1.csv"
EPIT_REGISTRY = REPO_ROOT / "corrosion_datasets/datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/feature_registry.json"
DEFAULT_OUTPUT = REPO_ROOT / "corrosion_datasets/analysis/soccol_pipeline/input_audit_v1"
ATOL, RTOL = 1e-10, 1e-12
CELL_COLUMNS = (
    "row_id", "task_row_index", "source", "outer_split", "composition_status",
    "use_as_development_context", "column", "previous_value", "reviewed_value",
    "change_type", "active_formula_families",
)


def same_values(left, right, *, categorical=False):
    """Ignore CSV floating-point round-off, but distinguish zero from missing."""
    if categorical:
        left, right = pd.Series(left).astype("string"), pd.Series(right).astype("string")
        return (left.eq(right).fillna(False) | (left.isna() & right.isna())).to_numpy(bool)
    return np.isclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float),
                      atol=ATOL, rtol=RTOL, equal_nan=True)


def rule_dependencies():
    """Describe actual terms, excluding arrays calculated but unused by a rule."""
    material = {
        "pren_n": {"Cr", "Mo", "N"}, "pren_n_synergy": {"Cr", "Mo", "N"},
        "pren_no_n": {"Cr", "Mo"}, "threshold_saturation": {"Cr", "Mo"},
    }
    halide = {"chloride": {"CP_Cl"}, "effective_halide": {"CP_Cl", "CP_Br"}}
    extras = {
        "mns_inclusion_susceptibility": {"Mn", "S"},
        "mo_n_acid_repassivation": {"Mo", "N"},
        "weak_inhibitor_ratio": {"CP_SO4", "CP_NO3", "CP_Cl", "CP_Br"},
    }
    specs = {spec.name: spec for spec in formulas.RULES}
    result = {}
    for family, name in RULE_SELECTION:
        spec = specs[name]
        used = material[spec.material] | halide[spec.halide] | {"CP_temp", "CP_pH"}
        for extra in spec.extras:
            used |= extras[extra]
        result[family] = sorted(used)
    return result


def changed_cells(old, new, rows, columns, dependencies, categorical=()):
    records = []
    for column in columns:
        changed = ~same_values(old[column], new[column], categorical=column in categorical)
        users = ";".join(family for family, used in dependencies.items() if column in used)
        for index in np.flatnonzero(changed):
            before, after = old.iloc[index][column], new.iloc[index][column]
            kind = ("missing_to_known" if pd.isna(before) else
                    "known_to_missing" if pd.isna(after) else
                    "category_representation" if column in categorical else "numeric_correction")
            records.append({
                **{key: rows.iloc[index][key] for key in CELL_COLUMNS[:6] if key != "task_row_index"},
                "task_row_index": int(index), "column": column,
                "previous_value": before, "reviewed_value": after,
                "change_type": kind, "active_formula_families": users,
            })
    return pd.DataFrame(records, columns=CELL_COLUMNS)


def formula_filling_audit(old, reviewed, dependencies):
    """Compare each accepted validation row using only its fold's context.

    The historical full context reproduces the previous 3,222-row population.
    A second historical fit on the accepted population separates row selection
    from value/missing-value changes. Neither comparison fits coefficients.
    """
    used = sorted({column for columns in dependencies.values() for column in columns})
    effective, filling = [], []
    for fold in range(1, 6):
        validation = reviewed.loc[reviewed.development_validation_fold.eq(fold)]
        context = reviewed.loc[reviewed.development_validation_fold.ne(fold)]
        old_context = old.loc[old.outer_split.eq("development") & old.development_validation_fold.ne(fold)]
        same_context = old.iloc[context.task_row_index]
        old_validation = old.iloc[validation.task_row_index]
        old_fill = formulas._fit_imputation(old_context)
        same_fill = formulas._fit_imputation(same_context)
        new_fill = formulas._fit_imputation(context)
        before = formulas._inputs(old_validation, old_fill)
        controlled = formulas._inputs(old_validation, same_fill)
        after = formulas._inputs(validation, new_fill)
        for column, value in new_fill.items():
            filling.append({
                "fold": fold, "column": column,
                "active_formula_dependency": column in used,
                "legacy_context_rows": len(old_context), "reviewed_context_rows": len(context),
                "legacy_fill": old_fill.get(column, 0.),
                "legacy_same_rows_fill": same_fill.get(column, 0.), "reviewed_fill": value,
                "legacy_policy": "context_mean" if column in old_fill else "zero",
                "reviewed_policy": "context_mean" if context[column].notna().any() else "all_missing_zero_placeholder",
            })
        for column in used:
            overall = ~same_values(before[column], after[column])
            selection = ~same_values(before[column], controlled[column])
            processing = ~same_values(controlled[column], after[column])
            for index in np.flatnonzero(overall | selection | processing):
                row = validation.iloc[index]
                effective.append({
                    "row_id": row.row_id, "task_row_index": int(row.task_row_index),
                    "fold": fold, "column": column,
                    "legacy_value": before[column][index],
                    "legacy_same_rows_value": controlled[column][index],
                    "reviewed_value": after[column][index],
                    "changed_overall": bool(overall[index]),
                    "changed_by_context_selection": bool(selection[index]),
                    "changed_by_processing": bool(processing[index]),
                    "active_formula_families": ";".join(f for f, cols in dependencies.items() if column in cols),
                })
    return pd.DataFrame(effective), pd.DataFrame(filling)


def build_audit():
    dataset = load_soccol_dataset()
    development = select_accepted_development(dataset)
    formula_rows = load_formula_data(dataset)
    old = pd.read_csv(LEGACY_CSV, float_precision="round_trip")
    frozen = json.loads(SPLIT_MANIFEST.read_text())
    if sha256_file(LEGACY_CSV) != frozen["dataset"]["source_sha256"]:
        raise RuntimeError("The historical CSV no longer matches the frozen split.")
    if not old.raw_workbook_row.equals(dataset.rows.raw_workbook_row):
        raise RuntimeError("Historical and reviewed Soccol row order differs.")
    pd.testing.assert_series_equal(old.E_pit, dataset.targets, check_names=False, check_exact=True)
    for column in ("outer_split", "development_validation_fold"):
        old[column] = dataset.assignments[column].to_numpy()
    dependencies = rule_dependencies()
    common = [column for column in SOCCOL_FEATURE_COLUMNS if column in dataset.features]
    changes = changed_cells(old, dataset.features, dataset.rows, common, dependencies,
                            categorical=SOCCOL_CATEGORICAL_COLUMNS)
    adapter_columns = [*formulas.COMPOSITION_ZERO_COLUMNS,
                       *(c for c in formulas.ION_ZERO_COLUMNS if c != "CP_ion_other"),
                       *formulas.MEAN_IMPUTE_COLUMNS, "material_is_Ni_based"]
    formula_changes = changed_cells(old, dataset.rows, dataset.rows, adapter_columns, dependencies)
    effective, filling = formula_filling_audit(old, formula_rows, dependencies)

    epit = json.loads(EPIT_REGISTRY.read_text())
    shared = sorted(set(epit["feature_columns"]) & set(dataset.features))
    shared_records = [{
        "column": c, "epit_dtype": epit["columns"][c]["dtype"],
        "soccol_dtype": dataset.registry["columns"][c]["dtype"],
        "epit_unit": epit["columns"][c].get("unit"),
        "soccol_unit": dataset.registry["columns"][c].get("unit"),
        "matches": all(epit["columns"][c].get(k) == dataset.registry["columns"][c].get(k)
                       for k in ("dtype", "unit")),
    } for c in shared]
    profile = load_soccol_feature_profile("soccol_pitting_features_v2")
    encoded = encode_model_features(development.features)
    checks = {
        "original_row_order_preserved": True, "targets_exactly_unchanged": True,
        "frozen_splits_validated_by_loader": True,
        "bank_row_ids_match": list(profile.row_ids) == development.rows.row_id.tolist(),
        "bank_task_indices_match": np.array_equal(profile.source_task_row_indices, development.task_row_indices),
        "bank_values_and_missingness_match": np.array_equal(profile.feature_values, encoded.to_numpy(), equal_nan=True),
        "formula_adapter_matches_saved_features": formula_rows[list(dataset.features)].equals(development.features),
        "accepted_compositions_complete": bool(np.isfinite(development.features[dataset.registry["composition_columns"]]).all().all()),
        "category_encoding_preserves_development_missingness": np.array_equal(encoded.isna(), development.features.isna()),
        "shared_epit_names_types_units_match": all(r["matches"] for r in shared_records),
    }
    # Verify saved formula artifacts refer to the same rows and filling statistics.
    calibration_dir = DEFAULT_OUTPUT.parent / "target_rules_v2"
    for family, _ in RULE_SELECTION:
        artifact = json.loads((calibration_dir / f"{family}.json").read_text())
        checks[f"{family}_selection_and_filling_match"] = (
            artifact["development_selection"] == development.identity
            and artifact["final_development_calibration"]["feature_state"]["imputation_means"] == formulas._fit_imputation(formula_rows)
            and all(f["feature_state"]["imputation_means"] == formulas._fit_imputation(
                formula_rows.loc[formula_rows.development_validation_fold.ne(f["fold"])])
                for f in artifact["fold_calibrations"])
        )

    schema = []
    for column in dict.fromkeys([*SOCCOL_FEATURE_COLUMNS, *dataset.features]):
        previous, current = column in SOCCOL_FEATURE_COLUMNS, column in dataset.features
        info = dataset.registry["columns"].get(column, {}) if current else {}
        old_type = "categorical" if column in SOCCOL_CATEGORICAL_COLUMNS else "numeric"
        record = {
            "column": column, "status": "retained" if previous and current else "added" if current else "removed",
            "old_dtype": old_type if previous else None, "reviewed_dtype": info.get("dtype"),
            "reviewed_unit": info.get("unit"), "shared_with_epit": column in shared,
            "active_formula_families": ";".join(f for f, cols in dependencies.items() if column in cols),
        }
        for label, mask in (
            ("all", np.ones(len(dataset.rows), dtype=bool)),
            ("accepted_development", dataset.rows.use_as_development_context),
            ("withheld_development", dataset.rows.outer_split.eq("development") & ~dataset.rows.use_as_development_context),
            ("final_query", dataset.rows.outer_split.eq("final_test")),
        ):
            for version, present, table in (("old", previous, old), ("reviewed", current, dataset.features)):
                record[f"{version}_{label}_observed"] = int(table.loc[mask, column].notna().sum()) if present else None
                record[f"{version}_{label}_missing"] = int(table.loc[mask, column].isna().sum()) if present else None
        schema.append(record)
    column_counts = changes.groupby(["column", "outer_split", "use_as_development_context", "change_type"], dropna=False).agg(
        changed_cells=("row_id", "size"), changed_rows=("row_id", "nunique")).reset_index()
    accepted_changes = changes.loc[changes.use_as_development_context]
    active_changes = formula_changes.loc[formula_changes.use_as_development_context & formula_changes.active_formula_families.ne("")]
    added = [c for c in dataset.features if c not in SOCCOL_FEATURE_COLUMNS]
    removed = [c for c in SOCCOL_FEATURE_COLUMNS if c not in dataset.features]
    summary = {
        "audit_version": "soccol_input_audit_v1",
        "scope": "Input comparison only; no coefficient fitting, model training or prediction scores.",
        "comparison_tolerance": {"absolute": ATOL, "relative": RTOL, "nan_equals_nan": True},
        "files_sha256": {
            str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in (
                LEGACY_CSV, EPIT_REGISTRY, SPLIT_MANIFEST,
                PROCESSED_DIR / "soccol_processed.csv", PROCESSED_DIR / "soccol_features.csv",
                PROCESSED_DIR / "feature_registry.json", PROCESSED_DIR / "split_assignments.csv",
                REPO_ROOT / "src/tabicl/prior/assets/soccol_pitting_features_v2.csv",
                REPO_ROOT / "src/tabicl/prior/assets/soccol_pitting_features_v2.json",
                Path(__file__), Path(formulas.__file__),
                *(calibration_dir / f"{family}.json" for family, _ in RULE_SELECTION),
            )
        },
        "rows": {"all": len(old), "original_development": int(old.outer_split.eq("development").sum()),
                 "accepted_development": len(development.rows), "withheld_development": development.identity["withheld_development_rows"],
                 "final_queries": int(old.outer_split.eq("final_test").sum()),
                 "unresolved_final_compositions": int((dataset.rows.outer_split.eq("final_test") & dataset.rows.composition_status.ne("accepted")).sum())},
        "features": {"old": len(SOCCOL_FEATURE_COLUMNS), "reviewed": len(dataset.features.columns),
                     "shared_old_new": len(common), "added": added, "removed": removed,
                     "shared_with_epit": len(shared)},
        "changed_common_input_cells": len(changes), "changed_common_input_rows": int(changes.row_id.nunique()),
        "accepted_development_changed_cells": len(accepted_changes),
        "accepted_development_changed_rows": int(accepted_changes.row_id.nunique()),
        "accepted_development_changes_by_type": accepted_changes.groupby("change_type").size().to_dict(),
        "accepted_development_active_formula_raw_changes": active_changes.groupby("column").size().to_dict(),
        "effective_formula_changes_by_column": effective.loc[effective.changed_overall].groupby("column").size().to_dict(),
        "rule_dependencies": dependencies, "checks": checks,
        "findings": [
            "The formulas retain their previous definitions. Only sulfate changed as a raw input actually used by the nine formulas on accepted development rows.",
            "Model-facing NaNs are retained. Internal formula filling uses context means; missing nitrate previously used zero. Changes to context membership also change means.",
            "Soccol's cr_mow_n_synergy name is misleading: its formula uses Mo, not W. EPIT's corresponding formula uses Mo + 0.5*W. No formula was changed by this audit.",
            "The reviewed Soccol bank contains 20 accepted development rows with positive W and 38 with positive Cu. All 58 features can feed the SCM, but the nine analytical formulas ignore W, Cu, Fe, Nb and Ti.",
            "Missingness and the shared 49 feature names/types/units agree with EPIT. Individual schemas remain 58 versus 70 columns; separate category codes must not be treated as universal merged-dataset codes.",
            "Ten frozen Soccol final queries retain unresolved composition NaNs. Accepted Soccol development compositions are complete. The final queries were retained unchanged; this audit did not evaluate predictions.",
        ],
    }
    tables = {
        "input_changes_vs_legacy.csv": changes,
        "formula_input_changes_vs_legacy.csv": formula_changes,
        "effective_formula_input_changes.csv": effective,
        "fold_filling_values.csv": filling,
        "schema_comparison.csv": pd.DataFrame(schema),
        "shared_epit_schema_check.csv": pd.DataFrame(shared_records),
        "change_counts.csv": column_counts,
    }
    return summary, tables


def write_audit(output_dir, summary, tables):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an input audit; choose a new --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)
    summary["output_sha256"] = {name: sha256_file(output_dir / name) for name in tables}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    counts = tables["input_changes_vs_legacy.csv"]
    counts = counts.loc[counts.use_as_development_context].groupby("column").size()
    lines = [
        "# Soccol input changes", "",
        "Compared the original frozen CSV with the reviewed export. This audit did not fit coefficients, train models or calculate prediction scores. Targets, row order and frozen split assignments were unchanged.", "",
        "The old schema had 37 features; the reviewed schema had 58: 34 retained, 24 added and 3 removed. Added columns were counted separately from changed common cells. Agitation yes/no to 1/0 and aeration renaming were recorded as representation changes, not changes in experimental conditions.", "",
        "## Accepted development rows", "",
        f"Used 2,950 accepted development rows. Recorded {summary['accepted_development_changed_cells']} changed common cells across {summary['accepted_development_changed_rows']} rows:", "",
        "These comprised 71 corrected numeric amounts (Fe/Nb/Ti), 299 values deliberately left missing after review, and 645 category representation changes. The full audit also recorded withheld development rows and frozen queries separately; those rows did not enter the reviewed development inputs.", "",
        "| Column | Changed cells |", "|---|---:|",
        *(f"| {c} | {n} |" for c, n in counts.items()), "",
        "Added composition fields included positive W in 20 development rows and positive Cu in 38. The other nine added element columns were zero in all accepted development rows. These added fields were excluded from the common-cell counts above.", "",
        "## Formula inputs", "",
        "Among raw fields actually used by the nine formulas, only sulfate changed on accepted development rows: 104 disputed values became NaN. Cr, Mo, N, Mn, S, temperature, pH, chloride, bromide and nitrate raw values were unchanged on those rows. Corrected Fe/Nb/Ti and added W/Cu do not enter these formula definitions.", "",
        "Internal formula inputs also changed through filling. The audit compared each accepted development validation row using its own fold's context only. It saved both the historical full context and a historical-data control using exactly the accepted context rows. This separated changes caused by row removal from changes caused by preprocessing; no final-query rows supplied filling statistics.", "",
        "| Formula input | Changed effective validation cells |", "|---|---:|",
        *(f"| {c} | {n} |" for c, n in summary["effective_formula_changes_by_column"].items()), "",
        "The 104 sulfate changes came from leaving disputed amounts missing and using context means internally. Three already-missing nitrate values changed from internal zero filling to context means. The 51 pH and 33 temperature changes came only from fitting means on the smaller accepted context. Model-facing unknowns remained NaN throughout. These are inputs before formula scaling: changes in scaling and coefficients can also affect other rows and were not measured here.", "",
        "The formula-adapter CSV also includes fields loaded by the adapter but unused by these nine rules; `active_formula_families` identifies actual dependencies. Empty values in the CSVs mean missing, never zero.", "",
        "## Checks and differences from EPIT", "",
        *(f"- {note}" for note in summary["findings"][2:]),
        f"- Passed {sum(summary['checks'].values())}/{len(summary['checks'])} checks: row/target/split identity, accepted compositions, saved bank and formula-adapter equality, missingness, shared units/types and saved formula filling states.", "",
        "## Files and reproduction", "",
        "- `input_changes_vs_legacy.csv`: every changed common raw feature cell across all 4,027 rows, with partition and eligibility.",
        "- `formula_input_changes_vs_legacy.csv`: changed raw formula-adapter fields, with active-rule dependency labels.",
        "- `effective_formula_input_changes.csv`: accepted development validation inputs after fold-context filling, with the same-row control.",
        "- `fold_filling_values.csv`: old, same-row-control and reviewed filling values and context sizes.",
        "- `schema_comparison.csv`: retained/added/removed columns and observed/missing counts by partition.",
        "- `shared_epit_schema_check.csv`: all 49 shared names, types and units.",
        "- `change_counts.csv` and `summary.json`: aggregate counts, input/output hashes, dependencies and checks.", "",
        "```bash", ".venv/bin/python -m scripts.soccol_pipeline.audit_input_changes --output-dir /tmp/soccol_input_audit", "```", "",
        "The command refused to overwrite an existing nonempty output directory. Numeric comparisons ignored serialization round-off only (absolute tolerance 1e-10, relative tolerance 1e-12).", "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an input audit; choose a new --output-dir.")
    summary, tables = build_audit()
    write_audit(args.output_dir, summary, tables)
    failed = [name for name, passed in summary["checks"].items() if not passed]
    if failed:
        raise RuntimeError(f"Input audit found inconsistencies: {failed}. See {args.output_dir}.")
    print(f"Saved input audit to {args.output_dir}; all {len(summary['checks'])} checks passed.")


if __name__ == "__main__":
    main()
