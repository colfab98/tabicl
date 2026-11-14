"""Explore explicit Soccol rule extensions using accepted development rows only.

Uses the EPIT nested development-fold selection procedure and reproduces the
nine current Soccol calibrations before comparing additional hypotheses. This
research entry point does not change training recipes or runtime formula assets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from corrosion_datasets.datasets.soccol_pitting_potential import compare_target_rules as old
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.discover_formula_extensions import (
    bootstrap_difference, coefficient_stability,
)
from scripts.epit_pipeline.formula_extension_study import (
    Candidate, FormulaStudy, PENALTIES, SELECTION_TOLERANCE,
    full_development_fits, nested_evaluation, transform_features,
)
from scripts.soccol_pipeline.data import REPO_ROOT
from scripts.soccol_pipeline.formula_data import load_formula_data
from scripts.soccol_pipeline.target_rules import RULE_SELECTION
from tabicl.prior.soccol_reviewed_schema import SOCCOL_REVIEWED_FEATURE_COLUMNS
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS


BASE_RULE = "pren_n_coupled_mns_weak_anions"
DEFAULT_OUTPUT = REPO_ROOT / "corrosion_datasets/analysis/soccol_pipeline/formula_extension_study_v1"
SOURCES = [
    {"topic": "Nitric-acid passivation and nonmonotone treatment effects",
     "url": "https://doi.org/10.1016/S0010-938X(00)00052-4"},
    {"topic": "Surface MnS removal and pitting resistance",
     "url": "https://doi.org/10.1155/2012/482730"},
    {"topic": "Scan-rate dependence of measured pitting potential",
     "url": "https://doi.org/10.1016/S0010-938X(03)00102-1"},
    {"topic": "Exposure area is not a universal proportional pitting law",
     "url": "https://doi.org/10.1016/0010-938X(95)00046-M"},
    {"topic": "Conventional tungsten-inclusive PREN definition",
     "url": "https://bssa.org.uk/bssa_articles/calculation-of-pitting-resistance-equivalent-numbers-pren/"},
    {"topic": "Aging-treatment effects depend on microstructure and temperature",
     "url": "https://doi.org/10.3390/ma18081823"},
]


def feature_library(frame):
    raw, definitions = {}, {}

    def column(name):
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)

    def add(name, value, definition):
        raw[name] = np.asarray(value, dtype=float)
        definitions[name] = definition

    for name in ("Cr", "N", "Cu", "Prep_pH", "Prep_redox"):
        add(name, column(name), name + " in saved units")
    mo, w = column("Mo"), column("W")
    add("log_Mo", np.log1p(mo), "log(1 + Mo [wt.%])")
    add("log_W", np.log1p(w), "log(1 + W [wt.%])")
    add("PREN", column("Cr") + 3.3 * (mo + .5*w) + 16*column("N"),
        "Cr + 3.3*(Mo + 0.5*W) + 16*N; wt.%")
    add("temperature", column("CP_temp"), "CP_temp (deg C)")
    halide = column("CP_Cl") + .5*column("CP_Br")
    add("log_halide", np.log10(np.maximum(halide, 1e-12)), "log10(max(CP_Cl + 0.5*CP_Br, 1e-12)); mol/L")
    add("acidic_pH", np.maximum(7-column("CP_pH"), 0), "max(7 - CP_pH, 0)")
    add("MnS", np.sqrt(column("Mn")*column("S")), "sqrt(Mn*S); wt.%")
    add("redox_squared", column("Prep_redox")**2, "Prep_redox squared; exploratory treatment-window term")
    for name, source in (
        ("log_scan", "scan_rate"), ("log_area", "Test_area_cm2"),
        ("log_exposure", "CP_time"), ("log_grit", "Prep_grinding_grit"),
        ("log_roughness", "Prep_Ra_micron"), ("log_prep_time", "Prep_time"),
        ("log_heat_hours", "heat_treatment_duration_h"),
    ):
        x = column(source)
        add(name, np.log10(np.where(x > 0, x, np.nan)),
            f"log10({source} in saved units); missing/nonpositive remains unknown")
    add("heat_temperature", column("heat_treatment_temperature_C"), "heat_treatment_temperature_C")
    category_names = ("surface_preparation_method", "preparation_solution_type",
                      "heat_treatment_type", "heat_treatment_cooling", "test_solution_type")
    categories = {c: frame[c].astype("string").fillna("").str.strip().to_numpy(dtype=str)
                  for c in category_names}
    known = categories["preparation_solution_type"] != ""
    nitric = np.char.find(categories["preparation_solution_type"], "HNO3") >= 0
    add("MnS_nitric_prep", np.where(known, raw["MnS"]*nitric, np.nan),
        "sqrt(Mn*S) * indicator(preparation solution contains HNO3); unknown solution stays unknown")
    return raw, categories, definitions


def candidates():
    specs = [Candidate("existing_"+family, "existing", existing_rule=family)
             for family, _ in RULE_SELECTION]
    for family in ("pren_n_linear", "cr_mow_n_synergy", BASE_RULE):
        specs.append(Candidate("updated_W_"+family, "extension", existing_rule="tungsten:"+family))
    surface_categories = ("surface_preparation_method", "preparation_solution_type")
    heat_categories = ("heat_treatment_type", "heat_treatment_cooling")
    surface_numeric = ("log_grit", "log_roughness", "Prep_pH", "Prep_redox", "log_prep_time")
    measurement = ("log_scan", "log_area", "log_exposure")
    groups = [
        ("surface_method", (), surface_categories[:1]),
        ("preparation_solution", (), surface_categories[1:]),
        ("surface_categories", (), surface_categories),
        ("surface_numeric", surface_numeric, ()),
        ("surface", surface_numeric, surface_categories),
        ("passivation", ("Prep_redox", "redox_squared", "log_prep_time", "MnS_nitric_prep"), surface_categories),
        ("heat_categories", (), heat_categories),
        ("heat_numeric", ("heat_temperature", "log_heat_hours"), ()),
        ("W_Cu", ("log_W", "Cu"), ()),
        ("measurement", measurement, ()),
        ("solution_type", (), ("test_solution_type",)),
        ("surface_measurement", surface_numeric+measurement, surface_categories),
        ("supported_new", ("log_W", "Cu"), surface_categories+heat_categories),
        ("supported_new_measurement", ("log_W", "Cu")+surface_numeric+measurement,
         surface_categories+heat_categories),
    ]
    for name, numeric, categorical in groups:
        specs.append(Candidate("extend_"+name, "extension", numeric, categorical, BASE_RULE))
    core = ("PREN", "temperature", "log_halide", "acidic_pH")
    specs.extend([
        Candidate("new_prenW_environment", "new", core),
        Candidate("new_prenW_surface", "new", core+surface_numeric, surface_categories),
    ])
    return specs


class SoccolFormulaStudy(FormulaStudy):
    def __init__(self, development):
        selection = development.attrs["soccol_development_selection"]
        if (not development.outer_split.eq("development").all()
                or not development.composition_status.eq("accepted").all()
                or development.row_id.tolist() != selection["source_row_ids"]
                or development.task_row_index.tolist() != selection["source_task_row_indices"]
                or development.development_validation_fold.astype(int).tolist() != selection["validation_folds"]):
            raise ValueError("Formula search requires the unchanged accepted development selection.")
        self.development = development.copy().reset_index(drop=True)
        self.frame = self.development.loc[:, list(SOCCOL_REVIEWED_FEATURE_COLUMNS)].copy()
        self.rows = development.task_row_index.to_numpy(dtype=int)
        self.y = development.E_pit.to_numpy(dtype=float)
        self.folds = development.development_validation_fold.to_numpy(dtype=int)
        self.groups = development.composition_group.to_numpy(dtype=str)
        self.source_groups = development.source.to_numpy(dtype=str)
        if not np.isfinite(self.y).all():
            raise ValueError("Development targets must be finite.")
        self.raw, self.categories, self.definitions = feature_library(self.frame)
        self.specs = {s.name: s for s in candidates()}
        self.old_cache, self.design_cache, self.model_cache = {}, {}, {}
        all_specs = {s.name: s for s in old.RULES}
        self.old_specs = {family: all_specs[name] for family, name in RULE_SELECTION}

    def old_frame(self, family):
        if family.startswith("tungsten:"):
            data = self.development.copy()
            data["Mo"] = data["Mo"] + .5*data["W"]
            return data, family.split(":", 1)[1]
        return self.development, family

    def old_model(self, train, family_name):
        key = (tuple(train), family_name)
        if key not in self.old_cache:
            frame, family = self.old_frame(family_name)
            spec = self.old_specs[family]
            terms, state = old._fit_term_state(frame.iloc[train], spec)
            weights, _ = old._fit_coefficients(terms, self.y[train], spec.anchor, spec.upper_bounds,
                                              .1, clamp_bound_roundoff=True)
            prediction = old._transform_terms(frame, spec, state) @ weights
            artifact = {"mode": "existing", "family": family_name, "feature_state": state,
                        "term_names": list(spec.term_names), "coefficients": weights.tolist()}
            self.old_cache[key] = prediction, artifact
        return self.old_cache[key]

    def predict_artifact(self, artifact):
        def old_prediction(model):
            frame, family = self.old_frame(model["family"])
            return old._transform_terms(frame, self.old_specs[family], model["feature_state"]) @ np.asarray(model["coefficients"])
        if artifact["mode"] == "existing":
            return old_prediction(artifact)
        X = transform_features(self.raw, self.categories, artifact["feature_state"], np.arange(len(self.y)))
        if artifact["mode"] == "extension":
            base = old_prediction(artifact["base"])
            X = np.column_stack([(base-artifact["base_center"])/artifact["base_scale"], X])
        return X @ np.asarray(artifact["coefficients"])


def coverage(study):
    records = []
    for column in study.frame:
        values = study.frame[column]
        observed = values.notna()
        positive = pd.to_numeric(values, errors="coerce").gt(0)
        records.append({"column": column, "new_column": column not in SOCCOL_FEATURE_COLUMNS,
                        "observed_rows": int(observed.sum()), "unique_values": int(values.nunique()),
                        "observed_folds": len(set(study.folds[observed])),
                        "observed_sources": len(set(study.source_groups[observed])),
                        "positive_rows": int(positive.sum()),
                        "positive_sources": len(set(study.source_groups[positive]))})
    return pd.DataFrame(records)


def summarize(rows):
    """Use Soccol's existing best formula as the descriptive reference."""
    frame = pd.DataFrame(rows)
    result = frame.groupby("name", sort=False).agg(
        mean_spearman=("spearman", "mean"), sd_fold_spearman=("spearman", lambda x: float(np.std(x))),
        minimum_fold_spearman=("spearman", "min"), maximum_fold_spearman=("spearman", "max"),
        mean_mae_mV=("mae_mV", "mean"), mean_terms=("terms", "mean")).reset_index()
    baseline = frame[frame.name.eq("existing_"+BASE_RULE)].set_index("fold").spearman
    result["gain_vs_current_best"] = result.mean_spearman - baseline.mean()
    result["folds_better_than_current"] = [int((group.set_index("fold").spearman.sort_index() > baseline.sort_index()).sum())
                                          for _, group in frame.groupby("name", sort=False)]
    return result.sort_values("mean_spearman", ascending=False).reset_index(drop=True)


def write_report(output, report, summaries, artifacts):
    selected = report["full_development_selected"]["name"]
    model = artifacts[selected]
    equation = "\n".join(f"{weight:+.6f} * z({name})" for name, weight in zip(model["term_names"], model["coefficients"]) if abs(weight)>1e-8)
    tables = "\n".join(f"<h2>{html.escape(label)}</h2>"+frame.round(5).to_html(index=False)
                       for label, frame in summaries.items())
    sources = "".join(f'<li><a href="{s["url"]}">{html.escape(s["topic"])}</a></li>' for s in SOURCES)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Soccol rule extensions</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 20px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;text-align:left}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:15px}}</style>
<h1>Soccol rule extensions</h1>
<p>Only the 2,950 accepted development rows were supplied to this study. All 805 final queries and 272 withheld development rows were excluded. These are development formula scores, not final-model results. Training formulas and sampling assets were not changed.</p>
<p>{len(report['candidate_specs'])} predeclared candidates: nine current rules, three tungsten updates, fourteen extensions, and two new equations. The current nine baseline scores were reproduced. The strongest existing development rule is the fixed anchor for extensions; it was previously selected on these development rows, so the study is exploratory.</p>
<p>The EPIT nested procedure selects regularization and formula inside each outer development fold. Filling values, scaling, category support, and base-rule coefficients use fitting rows only. The nested_new_and_existing row evaluates candidate selection; nested_existing_only applies the same selector to the original nine rules. Individual candidate scores can be optimistic when used to pick a winner.</p>
<p>The source_publication_blocked_folds diagnostic uses the same accepted development rows grouped by publication. It never replaces the frozen final-test partition. Uncertainty intervals resample held-out publication groups conditional on fitted models; they do not include research or fitting instability.</p>
{tables}
<h2>Full-development candidate: {html.escape(selected)}</h2><pre>{html.escape(equation)}</pre>
<p>Numeric terms use fitting-row median filling and standardization. Missing/unseen categories receive the mean known-category contribution. No missingness indicator, source identity, reference electrode label, or target-derived input is a predictor. Heat/solution fields with insufficient support are excluded within each fitting split. Category corrections and treatment-window terms are empirical hypotheses, not universal physical laws.</p>
<p>Positive bounds apply to base scores, PREN, Cr, N, log_Mo and log_W; temperature is constrained nonpositive. Other terms are regularized with unrestricted signs. The fitted score has a context-fitted affine mV calibration only for reporting MAE. Full-development coefficients are future-training candidates, never validation predictions.</p>
<h2>Feature availability</h2>{pd.read_csv(output/'feature_coverage.csv').to_html(index=False)}
<h2>Literature motivating the candidates</h2><ul>{sources}</ul>
<p>Detailed split identities, candidate definitions, coefficients, fold scores, exclusions and held-out predictions are stored alongside this report. Literature supports considering these effects, not the transferability of fitted Soccol coefficients.</p></html>'''
    (output/"report.html").write_text(content)


def run(output, bootstrap_samples=500):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory for formula research.")
    development = load_formula_data()
    study = SoccolFormulaStudy(development)
    output.mkdir(parents=True, exist_ok=True)
    coverage(study).to_csv(output/"feature_coverage.csv", index=False)
    report = {"schema_version": "soccol_formula_extension_study_v1", "completed": False,
              "development_selection": development.attrs["soccol_development_selection"],
              "final_test_targets_used": False, "candidate_specs": [asdict(s) for s in study.specs.values()],
              "feature_definitions": study.definitions, "sources": SOURCES,
              "penalties": list(PENALTIES), "selection_tolerance": SELECTION_TOLERANCE,
              "limitations": ["Exploratory study: the base rule was chosen using previous development scores.",
                              "Category and source associations do not establish causal material laws.",
                              "Positive tungsten/copper rows are sparse; heat measurements are sparse.",
                              "Reference electrode labels are excluded; no reference conversion is inferred."],
              "implementation_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in (
                  Path(__file__), REPO_ROOT/'scripts/epit_pipeline/formula_extension_study.py',
                  REPO_ROOT/'scripts/epit_pipeline/discover_formula_extensions.py', Path(old.__file__))}}
    historical = json.loads((REPO_ROOT/'corrosion_datasets/analysis/soccol_pipeline/target_rules_v2/calibration_summary.json').read_text())
    source_folds = np.zeros(len(study.y), dtype=int)
    for fold, (_, val) in enumerate(GroupKFold(5).split(study.frame, groups=study.source_groups), 1):
        source_folds[val] = fold
    summaries = {}
    for label, folds in (("frozen_development_folds", study.folds), ("source_publication_blocked_folds", source_folds)):
        print(label, flush=True)
        rows, predictions, selections, fitted = nested_evaluation(study, folds)
        summary = summarize(rows)
        summaries[label] = summary
        if label == "frozen_development_folds":
            for previous in historical["rules"]:
                actual = summary.set_index("name").loc["existing_"+previous["rule_family"], "mean_spearman"]
                if not np.isclose(actual, previous["mean_fold_spearman"], atol=1e-10, rtol=0):
                    raise RuntimeError("Existing rule score changed: "+previous["rule_family"])
            report["all_nine_current_baselines_reproduced"] = True
        pd.DataFrame(rows).to_csv(output/(label+"_fold_metrics.csv"), index=False)
        pd.DataFrame(predictions).to_csv(output/(label+"_oof_predictions.csv"), index=False)
        summary.to_csv(output/(label+"_summary.csv"), index=False)
        coefficient_stability(fitted).to_csv(output/(label+"_coefficient_stability.csv"), index=False)
        report[label] = {"selection": selections,
                         "fold_assignments": dict(zip(map(str, study.rows), map(int, folds))),
                         **bootstrap_difference(predictions, study, study.source_groups, samples=bootstrap_samples)}
        (output/(label+"_fits.json")).write_text(json.dumps({f"{name}/fold{fold}": artifact for (name,fold),artifact in fitted.items()}, indent=2, allow_nan=False)+"\n")
        (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(summary.head(8).to_string(index=False), flush=True)
        study.old_cache.clear(); study.design_cache.clear(); study.model_cache.clear()
    artifacts, selections, selected = full_development_fits(study)
    folder = output/"full_development_formulas"
    folder.mkdir()
    for name, artifact in artifacts.items():
        artifact.update(purpose="Research coefficient candidate; not used for development evaluation or production training.",
                        development_selection=development.attrs["soccol_development_selection"])
        (folder/(name+".json")).write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n")
    report.update(full_development_selected=selected, full_development_selection_scores=selections, completed=True)
    (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_report(output, report, summaries, artifacts)
    print("COMPLETE", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    with threadpool_limits(limits=1):
        run(args.output_dir, args.bootstrap_samples)


if __name__ == "__main__":
    main()
