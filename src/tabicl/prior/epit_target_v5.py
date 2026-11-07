"""EPIT final-training target experiment; original generic/Optuna paths stay intact.

Add the development-fitted heat/cooling equation to the nine v4 equations and
draw one formula/SCM weight per table, preserving the configured mean weight.
Only this opt-in dataset class uses the new target policy.
"""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
import torch

from .epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS, EPIT_EXPANDED_FEATURE_PROFILE
from .epit_feature_profile import load_epit_feature_profile
from .paired_streams import keyed_seed
from .target_mixing import (
    VariableTargetMixture, _VariableTargetMixPrior,
    mix_bounds, mix_policy_identity, table_mix_weight,
)


VERSION = "epit_target_v5_heat_categories_variable_mix"
ASSET_NAME = "epit_target_v5.json"


def load_recipe():
    payload = json.loads(resources.files("tabicl.prior.assets").joinpath(ASSET_NAME).read_text())
    if payload["schema_version"] != VERSION or payload["final_test_targets_used"] is not False:
        raise ValueError("Invalid EPIT v5 formula provenance.")
    profile = load_epit_feature_profile(EPIT_EXPANDED_FEATURE_PROFILE)
    if payload["category_mappings"] != profile.metadata["category_mappings"]:
        raise ValueError("EPIT v5 category codes differ from the empirical feature profile.")
    if payload["test_method_categories"] != list(profile.test_method_categories):
        raise ValueError("EPIT v5 test-method codes differ from the empirical feature profile.")
    if set(payload["formula"]["fitting_task_row_indices"]) != set(profile.source_task_row_indices.tolist()):
        raise ValueError("EPIT v5 formula was fitted on different development rows.")
    return payload


def recipe_identity(mean):
    payload = load_recipe()
    return {
        "version": VERSION,
        "artifact": ASSET_NAME,
        "artifact_sha256": hashlib.sha256(resources.files("tabicl.prior.assets").joinpath(ASSET_NAME).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_formula": payload["formula"]["name"],
        "new_formula_development_score": payload["mean_fold_spearman"],
        "formula_selection": "v4 score-weighted families plus extend_heat_categories",
        "target_mix": mix_policy_identity(mean),
        "coefficient_variation": "positive multiplicative factors on fitted extension coefficients; base fit fixed",
        "final_test_targets_used": False,
    }


def heat_formula_drive(X, method_codes, recipe, coefficients=None):
    """Evaluate the saved equation on raw 70-column inputs, before imputation/shuffling.

    Reuses *fitting-row* scales and method offsets, exactly as in the formula
    study. Unknown heat/cooling labels contribute zero, including missing ones.
    This returns the rank score; its mV calibration is not a training target.
    """
    if X.ndim != 2 or X.shape[1] != len(EPIT_EXPANDED_FEATURE_COLUMNS):
        raise ValueError("EPIT v5 requires the original ordered 70 physical columns.")
    model = recipe["formula"]
    base = model["base"]
    state = base["feature_state"]
    physical = state["base_rule_state"]

    def column(name):
        return X[:, EPIT_EXPANDED_FEATURE_COLUMNS.index(name)]

    def imputed(name, old_name):
        value = column(name)
        return torch.where(torch.isfinite(value), value, physical["imputation_means"][old_name])

    def scale(value, fitted):
        return (value - fitted["mean"]) / fitted["std"]

    cr, mo, w, nitrogen = [imputed(name, "Composition, wt.% " + name).clamp_min(0)
                           for name in ("Cr", "Mo", "W", "N")]
    mow = mo + 0.5 * w
    material = cr + 3.3 * mow + 16 * nitrogen + torch.sqrt(cr * mow)
    signals = physical["signal_scaling"]
    material_z = scale(material, signals["material_score"])
    temperature_z = scale(imputed("CP_temp", "Test Temp. oC"), signals["temperature"])
    chloride_z = scale(torch.log10(imputed("CP_Cl", "[Cl-] M").clamp_min(1e-12)), signals["chloride_log10"])
    ph_z = scale(torch.abs(imputed("CP_pH", "[Cl-] pH") - 7.25), signals["ph_distance"])
    environment_z = scale(0.075 * temperature_z + 0.925 * chloride_z + 0.115 * ph_z,
                          signals["environment_score"])
    raw_terms = (
        torch.tanh(material_z), -torch.sigmoid(environment_z),
        -torch.sigmoid(-material_z) * torch.sigmoid(chloride_z),
        -torch.sigmoid(temperature_z) * torch.sigmoid(chloride_z),
    )
    terms = [scale(value, physical["term_scaling"][name])
             for name, value in zip(base["term_names"][:4], raw_terms)]
    offsets = X.new_tensor(recipe["method_offsets_by_code"])
    if method_codes.shape != (len(X),) or not torch.isfinite(method_codes).all():
        raise ValueError("Invalid EPIT test-method codes.")
    labels = method_codes.long()
    if not torch.equal(labels.to(method_codes.dtype), method_codes) or torch.any((labels < 0) | (labels >= len(offsets))):
        raise ValueError("Invalid EPIT test-method codes.")
    terms.append(scale(offsets[labels], state["method_scaling"]))
    base_score = torch.stack(terms, dim=1) @ X.new_tensor(base["coefficients"])
    extension = [(base_score - model["base_center"]) / model["base_scale"]]
    for term in model["feature_state"]:
        codes = column(term["source"])
        mapping = recipe["category_mappings"][term["source"]]
        known = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        for level in term["levels"]:
            known |= codes == mapping[level]
        value = (codes == mapping[term["level"]]).to(X.dtype)
        extension.append(torch.where(known, (value - term["center"]) / term["scale"], 0.0))
    weights = X.new_tensor(model["coefficients"] if coefficients is None else coefficients)
    result = torch.stack(extension, dim=1) @ weights
    if not torch.isfinite(result).all():
        raise ValueError("EPIT v5 generated a non-finite formula score.")
    return result


class _EpitV5Prior(_VariableTargetMixPrior):
    target_policy = VERSION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_recipe = load_recipe()
        if self.fixed_hp.get("pitting_feature_profile") != EPIT_EXPANDED_FEATURE_PROFILE or self.fixed_hp.get("pitting_composition_mode") != "empirical_features_scm_target":
            raise ValueError("EPIT v5 is only supported for the reviewed EPIT SCM-target prior.")
        mix_bounds(self.fixed_hp.get("informed_target_mix_weight", 0.35))

    def _formula_drive_and_rule(self, X, params):
        seed = int(params["paired_task_seed"])
        _, scores, _ = self._fixed_epit_target_rule_distribution()
        new_score = float(self.target_recipe["mean_fold_spearman"])
        probability = new_score / (sum(scores.values()) + new_score)
        use_heat = np.random.default_rng(keyed_seed(seed, 52)).random() < probability
        rule = dict(self.last_pitting_target_rule)
        if use_heat:
            coefficients = np.asarray(self.target_recipe["formula"]["coefficients"])
            if self.coefficient_variation:
                rng = np.random.default_rng(keyed_seed(seed, 53, self.fixed_hp.get("pitting_coefficient_variation_seed", 42)))
                coefficients = coefficients * rng.uniform(1 - self.coefficient_variation, 1 + self.coefficient_variation, len(coefficients))
            batch = self.last_pitting_feature_batch
            raw = torch.as_tensor(batch.features.copy(), device=X.device, dtype=torch.float64)
            codes = torch.as_tensor(batch.test_method_codes.copy(), device=X.device)
            drive = heat_formula_drive(raw, codes, self.target_recipe, coefficients).to(X.dtype)
            rule.update(rule_type=VERSION, target_rule_family="extend_heat_categories",
                        target_rule_coefficients=dict(zip(self.target_recipe["formula"]["term_names"], coefficients.tolist())))
        else:
            drive = self.last_pitting_target_drive.to(device=X.device, dtype=X.dtype)
        all_scores = {**scores, "extend_heat_categories": new_score}
        all_probabilities = {name: value / sum(all_scores.values()) for name, value in all_scores.items()}
        family = rule["target_rule_family"]
        rule.update(target_recipe=VERSION,
                    target_rule_scores=all_scores, target_rule_probabilities=all_probabilities,
                    target_rule_score=all_scores[family], target_rule_probability=all_probabilities[family])
        return drive, rule


class EpitV5PriorMixture(VariableTargetMixture):
    """Original generic stream with the opt-in EPIT v5 informed target."""
    informed_prior_class = _EpitV5Prior

    def __init__(self, **kwargs):
        fixed = kwargs.get("scm_fixed_hp") or {}
        if fixed.get("pitting_feature_profile") != EPIT_EXPANDED_FEATURE_PROFILE:
            raise ValueError("EPIT v5 cannot be used for another dataset's prior.")
        super().__init__(**kwargs)
        # Hashed artifact and runtime implementation are part of resume identity.
        self._config["epit_target_recipe"] = recipe_identity(fixed.get("informed_target_mix_weight", 0.35))
