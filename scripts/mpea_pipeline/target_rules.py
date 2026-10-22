"""Explicit, reduced EPIT/Soccol hypotheses for the recorded MPEA inputs.

These are development candidates, not additions to either production registry.
Absent N, S, pH and extra ions are omitted, never filled with physical zeros.
Temperature has only four observations in the frozen development partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.epit_pipeline.split_data import EpitDataset
from scripts.epit_pipeline.target_rules import (
    RuleTermData, TargetRuleFamily, _apply_scale, _fit_scale, _sigmoid,
)


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    source: str
    kind: str
    mo_weight: float
    w_fraction: float

    @property
    def columns(self) -> tuple[str, ...]:
        return ("Cr", "Mo", "W") if self.w_fraction else ("Cr", "Mo")

    @property
    def formula(self) -> str:
        q = f"(Mo + {self.w_fraction:g}*W)" if self.w_fraction else "Mo"
        if self.kind == "threshold":
            return f"sigmoid((Cr-12)/2) + log1p({q})"
        linear = f"Cr + {self.mo_weight:g}*{q}"
        return linear + (f" + sqrt(Cr*{q})" if self.kind == "synergy" else "")

    def score(self, values: dict[str, np.ndarray]) -> np.ndarray:
        cr = np.clip(values["Cr"], 0.0, None)
        q = np.clip(values["Mo"] + (self.w_fraction * values["W"] if self.w_fraction else 0.0), 0.0, None)
        if self.kind == "threshold":
            return _sigmoid((cr - 12.0) / 2.0) + np.log1p(q)
        score = cr + self.mo_weight * q
        return score + (np.sqrt(cr * q) if self.kind == "synergy" else 0.0)


MATERIALS = {
    f"{source}_{kind}": MaterialSpec(f"{source}_{kind}", source, kind, mo, w)
    for source, tungsten in (("epit", True), ("soccol", False))
    for kind, mo, w in (
        ("pren", 3.3, 0.5 if tungsten else 0.0),
        ("synergy", 3.3, 0.5 if tungsten else 0.0),
        ("threshold", 3.25, 0.55 if tungsten else 0.0),
        ("coupled_pren", 3.25, 0.55 if tungsten else 0.0),
    )
}


class MpeaReducedRule(TargetRuleFamily):
    applicable_material_classes = None

    def __init__(self, material: MaterialSpec, environment: str, source_rules: tuple[str, ...] = ()):
        self.material = material
        self.environment = environment
        self.source_rules = source_rules
        self.name = f"{material.name}_{environment}"
        self.description = f"{material.source.upper()} {material.kind} hypothesis; {environment}"
        self.evaluation_role = "composition-only diagnostic" if environment == "material_only" else "MPEA development candidate"
        if environment == "interaction":
            self.term_names = ("material_passivity", "chloride_aggressiveness", "material_chloride_interaction")
            anchor, bounds = [0.575, 0.50, 0.775], [1.0, 1.0, 1.0]
        elif environment == "chloride":
            self.term_names = ("material_passivity", "chloride_aggressiveness")
            anchor, bounds = [0.42, 0.24], [1.0, 0.65]
        elif environment == "coupled":
            self.term_names = ("material_passivity", "coupled_chloride_breakdown")
            anchor, bounds = [0.55, 0.40], [1.0, 0.80]
        elif environment == "material_only":
            self.term_names = ("material_passivity",)
            anchor, bounds = [1.0], [1.0]
        else:
            raise ValueError(f"Unknown environment {environment}")
        self.coefficient_anchor = np.asarray(anchor, dtype=float) / sum(anchor)
        self.coefficient_upper_bounds = np.asarray(bounds, dtype=float)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.material.source,
            "source_rules": list(self.source_rules),
            "material_formula": self.material.formula,
            "environment": self.environment,
            "signed_terms": {
                "material_passivity": "tanh(z(material))",
                "chloride_aggressiveness": "-sigmoid(z(log10(max(CP_Cl,1e-12))))",
                "material_chloride_interaction": "-sigmoid(-z(material))*sigmoid(z(log10(max(CP_Cl,1e-12))))",
                "coupled_chloride_breakdown": "-sigmoid(0.65*z(log10(max(CP_Cl,1e-12)))-0.75*z(material))",
            },
            "required_columns": list(self.required_columns),
            "adaptation": "Omit unrecorded N/S/pH/extra-ion/method effects and very sparse temperature effects; no physical-zero substitutions.",
            "scope": "Borrowed Fe/Ni-alloy hypotheses tested on MPEA development rows; applicability to MPEA is being tested, not assumed.",
            "missing_chloride": "Mean of observed chloride in the current context only, before log transform, matching EPIT numeric imputation.",
            "scaling": "Context-only standardization of material/log-chloride, followed by context-only standardization of each signed term.",
            "bounds": "Inherited bounds for retained terms; anchors renormalized after removing terms.",
        }

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self.material.columns + (() if self.environment == "material_only" else ("CP_Cl",))

    def _inputs(self, dataset: EpitDataset, indices: np.ndarray, means: dict[str, float]) -> dict[str, np.ndarray]:
        values = {}
        for column in self.required_columns:
            raw = np.asarray([dataset.rows[int(i)][column] for i in indices], dtype=float)
            values[column] = np.where(np.isfinite(raw), raw, means[column])
        return values

    def _raw_terms(self, dataset: EpitDataset, indices: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        values = self._inputs(dataset, indices, state["imputation_means"])
        material_z = _apply_scale(self.material.score(values), state["signal_scaling"]["material"])
        terms = [np.tanh(material_z)]
        if self.environment != "material_only":
            chloride_z = _apply_scale(np.log10(np.clip(values["CP_Cl"], 1e-12, None)), state["signal_scaling"]["chloride_log10"])
            if self.environment == "coupled":
                terms.append(-_sigmoid(0.65 * chloride_z - 0.75 * material_z))
            else:
                terms.append(-_sigmoid(chloride_z))
                if self.environment == "interaction":
                    terms.append(-_sigmoid(-material_z) * _sigmoid(chloride_z))
        return np.column_stack(terms)

    def fit_terms(self, dataset: EpitDataset, row_indices: np.ndarray) -> RuleTermData:
        means = {}
        for column in self.required_columns:
            values = np.asarray([dataset.rows[int(i)][column] for i in row_indices], dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"{self.name}: no observed context values for {column}")
            means[column] = float(finite.mean())
        inputs = self._inputs(dataset, row_indices, means)
        scaling = {"material": _fit_scale(self.material.score(inputs))}
        if self.environment != "material_only":
            scaling["chloride_log10"] = _fit_scale(np.log10(np.clip(inputs["CP_Cl"], 1e-12, None)))
        state = {"imputation_means": means, "signal_scaling": scaling}
        raw = self._raw_terms(dataset, row_indices, state)
        state["term_scaling"] = {name: _fit_scale(raw[:, i]) for i, name in enumerate(self.term_names)}
        return RuleTermData(self.transform_terms(dataset, row_indices, state), state)

    def transform_terms(self, dataset: EpitDataset, row_indices: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        raw = self._raw_terms(dataset, row_indices, state)
        values = np.column_stack([_apply_scale(raw[:, i], state["term_scaling"][name]) for i, name in enumerate(self.term_names)])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite terms for {self.name}")
        return values


def get_mpea_rules() -> list[MpeaReducedRule]:
    rules = []
    for source in ("epit", "soccol"):
        for material, environment, original in (
            ("pren", "interaction", ("pren_n_linear",)),
            ("synergy", "interaction", ("cr_mow_n_synergy", "method_aware_pren_n") if source == "epit" else ("cr_mow_n_synergy",)),
            ("threshold", "interaction", ("threshold_saturation",)),
            ("pren", "chloride", ("pren_n_improved_environment", "mns_inclusion_penalty", "mo_n_acid_repassivation")),
            ("coupled_pren", "coupled", ("coupled_breakdown",)),
        ):
            rules.append(MpeaReducedRule(MATERIALS[f"{source}_{material}"], environment, original))
    rules.append(MpeaReducedRule(MATERIALS["soccol_pren"], "coupled", ("pren_n_coupled_mns", "pren_n_coupled_mns_weak_anions")))
    rules.extend(MpeaReducedRule(spec, "material_only") for spec in MATERIALS.values())
    return rules
