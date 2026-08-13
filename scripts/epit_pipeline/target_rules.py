"""Common interface for calibratable EPIT target-rule families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata

from scripts.epit_pipeline.split_data import EpitDataset, _method_family, _numeric


@dataclass(frozen=True)
class RuleTermData:
    """Signed, standardized terms used by a target-rule calibration."""

    values: np.ndarray
    state: dict[str, Any]


class TargetRuleFamily(ABC):
    """A material/environment hypothesis that can be calibrated consistently."""

    name: str
    description: str
    term_names: tuple[str, ...]
    coefficient_anchor: np.ndarray
    coefficient_upper_bounds: np.ndarray | None = None
    applicable_material_classes: tuple[str, ...] | None = None
    evaluation_role = "candidate"

    def eligible_rows(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> np.ndarray:
        """Limit a chemistry-specific rule to alloy classes where it is valid."""
        row_indices = np.asarray(row_indices, dtype=int)
        if self.applicable_material_classes is None:
            return row_indices
        allowed = set(self.applicable_material_classes)
        return np.asarray(
            [
                int(index)
                for index in row_indices
                if str(
                    dataset.rows[int(index)].get("Material class") or ""
                ).strip()
                in allowed
            ],
            dtype=int,
        )

    @property
    def applicability(self) -> str:
        if self.applicable_material_classes is None:
            return "all alloy classes"
        return ", ".join(self.applicable_material_classes)

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return a JSON-safe description of the fixed rule formula."""

    @abstractmethod
    def fit_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> RuleTermData:
        """Fit context-only preprocessing and return calibration terms."""

    @abstractmethod
    def transform_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        """Apply training-row preprocessing to new rows."""


def _fit_scale(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise RuntimeError("Cannot standardize non-finite target-rule values.")
    if std <= 1e-12:
        std = 1.0
    return {"mean": mean, "std": std}


def _apply_scale(values: np.ndarray, scale: dict[str, Any]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(scale["mean"])) / float(
        scale["std"]
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


class CurrentPrenRule(TargetRuleFamily):
    """Deterministic center of the project's current sampled PREN-like rule."""

    name = "current_pren"
    description = "Current PREN-like material, environment and interaction rule"
    evaluation_role = "historical all-alloy baseline"
    term_names = (
        "material_passivity",
        "environment_aggressiveness",
        "material_chloride_interaction",
    )

    # Current code defaults: material=0.575, environment=0.50, interaction=0.775.
    coefficient_anchor = np.asarray([0.575, 0.50, 0.775], dtype=float)
    coefficient_anchor /= coefficient_anchor.sum()

    material_columns = (
        "Composition, wt.% Cr",
        "Composition, wt.% Ni",
        "Composition, wt.% Mo",
        "Composition, wt.% W",
    )
    material_weights = np.asarray([1.0, 0.25, 3.3, 1.65], dtype=float)
    environment_columns = ("Test Temp. oC", "[Cl-] M", "[Cl-] pH")
    # Midpoints of the ranges sampled by the current implementation.
    environment_weights = np.asarray([0.075, 0.925, 0.115], dtype=float)
    ph_neutral = 7.25
    include_temperature_chloride_interaction = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "material_formula": "Cr + 0.25*Ni + 3.3*Mo + 1.65*W",
            "environment_formula": (
                "standardize(0.075*z(temperature) + "
                "0.925*z(log10(chloride)) + 0.115*z(abs(pH-7.25)))"
            ),
            "interaction_formula": (
                "sigmoid(-z(material))*sigmoid(z(log10(chloride)))"
            ),
            "signed_terms": [
                ("+" if index == 0 else "-") + name
                for index, name in enumerate(self.term_names)
            ],
            "process_effect": (
                "excluded from calibration because the current synthetic rule "
                "samples random method offsets for every task"
            ),
            "fixed_internal_material_weights": self.material_weights.tolist(),
            "fixed_internal_environment_weights": self.environment_weights.tolist(),
            "fixed_ph_neutral": self.ph_neutral,
            "applicability": self.applicability,
        }

    def _material_score(self, material: np.ndarray) -> np.ndarray:
        return material @ self.material_weights

    def _fit_imputation(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> dict[str, float]:
        columns = self.material_columns + self.environment_columns
        imputation: dict[str, float] = {}
        for column in columns:
            values = np.asarray(
                [
                    _numeric(dataset.table, dataset.rows[int(index)], column)
                    for index in row_indices
                ],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise RuntimeError(
                    f"Cannot calibrate {self.name}: {column!r} is entirely missing."
                )
            imputation[column] = float(np.mean(finite))
        return imputation

    @staticmethod
    def _column_values(
        dataset: EpitDataset,
        row_indices: np.ndarray,
        column: str,
        imputation: dict[str, Any],
    ) -> np.ndarray:
        values = np.asarray(
            [
                _numeric(dataset.table, dataset.rows[int(index)], column)
                for index in row_indices
            ],
            dtype=float,
        )
        return np.where(np.isfinite(values), values, float(imputation[column]))

    def _raw_inputs(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        imputation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        material = np.column_stack(
            [
                self._column_values(dataset, row_indices, column, imputation)
                for column in self.material_columns
            ]
        )
        temperature = self._column_values(
            dataset, row_indices, self.environment_columns[0], imputation
        )
        chloride = self._column_values(
            dataset, row_indices, self.environment_columns[1], imputation
        )
        ph = self._column_values(
            dataset, row_indices, self.environment_columns[2], imputation
        )
        return material, temperature, chloride, ph

    def _unscaled_signed_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        material, temperature, chloride, ph = self._raw_inputs(
            dataset, row_indices, state["imputation_means"]
        )
        scaling = state["signal_scaling"]

        material_score = self._material_score(material)
        material_z = _apply_scale(material_score, scaling["material_score"])
        material_passivity = np.tanh(material_z)
        material_susceptibility = _sigmoid(-material_z)

        temperature_z = _apply_scale(temperature, scaling["temperature"])
        chloride_log = np.log10(np.clip(chloride, 1e-12, None))
        chloride_z = _apply_scale(chloride_log, scaling["chloride_log10"])
        ph_distance = np.abs(ph - self.ph_neutral)
        ph_z = _apply_scale(ph_distance, scaling["ph_distance"])
        environment_score = (
            self.environment_weights[0] * temperature_z
            + self.environment_weights[1] * chloride_z
            + self.environment_weights[2] * ph_z
        )
        environment_z = _apply_scale(
            environment_score, scaling["environment_score"]
        )
        environment_aggressiveness = _sigmoid(environment_z)
        chloride_aggressiveness = _sigmoid(chloride_z)
        interaction = material_susceptibility * chloride_aggressiveness

        terms = [
            material_passivity,
            -environment_aggressiveness,
            -interaction,
        ]
        if self.include_temperature_chloride_interaction:
            terms.append(-_sigmoid(temperature_z) * chloride_aggressiveness)
        return np.column_stack(terms)

    def fit_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> RuleTermData:
        row_indices = np.asarray(row_indices, dtype=int)
        imputation = self._fit_imputation(dataset, row_indices)
        material, temperature, chloride, ph = self._raw_inputs(
            dataset, row_indices, imputation
        )
        material_score = self._material_score(material)
        chloride_log = np.log10(np.clip(chloride, 1e-12, None))
        ph_distance = np.abs(ph - self.ph_neutral)

        signal_scaling: dict[str, dict[str, float]] = {
            "material_score": _fit_scale(material_score),
            "temperature": _fit_scale(temperature),
            "chloride_log10": _fit_scale(chloride_log),
            "ph_distance": _fit_scale(ph_distance),
        }
        environment_score = (
            self.environment_weights[0]
            * _apply_scale(temperature, signal_scaling["temperature"])
            + self.environment_weights[1]
            * _apply_scale(chloride_log, signal_scaling["chloride_log10"])
            + self.environment_weights[2]
            * _apply_scale(ph_distance, signal_scaling["ph_distance"])
        )
        signal_scaling["environment_score"] = _fit_scale(environment_score)

        state: dict[str, Any] = {
            "imputation_means": imputation,
            "signal_scaling": signal_scaling,
        }
        raw_terms = self._unscaled_signed_terms(dataset, row_indices, state)
        term_scaling = {
            term_name: _fit_scale(raw_terms[:, term_index])
            for term_index, term_name in enumerate(self.term_names)
        }
        state["term_scaling"] = term_scaling
        values = self.transform_terms(dataset, row_indices, state)
        return RuleTermData(values=values, state=state)

    def transform_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        raw_terms = self._unscaled_signed_terms(
            dataset, np.asarray(row_indices, dtype=int), state
        )
        values = np.column_stack(
            [
                _apply_scale(raw_terms[:, index], state["term_scaling"][name])
                for index, name in enumerate(self.term_names)
            ]
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.name} generated non-finite rule terms.")
        return values


class LinearPrenRule(CurrentPrenRule):
    """PREN without the unsupported Ni contribution or minor-element noise."""

    name = "pren_linear"
    description = "Linear Cr-Mo-W PREN family for Fe and Ni-Cr-Mo alloys"
    evaluation_role = "candidate"
    applicable_material_classes = ("Fe Alloy", "NiCrMo Alloy")
    # Midpoints of approved a_M=[2.5, 4.0] and eta_W=[0.3, 0.8].
    a_m = 3.25
    eta_w = 0.55
    material_weights = np.asarray([1.0, 0.0, a_m, a_m * eta_w])

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata.update(
            {
                "material_formula": "Cr + 3.25*(Mo + 0.55*W)",
                "synthetic_parameter_ranges": {
                    "a_M": [2.5, 4.0],
                    "eta_W": [0.3, 0.8],
                },
                "removed_terms": [
                    "Ni contribution",
                    "Gaussian noise on unrelated minor-element weights",
                ],
                "applicability": self.applicability,
            }
        )
        return metadata


class CrMoWSynergyRule(LinearPrenRule):
    """Linear PREN plus a cooperative Cr-Mo/W contribution."""

    name = "cr_mow_synergy"
    description = "Cr-Mo/W synergy rule for Fe and Ni-Cr-Mo alloys"
    term_names = (
        "material_passivity",
        "environment_aggressiveness",
        "material_chloride_interaction",
        "temperature_chloride_interaction",
    )
    coefficient_anchor = np.asarray([0.575, 0.50, 0.775, 0.20], dtype=float)
    coefficient_anchor /= coefficient_anchor.sum()
    include_temperature_chloride_interaction = True
    synergy_strength = 1.0

    def _material_score(self, material: np.ndarray) -> np.ndarray:
        chromium = np.clip(material[:, 0], 0.0, None)
        q = np.clip(material[:, 2] + self.eta_w * material[:, 3], 0.0, None)
        linear = chromium + self.a_m * q
        return linear + self.synergy_strength * np.sqrt(chromium * q)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata.update(
            {
                "material_formula": (
                    "Cr + 3.25*(Mo + 0.55*W) + "
                    "sqrt(Cr*(Mo + 0.55*W))"
                ),
                "fixed_direct_evaluation_synergy_strength": self.synergy_strength,
                "additional_interaction": (
                    "-sigmoid(z(temperature))*sigmoid(z(log10(chloride)))"
                ),
                "applicability": self.applicability,
            }
        )
        return metadata


class ThresholdSaturationRule(LinearPrenRule):
    """Cr passivation threshold plus diminishing Mo/W returns."""

    name = "threshold_saturation"
    description = "Threshold and saturation rule for Fe and Ni-Cr-Mo alloys"
    term_names = CrMoWSynergyRule.term_names
    coefficient_anchor = CrMoWSynergyRule.coefficient_anchor.copy()
    include_temperature_chloride_interaction = True
    cr_threshold = 12.0
    cr_slope = 2.0
    q_scale = 1.0

    def _material_score(self, material: np.ndarray) -> np.ndarray:
        chromium = np.clip(material[:, 0], 0.0, None)
        q = np.clip(material[:, 2] + self.eta_w * material[:, 3], 0.0, None)
        cr_passivation = _sigmoid(
            (chromium - self.cr_threshold) / self.cr_slope
        )
        mow_saturation = np.log1p(q / self.q_scale)
        return cr_passivation + mow_saturation

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata.update(
            {
                "material_formula": (
                    "sigmoid((Cr-12)/2) + "
                    "log1p((Mo + 0.55*W)/1.0)"
                ),
                "fixed_direct_evaluation_parameters": {
                    "Cr_threshold_wt_percent": self.cr_threshold,
                    "Cr_slope_wt_percent": self.cr_slope,
                    "MoW_saturation_scale_wt_percent": self.q_scale,
                },
                "synthetic_use": (
                    "threshold, slope and saturation scale vary once per "
                    "synthetic task around these centers"
                ),
                "additional_interaction": (
                    "-sigmoid(z(temperature))*sigmoid(z(log10(chloride)))"
                ),
                "applicability": self.applicability,
            }
        )
        return metadata


class ImprovedEnvironmentRule(LinearPrenRule):
    """Separate, corrosion-motivated Fe/Ni environmental penalties."""

    name = "improved_environment"
    description = (
        "Log-chloride, high-temperature, temperature-chloride and acidic-pH rule"
    )
    term_names = (
        "material_passivity",
        "log_chloride_aggressiveness",
        "high_temperature_aggressiveness",
        "temperature_chloride_interaction",
        "acidic_ph_aggressiveness",
    )
    coefficient_anchor = np.asarray([0.42, 0.24, 0.14, 0.14, 0.06], dtype=float)
    coefficient_upper_bounds = np.asarray([1.0, 0.65, 0.45, 0.45, 0.15])
    temperature_threshold = 50.0
    temperature_slope = 10.0
    acidic_ph_threshold = 6.5
    acidic_ph_slope = 1.0

    def _material_score_for_rows(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        material: np.ndarray,
    ) -> np.ndarray:
        del dataset, row_indices
        return self._material_score(material)

    def _fit_rule_state(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> dict[str, Any]:
        imputation = self._fit_imputation(dataset, row_indices)
        material, temperature, chloride, ph = self._raw_inputs(
            dataset, row_indices, imputation
        )
        material_score = self._material_score_for_rows(
            dataset, row_indices, material
        )
        chloride_log = np.log10(np.clip(chloride, 1e-12, None))
        temperature_signal = _sigmoid(
            (temperature - self.temperature_threshold) / self.temperature_slope
        )
        acidic_ph_signal = _sigmoid(
            (self.acidic_ph_threshold - ph) / self.acidic_ph_slope
        )
        signal_scaling: dict[str, dict[str, float]] = {
            "material_score": _fit_scale(material_score),
            "chloride_log10": _fit_scale(chloride_log),
            "temperature_threshold_signal": _fit_scale(temperature_signal),
            "acidic_ph_signal": _fit_scale(acidic_ph_signal),
        }
        chloride_signal = _sigmoid(
            _apply_scale(chloride_log, signal_scaling["chloride_log10"])
        )
        signal_scaling["temperature_chloride_signal"] = _fit_scale(
            temperature_signal * chloride_signal
        )
        return {
            "imputation_means": imputation,
            "signal_scaling": signal_scaling,
        }

    def _unscaled_signed_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        material, temperature, chloride, ph = self._raw_inputs(
            dataset, row_indices, state["imputation_means"]
        )
        scaling = state["signal_scaling"]
        material_score = self._material_score_for_rows(
            dataset, row_indices, material
        )
        material_passivity = np.tanh(
            _apply_scale(material_score, scaling["material_score"])
        )
        chloride_log = np.log10(np.clip(chloride, 1e-12, None))
        chloride_signal = _sigmoid(
            _apply_scale(chloride_log, scaling["chloride_log10"])
        )
        temperature_signal = _sigmoid(
            (temperature - self.temperature_threshold) / self.temperature_slope
        )
        acidic_ph_signal = _sigmoid(
            (self.acidic_ph_threshold - ph) / self.acidic_ph_slope
        )
        return np.column_stack(
            [
                material_passivity,
                -chloride_signal,
                -temperature_signal,
                -(temperature_signal * chloride_signal),
                -acidic_ph_signal,
            ]
        )

    def fit_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> RuleTermData:
        row_indices = np.asarray(row_indices, dtype=int)
        state = self._fit_rule_state(dataset, row_indices)
        raw_terms = self._unscaled_signed_terms(dataset, row_indices, state)
        state["term_scaling"] = {
            name: _fit_scale(raw_terms[:, index])
            for index, name in enumerate(self.term_names)
        }
        return RuleTermData(
            values=self.transform_terms(dataset, row_indices, state),
            state=state,
        )

    def transform_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        raw_terms = self._unscaled_signed_terms(
            dataset, np.asarray(row_indices, dtype=int), state
        )
        values = np.column_stack(
            [
                _apply_scale(raw_terms[:, index], state["term_scaling"][name])
                for index, name in enumerate(self.term_names)
            ]
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.name} generated non-finite rule terms.")
        return values

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "material_formula": "Cr + 3.25*(Mo + 0.55*W)",
            "environment_formula": (
                "-log10(chloride) - sigmoid((temperature-50)/10) "
                "- temperature_signal*chloride_signal "
                "- sigmoid((6.5-pH)/1.0)"
            ),
            "fixed_direct_evaluation_parameters": {
                "temperature_threshold_C": self.temperature_threshold,
                "temperature_slope_C": self.temperature_slope,
                "acidic_pH_threshold": self.acidic_ph_threshold,
                "acidic_pH_slope": self.acidic_ph_slope,
            },
            "pH_design": "one-sided acidic penalty; no alkaline penalty",
            "applicability": self.applicability,
        }


class CoupledBreakdownRule(ImprovedEnvironmentRule):
    """Environmental breakdown becomes stronger as material protection falls."""

    name = "coupled_breakdown"
    description = "Environment-versus-material protection breakdown rule"
    term_names = (
        "material_passivity",
        "coupled_environment_breakdown",
        "acidic_ph_aggressiveness",
    )
    coefficient_anchor = np.asarray([0.55, 0.40, 0.05], dtype=float)
    coefficient_upper_bounds = np.asarray([1.0, 0.80, 0.15])
    protection_coupling = 0.75

    def _unscaled_signed_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        material, temperature, chloride, ph = self._raw_inputs(
            dataset, row_indices, state["imputation_means"]
        )
        scaling = state["signal_scaling"]
        material_score = self._material_score_for_rows(
            dataset, row_indices, material
        )
        material_z = _apply_scale(material_score, scaling["material_score"])
        material_passivity = np.tanh(material_z)

        chloride_log = np.log10(np.clip(chloride, 1e-12, None))
        chloride_z = _apply_scale(chloride_log, scaling["chloride_log10"])
        chloride_signal = _sigmoid(chloride_z)
        temperature_signal = _sigmoid(
            (temperature - self.temperature_threshold) / self.temperature_slope
        )
        temperature_z = _apply_scale(
            temperature_signal,
            scaling["temperature_threshold_signal"],
        )
        joint_z = _apply_scale(
            temperature_signal * chloride_signal,
            scaling["temperature_chloride_signal"],
        )
        aggressiveness = (
            0.65 * chloride_z + 0.25 * temperature_z + 0.10 * joint_z
        )
        coupled_breakdown = _sigmoid(
            aggressiveness - self.protection_coupling * material_z
        )
        acidic_ph_signal = _sigmoid(
            (self.acidic_ph_threshold - ph) / self.acidic_ph_slope
        )
        return np.column_stack(
            [material_passivity, -coupled_breakdown, -acidic_ph_signal]
        )

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata.update(
            {
                "coupled_breakdown_formula": (
                    "-sigmoid(0.65*z(log10(chloride)) + "
                    "0.25*z(high_temperature) + 0.10*z(joint_environment) "
                    "- 0.75*z(material))"
                ),
                "protection_coupling": self.protection_coupling,
                "double_counting_control": (
                    "one coupled breakdown term replaces separate generic "
                    "environment and material-chloride penalties"
                ),
            }
        )
        return metadata


class FeNiCrThresholdRule(ImprovedEnvironmentRule):
    """Separate smooth chromium passivation thresholds for Fe and Ni alloys."""

    name = "fe_ni_cr_threshold"
    description = "Fe/Ni-specific Cr threshold with the improved environment rule"
    fe_cr_threshold = 12.0
    ni_cr_threshold = 15.0
    cr_slope = 2.0
    q_scale = 1.0

    def _material_score_for_rows(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        material: np.ndarray,
    ) -> np.ndarray:
        classes = np.asarray(
            [
                str(dataset.rows[int(index)].get("Material class") or "").strip()
                for index in row_indices
            ]
        )
        thresholds = np.where(
            classes == "NiCrMo Alloy",
            self.ni_cr_threshold,
            self.fe_cr_threshold,
        )
        chromium = np.clip(material[:, 0], 0.0, None)
        q = np.clip(material[:, 2] + self.eta_w * material[:, 3], 0.0, None)
        cr_passivation = _sigmoid((chromium - thresholds) / self.cr_slope)
        mow_saturation = np.log1p(q / self.q_scale)
        return cr_passivation + mow_saturation

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata.update(
            {
                "material_formula": (
                    "sigmoid((Cr-threshold[class])/2) + "
                    "log1p((Mo + 0.55*W)/1.0)"
                ),
                "class_thresholds_wt_percent": {
                    "Fe Alloy": self.fe_cr_threshold,
                    "NiCrMo Alloy": self.ni_cr_threshold,
                },
                "fixed_direct_evaluation_parameters": {
                    "Cr_slope_wt_percent": self.cr_slope,
                    "MoW_saturation_scale_wt_percent": self.q_scale,
                },
                "small_class_warning": (
                    "NiCrMo has 41 development rows, so its threshold stays "
                    "fixed and is not independently fitted."
                ),
            }
        )
        return metadata


class MethodAwareRule(TargetRuleFamily):
    """Cr-Mo/W rule with context-learned test-method offsets."""

    name = "method_aware"
    description = "Cr-Mo/W synergy plus test-method correction"
    applicable_material_classes = ("Fe Alloy", "NiCrMo Alloy")
    term_names = CrMoWSynergyRule.term_names + ("test_method_correction",)
    coefficient_anchor = np.concatenate(
        [CrMoWSynergyRule.coefficient_anchor * 0.90, np.asarray([0.10])]
    )
    coefficient_upper_bounds = np.asarray([1.0, 1.0, 1.0, 1.0, 0.25])
    method_column = "[Cl-] Test Method"
    method_shrinkage_rows = 20.0

    @staticmethod
    def _rank_standardize(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(np.asarray(values, dtype=float), method="average")
        return _apply_scale(ranks, _fit_scale(ranks))

    def _method_labels(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                _method_family(dataset.rows[int(index)].get(self.method_column))
                for index in row_indices
            ]
        )

    def _fit_method_offsets(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        base_values: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, int]]:
        target = np.asarray(dataset.target[row_indices], dtype=float)
        if not np.isfinite(target).all():
            raise RuntimeError(
                f"{self.name} requires finite targets on context rows only."
            )
        base_prediction = base_values @ CrMoWSynergyRule.coefficient_anchor
        residual = self._rank_standardize(target) - _apply_scale(
            base_prediction, _fit_scale(base_prediction)
        )
        labels = self._method_labels(dataset, row_indices)
        offsets: dict[str, float] = {}
        counts: dict[str, int] = {}
        for label in sorted(set(labels)):
            selected = labels == label
            count = int(selected.sum())
            shrinkage = count / (count + self.method_shrinkage_rows)
            offsets[str(label)] = float(shrinkage * np.mean(residual[selected]))
            counts[str(label)] = count
        center = sum(counts[label] * offsets[label] for label in offsets) / len(labels)
        offsets = {label: float(value - center) for label, value in offsets.items()}
        return offsets, counts

    def fit_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> RuleTermData:
        row_indices = np.asarray(row_indices, dtype=int)
        base_family = CrMoWSynergyRule()
        base_prepared = base_family.fit_terms(dataset, row_indices)
        method_offsets, method_counts = self._fit_method_offsets(
            dataset, row_indices, base_prepared.values
        )
        method_values = np.asarray(
            [method_offsets.get(str(label), 0.0) for label in self._method_labels(dataset, row_indices)],
            dtype=float,
        )
        method_scaling = _fit_scale(method_values)
        state: dict[str, Any] = {
            "base_rule_state": base_prepared.state,
            "method_offsets": method_offsets,
            "method_counts": method_counts,
            "method_scaling": method_scaling,
        }
        return RuleTermData(
            values=np.column_stack(
                [
                    base_prepared.values,
                    _apply_scale(method_values, method_scaling),
                ]
            ),
            state=state,
        )

    def transform_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        row_indices = np.asarray(row_indices, dtype=int)
        base_values = CrMoWSynergyRule().transform_terms(
            dataset, row_indices, state["base_rule_state"]
        )
        method_values = np.asarray(
            [
                state["method_offsets"].get(str(label), 0.0)
                for label in self._method_labels(dataset, row_indices)
            ],
            dtype=float,
        )
        values = np.column_stack(
            [
                base_values,
                _apply_scale(method_values, state["method_scaling"]),
            ]
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.name} generated non-finite rule terms.")
        return values

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "base_rule": CrMoWSynergyRule.name,
            "scan_rate_excluded": (
                "scan rate is not present in the fixed 21-feature model schema"
            ),
            "test_method_families": [
                "potentiodynamic",
                "potentiostatic",
                "scratch",
                "other",
            ],
            "method_offset_fit": (
                "shrunken mean residual learned from each fold's context targets only"
            ),
            "method_shrinkage_rows": self.method_shrinkage_rows,
            "held_out_target_use": False,
            "synthetic_use": (
                "sample centered method offsets per synthetic task; do not copy "
                "real-dataset method directions"
            ),
            "applicability": self.applicability,
        }


class AlChlorideTemperatureRule(TargetRuleFamily):
    """Al-only chloride and high-temperature breakdown hypothesis."""

    name = "al_chloride_temperature"
    description = "Al-only log-chloride and above-30-C temperature rule"
    evaluation_role = "experimental Al-only candidate"
    applicable_material_classes = ("Al Alloy",)
    term_names = (
        "log_chloride_aggressiveness",
        "high_temperature_aggressiveness",
    )
    coefficient_anchor = np.asarray([0.80, 0.20], dtype=float)
    temperature_threshold = 30.0
    chloride_floor = 1e-12
    environment_columns = ("Test Temp. oC", "[Cl-] M")

    def _fit_imputation(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> dict[str, float]:
        imputation: dict[str, float] = {}
        for column in self.environment_columns:
            values = np.asarray(
                [
                    _numeric(dataset.table, dataset.rows[int(index)], column)
                    for index in row_indices
                ],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise RuntimeError(
                    f"{self.name} cannot impute an entirely missing {column!r}."
                )
            imputation[column] = float(np.mean(finite))
        return imputation

    def _raw_inputs(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        imputation: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        values: list[np.ndarray] = []
        for column in self.environment_columns:
            column_values = np.asarray(
                [
                    _numeric(dataset.table, dataset.rows[int(index)], column)
                    for index in row_indices
                ],
                dtype=float,
            )
            values.append(
                np.where(
                    np.isfinite(column_values),
                    column_values,
                    imputation[column],
                )
            )
        temperature, chloride = values
        return temperature, chloride

    def _unscaled_signed_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        temperature, chloride = self._raw_inputs(
            dataset,
            row_indices,
            state["imputation_means"],
        )
        return np.column_stack(
            [
                -np.log10(np.clip(chloride, self.chloride_floor, None)),
                -np.maximum(temperature - self.temperature_threshold, 0.0),
            ]
        )

    def fit_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
    ) -> RuleTermData:
        row_indices = np.asarray(row_indices, dtype=int)
        state: dict[str, Any] = {
            "imputation_means": self._fit_imputation(dataset, row_indices),
        }
        raw_terms = self._unscaled_signed_terms(dataset, row_indices, state)
        state["term_scaling"] = {
            name: _fit_scale(raw_terms[:, index])
            for index, name in enumerate(self.term_names)
        }
        return RuleTermData(
            values=self.transform_terms(dataset, row_indices, state),
            state=state,
        )

    def transform_terms(
        self,
        dataset: EpitDataset,
        row_indices: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        raw_terms = self._unscaled_signed_terms(
            dataset,
            np.asarray(row_indices, dtype=int),
            state,
        )
        values = np.column_stack(
            [
                _apply_scale(raw_terms[:, index], state["term_scaling"][name])
                for index, name in enumerate(self.term_names)
            ]
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.name} generated non-finite rule terms.")
        return values

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "environment_formula": (
                "-log10(max(chloride, 1e-12)) "
                "- max(temperature_C - 30, 0)"
            ),
            "fixed_direct_evaluation_parameters": {
                "chloride_floor_M": self.chloride_floor,
                "temperature_threshold_C": self.temperature_threshold,
            },
            "material_design": (
                "Al enters through the Al-Alloy applicability gate, not as a "
                "linear wt-percent term"
            ),
            "excluded_terms": [
                "pH because its effect is regime-dependent for Al",
                "composition terms until an Al-subfamily hypothesis is validated",
            ],
            "synthetic_use": (
                "Stage-2 experiment only until synthetic family routing exists"
            ),
            "applicability": self.applicability,
        }


class CurrentPrenFeNiBaseline(CurrentPrenRule):
    name = "current_pren_fe_ni"
    description = "Current PREN rule restricted to the Fe/Ni comparison rows"
    applicable_material_classes = ("Fe Alloy", "NiCrMo Alloy")
    evaluation_role = "matched-scope baseline"


RULE_FAMILIES: dict[str, TargetRuleFamily] = {
    CurrentPrenRule.name: CurrentPrenRule(),
    CurrentPrenFeNiBaseline.name: CurrentPrenFeNiBaseline(),
    LinearPrenRule.name: LinearPrenRule(),
    CrMoWSynergyRule.name: CrMoWSynergyRule(),
    ThresholdSaturationRule.name: ThresholdSaturationRule(),
    ImprovedEnvironmentRule.name: ImprovedEnvironmentRule(),
    CoupledBreakdownRule.name: CoupledBreakdownRule(),
    FeNiCrThresholdRule.name: FeNiCrThresholdRule(),
    MethodAwareRule.name: MethodAwareRule(),
    AlChlorideTemperatureRule.name: AlChlorideTemperatureRule(),
}


def get_rule_families(names: list[str] | None = None) -> list[TargetRuleFamily]:
    """Return requested registered families in stable order."""
    requested = list(RULE_FAMILIES) if not names or names == ["all"] else names
    unknown = sorted(set(requested) - set(RULE_FAMILIES))
    if unknown:
        choices = ", ".join(sorted(RULE_FAMILIES))
        raise ValueError(
            f"Unknown target-rule families: {', '.join(unknown)}. Available: {choices}."
        )
    return [RULE_FAMILIES[name] for name in requested]
