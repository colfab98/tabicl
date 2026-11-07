"""Shared formula/SCM target mixing for EPIT and Soccol final training."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .epit_expanded_schema import EPIT_EXPANDED_FEATURE_PROFILE
from .soccol_reviewed_schema import SOCCOL_REVIEWED_FEATURE_PROFILE
from .original_mixture import OriginalPriorMixture, _InformedPrior
from .paired_streams import keyed_seed


VERSION = "uniform_formula_scm_per_table_v1"


def resolve_mix_policy(policy=None, *, epit_v5=False, manual=False):
    effective = policy or ("uniform_per_table" if epit_v5 or manual else "fixed")
    if effective not in {"fixed", "uniform_per_table"}:
        raise ValueError(f"Unsupported formula/SCM mixing policy: {effective!r}.")
    if epit_v5 and effective != "uniform_per_table":
        raise ValueError("EPIT v5 requires uniform_per_table formula/SCM mixing.")
    return effective


def mix_bounds(mean):
    mean = float(mean)
    if not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
        raise ValueError("Formula/SCM mean weight must be in [0, 1].")
    radius = min(mean, 1.0 - mean)
    return mean - radius, mean + radius


def table_mix_weight(mean, task_seed):
    low, high = mix_bounds(mean)
    return float(np.random.default_rng(keyed_seed(task_seed, 51)).uniform(low, high))


def mix_policy_identity(mean):
    low, high = mix_bounds(mean)
    return {
        "version": VERSION,
        "distribution": "uniform_per_table",
        "mean": float(mean), "low": low, "high": high,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


class _VariableTargetMixPrior(_InformedPrior):
    """Use each source's formula, sharing the weight draw and target calculation."""
    target_policy = VERSION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mix_bounds(self.fixed_hp.get("informed_target_mix_weight", 0.35))

    def _generate_scm_target_from_physical_features(self, X, params, prior_cls):
        result = super()._generate_scm_target_from_physical_features(X, params, prior_cls)
        self._scm_target_for_mix = result
        return result

    def _formula_drive_and_rule(self, X, params):
        return (self.last_pitting_target_drive.to(device=X.device, dtype=X.dtype),
                dict(self.last_pitting_target_rule))

    @torch.no_grad()
    def _generate_scm_epit_target_from_physical_features(self, X, params, prior_cls):
        # New random draws never advance the physical/SCM/formula RNG streams.
        super()._generate_scm_epit_target_from_physical_features(X, params, prior_cls)
        mean = float(self.fixed_hp.get("informed_target_mix_weight", 0.35))
        weight = table_mix_weight(mean, int(params["paired_task_seed"]))
        drive, rule = self._formula_drive_and_rule(X, params)
        rule.update(target_mix_weight=weight, target_mix_mean=mean,
                    target_mix_distribution="uniform_per_table")
        component = weight * self._standardize_signal(drive)
        target = self._standardize_signal(
            (1 - weight) * self._standardize_signal(self._scm_target_for_mix) + component)
        self.last_pitting_target_rule = rule
        self.last_pitting_target_drive = drive.detach().cpu()
        self.last_pitting_target_component = component.detach().cpu()
        self.last_pitting_scm_target_metadata.update(
            target_policy=self.target_policy, target_mix_weight=weight, target_mix_mean=mean,
            target_mix_distribution="uniform_per_table", target_rule_family=rule["target_rule_family"])
        return target


class VariableTargetMixture(OriginalPriorMixture):
    """Keep original generic tables and vary only the informed target mixture."""
    informed_prior_class = _VariableTargetMixPrior

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fixed = kwargs.get("scm_fixed_hp") or {}
        if (fixed.get("pitting_feature_profile") not in {EPIT_EXPANDED_FEATURE_PROFILE, SOCCOL_REVIEWED_FEATURE_PROFILE}
                or fixed.get("pitting_composition_mode") != "empirical_features_scm_target"):
            raise ValueError("Variable target mixing requires a current reviewed EPIT or Soccol SCM-target prior.")
        if self.priors[1] is not None:
            previous = self.priors[1]
            shape = dict(self._config["shape_settings"])
            shape.update(batch_size=self.batch_size * self.world_size, seq_len_per_gp=False)
            self.priors[1] = self.informed_prior_class(
                **shape, min_features=previous.min_features, max_features=previous.max_features,
                prior_type="informed_scm", fixed_hp=previous.fixed_hp, sampled_hp=previous.sampled_hp)
        self._config["informed_target_mix_policy"] = mix_policy_identity(fixed.get("informed_target_mix_weight", 0.35))
