"""Portable identities for standalone empirical priors and their settings."""

from __future__ import annotations

import hashlib
import json


def configuration_value(value, _seen=None):
    """Represent constructor settings without pickling activation factories."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, type):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    # RandomChoiceFactory references the activation list containing itself.
    # Preserve that graph as stable references, never pickle callable objects.
    if _seen is None:
        _seen = {}
    if id(value) in _seen:
        return {"reference": _seen[id(value)]}
    _seen[id(value)] = len(_seen)
    if isinstance(value, dict):
        return {str(k): configuration_value(v, _seen) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration_value(v, _seen) for v in value]
    if callable(value) and hasattr(value, "__dict__"):
        return {"factory": f"{type(value).__module__}.{type(value).__qualname__}",
                "settings": configuration_value(vars(value), _seen)}
    raise TypeError(f"Unsupported prior setting for reproducible resume: {type(value).__name__}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def empirical_prior_identity(fixed_hp):
    from .epit_expanded_schema import EPIT_EXPANDED_FEATURE_PROFILE
    from .soccol_reviewed_schema import SOCCOL_REVIEWED_FEATURE_PROFILES

    name = fixed_hp.get("pitting_feature_profile")
    if name == EPIT_EXPANDED_FEATURE_PROFILE:
        from .epit_feature_profile import load_epit_feature_profile
        profile = load_epit_feature_profile(name)
    elif name in SOCCOL_REVIEWED_FEATURE_PROFILES:
        from .soccol_feature_profile import load_soccol_feature_profile
        profile = load_soccol_feature_profile(name)
    else:
        return None
    metadata = profile.metadata
    identity = {
        "version": 1,
        "feature_profile": name,
        "metadata_sha256": digest(metadata),
        "csv_sha256": metadata["csv_sha256"],
        "feature_columns": metadata["feature_columns"],
        "development_selection_sha256": digest(metadata.get("development_selection", metadata.get("sampling_scope"))),
        "preprocessing": metadata.get("preprocessing"),
        "sampling": metadata.get("sampling"),
        "category_mappings": metadata.get("category_mappings"),
        "reviewed_data": metadata.get("reviewed_composition", metadata.get("development_selection", {}).get("reviewed_data")),
    }
    if hasattr(profile, "composition_profile"):
        identity["composition_profile"] = profile.composition_profile.name
        identity["composition_metadata_sha256"] = digest(profile.composition_profile.metadata)
    return identity
