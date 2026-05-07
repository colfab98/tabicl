#!/usr/bin/env python3
"""Audit corrosion dataset columns for feature-group design.

This script is intentionally descriptive. It does not try to settle the final
feature ontology by itself. Instead, it turns the saved corrosion datasets into
an evidence table that can be reviewed when deciding how synthetic corrosion
tasks should be structured.

The key design choice is to separate several axes that were previously folded
into one coarse group label:

- current_group: the existing hand map used by analyze_structure.py
- semantic_domain: what the column describes in corrosion terms
- statistical_family: what its marginal/data type looks like
- causal_role: how it should probably enter a prediction task

That separation is important for inhibitor molecular descriptors: they may look
like dense numeric material descriptors, while causally describing an
intervention agent rather than the base alloy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from analyze_structure import (
    GROUPS,
    NA_STRINGS,
    Table,
    clean_name,
    load_all_tables,
    to_float,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "feature_group_audit"

SEMANTIC_DOMAINS = [
    "material",
    "molecular_descriptor",
    "environment",
    "electrochem",
    "history",
    "process",
    "intervention",
    "target",
    "metadata",
    "exclude",
]

CAUSAL_ROLES = [
    "base_material_condition",
    "intervention_agent_descriptor",
    "exposure_context",
    "process_or_history_condition",
    "direct_intervention_control",
    "downstream_response_measurement",
    "target_response",
    "metadata_identifier",
    "exclude",
]

INHIBITOR_DATASETS = {
    "datacor_aluminum_inhibitors",
    "datacortech_aluminum_inhibitors",
    "mg_az91_inhibitors",
    "mg_ze41_inhibitors",
}

TARGETISH_GROUPS = {"target", "metadata", "exclude"}

ELEMENT_SYMBOLS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Th",
    "U",
}

MATERIAL_RE = re.compile(
    r"\b("
    r"alloy|material|metal|substrate|steel|uns|phase|fcc|bcc|hcp|"
    r"composition|element|cement|binder|ggbs|fly ash|silic|silica|"
    r"porosity|water to binder|vec|entropy of mixing|microstructure"
    r")\b",
    re.IGNORECASE,
)
ENVIRONMENT_RE = re.compile(
    r"\b("
    r"environment|electrolyte|solution|chloride|cl-|salt|salinity|"
    r"ph|temperature|humidity|oxygen|saturation|water content|pore|"
    r"concentration|nacl|so4|no3|moo4|molar|mol/l"
    r")\b",
    re.IGNORECASE,
)
HISTORY_RE = re.compile(
    r"\b("
    r"time|duration|day|days|hour|hours|exposure|immersion|scan rate|"
    r"condition|history|age|cycle|cycles"
    r")\b",
    re.IGNORECASE,
)
PROCESS_RE = re.compile(
    r"\b("
    r"processing|process|am process|heat treatment|treatment|hip|"
    r"cold rolling|rolling|anneal|quenched|tempered|encapsulated|"
    r"synergistic"
    r")\b",
    re.IGNORECASE,
)
INTERVENTION_RE = re.compile(
    r"\b("
    r"inhibitor|inhib|inhibition|coating|coated|additive|dose|dosage|"
    r"concentrat|concentration|"
    r"synergistic|encapsulated|treatment|control"
    r")\b",
    re.IGNORECASE,
)
DIRECT_INTERVENTION_CONTROL_RE = re.compile(
    r"(inhib.*concentrat|synergistic|encapsulated|dose|dosage|coating|coated)",
    re.IGNORECASE,
)
ELECTROCHEM_RE = re.compile(
    r"\b("
    r"corrosion|potential|epit|epass|erp|ecrev|ecorr|ocp|icorr|ipass|"
    r"current|resistivity|resistance|polarisation|polarization|passive|"
    r"repassivation|pitting|crevice|localized attack|rate|rating"
    r")\b",
    re.IGNORECASE,
)
TARGET_RE = re.compile(
    r"\b("
    r"target|efficiency|score|corrosion rate|pitting potential|"
    r"repassivation|ecorr|icorr|ocp|epit|epass|er\.crev|rate|rating"
    r")\b",
    re.IGNORECASE,
)
METADATA_RE = re.compile(
    r"\b("
    r"doi|reference|ref|source|comment|note|notes|id|no\.|number|"
    r"name|formula|maps|file|descriptor_source|contributor"
    r")\b",
    re.IGNORECASE,
)
MOLECULAR_DESCRIPTOR_RE = re.compile(
    r"("
    r"\bmol\b|molecular|descriptor|fingerprint|logp|xlogp|slogp|mlogp|"
    r"tpsa|estate|vsa|bcut|chi|kappa|wiener|zagreb|balaban|"
    r"\bmats|gats|rdf|morse|mor\d|ats\d|aats|spmax|spmin|"
    r"topo|apol|bpol|bertz|kier|lipinski|hbond|donor|acceptor|"
    r"rotatable|aromatic|ring|charge|polar|surface|volume|"
    r"natom|nheavy|nhetero|nrot|nring|nsmallrings|naromrings|"
    r"nringblocks|naromblocks|nrings\d+|nhbd|nhba|"
    r"eta[_a-z0-9]*|ve[0-9]|vr[0-9]"
    r")",
    re.IGNORECASE,
)


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return text or "unnamed"


def json_default(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_default(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_default(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_default(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    return obj


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        fieldnames = fields
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key, "")) for key in fieldnames})


def csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(json_default(value), sort_keys=True)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def numeric_parse(value: Any) -> float:
    return to_float(value, allow_embedded_number=True, allow_rating=True)


def nonmissing_values(table: Table, column: str) -> list[str]:
    vals = []
    for row in table.rows:
        text = clean_name(row.get(column))
        if text and text.lower() not in NA_STRINGS:
            vals.append(text)
    return vals


def numeric_values(table: Table, column: str) -> np.ndarray:
    return np.array([numeric_parse(row.get(column)) for row in table.rows], dtype=float)


def finite_numeric_values(table: Table, column: str) -> np.ndarray:
    vals = numeric_values(table, column)
    return vals[np.isfinite(vals)]


def quantile(vals: np.ndarray, q: float) -> float:
    return float(np.quantile(vals, q)) if len(vals) else math.nan


def safe_mean(vals: np.ndarray) -> float:
    return float(np.mean(vals)) if len(vals) else math.nan


def safe_std(vals: np.ndarray) -> float:
    return float(np.std(vals)) if len(vals) else math.nan


def safe_skew(vals: np.ndarray) -> float:
    if len(vals) < 3 or float(np.std(vals)) == 0.0:
        return math.nan
    span = float(np.max(vals) - np.min(vals))
    scale = max(1.0, abs(float(np.mean(vals))))
    if span <= 1e-10 * scale:
        return math.nan
    return float(stats.skew(vals, nan_policy="omit"))


def integer_like_ratio(vals: np.ndarray) -> float:
    if len(vals) == 0:
        return math.nan
    return float(np.mean(np.isclose(vals, np.round(vals), atol=1e-8)))


def top_values(values: list[str], limit: int) -> list[dict[str, Any]]:
    total = len(values)
    counts = Counter(values)
    return [
        {"value": value, "count": count, "share": count / total if total else math.nan}
        for value, count in counts.most_common(limit)
    ]


def is_element_column(column: str) -> bool:
    return clean_name(column) in ELEMENT_SYMBOLS


def column_profile(table: Table, column: str, top_k: int) -> dict[str, Any]:
    text_vals = nonmissing_values(table, column)
    num_vals_all = numeric_values(table, column)
    num_vals = num_vals_all[np.isfinite(num_vals_all)]
    n_rows = len(table.rows)
    unique_vals = set(text_vals)
    nonmissing = len(text_vals)
    numeric_count = int(np.isfinite(num_vals_all).sum())
    numeric_ratio_nonmissing = numeric_count / nonmissing if nonmissing else 0.0
    numeric_ratio_rows = numeric_count / n_rows if n_rows else 0.0
    unique_count = len(unique_vals)
    avg_text_len = statistics.mean([len(v) for v in text_vals]) if text_vals else math.nan
    max_text_len = max([len(v) for v in text_vals], default=0)

    sorted_vals = np.sort(num_vals) if len(num_vals) else num_vals
    q01 = quantile(sorted_vals, 0.01)
    q05 = quantile(sorted_vals, 0.05)
    q25 = quantile(sorted_vals, 0.25)
    q50 = quantile(sorted_vals, 0.50)
    q75 = quantile(sorted_vals, 0.75)
    q95 = quantile(sorted_vals, 0.95)
    q99 = quantile(sorted_vals, 0.99)
    iqr = q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else math.nan
    min_value = float(sorted_vals[0]) if len(sorted_vals) else math.nan
    max_value = float(sorted_vals[-1]) if len(sorted_vals) else math.nan
    positive_ratio = float(np.mean(num_vals > 0)) if len(num_vals) else math.nan
    negative_ratio = float(np.mean(num_vals < 0)) if len(num_vals) else math.nan
    zero_ratio = float(np.mean(np.isclose(num_vals, 0.0))) if len(num_vals) else math.nan
    unique_numeric = len(set(float(v) for v in num_vals)) if len(num_vals) else 0

    return {
        "dataset": table.dataset,
        "table": table.table,
        "column": column,
        "current_group": table.groups.get(column, "metadata"),
        "n_rows": n_rows,
        "nonmissing_count": nonmissing,
        "nonmissing_ratio": nonmissing / n_rows if n_rows else math.nan,
        "unique_count": unique_count,
        "unique_ratio_nonmissing": unique_count / nonmissing if nonmissing else math.nan,
        "numeric_count": numeric_count,
        "numeric_ratio_nonmissing": numeric_ratio_nonmissing,
        "numeric_ratio_rows": numeric_ratio_rows,
        "unique_numeric_count": unique_numeric,
        "integer_like_ratio": integer_like_ratio(num_vals),
        "avg_text_len": avg_text_len,
        "max_text_len": max_text_len,
        "numeric_min": min_value,
        "numeric_q01": q01,
        "numeric_q05": q05,
        "numeric_q25": q25,
        "numeric_median": q50,
        "numeric_q75": q75,
        "numeric_q95": q95,
        "numeric_q99": q99,
        "numeric_max": max_value,
        "numeric_mean": safe_mean(num_vals),
        "numeric_std": safe_std(num_vals),
        "numeric_iqr": iqr,
        "numeric_skew": safe_skew(num_vals),
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "zero_ratio": zero_ratio,
        "top_values": top_values(text_vals, top_k),
        "is_element_symbol": is_element_column(column),
    }


def table_composition_candidates(table: Table, profiles_by_col: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates = []
    for col, profile in profiles_by_col.items():
        name = col.lower()
        if profile["numeric_ratio_rows"] < 0.5:
            continue
        if profile["is_element_symbol"] or re.search(
            r"\b(proportion|weight|fraction|composition|cement|ggbs|fly ash|silic fume)\b|wt\.?%|wt\s|wt_",
            name,
        ):
            candidates.append(col)

    out = {col: {"composition_set": "", "composition_sum_median": math.nan} for col in profiles_by_col}
    if len(candidates) < 2:
        return out

    matrix = np.column_stack([numeric_values(table, col) for col in candidates])
    mask = np.isfinite(matrix).sum(axis=1) >= max(2, min(len(candidates), 4))
    if not mask.any():
        return out
    row_sums = np.nansum(matrix[mask], axis=1)
    row_sums = row_sums[np.isfinite(row_sums)]
    if len(row_sums) == 0:
        return out
    median_sum = float(np.median(row_sums))
    if 0.85 <= median_sum <= 1.15:
        set_name = "closed_composition_fraction_candidate"
    elif 85.0 <= median_sum <= 115.0:
        set_name = "closed_composition_percent_candidate"
    elif 0.0 < median_sum <= 100.0:
        set_name = "partial_composition_or_mixture_candidate"
    else:
        set_name = ""
    if set_name:
        for col in candidates:
            out[col] = {"composition_set": set_name, "composition_sum_median": median_sum}
    return out


def is_likely_molecular_descriptor(table: Table, column: str, profile: dict[str, Any]) -> bool:
    name = column.lower()
    if profile["is_element_symbol"] and table.dataset not in INHIBITOR_DATASETS:
        return False
    if DIRECT_INTERVENTION_CONTROL_RE.search(column):
        return False
    if PROCESS_RE.search(column) and table.dataset not in INHIBITOR_DATASETS:
        return False
    if table.dataset not in INHIBITOR_DATASETS:
        if table.groups.get(column) in {"material", "environment", "history", "intervention", "electrochem", "target", "exclude"}:
            return False
        return bool(MOLECULAR_DESCRIPTOR_RE.search(column))
    metadata_names = {
        "inhibitor",
        "compound",
        "name",
        "no.",
        "number",
        "descriptor_source",
        "source_efficiency_column",
    }
    if name in metadata_names:
        return False
    if table.groups.get(column) in {"target", "material", "environment", "history", "metadata", "exclude"}:
        return False
    if DIRECT_INTERVENTION_CONTROL_RE.search(column) or any(token in name for token in ["concentrat", "synergistic", "encapsulated"]):
        return False
    if profile["numeric_ratio_nonmissing"] >= 0.75:
        return True
    return bool(MOLECULAR_DESCRIPTOR_RE.search(column))


def semantic_scores(table: Table, column: str, profile: dict[str, Any]) -> tuple[Counter[str], list[str]]:
    scores: Counter[str] = Counter()
    reasons: list[str] = []
    current = table.groups.get(column, "metadata")

    if current in SEMANTIC_DOMAINS:
        scores[current] += 1.25
        reasons.append(f"current_group:{current}")
    elif current == "intervention":
        scores["intervention"] += 1.25
        reasons.append("current_group:intervention")

    if current == "target":
        scores["target"] += 6.0
    if current == "metadata":
        scores["metadata"] += 2.0
    if current == "exclude":
        scores["exclude"] += 6.0

    if profile["is_element_symbol"]:
        scores["material"] += 5.0
        reasons.append("element_symbol")
    if MATERIAL_RE.search(column):
        scores["material"] += 2.0
        reasons.append("material_name_pattern")
    if ENVIRONMENT_RE.search(column):
        scores["environment"] += 2.0
        reasons.append("environment_name_pattern")
    if HISTORY_RE.search(column):
        scores["history"] += 1.7
        reasons.append("history_name_pattern")
    if PROCESS_RE.search(column):
        scores["process"] += 2.2
        reasons.append("process_name_pattern")
    if INTERVENTION_RE.search(column):
        scores["intervention"] += 2.5
        reasons.append("intervention_name_pattern")
    if DIRECT_INTERVENTION_CONTROL_RE.search(column):
        scores["intervention"] += 3.0
        reasons.append("direct_intervention_control_pattern")
    if ELECTROCHEM_RE.search(column):
        scores["electrochem"] += 2.3
        reasons.append("electrochem_name_pattern")
    if TARGET_RE.search(column) and current == "target":
        scores["target"] += 1.5
    if METADATA_RE.search(column):
        scores["metadata"] += 1.7
        reasons.append("metadata_name_pattern")

    if is_likely_molecular_descriptor(table, column, profile):
        scores["molecular_descriptor"] += 4.0
        scores["intervention"] += 0.5
        reasons.append("inhibitor_or_molecular_descriptor_pattern")

    if table.dataset in INHIBITOR_DATASETS and current == "intervention":
        # Current grouping calls many inhibitor-agent descriptors intervention.
        # Preserve intervention as semantic context, but make descriptor status
        # explicit when the column is dense numeric.
        if profile["numeric_ratio_nonmissing"] >= 0.75 and not DIRECT_INTERVENTION_CONTROL_RE.search(column):
            scores["molecular_descriptor"] += 2.0
            reasons.append("dense_numeric_in_inhibitor_dataset")

    if profile["unique_ratio_nonmissing"] >= 0.8 and profile["numeric_ratio_nonmissing"] < 0.5:
        scores["metadata"] += 0.8
        reasons.append("high_cardinality_text")

    if not scores:
        scores["metadata"] += 0.5
        reasons.append("fallback_metadata")

    return scores, reasons


def top_score(scores: Counter[str]) -> tuple[str, float, float]:
    if not scores:
        return "metadata", 0.0, 0.0
    ordered = scores.most_common()
    primary, primary_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    return primary, float(primary_score), float(primary_score - runner_up)


def data_kind(profile: dict[str, Any]) -> str:
    if profile["nonmissing_ratio"] < 0.2:
        return "mostly_missing"
    if profile["numeric_ratio_nonmissing"] >= 0.85 and profile["numeric_count"] >= 5:
        if profile["unique_numeric_count"] <= 2:
            return "numeric_binary"
        if profile["unique_numeric_count"] <= min(10, max(3, profile["numeric_count"] // 10)):
            return "numeric_discrete"
        return "numeric_continuous"
    if profile["unique_count"] <= 2:
        return "categorical_binary"
    if profile["unique_ratio_nonmissing"] >= 0.85:
        return "identifier_or_free_text"
    if profile["unique_count"] <= 80:
        return "categorical"
    return "high_cardinality_categorical"


def statistical_family(
    table: Table,
    column: str,
    profile: dict[str, Any],
    composition_info: dict[str, Any],
    semantic_domain: str,
) -> tuple[str, list[str]]:
    tags: list[str] = []
    kind = data_kind(profile)
    name = column.lower()
    min_v = profile["numeric_min"]
    max_v = profile["numeric_max"]
    q50 = profile["numeric_median"]
    skew = profile["numeric_skew"]
    numeric = profile["numeric_ratio_nonmissing"] >= 0.85 and profile["numeric_count"] >= 5

    if profile["nonmissing_ratio"] < 0.2:
        return "mostly_missing", ["sparse"]

    if profile["unique_count"] <= 2:
        tags.append("binary")

    if kind in {"identifier_or_free_text", "high_cardinality_categorical"}:
        if METADATA_RE.search(column) or profile["unique_ratio_nonmissing"] >= 0.85:
            tags.append("identifier_like")
        if profile["avg_text_len"] and profile["avg_text_len"] > 40:
            tags.append("free_text_like")
        return ("identifier_or_free_text" if "identifier_like" in tags else kind), tags

    if not numeric:
        if profile["unique_count"] <= 80:
            return "categorical_nominal", tags
        return "high_cardinality_categorical", tags

    if composition_info.get("composition_set"):
        tags.append(composition_info["composition_set"])
        return "composition_component", tags

    if re.search(r"\bph\b", name) and np.isfinite(min_v) and np.isfinite(max_v):
        if -0.5 <= min_v <= 14.5 and -0.5 <= max_v <= 14.5:
            return "pH_bounded", tags

    if semantic_domain == "molecular_descriptor":
        tags.append("dense_numeric_agent_descriptor")
        if profile["positive_ratio"] is not None and np.isfinite(profile["positive_ratio"]) and profile["positive_ratio"] > 0.95:
            tags.append("positive")
        return "dense_molecular_descriptor", tags

    if any(token in name for token in ["efficiency", "proportion", "percent", "wt", "%"]):
        if np.isfinite(min_v) and np.isfinite(max_v) and -1e-8 <= min_v and max_v <= 100.0 + 1e-8:
            return "percentage_or_fraction", tags

    if np.isfinite(min_v) and np.isfinite(max_v) and -1e-8 <= min_v and max_v <= 1.0 + 1e-8:
        return "bounded_0_1", tags

    if any(token in name for token in ["potential", "ocp", "epit", "epass", "erp", "ecorr", "ecrev"]):
        return "signed_electrochemical_potential", tags

    if any(token in name for token in ["current", "icorr", "ipass", "resist", "rate"]):
        if profile["positive_ratio"] is not None and np.isfinite(profile["positive_ratio"]) and profile["positive_ratio"] > 0.9:
            tags.append("positive_response")
        return "electrochemical_or_rate_response", tags

    if any(token in name for token in ["temperature", "temp", "t. °c"]):
        return "temperature_like", tags

    if any(token in name for token in ["concentration", "chloride", "salt", "salinity", "mol/l", "_m"]):
        if profile["positive_ratio"] is not None and np.isfinite(profile["positive_ratio"]) and profile["positive_ratio"] > 0.9:
            tags.append("positive")
        return "concentration_like", tags

    if any(token in name for token in ["time", "duration", "day", "hour", "scan rate"]):
        if profile["positive_ratio"] is not None and np.isfinite(profile["positive_ratio"]) and profile["positive_ratio"] > 0.9:
            tags.append("positive")
        return "time_or_rate_condition", tags

    if profile["unique_numeric_count"] <= 10 or profile["integer_like_ratio"] > 0.98:
        return "low_cardinality_numeric", tags

    if profile["positive_ratio"] is not None and np.isfinite(profile["positive_ratio"]) and profile["positive_ratio"] > 0.95:
        if np.isfinite(skew) and skew > 1.0 and np.isfinite(q50) and q50 > 0:
            return "positive_skewed_numeric", tags
        return "positive_continuous_numeric", tags

    return "signed_continuous_numeric", tags


def causal_role(column: str, current_group: str, semantic_domain: str, stat_family: str) -> tuple[str, list[str]]:
    tags: list[str] = []
    if current_group == "exclude" or semantic_domain == "exclude":
        return "exclude", tags
    if current_group == "target" or semantic_domain == "target":
        return "target_response", tags
    if semantic_domain == "metadata":
        return "metadata_identifier", tags
    if semantic_domain == "molecular_descriptor":
        tags.append("intervention_agent_not_base_material")
        return "intervention_agent_descriptor", tags
    if semantic_domain == "intervention":
        return "direct_intervention_control", tags
    if semantic_domain == "process":
        return "process_or_history_condition", tags
    if semantic_domain == "history":
        return "process_or_history_condition", tags
    if semantic_domain == "environment":
        return "exposure_context", tags
    if semantic_domain == "electrochem":
        tags.append("possible_downstream_or_leakage_feature")
        return "downstream_response_measurement", tags
    if semantic_domain == "material":
        return "base_material_condition", tags
    if stat_family in {"identifier_or_free_text", "high_cardinality_categorical"}:
        return "metadata_identifier", tags
    return "metadata_identifier", tags


def annotate_columns(tables: list[Table], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in tables:
        profiles = {col: column_profile(table, col, top_k) for col in table.columns}
        composition = table_composition_candidates(table, profiles)
        for col in table.columns:
            profile = profiles[col]
            scores, reasons = semantic_scores(table, col, profile)
            semantic, score, margin = top_score(scores)
            stat_family, stat_tags = statistical_family(table, col, profile, composition[col], semantic)
            role, role_tags = causal_role(col, profile["current_group"], semantic, stat_family)
            row = dict(profile)
            row.update(
                {
                    "semantic_domain": semantic,
                    "semantic_score": score,
                    "semantic_score_margin": margin,
                    "semantic_scores": dict(scores),
                    "semantic_reasons": reasons,
                    "data_kind": data_kind(profile),
                    "statistical_family": stat_family,
                    "distribution_tags": stat_tags,
                    "composition_set": composition[col].get("composition_set", ""),
                    "composition_sum_median": composition[col].get("composition_sum_median", math.nan),
                    "causal_role": role,
                    "causal_tags": role_tags,
                    "current_vs_proposed_changed": profile["current_group"] != semantic,
                    "review_priority": review_priority(profile["current_group"], semantic, role, margin, reasons),
                }
            )
            rows.append(row)
    return rows


def review_priority(
    current_group: str,
    semantic_domain: str,
    role: str,
    margin: float,
    reasons: list[str],
) -> str:
    if current_group == "intervention" and semantic_domain == "molecular_descriptor":
        return "high_molecular_vs_intervention"
    if role == "downstream_response_measurement":
        return "high_leakage_or_response_feature"
    if current_group not in {semantic_domain, "exclude"} and semantic_domain not in {"metadata", "exclude"}:
        return "medium_group_changed"
    if margin < 1.0 and semantic_domain not in {"metadata", "exclude"}:
        return "medium_ambiguous"
    if "high_cardinality_text" in reasons and current_group not in {"metadata", "exclude"}:
        return "medium_identifier_risk"
    return "low"


def valid_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def abs_spearman_pair(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[float, int]:
    xv, yv = valid_pair(x, y)
    n = len(xv)
    if n < min_n or np.std(xv) == 0 or np.std(yv) == 0:
        return math.nan, n
    rho = stats.spearmanr(xv, yv).correlation
    if rho is None or not np.isfinite(rho):
        return math.nan, n
    return abs(float(rho)), n


def summarize(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": math.nan, "median": math.nan, "q75": math.nan, "max": math.nan}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "q75": float(np.quantile(vals, 0.75)),
        "max": float(max(vals)),
    }


def numeric_audit_columns(
    table: Table,
    column_rows: list[dict[str, Any]],
    min_values: int,
    include_targets: bool,
) -> list[str]:
    by_name = {row["column"]: row for row in column_rows if row["dataset"] == table.dataset and row["table"] == table.table}
    out = []
    for col in table.columns:
        info = by_name[col]
        if not include_targets and info["current_group"] in TARGETISH_GROUPS:
            continue
        if info["numeric_count"] >= min_values and info["numeric_std"] and np.isfinite(info["numeric_std"]) and info["numeric_std"] > 0:
            out.append(col)
    return out


def abs_spearman_matrix(
    table: Table,
    cols: list[str],
    min_pair_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute Spearman correlations and pair counts for numeric cols.

    Large inhibitor descriptor tables can have 800-1200+ numeric columns. Calling
    scipy's spearmanr once per pair is too slow there, so this function uses a
    complete-case rank-correlation matrix when that is statistically reasonable.
    Smaller or sparse tables fall back to exact pairwise-complete correlations.
    """

    n_cols = len(cols)
    corr = np.full((n_cols, n_cols), math.nan, dtype=float)
    counts = np.zeros((n_cols, n_cols), dtype=int)
    if n_cols < 2:
        return corr, counts

    matrix = np.column_stack([numeric_values(table, col) for col in cols])
    finite = np.isfinite(matrix)
    complete = np.all(finite, axis=1)
    complete_n = int(complete.sum())
    median_col_n = float(np.median(finite.sum(axis=0))) if n_cols else 0.0

    if complete_n >= min_pair_n and (n_cols >= 80 or complete_n >= 0.8 * median_col_n):
        complete_matrix = matrix[complete]
        ranks = np.apply_along_axis(stats.rankdata, 0, complete_matrix)
        std = np.std(ranks, axis=0)
        valid = std > 0
        if valid.sum() >= 2:
            corr_valid = np.corrcoef(ranks[:, valid], rowvar=False)
            if corr_valid.ndim == 0:
                corr_valid = np.array([[float(corr_valid)]])
            idx = np.where(valid)[0]
            corr[np.ix_(idx, idx)] = np.abs(corr_valid)
            counts[np.ix_(idx, idx)] = complete_n
        np.fill_diagonal(corr, math.nan)
        np.fill_diagonal(counts, 0)
        return corr, counts

    vectors = [matrix[:, idx] for idx in range(n_cols)]
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            rho, n = abs_spearman_pair(vectors[i], vectors[j], min_pair_n)
            if np.isfinite(rho):
                corr[i, j] = corr[j, i] = rho
                counts[i, j] = counts[j, i] = n
    return corr, counts


def group_pair_correlations(
    tables: list[Table],
    column_rows: list[dict[str, Any]],
    *,
    min_pair_n: int,
    min_values: int,
    include_targets: bool,
) -> list[dict[str, Any]]:
    by_key = {(r["dataset"], r["table"], r["column"]): r for r in column_rows}
    axes = ["current_group", "semantic_domain", "causal_role", "statistical_family"]
    rows: list[dict[str, Any]] = []
    for table in tables:
        cols = numeric_audit_columns(table, column_rows, min_values, include_targets)
        if len(cols) < 2:
            continue
        corr_matrix, count_matrix = abs_spearman_matrix(table, cols, min_pair_n)
        for axis in axes:
            bucket: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
            for i, c1 in enumerate(cols):
                g1 = str(by_key[(table.dataset, table.table, c1)][axis])
                for j in range(i + 1, len(cols)):
                    c2 = cols[j]
                    g2 = str(by_key[(table.dataset, table.table, c2)][axis])
                    a, b = sorted([g1, g2])
                    rho = corr_matrix[i, j]
                    if np.isfinite(rho):
                        n = int(count_matrix[i, j])
                        bucket[(a, b)].append((rho, n))
            for (g1, g2), vals in bucket.items():
                corrs = [v[0] for v in vals]
                ns = [v[1] for v in vals]
                summary = summarize(corrs)
                rows.append(
                    {
                        "dataset": table.dataset,
                        "table": table.table,
                        "axis": axis,
                        "group_a": g1,
                        "group_b": g2,
                        "relation": "within" if g1 == g2 else "cross",
                        "n_pairs": summary["n"],
                        "mean_abs_spearman": summary["mean"],
                        "median_abs_spearman": summary["median"],
                        "q75_abs_spearman": summary["q75"],
                        "max_abs_spearman": summary["max"],
                        "median_pair_n": float(statistics.median(ns)) if ns else math.nan,
                    }
                )
    return rows


def eta_squared(categories: list[str], y: np.ndarray, min_n: int, max_levels: int) -> tuple[float, int, int]:
    groups: dict[str, list[float]] = defaultdict(list)
    for cat, yy in zip(categories, y):
        if cat and cat.lower() not in NA_STRINGS and np.isfinite(yy):
            groups[cat].append(float(yy))
    groups = {key: vals for key, vals in groups.items() if len(vals) >= 2}
    n = sum(len(vals) for vals in groups.values())
    if n < min_n or not (2 <= len(groups) <= max_levels):
        return math.nan, n, len(groups)
    all_vals = np.array([value for vals in groups.values() for value in vals], dtype=float)
    grand = float(np.mean(all_vals))
    ss_total = float(np.sum((all_vals - grand) ** 2))
    if ss_total <= 0:
        return math.nan, n, len(groups)
    ss_between = sum(len(vals) * (float(np.mean(vals)) - grand) ** 2 for vals in groups.values())
    return float(ss_between / ss_total), n, len(groups)


def feature_target_associations(
    tables: list[Table],
    column_rows: list[dict[str, Any]],
    *,
    min_pair_n: int,
    max_levels: int,
) -> list[dict[str, Any]]:
    by_key = {(r["dataset"], r["table"], r["column"]): r for r in column_rows}
    out: list[dict[str, Any]] = []
    for table in tables:
        targets = [
            col
            for col in table.columns
            if by_key[(table.dataset, table.table, col)]["current_group"] == "target"
            and by_key[(table.dataset, table.table, col)]["numeric_count"] >= min_pair_n
        ]
        features = [
            col
            for col in table.columns
            if by_key[(table.dataset, table.table, col)]["current_group"] not in TARGETISH_GROUPS
            and by_key[(table.dataset, table.table, col)]["nonmissing_count"] >= min_pair_n
        ]
        for target in targets:
            y = numeric_values(table, target)
            for feature in features:
                info = by_key[(table.dataset, table.table, feature)]
                if info["numeric_ratio_nonmissing"] >= 0.85:
                    score, n = abs_spearman_pair(numeric_values(table, feature), y, min_pair_n)
                    method = "abs_spearman"
                    levels = info["unique_numeric_count"]
                else:
                    cats = [clean_name(row.get(feature)) for row in table.rows]
                    score, n, levels = eta_squared(cats, y, min_pair_n, max_levels)
                    method = "eta_squared"
                if not np.isfinite(score):
                    continue
                out.append(
                    {
                        "dataset": table.dataset,
                        "table": table.table,
                        "target": target,
                        "feature": feature,
                        "current_group": info["current_group"],
                        "semantic_domain": info["semantic_domain"],
                        "causal_role": info["causal_role"],
                        "statistical_family": info["statistical_family"],
                        "method": method,
                        "score": score,
                        "n": n,
                        "levels": levels,
                    }
                )
    out.sort(key=lambda r: (-float(r["score"]), r["dataset"], r["table"], r["target"], r["feature"]))
    return out


def table_summaries(tables: list[Table], column_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in column_rows:
        by_table[(row["dataset"], row["table"])].append(row)
    out: list[dict[str, Any]] = []
    for table in tables:
        rows = by_table[(table.dataset, table.table)]
        current = Counter(row["current_group"] for row in rows)
        semantic = Counter(row["semantic_domain"] for row in rows)
        role = Counter(row["causal_role"] for row in rows)
        stat_family = Counter(row["statistical_family"] for row in rows)
        review = Counter(row["review_priority"] for row in rows)
        out.append(
            {
                "dataset": table.dataset,
                "table": table.table,
                "rows": len(table.rows),
                "columns": len(table.columns),
                "current_group_counts": dict(current),
                "semantic_domain_counts": dict(semantic),
                "causal_role_counts": dict(role),
                "statistical_family_counts": dict(stat_family),
                "review_priority_counts": dict(review),
                "n_numeric_columns": sum(1 for row in rows if row["numeric_ratio_nonmissing"] >= 0.85 and row["numeric_count"] >= 5),
                "n_targets": current.get("target", 0),
                "n_molecular_descriptor_candidates": semantic.get("molecular_descriptor", 0),
                "n_intervention_agent_descriptor_roles": role.get("intervention_agent_descriptor", 0),
            }
        )
    return out


def aggregate_group_pair_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    pair_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        key = (row["axis"], row["group_a"], row["group_b"], row["relation"])
        if np.isfinite(float(row.get("mean_abs_spearman", math.nan))):
            # Weight by the number of column pairs, because each row is already
            # a per-table summary of many pairwise correlations.
            n_pairs = int(row.get("n_pairs", 0) or 0)
            pair_counts[key] += n_pairs
            buckets[key].extend([float(row["mean_abs_spearman"])] * max(1, n_pairs))
    out = []
    for key, vals in buckets.items():
        summary = summarize(vals)
        out.append(
            {
                "axis": key[0],
                "group_a": key[1],
                "group_b": key[2],
                "relation": key[3],
                "weighted_n_pairs": pair_counts[key],
                "mean_abs_spearman": summary["mean"],
                "median_abs_spearman": summary["median"],
                "q75_abs_spearman": summary["q75"],
                "max_abs_spearman": summary["max"],
            }
        )
    out.sort(key=lambda r: (r["axis"], r["relation"], r["group_a"], r["group_b"]))
    return out


def fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def top_n(rows: list[dict[str, Any]], n: int, predicate: Any | None = None) -> list[dict[str, Any]]:
    filtered = [row for row in rows if predicate is None or predicate(row)]
    return filtered[:n]


def write_markdown(
    path: Path,
    table_rows: list[dict[str, Any]],
    column_rows: list[dict[str, Any]],
    corr_agg: list[dict[str, Any]],
    associations: list[dict[str, Any]],
) -> None:
    changed = [row for row in column_rows if row["current_vs_proposed_changed"]]
    high_review = [row for row in column_rows if str(row["review_priority"]).startswith("high")]
    molecular_split = [
        row
        for row in column_rows
        if row["current_group"] == "intervention" and row["semantic_domain"] == "molecular_descriptor"
    ]
    leakage = [row for row in column_rows if row["causal_role"] == "downstream_response_measurement"]
    domain_counts = Counter(row["semantic_domain"] for row in column_rows)
    family_counts = Counter(row["statistical_family"] for row in column_rows)
    role_counts = Counter(row["causal_role"] for row in column_rows)

    lines = [
        "# Feature Group Audit",
        "",
        "This report is generated by `corrosion_datasets/analysis/scripts/audit_feature_groups.py`.",
        "It is an evidence inventory for revising the corrosion feature ontology; it is not a final ontology by itself.",
        "",
        "## Main Point",
        "",
        "The audit keeps four labels separate: existing coarse group, proposed semantic domain, statistical/marginal family, and causal role. This makes columns such as inhibitor molecular descriptors reviewable as dense numeric descriptors of an intervention agent instead of forcing them into either base material or direct intervention.",
        "",
        "## Coverage",
        "",
        "| Dataset | Table | Rows | Columns | Targets | Molecular descriptor candidates | High-review columns |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    high_by_table = Counter((row["dataset"], row["table"]) for row in high_review)
    for row in table_rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | {row['rows']} | {row['columns']} | "
            f"{row['n_targets']} | {row['n_molecular_descriptor_candidates']} | "
            f"{high_by_table[(row['dataset'], row['table'])]} |"
        )

    lines += [
        "",
        "## Column Counts",
        "",
        "| Axis | Label | Columns |",
        "|---|---|---:|",
    ]
    for label, count in domain_counts.most_common():
        lines.append(f"| semantic_domain | `{label}` | {count} |")
    for label, count in role_counts.most_common():
        lines.append(f"| causal_role | `{label}` | {count} |")
    for label, count in family_counts.most_common(25):
        lines.append(f"| statistical_family | `{label}` | {count} |")

    lines += [
        "",
        "## Molecular Descriptor Versus Intervention Review",
        "",
        "| Dataset | Table | Column | Current Group | Proposed Domain | Family | Role |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in molecular_split[:80]:
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | `{row['column']}` | "
            f"{row['current_group']} | {row['semantic_domain']} | {row['statistical_family']} | {row['causal_role']} |"
        )
    if len(molecular_split) > 80:
        lines.append(f"| ... | ... | {len(molecular_split) - 80} more columns in CSV |  |  |  |  |")

    lines += [
        "",
        "## Downstream Or Leakage-Risk Feature Candidates",
        "",
        "| Dataset | Table | Column | Current Group | Proposed Domain | Family |",
        "|---|---|---|---|---|---|",
    ]
    for row in leakage[:80]:
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | `{row['column']}` | "
            f"{row['current_group']} | {row['semantic_domain']} | {row['statistical_family']} |"
        )
    if len(leakage) > 80:
        lines.append(f"| ... | ... | {len(leakage) - 80} more columns in CSV |  |  |  |")

    lines += [
        "",
        "## Ambiguous Or Changed Group Candidates",
        "",
        "| Dataset | Table | Column | Current | Proposed | Score Margin | Review | Reasons |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    interesting = sorted(
        changed,
        key=lambda r: (
            0 if str(r["review_priority"]).startswith("high") else 1,
            float(r["semantic_score_margin"]),
            r["dataset"],
            r["table"],
            r["column"],
        ),
    )
    for row in interesting[:120]:
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | `{row['column']}` | "
            f"{row['current_group']} | {row['semantic_domain']} | {fmt(row['semantic_score_margin'])} | "
            f"{row['review_priority']} | {', '.join(row['semantic_reasons'][:4])} |"
        )
    if len(interesting) > 120:
        lines.append(f"| ... | ... | {len(interesting) - 120} more columns in CSV |  |  |  |  |  |")

    lines += [
        "",
        "## Aggregate Correlation Evidence",
        "",
        "These summaries are pairwise absolute Spearman correlations among numeric columns. They are grouped by the audit axes, so they can be compared against the informed-prior mechanisms that use within-block and cross-block structure.",
        "",
        "| Axis | Relation | Group A | Group B | Weighted Pairs | Mean | Median | Q75 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    corr_focus = [
        row
        for row in corr_agg
        if row["axis"] in {"semantic_domain", "causal_role"}
        and row["group_a"]
        not in {"metadata", "exclude", "metadata_identifier", "target_response"}
        and row["group_b"]
        not in {"metadata", "exclude", "metadata_identifier", "target_response"}
    ]
    corr_focus.sort(key=lambda r: (r["axis"], r["relation"], -int(r["weighted_n_pairs"])))
    for row in corr_focus[:80]:
        lines.append(
            f"| {row['axis']} | {row['relation']} | `{row['group_a']}` | `{row['group_b']}` | "
            f"{row['weighted_n_pairs']} | {fmt(row['mean_abs_spearman'])} | "
            f"{fmt(row['median_abs_spearman'])} | {fmt(row['q75_abs_spearman'])} |"
        )

    lines += [
        "",
        "## Strongest Feature-Target Associations",
        "",
        "| Dataset | Table | Target | Feature | Domain | Role | Method | Score | N |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for row in top_n(associations, 80):
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | `{row['target']}` | `{row['feature']}` | "
            f"{row['semantic_domain']} | {row['causal_role']} | {row['method']} | {fmt(row['score'])} | {row['n']} |"
        )

    lines += [
        "",
        "## Output Files",
        "",
        "- `feature_inventory.csv`: one row per column with current/proposed labels, type statistics, marginals, and review flags.",
        "- `feature_inventory.json`: same inventory with nested score and top-value fields preserved.",
        "- `table_summary.csv`: table-level counts by group/domain/role/family.",
        "- `group_pair_correlations.csv`: per-table within/cross numeric correlation summaries by grouping axis.",
        "- `group_pair_correlations_aggregate.csv`: aggregated pairwise correlation summaries.",
        "- `feature_target_associations.csv`: feature-to-target association diagnostics.",
        "",
        "## Caveats",
        "",
        "- Candidate labels are heuristic. They should guide review, not replace domain judgment.",
        "- Numeric parsing accepts simple scalar values with units; text formulas and identifiers are intentionally left mostly categorical.",
        "- Correlations are descriptive and can be inflated by repeated-condition tables or identity-like columns.",
        "- A good synthetic prior may need separate semantic, marginal, and causal-role controls rather than a single flat group label.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", action="append", default=[], help="Limit to dataset id; repeatable.")
    parser.add_argument("--min-pair-n", type=int, default=20)
    parser.add_argument("--min-values", type=int, default=20)
    parser.add_argument("--max-categorical-levels", type=int, default=80)
    parser.add_argument("--top-values", type=int, default=8)
    parser.add_argument("--include-targets-in-correlations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_all_tables()
    if args.dataset:
        keep = set(args.dataset)
        tables = [table for table in tables if table.dataset in keep]
    if not tables:
        raise ValueError("No tables matched the requested dataset filter.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    column_rows = annotate_columns(tables, args.top_values)
    table_rows = table_summaries(tables, column_rows)
    corr_rows = group_pair_correlations(
        tables,
        column_rows,
        min_pair_n=args.min_pair_n,
        min_values=args.min_values,
        include_targets=args.include_targets_in_correlations,
    )
    corr_agg = aggregate_group_pair_correlations(corr_rows)
    assoc_rows = feature_target_associations(
        tables,
        column_rows,
        min_pair_n=args.min_pair_n,
        max_levels=args.max_categorical_levels,
    )

    inventory_fields = [
        "dataset",
        "table",
        "column",
        "current_group",
        "semantic_domain",
        "causal_role",
        "statistical_family",
        "data_kind",
        "review_priority",
        "current_vs_proposed_changed",
        "semantic_score",
        "semantic_score_margin",
        "semantic_scores",
        "semantic_reasons",
        "distribution_tags",
        "causal_tags",
        "n_rows",
        "nonmissing_count",
        "nonmissing_ratio",
        "unique_count",
        "unique_ratio_nonmissing",
        "numeric_count",
        "numeric_ratio_nonmissing",
        "numeric_ratio_rows",
        "unique_numeric_count",
        "integer_like_ratio",
        "numeric_min",
        "numeric_q01",
        "numeric_q05",
        "numeric_q25",
        "numeric_median",
        "numeric_q75",
        "numeric_q95",
        "numeric_q99",
        "numeric_max",
        "numeric_mean",
        "numeric_std",
        "numeric_iqr",
        "numeric_skew",
        "positive_ratio",
        "negative_ratio",
        "zero_ratio",
        "composition_set",
        "composition_sum_median",
        "is_element_symbol",
        "avg_text_len",
        "max_text_len",
        "top_values",
    ]

    write_csv(output_dir / "feature_inventory.csv", column_rows, inventory_fields)
    (output_dir / "feature_inventory.json").write_text(
        json.dumps(json_default(column_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(output_dir / "table_summary.csv", table_rows)
    (output_dir / "table_summary.json").write_text(
        json.dumps(json_default(table_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(output_dir / "group_pair_correlations.csv", corr_rows)
    write_csv(output_dir / "group_pair_correlations_aggregate.csv", corr_agg)
    write_csv(output_dir / "feature_target_associations.csv", assoc_rows)
    write_markdown(output_dir / "feature_group_audit.md", table_rows, column_rows, corr_agg, assoc_rows)

    print(f"Wrote feature-group audit to {output_dir}")
    print(f"Columns audited: {len(column_rows)}")
    print(f"Feature-target associations: {len(assoc_rows)}")
    print(f"Group-pair correlation summaries: {len(corr_rows)}")


if __name__ == "__main__":
    main()
