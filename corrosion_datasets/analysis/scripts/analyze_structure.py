#!/usr/bin/env python3
"""Quantitative structural analysis for external corrosion datasets.

The goal is not to fit a corrosion model. The goal is to estimate broad
structural signals that can justify informed-SCM prior settings without using
DatacorTech test performance as a tuning signal.

Dependencies are intentionally light: stdlib + numpy + scipy.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
OUT_JSON = ROOT / "analysis" / "structural_analysis_results.json"
OUT_MD = ROOT / "analysis" / "structural_analysis_results.md"

GROUPS = [
    "material",
    "environment",
    "process_history",
    "exposure_duration",
    "temporal_history",
    "direct_intervention",
    "molecular_descriptor",
    "electrochem_control",
    "electrochem_downstream",
    "target",
    "metadata",
    "exclude",
]

INHIBITOR_DATASETS = {
    "datacor_aluminum_inhibitors",
    "datacortech_aluminum_inhibitors",
    "mg_az91_inhibitors",
    "mg_ze41_inhibitors",
}

DIRECT_INTERVENTION_CONTROL_RE = re.compile(
    r"(inhib.*concentrat|synergistic|encapsulated|dose|dosage|coating|coated)",
    re.IGNORECASE,
)

EXPOSURE_DURATION_RE = re.compile(r"(time|duration|days?|hours?|exposure|immersion)", re.IGNORECASE)
ELECTROCHEM_CONTROL_RE = re.compile(
    r"(test\s+potential|scan[_\s-]*rate|applied\s+potential)",
    re.IGNORECASE,
)

MAX_WITHIN_CORRELATION_COLUMNS = 32

NA_STRINGS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
    "--",
}

RATING_TO_SEVERITY = {
    "a": 0.0,
    "b": 1.0,
    "c": 2.0,
    "d": 3.0,
}

NUMERIC_COLUMN_EXCLUDE_RE = re.compile(
    r"\b("
    r"alloy name|alloy formula|formula|phases? present|phase|"
    r"material group|material family|material$|uns|"
    r"test solution|test environment|test electrolyte|electrolyte$|environment$|"
    r"test method|heat treatment|microstructures?|condition/comment|comment|"
    r"processing|am process|reference|doi|source|id|maps"
    r")\b",
    re.IGNORECASE,
)

EMBEDDED_NUMERIC_COLUMN_RE = re.compile(
    r"\b("
    r"duration|days?|time|scan rate|temperature|concentration|chloride|"
    r"salinity|oxygen|humidity|saturation|water content|pore solution|ph|"
    r"rate|rating|potential|resistivity|current|density|window|ratio|"
    r"porosity|proportion|binder|entropy|vec"
    r")\b",
    re.IGNORECASE,
)

UNIT_SUFFIX_RE = re.compile(
    r"^[\s,;/()°%+\-.]*"
    r"(?:"
    r"m|mol|molar|wt|vol|ppm|ppb|ppt|"
    r"c|f|k|d|day|days|h|hr|hrs|hour|hours|min|s|sec|"
    r"v|mv|a|ma|ua|µa|ohm|cm2|cm|mm|yr|year|years|"
    r"max|min|approx|approximately|about"
    r")*"
    r"[\s,;/()°%+\-.]*$",
    re.IGNORECASE,
)


@dataclass
class Table:
    dataset: str
    table: str
    columns: list[str]
    groups: dict[str, str]
    rows: list[dict[str, Any]]
    notes: list[str] | None = None


def clean_name(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, col in enumerate(columns):
        base = clean_name(col) or f"unnamed_{i}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}__{count + 1}")
    return out


def normalize_numeric_text(text: str) -> str:
    text = text.strip().replace("−", "-").replace("–", "-").replace("—", "-")
    text = "".join(ch for ch in text if ch not in {"\u202a", "\u202b", "\u202c", "\u202d", "\u202e"})
    # Only treat commas as thousands separators in conventional numeric groups.
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return text


def to_float(value: Any, *, allow_embedded_number: bool = False, allow_rating: bool = False) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_numeric_text(clean_name(value))
    if text.lower() in NA_STRINGS:
        return math.nan
    # Ratings in CORR-DATA are ordered from resistant to poor.
    lower = text.lower()
    if allow_rating and lower and lower[0] in RATING_TO_SEVERITY and re.match(r"^[abcd]\b", lower):
        return RATING_TO_SEVERITY[lower[0]]
    # Common range form, e.g. 200-400.
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*$", text)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.match(r"^\s*[<>=~]*\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", text)
    if m:
        return float(m.group(1))
    try:
        return float(text)
    except ValueError:
        pass
    if not allow_embedded_number:
        return math.nan
    # Accept scalar values with simple measurement suffixes, such as
    # "0.05 max" or "414 d", while rejecting IDs/prose like "S30400".
    if not re.match(r"^\s*[<>=~]*\s*[+-]?\d", text):
        return math.nan
    found = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(found) == 1:
        tail = text[text.find(found[0]) + len(found[0]) :]
        if not UNIT_SUFFIX_RE.match(tail):
            return math.nan
        return float(found[0])
    return math.nan


def xlsx_rows_from_bytes(data: bytes, sheet_name: str | None = None) -> dict[str, list[list[str]]]:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "officeRel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    def col_to_idx(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha())
        idx = 0
        for ch in letters:
            idx = idx * 26 + ord(ch.upper()) - 64
        return idx - 1

    def load_shared(zf: zipfile.ZipFile) -> list[str]:
        try:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        except KeyError:
            return []
        vals = []
        for si in root.findall("main:si", ns):
            vals.append("".join((t.text or "") for t in si.iter("{%s}t" % ns["main"])))
        return vals

    def workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall("rel:Relationship", ns)}
        out = []
        for sh in wb.findall(".//main:sheet", ns):
            name = sh.attrib.get("name", "")
            rid = sh.attrib.get("{%s}id" % ns["officeRel"])
            target = rid_to_target.get(rid, "")
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            if sheet_name is None or name == sheet_name:
                out.append((name, target))
        return out

    def cell_value(cell: ET.Element, shared: list[str]) -> str:
        ctype = cell.attrib.get("t")
        if ctype == "inlineStr":
            return "".join((x.text or "") for x in cell.iter("{%s}t" % ns["main"]))
        v = cell.find("main:v", ns)
        if v is None or v.text is None:
            return ""
        raw = v.text
        if ctype == "s":
            try:
                return shared[int(raw)]
            except Exception:
                return raw
        return raw

    result: dict[str, list[list[str]]] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        shared = load_shared(zf)
        for sname, path in workbook_sheets(zf):
            root = ET.fromstring(zf.read(path))
            rows: list[list[str]] = []
            for row in root.findall(".//main:sheetData/main:row", ns):
                vals: dict[int, str] = {}
                for cell in row.findall("main:c", ns):
                    ref = cell.attrib.get("r", "")
                    idx = col_to_idx(ref) if ref else len(vals)
                    vals[idx] = clean_name(cell_value(cell, shared))
                if vals:
                    width = max(vals) + 1
                    rows.append([vals.get(i, "") for i in range(width)])
                else:
                    rows.append([])
            result[sname] = rows
    return result


def xlsx_rows(path: Path, sheet_name: str | None = None) -> dict[str, list[list[str]]]:
    return xlsx_rows_from_bytes(path.read_bytes(), sheet_name)


def xlsx_rows_from_zip(path: Path, internal_name: str, sheet_name: str | None = None) -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(path) as zf:
        return xlsx_rows_from_bytes(zf.read(internal_name), sheet_name)


def csv_rows_from_zip(path: Path, internal_name: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        data = zf.read(internal_name)
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:5000])
    except Exception:
        dialect = csv.excel
    return [[clean_name(c) for c in row] for row in csv.reader(io.StringIO(text), dialect)]


def csv_rows(path: Path, delimiter: str | None = None) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(text[:5000])
        except Exception:
            dialect = csv.excel
        return [[clean_name(c) for c in row] for row in csv.reader(io.StringIO(text), dialect)]
    return [[clean_name(c) for c in row] for row in csv.reader(io.StringIO(text), delimiter=delimiter)]


def docx_tables(path: Path) -> list[list[list[str]]]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    tables = []
    for tbl in root.findall(".//w:tbl", ns):
        rows = []
        for tr in tbl.findall("./w:tr", ns):
            row = []
            for tc in tr.findall("./w:tc", ns):
                texts = [t.text or "" for t in tc.findall(".//w:t", ns)]
                row.append(clean_name(" ".join("".join(texts).split())))
            if any(row):
                rows.append(row)
        if rows:
            tables.append(rows)
    return tables


def decimal_comma_to_dot(value: Any) -> Any:
    text = clean_name(value)
    if re.fullmatch(r"[+-]?\d+,\d+(?:[eE][+-]?\d+)?", text):
        return text.replace(",", ".")
    return value


def rows_to_dicts(columns: list[str], rows: list[list[Any]], start: int = 1) -> list[dict[str, Any]]:
    cols = unique_columns(columns)
    out = []
    for row in rows[start:]:
        if not any(clean_name(v) for v in row):
            continue
        out.append({col: row[i] if i < len(row) else "" for i, col in enumerate(cols)})
    return out


def make_table(dataset: str, table: str, columns: list[str], groups_by_col: dict[str, str], rows: list[list[Any]], start: int) -> Table:
    cols = unique_columns(columns)
    fixed_groups = {col: groups_by_col.get(col, "metadata") for col in cols}
    dict_rows = []
    for row in rows[start:]:
        if not any(clean_name(v) for v in row):
            continue
        dict_rows.append({col: row[i] if i < len(row) else "" for i, col in enumerate(cols)})
    return Table(dataset, table, cols, fixed_groups, dict_rows)


def audit_v2_group(table: Table, column: str, current_group: str) -> str:
    """Map the first-pass loader labels into the audit-derived ontology."""
    if current_group in {"target", "metadata", "exclude", "material", "environment"}:
        return current_group

    if table.dataset in INHIBITOR_DATASETS:
        direct_inhibitor_controls = {
            "Inhib_Concentrat_M",
            "Synergistic_inhib",
            "Synergistic_inhib_type",
            "Synergistic_inhib_Concentrat_M",
            "Encapsulated",
        }
        if current_group == "intervention":
            return "direct_intervention" if column in direct_inhibitor_controls else "molecular_descriptor"
        if current_group == "history":
            return "exposure_duration" if EXPOSURE_DURATION_RE.search(column) else "process_history"
        if current_group == "electrochem":
            return "electrochem_downstream"
        return current_group

    if current_group == "intervention":
        return "direct_intervention" if DIRECT_INTERVENTION_CONTROL_RE.search(column) else "process_history"

    if current_group == "history":
        if table.dataset == "mooring_steel_seawater" and column == "days":
            return "temporal_history"
        if ELECTROCHEM_CONTROL_RE.search(column):
            return "electrochem_control"
        if "heat treatment" in column.lower():
            return "direct_intervention"
        if EXPOSURE_DURATION_RE.search(column):
            return "exposure_duration"
        return "process_history"

    if current_group == "electrochem":
        return "electrochem_downstream"

    return current_group


def apply_audit_v2_groups(table: Table) -> Table:
    groups = {
        col: audit_v2_group(table, col, table.groups.get(col, "metadata"))
        for col in table.columns
    }
    return Table(table.dataset, table.table, table.columns, groups, table.rows, table.notes)


def group_by_positions(columns: list[str], position_groups: dict[str, list[int]]) -> dict[str, str]:
    cols = unique_columns(columns)
    groups = {col: "metadata" for col in cols}
    for group, idxs in position_groups.items():
        for idx in idxs:
            if 0 <= idx < len(cols):
                groups[cols[idx]] = group
    return groups


def combine_two_header_rows(row1: list[str], row2: list[str]) -> list[str]:
    width = max(len(row1), len(row2))
    out = []
    active = ""
    for i in range(width):
        top = clean_name(row1[i] if i < len(row1) else "")
        sub = clean_name(row2[i] if i < len(row2) else "")
        if top:
            active = top
        if active and sub:
            if sub.lower() in {"no", "no.", ""}:
                out.append(sub or active)
            elif active == sub:
                out.append(sub)
            else:
                out.append(f"{active} {sub}")
        else:
            out.append(sub or active or f"unnamed_{i}")
    return unique_columns(out)


def load_electrochemical_metrics() -> list[Table]:
    path = DATASETS / "electrochemical_metrics_alloys" / "raw" / "CRA_database_Scientific_Data_Publication_12102020.xlsx"
    sheets = xlsx_rows(path)
    configs = {
        "Pitting Potential": {
            "material": list(range(1, 25)),
            "target": [25, 26, 27],
            "environment": [28, 29, 30, 31],
            "history": [32, 33, 34, 35],
            "metadata": [0, 36, 37, 38],
        },
        "Repassivation Potential": {
            "material": list(range(1, 25)),
            "target": [25],
            "environment": [26, 27, 28, 29],
            "history": [30, 31, 32, 33],
            "metadata": [0, 34, 35, 36],
        },
        "Crevice Corrosion Potential": {
            "material": list(range(1, 15)),
            "target": [15, 16, 17],
            "environment": [18, 19, 20, 21],
            "history": [22, 23, 24, 25],
            "metadata": [0, 26, 27, 28],
        },
        "Pitting Temp": {
            "material": list(range(1, 15)),
            "target": [15, 16, 17],
            "environment": [18],
            "history": [19, 20, 21, 22, 23],
            "metadata": [0, 24, 25, 26],
        },
        "Crevice Corrosion Temp": {
            "material": list(range(1, 21)),
            "target": [21, 22, 23],
            "environment": [24],
            "history": [25, 26, 27, 28, 29],
            "metadata": [0, 30, 31, 32],
        },
        "HEAs_Ecorr, icorr, ipass, Rate": {
            "material": list(range(1, 22)),
            "target": [22, 23, 24, 25, 26, 27],
            "electrochem": [28, 29],
            "environment": [30, 31, 32, 33],
            "history": [34, 35, 36, 37],
            "metadata": [0, 38, 39, 40],
        },
    }
    tables = []
    for sheet, pos in configs.items():
        rows = sheets.get(sheet, [])
        if len(rows) < 3:
            continue
        columns = combine_two_header_rows(rows[0], rows[1])
        groups = group_by_positions(columns, pos)
        tables.append(make_table("electrochemical_metrics_alloys", sheet, columns, groups, rows, 2))
    return tables


def load_mpea() -> list[Table]:
    path = DATASETS / "mpea_corrosion" / "raw" / "Mendeley_MPEA_corrosion_database.xlsx"
    rows = xlsx_rows(path)["Sheet1"]
    columns = unique_columns(rows[0])
    groups = {col: "metadata" for col in columns}
    for col in columns:
        if col in {"FCC", "BCC", "HCP", "IM"} or col in {
            "Al",
            "B",
            "Be",
            "Co",
            "Cr",
            "Cu",
            "Fe",
            "Ga",
            "Hf",
            "La",
            "Mg",
            "Mn",
            "Mo",
            "Nb",
            "Ni",
            "Si",
            "Sn",
            "Ta",
            "Ti",
            "V",
            "W",
            "Y",
            "Zn",
            "Zr",
        }:
            groups[col] = "material"
        elif col in {"Phases present"}:
            groups[col] = "material"
        elif col in {"Test environment", "Concentration in M", "Electrolyte"}:
            groups[col] = "environment"
        elif col == "Processing":
            groups[col] = "intervention"
        elif col in {
            "Corrosion potential (mV vs SCE)",
            "Pitting potential (mV vs SCE)",
            "Corrosion current density (microA/cm2)",
            "Calculated passive window",
        }:
            groups[col] = "target"
    return [make_table("mpea_corrosion", "Sheet1", columns, groups, rows, 1)]


def load_am_mpea() -> list[Table]:
    archive = DATASETS / "am_mpea_corrosion" / "raw" / "Additively Manufactured Multi Principal Element Alloy Corrosion Database.zip"
    internal = "Additively Manufactured Multi Principal Element Alloy Corrosion Database/AM_MPEA_corrosion_database_V3.xlsx"
    rows = xlsx_rows_from_zip(archive, internal)["Sheet1"]
    columns = unique_columns(rows[0])
    groups = {col: "metadata" for col in columns}
    for col in columns:
        if col in {
            "Alloy name",
            "Alloy formula",
            "Phases present",
            "Al",
            "Co",
            "Cr",
            "Cu",
            "Fe",
            "Mn",
            "Mo",
            "Nb",
            "Ni",
            "Si",
            "Ti",
            "Entropy of mixing (J/K.mol)",
            "Average VEC",
            "St. Deviation of VEC",
        }:
            groups[col] = "material"
        elif col in {"Test electrolyte", "Electrolyte concentration (M)"}:
            groups[col] = "environment"
        elif col == "AM process":
            groups[col] = "intervention"
        elif col in {
            "Corrosion potential (mV vs. SCE)",
            "Pitting potential (mV vs. SCE)",
            "Corrosion current density (µA/cm2)",
        }:
            groups[col] = "target"
    return [make_table("am_mpea_corrosion", "AM_MPEA_corrosion_database_V3", columns, groups, rows, 1)]


def load_steel_mortar() -> list[Table]:
    path = DATASETS / "steel_mortar_corrosion" / "raw" / "Dataset of Steel Corrosion in Cementitious Mortar Due to Carbonation and Chlorides.xlsx"
    sheets = xlsx_rows(path)
    tables = []
    for sheet_name, rows in sheets.items():
        if len(rows) < 4:
            continue
        if sheet_name == "01_Carbonation":
            headers = [
                "Cement Proportion",
                "GGBS Proportion",
                "Fly Ash Proportion",
                "Silic Fume Proportion",
                "Water to Binder Ratio",
                "Incorporated Chloride Concentration of mortar",
                "Free Chloride Concentration of mortar",
                "pH value of Pore Solution of mortar",
                "Porosity of mortar",
                "Relative Humidity of Environment",
                "Degree of Saturation of mortar",
                "Water Content of mortar",
                "Corrosion Potential of Steel vs Cu/CuSO4",
                "Electrical Resistivity of mortar",
                "Chloride-to-hydroxide concentration ratio",
                "Corrosion Rate of Steel",
                "Notes",
            ]
            pos = {
                "material": [0, 1, 2, 3, 4, 8],
                "environment": [5, 6, 7, 9, 10, 11],
                "target": [12, 13, 14, 15],
                "metadata": [16],
            }
        else:
            headers = [
                "Cement Proportion",
                "GGBS Proportion",
                "Fly Ash Proportion",
                "Water to Binder Ratio",
                "Incorporated Chloride Concentration of mortar",
                "Free Chloride Concentration of mortar",
                "pH value of Pore Solution of mortar",
                "Porosity of mortar",
                "Relative Humidity of Environment",
                "Degree of Saturation of mortar",
                "Water Content of mortar",
                "Corrosion Potential of Steel vs Cu/CuSO4",
                "Electrical Resistivity of mortar",
                "Chloride-to-hydroxide concentration ratio",
                "Corrosion Rate of Steel",
                "Notes",
            ]
            pos = {
                "material": [0, 1, 2, 3, 7],
                "environment": [4, 5, 6, 8, 9, 10],
                "target": [11, 12, 13, 14],
                "metadata": [15],
            }
        groups = group_by_positions(headers, pos)
        tables.append(make_table("steel_mortar_corrosion", sheet_name, headers, groups, rows, 3))
    return tables


def parse_316l_condition(filename: str) -> tuple[float, float]:
    # Examples: Epit_Epass_005_50.csv, logj_uA-cm2_0.05_M_NaCl_50_mV_per_s.csv
    base = Path(filename).name
    if "0.05" in base or "_005_" in base:
        chloride = 0.05
    elif "0.01" in base or "_001_" in base:
        chloride = 0.01
    elif "0.005" in base or "_0005_" in base:
        chloride = 0.005
    else:
        chloride = math.nan
    scan_rate = 100.0 if "100" in base else 50.0 if "50" in base else math.nan
    return chloride, scan_rate


def load_316l_descriptors() -> list[Table]:
    archive = DATASETS / "316l_pitting_passivity" / "raw" / "Epit and Epass descriptors of 316L stainless steel estimated by Machine Learning.zip"
    out_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            rows = [[clean_name(c) for c in row] for row in csv.reader(io.StringIO(zf.read(name).decode("utf-8-sig")))]
            if not rows:
                continue
            chloride, scan_rate = parse_316l_condition(name)
            columns = unique_columns(rows[0])
            for row in rows[1:]:
                if not any(row):
                    continue
                record = {col: row[i] if i < len(row) else "" for i, col in enumerate(columns)}
                record["chloride_M"] = chloride
                record["scan_rate_mV_s"] = scan_rate
                record["source_file"] = Path(name).name
                out_rows.append(record)
    cols = ["chloride_M", "scan_rate_mV_s", "Maps", "Epit_x", "Epit_y", "Epass_x", "Epass_y", "source_file"]
    groups = {
        "chloride_M": "environment",
        "scan_rate_mV_s": "history",
        "Maps": "metadata",
        "Epit_x": "target",
        "Epit_y": "target",
        "Epass_x": "target",
        "Epass_y": "target",
        "source_file": "metadata",
    }
    return [Table("316l_pitting_passivity", "Epit_Epass_descriptors", cols, groups, out_rows)]


def load_nace() -> list[Table]:
    archive = DATASETS / "nace_nist_corr_data" / "raw" / "CORR-DATA_Database.zip"
    rows = csv_rows_from_zip(archive, "CORR-DATA_Database.csv")
    columns = unique_columns(rows[0])
    groups = {col: "metadata" for col in columns}
    for col in columns:
        if col in {"Material Group", "Material Family", "Material", "UNS"}:
            groups[col] = "material"
        elif col in {"Environment", "Concentration (Vol %)", "Temperature (deg C)", "Temperature (deg F)"}:
            groups[col] = "environment"
        elif col in {"Duration", "Condition/Comment"}:
            groups[col] = "history"
        elif col in {"Rate (mm/yr) or Rating", "Rate (mils/yr) or Rating", "Localized Attack"}:
            groups[col] = "target"
    return [make_table("nace_nist_corr_data", "CORR-DATA_Database", columns, groups, rows, 1)]


def load_mooring() -> list[Table]:
    archive_dir = DATASETS / "mooring_steel_seawater" / "raw"
    matches = sorted(archive_dir.glob("Data for The effect of environmental variables on corrosion*.zip"))
    if not matches:
        raise FileNotFoundError(f"No mooring steel data archive found in {archive_dir}")
    archive = matches[0]
    with zipfile.ZipFile(archive) as zf:
        internal_matches = [name for name in zf.namelist() if name.endswith("OCP R4 S31 Omax.xlsx")]
    if not internal_matches:
        raise FileNotFoundError(f"No OCP workbook found inside {archive}")
    rows = xlsx_rows_from_zip(archive, internal_matches[0])["Hoja1"]
    # Blocks observed in the workbook: A:B, E:F, I:J.
    blocks = [
        (0, 1, 31.0, 2.0, 1.0, "S=31 T=2 Omax"),
        (4, 5, 31.0, 32.0, 1.0, "S=31 T=32 Omax"),
        (8, 9, 31.0, 17.0, 1.0, "S=31 T=17 Omax"),
    ]
    out_rows = []
    for day_col, v_col, salinity, temp, oxygen, condition in blocks:
        for row in rows[6:]:
            day = row[day_col] if day_col < len(row) else ""
            ocp = row[v_col] if v_col < len(row) else ""
            if math.isnan(to_float(day)) or math.isnan(to_float(ocp)):
                continue
            out_rows.append(
                {
                    "condition": condition,
                    "salinity_pss": salinity,
                    "temperature_C": temp,
                    "oxygen_omax": oxygen,
                    "days": day,
                    "ocp_V": ocp,
                }
            )
    columns = ["condition", "salinity_pss", "temperature_C", "oxygen_omax", "days", "ocp_V"]
    groups = {
        "condition": "metadata",
        "salinity_pss": "environment",
        "temperature_C": "environment",
        "oxygen_omax": "environment",
        "days": "history",
        "ocp_V": "target",
    }
    return [Table("mooring_steel_seawater", "OCP_R4_S31_Omax", columns, groups, out_rows)]


def load_datacor_aluminum_inhibitors() -> list[Table]:
    path = DATASETS / "datacor_aluminum_inhibitors" / "raw" / "DataCor.xlsx"
    rows = xlsx_rows(path)["Main"]
    wide_columns = unique_columns(rows[0])
    wide_records = rows_to_dicts(wide_columns, rows, 1)

    target_pattern = re.compile(r"^Efficiency_(AA\d+)_pH(\d+)$")
    target_cols = [col for col in wide_columns if target_pattern.match(col)]
    descriptor_cols = [col for col in wide_columns if col not in {"Inhibitor", "number", *target_cols}]

    out_rows: list[dict[str, Any]] = []
    for record in wide_records:
        base = {
            "Inhibitor": record.get("Inhibitor", ""),
            "number": record.get("number", ""),
        }
        for col in descriptor_cols:
            base[col] = record.get(col, "")
        for target_col in target_cols:
            match = target_pattern.match(target_col)
            if not match:
                continue
            alloy, ph = match.groups()
            row = dict(base)
            row["alloy"] = alloy
            row["pH"] = ph
            row["efficiency_score"] = record.get(target_col, "")
            row["source_efficiency_column"] = target_col
            out_rows.append(row)

    columns = ["Inhibitor", "number", "alloy", "pH", *descriptor_cols, "efficiency_score", "source_efficiency_column"]
    groups = {col: "intervention" for col in columns}
    groups.update(
        {
            "Inhibitor": "metadata",
            "number": "metadata",
            "alloy": "material",
            "pH": "environment",
            "efficiency_score": "target",
            "source_efficiency_column": "metadata",
        }
    )
    return [Table("datacor_aluminum_inhibitors", "DataCor_long_efficiency", columns, groups, out_rows)]


def load_datacortech_aluminum_inhibitors() -> list[Table]:
    initial_path = DATASETS / "datacortech_aluminum_inhibitors" / "raw" / "Datacortech_initial_data.xlsx"
    final_path = DATASETS / "datacortech_aluminum_inhibitors" / "raw" / "Datacortech_final_data.xlsx"

    initial_rows = xlsx_rows(initial_path)["Efficiencies"]
    initial_columns = unique_columns(initial_rows[0])
    initial_records = rows_to_dicts(initial_columns, initial_rows, 1)

    final_rows = xlsx_rows(final_path)["Sheet 1"]
    final_columns = unique_columns(final_rows[0])
    descriptor_start = final_columns.index("Contributor") + 1 if "Contributor" in final_columns else len(final_columns)
    descriptor_cols = final_columns[descriptor_start:]
    descriptor_by_number: dict[str, dict[str, Any]] = {}
    for record in rows_to_dicts(final_columns, final_rows, 1):
        number = clean_name(record.get("Number"))
        if not number or number in descriptor_by_number:
            continue
        descriptor_by_number[number] = {col: record.get(col, "") for col in descriptor_cols}

    out_rows = []
    for record in initial_records:
        number = clean_name(record.get("Number"))
        out = dict(record)
        out["descriptor_source"] = "Datacortech_final_data.xlsx" if number in descriptor_by_number else ""
        out.update(descriptor_by_number.get(number, {col: "" for col in descriptor_cols}))
        out_rows.append(out)

    columns = [*initial_columns, "descriptor_source", *descriptor_cols]
    groups = {col: "metadata" for col in columns}
    for col in columns:
        if col in {"Metal", "Alloy"}:
            groups[col] = "material"
        elif col in {"Temperature_K", "pH", "Salt_Concentrat_M"}:
            groups[col] = "environment"
        elif col in {"Time_h"}:
            groups[col] = "history"
        elif col in {
            "Inhib_Concentrat_M",
            "Synergistic_inhib",
            "Synergistic_inhib_type",
            "Synergistic_inhib_Concentrat_M",
            "Encapsulated",
        } or col in descriptor_cols:
            groups[col] = "intervention"
        elif col == "Efficiency":
            groups[col] = "target"
    return [Table("datacortech_aluminum_inhibitors", "Efficiencies_with_descriptors", columns, groups, out_rows)]


def load_mg_az91_inhibitors() -> list[Table]:
    path = DATASETS / "mg_az91_inhibitors" / "raw" / "Features.csv"
    rows = csv_rows(path, delimiter=";")
    columns = unique_columns(["substrate_alloy", *rows[0]])
    out_rows = []
    for row in rows[1:]:
        if not any(row):
            continue
        record = {"substrate_alloy": "AZ91"}
        for idx, col in enumerate(columns[1:]):
            record[col] = row[idx] if idx < len(row) else ""
        out_rows.append(record)
    groups = {col: "intervention" for col in columns}
    groups.update({"substrate_alloy": "material", "No.": "metadata", "NAME": "metadata", "IE": "target"})
    return [Table("mg_az91_inhibitors", "Features", columns, groups, out_rows)]


def load_mg_ze41_inhibitors() -> list[Table]:
    path = DATASETS / "mg_ze41_inhibitors" / "raw" / "ze41_mol_desc_db_red.csv"
    rows = csv_rows(path, delimiter=";")
    columns = unique_columns(["substrate_alloy", *rows[0]])
    out_rows = []
    for row in rows[1:]:
        if not any(row):
            continue
        record = {"substrate_alloy": "ZE41"}
        for idx, col in enumerate(columns[1:]):
            value = row[idx] if idx < len(row) else ""
            record[col] = value if idx == 0 else decimal_comma_to_dot(value)
        out_rows.append(record)
    groups = {col: "intervention" for col in columns}
    groups.update(
        {
            "substrate_alloy": "material",
            "compound": "metadata",
            "inhibition efficiency ZE41 / %": "target",
            "LinIE ZE41": "exclude",
        }
    )
    return [Table("mg_ze41_inhibitors", "ze41_mol_desc_db_red", columns, groups, out_rows)]


def load_ni_crevice_repassivation() -> list[Table]:
    path = DATASETS / "ni_crevice_repassivation" / "raw" / "Research_data_Saenzetal.docx"
    tables = docx_tables(path)
    if not tables:
        return []
    rows = tables[0]
    columns = unique_columns(rows[0])
    groups = {col: "metadata" for col in columns}
    for col in columns:
        if col == "Alloy":
            groups[col] = "material"
        elif col in {"T. °C", "[Cl-]. mol/L", "[SO42-]. mol/L", "[NO3-]. mol/L", "[MoO42-]. mol/L"}:
            groups[col] = "environment"
        elif col == "iGS. µA/cm2":
            groups[col] = "electrochem"
        elif col == "ER.CREV. VECS":
            groups[col] = "target"
        elif col == "CC Attack?":
            groups[col] = "exclude"
    return [make_table("ni_crevice_repassivation", "Research_data_Saenzetal_table1", columns, groups, rows, 1)]


def load_all_tables() -> list[Table]:
    tables: list[Table] = []
    for loader in [
        load_electrochemical_metrics,
        load_mpea,
        load_am_mpea,
        load_steel_mortar,
        load_316l_descriptors,
        load_nace,
        load_mooring,
        load_datacor_aluminum_inhibitors,
        load_datacortech_aluminum_inhibitors,
        load_mg_az91_inhibitors,
        load_mg_ze41_inhibitors,
        load_ni_crevice_repassivation,
    ]:
        tables.extend(apply_audit_v2_groups(table) for table in loader())
    return tables


def is_numeric_measurement_column(table: Table, column: str) -> bool:
    group = table.groups.get(column, "metadata")
    if group in {"metadata", "exclude", "molecular_descriptor"}:
        return False
    if NUMERIC_COLUMN_EXCLUDE_RE.search(column):
        return False
    return True


def allows_embedded_number(table: Table, column: str) -> bool:
    if not is_numeric_measurement_column(table, column):
        return False
    return bool(EMBEDDED_NUMERIC_COLUMN_RE.search(column))


def allows_rating(table: Table, column: str) -> bool:
    group = table.groups.get(column, "metadata")
    return group == "target" and "rating" in column.lower()


def numeric_vector(table: Table, column: str) -> np.ndarray:
    if not is_numeric_measurement_column(table, column):
        return np.full(len(table.rows), math.nan, dtype=float)
    return np.array(
        [
            to_float(
                row.get(column),
                allow_embedded_number=allows_embedded_number(table, column),
                allow_rating=allows_rating(table, column),
            )
            for row in table.rows
        ],
        dtype=float,
    )


def text_vector(table: Table, column: str) -> list[str]:
    return [clean_name(row.get(column)) for row in table.rows]


def valid_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def abs_spearman(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> tuple[float, int]:
    xv, yv = valid_pair(x, y)
    if len(xv) < min_n:
        return math.nan, len(xv)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return math.nan, len(xv)
    rho = stats.spearmanr(xv, yv).correlation
    if rho is None or not np.isfinite(rho):
        return math.nan, len(xv)
    return abs(float(rho)), len(xv)


def summarize_values(values: list[float]) -> dict[str, float | int]:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": math.nan, "median": math.nan, "q75": math.nan}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "q75": float(np.quantile(vals, 0.75)),
    }


def numeric_columns_by_group(table: Table, min_values: int = 20) -> dict[str, list[str]]:
    out = {g: [] for g in GROUPS}
    for col in table.columns:
        group = table.groups.get(col, "metadata")
        vec = numeric_vector(table, col)
        if np.isfinite(vec).sum() >= min_values and np.nanstd(vec) > 0:
            out[group].append(col)
    return out


def categorical_columns_by_group(table: Table, min_values: int = 20, max_unique: int = 80) -> dict[str, list[str]]:
    out = {g: [] for g in GROUPS}
    for col in table.columns:
        group = table.groups.get(col, "metadata")
        vals = [v for v in text_vector(table, col) if v and v.lower() not in NA_STRINGS]
        unique = set(vals)
        if len(vals) >= min_values and 2 <= len(unique) <= max_unique:
            out[group].append(col)
    return out


def within_block_correlations(tables: list[Table]) -> list[dict[str, Any]]:
    results = []
    for table in tables:
        by_group = numeric_columns_by_group(table)
        for group in [
            "material",
            "environment",
            "process_history",
            "exposure_duration",
            "temporal_history",
            "direct_intervention",
            "electrochem_control",
            "electrochem_downstream",
            "target",
        ]:
            cols = by_group.get(group, [])
            n_numeric_columns_total = len(cols)
            if len(cols) > MAX_WITHIN_CORRELATION_COLUMNS:
                cols = cols[:MAX_WITHIN_CORRELATION_COLUMNS]
            vals, ns = [], []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    rho, n = abs_spearman(numeric_vector(table, cols[i]), numeric_vector(table, cols[j]))
                    if np.isfinite(rho):
                        vals.append(rho)
                        ns.append(n)
            if vals:
                summary = summarize_values(vals)
                results.append(
                    {
                        "dataset": table.dataset,
                        "table": table.table,
                        "group": group,
                        "n_numeric_columns": len(cols),
                        "n_numeric_columns_total": n_numeric_columns_total,
                        "n_pairs": len(vals),
                        "mean_abs_spearman": summary["mean"],
                        "median_abs_spearman": summary["median"],
                        "q75_abs_spearman": summary["q75"],
                        "median_pair_n": float(statistics.median(ns)) if ns else math.nan,
                    }
                )
    return results


def cross_block_correlations(tables: list[Table]) -> list[dict[str, Any]]:
    pairs = [
        ("material", "environment"),
        ("material", "process_history"),
        ("material", "exposure_duration"),
        ("environment", "process_history"),
        ("environment", "exposure_duration"),
        ("environment", "direct_intervention"),
    ]
    results = []
    for table in tables:
        by_group = numeric_columns_by_group(table)
        for g1, g2 in pairs:
            cols1, cols2 = by_group.get(g1, []), by_group.get(g2, [])
            vals, ns = [], []
            for c1 in cols1:
                for c2 in cols2:
                    rho, n = abs_spearman(numeric_vector(table, c1), numeric_vector(table, c2))
                    if np.isfinite(rho):
                        vals.append(rho)
                        ns.append(n)
            if vals:
                summary = summarize_values(vals)
                results.append(
                    {
                        "dataset": table.dataset,
                        "table": table.table,
                        "group_pair": f"{g1}:{g2}",
                        "n_pairs": len(vals),
                        "mean_abs_spearman": summary["mean"],
                        "median_abs_spearman": summary["median"],
                        "q75_abs_spearman": summary["q75"],
                        "median_pair_n": float(statistics.median(ns)) if ns else math.nan,
                    }
                )
    return results


def eta_squared(categories: list[str], y: np.ndarray, min_n: int = 20) -> tuple[float, int, int]:
    vals = []
    cats = []
    for cat, yy in zip(categories, y):
        if cat and cat.lower() not in NA_STRINGS and np.isfinite(yy):
            cats.append(cat)
            vals.append(float(yy))
    if len(vals) < min_n:
        return math.nan, len(vals), 0
    groups: dict[str, list[float]] = {}
    for cat, yy in zip(cats, vals):
        groups.setdefault(cat, []).append(yy)
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if len(groups) < 2:
        return math.nan, len(vals), len(groups)
    all_vals = np.array([v for group_vals in groups.values() for v in group_vals], dtype=float)
    grand = float(np.mean(all_vals))
    ss_total = float(np.sum((all_vals - grand) ** 2))
    if ss_total <= 0:
        return math.nan, len(all_vals), len(groups)
    ss_between = sum(len(v) * (float(np.mean(v)) - grand) ** 2 for v in groups.values())
    return float(ss_between / ss_total), len(all_vals), len(groups)


def target_associations(tables: list[Table]) -> list[dict[str, Any]]:
    results = []
    for table in tables:
        numeric_by_group = numeric_columns_by_group(table)
        cat_by_group = categorical_columns_by_group(table)
        targets = numeric_by_group.get("target", [])
        for target in targets:
            y = numeric_vector(table, target)
            for group in [
                "material",
                "environment",
                "process_history",
                "exposure_duration",
                "temporal_history",
                "direct_intervention",
                "electrochem_control",
                "electrochem_downstream",
            ]:
                vals = []
                best = None
                for col in numeric_by_group.get(group, []):
                    rho, n = abs_spearman(numeric_vector(table, col), y)
                    if np.isfinite(rho):
                        vals.append(rho)
                        if best is None or rho > best["score"]:
                            best = {"column": col, "score": rho, "method": "spearman", "n": n}
                eta_vals = []
                for col in cat_by_group.get(group, []):
                    eta, n, n_levels = eta_squared(text_vector(table, col), y)
                    if np.isfinite(eta):
                        eta_vals.append(eta)
                        if best is None or eta > best["score"]:
                            best = {"column": col, "score": eta, "method": "eta_squared", "n": n, "n_levels": n_levels}
                if vals or eta_vals:
                    results.append(
                        {
                            "dataset": table.dataset,
                            "table": table.table,
                            "target": target,
                            "group": group,
                            "n_numeric_associations": len(vals),
                            "mean_abs_spearman": summarize_values(vals)["mean"],
                            "max_abs_spearman": max(vals) if vals else math.nan,
                            "n_categorical_associations": len(eta_vals),
                            "mean_eta_squared": summarize_values(eta_vals)["mean"],
                            "max_eta_squared": max(eta_vals) if eta_vals else math.nan,
                            "best_column": best["column"] if best else "",
                            "best_method": best["method"] if best else "",
                            "best_score": best["score"] if best else math.nan,
                        }
                    )
    return results


def ridge_cv_r2(x: np.ndarray, y: np.ndarray, folds: int = 5, lam: float = 1.0) -> float:
    n = len(y)
    if n < 30:
        return math.nan
    order = np.arange(n)
    scores = []
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test, assume_unique=True)
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_train = np.column_stack([np.ones(len(train)), x_train])
        x_test = np.column_stack([np.ones(len(test)), x_test])
        penalty = np.eye(x_train.shape[1]) * lam
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y_train
        pred = x_test @ beta
        denom = float(np.sum((y_test - np.mean(y_train)) ** 2))
        if denom <= 0:
            continue
        scores.append(1.0 - float(np.sum((y_test - pred) ** 2)) / denom)
    return float(np.mean(scores)) if scores else math.nan


def interaction_probes(tables: list[Table]) -> list[dict[str, Any]]:
    results = []
    for table in tables:
        by_group = numeric_columns_by_group(table)
        material_cols = by_group.get("material", [])
        env_cols = by_group.get("environment", [])
        targets = by_group.get("target", [])
        if not material_cols or not env_cols or not targets:
            continue
        for target in targets:
            y_all = numeric_vector(table, target)
            # Select a few columns with strongest univariate association to target.
            scored_mat = []
            scored_env = []
            for col in material_cols:
                rho, _ = abs_spearman(numeric_vector(table, col), y_all)
                if np.isfinite(rho):
                    scored_mat.append((rho, col))
            for col in env_cols:
                rho, _ = abs_spearman(numeric_vector(table, col), y_all)
                if np.isfinite(rho):
                    scored_env.append((rho, col))
            mat_keep = [c for _, c in sorted(scored_mat, reverse=True)[:8]]
            env_keep = [c for _, c in sorted(scored_env, reverse=True)[:5]]
            if not mat_keep or not env_keep:
                continue
            cols = mat_keep + env_keep
            x_raw = np.column_stack([numeric_vector(table, c) for c in cols])
            mask = np.isfinite(y_all) & np.all(np.isfinite(x_raw), axis=1)
            if mask.sum() < 40:
                continue
            x_base = x_raw[mask]
            y = y_all[mask]
            interactions = []
            for i in range(len(mat_keep)):
                for j in range(len(env_keep)):
                    interactions.append((x_base[:, i] * x_base[:, len(mat_keep) + j]).reshape(-1, 1))
            x_inter = np.column_stack([x_base] + interactions)
            base_r2 = ridge_cv_r2(x_base, y)
            inter_r2 = ridge_cv_r2(x_inter, y)
            if np.isfinite(base_r2) and np.isfinite(inter_r2):
                # Very negative CV R2 values indicate an unstable probe rather
                # than meaningful interaction evidence. Keep the diagnostic
                # conservative so a single bad fold cannot dominate the summary.
                if base_r2 < -1.0 or inter_r2 < -1.0:
                    continue
                results.append(
                    {
                        "dataset": table.dataset,
                        "table": table.table,
                        "target": target,
                        "n": int(mask.sum()),
                        "material_features": mat_keep,
                        "environment_features": env_keep,
                        "base_cv_r2": base_r2,
                        "interaction_cv_r2": inter_r2,
                        "delta_cv_r2": inter_r2 - base_r2,
                    }
                )
    return results


def history_autocorrelation(tables: list[Table]) -> list[dict[str, Any]]:
    results = []
    for table in tables:
        if table.dataset != "mooring_steel_seawater":
            continue
        by_condition: dict[str, list[tuple[float, float]]] = {}
        for row in table.rows:
            condition = clean_name(row.get("condition"))
            day = to_float(row.get("days"))
            y = to_float(row.get("ocp_V"))
            if np.isfinite(day) and np.isfinite(y):
                by_condition.setdefault(condition, []).append((day, y))
        for condition, vals in by_condition.items():
            vals = sorted(vals)
            y = np.array([v for _, v in vals], dtype=float)
            if len(y) >= 5 and np.std(y[:-1]) > 0 and np.std(y[1:]) > 0:
                pearson = float(np.corrcoef(y[:-1], y[1:])[0, 1])
                spearman = float(stats.spearmanr(y[:-1], y[1:]).correlation)
                results.append(
                    {
                        "dataset": table.dataset,
                        "table": table.table,
                        "condition": condition,
                        "n": len(y),
                        "lag1_pearson": pearson,
                        "lag1_spearman": spearman,
                    }
                )
    return results


def table_summaries(tables: list[Table]) -> list[dict[str, Any]]:
    out = []
    for table in tables:
        numeric = numeric_columns_by_group(table)
        categorical = categorical_columns_by_group(table)
        out.append(
            {
                "dataset": table.dataset,
                "table": table.table,
                "rows": len(table.rows),
                "columns": len(table.columns),
                "columns_by_group": {g: sum(1 for c in table.columns if table.groups.get(c) == g) for g in GROUPS},
                "numeric_columns_by_group": {g: len(numeric.get(g, [])) for g in GROUPS},
                "categorical_columns_by_group": {g: len(categorical.get(g, [])) for g in GROUPS},
            }
        )
    return out


def aggregate_group_metric(rows: list[dict[str, Any]], key_filter: Callable[[dict[str, Any]], bool], value_key: str) -> dict[str, Any]:
    vals = [float(r[value_key]) for r in rows if key_filter(r) and np.isfinite(float(r.get(value_key, math.nan)))]
    return summarize_values(vals)


def recommendations(results: dict[str, Any]) -> dict[str, Any]:
    within = results["within_block_correlations"]
    cross = results["cross_block_correlations"]
    target = results["target_associations"]
    interactions = results["interaction_probes"]
    history = results["history_autocorrelation"]

    feature_groups = aggregate_group_metric(
        within,
        lambda r: r["group"]
        in {
            "material",
            "environment",
            "process_history",
            "exposure_duration",
            "temporal_history",
            "electrochem_control",
        }
        and r["dataset"] not in {"316l_pitting_passivity"},
        "mean_abs_spearman",
    )
    mat_env = aggregate_group_metric(cross, lambda r: r["group_pair"] == "material:environment", "mean_abs_spearman")
    interaction_delta = summarize_values([r["delta_cv_r2"] for r in interactions])
    history_corr = summarize_values([abs(r["lag1_spearman"]) for r in history])
    intervention_assoc = summarize_values(
        [
            r["best_score"]
            for r in target
            if r["group"] == "direct_intervention" and np.isfinite(float(r.get("best_score", math.nan)))
        ]
    )

    return {
        "informed_feature_block_strength": {
            "evidence": feature_groups,
            "suggested_range": [0.20, 0.35],
            "suggested_default": 0.25,
            "interpretation": "Observed within-block numeric dependence is moderate but not a calibrated prior strength; use a soft block signal rather than increasing it aggressively.",
        },
        "informed_interaction_strength": {
            "material_environment_correlation": mat_env,
            "interaction_cv_delta": interaction_delta,
            "suggested_range": [0.10, 0.35],
            "suggested_default": 0.25,
            "interpretation": "Material-environment coupling is structurally sensible, but simple interaction gains are small and mixed; keep this moderate.",
        },
        "informed_history_strength": {
            "history_autocorrelation": history_corr,
            "suggested_range": [0.00, 0.50],
            "suggested_default": 0.25,
            "interpretation": "Time-series evidence is narrow and mixed; path dependence is a corrosion motif, but a universal strong autoregressive component is not supported.",
        },
        "informed_intervention_strength": {
            "target_association": intervention_assoc,
            "suggested_range": [0.05, 0.20],
            "suggested_default": 0.10,
            "interpretation": "Processing/intervention labels are sometimes informative, but the evidence is sparse and categorical, so keep this weak and conditional.",
        },
        "informed_prior_ratio": {
            "evidence": {"n": 0, "mean": math.nan, "median": math.nan, "q75": math.nan},
            "suggested_range": [0.25, 0.75],
            "suggested_default": 0.50,
            "interpretation": "External datasets do not directly estimate this training-mixture parameter; choose via pre-specified ablation, not DatacorTech test selection.",
        },
    }


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def write_markdown(results: dict[str, Any], path: Path) -> None:
    lines = [
        "# Structural Analysis Results",
        "",
        "This is a first quantitative pass over the downloaded external corrosion datasets. It estimates structural evidence for informed-SCM settings without using DatacorTech performance.",
        "",
        "## Dataset Coverage",
        "",
        "| Dataset | Table | Rows | Columns | Numeric material | Numeric environment | Numeric process/history | Numeric molecular descriptors | Numeric targets |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["table_summaries"]:
        hist_int = (
            row["numeric_columns_by_group"].get("process_history", 0)
            + row["numeric_columns_by_group"].get("exposure_duration", 0)
            + row["numeric_columns_by_group"].get("temporal_history", 0)
        )
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | {row['rows']} | {row['columns']} | "
            f"{row['numeric_columns_by_group'].get('material', 0)} | {row['numeric_columns_by_group'].get('environment', 0)} | "
            f"{hist_int} | {row['numeric_columns_by_group'].get('molecular_descriptor', 0)} | "
            f"{row['numeric_columns_by_group'].get('target', 0)} |"
        )

    lines += [
        "",
        "## Within-Block Correlation Evidence",
        "",
        "| Dataset | Table | Group | Numeric Cols | Pairs | Mean | Median | Q75 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results["within_block_correlations"], key=lambda r: (r["dataset"], r["table"], r["group"])):
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | {row['group']} | {row['n_numeric_columns']} | {row['n_pairs']} | "
            f"{fmt_float(row['mean_abs_spearman'])} | {fmt_float(row['median_abs_spearman'])} | {fmt_float(row['q75_abs_spearman'])} |"
        )

    lines += [
        "",
        "## Material-Environment Interaction Probes",
        "",
        "The probe compares cross-validated ridge R2 from material+environment numeric features versus the same features plus material x environment products. Positive delta supports interaction structure, but this is a rough diagnostic rather than a final model.",
        "",
        "| Dataset | Table | Target | N | Base R2 | Interaction R2 | Delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(results["interaction_probes"], key=lambda r: (r["dataset"], r["table"], r["target"])):
        lines.append(
            f"| `{row['dataset']}` | `{row['table']}` | `{row['target']}` | {row['n']} | "
            f"{fmt_float(row['base_cv_r2'])} | {fmt_float(row['interaction_cv_r2'])} | {fmt_float(row['delta_cv_r2'])} |"
        )

    lines += [
        "",
        "## History Evidence",
        "",
        "| Dataset | Condition | N | Lag-1 Pearson | Lag-1 Spearman |",
        "|---|---|---:|---:|---:|",
    ]
    for row in results["history_autocorrelation"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | {row['n']} | {fmt_float(row['lag1_pearson'])} | {fmt_float(row['lag1_spearman'])} |"
        )

    lines += [
        "",
        "## Parameter Implications",
        "",
        "| Parameter | Suggested Default | Suggested Range | Evidence Summary | Interpretation |",
        "|---|---:|---:|---|---|",
    ]
    for param, rec in results["recommendations"].items():
        evidence = rec.get("evidence") or rec.get("material_environment_correlation") or rec.get("history_autocorrelation") or rec.get("target_association") or {}
        if "interaction_cv_delta" in rec:
            evidence_text = (
                f"mat-env corr median={fmt_float(rec['material_environment_correlation']['median'])}; "
                f"interaction delta median={fmt_float(rec['interaction_cv_delta']['median'])}"
            )
        elif evidence:
            evidence_text = f"mean={fmt_float(evidence.get('mean'))}; median={fmt_float(evidence.get('median'))}; n={evidence.get('n', 0)}"
        else:
            evidence_text = "not directly estimable"
        lines.append(
            f"| `{param}` | {fmt_float(rec['suggested_default'], 2)} | {rec['suggested_range'][0]:.2f}-{rec['suggested_range'][1]:.2f} | "
            f"{evidence_text} | {rec['interpretation']} |"
        )

    lines += [
        "",
        "## Caveats",
        "",
        "- This is structural evidence, not DatacorTech model selection.",
        "- Numeric correlations use pairwise complete Spearman correlations over column-gated scalar measurement fields.",
        "- Text identifiers, formulas, process labels, prose methods, and categorical environment labels are excluded from numeric evidence; bounded categorical target association is still estimated with eta squared.",
        "- The physical-range-prior question is still separate: this pass mostly addresses block and interaction settings.",
        "- Recommended values should become a small pre-specified ablation grid, not a final claim that one value is optimal.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write JSON and Markdown result files")
    args = parser.parse_args()

    tables = load_all_tables()
    results = {
        "table_summaries": table_summaries(tables),
        "within_block_correlations": within_block_correlations(tables),
        "cross_block_correlations": cross_block_correlations(tables),
        "target_associations": target_associations(tables),
        "interaction_probes": interaction_probes(tables),
        "history_autocorrelation": history_autocorrelation(tables),
    }
    results["recommendations"] = recommendations(results)

    if args.write:
        OUT_JSON.write_text(json.dumps(sanitize_for_json(results), indent=2), encoding="utf-8")
        write_markdown(results, OUT_MD)
        print(f"Wrote {OUT_JSON}")
        print(f"Wrote {OUT_MD}")
    else:
        print(json.dumps(sanitize_for_json(results["recommendations"]), indent=2))


if __name__ == "__main__":
    main()
