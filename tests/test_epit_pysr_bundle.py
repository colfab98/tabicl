from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1] / "standalone" / "epit_pysr"


def test_standalone_bundle_contains_development_rows_only() -> None:
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    with (BUNDLE / "epit_development.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert manifest["development_rows"] == 608
    assert manifest["eligible_development_rows"] == 452
    assert manifest["schema_version"] == "epit_pysr_discovery_bundle_v2"
    assert manifest["final_test_rows_exported"] == 0
    assert manifest["final_test_targets_exported"] is False
    assert len(rows) == 608
    assert {int(row["validation_fold"]) for row in rows} == {1, 2, 3, 4, 5}
    assert (
        sum(
            row["material_class"] in {"Fe Alloy", "NiCrMo Alloy"}
            for row in rows
        )
        == 452
    )
    assert len(manifest["physical_features"]) == 27


def test_standalone_runner_uses_discovery_grammar() -> None:
    spec = importlib.util.spec_from_file_location(
        "standalone_epit_pysr", BUNDLE / "run_pysr.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.BINARY_OPERATORS == ("+", "-", "*", "/")
    assert module.UNARY_OPERATORS == ("square", "sqrt", "log", "exp")
