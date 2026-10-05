from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.epit_pipeline.run_pysr_discovery import (
    label_template_equation,
    method_feature_names,
    method_indicators,
    varying_feature_mask,
)


def test_discovery_method_categories_are_numeric_indicators() -> None:
    categories = ["potentiodynamic", "potentiostatic", "scratch"]
    labels = pd.Series(["other", "scratch", "potentiodynamic"])

    result = method_indicators(labels, categories)

    assert method_feature_names(categories) == [
        "method_potentiodynamic",
        "method_potentiostatic",
        "method_scratch",
    ]
    assert result.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]


def test_discovery_drops_constant_candidates() -> None:
    values = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert varying_feature_mask(values).tolist() == [True, False]


def test_discovery_labels_template_placeholders() -> None:
    equation = "f = #2 + sqrt(#1); method_offset = [1.0, 2.0, 3.0]"
    assert label_template_equation(equation, ["Cr_wt_pct", "Mo_wt_pct"]) == (
        "f = Mo_wt_pct + sqrt(Cr_wt_pct); "
        "method_offset = [1.0, 2.0, 3.0]"
    )
