"""Machine-readable schema and composition policy for the EPIT task.

The v2 model schema exposes every elemental column present in the source
workbook.  Element order is fixed because the synthetic prior and analytical
target rules use named positions within the material block.
"""

from __future__ import annotations

from typing import Final


EPIT_SCHEMA_VERSION: Final[str] = "epit_24_element_v2"
EPIT_COMPOSITION_ELEMENTS: Final[tuple[str, ...]] = (
    "Fe",
    "Cr",
    "Ni",
    "Mo",
    "W",
    "N",
    "Nb",
    "C",
    "Si",
    "Mn",
    "Cu",
    "P",
    "S",
    "Al",
    "V",
    "Ta",
    "Re",
    "Ce",
    "Ti",
    "Co",
    "B",
    "Mg",
    "Y",
    "Gd",
)
EPIT_COMPOSITION_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"Composition, wt.% {element}" for element in EPIT_COMPOSITION_ELEMENTS
)
EPIT_COMPOSITION_INDEX: Final[dict[str, int]] = {
    element: index for index, element in enumerate(EPIT_COMPOSITION_ELEMENTS)
}

# The previous model exposed only these 17 columns.  The tuple is retained for
# frozen-artifact compatibility checks and documentation; it is not the v2
# model input schema.
EPIT_LEGACY_COMPOSITION_ELEMENTS: Final[tuple[str, ...]] = (
    "Fe",
    "Cr",
    "Ni",
    "Mo",
    "W",
    "Nb",
    "Al",
    "V",
    "Ta",
    "Re",
    "Ce",
    "Ti",
    "Co",
    "B",
    "Mg",
    "Y",
    "Gd",
)
EPIT_LEGACY_COMPOSITION_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"Composition, wt.% {element}"
    for element in EPIT_LEGACY_COMPOSITION_ELEMENTS
)

EPIT_ENVIRONMENT_COLUMNS: Final[tuple[str, ...]] = (
    "Test Temp. oC",
    "[Cl-] M",
    "[Cl-] pH",
)
EPIT_METHOD_COLUMN: Final[str] = "[Cl-] Test Method"
EPIT_MATERIAL_FEATURE_COUNT: Final[int] = len(EPIT_COMPOSITION_ELEMENTS)
EPIT_ENVIRONMENT_FEATURE_COUNT: Final[int] = len(EPIT_ENVIRONMENT_COLUMNS)
EPIT_PROCESS_FEATURE_COUNT: Final[int] = 1
EPIT_BASE_FEATURE_COUNT: Final[int] = (
    EPIT_MATERIAL_FEATURE_COUNT
    + EPIT_ENVIRONMENT_FEATURE_COUNT
    + EPIT_PROCESS_FEATURE_COUNT
)

# Missing elemental values are structural zeros only when the sum of all
# reported elemental values already closes to 100 wt.% within this tolerance.
EPIT_COMPOSITION_CLOSURE_TOLERANCE_WT_PERCENT: Final[float] = 0.1


def epit_feature_slices(
    material_feature_count: int = EPIT_MATERIAL_FEATURE_COUNT,
) -> dict[str, slice]:
    """Return EPIT feature blocks, defaulting to the v2 24/3/1 schema."""

    material_stop = int(material_feature_count)
    if material_stop <= 0:
        raise ValueError("material_feature_count must be positive.")
    environment_stop = material_stop + EPIT_ENVIRONMENT_FEATURE_COUNT
    base_feature_count = environment_stop + EPIT_PROCESS_FEATURE_COUNT
    return {
        "material": slice(0, material_stop),
        "environment": slice(material_stop, environment_stop),
        "process_history": slice(environment_stop, base_feature_count),
    }


__all__ = [
    "EPIT_BASE_FEATURE_COUNT",
    "EPIT_COMPOSITION_CLOSURE_TOLERANCE_WT_PERCENT",
    "EPIT_COMPOSITION_COLUMNS",
    "EPIT_COMPOSITION_ELEMENTS",
    "EPIT_COMPOSITION_INDEX",
    "EPIT_ENVIRONMENT_COLUMNS",
    "EPIT_LEGACY_COMPOSITION_COLUMNS",
    "EPIT_LEGACY_COMPOSITION_ELEMENTS",
    "EPIT_MATERIAL_FEATURE_COUNT",
    "EPIT_METHOD_COLUMN",
    "EPIT_SCHEMA_VERSION",
    "epit_feature_slices",
]
