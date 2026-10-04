"""Fixed 37-column model schema for the Soccol pitting-potential task."""

from __future__ import annotations

from typing import Final


SOCCOL_FEATURE_PROFILE: Final[str] = "soccol_pitting_features_v1"
SOCCOL_SCHEMA_VERSION: Final[str] = "soccol_pitting_37_feature_v1"

SOCCOL_COMPOSITION_COLUMNS: Final[tuple[str, ...]] = (
    "Fe", "C", "N", "Si", "P", "S", "Ti", "V", "Cr", "Mn", "Ni", "Nb", "Mo",
)
SOCCOL_CONTINUOUS_COLUMNS: Final[tuple[str, ...]] = (
    "Prep_grinding_grit",
    "Prep_Ra_micron",
    "Prep_pH",
    "Prep_redox",
    "Prep_time",
    "CP_time",
    "CP_temp",
    "CP_pH",
    "Test_area_cm2",
    "scan_rate",
)
SOCCOL_ION_COLUMNS: Final[tuple[str, ...]] = (
    "CP_Cl",
    "CP_Br",
    "CP_OH",
    "CP_SO4",
    "CP_CO3",
    "CP_NO3",
    "CP_PO4",
    "CP_MoO4",
    "CP_CrO4",
    "CP_ion_other",
)
SOCCOL_CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "Prep_medium", "CP_aeration", "CP_agitation", "CP_anions_info",
)
SOCCOL_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    *SOCCOL_COMPOSITION_COLUMNS,
    *SOCCOL_CONTINUOUS_COLUMNS,
    *SOCCOL_ION_COLUMNS,
)
SOCCOL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *SOCCOL_NUMERIC_COLUMNS,
    *SOCCOL_CATEGORICAL_COLUMNS,
)
SOCCOL_FEATURE_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(SOCCOL_FEATURE_COLUMNS)
}
SOCCOL_COMPOSITION_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(SOCCOL_COMPOSITION_COLUMNS)
}

SOCCOL_MATERIAL_FEATURE_COUNT: Final[int] = len(SOCCOL_COMPOSITION_COLUMNS)
SOCCOL_ENVIRONMENT_FEATURE_COUNT: Final[int] = (
    len(SOCCOL_CONTINUOUS_COLUMNS) + len(SOCCOL_ION_COLUMNS)
)
SOCCOL_PROCESS_FEATURE_COUNT: Final[int] = len(SOCCOL_CATEGORICAL_COLUMNS)
SOCCOL_BASE_FEATURE_COUNT: Final[int] = len(SOCCOL_FEATURE_COLUMNS)
SOCCOL_FIXED_BLOCK_ALLOCATION: Final[tuple[int, ...]] = (
    SOCCOL_MATERIAL_FEATURE_COUNT,
    SOCCOL_ENVIRONMENT_FEATURE_COUNT,
    SOCCOL_PROCESS_FEATURE_COUNT,
    0, 0, 0, 0, 0, 0,
)


def soccol_feature_slices() -> dict[str, slice]:
    material_stop = SOCCOL_MATERIAL_FEATURE_COUNT
    environment_stop = material_stop + SOCCOL_ENVIRONMENT_FEATURE_COUNT
    return {
        "material": slice(0, material_stop),
        "environment": slice(material_stop, environment_stop),
        "process_history": slice(environment_stop, SOCCOL_BASE_FEATURE_COUNT),
    }


__all__ = [name for name in globals() if name.startswith("SOCCOL_")] + [
    "soccol_feature_slices"
]
