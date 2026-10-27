"""Reviewed Soccol sampling-bank columns; historical 37-column schema is separate."""

from .epit_schema import EPIT_COMPOSITION_ELEMENTS


SOCCOL_REVIEWED_FEATURE_PROFILE = "soccol_pitting_features_v3"
SOCCOL_REVIEWED_FEATURE_PROFILES = ("soccol_pitting_features_v2", SOCCOL_REVIEWED_FEATURE_PROFILE)
SOCCOL_COMPOSITION_SAMPLING_POLICY = "uniform_distinct_compositions_v1"
SOCCOL_REVIEWED_SCHEMA_VERSION = "soccol_pitting_58_feature_v2"
SOCCOL_REVIEWED_COMPOSITION_COLUMNS = tuple(EPIT_COMPOSITION_ELEMENTS)
SOCCOL_REVIEWED_CONTEXT_COLUMNS = (
    "Prep_grinding_grit", "Prep_Ra_micron", "Prep_pH", "Prep_redox",
    "Prep_time", "CP_time", "CP_temp", "CP_pH", "Test_area_cm2", "scan_rate",
    "CP_Cl", "CP_Br", "CP_OH", "CP_SO4", "CP_CO3", "CP_NO3", "CP_PO4",
    "CP_MoO4", "CP_CrO4", "CP_aeration", "CP_agitation", "test_solution_type",
    "H2SO4_mol_L", "surface_preparation_method", "preparation_solution_type",
    "heat_treatment_type", "heat_treatment_temperature_C", "heat_treatment_duration_h",
    "heat_treatment_cooling", "heat_treatment_2_type", "heat_treatment_2_temperature_C",
    "heat_treatment_2_duration_h", "heat_treatment_2_cooling", "reference_type",
)
SOCCOL_REVIEWED_FEATURE_COLUMNS = (
    *SOCCOL_REVIEWED_COMPOSITION_COLUMNS, *SOCCOL_REVIEWED_CONTEXT_COLUMNS,
)
SOCCOL_REVIEWED_CATEGORICAL_COLUMNS = (
    "CP_aeration", "test_solution_type", "surface_preparation_method",
    "preparation_solution_type", "heat_treatment_type", "heat_treatment_cooling",
    "heat_treatment_2_type", "heat_treatment_2_cooling", "reference_type",
)
SOCCOL_REVIEWED_FEATURE_COUNT = len(SOCCOL_REVIEWED_FEATURE_COLUMNS)
SOCCOL_REVIEWED_FEATURE_INDEX = {
    name: index for index, name in enumerate(SOCCOL_REVIEWED_FEATURE_COLUMNS)
}
SOCCOL_REVIEWED_BLOCK_ALLOCATION = (24, 34, 0, 0, 0, 0, 0, 0, 0)


def soccol_reviewed_feature_slices():
    # Categories are interspersed in the saved context order, as in EPIT.
    return {"material": slice(0, 24), "environment": slice(24, 58)}
