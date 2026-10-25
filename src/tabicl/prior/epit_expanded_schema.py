"""Reviewed EPIT inputs; the legacy 28-column formula layout stays separate."""

from .epit_schema import EPIT_COMPOSITION_ELEMENTS

EPIT_EXPANDED_FEATURE_PROFILE = "epit_fe_nicrmo_features_v4"
EPIT_EXPANDED_ENVIRONMENT_COLUMNS = (
    "CP_temp", "CP_pH", "CP_Cl", "scan_rate", "test_solution_type", "CP_aeration",
    "H2SO4_eq_L", "H2SO4_mol_L", "H2S_partial_pressure_atm", "HCl_eq_L", "HCl_mol_L",
    "KCl_eq_L", "KCl_mol_L", "MgCl2_mol_L", "Na2S2O3_mol_L", "Na2SO4_mol_L",
    "NaBr_eq_L", "NaBr_mol_L", "NaClO4_mol_L", "NaCl_eq_L", "NaCl_mol_L",
    "NaCl_wt_percent", "NaHCO3_mol_L", "Test_area_cm2", "WO4_mol_L",
    "heat_treatment_type", "heat_treatment_temperature_C", "heat_treatment_duration_h",
    "heat_treatment_cooling", "heat_treatment_2_type", "heat_treatment_2_temperature_C",
    "heat_treatment_2_duration_h", "heat_treatment_2_cooling", "phase_FCC", "phase_BCC",
    "phase_HCP", "phase_IM", "Prep_grinding_grit", "surface_preparation_method",
    "preparation_solution_type", "Prep_time", "CP_time", "CP_agitation", "CP_Br",
    "CP_SO4", "reference_type",
)
EPIT_EXPANDED_CATEGORICAL_COLUMNS = (
    "test_solution_type", "CP_aeration", "heat_treatment_type", "heat_treatment_cooling",
    "heat_treatment_2_type", "heat_treatment_2_cooling", "surface_preparation_method",
    "preparation_solution_type", "reference_type",
)
EPIT_EXPANDED_FEATURE_COLUMNS = (*EPIT_COMPOSITION_ELEMENTS, *EPIT_EXPANDED_ENVIRONMENT_COLUMNS)
EPIT_EXPANDED_FEATURE_COUNT = len(EPIT_EXPANDED_FEATURE_COLUMNS)
EPIT_EXPANDED_BLOCK_ALLOCATION = (24, 46, 0, 0, 0, 0, 0, 0, 0)
