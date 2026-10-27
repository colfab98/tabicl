"""The existing nine Soccol formulas and their unchanged training term names."""

RULE_SELECTION = (
    ("pren_n_linear", "old_pren_n_linear"),
    ("cr_mow_n_synergy", "old_cr_mo_n_synergy"),
    ("threshold_saturation", "old_threshold_saturation"),
    ("pren_n_improved_environment", "old_pren_n_improved_environment"),
    ("mo_n_acid_repassivation", "old_mo_n_acid_repassivation"),
    ("mns_inclusion_penalty", "old_mns_inclusion_penalty"),
    ("coupled_breakdown", "old_coupled_breakdown"),
    ("pren_n_coupled_mns", "new_pren_n_coupled_mns"),
    ("pren_n_coupled_mns_weak_anions", "new_pren_n_coupled_mns_weak_anions"),
)
PROMOTED_RULES = tuple(family for family, _ in RULE_SELECTION)
TERM_RENAMES = {
    "log_halide_aggressiveness": "log_chloride_aggressiveness",
    "temperature_halide_interaction": "temperature_chloride_interaction",
}


def rename_terms(mapping):
    return {TERM_RENAMES.get(name, name): float(value) for name, value in mapping.items()}
