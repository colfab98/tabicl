from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
WORKBOOK = HERE / "raw" / "Pitting_potential_dataset_2024.xlsx"
ELEMENTS = ["C", "N", "Si", "P", "S", "Ti", "V", "Cr", "Mn", "Ni", "Nb", "Mo"]


def joined(values: list[str]) -> str:
    return ";".join(values)


data = pd.read_excel(WORKBOOK, sheet_name="pitting_potentials")
records = []

for source, group in data.groupby("source", sort=True):
    missing = group[ELEMENTS].isna()
    sample_count = len(group)
    partial = [
        element
        for element in ELEMENTS
        if 0 < int(missing[element].sum()) < sample_count
    ]
    all_blank = [element for element in ELEMENTS if missing[element].all()]
    complete = [element for element in ELEMENTS if not missing[element].any()]
    zero_and_blank = [
        element
        for element in ELEMENTS
        if missing[element].any()
        and pd.to_numeric(group[element], errors="coerce").eq(0).any()
    ]

    if not missing.any().any():
        classification = "fully_complete"
    elif partial:
        classification = "mixed_within_source"
    else:
        classification = "consistent_source_schema_with_blanks"

    records.append(
        {
            "source": source,
            "samples": sample_count,
            "classification": classification,
            "rows_with_any_composition_blank": int(missing.any(axis=1).sum()),
            "composition_blank_cells": int(missing.sum().sum()),
            "partial_elements": joined(partial),
            "zero_and_blank_elements": joined(zero_and_blank),
            "all_blank_elements": joined(all_blank),
            "complete_elements": joined(complete),
        }
    )

summary = pd.DataFrame(records)
summary.to_csv(HERE / "soccol_source_composition_audit.csv", index=False)
summary.loc[summary["classification"] == "mixed_within_source"].to_csv(
    HERE / "soccol_problematic_sources.csv", index=False
)

print(summary.groupby("classification")["samples"].agg(["count", "sum"]))
