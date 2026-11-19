"""Prepare evaluation-only rebar data; no coefficient fitting or prior generation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'corrosion_datasets/datasets/rebar_pitting'
RAW = BASE / 'raw/rebar_electrochemical_parameters.xlsx'
EXPECTED_RAW = '1ed10a3a6da71799e34051874844e86a1a1229a474051908bd3551c2feeab1cf'
TARGET = 'pitting_potential_mV_SCE'
# Conservative source-level decisions: these do not assert all withheld values are wrong.
DECISIONS = [
 ('Corrosion and tensile behavior of 316L', 'withheld_misaligned_composition_columns', None),
 ('Influence of phosphate ions on pit initiation', 'withheld_unresolved_composition_total', None),
 ('Potentiostatic determination of chloride threshold', 'withheld_applied_potential_target_unverified', '10.1016/j.electacta.2004.01.034'),
 ('Investigation on the effect of supplementary cementitious', 'withheld_threshold_target_and_placeholder_composition', '10.1617/s11527-015-0778-0'),
 ('Study on the corrosion behavior of reinforcing steel', 'withheld_pitting_target_unverified', '10.1016/j.electacta.2010.05.003'),
 ('Passivation and electrochemical behavior of 316L', 'excluded_known_soccol_source_overlap', '10.1016/j.apsusc.2016.12.180'),
 ('Effect of nucleation seeding and triisopropanolamine', 'withheld_source_target_mismatch', '10.1016/j.jobe.2022.104382'),
 ('Effect of alternating temperature on passivation', 'withheld_chloride_free_breakdown_target_unverified', None),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(value):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', str(value)).lower())


def number(value):
    """Only exact finite numbers; censored values/ranges remain missing."""
    if isinstance(value, bool) or value is None:
        return np.nan
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def source_rows(ws, header, refcol):
    """Carry bibliographic metadata only, reset on each changed reference ID."""
    by_id = defaultdict(set)
    for cells in ws.iter_rows(min_row=header + 1):
        ref, title = cells[0].value, cells[refcol - 1].value
        if ref is not None and title:
            by_id[str(ref)].add(str(title).strip())
    current_id, current_title = None, None
    for row in range(header + 1, ws.max_row + 1):
        ref, title = ws.cell(row, 1).value, ws.cell(row, refcol).value
        if ref is not None and str(ref) != current_id:
            current_id, current_title = str(ref), None
        if title:
            current_title = str(title).strip()
        if current_title is None and len(by_id[current_id]) == 1:
            current_title = next(iter(by_id[current_id]))
        yield row, current_id, current_title


def feature_map(ws, old):
    header = 4 if old else 3
    mapping = defaultdict(list)
    # Physical units follow workbook group headers. Never use outcome columns.
    for col in range(5 if old else 4, 18 if old else 30):
        name = str(ws.cell(header, col).value)
        mapping[f'{"Nb" if name == "Cb" else name}_wtpct'].append(col)
    mapping['temperature_C'] = [18 if old else 30]
    mapping['pH'] = [43 if old else 54]
    for col in range(24 if old else 35, 43 if old else 54):
        name = str(ws.cell(header, col).value)
        mapping[f'solution_{name}_M'].append(col)
    # Only first (weight-percent) cement block; no duplicate molar block.
    for col in range(46 if old else 57, 66 if old else 87):
        name = str(ws.cell(header, col).value)
        aliases = {'Tricalcium silicate (C3S)': 'C3S', 'Dicalcium silicate (C2S)': 'C2S',
                   'Tricalcium aluminate (C3A)': 'C3A', 'Tetracalcium aluminoferrite (C4AF)': 'C4AF'}
        mapping[f'cement_{aliases.get(name, name)}_wtpct'].append(col)
    return mapping


def prepare(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite frozen data: {output}')
    if sha(RAW) != EXPECTED_RAW:
        raise RuntimeError('Rebar raw workbook differs from audited version.')
    wb = openpyxl.load_workbook(RAW, data_only=True)
    formulas = openpyxl.load_workbook(RAW, data_only=False)
    records, features, cells, sources = [], [], [], {}
    for ws in wb:
        old = ws.title.endswith('1')
        header, targetcol, refcol = (4, 21, 87) if old else (3, 32, 95)
        mapping = feature_map(ws, old)
        for row, ref, source in source_rows(ws, header, refcol):
            raw_y = ws.cell(row, targetcol).value
            if raw_y is None:
                continue
            y = number(raw_y)
            row_id = f'{ws.title.replace(" ", "_")}:{row}'
            status, doi = 'included_workbook_reported', None
            if not np.isfinite(y):
                status = 'withheld_non_numeric_or_censored_target'
            elif source is None:
                status = 'withheld_missing_reference'
            else:
                for match, decision, source_doi in DECISIONS:
                    if norm(match) in norm(source):
                        status, doi = decision, source_doi
                        break
                # The Liu source has no established pitting-potential measurement.
                if 'liu' in source.lower() and '2017' in source and 'huang' in source.lower():
                    status, doi = 'withheld_pitting_target_unverified', '10.1016/j.conbuildmat.2017.09.104'
            values = {}
            for name, columns in mapping.items():
                candidates = []
                for col in columns:
                    cell = ws.cell(row, col)
                    value = number(cell.value)
                    # Only temperature uncertainty gets a documented central-value conversion.
                    if name == 'temperature_C' and isinstance(cell.value, str) and '±' in cell.value:
                        value = number(cell.value.split('±')[0])
                    color = cell.font.color
                    style = str(color.rgb) if color and color.type == 'rgb' else None
                    cells.append(dict(row_id=row_id, feature=name, coordinate=cell.coordinate,
                        raw=str(cell.value) if cell.value is not None else None,
                        formula=str(formulas[ws.title][cell.coordinate].value), font_rgb=style,
                        parsed=value if np.isfinite(value) else None))
                    if np.isfinite(value):
                        candidates.append(value)
                unique = set(candidates)
                values[name] = candidates[0] if len(unique) == 1 else np.nan
                if len(unique) > 1:
                    cells.append(dict(row_id=row_id, feature=name, issue='conflicting_duplicate_columns'))
            ecp = number(ws.cell(row, 19 if old else 31).value)
            source_id = hashlib.sha256(norm(source).encode()).hexdigest()[:16] if source else None
            record = dict(row_id=row_id, sheet=ws.title, excel_row=row, reference_id=ref,
                          source_id=source_id, source=source, target_raw=str(raw_y),
                          target_coordinate=ws.cell(row, targetcol).coordinate,
                          target_mV_SCE=y*1000 if np.isfinite(y) else None,
                          equal_to_ecp=bool(np.isfinite(y) and y == ecp), status=status)
            records.append(record)
            if source:
                sources.setdefault(source_id, dict(source=source, doi=doi,
                    evidence_url=f'https://doi.org/{doi}' if doi else None,
                    overlap_review='known_soccol_overlap' if 'overlap' in status else 'not_fully_resolved', row_decisions={}))
                source_decisions = sources[source_id]['row_decisions']
                source_decisions[status] = source_decisions.get(status, 0) + 1
            if status.startswith('included'):
                features.append({'row_id': row_id, **values})
    X = pd.DataFrame(features).set_index('row_id')
    # Columns unobserved anywhere carry no information. No global value imputation/scaling.
    X = X.loc[:, X.notna().any()]
    kept = pd.DataFrame([r for r in records if r['status'].startswith('included')]).set_index('row_id').loc[X.index]
    # Entire publications stay together, including repeated/digitized observations.
    sizes = kept.groupby('source_id').size().to_dict()
    rng = np.random.default_rng(42)
    groups = list(rng.permutation(sorted(sizes)))
    groups.sort(key=lambda g: -sizes[g])
    loads, assignment = [0]*5, {}
    for group in groups:
        fold = int(np.argmin(loads))
        assignment[group] = fold + 1
        loads[fold] += sizes[group]
    rows = [dict(task_row_index=i, row_id=rid, source_id=kept.loc[rid, 'source_id'],
                 cv_fold=assignment[kept.loc[rid, 'source_id']],
                 outer_split='final_test' if assignment[kept.loc[rid, 'source_id']] == 1 else 'development')
            for i, rid in enumerate(X.index)]
    output.mkdir(parents=True, exist_ok=True)
    X.reset_index().to_csv(output/'features.csv', index=False)
    pd.DataFrame({'row_id': X.index, TARGET: kept['target_mV_SCE'].to_numpy()}).to_csv(output/'targets.csv', index=False)
    pd.DataFrame(records).to_csv(output/'row_audit.csv', index=False)
    (output/'cell_audit.json').write_text(json.dumps(cells, indent=2)+'\n')
    (output/'source_audit.json').write_text(json.dumps(sources, indent=2)+'\n')
    report = dict(status='provisional_workbook_reported_not_fully_source_verified',
        numeric_target_candidates=sum(r['target_mV_SCE'] is not None for r in records),
        included=len(X), features=len(X.columns), publications=len(sizes), fold_sizes=loads,
        decisions=pd.Series([r['status'] for r in records]).value_counts().to_dict(),
        limitations=['Original-paper row verification and training-source overlap screening are incomplete.',
            'Workbook zeros may include author assumptions; no additional zero filling is applied.',
            'Unreported inhibitors, exposure and preparation conditions may remain unobserved.',
            'CV is publication-grouped, unlike the existing EPIT/Soccol row-based exploratory folds.'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    manifest = dict(schema_version='rebar_evaluation_split_v1', seed=42,
        dataset=dict(target_column=TARGET, target_unit='mV SCE', feature_columns_in_order=list(X.columns)),
        raw=dict(path=str(RAW.relative_to(ROOT)), sha256=sha(RAW)),
        preparation_sha256=sha(Path(__file__)), rows=rows,
        exports={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        status=report['status'])
    (output/'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'split_lock.json').write_text(json.dumps({'manifest_sha256': sha(output/'split_manifest.json')}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE/'processed/splits_v1')
    prepare(parser.parse_args().output_dir)
