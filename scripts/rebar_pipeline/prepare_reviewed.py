"""Versioned rebar candidate review; preserve every frozen outer-query row.

This prepares data only. Eligibility is a composition/known-target-issue screen,
not certification of measurement independence or permission to fit a prior.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import openpyxl
import pandas as pd

from scripts.rebar_pipeline.evaluation_data import DEFAULT_SPLIT, frozen_split
from scripts.rebar_pipeline.prepare_evaluation import BASE, RAW, ROOT, TARGET, feature_map, norm, sha

DEFAULT_OUTPUT = BASE / 'processed/rebar_reviewed_v2'
RULES = BASE / 'review_rules_v2.json'
EPIT_REF = ROOT / 'corrosion_datasets/datasets/electrochemical_metrics_alloys/raw/CRA_database_Scientific_Data_Publication_12102020.xlsx'
SOCCOL_REF = ROOT / 'corrosion_datasets/datasets/soccol_pitting_potential/raw/Pitting_potential_dataset_2024.xlsx'
EPIT_ELEMENTS = 'Fe Cr Ni Mo W N Nb C Si Mn Cu P S Al V Ta Re Ce Ti Co B Mg Y Gd'.split()
RENAMES = {
    'temperature_C': 'CP_temp', 'pH': 'CP_pH',
    'solution_Cl-_M': 'CP_Cl', 'solution_OH-_M': 'CP_OH',
    'solution_(SO4)2-_M': 'CP_SO4', 'solution_(NO2)-_M': 'CP_NO2',
    'solution_(AlO2)-_M': 'CP_AlO2', 'solution_(HPO4)2-_M': 'CP_HPO4',
    'solution_(HCO3)-_M': 'CP_HCO3', 'solution_(WO4)2-_M': 'WO4_mol_L',
    'solution_Na+_M': 'CP_Na', 'solution_K+_M': 'CP_K',
    'solution_Ca(OH)2_M': 'CaOH2_mol_L',
}
EXCLUDED = {
    'solution_Sodium oleate hydrocarbon tail_M': 'Fragment amount is not a verified whole-inhibitor molarity.',
    'solution_(MO4)2-_M': 'Ambiguous element notation MO4; do not silently reinterpret as MoO4.',
}
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def parse_amount(raw):
    if raw is None or str(raw).strip() in {'', '-', '--'}:
        return math.nan, 'unreported'
    text = str(raw).strip()
    if text.lower() in {'bal.', 'bal', 'balance', 'remainder'}:
        return math.nan, 'explicit_balance'
    if text.lower() in {'n/a', 'na', 'nan', 'none'}:
        return math.nan, 'explicit_unknown'
    limit = re.fullmatch(rf'(?:<\s*=?\s*({NUMBER})|({NUMBER})\s+[Mm]ax)', text)
    if limit:
        value, action = float(limit[1] or limit[2]), 'upper_limit_as_numeric'
    elif not isinstance(raw, bool) and re.fullmatch(NUMBER, text):
        value, action = float(text), 'reported_numeric'
    else:
        return math.nan, 'unresolved_notation'
    if not math.isfinite(value) or not 0 <= value <= 100:
        return math.nan, 'invalid_amount'
    return value, action


def process_composition(raw, *, infer_fe=False, known_omissions=()):
    """Never close an unexplained shortfall or conceal a known missing element."""
    parsed = {e: parse_amount(v) for e, v in raw.items()}
    values = {e: v for e, (v, _) in parsed.items()}
    actions = {e: a for e, (_, a) in parsed.items()}
    reasons = []
    for e in known_omissions:
        values[e], actions[e] = math.nan, 'known_omission'
    if any(a in {'unresolved_notation', 'invalid_amount', 'explicit_unknown', 'known_omission'}
           for a in actions.values()):
        reasons.append('unresolved_element_amount')
    if any(a == 'explicit_balance' for e, a in actions.items() if e != 'Fe'):
        reasons.append('unsupported_balance_element')
    n_known = sum(math.isfinite(v) for v in values.values())
    before = math.fsum(v for v in values.values() if math.isfinite(v))
    if not n_known:
        reasons.append('no_numeric_composition')
    if not reasons and not math.isfinite(values['Fe']):
        if actions['Fe'] == 'explicit_balance' or infer_fe:
            values['Fe'] = 100.0 - before
            actions['Fe'] = 'calculated_explicit_Fe_balance' if actions['Fe'] == 'explicit_balance' else 'calculated_reviewed_Fe_balance'
        else:
            reasons.append('unsupported_missing_Fe')
    total = math.fsum(v for v in values.values() if math.isfinite(v))
    if total < 90:
        reasons.append('composition_total_below_90')
    elif not 99 - 1e-10 <= total <= 101 + 1e-10:
        reasons.append('composition_total_outside_99_101')
    if any(not 0 <= v <= 100 for v in values.values() if math.isfinite(v)):
        reasons.append('invalid_completed_amount')
    if not reasons:
        for e in values:
            if not math.isfinite(values[e]):
                values[e], actions[e] = 0.0, 'assumed_zero_after_completeness_review'
    return values, actions, before, total, sorted(set(reasons))


def doi_tokens(text):
    return {s.lower().rstrip('.,;)') for s in re.findall(r'10\.\d{4,9}/[^\s<>"\]]+', str(text))}


def review_sources(original, rules):
    references = []
    for path, sheet, dataset in [(EPIT_REF, 'References', 'EPIT'), (SOCCOL_REF, 'references', 'Soccol')]:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for row in list(wb[sheet].values)[1:]:
            text = ' '.join(str(v) for v in row if v is not None)
            references.append((dataset, text, doi_tokens(text)))
        wb.close()
    reviewed = {}
    for sid, old in original.items():
        rule = rules['sources'].get(sid, {})
        dois = doi_tokens(old.get('doi') or '') | doi_tokens(old['source']) | doi_tokens(rule.get('doi', ''))
        matches = []
        for dataset, text, ref_dois in references:
            common = sorted(dois & ref_dois)
            exact_text = len(norm(old['source'])) >= 40 and norm(old['source']) in norm(text)
            if common or exact_text:
                matches.append(dict(dataset=dataset, reference=text, matching_dois=common,
                                    method='doi' if common else 'normalized_full_reference_containment'))
        known_overlap = old['overlap_review'] == 'known_soccol_overlap'
        reviewed[sid] = {**old, 'resolved_dois': sorted(dois), 'matches': matches,
                        'known_training_source_overlap': bool(matches or known_overlap),
                        'overlap_review': 'known_overlap' if matches or known_overlap else 'no_match_in_limited_screen_not_verified_independent',
                        'source_review': rule}
    return reviewed


def prepare(output=DEFAULT_OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite reviewed data: {output}')
    manifest = frozen_split()
    rules = json.loads(RULES.read_text())
    parent = DEFAULT_SPLIT.parent
    old = pd.read_csv(parent / 'features.csv').set_index('row_id')
    targets = pd.read_csv(parent / 'targets.csv').set_index('row_id')
    all_audit = pd.read_csv(parent / 'row_audit.csv').set_index('row_id')
    assignments = pd.DataFrame(manifest['rows'])
    ids = assignments.row_id.tolist()
    if old.index.tolist() != ids or targets.index.tolist() != ids:
        raise RuntimeError('Frozen row order mismatch.')
    source_audit = review_sources(json.loads((parent / 'source_audit.json').read_text()), rules)
    wb = openpyxl.load_workbook(RAW, data_only=True)
    formulas = openpyxl.load_workbook(RAW, data_only=False)
    elements = list(EPIT_ELEMENTS)
    for ws in wb:
        for name in feature_map(ws, ws.title.endswith('1')):
            if name.endswith('_wtpct') and not name.startswith('cement_') and name[:-6] not in elements:
                elements.append(name[:-6])
    if 'La' not in elements:
        elements.append('La')
    env_columns = [c for c in old if not (c.endswith('_wtpct') and not c.startswith('cement_')) and c not in EXCLUDED]
    rows, audits, logs = [], [], []
    for rid in ids:
        record = all_audit.loc[rid]
        ws, row = wb[record['sheet']], int(record['excel_row'])
        mapping = feature_map(ws, ws.title.endswith('1'))
        raw, coordinates, raw_formulas = {}, {}, {}
        for element in elements:
            columns = mapping.get(element + '_wtpct', [])
            observed = [(col, ws.cell(row, col).value) for col in columns if ws.cell(row, col).value is not None]
            coordinates[element] = ';'.join(ws.cell(row, col).coordinate for col in columns)
            raw_formulas[element] = [str(formulas[ws.title].cell(row, col).value) for col in columns
                                     if formulas[ws.title].cell(row, col).data_type == 'f']
            unique = {str(value).strip() for _, value in observed}
            if len(unique) > 1 and len({parse_amount(value)[0] for _, value in observed}) != 1:
                raw[element] = 'CONFLICT: ' + ' | '.join(sorted(unique))
            else:
                raw[element] = observed[0][1] if observed else None
        source_rule = rules['sources'].get(record.source_id, {})
        omitted = source_rule.get('known_omissions_by_row', {}).get(rid, [])
        values, actions, before, total, reasons = process_composition(
            raw, infer_fe=rid in rules['supported_Fe_balance_rows'], known_omissions=omitted)
        source_info = source_audit[record.source_id]
        target_issue = source_rule.get('target_issue', '')
        eligibility_reasons = reasons + ([target_issue] if target_issue else [])
        if source_info['known_training_source_overlap']:
            eligibility_reasons.append('known_training_source_overlap')
        env = {RENAMES.get(c, c): old.loc[rid, c] for c in env_columns}
        rows.append({'row_id': rid, **values, **env, 'reference_type': 'SCE'})
        audits.append(dict(row_id=rid, source_id=record.source_id, steel_type=ws.cell(row, 3).value,
            composition_sum_before=before, composition_sum_after=total,
            composition_eligible=not reasons, composition_reasons=';'.join(reasons),
            eligible_candidate=not eligibility_reasons, eligibility_reasons=';'.join(eligibility_reasons),
            Fe_action=actions['Fe'], assumed_zero_count=sum(a.startswith('assumed_zero') for a in actions.values()),
            target_status='needs_primary_source_resolution' if target_issue else 'workbook_reported_not_fully_verified',
            reference_evidence='workbook_V_SCE_header_and_usage_notes',
            source_overlap_status=source_info['overlap_review']))
        for e in elements:
            logs.append(dict(row_id=rid, feature=e, sheet=ws.title, coordinate=coordinates[e],
                original_value=raw[e], workbook_formula=';'.join(raw_formulas[e]), value=values[e],
                action=actions[e], evidence=source_rule.get('evidence_url', '') if e in omitted else
                (rules['supported_Fe_balance_rows'].get(rid, '') if actions[e] == 'calculated_reviewed_Fe_balance' else 'raw_workbook')))
    wb.close()
    formulas.close()
    frame = pd.DataFrame(rows).set_index('row_id')
    audit = pd.DataFrame(audits).set_index('row_id')
    eligible = audit.eligible_candidate
    development = assignments.set_index('row_id').outer_split.eq('development')
    final_ids = assignments.loc[assignments.outer_split.eq('final_test'), 'row_id'].tolist()
    query = frame.loc[final_ids]
    units = {c: ('wt.%' if c in elements or c.startswith('cement_') else
                  'degrees_C' if c == 'CP_temp' else 'dimensionless' if c == 'CP_pH' else
                  'category' if c == 'reference_type' else 'mol/L') for c in frame}
    output.mkdir(parents=True)
    frame.to_csv(output / 'features_all_frozen_rows.csv')
    frame.loc[eligible].to_csv(output / 'eligible_candidate_features.csv')
    targets.loc[eligible].to_csv(output / 'eligible_candidate_targets.csv')
    frame.loc[eligible & development].to_csv(output / 'development_context_features.csv')
    targets.loc[eligible & development].to_csv(output / 'development_context_targets.csv')
    query.to_csv(output / 'final_test_queries.csv')
    targets.loc[final_ids].to_csv(output / 'final_test_scoring_targets.csv')
    assignments.to_csv(output / 'split_assignments.csv', index=False)
    audit.to_csv(output / 'composition_row_audit.csv')
    all_audit.join(audit.drop(columns='source_id'), how='left').to_csv(output / 'row_audit.csv')
    audit.loc[final_ids].to_csv(output / 'final_query_audit.csv')
    pd.DataFrame(logs).to_csv(output / 'composition_processing_log.csv', index=False)
    (output / 'source_audit.json').write_text(json.dumps(source_audit, indent=2) + '\n')
    registry = dict(schema_version='rebar_reviewed_v2', target_column=TARGET, target_unit='mV vs SCE (workbook convention)',
        feature_columns_in_order=frame.columns.tolist(), composition_features=elements,
        categorical_features=['reference_type'], units=units, approved_renames=RENAMES,
        excluded_predictors=EXCLUDED, missingness='Unknown stays NaN; only accepted complete compositions receive audited assumed zeros.',
        scope='Evaluation candidates only; original-paper target and independence verification incomplete.')
    (output / 'feature_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    report = dict(status='composition_screened_candidates_not_fully_source_verified',
        frozen_rows=len(frame), eligible_candidates=int(eligible.sum()),
        composition_eligible=int(audit.composition_eligible.sum()),
        development_context_rows=int((eligible & development).sum()),
        frozen_final_query_rows=len(query), final_queries_needing_review=int((~eligible.loc[final_ids]).sum()),
        all_final_rows_pass_candidate_screen=bool(eligible.loc[final_ids].all()),
        final_evaluation_ready=False,
        fully_verified_independent_benchmark=False,
        feature_count=len(frame.columns), composition_feature_count=len(elements),
        distinct_eligible_compositions=len(frame.loc[eligible, elements].drop_duplicates()),
        eligible_publications=int(audit.loc[eligible, 'source_id'].nunique()),
        Fe_actions=audit.Fe_action.value_counts().to_dict(),
        composition_decisions=audit.composition_reasons.value_counts().to_dict(),
        eligible_feature_coverage=frame.loc[eligible].notna().sum().to_dict(),
        limitations=['Original-paper verification and source-overlap checks remain incomplete.',
            'Workbook numeric zeros may include upstream assumptions.',
            'Unknown inhibitor and preparation details remain unobserved.',
            'All outer queries are preserved, including flagged rows; eligible candidates are not a replacement final-query set.',
            'Existing v1 evaluator defaults have not been migrated to this candidate export.'])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    evidence_paths = [BASE / p for p in rules['evidence_files']]
    inputs = [RAW, RULES, EPIT_REF, SOCCOL_REF, DEFAULT_SPLIT, parent / 'split_lock.json',
              Path(__file__), Path(__file__).with_name('prepare_evaluation.py'),
              Path(__file__).with_name('evaluation_data.py'), *evidence_paths,
              *[parent / name for name in manifest['exports']]]
    provenance = dict(schema_version='rebar_reviewed_v2',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        exports={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        frozen_final_query_row_ids=final_ids, split_policy='Original memberships retained verbatim; no new split.',
        data_policy='No target fitting, imputation, scaling, prior generation, training or evaluation.')
    (output / 'processing_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'processing_lock.json').write_text(json.dumps({'manifest_sha256': sha(output / 'processing_manifest.json')}, indent=2) + '\n')
    return report


def validate(output=DEFAULT_OUTPUT):
    output = Path(output)
    path = output / 'processing_manifest.json'
    if sha(path) != json.loads((output / 'processing_lock.json').read_text())['manifest_sha256']:
        raise RuntimeError('Reviewed manifest changed.')
    manifest = json.loads(path.read_text())
    for group, base in [('inputs', ROOT), ('exports', output)]:
        for name, digest in manifest[group].items():
            if sha(base / name) != digest:
                raise RuntimeError(f'Reviewed {group} changed: {name}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        validate(args.output_dir)
        print('Reviewed input/output hashes verified.')
    else:
        result = prepare(args.output_dir)
        print(json.dumps({k: v for k, v in result.items() if k != 'eligible_feature_coverage'}, indent=2))
