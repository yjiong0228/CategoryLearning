"""Bounded full-sequence boundary probes using the production compact scorer.

Historical files supply parameter points only. All scores are recomputed. The
screen selects a family-balanced bank; a fresh high-precision phase nominates a
point, then independent seeds audit that frozen nomination. No audit reselects
its own winner. This is conditional sensitivity, not profile likelihood or a
new global fit. See optimization/BOUNDARY_CALIBRATION.md.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from itertools import product
import json
from pathlib import Path

import numpy as np
import yaml

from ...optimization.adaptive_boundary import adaptive_support, boundary_report
from ...optimization.adaptive_fit import load_fit_config, make_context, merge_proposals, select_subjects
from ...optimization.adaptive_runtime import Scorer, digest, freeze_json, run_manifest, verify_manifest
from ...optimization.diagnostics.decision_precision import decision_diagnostics
from ...optimization.model_0826 import (
    ACCUMULATOR_GAIN_PATH, CAPACITY_PATH, EVENT_CORRECT_PATH, EVENT_ERROR_PATH,
    GAMMA_PATH, INITIAL_EVENT_PATH, REACTIVE_PROFILE_KEY, WORKSPACE_PROFILE_KEY,
    extract_model_0826_parameters,
)
from ...optimization.parameter_space import load_model_parameter_space, reactive_error_probability
from ...optimization.search.adaptive_proposals import candidate, point_id, ranking
from ...utils.paths import ROOT_DIR
from ...utils.seeding import stable_seed


DEFAULT_CONFIG = ROOT_DIR/'configs/exp123/specific_models/model_0826_boundary_calibration.yaml'


def load_protocol(path: Path, smoke: bool = False) -> tuple[dict, dict]:
    protocol = yaml.safe_load(path.read_text())
    expected = {'schema_version', 'fit_config', 'base_seed', 'screen', 'selection',
                'condition3_selection', 'condition3_audit_max_seeds', 'alpha', 'tolerance',
                'bootstrap_replicates', 'historical_banks', 'historical_stop', 'extensions', 'cases'}
    if set(protocol) != expected or protocol['schema_version'] != 1:
        raise ValueError('Expected boundary calibration schema 1')
    for key in ('fit_config', 'historical_banks', 'historical_stop'):
        protocol[key] = str((path.parent/protocol[key]).resolve())
    config = load_fit_config(Path(protocol['fit_config']))
    if type(protocol['base_seed']) is not int or protocol['base_seed'] < 1:
        raise ValueError('Invalid base seed')
    config['base_seed'] = protocol['base_seed']
    for name in ('screen', 'selection', 'condition3_selection'):
        budget = protocol[name]
        if set(budget) != {'particle_count', 'filter_seed_count'} or any(type(v) is not int or v < 2 for v in budget.values()):
            raise ValueError('Each budget needs particle/seed counts >= 2')
    if any(protocol['screen'][k] > protocol[name][k] for name in ('selection', 'condition3_selection') for k in protocol['screen']):
        raise ValueError('Selection cannot reduce the screen budget')
    maximum = protocol['condition3_audit_max_seeds']
    if type(maximum) is not int or maximum <= protocol['condition3_selection']['filter_seed_count']:
        raise ValueError('The optional second audit look must add seeds')
    if not 0 < protocol['alpha'] < .5 or not np.isfinite(protocol['tolerance']) or protocol['tolerance'] <= 0:
        raise ValueError('Invalid tolerance/alpha')
    if type(protocol['bootstrap_replicates']) is not int or protocol['bootstrap_replicates'] < 100:
        raise ValueError('At least 100 bootstrap draws required')
    ids = [c['subject'] for c in protocol['cases']]
    if not ids or any(type(s) is not int for s in ids) or len(set(ids)) != len(ids):
        raise ValueError('Need unique integer subject IDs')
    for case in protocol['cases']:
        if set(case) != {'subject', 'anchor', 'references', 'shortlist_size', 'families'}:
            raise ValueError('Invalid case keys')
        names = [f['name'] for f in case['families']]
        if len(names) != len(set(names)) or case['anchor'] not in case['references']:
            raise ValueError('Need distinct families and a mandatory anchor')
        if type(case['shortlist_size']) is not int or case['shortlist_size'] < len(case['references'])+len(names):
            raise ValueError('Shortlist must retain references and each family winner')
    if smoke:
        protocol['cases'] = protocol['cases'][:1]
        for name in ('screen', 'selection', 'condition3_selection'):
            protocol[name] = {'particle_count': 2, 'filter_seed_count': 2}
        protocol['condition3_audit_max_seeds'] = 4
        protocol['bootstrap_replicates'] = 100
        config['parallel_budget'] = 1
    config['calibration'] = protocol
    return config, protocol


def change_point(point: dict, changes: dict, *, preserve_error: bool = False) -> dict:
    """Change named scientific parameters, preserving untouched executable values."""
    if set(changes)-{'M', 'gamma', 'E_C', 'delta_E', 'c_A'}:
        raise ValueError('Unsupported boundary probe parameter')
    named = extract_model_0826_parameters(point)
    updated = deepcopy(point)
    if 'M' in changes:
        updated[WORKSPACE_PROFILE_KEY][CAPACITY_PATH] = changes['M']
    for key, path in [('gamma', GAMMA_PATH), ('c_A', ACCUMULATOR_GAIN_PATH)]:
        if key in changes:
            updated[path] = changes[key]
    if {'E_C', 'delta_E'} & changes.keys():
        ec = changes.get('E_C', named['E_C'])
        if preserve_error:
            if 'delta_E' in changes:
                raise ValueError('Preserving E_E cannot also prescribe delta_E')
            ee = named['E_E']
        else:
            delta = changes.get('delta_E', named['delta_E'])
            # Keep the original representation when the requested pair is unchanged.
            ee = (named['E_E'] if ec == named['E_C'] and np.isclose(delta, named['delta_E'], rtol=0, atol=1e-12)
                  else reactive_error_probability(ec, delta))
        if not 0 < ec <= ee < 1:
            raise ValueError('Probe violates reactive-event domain')
        updated[REACTIVE_PROFILE_KEY] = {EVENT_CORRECT_PATH: ec, EVENT_ERROR_PATH: ee,
                                        INITIAL_EVENT_PATH: ec}
    return updated


def build_banks(protocol: dict) -> tuple[dict[int, list[dict]], dict]:
    historical = json.loads(Path(protocol['historical_banks']).read_text())['banks']
    stop = json.loads(Path(protocol['historical_stop']).read_text())
    banks, extensions = {}, deepcopy(protocol['extensions'])
    for case in protocol['cases']:
        sid = case['subject']
        old = {r['id']: r for r in historical[str(sid)]}
        if sid == 103:
            old.update({r['id']: r for r in stop['base']})
        needed = set(case['references']) | {f.get('anchor', case['anchor']) for f in case['families']}
        if not needed <= old.keys():
            raise ValueError(f'Missing historical points for S{sid}')
        for pid in needed:
            if point_id(old[pid]['hyperparams']) != pid:
                raise ValueError('Historical point identity mismatch')
        groups = [('reference', [candidate(old[pid]['hyperparams'], 'reference', historical_id=pid)
                                  for pid in case['references']])]
        for family in case['families']:
            if set(family)-{'name', 'anchor', 'grid', 'points', 'preserve_E_E'} or ('grid' in family) == ('points' in family):
                raise ValueError('Specify exactly one grid or explicit points per family')
            anchor_id = family.get('anchor', case['anchor'])
            source = old[anchor_id]['hyperparams']
            if 'grid' in family:
                grid = family['grid']
                changes = [dict(zip(grid, values)) for values in product(*grid.values())]
            else:
                changes = family['points']
            rows = []
            for change in changes:
                point = change_point(source, change, preserve_error=family.get('preserve_E_E', False))
                if family.get('preserve_E_E', False):
                    extensions.setdefault('delta_E', []).append(extract_model_0826_parameters(point)['delta_E'])
                rows.append(candidate(point, family['name'], anchor=anchor_id, changes=change,
                                      preserve_E_E=family.get('preserve_E_E', False)))
            groups.append((family['name'], rows))
        banks[sid] = merge_proposals(groups)
    return banks, extensions


def validate_banks(banks: dict[int, list[dict]], support: dict) -> None:
    specs = support['subject_parameters']
    cells = {(p['M'], p['chi']) for p in specs['workspace_execution']['fine_candidates']}
    for rows in banks.values():
        for row in rows:
            named = extract_model_0826_parameters(row['hyperparams'])
            if (named['M'], named['chi']) not in cells or not 0 < named['E_C'] <= named['E_E'] < 1:
                raise ValueError('Candidate outside workspace/reactive support')
            for name, value in named.items():
                if name not in specs:
                    continue
                spec = specs[name]
                values = spec.get('fine_values', [spec.get('zero_value', 0), *spec.get('fine_positive_values', [])])
                if not any(np.isclose(value, v, rtol=0, atol=1e-9) for v in values):
                    raise ValueError(f'Undeclared probe {name}={value}')


def select_bank(rows: list[dict], case: dict) -> list[dict]:
    """Mandatory historical controls, then each family winner, then score order."""
    ranked, selected = ranking(rows), {}
    by_id = {r['id']: r for r in rows}
    for pid in case['references']:
        selected[pid] = by_id[pid]
    for family in case['families']:
        winner = next(r for r in ranked if family['name'] in r['sources'])
        selected[winner['id']] = winner
    for row in ranked:
        if len(selected) >= case['shortlist_size']:
            break
        selected[row['id']] = row
    return list(selected.values())


def audit_result(scorer: Scorer, sid: int, rows: list[dict], budget: dict,
                 nominee: str, baseline: str, protocol: dict, condition: int) -> dict:
    arrays = {r['id']: scorer.arrays(sid, r, budget, 'boundary_audit') for r in rows}
    alpha = protocol['alpha']/(2 if condition == 3 else 1)
    diagnostic = decision_diagnostics(
        arrays, nominee, protocol['tolerance'], alpha, protocol['bootstrap_replicates'],
        stable_seed({'base': protocol['base_seed'], 'subject': sid, 'role': 'boundary_diagnostic', 'budget': budget}))
    pair = diagnostic['selected_minus_candidates'].get(baseline)
    gain = ({'baseline_minus_nominee': -pair['selected_minus_candidate'],
             'descriptive_interval95': [-pair['interval95'][1], -pair['interval95'][0]]}
            if pair else {'baseline_minus_nominee': 0., 'descriptive_interval95': [0., 0.]})
    return {'budget': budget, 'decision': diagnostic, 'gain_over_historical_anchor': gain,
            'scored_rows': rows, 'trials': len(arrays[nominee]['observed']),
            'scored_trials': int(arrays[nominee]['mask'].sum())}


def calibrate(config: dict, protocol: dict, contexts: dict, banks: dict, support: dict, root: Path) -> dict:
    scorer = Scorer(root, config, contexts)
    screen = scorer.batch(banks, protocol['screen'], 'boundary_screen', 'screen')
    cases = {c['subject']: c for c in protocol['cases']}
    shortlist = {sid: select_bank(rows, cases[sid]) for sid, rows in screen.items()}
    freeze_json(root/'shortlists.json', {str(s): rows for s, rows in shortlist.items()})
    results = {}
    for group, condition3 in [('conditions12', False), ('condition3', True)]:
        subset = {s: rows for s, rows in shortlist.items() if (contexts[s]['condition'] == 3) == condition3}
        if not subset:
            continue
        budget = protocol['condition3_selection' if condition3 else 'selection']
        selected = scorer.batch(subset, budget, 'boundary_selection', group+'/selection')
        nominations = {s: ranking(rows)[0]['id'] for s, rows in selected.items()}
        freeze_json(root/group/'nominations.json', {str(s): p for s, p in nominations.items()})
        audited = scorer.batch(subset, budget, 'boundary_audit', group+'/audit_1')
        for sid, rows in audited.items():
            nominee = nominations[sid]
            looks = [audit_result(scorer, sid, rows, budget, nominee, cases[sid]['anchor'], protocol, contexts[sid]['condition'])]
            freeze_json(root/'subjects'/str(sid)/'audit_1.json', looks[0])
            if condition3 and looks[0]['decision']['status'] == 'unresolved':
                extended = {**budget, 'filter_seed_count': protocol['condition3_audit_max_seeds']}
                extra = scorer.batch({sid: subset[sid]}, extended, 'boundary_audit', group+f'/audit_2_{sid}')
                rows = extra[sid]
                looks.append(audit_result(scorer, sid, rows, extended, nominee, cases[sid]['anchor'], protocol, 3))
            results[str(sid)] = {
                'condition': contexts[sid]['condition'], 'historical_anchor': cases[sid]['anchor'],
                'nominee': nominee, 'parameters': extract_model_0826_parameters(next(r['hyperparams'] for r in rows if r['id'] == nominee)),
                'screen_count': len(screen[sid]), 'shortlist_count': len(rows),
                'selection_rows': selected[sid], 'audit_looks': looks,
                'boundary': boundary_report(rows, support, protocol['tolerance'], len(rows)),
                'status': looks[-1]['decision']['status'],
            }
            freeze_json(root/'subjects'/str(sid)/'result.json', results[str(sid)])
    return {'subjects': results, 'scope': 'Conditional full-sequence probes; no global optimization, parameter uncertainty, recovery, or population boundary-hit estimate. All scores recomputed. Audit nominee frozen; C3 has at most two looks with alpha/2 each (approximate bootstrap).'}


def run(path: Path, output: Path | None, *, smoke: bool = False, resume: bool = False, dry_run: bool = False) -> dict:
    path = path.resolve()
    config, protocol = load_protocol(path, smoke)
    subjects = select_subjects(config, [c['subject'] for c in protocol['cases']], None, smoke)
    contexts = {s['subject']: make_context(s, config, smoke) for s in subjects}
    banks, extensions = build_banks(protocol)
    original = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    _, _, support = adaptive_support(original, extensions)
    validate_banks(banks, support)
    plan = {'cases': [{'subject': s['subject'], 'condition': s['condition'], 'trials': len(contexts[s['subject']]['arrays'].choices),
                       'candidates': len(banks[s['subject']]), 'shortlist_size': c['shortlist_size']}
                      for s, c in zip(subjects, sorted(protocol['cases'], key=lambda c: c['subject']))],
            'protocol': protocol, 'parallel_budget': config['parallel_budget'], 'smoke': smoke}
    if dry_run:
        return plan
    if output is None:
        raise ValueError('Need a new --output-dir or an explicit --resume')
    output = output.resolve()
    if resume:
        if not (output/'manifest.json').exists():
            raise ValueError('Resume requires a manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = run_manifest(config, path, subjects, smoke)
        manifest['scope'] = 'Full-sequence conditional boundary calibration; historical files supply points, never cached scores.'
        for key in ('fit_config', 'historical_banks', 'historical_stop'):
            source = Path(protocol[key])
            manifest['input_sha256'][str(source)] = digest(source)
        freeze_json(output/'manifest.json', manifest)
        freeze_json(output/'plan.json', plan)
        freeze_json(output/'frozen_banks.json', {str(s): rows for s, rows in banks.items()})
        freeze_json(output/'effective_support.json', support)
        results = calibrate(config, protocol, contexts, banks, support, output)
        verify_manifest(manifest)
        freeze_json(output/'results.json', results)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    result = run(args.config, args.output_dir, smoke=args.smoke, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result if args.dry_run else {s: r['status'] for s, r in result['subjects'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
