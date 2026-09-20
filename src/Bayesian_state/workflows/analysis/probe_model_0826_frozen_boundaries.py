"""One bounded outer-support check after the frozen three-subject validation.

Reuse the shared scorer and decision diagnostic. Keep every historical finalist;
compare each fixed inner point with every outer point, with a preallocated alpha.
No adaptive search, nominee replacement, default-support change, or extra tier.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path

import numpy as np

from ...optimization.adaptive_boundary import adaptive_support
from ...optimization.adaptive_fit import make_context, merge_proposals, proposal_seed, select_subjects
from ...optimization.adaptive_runtime import (
    Scorer, cache_path, digest, freeze_json, phase_seeds, run_manifest, verify_manifest,
)
from ...optimization.diagnostics.decision_precision import decision_diagnostics
from ...optimization.model_0826 import ETA_MINUS_PATH, ETA_PLUS_PATH, extract_model_0826_parameters
from ...optimization.parameter_space import load_model_parameter_space
from ...optimization.search.adaptive_proposals import candidate, point_id
from ...utils.paths import ROOT_DIR
from ...utils.seeding import stable_seed
from .calibrate_model_0826_boundaries import change_point, validate_banks
from .diagnose_model_0826_numerics import load_protocol

DEFAULT_CONFIG = ROOT_DIR/'configs/exp123/specific_models/model_0826_frozen_boundary_check.yaml'
FAMILY = 'frozen_boundary_check'


def outer_point(point: dict, changes: dict, preserve_error: bool = False) -> dict:
    """Extend the existing boundary helper with the two declared update rates."""
    rates = {'eta_plus': ETA_PLUS_PATH, 'eta_minus': ETA_MINUS_PATH}
    updated = change_point(point, {k: v for k, v in changes.items() if k not in rates},
                           preserve_error=preserve_error)
    for name, path in rates.items():
        if name in changes:
            value = changes[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError('Update-rate probes must be positive and <= 1; zero is a separate ablation')
            updated[path] = value
    return updated


def fixed_banks(protocol: dict, config: dict) -> tuple[dict, dict]:
    archived = json.loads((Path(protocol['previous'])/'fit_results.json').read_text())['subjects']
    original = load_model_parameter_space(config['parameter_space'])
    _, _, original_support = adaptive_support(original, {})
    extensions = deepcopy(protocol['extensions'])
    banks = {}
    for case in protocol['cases']:
        sid = case['subject']
        if case['kind'] != 'boundary' or len(case['budgets']) != 1:
            raise ValueError('Boundary checks have exactly one unconditional numerical budget')
        old = archived[str(sid)]
        by_id = {r['id']: r for r in old['candidate_bank']}
        if set(case['references']) != set(by_id) or case['nominee'] != old['selected']:
            raise ValueError('Keep the entire historical bank and original nominee')
        names = [p['name'] for p in case['probes']]
        if not names or len(names) != len(set(names)):
            raise ValueError('Need unique nonempty probe names')
        refs = []
        for pid in case['references']:
            point = by_id[pid]['hyperparams']
            if point_id(point) != pid:
                raise ValueError('Historical point identity mismatch')
            refs.append(candidate(point, 'historical_reference', historical_id=pid))
        validate_banks({sid: refs}, original_support)
        groups = [('historical_reference', refs)]
        for probe in case['probes']:
            preserve = probe.get('preserve_E_E', False)
            point = outer_point(by_id[probe['anchor']]['hyperparams'], probe['changes'], preserve)
            if preserve:
                extensions.setdefault('delta_E', []).append(extract_model_0826_parameters(point)['delta_E'])
            row = candidate(point, probe['name'], **probe)
            try:
                validate_banks({sid: [row]}, original_support)
            except ValueError:
                pass
            else:
                raise ValueError('Every outer probe must actually leave the original support')
            groups.append((probe['name'], [row]))
        banks[sid] = merge_proposals(groups)
    _, _, support = adaptive_support(original, extensions)
    validate_banks(banks, support)
    return banks, support


def boundary_decision(arrays: dict, inner_ids: list[str], protocol: dict, sid: int) -> dict:
    """Simultaneous fixed-inner comparisons, not a newly selected-winner audit.

    An inner point within tolerance of all outer points suffices for this finite
    bank. Outer improvement is supported only when EVERY inner point is worse
    beyond tolerance. Otherwise retain uncertainty. Alpha is divided over all
    predeclared inner points; each comparison already uses maximum outer regret.
    """
    if not inner_ids or len(set(inner_ids)) != len(inner_ids) or not set(inner_ids) < set(arrays):
        raise ValueError('Need distinct inner references and at least one outer candidate')
    outer = {pid: a for pid, a in arrays.items() if pid not in inner_ids}
    alpha = protocol['alpha']/len(inner_ids)
    seed = stable_seed({'base': protocol['base_seed'], 'subject': sid, 'role': 'inner_vs_outer'})
    checks = {pid: decision_diagnostics({pid: arrays[pid], **outer}, pid, protocol['tolerance'],
               alpha, protocol['bootstrap_replicates'], seed) for pid in inner_ids}
    acceptable = [pid for pid, d in checks.items() if d['status'] == 'acceptable_within_bank']
    status = ('no_material_gain_in_tested_bank' if acceptable else
              'outer_gain_confirmed_in_tested_bank' if all(d['status'] == 'selected_point_inferior' for d in checks.values())
              else 'unresolved')
    return {'status': status, 'acceptable_inner_references': acceptable,
            'inner_comparisons': checks, 'per_inner_alpha': alpha,
            'best_inner_regret_lower': min(d['regret_lower'] for d in checks.values()),
            'best_inner_regret_upper': min(d['regret_upper'] for d in checks.values()),
            'scope': 'Approximate numerical-seed comparisons of finite fixed inner/outer banks. No unrestricted profile search, global range validation, unique parameter, or original-fit acceptance claim.'}


def cached_sensitivity(protocol: dict, historical: dict) -> tuple[dict, dict]:
    """One 60000-draw sensitivity check, preserving the original test and seed."""
    previous = Path(protocol['previous'])
    old = json.loads((previous/'fit_results.json').read_text())['subjects']
    scorer = Scorer(previous, historical['config'], {})
    results, hashes = {}, {}
    for case in protocol['cases']:
        sid = case['subject']
        if sid not in (222, 315):
            continue
        row = old[str(sid)]
        budget, family = row['audit_budget'], row['audit_family']
        cycle = family.split('/')[0]
        index = next(i for i, b in enumerate(historical['config']['precision']['tiers']) if b == budget)
        receipt_path = previous/'batches'/cycle/f'audit_{index}'/'scores.json'
        receipt = json.loads(receipt_path.read_text())
        hashes[str(receipt_path)] = digest(receipt_path)
        arrays = {}
        for point in row['candidate_bank']:
            for seed in phase_seeds(historical['config']['base_seed'], sid, family, budget['filter_seed_count']):
                path = cache_path(previous, sid, point['id'], budget['particle_count'], seed)
                actual = digest(path)
                if actual != receipt['cache_sha256'][str(path.relative_to(previous))]:
                    raise ValueError('Historical cache checksum differs')
                hashes[str(path)] = actual
            arrays[point['id']] = scorer.arrays(sid, point, budget, family)
        diagnostic = row['independent_audit']
        seed = proposal_seed(historical['config'], sid, f'{cycle}/audit_bootstrap_{index}')
        updated = decision_diagnostics(arrays, row['selected'], diagnostic['tolerance'], diagnostic['alpha'], 60000, seed)
        results[str(sid)] = {'original_6000': diagnostic, 'sensitivity_60000': updated,
                            'bootstrap_seed': seed, 'budget': budget, 'new_pf_runs': 0,
                            'scope': 'Post-hoc numerical resampling sensitivity, not new data or a replacement acceptance decision.'}
    return results, hashes


def run(path: Path, output: Path | None, *, smoke: bool = False, resume: bool = False,
        dry_run: bool = False, cached_only: bool = False) -> dict:
    path = path.resolve()
    config, protocol = load_protocol(path, smoke)
    previous = Path(protocol['previous'])
    historical = json.loads((previous/'manifest.json').read_text())
    verify_manifest(historical)
    banks, support = fixed_banks(protocol, config)
    specs = select_subjects(config, [c['subject'] for c in protocol['cases']], None, smoke)
    contexts = {s['subject']: make_context(s, config, smoke) for s in specs}
    for case in protocol['cases']:
        sid = case['subject']
        fresh = set(phase_seeds(config['base_seed'], sid, FAMILY, case['budgets'][0]['filter_seed_count']))
        old_seeds = {int(p.name.split('_seed')[1]) for p in (previous/'cache'/str(sid)).glob('R*_seed*')}
        if fresh & old_seeds:
            raise ValueError('Boundary evaluation shares historical seeds')
    plan = {'protocol': protocol, 'smoke': smoke, 'cached_only': cached_only, 'parallel_budget': config['parallel_budget'],
            'subjects': [{**s, 'trials': len(contexts[s['subject']]['arrays'].choices),
                          'candidates': len(banks[s['subject']])} for s in specs],
            'max_new_pf_runs': 0 if cached_only else sum(len(banks[c['subject']])*c['budgets'][0]['filter_seed_count'] for c in protocol['cases'])}
    if dry_run:
        return plan
    if output is None:
        raise ValueError('Need --output-dir')
    output = output.resolve()
    if resume:
        if not (output/'manifest.json').is_file():
            raise ValueError('Resume requires a manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cached, hashes = cached_sensitivity(protocol, historical)
        manifest = run_manifest(config, path, specs, smoke)
        manifest['input_sha256'].update(hashes)
        for source in (Path(protocol['fit_config']), previous/'manifest.json', previous/'fit_results.json'):
            manifest['input_sha256'][str(source)] = digest(source)
        freeze_json(output/'manifest.json', manifest)
        freeze_json(output/'plan.json', plan)
        freeze_json(output/'frozen_banks.json', {str(s): rows for s, rows in banks.items()})
        freeze_json(output/'support.json', support)
        freeze_json(output/'cached_sensitivity.json', cached)
        result = {'subjects': {}, 'smoke_only': smoke, 'cached_only': cached_only}
        if not cached_only:
            scorer = Scorer(output, config, contexts)
            groups = {}
            for case in protocol['cases']:
                budget = case['budgets'][0]
                key = (budget['particle_count'], budget['filter_seed_count'])
                groups.setdefault(key, {})[case['subject']] = banks[case['subject']]
            scored = {}
            for (particles, seeds), group in groups.items():
                budget = {'particle_count': particles, 'filter_seed_count': seeds}
                scored.update(scorer.batch(group, budget, FAMILY, f'R{particles}_B{seeds}'))
            for case in protocol['cases']:
                sid, budget = case['subject'], case['budgets'][0]
                arrays = {r['id']: scorer.arrays(sid, r, budget, FAMILY) for r in scored[sid]}
                diagnostic = boundary_decision(arrays, case['references'], protocol, sid)
                row = {'budget': budget, 'boundary_diagnostic': diagnostic, 'rows': scored[sid],
                       'original_nominee': case['nominee'], 'whole_fit_status': 'not_reassessed'}
                freeze_json(output/'subjects'/str(sid)/'boundary_result.json', row)
                result['subjects'][str(sid)] = row
        verify_manifest(manifest)
        freeze_json(output/'results.json', result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cached-only', action='store_true')
    args = parser.parse_args()
    result = run(args.config, args.output_dir, smoke=args.smoke, resume=args.resume,
                 dry_run=args.dry_run, cached_only=args.cached_only)
    print(json.dumps(result if args.dry_run else {'output': str(args.output_dir), 'complete': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
