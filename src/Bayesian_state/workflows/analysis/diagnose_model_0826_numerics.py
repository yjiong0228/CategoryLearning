"""Read archived seed variability, then audit predeclared numerical/boundary probes.

This bounded follow-up reuses the production scorer. It does not search, change
production defaults, promote a winner seen in its own audit, or certify a fit.
Only each case's last budget supplies a primary fixed-bank decision.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import yaml

from ...optimization.adaptive_boundary import adaptive_support
from ...optimization.adaptive_fit import load_fit_config, make_context, merge_proposals, select_subjects
from ...optimization.adaptive_runtime import (
    Scorer, cache_path, digest, freeze_json, phase_seeds, run_manifest, verify_manifest,
)
from ...optimization.diagnostics.decision_precision import decision_diagnostics, mixture_nll
from ...optimization.model_0826 import extract_model_0826_parameters
from ...optimization.parameter_space import load_model_parameter_space
from ...optimization.search.adaptive_proposals import candidate, point_id
from ...utils.paths import ROOT_DIR
from ...utils.seeding import stable_seed
from .calibrate_model_0826_boundaries import change_point, validate_banks

DEFAULT_CONFIG = ROOT_DIR/'configs/exp123/specific_models/model_0826_numerical_boundary_followup.yaml'
FAMILY = 'numerical_boundary_followup'


def load_protocol(path: Path, smoke: bool = False) -> tuple[dict, dict]:
    protocol = yaml.safe_load(path.read_text())
    if set(protocol) != {'schema_version', 'fit_config', 'previous', 'base_seed', 'alpha',
                         'tolerance', 'bootstrap_replicates', 'extensions', 'cases'} or protocol['schema_version'] != 1:
        raise ValueError('Expected numerical/boundary follow-up schema 1')
    for key in ('fit_config', 'previous'):
        protocol[key] = str((path.parent/protocol[key]).resolve())
    config = load_fit_config(Path(protocol['fit_config']), smoke)
    if type(protocol['base_seed']) is not int or protocol['base_seed'] < 1:
        raise ValueError('Need a positive integer base seed')
    config['base_seed'] = protocol['base_seed']
    if (not 0 < protocol['alpha'] < .5 or not np.isfinite(protocol['tolerance']) or
            protocol['tolerance'] <= 0 or type(protocol['bootstrap_replicates']) is not int or
            protocol['bootstrap_replicates'] < 100):
        raise ValueError('Invalid diagnostic settings')
    cases = protocol['cases']
    if not cases or len({c['subject'] for c in cases}) != len(cases):
        raise ValueError('Need unique nonempty cases')
    for case in cases:
        if set(case) != {'subject', 'kind', 'nominee', 'references', 'probes', 'budgets'}:
            raise ValueError('Invalid case fields')
        ids = case['references']
        if (type(case['subject']) is not int or case['kind'] not in ('numerical', 'boundary') or
                not ids or len(set(ids)) != len(ids) or case['nominee'] not in ids):
            raise ValueError('Invalid subject, fixed references or nominee')
        budgets = case['budgets']
        if not budgets or len({b['particle_count'] for b in budgets}) != len(budgets):
            raise ValueError('Need distinct particle budgets')
        if len({b['filter_seed_count'] for b in budgets}) != 1:
            raise ValueError('Particle comparisons require equal seed counts')
        if [b['particle_count'] for b in budgets] != sorted(b['particle_count'] for b in budgets):
            raise ValueError('Final particle budget must be the largest')
        for budget in budgets:
            if set(budget) != {'particle_count', 'filter_seed_count'} or any(type(v) is not int or v < 2 for v in budget.values()):
                raise ValueError('Invalid numerical budget')
        for probe in case['probes']:
            if (set(probe)-{'name', 'anchor', 'changes', 'preserve_E_E'} or
                    not {'name', 'anchor', 'changes'} <= probe.keys() or probe['anchor'] not in ids or
                    type(probe.get('preserve_E_E', False)) is not bool):
                raise ValueError('Invalid boundary probe')
    if smoke:
        protocol['cases'] = deepcopy(cases[:1])
        protocol['cases'][0]['budgets'] = [{'particle_count': 2, 'filter_seed_count': 2}]
        protocol['bootstrap_replicates'] = 100
        config['parallel_budget'] = 1
    return config, protocol


def fixed_banks(protocol: dict, config: dict) -> tuple[dict, dict, dict]:
    archived = json.loads((Path(protocol['previous'])/'fit_results.json').read_text())['subjects']
    banks, excluded, extensions = {}, {}, deepcopy(protocol['extensions'])
    for case in protocol['cases']:
        sid = case['subject']
        by_id = {r['id']: r for r in archived[str(sid)]['candidate_bank']}
        if not set(case['references']) <= by_id.keys():
            raise ValueError(f'Missing historical reference for S{sid}')
        refs = []
        for pid in case['references']:
            point = by_id[pid]['hyperparams']
            if point_id(point) != pid:
                raise ValueError('Historical point identity mismatch')
            refs.append(candidate(point, 'historical_reference', historical_id=pid))
        groups = [('historical_reference', refs)]
        for probe in case['probes']:
            preserve = probe.get('preserve_E_E', False)
            point = change_point(by_id[probe['anchor']]['hyperparams'], probe['changes'], preserve_error=preserve)
            if preserve:
                # E_E is fixed, so delta_E is a derived coordinate, not a new search grid.
                extensions.setdefault('delta_E', []).append(extract_model_0826_parameters(point)['delta_E'])
            groups.append((probe['name'], [candidate(point, probe['name'], **probe)]))
        banks[sid] = merge_proposals(groups)
        if len(banks[sid]) < 2:
            raise ValueError('A fixed comparison requires at least two points')
        excluded[str(sid)] = sorted(set(by_id)-set(case['references']))
    _, _, support = adaptive_support(load_model_parameter_space(config['parameter_space']), extensions)
    validate_banks(banks, support)
    return banks, support, excluded


def trial_losses(arrays: dict) -> np.ndarray:
    p = arrays['probabilities'].mean(axis=0)
    return -np.log(np.clip(p[np.arange(len(p)), arrays['observed']][arrays['mask']], 1e-12, 1.))


def numerical_summary(arrays: dict) -> dict:
    """Descriptive seed sensitivity; no trial-independent uncertainty estimate."""
    p, y, mask = arrays['probabilities'], arrays['observed'], arrays['mask']
    score = mixture_nll(p, y, mask)
    midpoint = len(p)//2
    loo = [mixture_nll(np.delete(p, i, axis=0), y, mask) for i in range(len(p))]
    prefixes = {str(n): mixture_nll(p[:n], y, mask) for n in (16, 32, 64) if n <= len(p)}
    return {'mean_nll': score, 'seed_count': len(p), 'trials': len(y), 'scored_trials': int(mask.sum()),
            'prefix_mean_nll': prefixes,
            'half_mean_nll': [mixture_nll(q, y, mask) for q in (p[:midpoint], p[midpoint:])],
            'max_leave_one_seed_out_change': float(np.max(np.abs(np.asarray(loo)-score))),
            'min_observed_mean_probability': float(np.exp(-trial_losses(arrays).max())),
            'single_seed_nll_sd': float(np.std([mixture_nll(q[None], y, mask) for q in p], ddof=1)),
            'split_half_probability_rmse': float(np.sqrt(np.mean((p[:midpoint, mask].mean(0)-p[midpoint:, mask].mean(0))**2)))}


def trial_contrast(left: dict, right: dict) -> dict:
    """Locate absolute loss changes without removing or reweighting any trial."""
    for key in ('observed', 'mask'):
        np.testing.assert_array_equal(left[key], right[key])
    delta = trial_losses(right)-trial_losses(left)
    index = np.flatnonzero(left['mask'])+1
    order = np.argsort(-np.abs(delta))[:10]
    total = np.abs(delta).sum()
    return {'right_minus_left_mean_nll': float(delta.mean()),
            'top10_absolute_change_fraction': float(np.abs(delta[order]).sum()/total) if total else 0.,
            'top_trials': [{'trial': int(index[i]), 'loss_change': float(delta[i])} for i in order],
            'block64_mean_contributions': [float(delta[(index > start) & (index <= start+64)].sum()/len(delta))
                                           for start in range(0, len(left['mask']), 64)]}


def cached_diagnosis(protocol: dict, historical: dict, banks: dict) -> tuple[dict, dict]:
    root = Path(protocol['previous'])
    scorer = Scorer(root, historical['config'], {})
    available, hashes = {}, {}
    for path in sorted((root/'batches').rglob('plan.json')):
        plan = json.loads(path.read_text())
        if plan['budget']['particle_count'] < 128:
            continue
        receipt_path = path.with_name('scores.json')
        receipt = json.loads(receipt_path.read_text())
        for sid, rows in plan['banks'].items():
            for row in rows:
                key = (int(sid), row['id'], plan['family'], plan['budget']['particle_count'], plan['budget']['filter_seed_count'])
                available[key] = (plan['budget'], receipt, receipt_path)
    output = {}
    for case in protocol['cases']:
        if case['kind'] != 'numerical':
            continue
        sid, points = case['subject'], {}
        for row in banks[sid]:
            looks, raw = {}, {}
            for key, (budget, receipt, receipt_path) in sorted(available.items()):
                if key[:2] != (sid, row['id']):
                    continue
                family = key[2]
                for seed in phase_seeds(historical['config']['base_seed'], sid, family, budget['filter_seed_count']):
                    path = cache_path(root, sid, row['id'], budget['particle_count'], seed)
                    actual = digest(path)
                    if actual != receipt['cache_sha256'][str(path.relative_to(root))]:
                        raise ValueError('Archived PF cache checksum differs')
                    hashes[str(path)] = actual
                hashes[str(receipt_path)] = digest(receipt_path)
                label = f'{family}/R{key[3]}_B{key[4]}'
                raw[label] = scorer.arrays(sid, row, budget, family)
                looks[label] = numerical_summary(raw[label])
            comparisons = {f'{a} -> {b}': trial_contrast(raw[a], raw[b]) for a, b in combinations(raw, 2)
                           if a.rsplit('/', 1)[1] == b.rsplit('/', 1)[1]}
            points[row['id']] = {'looks': looks, 'equal_budget_family_contrasts': comparisons}
        output[str(sid)] = points
    final = json.loads((root/'fit_results.json').read_text())['subjects']
    boundaries = {sid: {'selected': r['selected'], 'selected_parameters': r['parameters'],
                        'representative_hits': [h for h in r['boundary']['hits'] if h['candidate'] == r['selected']],
                        'alternative_hits': [h for h in r['boundary']['hits'] if h['candidate'] != r['selected']]}
                  for sid, r in final.items()}
    return {'numerical': output, 'boundaries': boundaries,
            'scope': 'Read-only historical diagnostics. Single-seed NLL SD is not the error of the mean-probability estimator. No new candidate selection or trial exclusions.'}, hashes


def execute(scorer: Scorer, protocol: dict, banks: dict, output: Path) -> dict:
    # Group equal budgets across subjects; every declared budget runs unconditionally.
    groups = {}
    for case in protocol['cases']:
        for budget in case['budgets']:
            key = (budget['particle_count'], budget['filter_seed_count'])
            groups.setdefault(key, {})[case['subject']] = banks[case['subject']]
    scored = {}
    for (particles, seeds), group in sorted(groups.items()):
        budget = {'particle_count': particles, 'filter_seed_count': seeds}
        scored[(particles, seeds)] = scorer.batch(group, budget, FAMILY, f'R{particles}_B{seeds}')
    results = {}
    for case in protocol['cases']:
        sid, looks = case['subject'], []
        for index, budget in enumerate(case['budgets']):
            rows = scored[(budget['particle_count'], budget['filter_seed_count'])][sid]
            arrays = {r['id']: scorer.arrays(sid, r, budget, FAMILY) for r in rows}
            diagnostic = decision_diagnostics(arrays, case['nominee'], protocol['tolerance'], protocol['alpha'],
                protocol['bootstrap_replicates'], stable_seed({'base': protocol['base_seed'], 'subject': sid, 'budget': budget}))
            look = {'budget': budget, 'primary': index == len(case['budgets'])-1, 'diagnostic': diagnostic,
                    'rows': rows, 'numerical': {pid: numerical_summary(a) for pid, a in arrays.items()}}
            freeze_json(output/'subjects'/str(sid)/f'R{budget["particle_count"]}.json', look)
            looks.append(look)
        results[str(sid)] = {'kind': case['kind'], 'nominee': case['nominee'], 'looks': looks,
                             'fixed_bank_status': looks[-1]['diagnostic']['status'],
                             'whole_fit_status': 'not_reassessed'}
    return {'subjects': results, 'scope': 'Only final budget is primary; fixed banks, approximate seed bootstrap at finite R. No whole-fit/global optimum, parameter interval, holdout or state-precision claim.'}


def run(path: Path, output: Path | None, *, smoke: bool = False, resume: bool = False,
        dry_run: bool = False, cache_only: bool = False) -> dict:
    path = path.resolve()
    config, protocol = load_protocol(path, smoke)
    previous = Path(protocol['previous'])
    historical = json.loads((previous/'manifest.json').read_text())
    verify_manifest(historical)
    subjects = select_subjects(config, [c['subject'] for c in protocol['cases']], None, smoke)
    contexts = {s['subject']: make_context(s, config, smoke) for s in subjects}
    banks, support, excluded = fixed_banks(protocol, config)
    for case in protocol['cases']:
        fresh = set(phase_seeds(config['base_seed'], case['subject'], FAMILY, max(b['filter_seed_count'] for b in case['budgets'])))
        old = {int(p.name.split('_seed')[1]) for p in (previous/'cache'/str(case['subject'])).glob('R*_seed*')}
        if fresh & old:
            raise ValueError('Fresh audit shares historical seeds')
    plan = {'protocol': protocol, 'excluded_historical_candidates': excluded,
            'parallel_budget': config['parallel_budget'], 'smoke': smoke, 'cache_only': cache_only,
            'max_new_pf_runs': 0 if cache_only else sum(len(banks[c['subject']])*sum(b['filter_seed_count'] for b in c['budgets']) for c in protocol['cases']),
            'subjects': [{'subject': s['subject'], 'condition': s['condition'], 'trials': len(contexts[s['subject']]['arrays'].choices)} for s in subjects]}
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
        cached, hashes = cached_diagnosis(protocol, historical, banks)
        manifest = run_manifest(config, path, subjects, smoke)
        manifest['scope'] = 'Bounded numerical/boundary diagnosis; historical cache read-only; no search or default change.'
        manifest['input_sha256'].update(hashes)
        for source in (Path(protocol['fit_config']), previous/'manifest.json', previous/'fit_results.json'):
            manifest['input_sha256'][str(source)] = digest(source)
        freeze_json(output/'manifest.json', manifest)
        freeze_json(output/'plan.json', plan)
        freeze_json(output/'frozen_banks.json', {str(s): rows for s, rows in banks.items()})
        freeze_json(output/'support.json', support)
        freeze_json(output/'nominations.json', {str(c['subject']): c['nominee'] for c in protocol['cases']})
        freeze_json(output/'cached_diagnosis.json', cached)
        result = cached if cache_only else execute(Scorer(output, config, contexts), protocol, banks, output)
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
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    result = run(args.config, args.output_dir, smoke=args.smoke, resume=args.resume,
                 dry_run=args.dry_run, cache_only=args.cache_only)
    print(json.dumps(result if args.dry_run else {'output': str(args.output_dir), 'complete': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
