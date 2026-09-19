"""Fresh fixed-pair audits and a cached S129 prediction comparison.

Only the final, prospectively chosen budget supplies a decision. Earlier
budgets diagnose particle sensitivity; neither pass promotes an entire fit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path

import numpy as np
import yaml

from ...optimization.adaptive_fit import load_fit_config, make_context, select_subjects
from ...optimization.adaptive_runtime import (
    Scorer, cache_path, digest, freeze_json, phase_seeds, run_manifest, verify_manifest,
)
from ...optimization.diagnostics.decision_precision import decision_diagnostics, mixture_nll
from ...optimization.model_0826 import extract_model_0826_parameters
from ...optimization.search.adaptive_proposals import point_id
from ...utils.paths import ROOT_DIR
from ...utils.seeding import stable_seed

DEFAULT_CONFIG = ROOT_DIR/'configs/exp123/specific_models/model_0826_precision_followup.yaml'
FAMILY = 'precision_followup_audit'


def load_protocol(path: Path, smoke: bool = False) -> tuple[dict, dict]:
    protocol = yaml.safe_load(path.read_text())
    if protocol['schema_version'] != 1:
        raise ValueError('Unsupported protocol version')
    for key in ('fit_config', 'previous'):
        protocol[key] = str((path.parent/protocol[key]).resolve())
    config = load_fit_config(Path(protocol['fit_config']), smoke)
    config['base_seed'] = int(protocol['base_seed'])
    if not 0 < protocol['alpha'] < 1 or protocol['tolerance'] <= 0 or protocol['bootstrap_replicates'] < 100:
        raise ValueError('Invalid diagnostic settings')
    cases = protocol['cases']
    if not cases or len({c['subject'] for c in cases}) != len(cases):
        raise ValueError('Need unique nonempty cases')
    for case in cases:
        ids = case['candidates']
        if len(ids) < 2 or len(set(ids)) != len(ids) or case['nominee'] not in ids:
            raise ValueError('Invalid fixed bank or nominee')
        if not case['budgets'] or len({b['particle_count'] for b in case['budgets']}) != len(case['budgets']):
            raise ValueError('Need distinct particle budgets')
        if len({b['filter_seed_count'] for b in case['budgets']}) != 1:
            raise ValueError('Particle sensitivity requires the same seed count')
        for budget in case['budgets']:
            if any(type(v) is not int or v < 2 for v in budget.values()):
                raise ValueError('Particle and seed budgets must be integers >= 2')
    if smoke:
        protocol['cases'] = deepcopy(cases[:1])
        protocol['cases'][0]['budgets'] = [{'particle_count': 2, 'filter_seed_count': 2}]
        protocol['bootstrap_replicates'] = 100
    return config, protocol


def fixed_banks(protocol: dict) -> tuple[dict, dict]:
    archived = json.loads((Path(protocol['previous'])/'shortlists.json').read_text())
    banks, excluded = {}, {}
    for case in protocol['cases']:
        sid = case['subject']
        by_id = {r['id']: r for r in archived[str(sid)]}
        if not set(case['candidates']) <= by_id.keys():
            raise ValueError('Requested candidate absent from historical shortlist')
        rows = []
        for pid in case['candidates']:
            row = by_id[pid]
            if point_id(row['hyperparams']) != pid:
                raise ValueError('Historical candidate ID mismatch')
            # Historical numbers must never masquerade as a new audit score.
            rows.append({k: deepcopy(row[k]) for k in ('id', 'hyperparams', 'sources', 'origin')})
        banks[sid] = rows
        excluded[str(sid)] = sorted(set(by_id)-set(case['candidates']))
    return banks, excluded


def prediction_comparison(left: dict, right: dict) -> dict:
    """Descriptive trialwise total variation, using the original score mask."""
    for key in ('observed', 'mask', 'seeds'):
        np.testing.assert_array_equal(left[key], right[key])
    mask = left['mask']
    p, q = (a['probabilities'].mean(axis=0)[mask] for a in (left, right))
    tv = np.abs(p-q).sum(axis=1)/2
    halves = []
    for a in (left, right):
        data = a['probabilities'][:, mask]
        mid = len(data)//2
        halves.append(float(np.sqrt(np.mean((data[:mid].mean(0)-data[mid:].mean(0))**2))))
    return {
        'scored_trials': int(mask.sum()),
        'mean_total_variation': float(tv.mean()),
        'q95_total_variation': float(np.quantile(tv, .95)),
        'max_total_variation': float(tv.max()),
        'probability_rmse': float(np.sqrt(np.mean((p-q)**2))),
        'argmax_disagreements': int((p.argmax(1) != q.argmax(1)).sum()),
        'split_half_probability_rmse': halves,
        'mean_nll': [mixture_nll(a['probabilities'], a['observed'], mask) for a in (left, right)],
        'scope': 'Descriptive existing-data predictions, not held-out accuracy or latent-state equivalence. Split halves are one numerical sensitivity check, not a confidence interval.',
    }


def cached_check(protocol: dict) -> tuple[dict, dict]:
    root = Path(protocol['previous'])
    settings = protocol['cached_prediction_check']
    historical = json.loads((root/'manifest.json').read_text())
    receipt_path = root/settings['receipt']
    receipt = json.loads(receipt_path.read_text())
    rows = json.loads((root/'shortlists.json').read_text())[str(settings['subject'])]
    rows = [next(r for r in rows if r['id'] == pid) for pid in settings['candidates']]
    scorer = Scorer(root, historical['config'], {})
    arrays, hashes = [], {str(receipt_path): digest(receipt_path)}
    for row in rows:
        seeds = phase_seeds(historical['config']['base_seed'], settings['subject'], settings['family'], settings['budget']['filter_seed_count'])
        for seed in seeds:
            path = cache_path(root, settings['subject'], row['id'], settings['budget']['particle_count'], seed)
            actual = digest(path)
            if actual != receipt['cache_sha256'][str(path.relative_to(root))]:
                raise ValueError('Archived prediction cache checksum differs')
            hashes[str(path)] = actual
        arrays.append(scorer.arrays(settings['subject'], row, settings['budget'], settings['family']))
    result = prediction_comparison(*arrays)
    result.update({'subject': settings['subject'], 'candidate_ids': settings['candidates'],
                   'parameters': [extract_model_0826_parameters(r['hyperparams']) for r in rows],
                   'settings': settings})
    return result, hashes


def execute(scorer: Scorer, protocol: dict, banks: dict, output: Path) -> dict:
    results = {}
    for case in protocol['cases']:
        sid, nominee = case['subject'], case['nominee']
        looks = []
        for index, budget in enumerate(case['budgets']):
            rows = scorer.batch({sid: banks[sid]}, budget, FAMILY, f'{sid}/R{budget["particle_count"]}')[sid]
            arrays = {r['id']: scorer.arrays(sid, r, budget, FAMILY) for r in rows}
            diagnostic = decision_diagnostics(
                arrays, nominee, protocol['tolerance'], protocol['alpha'],
                protocol['bootstrap_replicates'], stable_seed({'base': protocol['base_seed'], 'subject': sid, 'budget': budget}))
            item = {'budget': budget, 'diagnostic': diagnostic, 'rows': rows,
                    'primary': index == len(case['budgets'])-1,
                    'trials': len(arrays[nominee]['mask']), 'scored_trials': int(arrays[nominee]['mask'].sum())}
            freeze_json(output/'subjects'/str(sid)/f'R{budget["particle_count"]}.json', item)
            looks.append(item)
        results[str(sid)] = {'nominee': nominee, 'looks': looks,
                             'pair_status': looks[-1]['diagnostic']['status'],
                             'whole_bank_status': 'not_reassessed', 'parameter_identification': 'not_assessed'}
    return {'subjects': results, 'scope': 'Fixed-pair fresh-seed audit. Only final budget is primary. No whole-bank/global stopping, parameter interval, behavioral holdout or latent-state precision claim.'}


def run(path: Path, output: Path | None, *, smoke: bool = False, resume: bool = False, dry_run: bool = False) -> dict:
    path = path.resolve()
    config, protocol = load_protocol(path, smoke)
    previous = Path(protocol['previous'])
    historical = json.loads((previous/'manifest.json').read_text())
    verify_manifest(historical)
    subjects = select_subjects(config, [c['subject'] for c in protocol['cases']], None, smoke)
    contexts = {s['subject']: make_context(s, config, smoke) for s in subjects}
    banks, excluded = fixed_banks(protocol)
    # Explicitly reject collisions with any previously used seed family.
    for case in protocol['cases']:
        fresh = set(phase_seeds(config['base_seed'], case['subject'], FAMILY, max(b['filter_seed_count'] for b in case['budgets'])))
        old = set().union(*(phase_seeds(historical['config']['base_seed'], case['subject'], family, 64)
                            for family in ('boundary_screen', 'boundary_selection', 'boundary_audit')))
        if fresh & old:
            raise ValueError('Fresh audit shares historical seeds')
    plan = {'protocol': protocol, 'excluded_historical_candidates': excluded,
            'parallel_budget': config['parallel_budget'], 'smoke': smoke,
            'max_new_pf_runs': sum(len(c['candidates'])*sum(b['filter_seed_count'] for b in c['budgets']) for c in protocol['cases']),
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
        prediction, hashes = cached_check(protocol)
        manifest = run_manifest(config, path, subjects, smoke)
        manifest['scope'] = 'Bounded fixed-pair precision follow-up, plus read-only archived S129 prediction comparison.'
        manifest['input_sha256'].update(hashes)
        for source in (Path(protocol['fit_config']), previous/'manifest.json', previous/'shortlists.json'):
            manifest['input_sha256'][str(source)] = digest(source)
        freeze_json(output/'manifest.json', manifest)
        freeze_json(output/'plan.json', plan)
        freeze_json(output/'frozen_banks.json', {str(s): rows for s, rows in banks.items()})
        freeze_json(output/'nominations.json', {str(c['subject']): c['nominee'] for c in protocol['cases']})
        freeze_json(output/'S129_prediction_comparison.json', prediction)
        result = execute(Scorer(output, config, contexts), protocol, banks, output)
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
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, smoke=args.smoke, resume=args.resume, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
