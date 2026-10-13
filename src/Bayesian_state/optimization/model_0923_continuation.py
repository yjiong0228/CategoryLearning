"""One bounded outer-support check and a fresh, fixed-bank numerical audit.

The saved fit stays immutable. All old finalists and declared boundary probes
receive common fresh screening seeds. Screening freezes the nominee and bank;
one independent audit then assesses that nominee without selecting it again.
Both accepted and unresolved outcomes can feed the subject evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import shutil
from typing import Any

import yaml

from .adaptive_runtime import Scorer, digest, freeze_json, phase_seeds, run_manifest, verify_manifest
from .diagnostics.decision_precision import decision_diagnostics
from .model_0923_adaptive import Model0923Policy, make_contexts, parameter_support
from .search.adaptive_proposals import candidate, point_id
from ..evaluation.model_0923_fit_source import validate_fit_source
from ..model import ModelContext, StateModel
from ..utils.seeding import stable_seed


PROTOCOL = 'model0923_complete_pipeline_v1'
SCREEN_FAMILY = 'bounded_continuation_v1/screen'
AUDIT_FAMILY = 'bounded_continuation_v1/independent_audit'


def load_policy(path: Path, smoke: bool = False) -> dict[str, Any]:
    policy = yaml.safe_load(path.read_text())
    keys = {'protocol', 'base_seed', 'parallel_budget', 'screen', 'audit', 'alpha',
            'tolerance', 'bootstrap_replicates', 'max_new_candidates',
            'audit_new_candidates', 'evaluation_attempts', 'extensions', 'jobs'}
    if not isinstance(policy, dict) or set(policy) != keys or policy['protocol'] != PROTOCOL:
        raise ValueError('Invalid complete-pipeline policy')
    for budget in (policy['screen'], policy['audit']):
        if set(budget) != {'particle_count', 'filter_seed_count'} or any(type(v) is not int or v < 2 for v in budget.values()):
            raise ValueError('Need explicit particle and seed budgets >= 2')
    for name in ('parallel_budget', 'bootstrap_replicates', 'max_new_candidates', 'audit_new_candidates', 'evaluation_attempts'):
        if type(policy[name]) is not int or policy[name] < 1:
            raise ValueError(f'Invalid continuation limit: {name}')
    if type(policy['base_seed']) is not int or policy['base_seed'] < 0 or not 0 < policy['alpha'] < 1 or not math.isfinite(policy['tolerance']) or policy['tolerance'] <= 0:
        raise ValueError('Invalid seed or numerical audit settings')
    if policy['bootstrap_replicates'] < 100:
        raise ValueError('Need at least 100 numerical bootstrap draws')
    if not isinstance(policy['extensions'], dict) or any(not isinstance(v, dict) or not set(v) <= {'lower', 'upper'} for v in policy['extensions'].values()):
        raise ValueError('Extensions must declare lower/upper edge values')
    jobs = policy['jobs']
    if not jobs or len({j['subject'] for j in jobs}) != len(jobs):
        raise ValueError('Need distinct continuation subjects')
    for job in jobs:
        if set(job) != {'subject', 'evaluation_config'} or type(job['subject']) is not int:
            raise ValueError('Each job needs a subject and evaluation configuration')
        job['evaluation_config'] = str((path.parent / job['evaluation_config']).resolve())
    if smoke:
        policy.update(parallel_budget=1, bootstrap_replicates=100)
        policy['screen'] = dict(particle_count=2, filter_seed_count=2)
        policy['audit'] = dict(particle_count=4, filter_seed_count=2)
    return policy


def boundary_candidates(result: dict, support: dict, policy: dict) -> tuple[list[dict], dict]:
    """Preserve every finalist; add single-coordinate probes at artificial edges.

    All proposals are fixed before any new scoring. Each affected finalist is
    checked, including near-equivalent finalists with an interior nominee.
    Natural closed-domain endpoints retain their original meaning.
    """
    originals = {r['id']: candidate(r['hyperparams'], 'previous_finalist',
                                  previous_candidate=r['id']) for r in result['candidate_bank']}
    if any(point_id(r['hyperparams']) != r['id'] for r in result['candidate_bank']):
        raise ValueError('Saved candidate identity differs')
    proposals, additions = dict(originals), {}
    for hit in result['boundary']['hits']:
        if hit['kind'] != 'artificial_boundary':
            continue
        name, side = hit['parameter'], hit['side']
        values = policy['extensions'].get(name, {}).get(side, [])
        if not values:
            continue
        spec = support['parameters'][name]
        edge = min(spec['values']) if side == 'lower' else max(spec['values'])
        if hit['value'] != edge:
            raise ValueError('Saved boundary hit differs from its frozen support')
        for value in values:
            if not (value < edge if side == 'lower' else value > edge):
                raise ValueError('Continuation values must extend the indicated edge')
            point = deepcopy(originals[hit['candidate']]['hyperparams'])
            point[spec['path']] = value
            row = candidate(point, 'outer_boundary', anchor=hit['candidate'],
                            parameter=name, side=side, old_edge=edge)
            proposals.setdefault(row['id'], row)
            additions.setdefault(name, set()).add(value)
    if len(proposals) - len(originals) > policy['max_new_candidates']:
        raise ValueError('Declared boundary probes exceed the fixed candidate budget')
    return list(proposals.values()), {name: sorted(values) for name, values in additions.items()}


def freeze_nomination(rows: list[dict], original_ids: set[str], limit: int) -> tuple[str, list[dict]]:
    """Nominate on screening alone, retaining every historical finalist in audit."""
    ranked = sorted(rows, key=lambda r: (r['mean_nll'], r['id']))
    nominee = ranked[0]['id']
    outer = [r for r in ranked if r['id'] not in original_ids][:limit]
    bank = {r['id']: r for r in rows if r['id'] in original_ids}
    bank.update({r['id']: r for r in outer})
    if nominee not in bank:
        raise ValueError('Screening nominee must be in the fixed independent audit bank')
    return nominee, sorted(bank.values(), key=lambda r: r['id'])


def seed_overlap_check(source: Path, config: dict, subject: int, policy: dict) -> None:
    """Reject overlap with any actual saved seed, including prior continuations."""
    old = set()
    for path in (source / 'cache' / str(subject)).glob('R*_seed*'):
        old.add(int(path.name.split('_seed')[1]))
    fresh = []
    for family, budget in ((SCREEN_FAMILY, policy['screen']), (AUDIT_FAMILY, policy['audit'])):
        fresh.append(set(phase_seeds(config['base_seed'], subject, family, budget['filter_seed_count'],
                                    namespace=config['seed_namespace'])))
    if fresh[0] & fresh[1] or old & (fresh[0] | fresh[1]):
        raise ValueError('Continuation screening or audit seed family overlaps earlier scoring')


def execute(scorer: Scorer, config: dict, policy: dict, original: dict,
            rows: list[dict], support: dict, output: Path) -> dict:
    sid = original['subject']
    screened = scorer.batch({sid: rows}, policy['screen'], SCREEN_FAMILY, 'continuation/screen')[sid]
    nominee, bank = freeze_nomination(screened, {r['id'] for r in original['candidate_bank']},
                                     policy['audit_new_candidates'])
    freeze_json(output / 'nomination.json', dict(selected=nominee, bank=bank,
                                               role='Frozen before independent audit'))
    audited = scorer.batch({sid: bank}, policy['audit'], AUDIT_FAMILY, 'continuation/audit')[sid]
    arrays = {r['id']: scorer.arrays(sid, r, policy['audit'], AUDIT_FAMILY) for r in audited}
    # One primary look per subject in this separately declared continuation.
    # The old alpha spending and old acceptance records remain unchanged.
    diagnostic = decision_diagnostics(arrays, nominee, policy['tolerance'],
        policy['alpha'] / len(policy['jobs']), policy['bootstrap_replicates'],
        stable_seed(dict(base=config['base_seed'], subject=sid, role='continuation_bootstrap')))
    ranked = [{**r, 'mean_nll': diagnostic['scores'][r['id']]} for r in audited]
    boundary = Model0923Policy().boundary_report(ranked, support, policy['tolerance'], len(ranked))
    issues = [i for i in original['issues'] if i not in {'boundary_review_required', 'independent_audit_unresolved'}]
    if diagnostic['status'] != 'acceptable_within_bank':
        issues.append('independent_audit_unresolved')
    if boundary['review_required']:
        issues.append('boundary_review_required')
    result = deepcopy(original)
    point = next(r['hyperparams'] for r in bank if r['id'] == nominee)
    result.update(status='unresolved' if issues else 'provisional_stop_within_tested_scope',
        issues=issues, selected=nominee, hyperparams=point,
        parameters=Model0923Policy().extract_parameters(point), candidate_bank=bank,
        near_candidates=[r['id'] for r in ranked if r['mean_nll'] <= min(x['mean_nll'] for x in ranked) + policy['tolerance']],
        selected_mean_nll=diagnostic['scores'][nominee], independent_audit=diagnostic,
        audit_family=AUDIT_FAMILY, audit_budget=policy['audit'], boundary=boundary,
        tiers=[dict(role='continuation_independent_audit', budget=policy['audit'], diagnostic=diagnostic)],
        precision_policy='one_bounded_continuation_audit',
        trial_count=len(arrays[nominee]['mask']), scored_trial_count=int(arrays[nominee]['mask'].sum()),
        continuation=dict(previous_selected=original['selected'], previous_status=original['status'],
                          previous_issues=original['issues'], outer_proposals=len(rows)-len(original['candidate_bank']),
                          excluded_screened_candidates=sorted({r['id'] for r in rows}-{r['id'] for r in bank}),
                          audit_reselection=False, further_automatic_continuations=0,
                          scope='Single-coordinate outer probes and a fixed-bank, finite-particle numerical audit'))
    return result


def run(source: Path, policy_path: Path, output: Path, *, smoke: bool = False,
        resume: bool = False, dry_run: bool = False) -> dict:
    source, policy_path, output = source.resolve(), policy_path.resolve(), output.resolve()
    metadata = validate_fit_source(source)
    old_manifest = json.loads((source / 'manifest.json').read_text())
    if metadata['source_kind'] != 'adaptive' or len(metadata['subjects']) != 1:
        raise ValueError('Continuation needs one completed adaptive subject')
    if metadata['smoke_only'] and not smoke:
        raise ValueError('A smoke source supports smoke continuation only')
    original = json.loads((source / 'subjects' / str(metadata['subjects'][0]) / 'fit_result.json').read_text())
    if original.get('continuation'):
        raise ValueError('One continuation per source fit; further automatic expansion is disabled')
    policy = load_policy(policy_path, smoke)
    if original['subject'] not in {j['subject'] for j in policy['jobs']}:
        raise ValueError('Subject is absent from the continuation policy')
    support = json.loads((source / 'effective_parameter_support.json').read_text())
    rows, extensions = boundary_candidates(original, support, policy)
    config = deepcopy(old_manifest['config'])
    config.update(base_seed=policy['base_seed'], parallel_budget=policy['parallel_budget'])
    config['boundary']['extensions'].update({name: sorted(set(config['boundary']['extensions'].get(name, []) + values))
                                            for name, values in extensions.items()})
    _, _, extended_support = parameter_support(config)
    specs, contexts, files = make_contexts(config, None, smoke)
    if max(extended_support['parameters']['max_capacity']['values']) >= (29 if original['condition'] == 1 else 152):
        raise ValueError('Extended capacity must stay below the rule catalogue size')
    config['processed_dir'] = str(contexts[original['subject']]['processed_dir'])
    seed_overlap_check(source, config, original['subject'], policy)
    plan = dict(source_fit=str(source), subject=original['subject'], policy=policy, smoke=smoke,
                extensions=extensions, screened_candidates=len(rows),
                max_audit_candidates=len(original['candidate_bank'])+policy['audit_new_candidates'],
                max_new_pf_runs=len(rows)*policy['screen']['filter_seed_count']+
                    (len(original['candidate_bank'])+policy['audit_new_candidates'])*policy['audit']['filter_seed_count'],
                original_acceptance=original['status'], scored_trials=len(contexts[original['subject']]['arrays'].choices)-1,
                scope='One bounded continuation; report generation follows every completed acceptance state')
    if dry_run:
        return plan
    if resume:
        if not (output / 'manifest.json').is_file():
            raise ValueError('Resume needs a frozen continuation manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (source / 'resources_cache').is_dir() and not (output / 'resources_cache').exists():
            shutil.copytree(source / 'resources_cache', output / 'resources_cache')
        specs, contexts, _ = make_contexts(config, None, smoke, output / 'resources_cache')
        for context in contexts.values():
            StateModel(context['engine'], context=ModelContext(condition=context['condition'], subject_id=context['subject'],
                processed_data_dir=context['processed_dir'], dataset_paths=context['dataset_paths']))
        manifest = run_manifest(config, policy_path, specs, smoke)
        manifest['input_sha256'].update(metadata['input_sha256'])
        manifest['input_sha256'].update({str(p.resolve()): digest(p) for p in files})
        manifest.update(scope=plan['scope'], continuation_policy=policy,
                        git_commit=metadata['source_fit_commit'])
        freeze_json(output / 'manifest.json', manifest)
        freeze_json(output / 'plan.json', plan)
        freeze_json(output / 'effective_parameter_support.json', extended_support)
        freeze_json(output / 'boundary_proposals.json', rows)
        if (output / 'fit_results.json').is_file():
            verify_manifest(manifest)
            for receipt in (output / 'batches').rglob('scores.json'):
                for path, expected in json.loads(receipt.read_text())['cache_sha256'].items():
                    if digest(output / path) != expected:
                        raise ValueError('Completed continuation cache changed')
            return json.loads((output / 'fit_results.json').read_text())
        result = execute(Scorer(output, config, contexts), config, policy, original, rows, extended_support, output)
        result['smoke_only'] = smoke
        result['continuation']['source_fit'] = str(source)
        verify_manifest(manifest)
        freeze_json(output / 'subjects' / str(original['subject']) / 'fit_result.json', result)
        completed = dict(schema_version=2, backend='model0923_adaptive', smoke_only=smoke,
            subjects={str(original['subject']): result}, scope=plan['scope'])
        freeze_json(output / 'fit_results.json', completed)
        return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source-fit', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.source_fit, args.policy, args.output_dir,
                 smoke=args.smoke, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
