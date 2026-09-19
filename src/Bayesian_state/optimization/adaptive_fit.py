"""Default bounded PMH fit controller for observed exp123 Model 0826 data.

Software stopping flags describe only tested search/finite-particle precision.
They never certify recovery, latent-state precision or a global optimum.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .adaptive_boundary import adaptive_support, boundary_report
from .adaptive_runtime import Scorer, freeze_json, run_manifest, verify_manifest
from .diagnostics.decision_precision import decision_diagnostics
from .model_0826 import WORKSPACE_PROFILE_KEY, build_model_0826_cell_engine, extract_model_0826_parameters
from .parameter_space import load_model_parameter_space
from .search.adaptive_proposals import (
    candidate, initial_points, propose_round, ranking,
    ray_proposals, select_elites,
)
from ..simulation.data import SubjectTrialDataLoader
from ..utils.paths import ROOT_DIR
from ..utils.seeding import stable_seed


DEFAULT_CONFIG = ROOT_DIR/'configs/exp123/specific_models/model_0826_adaptive_fit.yaml'


def load_fit_config(path: Path, smoke: bool = False) -> dict:
    config = yaml.safe_load(path.read_text())
    required = {'schema_version', 'backend', 'analysis_id', 'parameter_space', 'processed_dir',
                'conditions', 'base_seed', 'parallel_budget', 'search', 'precision', 'boundary'}
    if not isinstance(config, dict) or set(config) != required or config['schema_version'] != 1 or config['backend'] != 'model0826_adaptive':
        raise ValueError('Expected Model 0826 adaptive schema version 1; unknown keys are not ignored')
    for key in ('parameter_space', 'processed_dir'):
        config[key] = str((path.parent/config[key]).resolve())
    if not config['conditions'] or any(type(c) is not int or c not in (1, 2, 3) for c in config['conditions']):
        raise ValueError('Only exp123 conditions 1, 2, 3 are supported')
    config['conditions'] = {str(c): str((path.parent/v).resolve()) for c, v in config['conditions'].items()}
    for name in ('base_seed', 'parallel_budget'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    search = config['search']; precision = config['precision']; boundary = config['boundary']
    search_keys = {'discovery', 'guide', 'initial_count', 'max_rounds', 'max_cycles', 'elite_count',
                   'proposals_per_elite', 'fresh_global_count', 'local_fraction', 'jump_fraction',
                   'guide_top', 'guide_diverse', 'guide_random', 'patience', 'min_improvement',
                   'challenge_rounds', 'challenge_global_count', 'audit_top_per_source'}
    if set(search) != search_keys or set(precision) != {'tiers', 'audit', 'alpha', 'tolerance', 'bootstrap_replicates'} or set(boundary) != {'extensions', 'near_tolerance', 'max_candidates', 'proposals_per_elite'}:
        raise ValueError('Unknown or missing search/precision/boundary keys')
    for key in search_keys - {'discovery', 'guide', 'local_fraction', 'jump_fraction', 'min_improvement'}:
        if type(search[key]) is not int or search[key] < 1:
            raise ValueError(f'search.{key} must be a positive integer')
    if not 0 <= search['local_fraction'] < 1 or not 0 <= search['jump_fraction'] < 1 or search['local_fraction']+search['jump_fraction'] >= 1:
        raise ValueError('Proposal quotas must leave a positive share for joint moves')
    if not np.isfinite(search['min_improvement']) or search['min_improvement'] < 0:
        raise ValueError('Invalid minimum improvement')
    if not 0 < precision['alpha'] < .5 or not np.isfinite(precision['tolerance']) or precision['tolerance'] <= 0:
        raise ValueError('Invalid numerical tolerance/alpha')
    if type(precision['bootstrap_replicates']) is not int or precision['bootstrap_replicates'] < 100:
        raise ValueError('Need at least 100 bootstrap draws')
    if not isinstance(precision['tiers'], list) or not precision['tiers']:
        raise ValueError('Need numerical precision tiers')
    budgets = [search['discovery'], search['guide'], *precision['tiers'], precision['audit']]
    for budget in budgets:
        if set(budget) != {'particle_count', 'filter_seed_count'} or any(type(v) is not int or v < 2 for v in budget.values()):
            raise ValueError('Each numerical budget needs integer particle/seed counts >= 2')
    ordered = [search['guide'], *precision['tiers'], precision['audit']]
    if any(any(b[k] < a[k] for k in a) for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Precision tiers and audit must not decrease R or B below guide')
    if not isinstance(boundary['extensions'], dict) or any(type(boundary[k]) is not int or boundary[k] < 1 for k in ('max_candidates', 'proposals_per_elite')):
        raise ValueError('Invalid boundary policy')
    if not np.isfinite(boundary['near_tolerance']) or boundary['near_tolerance'] < precision['tolerance']:
        raise ValueError('Boundary near-candidate tolerance must cover selection tolerance')
    if smoke:
        config['parallel_budget'] = 1
        for budget in budgets:
            budget.update(particle_count=2, filter_seed_count=2)
        search.update(initial_count=9, max_rounds=2, max_cycles=1, elite_count=1,
                      proposals_per_elite=2, fresh_global_count=9, guide_top=1,
                      guide_diverse=1, guide_random=1, challenge_rounds=1,
                      challenge_global_count=9, audit_top_per_source=2)
        precision['tiers'] = [precision['tiers'][0]]
        precision['bootstrap_replicates'] = 100
        boundary.update(max_candidates=1, proposals_per_elite=2)
    return config


def select_subjects(config: dict, subjects: list[int] | None, conditions: list[int] | None, smoke: bool) -> list[dict]:
    frame = pd.read_csv(Path(config['processed_dir'])/'Task2_processed.csv', usecols=['iSub', 'condition'])
    if frame.isna().any().any() or any(not np.equal(frame[k], np.floor(frame[k])).all() for k in frame):
        raise ValueError('Subject and condition IDs must be finite integers')
    pairs = frame.drop_duplicates().sort_values('iSub')
    if pairs['iSub'].duplicated().any():
        raise ValueError('Subject ID belongs to more than one condition')
    allowed = set(conditions) if conditions is not None else {int(k) for k in config['conditions']}
    if not allowed or not allowed.issubset({int(k) for k in config['conditions']}):
        raise ValueError('Requested condition is not configured')
    pairs = pairs[pairs['condition'].isin(allowed)]
    if subjects is not None:
        if len(set(subjects)) != len(subjects) or not set(subjects).issubset(set(pairs['iSub'])):
            raise ValueError('Missing, duplicated, or condition-mismatched subjects')
        pairs = pairs[pairs['iSub'].isin(subjects)]
    if pairs.empty:
        raise ValueError('No subjects selected')
    if smoke:
        if subjects is not None and len(subjects) != 1:
            raise ValueError('Smoke requires exactly one subject')
        pairs = pairs.iloc[:1]
    return [{'subject': int(sid), 'condition': int(cond), 'engine': config['conditions'][str(int(cond))]}
            for sid, cond in pairs[['iSub', 'condition']].itertuples(index=False, name=None)]


def make_context(spec: dict, config: dict, smoke: bool) -> dict:
    engine = yaml.safe_load(Path(spec['engine']).read_text())
    if engine.get('provenance', {}).get('model_id') != 'model_0826' or engine.get('recovery', {}).get('architecture_cell', 'PMH') != 'PMH':
        raise ValueError('Adaptive default fits only declared Model 0826 PMH engines')
    validated = build_model_0826_cell_engine(engine, 'PMH')
    original = deepcopy(engine)
    validated.pop('recovery', None); original.pop('recovery', None)
    if validated != original:
        raise ValueError('Engine is not already PMH; adaptive fitting does not change architecture')
    if int(engine['provenance'].get('condition', 1)) != spec['condition'] or engine['inference']['backend'] != 'particle_filter':
        raise ValueError('Engine condition/backend mismatch')
    loader = SubjectTrialDataLoader(engine, config['processed_dir'])
    frame = loader._get_subject_frame(spec['subject'], 1.)
    if set(frame['condition']) != {spec['condition']}:
        raise ValueError('Data condition differs from engine')
    arrays = loader._extract_arrays(frame, 32 if smoke else None)
    if len(arrays.choices) < 2:
        raise ValueError('At least two trials required for default scoring')
    return {**spec, 'engine': engine, 'arrays': arrays, 'processed_dir': Path(config['processed_dir'])}


def proposal_seed(config: dict, subject: int, role: str) -> int:
    return stable_seed({'base': config['base_seed'], 'subject': subject, 'proposal_role': role})


def merge_proposals(groups: list[tuple[str, list[dict]]]) -> list[dict]:
    """Deduplicate while retaining how and from where each point was proposed."""
    bank = {}
    for source, rows in groups:
        for row in rows:
            item = bank.setdefault(row['id'], deepcopy(row))
            for label in [*row.get('sources', []), source]:
                if label not in item['sources']:
                    item['sources'].append(label)
            if not item.get('origin') and row.get('origin'):
                item['origin'] = deepcopy(row['origin'])
    return list(bank.values())


def shortlist(rows: list[dict], config: dict, seed: int) -> list[dict]:
    search = config['search']
    picked = ranking(rows)[:search['guide_top']] + select_elites(rows, search['guide_diverse'])
    used = {r['id'] for r in picked}
    remaining = [r for r in rows if r['id'] not in used]
    rng = np.random.default_rng(seed)
    picked += [remaining[i] for i in rng.permutation(len(remaining))[:search['guide_random']]]
    return merge_proposals([('guide', picked)])


def stop_status(plateau: bool, challenge_improved: bool, decision: str, audit: str, boundary: bool) -> tuple[str, list[str]]:
    issues = []
    if not plateau:
        issues.append('search_budget_without_plateau')
    if challenge_improved:
        issues.append('challenge_improved')
    if decision != 'acceptable_within_bank':
        issues.append('selection_precision_unresolved')
    if audit != 'acceptable_within_bank':
        issues.append('independent_audit_unresolved')
    if boundary:
        issues.append('boundary_review_required')
    return ('unresolved' if issues else 'provisional_stop_within_tested_scope'), issues


def fit_subjects(config: dict, contexts: dict[int, dict], space: dict, anchor: dict,
                 support: dict, root: Path, smoke: bool) -> dict:
    scorer = Scorer(root, config, contexts)
    search, precision = config['search'], config['precision']
    states = {sid: {'guided': {}, 'seen': set(), 'trace': [], 'rejected': set()} for sid in contexts}
    active = list(contexts)
    results: dict[int, dict] = {}

    def evaluate(banks: dict[int, list[dict]], label: str, *, direct: bool = False) -> dict[int, list[dict]]:
        low = scorer.batch(banks, search['discovery'], 'discovery', label+'/discovery')
        guides = {sid: (merge_proposals([('mandatory_boundary', rows)]) if direct else
                       shortlist(rows, config, proposal_seed(config, sid, label))) for sid, rows in low.items()}
        high = scorer.batch(guides, search['guide'], 'guide', label+'/guide')
        for sid, rows in low.items():
            states[sid]['seen'].update(row['id'] for row in rows)
        for sid, rows in high.items():
            states[sid]['guided'].update({row['id']: row for row in rows})
        return high

    for cycle in range(search['max_cycles']):
        if not active:
            break
        cycle_name = f'cycle_{cycle}'
        stagnant = {sid: 0 for sid in active}
        plateau = {sid: False for sid in active}
        if cycle == 0:
            banks = {sid: [candidate(p, 'initial') for p in initial_points(
                space, anchor, search['initial_count'], proposal_seed(config, sid, 'initial'))] for sid in active}
            evaluate(banks, cycle_name+'/initial')
        previous = {sid: ranking(list(states[sid]['guided'].values()))[0]['mean_nll'] for sid in active}
        working = list(active)
        for iteration in range(search['max_rounds']):
            label = f'{cycle_name}/round_{iteration}'
            banks = {}
            for sid in working:
                state = states[sid]
                elites = select_elites(list(state['guided'].values()), search['elite_count'])
                if state.get('preferred'):
                    preferred = state['guided'][state['preferred']]
                    elites = [preferred] + [r for r in elites if r['id'] != preferred['id']][:search['elite_count']-1]
                proposals = propose_round(elites, space, state['seen'], search['proposals_per_elite'],
                                          proposal_seed(config, sid, label), search['local_fraction'], search['jump_fraction'])
                proposals += [candidate(p, 'fresh_global') for p in initial_points(
                    space, anchor, search['fresh_global_count'], proposal_seed(config, sid, label+'/global'))]
                banks[sid] = [r for r in merge_proposals([('search', proposals)]) if r['id'] not in state['seen']]
            evaluate(banks, label)
            for sid in working:
                current = ranking(list(states[sid]['guided'].values()))[0]['mean_nll']
                gain = previous[sid]-current
                stagnant[sid] = stagnant[sid]+1 if gain <= search['min_improvement'] else 0
                plateau[sid] = stagnant[sid] >= search['patience']
                previous[sid] = current
                states[sid]['trace'].append({'cycle': cycle, 'round': iteration, 'best_guide_nll': current,
                                             'gain': gain, 'stagnant_rounds': stagnant[sid], 'plateau': plateau[sid]})
            freeze_json(root/'progress'/f'{cycle_name}_round_{iteration}.json',
                        {str(sid): states[sid]['trace'][-1] for sid in working})
            working = [sid for sid in working if not plateau[sid]]
            if not working:
                break
        base = {sid: ranking(list(states[sid]['guided'].values()))[:search['audit_top_per_source']] for sid in active}
        freeze_json(root/'progress'/f'{cycle_name}_before_challenge.json', {str(sid): rows for sid, rows in base.items()})
        challenges = {sid: [] for sid in active}
        for iteration in range(search['challenge_rounds']):
            label = f'{cycle_name}/challenge_{iteration}'
            banks = {}
            for sid in active:
                state = states[sid]
                proposals = propose_round(select_elites(list(state['guided'].values()), search['elite_count']),
                    space, state['seen'], search['proposals_per_elite'], proposal_seed(config, sid, label), .25, .25)
                if iteration == 0:
                    proposals += [candidate(p, 'challenge_global') for p in initial_points(
                        space, anchor, search['challenge_global_count'], proposal_seed(config, sid, label+'/global'))]
                banks[sid] = [r for r in merge_proposals([('challenge', proposals)]) if r['id'] not in state['seen']]
            high = evaluate(banks, label)
            for sid, rows in high.items():
                challenges[sid] += rows
        # Boundary neighborhoods are mandatory guide evaluations, including
        # cross-block moves; no cheap screening discards these diagnostics.
        boundary_banks = {}
        for sid in active:
            state = states[sid]; rows = list(state['guided'].values())
            report = boundary_report(rows, support, config['boundary']['near_tolerance'], config['boundary']['max_candidates'])
            ids = {h['candidate'] for h in report['hits'] if h['kind'] == 'artificial_boundary'}
            elites = [r for r in rows if r['id'] in ids]
            count = config['boundary']['proposals_per_elite']
            proposals = propose_round(elites, space, state['seen'], count, proposal_seed(config, sid, cycle_name+'/boundary'), .25, .25)
            proposals += ray_proposals(elites, space, state['seen'] | {r['id'] for r in proposals},
                                       count, proposal_seed(config, sid, cycle_name+'/boundary_rays'), 'boundary')
            boundary_banks[sid] = merge_proposals([('boundary', proposals)])
            freeze_json(root/'progress'/f'{cycle_name}_boundary_{sid}.json', report)
        high = evaluate(boundary_banks, cycle_name+'/boundary', direct=True)
        for sid, rows in high.items():
            challenges[sid] += rows
        improved = {sid: base[sid][0]['mean_nll']-ranking(list(states[sid]['guided'].values()))[0]['mean_nll'] > search['min_improvement'] for sid in active}
        next_cycle = [sid for sid in active if improved[sid] and cycle+1 < search['max_cycles']]
        testing = [sid for sid in active if sid not in next_cycle]
        banks, primary = {}, {}
        for sid in testing:
            state = states[sid]
            eligible = [r for r in ranking(list(state['guided'].values())) if r['id'] not in state['rejected']]
            chosen = eligible[0] if eligible else ranking(list(state['guided'].values()))[0]
            if state.get('preferred') and not improved[sid]:
                chosen = state['guided'][state['preferred']]
            primary[sid] = chosen['id']
            banks[sid] = merge_proposals([('base', base[sid]), ('challenge', ranking(challenges[sid])[:search['audit_top_per_source']]), ('selected', [chosen])])
        if not testing:
            active = next_cycle
            continue
        freeze_json(root/'progress'/f'{cycle_name}_frozen_selection.json',
                    {'primary': {str(sid): pid for sid, pid in primary.items()}, 'banks': {str(sid): rows for sid, rows in banks.items()}})
        decisions, tier_history = {}, {sid: [] for sid in testing}
        unresolved = list(testing)
        for tier, budget in enumerate(precision['tiers']):
            family = f'{cycle_name}/decision'
            scored = scorer.batch({sid: banks[sid] for sid in unresolved}, budget, family, f'{cycle_name}/decision_{tier}')
            for sid in unresolved:
                arrays = {row['id']: scorer.arrays(sid, row, budget, family) for row in scored[sid]}
                diagnostic = decision_diagnostics(arrays, primary[sid], precision['tolerance'],
                    precision['alpha']/(len(precision['tiers'])*search['max_cycles']), precision['bootstrap_replicates'],
                    proposal_seed(config, sid, f'{cycle_name}/bootstrap_{tier}'))
                decisions[sid] = diagnostic
                tier_history[sid].append({'budget': budget, 'diagnostic': diagnostic})
            unresolved = [sid for sid in unresolved if decisions[sid]['status'] == 'unresolved']
            if not unresolved:
                break
        audit_family = f'{cycle_name}/independent_audit'
        audits = scorer.batch(banks, precision['audit'], audit_family, cycle_name+'/audit')
        for sid in testing:
            arrays = {row['id']: scorer.arrays(sid, row, precision['audit'], audit_family) for row in audits[sid]}
            audit = decision_diagnostics(arrays, primary[sid], precision['tolerance'],
                precision['alpha']/search['max_cycles'], precision['bootstrap_replicates'],
                proposal_seed(config, sid, cycle_name+'/audit_bootstrap'))
            near_rows = [r for r in audits[sid] if r['id'] == primary[sid] or r['mean_nll'] <= min(audit['scores'].values()) + config['boundary']['near_tolerance']]
            # Include the nominated representative even if numerical inspection
            # later shows it inferior; boundaries must not disappear with rank.
            edge = boundary_report(near_rows, support, float('inf'), len(near_rows))
            status, issues = stop_status(plateau[sid], improved[sid], decisions[sid]['status'], audit['status'], edge['review_required'])
            selected = next(r for r in banks[sid] if r['id'] == primary[sid])
            result = {'subject': sid, 'condition': contexts[sid]['condition'], 'status': status,
                      'issues': issues, 'smoke_only': smoke, 'selected': primary[sid],
                      'hyperparams': selected['hyperparams'], 'parameters': extract_model_0826_parameters(selected['hyperparams']),
                      'candidate_bank': banks[sid], 'near_candidates': [r['id'] for r in near_rows],
                      'selected_mean_nll': audit['scores'][primary[sid]], 'tiers': tier_history[sid],
                      'independent_audit': audit, 'boundary': edge, 'search_trace': states[sid]['trace'],
                      'audit_family': audit_family, 'audit_budget': precision['audit'],
                      'trial_count': len(contexts[sid]['arrays'].choices),
                      'scored_trial_count': int(next(iter(arrays.values()))['mask'].sum()),
                      'state_precision': 'not_checked', 'parameter_recovery': 'not_checked'}
            freeze_json(root/'subjects'/str(sid)/f'cycle_{cycle}.json', result)
            inferior = audit['status'] == 'selected_point_inferior' or decisions[sid]['status'] == 'selected_point_inferior'
            if inferior and cycle+1 < search['max_cycles']:
                diagnostic = audit if audit['status'] == 'selected_point_inferior' else decisions[sid]
                states[sid]['preferred'] = min(diagnostic['scores'], key=diagnostic['scores'].get)
                states[sid]['rejected'].add(primary[sid])
                next_cycle.append(sid)
            elif not plateau[sid] and cycle+1 < search['max_cycles']:
                next_cycle.append(sid)
            else:
                results[sid] = result
                freeze_json(root/'subjects'/str(sid)/'fit_result.json', result)
        active = next_cycle
    return {'schema_version': 1, 'backend': 'model0826_adaptive', 'smoke_only': smoke,
            'subjects': {str(sid): result for sid, result in results.items()},
            'scope': 'Representative parameters and choice-scoring diagnostics; latent-state output precision, recovery and global optimality unverified.'}


def run_fit(config_path: Path, output: Path | None, subjects: list[int] | None = None,
            conditions: list[int] | None = None, *, smoke: bool = False,
            resume: bool = False, dry_run: bool = False) -> dict:
    config_path = config_path.resolve()
    config = load_fit_config(config_path, smoke)
    specs = select_subjects(config, subjects, conditions, smoke)
    original = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    space, anchor, support = adaptive_support(original, config['boundary']['extensions'])
    cell_count = len(space[WORKSPACE_PROFILE_KEY])
    if any(config['search'][key] < cell_count for key in ('initial_count', 'fresh_global_count', 'challenge_global_count')):
        raise ValueError('Every global proposal batch must cover declared workspace cells')
    contexts = {spec['subject']: make_context(spec, config, smoke) for spec in specs}
    plan = {'config': config, 'subjects': [{**spec, 'trials': len(contexts[spec['subject']]['arrays'].choices)} for spec in specs],
            'workspace_cells': cell_count, 'scope': 'exp123 PMH observed-data choice fitting; default support remains provisional'}
    if dry_run:
        return plan
    if output is None:
        raise ValueError('A new --output-dir is required (or explicitly --resume an existing run)')
    output = output.resolve()
    if resume:
        if not (output/'manifest.json').is_file():
            raise ValueError('Resume requires a complete run manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = run_manifest(config, config_path, specs, smoke)
        freeze_json(output/'manifest.json', manifest)
        freeze_json(output/'plan.json', plan)
        freeze_json(output/'effective_parameter_support.json', support)
        result = fit_subjects(config, contexts, space, anchor, support, output, smoke)
        verify_manifest(manifest)
        freeze_json(output/'fit_results.json', result)
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Default Model 0826 adaptive PMH fit (exp123)', allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--conditions', type=int, nargs='+', choices=(1, 2, 3))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    result = run_fit(args.config, args.output_dir, args.subjects, args.conditions,
                     smoke=args.smoke, resume=args.resume, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({sid: {'status': row['status'], 'issues': row['issues']}
                          for sid, row in result['subjects'].items()}, indent=2))


if __name__ == '__main__':
    main()
