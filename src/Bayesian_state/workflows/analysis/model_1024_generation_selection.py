"""Select execution-model parameters using training likelihood and generation.

Use --phase train --config YAML --output NEW [--smoke], then --phase evaluate
--training TRAIN --config YAML --output NEW. Training workers deny access to
sealed_test.npz. Shared cognitive workers, generation, scores and paired guards
supply all scientific algorithms. The fixed rule has one confirmation batch.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
from copy import deepcopy
import io
import os
from pathlib import Path
import time

if __package__ in (None, ''):
    # Workers keep the archived cognitive package first. Only workflow and
    # numerical-metric namespaces need access to the newly added analyses.
    import src.Bayesian_state.workflows.analysis as analysis_package
    import src.Bayesian_state.metrics as metric_package
    repository = Path(__file__).resolve().parents[4]
    analysis_package.__path__ = [str(repository/'src/Bayesian_state/workflows/analysis'), *analysis_package.__path__]
    metric_package.__path__ = [str(repository/'src/Bayesian_state/metrics'), *metric_package.__path__]
    __package__ = 'src.Bayesian_state.workflows.analysis'

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from . import model_1023_execution_study as execution
from .model_1022_belief_summary import arrays, digest, nll, read
from .model_1022_belief_balanced_selection import choose_energy, likelihood_guard
from .model_1022_belief_energy_guard import paired_energy_guard, validate_fresh_seeds
from .model_1024_trajectory_diagnosis import manifest
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
TRAINING_GENERATOR = 'src/Bayesian_state/workflows/analysis/model_0923_h_objective_comparison.py'
write = execution.write


def sources() -> dict:
    names = [ENTRY, TRAINING_GENERATOR, 'src/Bayesian_state/metrics/autonomous_fit.py',
             'src/Bayesian_state/workflows/analysis/model_1022_belief_balanced_selection.py',
             'src/Bayesian_state/workflows/analysis/model_1022_belief_energy_guard.py']
    return {n: digest(ROOT/n) for n in names}


@contextmanager
def training_boundary():
    """Deny future-array access through Python and numpy's file opening."""
    old_builtin, old_io = builtins.open, io.open

    def wrap(original):
        def checked(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)) and Path(os.fsdecode(file)).name == 'sealed_test.npz':
                raise PermissionError('Training workers cannot open sealed_test.npz')
            return original(file, *args, **kwargs)
        return checked

    builtins.open, io.open = wrap(old_builtin), wrap(old_io)
    try:
        yield
    finally:
        builtins.open, io.open = old_builtin, old_io


def worker(file: Path) -> None:
    job = read(file)
    if job['selection_sources'] != sources():
        raise ValueError('Selection workflow changed')
    if job.get('freeze_file'):
        frozen = read(Path(job['freeze_file']))
        if (digest(Path(job['freeze_file'])) != job['freeze_sha256']
                or frozen['status'] != 'frozen_before_test' or frozen['later_choices_used']):
            raise ValueError('Evaluation requires the intact pre-test freeze')
    if job.get('training_generation'):
        original = execution.autonomous
        generator = execution.base.load('selection_shared_training_generator', TRAINING_GENERATOR)
        execution.autonomous = lambda j, workflow: generator.generation_score(j, execution.driver.io)
        try:
            with training_boundary():
                execution.worker(file)
        finally:
            execution.autonomous = original
    elif job['phase'] == 'training':
        with training_boundary():
            execution.worker(file)
    else:
        execution.worker(file)
    write(Path(job['target']).with_suffix('.boundary.json'), dict(
        subject=job['subject'], execution_mode=job['execution_mode'],
        training_only=bool(job.get('training_generation') or job['phase'] == 'training'),
        sealed_test_denied=bool(job.get('training_generation') or job['phase'] == 'training'),
        freeze_sha256=job.get('freeze_sha256')))


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    if not jobs:
        return
    for job in jobs:
        job['selection_sources'] = sources()
    # The dispatcher is reused unchanged; workers import execution with its
    # own original ENTRY, preserving the frozen cognitive source identity.
    previous = execution.ENTRY
    execution.ENTRY = ENTRY
    try:
        execution.execute(jobs, output, budget)
    finally:
        execution.ENTRY = previous


def settings(config: Path) -> dict:
    p = yaml.safe_load(config.read_text())
    if (p['modes'] != ['MIX', 'MAP', 'STICKY'] or p['fallback'] != 'retain_original_nll_nominee'
            or not 1 <= p['parallel_budget'] <= 128 or p['training_nll_tolerance'] < 0
            or p['mcse_multiplier'] < 0 or p['ranking_paths'] < 4 or p['confirmation_paths'] < 4):
        raise ValueError('Require the registered constrained selection policy')
    return p


def seed_bank(policy: dict, subject: int, role: str, count: int, forbidden: set[int]) -> list[int]:
    seeds = [stable_seed(dict(protocol=policy['protocol'], base=policy['seed'],
                             subject=subject, stage=role, repeat=i)) for i in range(count)]
    validate_fresh_seeds(seeds, forbidden)
    return seeds


def point_jobs(template: dict, point: dict, output: Path, role: str, seeds: list[int],
               policy: dict, *, training_generation: bool = False, phase: str = 'autonomous',
               particles: int | None = None, smoke: bool = False, freeze: Path | None = None) -> list[dict]:
    sid, mode = template['subject'], template['model']
    jobs = []
    chunk = policy['generation_chunk'] if phase == 'autonomous' else 1
    for index in range(0, len(seeds), chunk):
        name = f'{role}_S{sid}_{mode}_{point["id"]}_C{index:03d}'
        job = deepcopy(template)
        job.update(name=name, candidate=point, phase=phase, training_generation=training_generation,
            case_id=sid, features=policy['features'], seeds=seeds[index:index+chunk],
            smoke=smoke, source_sha256=execution.sources(execution.driver.io),
            target=str((output/role/name).resolve()))
        if particles is not None:
            job['particles'] = particles
        if freeze is not None:
            job.update(freeze_file=str(freeze.resolve()), freeze_sha256=digest(freeze))
        jobs.append(job)
    return jobs


def training_features(jobs: list[dict], empirical: np.ndarray) -> np.ndarray:
    parts, seeds = [], []
    for job in jobs:
        file = Path(job['target'])
        saved, meta = arrays(file), read(file.with_suffix('.json'))
        if (meta['data_boundary'] != 'training_only' or meta['candidate']['hyperparams'] != job['candidate']['hyperparams']
                or meta['subject'] != job['subject']):
            raise ValueError('Training path boundary or parameter identity changed')
        np.testing.assert_array_equal(saved['seeds'], job['seeds'])
        np.testing.assert_array_equal(saved['observed_features'], empirical)
        ncat = 2 if meta['condition'] == 1 else 4
        recalculated = behavior_features(saved['choices'], saved['categories'], n_categories=ncat, **job['features'])
        np.testing.assert_array_equal(saved['features'], recalculated)
        boundary = read(file.with_suffix('.boundary.json'))
        if not boundary['sealed_test_denied']:
            raise ValueError('Missing enforced training-only boundary')
        parts.append(recalculated)
        seeds.extend(saved['seeds'].tolist())
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated generation seeds')
    return np.concatenate(parts)


def filter_arrays(jobs: list[dict]) -> dict:
    parts = []
    for job in jobs:
        file = Path(job['target'])
        saved, meta = arrays(file), read(file.with_suffix('.json'))
        if meta['candidate']['hyperparams'] != job['candidate']['hyperparams'] or meta['subject'] != job['subject']:
            raise ValueError('Filter parameter identity changed')
        np.testing.assert_array_equal(saved['seeds'], job['seeds'])
        if job['phase'] == 'training' and not read(file.with_suffix('.boundary.json'))['sealed_test_denied']:
            raise ValueError('Missing training-only filter boundary')
        parts.append(saved)
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]
              if k not in ('choices', 'categories')}
    if len(set(result['seeds'].tolist())) != len(result['seeds']):
        raise ValueError('Repeated filter seeds')
    return result


def final_decision(old: str, proposed: str, likelihood: dict, generation: dict) -> dict:
    same = old == proposed
    accepted = bool(not same and likelihood['accepted'] and generation['energy_accepted'])
    reason = ('original_is_energy_leader' if same else 'both_guards_pass' if accepted else
              'likelihood_and_energy_guard_fail' if not likelihood['accepted'] and not generation['energy_accepted'] else
              'likelihood_guard_fail' if not likelihood['accepted'] else 'energy_guard_fail')
    return dict(original=old, proposed=proposed, selected=proposed if accepted else old,
                replaced=accepted, reason=reason, likelihood=likelihood, generation=generation)


def train(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new training output')
    p = settings(config)
    study = ROOT/p['source_study']
    frozen, done, old_protocol = [read(study/n) for n in ('nominees.json', 'completion.json', 'protocol.json')]
    if (done['status'] != 'complete' or done['smoke'] or frozen['status'] != 'frozen_before_test'
            or frozen['later_choices_used'] or frozen['oral_used']
            or done['nominees_sha256'] != digest(study/'nominees.json')
            or old_protocol['source_sha256'] != execution.sources(execution.driver.io)):
        raise ValueError('Require intact training-only estimates and the same cognitive model')
    templates = read(study/'autonomous_jobs.json')['jobs']
    output.mkdir(parents=True)
    cases, audits, inputs, ranking_jobs = [], [], {}, []
    selected = [r for r in frozen['nominees'] if r['subject'] in p['subjects'] and r['arm'] in p['modes']]
    if smoke:
        selected = selected[:1]
    started = time.monotonic()
    for r in selected:
        sid, mode = r['subject'], r['arm']
        template = next(j for j in templates if j['subject'] == sid and j['model'] == mode)
        folder = Path(template['folder'])
        meta = read(folder/'prepared.json')
        training_file = folder/'training.npz'
        if digest(training_file) != meta['array_sha256']['training.npz']:
            raise ValueError('Prepared training data changed')
        inputs[str(training_file)] = digest(training_file)
        with np.load(training_file) as z:
            n = min(8, len(z['choices'])) if smoke else len(z['choices'])
            obs, categories = z['choices'][:n], z['categories'][:n]
        empirical = behavior_features(obs[None], categories, n_categories=2 if meta['condition'] == 1 else 4,
                                      **p['features'])[0]
        old = r['candidate']
        eligible = [c for c in r['finalists'] if c['mean_nll'] <= old['mean_nll']+p['training_nll_tolerance']]
        if smoke:
            eligible = [old] + [c for c in eligible if c['id'] != old['id']][:1]
        for candidate in r['finalists']:
            f = study/'confirmation'/f'subject_{sid}'/mode/(candidate['id']+'.json')
            saved = arrays(f)
            if saved['observed_probability'].shape[1] != meta['training_trials']:
                raise ValueError('Original confirmation includes non-training choices')
            np.testing.assert_allclose(nll(saved['observed_probability']), candidate['mean_nll'], atol=1e-12, rtol=0)
            inputs[str(f)] = digest(f)
            inputs[str(f.with_suffix('.npz'))] = digest(f.with_suffix('.npz'))
            audits.append(dict(subject=sid, mode=mode, candidate=candidate['id'],
                training_nll=candidate['mean_nll'], eligible=candidate['id'] in {x['id'] for x in eligible}))
        forbidden = {s for j in templates if j['subject'] == sid for s in j['seeds']}
        seeds = seed_bank(p, sid, 'ranking', 4 if smoke else p['ranking_paths'], forbidden)
        jobs_by_point = {}
        for point in eligible:
            jobs = point_jobs(template, point, output, 'ranking', seeds, p, training_generation=True, smoke=smoke)
            jobs_by_point[point['id']] = jobs
            ranking_jobs.extend(jobs)
        cases.append(dict(subject=sid, mode=mode, template=template, original=old, candidates=eligible,
            empirical=empirical.tolist(), training_trials=n, ranking_jobs=jobs_by_point,
            forbidden=list(forbidden | set(seeds))))
    pd.DataFrame(audits).to_csv(output/'candidate_eligibility.csv', index=False)
    write(output/'protocol.json', dict(policy=p, smoke=smoke, cases=cases, inputs_sha256=inputs,
        source_study_nominees_sha256=digest(study/'nominees.json'), provenance=manifest(config),
        selection_sources=sources(), later_choices_used=False, oral_used=False,
        decision='fixed Energy minimum within training NLL tolerance, independent paired guards'))
    budget = 1 if smoke else p['parallel_budget']
    execute(ranking_jobs, output, budget)
    scores, confirmation_jobs, filter_jobs = [], [], []
    for case in cases:
        sid, mode = case['subject'], case['mode']
        own = []
        for point in case['candidates']:
            x = training_features(case['ranking_jobs'][point['id']], np.array(case['empirical']))
            score = energy_u(x, np.array(case['empirical']))
            own.append(dict(subject=sid, mode=mode, candidate=point['id'],
                confirmation_nll=point['mean_nll'], training_energy=score['value'], training_energy_mcse=score['mc_se']))
        scores.extend(own)
        chosen = choose_energy(own, p['training_nll_tolerance'])
        case['proposed'] = next(c for c in case['candidates'] if c['id'] == chosen['candidate'])
        case['energy_jobs'], case['filter_jobs'] = {}, {}
        if case['proposed']['id'] == case['original']['id'] and not smoke:
            continue
        forbidden = set(case['forbidden'])
        seeds = seed_bank(p, sid, 'energy_confirmation', 4 if smoke else p['confirmation_paths'], forbidden)
        filter_seeds = seed_bank(p, sid, 'nll_confirmation', 2 if smoke else p['guard']['seeds'], forbidden | set(seeds))
        for point in {c['id']: c for c in (case['original'], case['proposed'])}.values():
            jobs = point_jobs(case['template'], point, output, 'energy_confirmation', seeds, p,
                              training_generation=True, smoke=smoke)
            case['energy_jobs'][point['id']] = jobs
            confirmation_jobs.extend(jobs)
            jobs = point_jobs(case['template'], point, output, 'nll_confirmation', filter_seeds, p,
                phase='training', particles=2 if smoke else p['guard']['particles'], smoke=smoke)
            case['filter_jobs'][point['id']] = jobs
            filter_jobs.extend(jobs)
    pd.DataFrame(scores).to_csv(output/'ranking_scores.csv', index=False)
    write(output/'proposals.json', dict(cases=cases, later_choices_used=False, oral_used=False))
    execute(confirmation_jobs+filter_jobs, output, budget)
    decisions, nominees = [], []
    for case in cases:
        old, proposed = case['original'], case['proposed']
        likelihood, generation = {}, {}
        if case['energy_jobs']:
            x, y = [training_features(case['energy_jobs'][c['id']], np.array(case['empirical'])) for c in (old, proposed)]
            generation = paired_energy_guard(x, y, np.array(case['empirical']), p['mcse_multiplier'])
            a, b = [filter_arrays(case['filter_jobs'][c['id']]) for c in (old, proposed)]
            np.testing.assert_array_equal(a['seeds'], b['seeds'])
            likelihood = likelihood_guard(a['observed_probability'], b['observed_probability'],
                                          p['training_nll_tolerance'], p['mcse_multiplier'])
        decision = final_decision(old['id'], proposed['id'], likelihood, generation)
        decisions.append(dict(subject=case['subject'], mode=case['mode'], **decision))
        nominees.append(dict(subject=case['subject'], mode=case['mode'], original=old,
            candidate=next(c for c in case['candidates'] if c['id'] == decision['selected']),
            candidates=case['candidates'], template=case['template']))
    for filename, sha in inputs.items():
        if digest(Path(filename)) != sha:
            raise ValueError('Training input changed before freezing')
    write(output/'decisions.json', dict(decisions=decisions, later_choices_used=False, oral_used=False))
    write(output/'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
        later_choices_used=False, oral_used=False, created_unix=time.time(),
        decisions_sha256=digest(output/'decisions.json'), selection_sources=sources()))
    write(output/'completion.json', dict(status='complete', smoke=smoke, groups=len(cases),
        replaced=sum(d['replaced'] for d in decisions), ranking_paths=sum(len(j['seeds']) for j in ranking_jobs),
        confirmation_paths=sum(len(j['seeds']) for j in confirmation_jobs), filter_jobs=len(filter_jobs),
        seconds=time.monotonic()-started, nominees_sha256=digest(output/'nominees.json')))
    print('Training complete; all parameter choices frozen before evaluation', flush=True)


def evaluate(config: Path, training: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new evaluation output')
    p = settings(config)
    frozen, complete, protocol = [read(training/n) for n in ('nominees.json', 'completion.json', 'protocol.json')]
    if (complete['status'] != 'complete' or complete['smoke'] or frozen['status'] != 'frozen_before_test'
            or frozen['later_choices_used'] or protocol['policy'] != p
            or complete['nominees_sha256'] != digest(training/'nominees.json')
            or frozen['selection_sources'] != sources()):
        raise ValueError('Evaluation requires matching intact formal training freeze')
    output.mkdir(parents=True)
    jobs, cases = [], []
    previous_cases = read(training/'proposals.json')['cases']
    for row in frozen['nominees']:
        sid, mode = row['subject'], row['mode']
        previous = next(c for c in previous_cases if c['subject'] == sid and c['mode'] == mode)
        forbidden = set(previous['forbidden'])
        for mapping in (previous['energy_jobs'], previous['filter_jobs']):
            forbidden.update(s for js in mapping.values() for j in js for s in j['seeds'])
        filters = seed_bank(p, sid, 'frozen_filter', p['evaluate']['seeds'], forbidden)
        seeds = seed_bank(p, sid, 'frozen_autonomous', p['evaluation_paths'], forbidden | set(filters))
        for point in {c['id']: c for c in (row['original'], row['candidate'])}.values():
            fj = point_jobs(row['template'], point, output, 'evaluation_filter', filters, p, phase='evaluate',
                            particles=p['evaluate']['particles'], freeze=training/'nominees.json')
            aj = point_jobs(row['template'], point, output, 'evaluation_paths', seeds, p,
                            freeze=training/'nominees.json')
            jobs.extend(fj+aj)
            cases.append(dict(subject=sid, mode=mode, candidate=point, folder=row['template']['folder'],
                filter_jobs=fj, autonomous_jobs=aj, is_original=point['id']==row['original']['id'],
                is_selected=point['id']==row['candidate']['id']))
    write(output/'protocol.json', dict(policy=p, training=str(training.resolve()), cases=cases,
        frozen_nominees_sha256=digest(training/'nominees.json'), provenance=manifest(config), selection_sources=sources()))
    execute(jobs, output, p['parallel_budget'])
    write(output/'completion.json', dict(status='complete', cases=len(cases),
        filter_jobs=sum(len(c['filter_jobs']) for c in cases), paths=sum(len(j['seeds']) for c in cases for j in c['autonomous_jobs']),
        protocol_sha256=digest(output/'protocol.json'), frozen_nominees_sha256=digest(training/'nominees.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'evaluate'), required=True)
    for name in ('config', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == 'train':
            train(args.config, args.output, args.smoke)
        elif args.training and not args.smoke:
            evaluate(args.config, args.training, args.output)
        else:
            parser.error('Evaluation requires --training and a formal run')


if __name__ == '__main__':
    # Shared dispatcher invokes this standalone script with --worker only.
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        main()
