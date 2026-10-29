"""Bounded nested execution comparison with isolated prefix and recovery fits.

Use --phase smoke --output NEW first, then --phase prepare and --phase run
with the same fresh formal output. --phase report reads completed artifacts.
Numerical jobs reuse model_1024_individual_regions.worker and its frozen
shared scientific runtime. This module changes estimation, not cognition.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

from . import model_1024_individual_regions as regions
from .model_1023_execution_summary import trajectory_diagnostics
from ...evaluation.execution_comparison import conditional_score, score_difference, precision_check
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...utils.seeding import stable_seed

ROOT = regions.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
METRIC = 'src/Bayesian_state/evaluation/execution_comparison.py'
KEYS = regions.KEYS
read, write, digest = regions.read, regions.write, regions.digest


def sources() -> dict:
    return dict(regions.sources(), **{p: digest(ROOT / p) for p in
                (ENTRY, METRIC, 'src/Bayesian_state/metrics/autonomous_fit.py',
                 'src/Bayesian_state/workflows/analysis/model_1023_execution_summary.py')})


def seed(p: dict, case_id: str, role: str, repeat: int) -> int:
    return stable_seed(dict(protocol=p['protocol'], base=p['seed'], case=case_id, role=role, repeat=repeat))


def with_kappa(point: dict, value: float, condition: int, origin: str) -> dict:
    values = dict(point['hyperparams'], **{KEYS['inertia']: float(value)})
    return regions.canonical(values, condition, origin)


def neutral_point(template: dict, p: dict, condition: int) -> dict:
    values = deepcopy(template['candidate']['hyperparams'])
    for name, candidates in p['parameter_values'].items():
        values[KEYS[name]] = candidates[len(candidates) // 2]
    return regions.canonical(values, condition, 'support_midpoint')


def paired_candidates(points: list[dict], condition: int, kappas: list[float]) -> list[dict]:
    """Every nonzero candidate has an evaluated zero-boundary projection."""
    bank = {}
    positive = [k for k in kappas if k > 0]
    for index, point in enumerate(points):
        kappa = point['hyperparams'][KEYS['inertia']]
        for k in (0., kappa if kappa > 0 else positive[index % len(positive)]):
            own = with_kappa(point, k, condition, 'paired_boundary_projection')
            bank.setdefault(own['id'], own)
    return list(bank.values())


def propose(p: dict, case: dict, count: int, round_index: int,
            previous: list[dict] | None = None) -> list[dict]:
    """Use both boundary leaders, joint perturbations and dispersed starts."""
    rng = np.random.default_rng(seed(p, case['id'], f'proposal_{round_index}', 0))
    anchors = []
    if previous:
        for positive in (False, True):
            rows = [r for r in previous if (r['candidate']['hyperparams'][KEYS['inertia']] > 0) == positive]
            anchors.extend(r['candidate'] for r in sorted(rows, key=lambda r: r['score']['nll'])[:3])
    if not anchors:
        anchors = case['warm'] or [neutral_point(case['template'], p, case['condition'])]
    points = list(case['warm']) if round_index == 0 else []
    names = [k for k in KEYS if k != 'gamma' or case['condition'] == 3]
    seen = {r['candidate']['id'] for r in previous or []}
    bank = {r['id']: r for r in paired_candidates(points, case['condition'], p['parameter_values']['inertia'])}
    target = 2 * count
    for attempt in range(10000):
        if len(bank) >= target:
            break
        values = deepcopy(anchors[attempt % len(anchors)]['hyperparams'])
        changed = names if round_index == 0 or attempt % 5 == 0 else rng.choice(names, size=1 + attempt % 3, replace=False)
        for name in changed:
            options = p['parameter_values'][name]
            values[KEYS[name]] = options[int(rng.integers(len(options)))]
        point = regions.canonical(values, case['condition'], f'round_{round_index}')
        for own in paired_candidates([point], case['condition'], p['parameter_values']['inertia']):
            if own['id'] not in seen:
                bank.setdefault(own['id'], own)
    # Both members of a pair remain available, even if one already exists.
    return list(bank.values())


def freeze_shortlist(rows: list[dict], per_boundary: int, protected: list[dict] | None = None) -> list[dict]:
    """Retain leading new points and prior anchors despite noisy screening."""
    selected = {}
    for positive in (False, True):
        eligible = [r for r in rows if (r['candidate']['hyperparams'][KEYS['inertia']] > 0) == positive]
        ranked = sorted(eligible, key=lambda r: (r['score']['nll'], r['candidate']['id']))
        by_id = {r['candidate']['id']: r for r in eligible}
        ordering = ranked[:min(2, per_boundary)]
        ordering += [by_id[p['id']] for p in protected or [] if p['id'] in by_id]
        ordering += ranked
        own = {}
        for row in ordering:
            own.setdefault(row['candidate']['id'], row)
            if len(own) >= per_boundary:
                break
        for row in own.values():
            selected[row['candidate']['id']] = row['candidate']
    return list(selected.values())


def copy_fixture(original: Path, target: Path, arrays: dict, cut: int, role: str) -> dict:
    """Create a new derived fixture; preserve the source task and engine."""
    target.mkdir(parents=True, exist_ok=False)
    meta = deepcopy(read(original / 'prepared.json'))
    if digest(original / 'engine.json') != meta['engine_sha256']:
        raise ValueError('Source engine identity changed')
    shutil.copy2(original / 'engine.json', target / 'engine.json')
    n = len(arrays['choices'])
    if not 0 < cut < n:
        raise ValueError('Fixture needs nonempty prefix and suffix')
    hashes = {}
    for name, part in [('training.npz', slice(0, cut)), ('sealed_test.npz', slice(cut, n))]:
        with (target / name).open('xb') as stream:
            np.savez_compressed(stream, **{k: v[part] for k, v in arrays.items()})
        hashes[name] = digest(target / name)
    meta.update(trials=n, training_trials=cut, test_trials=n-cut, array_sha256=hashes,
                data_role=role, original_prepared_sha256=digest(original / 'prepared.json'))
    write(target / 'prepared.json', meta)
    return meta


def prepare(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new study output')
    p = yaml.safe_load(config.read_text())
    if sum(p['budget']['stage_cpu_hours'].values()) != p['budget']['cpu_hours']:
        raise ValueError('CPU allocations do not sum to the registered total')
    if smoke:
        p['subjects'] = p['subjects'][:1]
        p['budget'].update(workers=1, wall_hours=1, poll_seconds=.1)
        for group in ('full', 'prefix', 'recovery'):
            p['search'][group].update(initial_pairs=2, rounds=0, round_pairs=1, finalists_per_boundary=1)
        p['search'].update(particles=2, repeats=2, confirmation_particles=4, confirmation_repeats=2,
                           audit_per_boundary=1, audit_particles=8, audit_repeats=2,
                           escalation_particles=16, escalation_repeats=2)
        p['generation'].update(paths=8, chunk=4)
        p['recovery'].update(subjects=p['subjects'], kappas=[0, 2], replicates=1,
                             confirmation_repeats=2, audit_particles=8, audit_repeats=2,
                             escalation_particles=16, evaluation_paths=8)
    output.mkdir(parents=True)
    old = ROOT / p['source_study']
    templates = read(old / 'autonomous_jobs.json')['jobs']
    nominees = read(old / 'nominees.json')
    if nominees['later_choices_used'] or nominees['smoke'] or nominees['status'] != 'frozen_before_test':
        raise ValueError('Prefix warm starts must come from frozen prefix-only fits')
    cases, bindings, generation = [], {}, []
    for sid in p['subjects']:
        template = deepcopy(next(j for j in templates if j['subject'] == sid and j['model'] == 'STICKY'))
        original = Path(template['folder'])
        meta, observed = regions.load_observed(original)
        if smoke:
            arrays = {k: v[:32] for k, v in observed.items()}
            folder = output / 'smoke_data' / f'subject_{sid}'
            meta = copy_fixture(original, folder, arrays, 16, 'short_smoke_only')
            template['folder'] = str(folder.resolve())
        prefix_warm = [r['candidate'] for r in nominees['nominees'] if r['subject'] == sid]
        full_file = ROOT / p['full_record_study'] / f'confirmed_S{sid}.json'
        full_warm = [r['candidate'] for r in read(full_file)]
        bindings[str(full_file.relative_to(ROOT))] = digest(full_file)
        for group, warm in [('full', full_warm + prefix_warm), ('prefix', prefix_warm)]:
            case = dict(id=f'{group}_S{sid}', group=group, subject=sid, condition=meta['condition'],
                        trials=meta['trials'], cut=meta['training_trials'], template=deepcopy(template),
                        warm=warm, input_hashes={n: digest(Path(template['folder']) / n) for n in
                            ('prepared.json', 'engine.json', 'training.npz', 'sealed_test.npz')})
            if smoke:
                case['warm'] = case['warm'][:1]
            cases.append(case)
        if sid in p['recovery']['subjects']:
            for kappa in p['recovery']['kappas']:
                generation.append(dict(subject=sid, condition=meta['condition'],
                    template=deepcopy(template), point=with_kappa(template['candidate'], kappa, meta['condition'], 'fixed_generation'),
                    id=f'G{len(generation):03d}', trials=meta['trials'], cut=meta['training_trials']))
    bindings[str((old / 'nominees.json').relative_to(ROOT))] = digest(old / 'nominees.json')
    current = sources()
    for name in (ENTRY, METRIC):
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(config, output / 'registered_config.yaml')
    protocol = dict(policy=p, cases=cases, generation=generation, smoke=smoke, sources=current,
                    region_sources=regions.sources(), input_bindings=bindings, created_unix=time.time(),
                    python=sys.version, versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'PyYAML')},
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    oral_used=False, human_scope='previously_inspected_development_cases',
                    selection='sequence_likelihood; autonomous_distribution_evaluated_after_freeze')
    write(output / 'protocol.json', protocol)
    write(output / 'protocol_digest.json', dict(sha256=digest(output / 'protocol.json')))


def jobs_for(protocol: dict, case: dict, points: list[dict], output: Path, role: str,
             particles: int = 0, repeats: int = 0, paths: int = 0, evaluate: bool = False) -> list[dict]:
    p = protocol['policy']
    bank = [seed(p, case['id'], role, i) for i in range(paths or repeats)]
    size = p['generation']['chunk'] if paths else 1
    jobs = []
    for point in points:
        for offset in range(0, len(bank), size):
            name = f'{case["id"]}_{role}_{point["id"]}_{offset:03d}'
            job = deepcopy(case['template'])
            phase = 'autonomous' if paths else ('evaluate' if case['group'] == 'full' or evaluate else 'training')
            mode = 'STICKY' if paths or point['hyperparams'][KEYS['inertia']] else 'MIX'
            job.update(name=name, phase=phase, model='COMPACT', comparison_arm='ADAPT', execution_mode=mode,
                candidate=point, seeds=bank[offset:offset+size], particles=particles, smoke=False,
                source_sha256=regions.study.sources(regions.study.driver.io), region_sources=protocol['region_sources'],
                target=str((output / 'numerical' / name).resolve()),
                expected_trials=case['trials'] if phase in ('evaluate', 'autonomous') else case['cut'],
                comparison_case=case['id'], comparison_role=role, budget_stage=case['group'])
            job['mechanism']['capacity'] = 'evidence'
            jobs.append(job)
    return jobs


class Queue:
    """One process layer, measured CPU receipts, hard wall and CPU limits."""
    def __init__(self, output: Path, protocol: dict):
        self.output, self.protocol = output, protocol
        self.policy = protocol['policy']['budget']
        for name in ('jobs', 'ledger', 'batches'):
            (output / name).mkdir(exist_ok=True)
        clock = output / 'run_clock.json'
        if not clock.exists():
            write(clock, dict(started_unix=time.time()))
        self.started = read(clock)['started_unix']
        self.deadline = self.started + self.policy['wall_hours'] * 3600
        self.records = {f.stem: read(f) for f in (output / 'ledger').glob('*.json')}
        self.coordinator_file = output / 'coordinator_usage.json'
        self.coordinator_prior = read(self.coordinator_file)['cpu_seconds'] if self.coordinator_file.exists() else 0.
        self.coordinator_started = time.process_time()
        self.stop = None
        self.peak = 0
        self.exhausted_stages = set()

    def used(self, stage: str | None = None) -> float:
        workers = sum(r['cpu_seconds'] for r in self.records.values() if stage is None or r['stage'] == stage)
        coordinator = self.coordinator_prior + time.process_time() - self.coordinator_started
        return workers + (coordinator if stage in (None, 'report') else 0.)

    def estimate(self, job: dict) -> float:
        if self.protocol['smoke']:
            return 15.
        k = job['candidate']['hyperparams'][KEYS['capacity']]
        sid = job['subject']
        scale = max(.5, k / regions.OLD_CAPACITY[sid])
        if job['phase'] == 'autonomous':
            return max(15., regions.PATH_COST[sid] * len(job['seeds']) * scale * 1.5)
        n = read(Path(job['folder']) / 'prepared.json')['trials']
        fraction = job['expected_trials'] / n
        estimate = regions.HIGH_COST[sid] * job['particles'] / 512 * scale * fraction
        return max(30., estimate * 1.5)

    def execute(self, jobs: list[dict], label: str) -> None:
        batch = self.output / 'batches' / (label + '.json')
        if not batch.exists():
            write(batch, dict(jobs=jobs, created_unix=time.time()))
        elif read(batch)['jobs'] != jobs:
            raise ValueError('Registered batch identity changed: ' + label)
        pending = [j for j in jobs if j['name'] not in self.records]
        active = {}
        workers = min(self.policy['workers'], len(os.sched_getaffinity(0)))
        last_print = 0.
        terminate_at = None
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='model1024_execution_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                       NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            while pending or active:
                now = time.time()
                live = {}
                for name, item in active.items():
                    try:
                        stat = Path(f'/proc/{item["process"].pid}/stat').read_text().split(') ', 1)[1].split()
                        live[name] = (float(stat[11]) + float(stat[12])) / os.sysconf('SC_CLK_TCK')
                    except FileNotFoundError:
                        live[name] = 0.
                if self.stop is None:
                    if now >= self.deadline:
                        self.stop = 'wall_budget'
                    elif self.used() + sum(live.values()) >= self.policy['cpu_hours'] * 3600:
                        self.stop = 'cpu_budget'
                    elif sum(r['status'] == 'failed' for r in self.records.values()) >= self.policy['maximum_failures']:
                        self.stop = 'repeated_worker_failure'
                if self.stop and active:
                    if terminate_at is None:
                        terminate_at = now
                        for item in active.values():
                            item['process'].terminate()
                    elif now - terminate_at > 15:
                        for item in active.values():
                            item['process'].kill()
                for group, limit in self.policy['stage_cpu_hours'].items():
                    group_live = sum(live.get(name, 0.) for name, item in active.items()
                                     if item['job']['budget_stage'] == group)
                    if self.used(group) + group_live >= limit * 3600:
                        self.exhausted_stages.add(group)
                for item in active.values():
                    if item['job']['budget_stage'] in self.exhausted_stages:
                        if 'stage_terminated_at' not in item:
                            item['stage_terminated_at'] = now
                            item['process'].terminate()
                        elif now - item['stage_terminated_at'] > 15:
                            item['process'].kill()
                while pending and len(active) < workers and not self.stop:
                    feasible = None
                    for index, job in enumerate(pending):
                        estimate = self.estimate(job)
                        group = job['budget_stage']
                        if group in self.exhausted_stages:
                            continue
                        reserved = sum(max(v['estimate'], live.get(k, 0.)) for k, v in active.items())
                        group_reserved = sum(max(v['estimate'], live.get(k, 0.)) for k, v in active.items()
                                             if v['job']['budget_stage'] == group)
                        if (self.used() + reserved + estimate <= self.policy['cpu_hours'] * 3600
                            and self.used(group) + group_reserved + estimate <= self.policy['stage_cpu_hours'][group] * 3600
                            and estimate * 2 < self.deadline - now):
                            feasible = index
                            break
                    if feasible is None:
                        break
                    job = deepcopy(pending.pop(feasible))
                    estimate = self.estimate(job)
                    job['cpu_limit_seconds'] = int(max(180., estimate * 4))
                    file = self.output / 'jobs' / (job['name'] + '.json')
                    if file.with_suffix('.log').exists():
                        raise RuntimeError('Interrupted job needs an explicit receipt audit: ' + job['name'])
                    write(file, job)
                    stream = file.with_suffix('.log').open('x')
                    worker_entry = job.get('worker_entry', str(ROOT / regions.ENTRY))
                    proc = subprocess.Popen([sys.executable, '-c', launcher, job['runtime'], worker_entry,
                        '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    active[job['name']] = dict(process=proc, stream=stream, job=job, estimate=estimate, started=now)
                    self.peak = max(self.peak, len(active))
                completed_this_poll = False
                for name, item in list(active.items()):
                    pid, status, usage = os.wait4(item['process'].pid, os.WNOHANG)
                    if not pid:
                        continue
                    code = os.waitstatus_to_exitcode(status)
                    item['process'].returncode = code
                    item['stream'].close()
                    receipt = self.output / 'jobs' / (name + '.usage.json')
                    success = code == 0 and receipt.exists() and read(receipt)['status'] == 'complete'
                    row = dict(name=name, stage=item['job']['budget_stage'], status='complete' if success else
                               ('budget_terminated' if self.stop or 'stage_terminated_at' in item else 'failed'), returncode=code,
                               cpu_seconds=usage.ru_utime + usage.ru_stime, wall_seconds=now-item['started'],
                               max_rss_kib=usage.ru_maxrss, finished_unix=now, estimate=item['estimate'])
                    write(self.output / 'ledger' / (name + '.json'), row)
                    self.records[name] = row
                    del active[name]
                    completed_this_poll = True
                progress = dict(batch=label, complete=sum(r['status'] == 'complete' for r in self.records.values()),
                    failed=sum(r['status'] == 'failed' for r in self.records.values()), active=len(active),
                    pending=len(pending), peak_workers=self.peak, cpu_hours=(self.used()+sum(live.values()))/3600,
                    elapsed_hours=(now-self.started)/3600, stop_reason=self.stop, updated_unix=now)
                write(self.output / 'progress.json', progress)
                write(self.coordinator_file, dict(cpu_seconds=self.used('report'), updated_unix=now))
                if now-last_print >= 60:
                    print(json.dumps(progress), flush=True)
                    last_print = now
                if not active:
                    if pending and completed_this_poll and not self.stop:
                        continue
                    break
                time.sleep(self.policy['poll_seconds'])
        write(self.output / 'batches' / (label + '.completion.json'),
              dict(status='complete' if not pending and not self.stop else 'bounded_partial',
                   stop_reason=self.stop, unsubmitted=[j['name'] for j in pending],
                   succeeded=sum(self.records.get(j['name'], {}).get('status') == 'complete' for j in jobs),
                   expected=len(jobs)))


def score_bank(jobs: list[dict], points: list[dict]) -> list[dict]:
    rows = []
    for point in points:
        own = [j for j in jobs if j['candidate']['id'] == point['id']]
        probability = regions.completed_arrays(own, 'observed_probability')
        if probability is not None:
            rows.append(dict(candidate=point, score=conditional_score(probability), jobs=own))
    return rows


def filter_stage(protocol: dict, cases: list[dict], banks: dict, output: Path, queue: Queue,
                 role: str, particles: int | dict, repeats: int | dict, evaluate: bool = False) -> dict:
    jobs = []
    for case in cases:
        jobs.extend(jobs_for(protocol, case, banks.get(case['id'], []), output, role,
                             particles=particles[case['group']] if isinstance(particles, dict) else particles,
                             repeats=repeats[case['group']] if isinstance(repeats, dict) else repeats,
                             evaluate=evaluate))
    queue.execute(jobs, role)
    result = {c['id']: score_bank([j for j in jobs if j['comparison_case'] == c['id']], banks.get(c['id'], []))
              for c in cases}
    write(output / 'scores' / (role + '.json'), result)
    return result


def fit_all(protocol: dict, cases: list[dict], output: Path, queue: Queue) -> dict:
    """Interleave ready cases while preserving each case's scientific boundary."""
    p, search = protocol['policy'], protocol['policy']['search']
    banks = {c['id']: propose(p, c, search[c['group']]['initial_pairs'], 0) for c in cases}
    scores = filter_stage(protocol, cases, banks, output, queue, 'screen_0', search['particles'], search['repeats'])
    for index in range(max(search[c['group']]['rounds'] for c in cases)):
        new = {c['id']: propose(p, c, search[c['group']]['round_pairs'], index+1, scores[c['id']])
               if index < search[c['group']]['rounds'] else [] for c in cases}
        extra = filter_stage(protocol, cases, new, output, queue, f'screen_{index+1}', search['particles'], search['repeats'])
        for case in cases:
            scores[case['id']].extend(extra[case['id']])
    protected = {c['id']: [point for warm in c['warm'] for point in
                          (warm, with_kappa(warm, 0., c['condition'], 'protected_zero_projection'))]
                 for c in cases}
    shortlist = {c['id']: freeze_shortlist(scores[c['id']], search[c['group']]['finalists_per_boundary'], protected[c['id']])
                 for c in cases}
    repeats = {g: p['recovery']['confirmation_repeats'] if g == 'recovery' else search['confirmation_repeats']
               for g in ('full', 'prefix', 'recovery')}
    confirmed = filter_stage(protocol, cases, shortlist, output, queue, 'confirm', search['confirmation_particles'], repeats)
    selected = {c['id']: freeze_shortlist(confirmed[c['id']], search['audit_per_boundary']) for c in cases}
    high_n = {g: p['recovery']['audit_particles'] if g == 'recovery' else search['audit_particles'] for g in repeats}
    high_r = {g: p['recovery']['audit_repeats'] if g == 'recovery' else search['audit_repeats'] for g in repeats}
    audited = filter_stage(protocol, cases, selected, output, queue, 'audit', high_n, high_r)
    escalation = {}
    for case in cases:
        low = {r['candidate']['id']: r for r in confirmed[case['id']]}
        escalation[case['id']] = []
        for row in audited[case['id']]:
            row['precision'] = precision_check(low[row['candidate']['id']]['score'], row['score'], search)
            if not row['precision']['passed']:
                escalation[case['id']].append(row['candidate'])
    higher_n = {g: p['recovery']['escalation_particles'] if g == 'recovery' else search['escalation_particles'] for g in repeats}
    higher = {c['id']: [] for c in cases}
    # All finalists receive the same last resolution if any requires escalation;
    # selection never compares scores computed at different resolutions.
    escalate_all = {c['id']: selected[c['id']] if escalation[c['id']] else [] for c in cases}
    if any(escalate_all.values()):
        higher = filter_stage(protocol, cases, escalate_all, output, queue, 'escalation_all', higher_n, search['escalation_repeats'])
    frozen = {}
    for case in cases:
        cid = case['id']
        previous = {r['candidate']['id']: r for r in audited[cid]}
        final = [r for r in higher[cid] if r['candidate']['id'] in previous] if escalate_all[cid] else audited[cid]
        for row in final:
            if escalate_all[cid]:
                row['precision'] = precision_check(previous[row['candidate']['id']]['score'], row['score'], search)
        zero = sorted([r for r in final if r['candidate']['hyperparams'][KEYS['inertia']] == 0], key=lambda r: r['score']['nll'])
        positive = sorted([r for r in final if r['candidate']['hyperparams'][KEYS['inertia']] > 0], key=lambda r: r['score']['nll'])
        if not zero or not positive:
            frozen[cid] = dict(status='incomplete', candidates=final)
            continue
        free = min([zero[0], positive[0]], key=lambda r: r['score']['nll'])
        frozen[cid] = dict(status='frozen', zero=zero[0], free=free, positive=positive[0],
                           candidates=final, frozen_unix=time.time(), numerical_resolution=all(r['precision']['passed'] for r in (zero[0], free, positive[0])),
                           source_scope=case['group'], candidate_count=len(scores[cid]))
    for group in ('full', 'prefix', 'recovery'):
        write(output / 'frozen' / (group+'.json'), {c['id']: frozen[c['id']] for c in cases if c['group'] == group})
    return frozen


def unique_nominees(frozen: dict, case_id: str) -> list[dict]:
    row = frozen[case_id]
    if row['status'] != 'frozen':
        return []
    return list({row[k]['candidate']['id']: row[k]['candidate'] for k in ('zero', 'free', 'positive')}.values())


def prepare_recovery(protocol: dict, output: Path, queue: Queue) -> list[dict]:
    existing = output / 'recovery_cases.json'
    if existing.exists():
        return read(existing)['cases']
    jobs, backgrounds = [], []
    p = protocol['policy']
    for item in protocol['generation']:
        case = dict(id=item['id'], group='recovery', subject=item['subject'], condition=item['condition'],
                    trials=item['trials'], cut=item['cut'], template=item['template'])
        own = jobs_for(protocol, case, [item['point']], output, 'recovery_generate', paths=p['recovery']['replicates'])
        jobs.extend(own)
        backgrounds.append((item, own))
    queue.execute(jobs, 'recovery_truth_generation')
    cases, truths = [], []
    for item, own in backgrounds:
        choices = regions.completed_arrays(own, 'choices')
        feedback = regions.completed_arrays(own, 'feedback')
        if choices is None or feedback is None:
            continue
        original = Path(item['template']['folder'])
        meta, data = regions.load_observed(original)
        for index in range(len(choices)):
            cid = f'R{len(cases):03d}'
            arrays = dict(stimulus=data['stimulus'], categories=data['categories'], choices=choices[index], feedback=feedback[index])
            folder = output / 'recovery_data' / cid
            copy_fixture(original, folder, arrays, meta['training_trials'], 'new_synthetic_recovery')
            template = deepcopy(item['template'])
            template['folder'] = str(folder.resolve())
            template['candidate'] = neutral_point(template, p, meta['condition'])
            case = dict(id=cid, group='recovery', subject=item['subject'], condition=item['condition'],
                trials=len(choices[index]), cut=meta['training_trials'], template=template, warm=[],
                input_hashes={n: digest(folder/n) for n in ('prepared.json', 'engine.json', 'training.npz', 'sealed_test.npz')})
            cases.append(case)
            truths.append(dict(id=cid, point=item['point'], generation_jobs=own, path_index=index,
                               source='fixed_before_generation; all_paths_retained'))
    write(output / 'recovery_truths.json', dict(rows=truths, role='open_after_recovery_estimates_freeze'))
    write(existing, dict(cases=cases, status='complete' if len(cases) == len(backgrounds)*p['recovery']['replicates'] else 'partial'))
    return cases


def evaluate_group(protocol: dict, cases: list[dict], frozen: dict, output: Path, queue: Queue, group: str) -> None:
    p = protocol['policy']
    banks = {c['id']: unique_nominees(frozen, c['id']) for c in cases}
    if group != 'full':
        n = p['recovery']['audit_particles'] if group == 'recovery' else p['search']['audit_particles']
        r = p['recovery']['audit_repeats'] if group == 'recovery' else p['search']['audit_repeats']
        filter_stage(protocol, cases, banks, output, queue, group+'_evaluation_low', max(2, n//2), r, evaluate=True)
        filter_stage(protocol, cases, banks, output, queue, group+'_evaluation', n, r, evaluate=True)
    jobs = []
    count = p['recovery']['evaluation_paths'] if group == 'recovery' else p['generation']['paths']
    for case in cases:
        jobs.extend(jobs_for(protocol, case, banks[case['id']], output, group+'_generation', paths=count))
    queue.execute(jobs, group+'_generation')
    write(output / 'generation_jobs' / (group+'.json'), jobs)
    if group == 'recovery':
        truths = {r['id']: r['point'] for r in read(output/'recovery_truths.json')['rows']}
        truth_banks = {c['id']: [truths[c['id']]] for c in cases}
        # Open truth only after fitting has written its frozen record. Evaluate
        # generating and estimated points with the same fresh systems.
        joint = {cid: list({r['id']: r for r in banks[cid]+truth_banks[cid]}.values()) for cid in banks}
        filter_stage(protocol, cases, joint, output, queue, 'recovery_truth_check',
                     p['recovery']['audit_particles'], p['recovery']['audit_repeats'], evaluate=True)


def report(output: Path) -> None:
    protocol = read(output/'protocol.json')
    all_cases = protocol['cases'] + (read(output/'recovery_cases.json')['cases'] if (output/'recovery_cases.json').exists() else [])
    rows, trajectory_rows, recovery_rows = [], [], []
    for group in ('full', 'prefix', 'recovery'):
        frozen_file = output/'frozen'/(group+'.json')
        if not frozen_file.exists():
            continue
        frozen = read(frozen_file)
        evaluation_file = output/'scores'/(group+'_evaluation.json')
        evaluations = read(evaluation_file) if evaluation_file.exists() else {}
        low_file = output/'scores'/(group+'_evaluation_low.json')
        low_evaluations = read(low_file) if low_file.exists() else {}
        gen_file = output/'generation_jobs'/(group+'.json')
        generations = read(gen_file) if gen_file.exists() else []
        for case in [c for c in all_cases if c['group'] == group]:
            cid = case['id']; result = frozen[cid]
            row = dict(case=cid, subject=case['subject'], group=group, status=result['status'])
            if result['status'] != 'frozen':
                rows.append(row); continue
            row.update(numerical_resolution=result['numerical_resolution'],
                       zero_nll=result['zero']['score']['nll'], free_nll=result['free']['score']['nll'],
                       positive_nll=result['positive']['score']['nll'],
                       free_kappa=result['free']['candidate']['hyperparams'][KEYS['inertia']],
                       positive_kappa=result['positive']['candidate']['hyperparams'][KEYS['inertia']],
                       candidate_count=result['candidate_count'])
            a = regions.completed_arrays(result['free']['jobs'], 'observed_probability')
            b = regions.completed_arrays(result['zero']['jobs'], 'observed_probability')
            if a is not None and b is not None:
                row.update({'fit_'+k: v for k, v in score_difference(a, b).items()})
            eval_by_id = {r['candidate']['id']: r for r in evaluations.get(cid, [])}
            if all(result[k]['candidate']['id'] in eval_by_id for k in ('zero', 'free')):
                a = regions.completed_arrays(eval_by_id[result['free']['candidate']['id']]['jobs'], 'observed_probability')
                b = regions.completed_arrays(eval_by_id[result['zero']['candidate']['id']]['jobs'], 'observed_probability')
                row.update({'suffix_'+k: v for k, v in score_difference(a, b, case['cut']).items()})
                low_by_id = {r['candidate']['id']: r for r in low_evaluations.get(cid, [])}
                checks = []
                for label, high_array in [('free', a), ('zero', b)]:
                    identity = result[label]['candidate']['id']
                    if identity in low_by_id:
                        low_array = regions.completed_arrays(low_by_id[identity]['jobs'], 'observed_probability')
                        check = precision_check(conditional_score(low_array, case['cut']),
                                                conditional_score(high_array, case['cut']), protocol['policy']['search'])
                        row[label+'_suffix_precision'] = check['passed']
                        row[label+'_suffix_particle_gap'] = check['absolute_difference']
                        checks.append(check['passed'])
                row['suffix_numerical_resolution'] = len(checks) == 2 and all(checks)
            _, observed = regions.load_observed(Path(case['template']['folder']))
            for label in ('zero', 'free', 'positive'):
                point = result[label]['candidate']
                own = [j for j in generations if j['comparison_case'] == cid and j['candidate']['id'] == point['id']]
                choices = regions.completed_arrays(own, 'choices')
                if choices is None:
                    continue
                for part, start in [('full', 0), ('suffix', case['cut'])]:
                    categories = observed['categories'][start:]
                    kwargs = dict(n_categories=2 if case['condition'] == 1 else 4,
                                  block_width=protocol['policy']['generation']['block_width'],
                                  temporal_weight=protocol['policy']['generation']['temporal_weight'])
                    energy = energy_u(behavior_features(choices[:, start:], categories, **kwargs),
                                      behavior_features(observed['choices'][None, start:], categories, **kwargs)[0])
                    row[f'{label}_{part}_energy'] = energy['value']
                    row[f'{label}_{part}_energy_mcse'] = energy['mc_se']
                for index, path in enumerate(choices):
                    if len(path) >= 128:
                        d = {k: v[0].item() for k, v in trajectory_diagnostics(path[None, :], observed['categories']).items()}
                        trajectory_rows.append(dict(case=cid, model=label, path=index, **d))
            rows.append(row)
    truth_file = output/'recovery_truths.json'
    check_file = output/'scores/recovery_truth_check.json'
    freeze_file = output/'frozen/recovery.json'
    if truth_file.exists() and check_file.exists() and freeze_file.exists():
        truths = {r['id']: r for r in read(truth_file)['rows']}
        checked, frozen = read(check_file), read(freeze_file)
        case_map = {c['id']: c for c in all_cases}
        for cid, result in frozen.items():
            if result['status'] != 'frozen':
                continue
            truth = truths[cid]['point']
            by_id = {r['candidate']['id']: r for r in checked.get(cid, [])}
            ids = {k: result[k]['candidate']['id'] for k in ('zero', 'free')}
            if not all(i in by_id for i in [truth['id'], *ids.values()]):
                continue
            arrays = {k: regions.completed_arrays(by_id[i]['jobs'], 'observed_probability') for k, i in ids.items()}
            generating = regions.completed_arrays(by_id[truth['id']]['jobs'], 'observed_probability')
            cut = case_map[cid]['cut']
            difference = score_difference(arrays['free'], arrays['zero'], cut)
            generating_kappa = truth['hyperparams'][KEYS['inertia']]
            fitted_kappa = result['free']['candidate']['hyperparams'][KEYS['inertia']]
            winner = ('zero_boundary' if fitted_kappa == 0 else
                      'nonzero_persistence' if difference['difference'] + 2*difference['mcse'] < 0 else
                      'zero_boundary' if difference['difference'] - 2*difference['mcse'] > 0 else 'numerically_unresolved')
            within = arrays['zero'] if generating_kappa == 0 else arrays['free']
            optimization = score_difference(within[:, :cut], generating[:, :cut])
            recovery_rows.append(dict(case=cid, generating_kappa=generating_kappa, fitted_kappa=fitted_kappa,
                suffix_winner=winner, same_behavior_class=(winner == ('zero_boundary' if generating_kappa == 0 else 'nonzero_persistence')),
                suffix_nll_difference=difference['difference'], suffix_mcse=difference['mcse'],
                fitted_minus_generating_prefix_nll=optimization['difference'], optimization_mcse=optimization['mcse'],
                generating_point_better_than_search=optimization['difference'] > 2*optimization['mcse'],
                scope='specified_backgrounds_and_two_generated_paths_per_point'))
    destination = output/'report_v1'; destination.mkdir(exist_ok=True)
    for name, data in [('comparison.csv', rows), ('path_diagnostics.csv', trajectory_rows), ('recovery.csv', recovery_rows)]:
        if data:
            columns = list(dict.fromkeys(k for r in data for k in r))
            with (destination/name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(data)
    ledger = [read(f) for f in (output/'ledger').glob('*.json')]
    text = ['# 执行维持与零维持边界：本轮结果', '',
        '全部真人均为已查看的开发资料。完整记录拟合、前段冻结后的后段评价及新模拟恢复分别报告。',
        'NLL和Energy均以较小为优；差值为自由维持减零维持。标准误只描述数值计算误差。', '',
        f'完成数值任务{sum(r["status"]=="complete" for r in ledger)}项；数值工作进程CPU消耗{sum(r["cpu_seconds"] for r in ledger)/3600:.2f}小时，调度与汇总另存coordinator_usage.json。', '',
        '| 案例 | 状态 | 自由模型κ | 拟合NLL差 | 后段NLL差 | 数值检查 |',
        '|---|---|---:|---:|---:|---|']
    for row in rows:
        def number(key):
            return f'{row[key]:.5f}' if key in row else '待完成'
        text.append(f'| {row["case"]} | {row["status"]} | {row.get("free_kappa", "—")} | {number("fit_difference")} | {number("suffix_difference")} | {row.get("numerical_resolution", "—")} |')
    text += ['', '数值未确定的比较保留限制。有限候选的最优点与参数或机制已获科学验证分别判断。',
             '工作集合没有经校准的参数置信覆盖含义。固定实际刺激日程与记录长度，实验自适应日程和停止规则尚待另行检验。',
             '口述未参与本轮。逐参数、逐路径与完成记录见同目录CSV及上级冻结记录。', '']
    (destination/'report.md').write_text('\n'.join(text))


def run(output: Path) -> None:
    protocol = read(output/'protocol.json')
    if digest(output/'protocol.json') != read(output/'protocol_digest.json')['sha256'] or sources() != protocol['sources']:
        raise ValueError('Registered protocol or scientific sources changed')
    for name, sha in protocol['input_bindings'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Registered source artifact changed: '+name)
    queue = Queue(output, protocol)
    calibration = []
    for case in [c for c in protocol['cases'] if c['group'] == 'full']:
        anchor = case['warm'][0]
        points = [with_kappa(anchor, k, case['condition'], 'fixed_calibration') for k in (0., 2.)]
        for label, particles in [('low', protocol['policy']['search']['particles']),
                                 ('high', protocol['policy']['search']['confirmation_particles'])]:
            own = jobs_for(protocol, case, points, output, 'calibration_'+label, particles=particles, repeats=2)
            for job in own:
                job['budget_stage'] = 'calibration'
            calibration.extend(own)
    queue.execute(calibration, 'calibration')
    # Fresh simulated paths are registered before observing this run's fits.
    recovery = prepare_recovery(protocol, output, queue)
    all_cases = protocol['cases'] + recovery
    frozen = fit_all(protocol, all_cases, output, queue)
    for group in ('full', 'prefix', 'recovery'):
        cases = [c for c in all_cases if c['group'] == group]
        if not cases or queue.stop:
            continue
        evaluate_group(protocol, cases, frozen, output, queue, group)
        report(output)
    report(output)
    write(output/'completion.json', dict(status='bounded_run_finished', stop_reason=queue.stop,
          cpu_hours=queue.used()/3600, elapsed_hours=(time.time()-queue.started)/3600,
          peak_workers=queue.peak, successful=sum(r['status']=='complete' for r in queue.records.values()),
          failed=sum(r['status']=='failed' for r in queue.records.values()), scientific_interpretation_pending=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'smoke', 'run', 'report'))
    parser.add_argument('--config', type=Path, default=Path('configs/shared/model1024_execution_comparison.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase in ('prepare', 'smoke'):
        prepare(args.config, args.output, args.phase == 'smoke')
    if args.phase in ('smoke', 'run'):
        run(args.output)
    if args.phase == 'report':
        report(args.output)


if __name__ == '__main__':
    main()
