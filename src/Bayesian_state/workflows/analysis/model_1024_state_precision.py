"""Bounded particle-resolution study using the existing passive state worker.

From the repository root: --phase prepare|run|summarize --output NEW.
``run`` accepts --cases CASE... --particles N... --systems R and creates a
registered batch before launching jobs. Indices are permanent; additional
systems extend the same bank. Use --confirmation for a separate seed bank.
Scientific states and choices are computed by the existing frozen runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from . import model_1024_individual_states as study
from . import model_1024_individual_state_summary as summary

ROOT, read, write, digest = study.ROOT, study.read, study.write, study.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def available_memory() -> int:
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:')))*1024


def child_cpu(pid: int) -> float:
    """Read live child CPU without reaping it; wait4 supplies final usage."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
    except FileNotFoundError:
        return 0.0


def bank_pass(row: dict, policy: dict, mode: str) -> bool:
    """Assess available estimates separately from planned repetition completion."""
    t = policy['numerical_checks']
    return bool(row['systems'] >= policy['minimum_diagnostic_systems']
        and row[mode+'_ess_min'] >= t['minimum_system_ess']
        and row[mode+'_maximum_weight'] <= t['maximum_system_weight']
        and row[mode+'_half_mean'] <= t['half_mean_absolute_difference']
        and row[mode+'_half_q95'] <= t['half_q95_absolute_difference'])


def difference_pass(difference: dict, policy: dict) -> bool:
    t = policy['numerical_checks']
    return bool(difference['mean'] <= t['resolution_mean_absolute_difference']
                and difference['q95'] <= t['resolution_q95_absolute_difference'])


def may_expand(stable: list[str], policy: dict) -> bool:
    return (len(set(stable) & set(policy['pilot_cases'])) >= policy['expansion_minimum_stable_pilots']
            and (not policy['expansion_requires_stable_hard_case']
                 or bool(set(stable) & set(policy['hard_cases']))))


def prepare(config: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    p = yaml.safe_load(config.read_text()); source = ROOT/p['source']
    old = read(source/'protocol.json')
    if old['sources'] != study.capture.sources():
        raise ValueError('Previously used numerical sources changed')
    cases = [c for c in old['cases'] if c['kind']=='human']
    if set(p['pilot_cases'])-set(c['id'] for c in cases):
        raise ValueError('Unknown pilot')
    inputs = {str(config): digest(config), str(source/'protocol.json'): digest(source/'protocol.json')}
    for path, expected in old['inputs'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Previous input changed: {path}')
        inputs[path] = expected
    baseline = []
    for file in sorted((source/'jobs').glob('*.json')):
        j = read(file)
        if j['state_case'] not in {c['id'] for c in cases}:
            continue
        stage = j['state_stage']; index = int(j['name'].rsplit('_', 1)[1])
        if stage == 'primary' or stage == 'resolution' or (stage == 'extension' and index < 8):
            ledger = read(source/'ledger'/file.name)
            if ledger['status'] != 'complete':
                raise ValueError('Expected completed baseline')
            baseline.append(j); inputs[str(file)] = digest(file)
    write(output/'baseline_jobs.json', baseline)
    estimates = {}
    for c in cases:
        own = [j for j in baseline if j['state_case']==c['id'] and j['particles']==1024]
        meta = [read(Path(j['individual_target']).with_suffix('.json')) for j in own]
        estimates[c['id']] = dict(cpu_seconds=max(m['cpu_seconds'] for m in meta),
            wall_seconds=max(m['wall_seconds'] for m in meta),
            rss_bytes=max(m['max_rss_kib']*1024 for m in meta), particles=1024)
    sources = dict(study.capture.sources(), **{ENTRY: digest(ROOT/ENTRY),
        study.ENTRY: digest(ROOT/study.ENTRY),
        str(Path(summary.__file__).resolve().relative_to(ROOT)): digest(Path(summary.__file__))})
    protocol = dict(policy=p, cases=cases, sources=sources, inputs=inputs, estimates=estimates,
        registered_unix=time.time(), affinity=sorted(os.sched_getaffinity(0)),
        initial_memory_available_bytes=available_memory(),
        versions=dict(python=sys.version, numpy=np.__version__, pyyaml=yaml.__version__),
        baseline='512: primary 0..15; 1024: resolution 0..7 + extension 0..7; fixed indices',
        conditioning='after current choice, before feedback; filtering and complete observed sequence',
        parameter_pooling='none', oral_used_for_inference=False)
    write(output/'protocol.json', protocol)
    for rel in sources:
        target = output/'source'/rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
    shutil.copy2(config, output/'config.yaml')
    print(json.dumps(dict(registered=str(output), cases=len(cases), baseline_jobs=len(baseline))), flush=True)


def estimate(job: dict, protocol: dict, completed: list[dict], expected_workers: int | None = None) -> dict:
    old = protocol['estimates'][job['state_case']]
    own = [r for r in completed if r['status']=='complete' and r['case']==job['state_case']]
    scale = job['particles']/old['particles']
    cpu = old['cpu_seconds']*scale
    wall = old['wall_seconds']*scale
    memory = old['rss_bytes']*scale
    if own:
        physical = protocol.get('resource_policy', {}).get('physical_cores')
        latest_n = max(r['particles'] for r in own)
        recent = [r for r in own if r['particles']==latest_n]
        def adjustment(row):
            if expected_workers is None or physical is None:
                return 1.0
            observed = row.get('peak_concurrent_workers', protocol['policy']['budget']['workers'])
            # Admission heuristic for shared hardware threads. The measured
            # hard CPU/wall stops and all scientific thresholds are unchanged.
            return min(1.0, max(1.0, expected_workers/physical)/max(1.0, observed/physical))
        cpu = max(cpu, max(r['cpu_seconds']/r['particles']*adjustment(r) for r in recent)*job['particles'])
        wall = max(wall, max(r['wall_seconds']/r['particles']*adjustment(r) for r in recent)*job['particles'])
        unshared = [r for r in recent if physical is not None
                    and r.get('peak_concurrent_workers', protocol['policy']['budget']['workers']) <= physical]
        minimum = protocol.get('resource_policy', {}).get('minimum_unshared_observations')
        if minimum is not None and len(unshared) >= minimum and latest_n >= job['particles']/2:
            # Large-array work can scale better than extrapolation from 1024
            # particles. Replace that old floor only after replicated timings
            # at the immediately preceding or same resolution, without SMT.
            sharing = max(1.0, (expected_workers or physical)/physical)
            cpu = max(r['cpu_seconds']/r['particles'] for r in unshared)*job['particles']*sharing
            wall = max(r['wall_seconds']/r['particles'] for r in unshared)*job['particles']*sharing
        memory = max(memory, max(r['max_rss_kib']*1024/r['particles'] for r in own)*job['particles'])
    if 'state_storage_dimensions' in job:
        d = job['state_storage_dimensions']; n = job['particles']
        # One retained sparse history plus ancestry/weight arrays, current
        # dense workspace and a fixed allowance for Python and compression.
        memory = (1024**3 + d['trials']*n*(d['capacity']*(26+8*d['components'])+40)
                  + n*d['rules']*8*24)
        observed = [r for r in own if r.get('worker_entry')==job['state_worker_entry']]
        if observed:
            memory = max(memory, max(r['max_rss_kib']*1024/r['particles'] for r in observed)*n)
    factor = protocol['policy']['budget']['estimate_safety_factor']
    return dict(cpu=cpu*factor+30, wall=wall*factor+30, memory=memory*factor)


def extend_confirmation_plan(output: Path, plan: list[dict], policy: dict, execution: dict) -> list[dict]:
    """Reserve independent seed blocks alongside costly high-resolution work."""
    minimum = execution.get('prefetch_confirmation_minimum_particles')
    result = list(plan)
    if minimum is None:
        return result
    for spec in plan:
        if spec.get('confirmation') or spec['systems'] < policy['systems']:
            continue
        for n in spec['particles']:
            if n < minimum:
                continue
            for cid in spec['cases']:
                if (output/'confirmations'/f'{cid}_n{n}_r{spec["systems"]}.json').exists():
                    continue
                if any(x.get('confirmation') and cid in x['cases'] and n in x['particles'] for x in result):
                    continue
                count = len(list((output/'jobs').glob(f'confirm_n{n}_{cid}_*.json')))
                total = count+policy['confirmation_systems']
                if total <= policy['maximum_systems']:
                    result.append(dict(cases=[cid], particles=[n], systems=total, confirmation=True,
                        primary_systems=spec['systems'], reason='prefetched_independent_confirmation'))
    return result


def configure_storage(job: dict, execution: dict) -> dict:
    """Attach the separately verified passive adapter to newly registered jobs."""
    if execution and job['particles'] >= execution['storage_minimum_particles']:
        job.update(state_worker_entry=execution['worker_entry'],
            precision_storage_sources=execution['worker_sources'],
            state_summary_block_trials=execution['block_trials'],
            state_storage_dimensions=execution['case_dimensions'][job['state_case']])
    return job


def run(output: Path, cases: list[str], particles: list[int], systems: int, label: str,
        confirmation: bool = False, plan: list[dict] | None = None) -> None:
    if (output/'completions'/(label+'.json')).exists():
        print('Batch already accounted', flush=True); return
    protocol = read(output/'protocol.json'); p = protocol['policy']; b = p['budget']
    sources = dict(protocol['sources'])
    resource_policy_path = output/'resource_policy.json'
    execution = {}
    for amendment in sorted((output/'amendments').glob('*.json')):
        record = read(amendment)
        if record.get('resource_policy_file'):
            resource_policy_path = output/record['resource_policy_file']
            if digest(resource_policy_path) != record['resource_policy_sha256']:
                raise ValueError('Registered resource policy changed')
        if record.get('execution_policy_file'):
            path = output/record['execution_policy_file']
            if digest(path) != record['execution_policy_sha256']:
                raise ValueError('Registered execution policy changed')
            execution = read(path)
        for path, change in record.get('source_changes', {}).items():
            if sources[path] != change['before']:
                raise ValueError('Amendment chain changed')
            sources[path] = change['after']
    if any(digest(ROOT/k) != v for k, v in sources.items()):
        raise ValueError('Registered source changed; register an explicit amendment')
    if resource_policy_path.exists():
        protocol['resource_policy'] = read(resource_policy_path)
    if execution:
        if any(digest(ROOT/k)!=v for k,v in execution['worker_sources'].items()):
            raise ValueError('Registered storage adapter changed')
        evidence = output/execution['validation_file']
        if digest(evidence)!=execution['validation_sha256'] or read(evidence)['status']!='passed':
            raise ValueError('Storage adapter requires completed exact replay validation')
    case_map = {c['id']: c for c in protocol['cases']}
    plan = plan or [dict(cases=cases, particles=particles, systems=systems, confirmation=confirmation)]
    batch_path = output/'batches'/(label+'.json')
    plan = read(batch_path)['plan'] if batch_path.exists() else extend_confirmation_plan(output,plan,p,execution)
    for spec in plan:
        if set(spec['cases'])-case_map.keys() or not 1 <= spec['systems'] <= p['maximum_systems']:
            raise ValueError('Unknown case or invalid systems')
        if set(spec['particles'])-set(p['initial_particles']+p['escalation_particles']):
            raise ValueError('Unregistered resolution')
    jobs = []
    for i in range(max(spec['systems'] for spec in plan)):
        for spec in plan:
            if i >= spec['systems']:
                continue
            for n in spec['particles']:
                for cid in spec['cases']:
                    stage = ('confirm' if spec.get('confirmation', False) else 'precision')+f'_n{n}'
                    j = study.make_job(case_map[cid], p, output, stage, n, i)
                    file = output/'jobs'/(j['name']+'.json')
                    j = read(file) if file.exists() else configure_storage(j,execution)
                    write(output/'jobs'/(j['name']+'.json'), j); jobs.append(j)
    if len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Overlapping plan entries')
    write(output/'batches'/(label+'.json'), dict(plan=plan, jobs=[j['name'] for j in jobs]))
    clock = output/'run_clock.json'
    if not clock.exists():
        write(clock, dict(started_unix=time.time()))
    started = read(clock)['started_unix']; deadline = started+b['wall_hours']*3600
    ledger = output/'ledger'; ledger.mkdir(exist_ok=True)
    completed = [read(f) for f in ledger.glob('*.json')]
    existing = {r['name'] for r in completed}
    pending = [j for j in jobs if j['name'] not in existing]
    workers = min(b['workers'], len(os.sched_getaffinity(0)))
    memory_cap = available_memory()*b['memory_fraction']
    median_memory = float(np.median([estimate(j, protocol, completed)['memory'] for j in pending])) if pending else 1.0
    expected_workers = min(workers, len(pending), max(1, int(memory_cap/median_memory)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
        NUMBA_CACHE_DIR='/tmp/model1024_precision_numba')
    launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                'sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")')
    active = {}; stop = None; peak = 0; last_print = 0.; terminated_at = None
    supervisor_start = resource.getrusage(resource.RUSAGE_SELF)
    prior_supervisor = sum(read(f).get('supervisor_cpu_seconds', 0)
                           for f in (output/'completions').glob('*.json'))
    while pending or active:
        now = time.time()
        own_usage = resource.getrusage(resource.RUSAGE_SELF)
        supervisor_cpu = own_usage.ru_utime+own_usage.ru_stime-supervisor_start.ru_utime-supervisor_start.ru_stime
        used = sum(r['cpu_seconds'] for r in completed)+prior_supervisor+supervisor_cpu
        live_cpu = sum(child_cpu(a['process'].pid) for a in active.values())
        if stop is None:
            if now >= deadline:
                stop = 'wall_budget'
            elif used+live_cpu >= b['cpu_hours']*3600:
                stop = 'cpu_budget'
            elif sum(r['status']=='failed' for r in completed) >= b['failed_jobs_before_stop']:
                stop = 'repeated_worker_failure'
        if stop and active:
            if terminated_at is None:
                terminated_at = now
                for a in active.values():
                    a['process'].terminate()
            elif now-terminated_at > 30:
                for a in active.values():
                    a['process'].kill()
        while pending and len(active) < workers and not stop:
            # Active CPU reservations use estimates, while the hard stop uses
            # measured usage. Large per-job ceilings no longer idle the queue.
            reserved = sum(max(0, a['estimate']['cpu']-child_cpu(a['process'].pid)) for a in active.values())
            memory_reserved = sum(a['estimate']['memory'] for a in active.values())
            feasible = []
            for index, j in enumerate(pending):
                est = estimate(j, protocol, completed, expected_workers)
                if (used+live_cpu+reserved+est['cpu'] <= b['cpu_hours']*3600
                    and est['wall'] < deadline-now
                    and est['memory']+memory_reserved <= memory_cap
                    and est['memory'] < available_memory()*.9):
                    feasible.append((index, est)); break
            if not feasible:
                if not active:
                    stop = 'remaining_budget_or_memory_insufficient'
                break
            index, est = feasible[0]; j = pending.pop(index)
            log = output/'logs'/(j['name']+'.log'); log.parent.mkdir(exist_ok=True)
            stream = log.open('x')
            proc = subprocess.Popen([sys.executable, '-c', launcher, j['runtime'],
                str(ROOT/j.get('state_worker_entry',study.capture.ENTRY)),
                '--worker', str((output/'jobs'/(j['name']+'.json')).resolve())], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            active[j['name']] = dict(process=proc, job=j, stream=stream, started=time.time(), estimate=est)
            peak = max(peak, len(active))
        for name, a in list(active.items()):
            pid, status, usage = os.wait4(a['process'].pid, os.WNOHANG)
            if not pid:
                continue
            code = os.waitstatus_to_exitcode(status); a['process'].returncode = code
            a['stream'].close(); j = a['job']
            file = Path(j['individual_target']).with_suffix('.json')
            complete = code == 0 and file.exists() and read(file)['status']=='complete'
            row = dict(name=name, case=j['state_case'], particles=j['particles'],
                status='complete' if complete else ('budget_terminated' if stop else 'failed'),
                returncode=code, cpu_seconds=usage.ru_utime+usage.ru_stime,
                wall_seconds=time.time()-a['started'], max_rss_kib=usage.ru_maxrss,
                finished_unix=time.time(), stop_reason=stop, estimate=a['estimate'],
                peak_concurrent_workers=peak, admission_expected_workers=expected_workers,
                worker_entry=j.get('state_worker_entry',study.capture.ENTRY),
                job_sha256=digest(output/'jobs'/(name+'.json')))
            write(ledger/(name+'.json'), row); completed.append(row); del active[name]
        progress = dict(batch=label, active=len(active), pending=len(pending), peak_workers=peak,
            complete=sum(r['status']=='complete' for r in completed),
            failed=sum(r['status']=='failed' for r in completed),
            measured_cpu_hours=(used+live_cpu)/3600, elapsed_wall_hours=(now-started)/3600,
            stop_reason=stop, updated_unix=now)
        write(output/'progress.json', progress)
        if now-last_print >= 60:
            print(json.dumps(progress), flush=True); last_print=now
        if stop and not active:
            break
        if active:
            time.sleep(b['poll_seconds'])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(output/'completions'/(label+'.json'), dict(batch=label, stop_reason=stop or 'batch_complete',
        expected_jobs=len(jobs), completed_jobs=sum(r['name'] in {j['name'] for j in jobs}
            and r['status']=='complete' for r in completed), unstarted=[j['name'] for j in pending],
        peak_workers=peak, finished_unix=time.time(),
        supervisor_cpu_seconds=usage.ru_utime+usage.ru_stime-supervisor_start.ru_utime-supervisor_start.ru_stime))
    print(json.dumps(read(output/'completions'/(label+'.json'))), flush=True)
    prefetched = [x for x in plan if x.get('reason')=='prefetched_independent_confirmation']
    if prefetched:
        decision = output/'decisions'/(label+'.json')
        scope = read(decision)['scope'] if decision.exists() else sorted({c for x in plan for c in x['cases']})
        summarize(output,label,scope)
        rows = read(output/'summaries'/label/'diagnostics.json')
        ready = [x for x in prefetched if any(r['case']==x['cases'][0]
            and r['particles']==x['particles'][0] and r['systems']==x['primary_systems']
            and not r['confirmation'] for r in rows)]
        from .model_1024_state_precision_control import check_confirmations
        check_confirmations(output,rows,ready,label)


def summarize(output: Path, label: str, cases: list[str] | None = None) -> None:
    summary_started = time.time()
    cpu_started = time.process_time()
    destination = output/'summaries'/label
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    protocol = read(output/'protocol.json'); p = protocol['policy']
    baseline_jobs = read(output/'baseline_jobs.json')
    registered_jobs = [read(f) for f in sorted((output/'jobs').glob('*.json'))]
    all_jobs = list(baseline_jobs)
    for job in registered_jobs:
        ledger = output/'ledger'/(job['name']+'.json')
        if ledger.exists() and read(ledger)['status']=='complete':
            all_jobs.append(job)
    rows = []; saved = {}
    for case in protocol['cases']:
        cid = case['id']
        if cases and cid not in cases:
            continue
        own = [j for j in all_jobs if j['state_case']==cid]
        keys = sorted({(j['particles'], j['state_stage'].startswith('confirm')) for j in own})
        for n, confirm in keys:
            jobs = [j for j in own if j['particles']==n and j['state_stage'].startswith('confirm')==confirm]
            jobs.sort(key=lambda j: (j['state_stage']!='resolution', j['name']))
            pooled, row = summary.summarize_bank(jobs)
            expected = sum(j['state_case']==cid and j['particles']==n
                and j['state_stage'].startswith('confirm')==confirm for j in baseline_jobs+registered_jobs)
            row.update(case=cid, subject=case['subject'], confirmation=confirm,
                expected_systems=expected, bank_complete=len(jobs)==expected,
                planned_minimum_complete=len(jobs) >= p['systems'],
                job_names=[j['name'] for j in jobs])
            anc = pooled['full_unique_ancestors_per_system']
            row.update(ancestor_first_median=float(np.median(anc[:, 0])),
                ancestor_quarter_median=float(np.median(anc[:, anc.shape[1]//4])),
                ancestor_mid_median=float(np.median(anc[:, anc.shape[1]//2])),
                ancestor_first_min=int(anc[:, 0].min()))
            for mode in ('filtered', 'full'):
                row[mode+'_repeat_pass'] = bank_pass(row, p, mode)
            lower = [key for key in saved if key[0]==cid and key[1]<n and not key[2]]
            refkey = (cid, n, False) if confirm else (max(lower, key=lambda k:k[1]) if lower else None)
            if refkey:
                reference = saved[refkey]
                row['comparison_particles'] = refkey[1]
                for mode in ('filtered', 'full'):
                    d = summary.discrepancy(pooled, reference, mode)
                    row.update({mode+'_comparison_'+k:v for k,v in d.items()})
                    row[mode+'_comparison_pass'] = difference_pass(d, p)
                    row[mode+'_numerically_consistent'] = row[mode+'_repeat_pass'] and difference_pass(d, p)
                    quarter = case['trials']//4
                    early = summary.discrepancy({k:v[:quarter] for k,v in pooled.items() if k.startswith(mode+'_')},
                        {k:v[:quarter] for k,v in reference.items() if k.startswith(mode+'_')}, mode)
                    row.update({mode+'_early_comparison_'+k:v for k,v in early.items()})
            file = destination/(cid+f'_n{n}'+('_confirm' if confirm else '')+'.npz')
            with file.open('xb') as stream:
                np.savez_compressed(stream, **pooled)
            row['pooled_file'] = str(file.resolve()); row['pooled_sha256'] = digest(file)
            rows.append(row); saved[(cid,n,confirm)] = pooled
    write(destination/'diagnostics.json', rows)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(destination/'manifest.json', dict(protocol_sha256=digest(output/'protocol.json'),
        completed_unix=time.time(), sources=protocol['sources'], cases=cases,
        analysis_source_sha256=digest(Path(__file__)),
        amendments={f.name:digest(f) for f in (output/'amendments').glob('*.json')},
        wall_seconds=time.time()-summary_started, process_cpu_seconds=usage.ru_utime+usage.ru_stime,
        summary_cpu_seconds=time.process_time()-cpu_started,
        note='Available-data numerical checks and scheduled-work completion are reported separately.'))
    for r in rows:
        print(json.dumps({k:r.get(k) for k in ('case','particles','systems','confirmation',
            'full_half_mean','full_half_q95','full_comparison_q95','full_ess_min',
            'full_numerically_consistent','ancestor_first_median')}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare','run','summarize'], required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/shared/model1024_state_precision.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--particles', type=int, nargs='+', default=[2048,4096])
    parser.add_argument('--systems', type=int, default=16)
    parser.add_argument('--label', default='pilot')
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--plan', type=Path, help='Registered JSON list of independent batch specifications')
    args = parser.parse_args()
    if args.phase=='prepare':
        prepare(args.config, args.output)
    elif args.phase=='run':
        run(args.output, args.cases or read(args.output/'protocol.json')['policy']['pilot_cases'],
            args.particles, args.systems, args.label, args.confirmation, read(args.plan) if args.plan else None)
    else:
        summarize(args.output, args.label, args.cases)


if __name__ == '__main__':
    main()
