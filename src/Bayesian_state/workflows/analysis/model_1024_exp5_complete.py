"""Complete unstarted Exp5 precision jobs under the original global budget.

The initial queue may reserve twice an already conservative job estimate.
This coordinator launches only jobs that the initial queue cannot admit before
its deadline. Original workers keep ownership of every job already started.
Scientific inputs, seeds, candidates and particle tolerances stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import yaml

from . import model_1024_exp5_fit as fit


def split_ownership(source: Path, jobs: list[dict], queue: fit.Exp5Queue,
                    now: float) -> tuple[list[dict], list[dict]]:
    """Use started-job files and monotone deadline eligibility to prevent overlap."""
    original, supplement = [], []
    factor = queue.policy.get('wall_reserve_factor', 2.)
    for job in jobs:
        if (source / 'jobs' / (job['name'] + '.json')).exists():
            original.append(job)
        elif queue.estimate(job) * factor >= queue.deadline - now:
            supplement.append(job)
        else:
            raise ValueError('The original queue can still launch ' + job['name'])
    return original, supplement


def prepare(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Use a new supplement directory')
    old = fit.read(source / 'protocol.json')
    if old['smoke'] or old['policy']['subject'] != 501:
        raise ValueError('This completion is scoped to the formal subject 501 fit')
    audited = fit.read(source / 'scores/audit.json')
    if len(audited) != old['policy']['search']['audit_points']:
        raise ValueError('Require a complete audit bank')
    # Read-only access to the existing queue clock and records; no execute call.
    original_queue = fit.Exp5Queue(source, old)
    original_queue.coordinator_started = time.process_time()
    now = time.time()
    batch = fit.read(source / 'batches/escalation.json')['jobs']
    original, pending = split_ownership(source, batch, original_queue, now)
    if not pending or now >= original_queue.deadline:
        raise ValueError('No eligible supplemental work within the original deadline')
    # Reserve original workers' full CPU limits, the entire original generation
    # allowance, and one coordinator CPU for every remaining wall second.
    finished_cpu = sum(r['cpu_seconds'] for r in original_queue.records.values()) + original_queue.coordinator_prior
    active = [j for j in original if j['name'] not in original_queue.records]
    active_cpu = sum(fit.read(source / 'jobs' / (j['name'] + '.json'))['cpu_limit_seconds'] + 1 for j in active)
    original_upper = finished_cpu + active_cpu + old['policy']['budget']['stage_cpu_hours']['generation'] * 3600 + (original_queue.deadline-now)
    available_hours = old['policy']['budget']['cpu_hours'] - original_upper/3600
    cpu_hours = min(64., available_hours)
    if cpu_hours <= 8:
        raise ValueError('Insufficient CPU budget after reserving the original run')
    p = deepcopy(old['policy'])
    p['budget'].update(cpu_hours=cpu_hours, wall_reserve_factor=1.1,
                       workers=max(1, old['policy']['budget']['workers']-len(active)),
                       stage_cpu_hours=dict(search=0, confirmation=0, audit=cpu_hours-8, generation=7, report=1))
    output.mkdir(parents=True)
    for name, expected in old['input_sha256'].items():
        if fit.digest(source / name) != expected:
            raise ValueError('Original input identity changed: ' + name)
        shutil.copy2(source / name, output / name)
    if fit.digest(source / 'source_snapshot.tar') != old['snapshot_sha256']:
        raise ValueError('Original source archive changed')
    shutil.copy2(source / 'source_snapshot.tar', output / 'source_snapshot.tar')
    shutil.copy2(source / 'preflight_validation.json', output / 'preflight_validation.json')
    (output / 'registered_config.yaml').write_text(yaml.safe_dump(p, sort_keys=False))
    protocol = deepcopy(old)
    protocol.update(policy=p, source_run=str(source), supplement_prepared_unix=now,
        input_sha256={name: fit.digest(output / name) for name in old['input_sha256']},
        source_bindings={name: fit.digest(source / name) for name in
                         ('protocol.json', 'scores/audit.json', 'batches/escalation.json')})
    coordinator = output / 'coordinator_source'
    coordinator.mkdir()
    for name in ('model_1024_exp5_fit.py', 'model_1024_exp5_complete.py', 'model_1024_execution_comparison.py'):
        shutil.copy2(Path(__file__).with_name(name), coordinator / name)
    protocol['coordinator_sha256'] = {p.name: fit.digest(p) for p in coordinator.iterdir()}
    fit.write(output / 'protocol.json', protocol)
    fit.write(output / 'run_clock.json', dict(started_unix=original_queue.started))
    plan = dict(source_run=str(source), original_jobs=original, supplemental_names=[j['name'] for j in pending],
                original_upper_cpu_hours=original_upper/3600, supplemental_cpu_hours=cpu_hours,
                global_cpu_hours=old['policy']['budget']['cpu_hours'], global_deadline=original_queue.deadline,
                prepared_unix=now, reason='Original wall reservation rejects these unstarted jobs; reserve factor 1.1 for completion')
    fit.write(output / 'supplement.json', plan)
    return protocol


def run(output: Path) -> dict:
    output = output.resolve()
    if (output / 'completion.json').exists():
        raise FileExistsError('Supplement already completed')
    protocol = fit.read(output / 'protocol.json')
    plan = fit.read(output / 'supplement.json')
    source = Path(protocol['source_run'])
    for name, expected in protocol['source_bindings'].items():
        if fit.digest(source / name) != expected:
            raise ValueError('Original fitting record changed: ' + name)
    for name, expected in protocol['coordinator_sha256'].items():
        if fit.digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Supplement coordinator changed after registration')
    audited = fit.read(source / 'scores/audit.json')
    s = protocol['policy']['search']
    jobs = fit.jobs_for(protocol, output, audited, 'escalation', 'audit',
                        s['escalation_particles'], s['escalation_repeats'])
    own = [j for j in jobs if j['name'] in plan['supplemental_names']]
    queue = fit.Exp5Queue(output, protocol)
    queue.execute(own, 'escalation')
    # Original jobs have lower capacity and normally finish before this bank.
    while not (source / 'completion.json').exists() and time.time() < queue.deadline:
        time.sleep(2)
    if not (source / 'completion.json').exists():
        raise TimeoutError('Original coordinator has not closed its budget ledger')
    mixed = own + plan['original_jobs']
    final = []
    for point in audited:
        point_jobs = [j for j in mixed if j['candidate']['id'] == point['id']]
        values = fit.collect(point_jobs, 'observed_probability')
        if values is None:
            continue
        score = fit.sequence_score(values)
        row = {k: point[k] for k in ('id', 'hyperparams', 'sources', 'origin')}
        row.update(mean_nll=score['nll'], score=score, particles=s['escalation_particles'], jobs=point_jobs,
                   precision=fit.precision_check(point['score'], score, s))
        final.append(row)
    fit.write(output / 'scores/escalation.json', final)
    previous = fit.read(source / 'frozen.json')
    return fit.finalize(output, protocol, queue, final, audited,
                        previous['evaluated_candidates'], previous['proposed_candidates'], True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.source is None:
            parser.error('--source is required for preparation')
        prepare(args.source, args.output)
    else:
        print(run(args.output), flush=True)


if __name__ == '__main__':
    main()
