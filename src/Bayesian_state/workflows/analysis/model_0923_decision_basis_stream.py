"""Replay each completed evaluation seed while the remaining seeds finish.

Use the existing replay YAML with --config CONFIG --output NEW_DIRECTORY.
Every original conditional seed must be included. All autonomous evaluation
jobs must already be complete. Each conditional job and its replay share one
reserved slot, so downstream recovery can use the remaining CPU allocation.
The existing replay worker performs the cognitive calculation and exact check.
--smoke requires a complete source evaluation and uses its first 16-trial job.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

from . import model_0923_decision_basis_replay as replay

ROOT = replay.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def reference_ready(path: Path) -> bool:
    """Wait for the original worker's final marker, including a complete JSON write."""
    marker = path.with_suffix('.experiment.json')
    try:
        replay.read(marker)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return True


def planned_jobs(settings: dict, protocol: dict, *, smoke: bool, cpus: int) -> list[dict]:
    """Cover all conditional tasks so the reservation also covers pending sources."""
    jobs = [deepcopy(j) for j in protocol['jobs'] if j['phase'] == 'evaluate']
    expected = {(s, a) for s in settings['subjects'] for a in settings['arms']}
    if (settings['sample_role'] != 'development' or len(expected) != len(settings['subjects']) * len(settings['arms'])
            or settings['seeds_per_arm'] < 1 or not jobs):
        raise ValueError('Need unique development subjects, arms and independent seeds')
    groups = {}
    for job in jobs:
        subject = replay.read(Path(job['folder']) / 'prepared.json')['subject']
        key = subject, job['model']
        if key not in expected or len(job['seeds']) != 1:
            raise ValueError('Replay must cover every original conditional job, one seed each')
        groups.setdefault(key, []).extend(job['seeds'])
    if set(groups) != expected or any(len(seeds) != settings['seeds_per_arm'] or len(set(seeds)) != len(seeds)
                                      for seeds in groups.values()):
        raise ValueError('Missing or duplicated original filter seeds')
    if len({j['name'] for j in jobs}) != len(jobs) or len({j['target'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate original job names or targets')
    jobs.sort(key=lambda j: j['name'])
    if smoke:
        return jobs[:1]
    if min(settings['workers'], cpus) < len(jobs):
        raise ValueError('Reserve one slot for each original conditional job and its replay')
    return jobs


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, list[dict]]:
    """Freeze sources and task identities before publishing the CPU reservation."""
    if output.exists():
        raise FileExistsError('Use a new streaming replay directory')
    settings = yaml.safe_load(config.read_text()); source = ROOT / settings['source']
    original = replay.read(source / 'protocol.json')
    fit = Path(original['fit']); frozen = replay.read(fit / 'nominees.json')
    done = replay.read(fit / 'completion.json')
    if (done['status'] != 'complete' or frozen['status'] != 'frozen_before_test'
            or (not smoke and (done['smoke'] or frozen['smoke']))
            or original['nomination_sha256'] != replay.digest(fit / 'nominees.json')
            or done['nomination_sha256'] != original['nomination_sha256']
            or original['source_sha256'] != frozen['source_sha256']):
        raise ValueError('Need a completed frozen fit and its own evaluation')
    jobs = planned_jobs(settings, original, smoke=smoke, cpus=len(os.sched_getaffinity(0)))
    if smoke and not (source / 'completion.json').exists():
        raise ValueError('Smoke replay requires a complete source evaluation')
    for job in original['jobs']:
        if job['phase'] == 'autonomous':
            target = Path(job['target']); record = replay.read(target.with_suffix('.json'))
            if not reference_ready(target) or replay.digest(target.with_suffix('.npz')) != record['predictions_sha256']:
                raise ValueError('All autonomous tasks must be complete before reserving conditional slots')
        elif job['phase'] != 'evaluate':
            raise ValueError('Unexpected source task outside the reserved evaluation')
    nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
    hashes = {name: replay.digest(ROOT / name) for name in (ENTRY, replay.ENTRY, replay.PF, replay.COLLECTOR, replay.REFIT)}
    if 'SEARCH' in settings['arms']:
        hashes[replay.PROPOSAL] = replay.digest(ROOT / replay.PROPOSAL)
    for job in jobs:
        subject = replay.read(Path(job['folder']) / 'prepared.json')['subject']
        nominee = nominees[subject, job['model']]
        if (job['candidate']['id'] != nominee['id'] or job['candidate']['hyperparams'] != nominee['hyperparams']
                or job['source_sha256'] != original['source_sha256']):
            raise ValueError('Original job differs from its frozen nominee or sources')
        for name, expected in job['source_sha256'].items():
            if replay.digest(ROOT / name) != expected:
                raise ValueError(f'Original source changed: {name}')
        job.update(reference=str(Path(job['target']).with_suffix('.npz')),
                   original_job=str((source / 'jobs' / (job['name'] + '.json')).resolve()),
                   replay_source_sha256=hashes, archived_pf_sha256=replay.digest(Path(job['runtime']) / replay.PF),
                   trial_limit=16 if smoke else 0, target=str((output / 'states' / job['name']).resolve()))
    output.mkdir(parents=True); (output / 'jobs').mkdir()
    for relative in hashes:
        target = output / 'source' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(config, output / 'protocol.yaml')
    protocol = dict(settings=settings, smoke=smoke, workers=len(jobs), jobs=len(jobs), source_sha256=hashes,
        config_sha256=replay.digest(config), original_protocol_sha256=replay.digest(source / 'protocol.json'),
        nomination_sha256=original['nomination_sha256'], python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        role='fixed_parameter_replay_no_oral_input', scheduling='each completed source seed immediately replayed',
        reservation='one slot per unfinished source/replay pair; all autonomous sources complete',
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    # Publish the complete reservation atomically because another queue watches it.
    with tempfile.NamedTemporaryFile(dir=output, prefix='.protocol-', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
        os.link(temporary_path, output / 'protocol.json')
    finally:
        temporary_path.unlink()
    return protocol, jobs


def bind_reference(job: dict) -> dict:
    """Verify a finished original seed before exposing it to the existing worker."""
    if not reference_ready(Path(job['reference'])):
        raise ValueError('Original worker has not finished its final source checks')
    bound = deepcopy(job); original = replay.read(Path(job['original_job']))
    for key in ('name', 'phase', 'folder', 'model', 'candidate', 'runtime', 'seeds', 'particles',
                'source_sha256', 'arm_settings', 'mechanism', 'smoke'):
        if original[key] != job[key]:
            raise ValueError(f'Original job changed: {key}')
    record = replay.read(Path(job['reference']).with_suffix('.json'))
    if (record['candidate'] != job['candidate'] or record['seeds'] != job['seeds']
            or record['particles'] != job['particles'] or record['model'] != job['model']):
        raise ValueError('Finished original seed has another model, parameter or numerical budget')
    bound.update(reference_sha256=record['predictions_sha256'], original_job_sha256=replay.digest(Path(job['original_job'])))
    replay.verify_sources(bound)
    return bound


def run(config: Path, output: Path, smoke: bool) -> None:
    protocol, jobs = prepare(config, output, smoke); source = ROOT / protocol['settings']['source']
    print(f"{len(jobs)} source/replay pairs reserved; existing exact-replay worker; numeric threads=1", flush=True)

    def launch(pending):
        while not reference_ready(Path(pending['reference'])):
            if (source / 'completion.json').exists():
                raise ValueError('Completed evaluation is missing a declared reference')
            time.sleep(2)
        job = bind_reference(pending); file = output / 'jobs' / (job['name'] + '.json'); replay.write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='stream_replay_') as cache:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / replay.ENTRY),
                    '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Replay worker failed: {file.with_suffix(".log")}')
        record = replay.read(Path(job['target']).with_suffix('.json'))
        if replay.digest(Path(job['target']).with_suffix('.npz')) != record['array_sha256']:
            raise ValueError('Replay artifact changed')
        return record

    records = []
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            records.append(future.result()); print(f'{len(records)}/{len(jobs)} exact replays complete', flush=True)
    while not (source / 'completion.json').exists():
        time.sleep(2)
    done = replay.read(source / 'completion.json')
    if (done['status'] != 'complete' or done['nomination_sha256'] != protocol['nomination_sha256']
            or replay.digest(source / 'protocol.json') != protocol['original_protocol_sha256']):
        raise ValueError('Original evaluation completion differs from the reserved source')
    replay.write(output / 'reference_validation.json', dict(status='complete',
        original_completion_sha256=replay.digest(source / 'completion.json'),
        original_protocol_sha256=protocol['original_protocol_sha256']))
    replay.write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(records),
        exact_behavior_replays=sum(r['exact_behavior_replay'] for r in records),
        cpu_seconds=sum(r['cpu_seconds'] for r in records),
        records=[dict(subject=r['subject'], arm=r['arm'], seed=r['seeds'][0], trials=r['trials']) for r in records]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); run(args.config, args.output, args.smoke)
