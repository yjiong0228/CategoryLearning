"""Prepare a fresh bounded continuation using verified completed worker results.

The registered comparison runner remains the numerical entry point. This tool
changes only the output location and the additional computing allowance. It
requires a terminal source run, preserves its artifacts, and never launches jobs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import yaml


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def verify_source(source: Path, checkout: Path) -> tuple[dict, list[dict], dict]:
    """Check terminal state, scientific identity, inputs and successful workers."""
    protocol = read(source / 'protocol.json')
    if digest(source / 'protocol.json') != read(source / 'protocol_digest.json')['sha256']:
        raise ValueError('Source protocol digest mismatch')
    completion = source / 'completion.json'
    if not completion.exists() or read(completion).get('status') != 'bounded_run_finished':
        raise ValueError('Source numerical run must have a terminal completion record')
    supervisor = source / 'supervisor_completion.json'
    if not protocol['smoke'] and not supervisor.exists():
        raise ValueError('Formal source run requires its terminal supervisor receipt')
    for name, expected in protocol['sources'].items():
        if digest(checkout / name) != expected:
            raise ValueError('Registered scientific source changed: ' + name)
    for name, expected in protocol['input_bindings'].items():
        if digest(checkout / name) != expected:
            raise ValueError('Registered historical input changed: ' + name)
    recovery = read(source / 'recovery_cases.json')
    if recovery['status'] != 'complete':
        raise ValueError('Recovery path generation must be complete before continuation')
    for case in protocol['cases'] + recovery['cases']:
        for name, expected in case['input_hashes'].items():
            if digest(Path(case['template']['folder']) / name) != expected:
                raise ValueError('Registered case input changed: ' + case['id'] + '/' + name)
    records, rejected = [], {}
    for ledger in sorted((source / 'ledger').glob('*.json')):
        row = read(ledger)
        if row['status'] != 'complete':
            rejected[ledger.stem] = dict(status=row['status'], cpu_seconds=row['cpu_seconds'])
            continue
        name = ledger.stem
        if row['name'] != name or row['returncode'] != 0:
            raise ValueError('Successful ledger identity mismatch: ' + name)
        job_file = source / 'jobs' / (name + '.json')
        job = read(job_file)
        usage_file = source / 'jobs' / (name + '.usage.json')
        usage = read(usage_file)
        metadata = source / 'numerical' / (name + '.json')
        predictions = source / 'numerical' / (name + '.npz')
        meta = read(metadata)
        if job['name'] != name or not predictions.samefile(Path(job['target']).with_suffix('.npz')):
            raise ValueError('Worker output identity mismatch: ' + name)
        if (usage['status'] != 'complete' or usage['source_sha256'] != protocol['region_sources']
                or job['region_sources'] != protocol['region_sources']):
            raise ValueError('Worker scientific source mismatch: ' + name)
        if (meta['candidate'] != job['candidate'] or meta['trials'] != job['expected_trials']
                or digest(predictions) != meta['predictions_sha256']):
            raise ValueError('Worker metadata or prediction digest mismatch: ' + name)
        if job['phase'] == 'autonomous':
            if job['particles'] != 0 or meta['paths'] != len(job['seeds']):
                raise ValueError('Autonomous path count mismatch: ' + name)
        elif meta['particles'] != job['particles'] or meta['seeds'] != job['seeds']:
            raise ValueError('Filter particle or seed metadata mismatch: ' + name)
        with np.load(predictions, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
            keys = ('choices', 'feedback') if job['phase'] == 'autonomous' else (
                'observed_probability', 'correct_probability')
            for key in keys:
                values = arrays[key]
                if values.shape != (len(job['seeds']), job['expected_trials']) or not np.isfinite(values).all():
                    raise ValueError('Worker array shape or finite-value check failed: ' + name)
                if job['phase'] != 'autonomous' and not ((values > 0) & (values <= 1)).all():
                    raise ValueError('Worker probability bounds failed: ' + name)
        files = [ledger, job_file, usage_file, metadata, predictions, source / 'jobs' / (name + '.log')]
        records.append(dict(name=name, stage=row['stage'], cpu_seconds=row['cpu_seconds'],
                            files={str(f.relative_to(source)): digest(f) for f in files}))
    return protocol, records, rejected


def prepare_continuation(source: Path, output: Path, checkout: Path, allowance: dict) -> dict:
    """Create a separate run with unchanged scientific policy and reused arrays.

    Stage allowances specify additional CPU hours. The unmodified runner counts
    reused ledger CPU, so its caps include that exact offset. New run CPU minus
    the recorded offset gives the additional worker and coordinator consumption.
    """
    source, output, checkout = source.resolve(), output.resolve(), checkout.resolve()
    if output.exists():
        raise FileExistsError('Use a new continuation output directory')
    protocol, records, rejected = verify_source(source, checkout)
    stages = protocol['policy']['budget']['stage_cpu_hours']
    extra = allowance['stage_cpu_hours']
    if set(extra) != set(stages):
        raise ValueError('Additional allowances must specify every registered budget stage')
    wall = float(allowance['wall_hours'])
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError('Additional wall hours must be finite and positive')
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in extra.values()) or sum(extra.values()) <= 0:
        raise ValueError('Additional CPU allowances must be finite, nonnegative and nonzero in total')
    offsets = {stage: sum(r['cpu_seconds'] for r in records if r['stage'] == stage) / 3600 for stage in stages}
    if any(r['stage'] not in stages for r in records):
        raise ValueError('Worker ledger contains an unregistered budget stage')
    changed = deepcopy(protocol)
    budget = changed['policy']['budget']
    budget['wall_hours'] = wall
    budget['stage_cpu_hours'] = {stage: offsets[stage] + float(extra[stage]) for stage in stages}
    budget['cpu_hours'] = sum(budget['stage_cpu_hours'].values())
    science = deepcopy(changed)
    science['policy']['budget'] = protocol['policy']['budget']
    assert science == protocol
    manifest = dict(created_unix=time.time(), source=str(source), checkout=str(checkout),
                    original_protocol_sha256=digest(source / 'protocol.json'),
                    original_completion_sha256=digest(source / 'completion.json'),
                    additional_allowance=allowance, reused_cpu_hours_by_stage=offsets,
                    reused_cpu_hours=sum(offsets.values()), reused_jobs=len(records),
                    excluded_noncomplete_jobs=rejected, records=records,
                    scientific_policy_unchanged=True, recovery_truth_content_inspected=False,
                    accounting='additional CPU = continuation completion.cpu_hours - reused_cpu_hours; '
                               'add original run consumption once when reporting the combined study',
                    retention='Numerical outputs and worker receipts link to retained source artifacts; '
                              'source directory must remain available',
                    command=['python', '-m', 'src.Bayesian_state.workflows.analysis.model_1024_execution_comparison',
                             '--phase', 'run', '--output', str(output)])
    output.mkdir(parents=True)
    for directory in ('jobs', 'ledger', 'numerical'):
        (output / directory).mkdir()
    for record in records:
        for relative in record['files']:
            target = output / relative
            if target.parent.name == 'ledger':
                shutil.copy2(source / relative, target)
            else:
                target.symlink_to(source / relative)
    # Preserve recovery data and truth identities without opening truth content.
    for name in ('recovery_cases.json', 'recovery_truths.json'):
        shutil.copy2(source / name, output / name)
    for name in ('recovery_truth_generation.json', 'recovery_truth_generation.completion.json'):
        path = source / 'batches' / name
        if path.exists():
            (output / 'batches').mkdir(exist_ok=True)
            shutil.copy2(path, output / 'batches' / name)
    if (source / 'source').exists():
        shutil.copytree(source / 'source', output / 'source')
    write_new(output / 'protocol.json', changed)
    write_new(output / 'protocol_digest.json', dict(sha256=digest(output / 'protocol.json')))
    with (output / 'registered_config.yaml').open('x') as stream:
        yaml.safe_dump(changed['policy'], stream, allow_unicode=True, sort_keys=False)
    for name in ('recovery_cases.json', 'recovery_truths.json'):
        if digest(output / name) != digest(source / name):
            raise ValueError('Recovery identity changed during copy')
    write_new(output / 'continuation_registration.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--budget', type=Path, required=True,
                        help='YAML with wall_hours and additional stage_cpu_hours')
    args = parser.parse_args()
    result = prepare_continuation(args.source, args.output, args.checkout, yaml.safe_load(args.budget.read_text()))
    print(json.dumps({k: result[k] for k in ('reused_jobs', 'reused_cpu_hours', 'command')}, indent=2))


if __name__ == '__main__':
    main()
