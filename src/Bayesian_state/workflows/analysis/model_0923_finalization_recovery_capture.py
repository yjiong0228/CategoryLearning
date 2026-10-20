"""Record target states during the bounded ONLINE recovery confirmation filters.

Use --phase search|score-truth --prepared PREP --subject ID --replicate INDEX
--output NEW [--fit FIT] [--smoke]. Preparation uses the existing recovery entry.
Search policy, numerical budgets and worker cognition are delegated unchanged.
Each confirmation filter also records the target rule before current feedback,
before and after current-choice conditioning. This avoids a second execution
of the selected and generating points solely to recover their state arrays.
All confirmation alternatives are captured; nomination uses behavior alone.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
DRIVER = 'src/Bayesian_state/workflows/analysis/model_0923_finalization_recovery.py'
OBSERVER = 'src/Bayesian_state/workflows/analysis/model_0923_target_state_recovery.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
spec = importlib.util.spec_from_file_location('finalization_recovery_driver_capture', ROOT / DRIVER)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
read, write, digest = driver.read, driver.write, driver.digest


def target_from_protocol(settings: dict, inputs: dict) -> int:
    """Use the registered target identity without reading generated states."""
    cases = []
    for path in inputs:
        if Path(path).name != 'protocol.json':
            continue
        protocol = read(Path(path))
        if protocol.get('config', {}).get('scope') not in driver.SCOPES:
            continue
        cases.extend(c for c in protocol['cases']
                     if {k: c[k] for k in ('subject', 'replicate')} == settings['recovery_case'])
    if len(cases) != 1 or type(cases[0]['target_rule']) is not int or cases[0]['target_rule'] < 0:
        raise ValueError('Require exactly one registered target rule for this recovery case')
    return cases[0]['target_rule']


original_configure = driver.configure


def configure(settings: dict, inputs: dict):
    target = target_from_protocol(settings, inputs)
    own = dict(inputs)
    for name in (ENTRY, DRIVER, OBSERVER, PF):
        own[str(ROOT / name)] = digest(ROOT / name)
    module = original_configure(settings, own)
    old_execute, old_hashes, old_archive = module.base.execute, module.base.source_hashes, module.base.archive

    def hashes(v):
        return {**old_hashes(v), **{name: digest(ROOT / name) for name in (ENTRY, DRIVER, OBSERVER, PF)}}

    def execute(jobs, output, budget):
        tagged = []
        for job in jobs:
            row = deepcopy(job)
            if row['name'].startswith('confirmation_'):
                if len(row['seeds']) != 1:
                    raise ValueError('Capture requires one complete filter per confirmation job')
                directory = output / 'state_capture'
                directory.mkdir(exist_ok=True)
                row.update(state_capture_target=str((directory / row['name']).resolve()),
                           state_capture_rule=target)
            tagged.append(row)
        old_execute(tagged, output, budget)

    def archive(output, config, sources):
        old_archive(output, config, sources)
        write(output / 'state_capture_protocol.json', dict(target_rule=target,
            scope='All original confirmation alternatives; target-state observation only',
            nomination='Behavioral likelihood only', additional_filters=0,
            timing='Predictive and choice-filtered states, both before current feedback',
            inputs_sha256=own))

    module.base.execute, module.base.source_hashes, module.base.archive = execute, hashes, archive
    return module


def worker(file: Path) -> None:
    job = read(file)
    driver.check_job(job)
    if not job['name'].startswith('confirmation_'):
        if 'state_capture_target' in job:
            raise ValueError('State capture is restricted to confirmation filters')
        driver.worker(file)
        return
    if len(job['seeds']) != 1 or 'state_capture_target' not in job:
        raise ValueError('Missing confirmation capture declaration')
    from src.Bayesian_state.inference.backends import particle_filter as archived
    observer = driver.load_module('finalization_recovery_target_observer', OBSERVER)
    capture = driver.load_module('src.Bayesian_state.inference.backends._finalization_recovery_capture', PF)
    trials = 8 if job['smoke'] else read(Path(job['folder']) / 'prepared.json')['training_trials']
    collector = observer.TargetCollector(job['state_capture_rule'], trials)
    original_filter = archived.run_state_model_particle_filter
    calls = []

    def observe(**kwargs):
        if calls or kwargs['filter_seed'] != job['seeds'][0]:
            raise ValueError('Capture changed the declared confirmation filter sequence')
        calls.append(kwargs['filter_seed'])
        return capture.run_state_model_particle_filter(**kwargs, state_observer=collector)

    archived.run_state_model_particle_filter = observe
    try:
        driver.worker(file)
    finally:
        archived.run_state_model_particle_filter = original_filter
    if calls != job['seeds']:
        raise ValueError('Confirmation worker did not execute its declared filter')
    target = Path(job['state_capture_target'])
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **collector.arrays(), seeds=job['seeds'])
    score = Path(job['target']).with_suffix('.npz')
    write(target.with_suffix('.json'), dict(status='complete', target_rule=job['state_capture_rule'],
        trials=trials, particles=job['particles'], seeds=job['seeds'], candidate=job['candidate'],
        belief_mechanism=job['cohort_belief_mechanism'], precision_storage='RETAIN',
        array_sha256=digest(target.with_suffix('.npz')), score_file=str(score), score_sha256=digest(score),
        original_job=str(file.resolve()), original_job_sha256=digest(file),
        timing='Predictive and choice-filtered states, both before current feedback', additional_filters=0))


driver.configure = configure
driver.ENTRY = ENTRY


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('search', 'score-truth'))
    for name in ('prepared', 'fit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--subject', type=int)
    parser.add_argument('--replicate', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.prepared and args.output and args.subject is not None and args.replicate is not None:
        if args.phase == 'search':
            driver.search(args.prepared, args.subject, args.replicate, args.output, args.smoke)
        elif args.phase == 'score-truth' and args.fit:
            driver.score_truth(args.prepared, args.subject, args.replicate, args.fit, args.output, args.smoke)
        else:
            parser.error('Select search or score-truth; truth scoring needs --fit')
    else:
        parser.error('Provide prepared data, case identity and a new output directory')
