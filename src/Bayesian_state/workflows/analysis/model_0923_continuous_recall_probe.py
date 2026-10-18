"""Probe entry-only versus every-trial recalled belief at fixed RETAIN points.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Baseline autonomous paths
are reused; one path per point is also replayed with recall off and must match
every saved array exactly. New arms share those seeds and parameters. The
scientific output is a fixed-parameter development screen, with no oral input
or parameter selection. --smoke keeps one point, one seed, sixteen trials and
one worker. Mechanisms execute inside the archived cognitive runtime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/continuous_recalled_belief.py'
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_reentry_precision_refit.py'
CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'
GLOBAL = 'engine.modules.hypo_transitions_mod.kwargs.search_controller.global_search'


def helpers():
    spec = importlib.util.spec_from_file_location('continuous_recall_retained_base', ROOT / BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    retention = helpers()
    recalled = retention.helpers()
    retention.helpers = lambda: recalled
    original = recalled.refit.autonomous
    v = recalled.base.validation()

    def generate(job, workflow):
        module = recalled.base.load(
            'src.Bayesian_state.model.modules.hypothesis_transition.continuous_recalled_belief', MECHANISM)
        cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
        with module.continuous_recalled_belief(cls, schedule=job['belief_schedule']) as events:
            original(job, workflow)
        target = Path(job['target'])
        row = v.read(target.with_suffix('.json'))
        if any(event['belief_recalled_episodes'] != event['trial'] for event in events):
            raise ValueError('Belief reconstruction used an incorrect completed history')
        if job['belief_schedule'] == 'every_trial' and len(events) != row['paths'] * (row['trials'] - 1):
            raise ValueError('Every-trial belief omitted a completed feedback step')
        if job['belief_schedule'] == 'entry' and any(event['newcomer_count'] != 1 for event in events):
            raise ValueError('Entry-only reconstruction ran without admission')
        if job['belief_schedule'] == 'off':
            reference = Path(job['baseline_arrays'])
            if v.digest(reference) != job['baseline_arrays_sha256']:
                raise ValueError('Baseline replay source changed')
            with np.load(target.with_suffix('.npz')) as current, np.load(reference) as previous:
                indices = [int(np.flatnonzero(previous['seeds'] == seed).item()) for seed in job['seeds']]
                n = row['trials']
                for key in current.files:
                    expected = (previous[key][:n] if key == 'categories' else previous[key][indices]
                                if key == 'seeds' else previous[key][indices, :n])
                    np.testing.assert_array_equal(current[key], expected)
        v.write(target.with_suffix('.continuous.json'), dict(status='complete',
            schedule=job['belief_schedule'], reconstructions=len(events),
            nonentry_reconstructions=sum(event['newcomer_count'] == 0 for event in events),
            completed_history_timing_exact=True, exact_baseline=job['belief_schedule'] == 'off'))

    recalled.refit.autonomous = generate
    retention.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    from .model_0923_reentry_finalist_generation import checked_chunk
    if output.exists():
        raise FileExistsError('Use a new continuous-recall output')
    settings = yaml.safe_load(config.read_text())
    if (settings['schedules'] != ['entry', 'every_trial'] or type(settings['paths']) is not int
            or settings['paths'] < 2 or not 1 <= settings['parallel_budget'] <= 128):
        raise ValueError('Require the bounded paired entry/every-trial protocol')
    retention = helpers()
    recalled = retention.helpers()
    v = recalled.base.validation()
    source = ROOT / settings['source']
    inputs = {str(config.resolve()): v.digest(config)}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = v.digest(path)
        return path

    original = v.read(bind(source / 'protocol.json'))
    done = v.read(bind(source / 'completion.json'))
    if (done['status'] != 'complete' or done['smoke'] or done['nominees_changed']
            or done['protocol_sha256'] != v.digest(source / 'protocol.json')):
        raise ValueError('Require a complete unchanged fixed-parameter crossing')
    for field in ('source_sha256', 'input_sha256'):
        for name, sha in original[field].items():
            if v.digest(bind(ROOT / name)) != sha:
                raise ValueError('Original implementation or input changed')
    selected = [bank for bank in original['banks']
                if bank['model'] in settings['models']
                and bank['candidate']['hyperparams'][CAPACITY] in settings['capacities']
                and bank['candidate']['hyperparams'][GLOBAL] == settings['global_proposal']]
    if len(selected) != len(settings['models']) * len(settings['capacities']):
        raise ValueError('Incomplete declared capacity/proposal baseline cells')
    hashes = dict(original['source_sha256'])
    hashes.update({name: v.digest(ROOT / name) for name in (ENTRY, MECHANISM, BASE)})
    jobs, banks = [], []
    for bank in selected[:1] if smoke else selected:
        left = 1 if smoke else settings['paths']
        baselines = []
        for original_job in bank['chunks']:
            if left == 0:
                break
            checked_chunk(original_job, v)
            job = deepcopy(original_job)
            job['seeds'] = job['seeds'][:left]
            left -= len(job['seeds'])
            for suffix in ('.json', '.npz', '.precision.json', '.experiment.json'):
                bind(Path(job['target']).with_suffix(suffix))
            baselines.append(job)
        if left:
            raise ValueError('Insufficient completed baseline seeds')
        tag = {key: bank[key] for key in ('subject', 'model', 'candidate')}
        banks.append(dict(tag, schedule='off', reused=True, chunks=baselines))
        for schedule in ['off', *settings['schedules']]:
            chunks = []
            for index, baseline in enumerate(baselines[:1] if schedule == 'off' else baselines):
                job = deepcopy(baseline)
                name = f"{schedule}_S{bank['subject']}_{bank['model']}_{bank['candidate']['id']}_C{index:03d}"
                job.update(name=name, target=str((output / 'chunks' / name).resolve()),
                           belief_schedule=schedule, smoke=smoke, source_sha256=hashes,
                           baseline_arrays=str(Path(baseline['target']).with_suffix('.npz')),
                           baseline_arrays_sha256=v.digest(Path(baseline['target']).with_suffix('.npz')))
                if schedule == 'off':
                    job['seeds'] = job['seeds'][:1]
                chunks.append(job)
                jobs.append(job)
            if schedule != 'off':
                banks.append(dict(tag, schedule=schedule, reused=False, chunks=chunks))
    references = retention.reference_inputs(recalled, original['settings'])
    inputs.update(references)
    retention.configure(recalled, references)
    recalled.ENTRY = ENTRY
    output.mkdir(parents=True)
    v.write(output / 'protocol.json', dict(settings=original['settings'], probe_settings=settings,
        source=str(source.resolve()), smoke=smoke, input_sha256=inputs, source_sha256=hashes,
        jobs=jobs, banks=banks, scope='fixed-parameter development mechanism comparison',
        data_use='own-choice autonomous feedback; no refitting; oral excluded; original nominations unchanged'))
    recalled.base.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    for job in jobs:
        checked_chunk(job, v)
        sidecar = v.read(Path(job['target']).with_suffix('.continuous.json'))
        if sidecar['status'] != 'complete' or not sidecar['completed_history_timing_exact']:
            raise ValueError('Incomplete mechanism execution audit')
    for path, sha in inputs.items():
        if v.digest(Path(path)) != sha:
            raise ValueError('Bound source changed during execution')
    v.write(output / 'generation.json', dict(status='complete', rows=banks))
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        generated_paths=sum(len(job['seeds']) for job in jobs if job['belief_schedule'] != 'off'),
        replayed_baseline_paths=sum(len(job['seeds']) for job in jobs if job['belief_schedule'] == 'off'),
        reused_paths=sum(len(job['seeds']) for bank in banks if bank['reused'] for job in bank['chunks']),
        nominees_changed=False, protocol_sha256=v.digest(output / 'protocol.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Use --config and a new --output, or an internal --worker job')
