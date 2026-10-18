"""Extend online evidence belief to completed RETAIN development path banks.

Run --config YAML --output NEW_DIRECTORY [--smoke]. Each declared subject keeps
all original parameters and the declared first seeds. Retained-precision paths
are reused as baseline. One disabled path per subject is replayed exactly;
the new online variant uses the same archived cognitive runtime and precision
storage wrapper. --smoke uses one seed per subject, sixteen trials and one job
at a time. No oral reports, reserved participants or parameter search are used.
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
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_reentry_precision_probe.py'
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/online_evidence_belief.py'


def helpers():
    spec = importlib.util.spec_from_file_location('online_development_retention_base', ROOT / BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    precision = helpers()
    recalled = precision.helpers()
    precision.helpers = lambda: recalled
    v = recalled.base.validation()
    original = recalled.refit.autonomous

    def generate(job, workflow):
        module = recalled.base.load(
            'src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief', MECHANISM)
        cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
        with module.online_evidence_belief(cls, enabled=job['online_belief']) as events:
            original(job, workflow)
        target = Path(job['target'])
        row = v.read(target.with_suffix('.json'))
        if job['online_belief']:
            np.testing.assert_array_equal([e['trial'] for e in events],
                np.tile(np.arange(1, row['trials']), row['paths']))
        else:
            reference = Path(job['baseline_arrays'])
            if v.digest(reference) != job['baseline_arrays_sha256']:
                raise ValueError('Retained-precision baseline changed')
            with np.load(target.with_suffix('.npz')) as a, np.load(reference) as b:
                indices = [int(np.flatnonzero(b['seeds'] == seed).item()) for seed in job['seeds']]
                n = row['trials']
                for key in a.files:
                    expected = b[key][:n] if key == 'categories' else b[key][indices] if key == 'seeds' else b[key][indices,:n]
                    np.testing.assert_array_equal(a[key], expected)
        v.write(target.with_suffix('.online.json'), dict(status='complete', enabled=job['online_belief'],
            reconstructions=len(events), nonentry_reconstructions=sum(e['newcomer_count'] == 0 for e in events),
            completed_history_timing_exact=True, evidence_matches_retained_state=True,
            exact_retained_baseline=not job['online_belief']))

    recalled.refit.autonomous = generate
    precision.worker(file)


def checked_source(job: dict, v) -> dict:
    """Require the actual completed retained-precision path and mechanism record."""
    target = Path(job['target'])
    record = v.read(target.with_suffix('.json'))
    retained = v.read(target.with_suffix('.precision.json'))
    if (job['smoke'] or job['phase'] != 'autonomous' or job['model'] != 'SEARCH'
            or not job['retain_precision'] or not retained['retained_precision']
            or retained['status'] != 'complete' or record['candidate'] != job['candidate']
            or record['paths'] != len(job['seeds'])
            or record['model'] != 'SEARCH' or record['predictions_sha256'] != v.digest(target.with_suffix('.npz'))
            or retained['arrays_sha256'] != v.digest(target.with_suffix('.precision.npz'))):
        raise ValueError('Incomplete or inconsistent retained-precision baseline')
    with np.load(target.with_suffix('.npz')) as arrays:
        np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
    return record


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new online-evidence development output')
    settings = yaml.safe_load(config.read_text())
    if (not settings['subjects'] or type(settings['paths']) is not int or settings['paths'] < 3
            or not 1 <= settings['parallel_budget'] <= 128):
        raise ValueError('Require declared subjects and a bounded path budget')
    precision = helpers()
    recalled = precision.helpers()
    v = recalled.base.validation()
    source = ROOT / settings['source']
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = v.digest(path)
        return path

    bind(config)
    protocol = v.read(bind(source / 'protocol.json'))
    completion = v.read(bind(source / 'completion.json'))
    if completion['status'] != 'complete' or completion['smoke'] or protocol['smoke']:
        raise ValueError('Require the completed scientific RETAIN development study')
    groups = {row['subject']:row for row in protocol['groups']}
    if (len(settings['subjects']) != len(set(settings['subjects']))
            or not set(settings['subjects']) <= set(groups)):
        raise ValueError('Use only subjects already present in the completed development study')
    hashes = dict(protocol['source_sha256'])
    for field in ('input_sha256', 'source_sha256'):
        for path, sha in protocol[field].items():
            if v.digest(bind(ROOT / path)) != sha:
                raise ValueError('Existing development source changed')
    hashes.update({path:v.digest(ROOT / path) for path in (ENTRY, BASE, MECHANISM)})
    jobs, banks = [], []
    for subject in settings['subjects']:
        group = groups[subject]
        subdone = v.read(bind(source / f'subject_{subject}' / 'completion.json'))
        expected = next(r for r in completion['groups'] if r['subject'] == subject)
        if expected['completion_sha256'] != v.digest(source / f'subject_{subject}' / 'completion.json'):
            raise ValueError('Subject completion record changed')
        if subdone['status'] != 'complete' or subdone['smoke']:
            raise ValueError('Subject generation is incomplete')
        left = 1 if smoke else settings['paths']
        baselines = []
        for file in sorted((source / 'jobs').glob(f'RETAIN_S{subject}_*.json')):
            if left == 0:
                break
            original = v.read(bind(file))
            record = checked_source(original, v)
            if record['subject'] != subject or record['candidate'] != group['candidate']:
                raise ValueError('Baseline subject or parameter provenance differs')
            for suffix in ('.json', '.npz', '.precision.json', '.precision.npz', '.experiment.json'):
                bind(Path(original['target']).with_suffix(suffix))
            job = deepcopy(original)
            job['seeds'] = job['seeds'][:left]
            left -= len(job['seeds'])
            baselines.append(job)
        if left or not baselines:
            raise ValueError('Insufficient original development seeds')
        tag = dict(subject=subject, model='SEARCH', candidate=group['candidate'],
                   parameter_provenance=group['parameter_provenance'])
        banks.append(dict(tag, schedule='off', reused=True, chunks=baselines))
        for enabled in (False, True):
            chunks = []
            for i, baseline in enumerate(baselines if enabled else baselines[:1]):
                name = f"{'online' if enabled else 'off'}_S{subject}_C{i:03d}"
                target = Path(baseline['target']).with_suffix('.npz')
                job = dict(deepcopy(baseline), name=name, online_belief=enabled, smoke=smoke,
                    source_sha256=hashes, baseline_arrays=str(target), baseline_arrays_sha256=v.digest(target),
                    target=str((output / 'chunks' / name).resolve()))
                if not enabled:
                    job['seeds'] = job['seeds'][:1]
                chunks.append(job)
                jobs.append(job)
            if enabled:
                banks.append(dict(tag, schedule='online', reused=False, chunks=chunks))
    output.mkdir(parents=True)
    v.write(output / 'protocol.json', dict(settings=protocol['settings'], probe_settings=settings,
        source=str(source.resolve()), smoke=smoke, source_sha256=hashes, input_sha256=inputs,
        banks=banks, jobs=jobs, scope='Same frozen development points and RETAIN mechanism; online evidence off/on',
        data_use='Own-choice feedback; no fitting, oral data or reserved subjects; original nominations unchanged'))
    recalled.ENTRY = ENTRY
    recalled.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    for job in jobs:
        target = Path(job['target'])
        record, side = v.read(target.with_suffix('.json')), v.read(target.with_suffix('.online.json'))
        if record['predictions_sha256'] != v.digest(target.with_suffix('.npz')) or not side['evidence_matches_retained_state']:
            raise ValueError('Incomplete online evidence generation')
        if not job['online_belief']:
            with np.load(target.with_suffix('.precision.npz')) as a, np.load(Path(job['baseline_arrays']).with_suffix('.precision.npz')) as b:
                indices = [int(np.flatnonzero(b['seeds'] == seed).item()) for seed in job['seeds']]
                for key in a.files:
                    expected = b[key][indices] if key == 'seeds' else b[key][indices,:record['trials']]
                    np.testing.assert_array_equal(a[key], expected)
    for path, sha in dict(inputs, **hashes).items():
        if v.digest(Path(path)) != sha:
            raise ValueError('Bound development input changed')
    v.write(output / 'generation.json', dict(status='complete', rows=banks))
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        new_paths=sum(len(j['seeds']) for j in jobs if j['online_belief']),
        reused_paths=sum(len(j['seeds']) for b in banks if b['reused'] for j in b['chunks']),
        exact_retained_baseline_paths=len(settings['subjects']), nominees_changed=False,
        protocol_sha256=v.digest(output / 'protocol.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config,args.output,args.smoke)
    else: parser.error('Use --config and --output, or an internal --worker')
