"""Test existing online H evidence as choice belief at fixed RETAIN points.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Reuse the baseline banks
of a completed continuous-recall probe, and generate the declared same seeds
under online evidence belief. One disabled path per point must reproduce every
baseline array exactly. The smoke uses one point, one seed and sixteen trials.
Parameters and behavioral nominations stay frozen; oral is excluded.
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
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/online_evidence_belief.py'


def helpers():
    # Workers are invoked with run_path inside an archived package namespace.
    # Load orchestration by path before the existing worker installs that core.
    path = ROOT / 'src/Bayesian_state/workflows/analysis/model_0923_continuous_recall_probe.py'
    spec = importlib.util.spec_from_file_location('online_evidence_probe_base', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.helpers()


def worker(file: Path) -> None:
    retention = helpers()
    recalled = retention.helpers()
    retention.helpers = lambda: recalled
    original = recalled.refit.autonomous
    v = recalled.base.validation()

    def generate(job, workflow):
        module = recalled.base.load(
            'src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief', MECHANISM)
        cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
        with module.online_evidence_belief(cls, enabled=job['online_belief']) as events:
            original(job, workflow)
        target = Path(job['target'])
        row = v.read(target.with_suffix('.json'))
        if job['online_belief']:
            expected = np.tile(np.arange(1, row['trials']), row['paths'])
            np.testing.assert_array_equal([e['trial'] for e in events], expected)
            if not all(e['evidence_matches_retained_state'] for e in events):
                raise ValueError('Online belief used evidence inconsistent with H state')
        else:
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
        v.write(target.with_suffix('.online.json'), dict(status='complete',
            enabled=job['online_belief'], reconstructions=len(events),
            nonentry_reconstructions=sum(e['newcomer_count'] == 0 for e in events),
            completed_history_timing_exact=True, evidence_matches_retained_state=True,
            exact_baseline=not job['online_belief']))

    recalled.refit.autonomous = generate
    retention.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    from .model_0923_reentry_finalist_generation import checked_chunk
    if output.exists():
        raise FileExistsError('Use a new online-evidence output')
    settings = yaml.safe_load(config.read_text())
    if (type(settings['paths']) is not int or settings['paths'] < 2
            or not 1 <= settings['parallel_budget'] <= 128):
        raise ValueError('Require a bounded paired-path protocol')
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
    generation = v.read(bind(source / 'generation.json'))
    if (done['status'] != 'complete' or done['smoke'] or done['nominees_changed']
            or done['protocol_sha256'] != v.digest(source / 'protocol.json')
            or generation['rows'] != original['banks']):
        raise ValueError('Require a complete unchanged continuous-recall comparison')
    for field in ('source_sha256', 'input_sha256'):
        for name, sha in original[field].items():
            if v.digest(bind(ROOT / name)) != sha:
                raise ValueError('Original implementation or input changed')
    selected = [bank for bank in generation['rows'] if bank['schedule'] == 'off']
    if not selected or any(not row['reused'] for row in selected):
        raise ValueError('Completed source lacks its original baseline banks')
    hashes = dict(original['source_sha256'])
    hashes.update({name: v.digest(ROOT / name) for name in (ENTRY, MECHANISM)})
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
            raise ValueError('Insufficient original paired baseline seeds')
        tag = {key: bank[key] for key in ('subject', 'model', 'candidate')}
        banks.append(dict(tag, schedule='off', reused=True, chunks=baselines))
        for enabled in (False, True):
            chunks = []
            for index, baseline in enumerate(baselines if enabled else baselines[:1]):
                job = deepcopy(baseline)
                name = f"{'online' if enabled else 'off'}_S{bank['subject']}_{bank['model']}_{bank['candidate']['id']}_C{index:03d}"
                job.update(name=name, target=str((output / 'chunks' / name).resolve()),
                           online_belief=enabled, smoke=smoke, source_sha256=hashes,
                           baseline_arrays=str(Path(baseline['target']).with_suffix('.npz')),
                           baseline_arrays_sha256=v.digest(Path(baseline['target']).with_suffix('.npz')))
                if not enabled:
                    job['seeds'] = job['seeds'][:1]
                chunks.append(job)
                jobs.append(job)
            if enabled:
                banks.append(dict(tag, schedule='online', reused=False, chunks=chunks))
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
        sidecar = v.read(Path(job['target']).with_suffix('.online.json'))
        if sidecar['status'] != 'complete' or not sidecar['evidence_matches_retained_state']:
            raise ValueError('Incomplete online evidence audit')
    for path, sha in inputs.items():
        if v.digest(Path(path)) != sha:
            raise ValueError('Bound source changed during execution')
    v.write(output / 'generation.json', dict(status='complete', rows=banks))
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        generated_paths=sum(len(job['seeds']) for job in jobs if job['online_belief']),
        replayed_baseline_paths=sum(len(job['seeds']) for job in jobs if not job['online_belief']),
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
