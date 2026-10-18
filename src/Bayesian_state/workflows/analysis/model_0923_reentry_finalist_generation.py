"""Generate the remaining behavior-selected RETAIN finalists with paired seeds.

Reuse every completed autonomous chunk for the frozen nominee and run only the
other frozen confirmation finalists. Candidate banks and nominees stay fixed.
This is parameter-point sensitivity within RETAIN, not a posterior sample or a
new selection criterion. The original archived runtime and precision-retention
worker execute all new paths. Full runs inherit path counts and chunk sizes from
the fit; --smoke runs one other finalist for one seed and 16 trials on one worker.
An existing evaluation may still be filtering, provided its autonomous chunks
are complete. Output: protocol.json, jobs/, generation.json and completion.json.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import model_0923_reentry_precision_refit as retention
from ...utils.seeding import stable_seed

ENTRY = str(Path(__file__).resolve().relative_to(retention.ROOT))


def checked_chunk(job: dict, v) -> dict:
    """Require an executed RETAIN chunk with exactly the declared paths."""
    target = Path(job['target'])
    row = v.read(target.with_suffix('.json'))
    precision = v.read(target.with_suffix('.precision.json'))
    if (job.get('retain_precision') is not True
            or job.get('structural_intervention') != retention.INTERVENTION
            or precision.get('retained_precision') is not True
            or precision.get('structural_intervention') != retention.INTERVENTION
            or precision.get('event_unit') != 'autonomous path transitions'):
        raise ValueError('Require actual retained-precision autonomous execution')
    if (row['candidate'] != job['candidate'] or row['model'] != job['model']
            or row['paths'] != len(job['seeds'])
            or row['role'] != 'autonomous_from_start'
            or row['predictions_sha256'] != v.digest(target.with_suffix('.npz'))):
        raise ValueError('Completed chunk does not match its job')
    if precision['transitions'] != row['paths'] * row['trials']:
        raise ValueError('Incomplete precision-retention execution')
    if any(pool['num_threads'] != 1 for pool in row['numeric_pools']):
        raise ValueError('Require one numeric thread per worker')
    with np.load(target.with_suffix('.npz')) as arrays:
        np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
        if arrays['choices'].shape != (row['paths'], row['trials']):
            raise ValueError('Autonomous shape differs from metadata')
    return row


def plan(fit: Path, evaluation: Path, output: Path, v, smoke: bool) -> tuple:
    """Bind completed inputs and prepare jobs without changing either source."""
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = v.digest(path)
        return path

    frozen = v.read(bind(fit / 'nominees.json'))
    done = v.read(bind(fit / 'completion.json'))
    original = v.read(bind(evaluation / 'protocol.json'))
    if (done['status'] != 'complete' or done['smoke'] or done['test_choices_used']
            or frozen['smoke'] or frozen['status'] != 'frozen_before_test'
            or done['nomination_sha256'] != v.digest(fit / 'nominees.json')
            or original['nomination_sha256'] != done['nomination_sha256']
            or Path(original['fit']).resolve() != fit.resolve()
            or original['source_sha256'] != frozen['source_sha256']):
        raise ValueError('Require unchanged completed behavioral fit and its evaluation')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(bind(retention.ROOT / name)) != sha:
            raise ValueError('Fitted implementation changed')
    settings = frozen['settings']
    jobs, banks = [], []
    for nominee in frozen['nominees']:
        subject, arm = nominee['subject'], nominee['model']
        finalists = nominee['finalists']
        ids = [candidate['id'] for candidate in finalists]
        if len(ids) < 2 or len(ids) != len(set(ids)) or nominee['candidate'] not in finalists:
            raise ValueError('Require distinct frozen finalists including the nominee')
        expected = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                    role='autonomous', subject=subject, path=i))
                    for i in range(settings['autonomous_paths'])]
        old = [job for job in original['jobs'] if job['phase'] == 'autonomous'
               and job['model'] == arm
               and v.read(Path(job['folder']) / 'prepared.json')['subject'] == subject]
        old.sort(key=lambda job: expected.index(job['seeds'][0]))
        if [seed for job in old for seed in job['seeds']] != expected:
            raise ValueError('Primary trajectory seeds differ from the frozen schedule')
        templates = []
        for job in old:
            actual = v.read(bind(evaluation / 'jobs' / (job['name'] + '.json')))
            if (any(actual[key] != value for key, value in job.items())
                    or job['candidate'] != nominee['candidate']
                    or actual['source_sha256'] != frozen['source_sha256']):
                raise ValueError('Executed nominee job differs from the evaluation plan')
            checked_chunk(actual, v)
            for suffix in ('.json', '.npz', '.precision.json', '.experiment.json'):
                bind(Path(actual['target']).with_suffix(suffix))
            templates.append(actual)
        folder = Path(templates[0]['folder'])
        meta = v.read(bind(folder / 'prepared.json'))
        for name in ('training.npz', 'sealed_test.npz'):
            if v.digest(bind(folder / name)) != meta['array_sha256'][name]:
                raise ValueError('Observed task arrays changed')
        for candidate in finalists:
            bind(candidate['score_file'])
            selected = candidate['id'] == nominee['candidate']['id']
            if selected and smoke:
                continue
            chunks = []
            for index, template in enumerate(templates[:1] if smoke else templates):
                job = deepcopy(template)
                if not selected:
                    name = f"FINALIST_S{subject}_{arm}_{candidate['id']}_C{index:03d}"
                    job.update(name=name, candidate=deepcopy(candidate), smoke=smoke,
                               target=str((output / 'chunks' / name).resolve()))
                    job['source_sha256'][ENTRY] = v.digest(Path(__file__))
                    if smoke:
                        job['seeds'] = job['seeds'][:1]
                    jobs.append(job)
                chunks.append(job)
            banks.append(dict(subject=subject, model=arm, candidate=candidate,
                              frozen_nominee=selected, reused=selected, chunks=chunks))
            if smoke:
                return frozen, inputs, jobs, banks
    return frozen, inputs, jobs, banks


def run(fit: Path, evaluation: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new finalist generation output')
    recalled = retention.helpers()
    v = recalled.base.validation()
    frozen, inputs, jobs, banks = plan(fit, evaluation, output, v, smoke)
    references = retention.reference_inputs(recalled, frozen['settings'])
    inputs.update(references)
    retention.configure(recalled, references)
    output.mkdir(parents=True)
    v.write(output / 'protocol.json', dict(
        fit=str(fit.resolve()), evaluation=str(evaluation.resolve()),
        settings=frozen['settings'], smoke=smoke, input_sha256=inputs,
        source_sha256={ENTRY: v.digest(Path(__file__)), **frozen['source_sha256']},
        jobs=jobs, banks=banks, role='frozen_behavior_finalist_parameter_sensitivity',
        selection='original training-choice nominations unchanged; oral excluded',
        data_use='development', uncertainty='candidate points are not posterior draws'))
    recalled.base.execute(jobs, output, 1 if smoke else frozen['settings']['parallel_budget'])
    rows = []
    for bank in banks:
        for job in bank['chunks']:
            checked_chunk(job, v)
        row = {key: value for key, value in bank.items() if key != 'chunks'}
        row.update(chunk_files=[job['target'] for job in bank['chunks']],
                   seeds=[seed for job in bank['chunks'] for seed in job['seeds']])
        rows.append(row)
    for name, sha in inputs.items():
        if v.digest(Path(name)) != sha:
            raise ValueError('Bound input changed during generation')
    v.write(output / 'generation.json', dict(status='complete', rows=rows))
    v.write(output / 'completion.json', dict(
        status='complete', smoke=smoke, candidate_points=len(banks),
        generated_paths=sum(len(job['seeds']) for job in jobs),
        reused_paths=sum(len(row['seeds']) for row in rows if row['reused']),
        nominees_changed=False, protocol_sha256=v.digest(output / 'protocol.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.fit, args.evaluation, args.output, args.smoke)


if __name__ == '__main__':
    main()
