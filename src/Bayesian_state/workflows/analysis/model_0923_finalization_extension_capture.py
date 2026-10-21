"""Save oral decision bases inside the original extension evaluation filters.

Use --fit FIT --output NEW. The frozen extension evaluation schedules its
original filters and autonomous paths. Each filter passively records the
complete DecisionBasisCollector arrays before current feedback. Parameters,
seeds, particles, trial order and cognitive interventions stay with the
original worker. No additional filters are executed for state collection.
The capture protocol is separate from the already frozen fitting sources.
Use model_0923_finalization_extension_captured_states to validate and align
the resulting arrays after the original behavior summary is complete.
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
DRIVER = 'src/Bayesian_state/workflows/analysis/model_0923_finalization_extension_fit.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
COLLECTOR = 'src/Bayesian_state/evaluation/oral/decision_basis.py'
CAPTURE_KEYS = ('decision_basis_target', 'decision_basis_sources')
spec = importlib.util.spec_from_file_location('extension_capture_driver', ROOT / DRIVER)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
read, write, digest = driver.read, driver.write, driver.digest


def sources() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in (ENTRY, DRIVER, PF, COLLECTOR)}


def check_capture_job(job: dict) -> None:
    driver.check_job(job)
    if (job.get('phase') != 'evaluate' or len(job['seeds']) != 1
            or not job.get('decision_basis_target') or job.get('decision_basis_sources') != sources()
            or job.get('online_belief') is not True or job.get('retain_precision') is not True):
        raise ValueError('Require one unchanged ONLINE/RETAIN evaluation filter and bound capture sources')


def configure(settings: dict, inputs: dict, config: dict):
    # Changing the launcher entry only routes workers through passive capture.
    # Evaluation continues to bind the original frozen fitting source hashes.
    driver.ENTRY = ENTRY
    module = driver.configure(settings, inputs, config)
    original_execute = module.base.execute

    def execute(jobs, output, budget):
        tagged = []
        own = sources()
        write(output / 'decision_basis_capture_protocol.json', dict(
            status='declared', source_sha256=own, additional_filters=0,
            timing='after_choice_before_current_feedback',
            scope='All original evaluation filters; autonomous paths unchanged',
            scientific_settings_changed=False))
        directory = output / 'state_capture'
        directory.mkdir()
        for job in jobs:
            row = deepcopy(job)
            if row['phase'] == 'evaluate':
                if len(row['seeds']) != 1:
                    raise ValueError('Capture requires one original evaluation filter per job')
                row.update(decision_basis_target=str((directory / row['name']).resolve()),
                           decision_basis_sources=own)
            tagged.append(row)
        original_execute(tagged, output, budget)

    module.base.execute = execute
    return module


def worker(file: Path) -> None:
    job = read(file)
    driver.check_job(job)
    if job['phase'] == 'autonomous':
        if any(key in job for key in CAPTURE_KEYS):
            raise ValueError('Autonomous jobs cannot contain a decision-basis capture')
        driver.worker(file)
        return
    check_capture_job(job)
    from src.Bayesian_state.inference.backends import particle_filter as archived
    observer = driver.load_module('src.Bayesian_state.evaluation.oral._extension_basis_capture', ROOT / COLLECTOR)
    backend = driver.load_module('src.Bayesian_state.inference.backends._extension_basis_capture', ROOT / PF)
    from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
    from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    meta = read(Path(job['folder']) / 'prepared.json')
    space = build_continuous_hypothesis_space(4, 2) if meta['condition'] == 1 else build_relational_0923_space()
    calls, captured = [], {}
    original_filter = archived.run_state_model_particle_filter

    def observe(**kwargs):
        if (calls or kwargs['filter_seed'] != job['seeds'][0]
                or kwargs['particle_count'] != job['particles']
                or kwargs['subject_id'] != meta['subject'] or kwargs['condition'] != meta['condition']):
            raise ValueError('Capture changed the original evaluation identity')
        calls.append(kwargs['filter_seed'])
        collector = observer.DecisionBasisCollector(space, kwargs['choices'])
        result = backend.run_state_model_particle_filter(
            **kwargs, state_observer=collector, observe_decision_basis=True)
        arrays = collector.arrays()
        predicted = result.observation_probabilities['prior_t']
        roundoff = float(np.max(np.abs(arrays['prediction'] - predicted)))
        np.testing.assert_allclose(arrays['prediction'], predicted, atol=1e-14, rtol=0)
        arrays['prediction'] = predicted.copy()
        np.testing.assert_allclose(arrays['choice_component_joint_mass'].sum(axis=(1, 2)),
            predicted[np.arange(len(arrays['choices'])), arrays['choices'] - 1], atol=1e-12, rtol=0)
        captured.update(arrays=arrays, aggregation_roundoff_max=roundoff)
        return result

    archived.run_state_model_particle_filter = observe
    try:
        driver.worker(file)
    finally:
        archived.run_state_model_particle_filter = original_filter
    check_capture_job(job)
    if calls != job['seeds']:
        raise ValueError('Original evaluation filter was not captured exactly once')
    score = Path(job['target']).with_suffix('.npz')
    record = read(score.with_suffix('.json'))
    precision, belief = [read(score.with_suffix(s)) for s in ('.precision.json', '.online.json')]
    arrays = captured['arrays']
    with np.load(score) as saved:
        np.testing.assert_array_equal(arrays['prediction'], saved['probabilities'][0])
        np.testing.assert_array_equal(arrays['choices'], saved['choices'])
        np.testing.assert_array_equal(job['seeds'], saved['seeds'])
    if (record['predictions_sha256'] != digest(score)
            or not precision['retained_precision'] or not belief['evidence_matches_retained_state']
            or not belief['exact_transition_timing']):
        raise ValueError('Captured filter lost its original score or cognitive interventions')
    target = Path(job['decision_basis_target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays, seeds=job['seeds'])
    write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'],
        condition=meta['condition'], arm=job['model'], particles=job['particles'], seeds=job['seeds'],
        trials=len(arrays['choices']), training_trials=record['training_trials'], candidate=job['candidate'],
        exact_behavior_replay=False, exact_original_filter_capture=True, additional_filters=0,
        aggregation_roundoff_max=captured['aggregation_roundoff_max'],
        catalogue_version=space.version, catalogue_signature=space.signature,
        component_policy='nearest_internal_stimulus_uniform_ties', timing='after_choice_before_current_feedback',
        data_use='behavior_only; no_oral_input', parameter_estimation_scope='prefix',
        evaluation_use='registered_extension_validation', structural_intervention='retained_rule_precision',
        particle_transitions=precision['transitions'], online_belief=True,
        online_reconstructions=belief['reconstructions'], online_evidence_matches_retained_state=True,
        numeric_pools=record['numeric_pools'], array_sha256=digest(target.with_suffix('.npz')),
        original_job=str(file.resolve()), original_job_sha256=digest(file),
        score_file=str(score), score_sha256=digest(score), capture_source_sha256=job['decision_basis_sources']))


def evaluate(fit: Path, output: Path) -> None:
    settings, inputs, config = driver.restore(fit)
    configure(settings, inputs, config).base.evaluate(fit, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.fit and args.output:
        evaluate(args.fit, args.output)
    else:
        parser.error('Use --fit and --output, or an internal --worker')
