"""Evaluate frozen simplification fits and passively capture oral decision bases.

Use --fit FIT --output NEW. Every original evaluation filter records its joint
rule/component masses before current feedback. No additional filters or oral
inputs enter this capture. The archived cognitive worker remains unchanged.
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
DRIVER = 'src/Bayesian_state/workflows/analysis/model_1021_simplification.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
COLLECTOR = 'src/Bayesian_state/evaluation/oral/decision_basis.py'
spec = importlib.util.spec_from_file_location('compact_capture_driver', ROOT / DRIVER)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
read, write, digest = driver.io.read, driver.io.write, driver.io.digest


def sources() -> dict:
    return {name: digest(ROOT / name) for name in (ENTRY, DRIVER, PF, COLLECTOR)}


def worker(file: Path) -> None:
    job = read(file)
    if job['phase'] == 'autonomous':
        driver.worker(file)
        return
    if (job['phase'] != 'evaluate' or len(job['seeds']) != 1
            or job.get('decision_basis_sources') != sources()):
        raise ValueError('Capture requires one declared original evaluation filter')
    from src.Bayesian_state.inference.backends import particle_filter as archived
    observer = driver.load('src.Bayesian_state.evaluation.oral._compact_capture', COLLECTOR)
    backend = driver.load('src.Bayesian_state.inference.backends._compact_capture', PF)
    from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
    from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    meta = read(Path(job['folder']) / 'prepared.json')
    space = build_continuous_hypothesis_space(4, 2) if meta['condition'] == 1 else build_relational_0923_space()
    original_filter = archived.run_state_model_particle_filter
    captured = {}

    def observe(**kwargs):
        if (captured or kwargs['filter_seed'] != job['seeds'][0] or kwargs['particle_count'] != job['particles']
                or kwargs['subject_id'] != meta['subject'] or kwargs['condition'] != meta['condition']):
            raise ValueError('Capture changed the frozen evaluation identity')
        collector = observer.DecisionBasisCollector(space, kwargs['choices'])
        result = backend.run_state_model_particle_filter(**kwargs, state_observer=collector, observe_decision_basis=True)
        arrays = collector.arrays()
        predicted = result.observation_probabilities['prior_t']
        error = float(np.max(np.abs(arrays['prediction'] - predicted)))
        np.testing.assert_allclose(arrays['prediction'], predicted, atol=1e-14, rtol=0)
        arrays['prediction'] = predicted.copy()
        np.testing.assert_allclose(arrays['choice_component_joint_mass'].sum((1, 2)),
            predicted[np.arange(len(arrays['choices'])), arrays['choices'] - 1], atol=1e-12, rtol=0)
        captured.update(arrays=arrays, aggregation_roundoff_max=error)
        return result

    archived.run_state_model_particle_filter = observe
    try:
        driver.worker(file)
    finally:
        archived.run_state_model_particle_filter = original_filter
    if not captured or job['decision_basis_sources'] != sources():
        raise ValueError('Incomplete capture or changing capture source')
    score = Path(job['target']).with_suffix('.npz')
    row = read(score.with_suffix('.json'))
    arrays = captured['arrays']
    with np.load(score) as saved:
        np.testing.assert_array_equal(arrays['prediction'], saved['probabilities'][0])
        np.testing.assert_array_equal(arrays['choices'], saved['choices'])
    target = Path(job['decision_basis_target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays, seeds=job['seeds'])
    write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'], condition=meta['condition'],
        model=job['model'], arm=job['model'], candidate=job['candidate'], particles=job['particles'], seeds=job['seeds'],
        trials=len(arrays['choices']), training_trials=row['training_trials'], smoke=job['smoke'],
        timing='after_choice_before_current_feedback', data_use='behavior_only; no_oral_input',
        exact_behavior_replay=False, exact_original_filter_capture=True, additional_filters=0,
        catalogue_version=space.version, catalogue_signature=space.signature,
        component_policy='nearest_internal_stimulus_uniform_ties',
        aggregation_roundoff_max=captured['aggregation_roundoff_max'],
        array_sha256=digest(target.with_suffix('.npz')), original_job=str(file.resolve()),
        original_job_sha256=digest(file), score_file=str(score), score_sha256=digest(score),
        capture_source_sha256=job['decision_basis_sources']))


def evaluate(fit: Path, output: Path) -> None:
    driver.configure()
    original_execute = driver.execute
    # The numerical evaluator supplies the original frozen fitting hashes.
    # Only its worker launcher is routed through this observation adapter.
    driver.ENTRY = ENTRY

    def execute(jobs, folder, budget):
        write(folder / 'decision_basis_capture_protocol.json', dict(source_sha256=sources(),
            additional_filters=0, scientific_settings_changed=False,
            timing='after_choice_before_current_feedback'))
        directory = folder / 'state_capture'
        directory.mkdir()
        tagged = []
        for job in jobs:
            row = deepcopy(job)
            if row['phase'] == 'evaluate':
                row.update(decision_basis_target=str((directory / row['name']).resolve()),
                           decision_basis_sources=sources())
            tagged.append(row)
        original_execute(tagged, folder, budget)

    driver.base.execute = execute
    driver.base.evaluate(fit, output)


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
