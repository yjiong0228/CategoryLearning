"""Passively record one target rule in original COMPACT confirmation filters.

Internal --worker JOB entry used by joint recovery. Predictive and choice-
filtered states both precede the current feedback. The same original filter
produces behavior probabilities and target-state arrays; capture adds no fits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
DRIVER = 'src/Bayesian_state/workflows/analysis/model_1021_simplification.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
COLLECTOR = 'src/Bayesian_state/workflows/analysis/model_0923_target_state_recovery.py'


def worker(file: Path) -> None:
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('compact_recovery_capture_driver', ROOT/DRIVER)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    read, write, digest = driver.io.read, driver.io.write, driver.io.digest
    job = read(file)
    expected = {name: digest(ROOT/name) for name in (ENTRY, PF, COLLECTOR)}
    if (job['phase'] != 'training' or job['model'] != 'COMPACT' or len(job['seeds']) != 1
            or job.get('target_capture_sources') != expected):
        raise ValueError('Require one original COMPACT confirmation filter and bound capture sources')
    from src.Bayesian_state.inference.backends import particle_filter as archived
    observer = driver.load('compact_target_recovery_observer', COLLECTOR)
    backend = driver.load('src.Bayesian_state.inference.backends._compact_target_capture', PF)
    meta = read(Path(job['folder'])/'prepared.json')
    original_filter = archived.run_state_model_particle_filter
    captured = {}

    def observe(**kwargs):
        if (captured or kwargs['filter_seed'] != job['seeds'][0] or kwargs['particle_count'] != job['particles']
                or kwargs['subject_id'] != meta['subject'] or kwargs['condition'] != meta['condition']):
            raise ValueError('Target capture differs from original confirmation identity')
        collector = observer.TargetCollector(job['target_rule'], len(kwargs['choices']))
        result = backend.run_state_model_particle_filter(**kwargs, state_observer=collector)
        prediction = result.observation_probabilities['prior_t'].copy()
        choices = np.asarray(kwargs['choices'], dtype=int)
        captured.update(arrays=collector.arrays(), prediction=prediction,
                        observed_probability=prediction[np.arange(len(choices)), choices-1])
        return result

    archived.run_state_model_particle_filter = observe
    try:
        driver.worker(file)
    finally:
        archived.run_state_model_particle_filter = original_filter
    if not captured:
        raise ValueError('Original confirmation filter was not captured')
    score = Path(job['target']).with_suffix('.npz')
    with np.load(score) as saved:
        np.testing.assert_array_equal(captured['observed_probability'], saved['observed_probability'][0])
        np.testing.assert_array_equal(saved['seeds'], job['seeds'])
    target = Path(job['state_capture_target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **captured['arrays'], seeds=job['seeds'])
    write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'], condition=meta['condition'],
        model=job['model'], candidate=job['candidate'], target_rule=job['target_rule'], particles=job['particles'],
        seeds=job['seeds'], trials=len(captured['prediction']), smoke=job['smoke'], additional_filters=0,
        exact_original_filter_capture=True, array_sha256=digest(target.with_suffix('.npz')),
        timing='Predictive and choice-filtered; both precede current feedback',
        original_job=str(file.resolve()), original_job_sha256=digest(file),
        score_file=str(score.resolve()), score_sha256=digest(score), capture_source_sha256=expected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    worker(parser.parse_args().worker)
