"""Capture belief, execution, error sources and oral bases in one behavior filter.

The internal --worker JOB entry accepts an execution-study scoring job with
one seed, capture_sources, target_rules and state_capture_target. Setting
capture_oral retains the shared local oral basis. original_score optionally
requires exact replay of a saved behavioral filter, including its seed.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
STUDY = 'src/Bayesian_state/workflows/analysis/model_1023_execution_study.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
COLLECTOR = 'src/Bayesian_state/evaluation/execution_states.py'
ORAL = 'src/Bayesian_state/evaluation/oral/decision_basis.py'
spec = importlib.util.spec_from_file_location('execution_capture_study', ROOT / STUDY)
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)
read, write, digest = study.read, study.write, study.digest


def sources() -> dict:
    return {p: digest(ROOT / p) for p in (ENTRY, STUDY, PF, COLLECTOR, ORAL)}


def worker(file: Path) -> None:
    job = read(file)
    if job['capture_sources'] != sources() or len(job['seeds']) != 1:
        raise ValueError('Require one registered capture filter')
    from src.Bayesian_state.inference.backends import particle_filter as archived
    backend = study.base.load('src.Bayesian_state.inference.backends._execution_capture', PF)
    observer = study.base.load('src.Bayesian_state.evaluation._execution_capture', COLLECTOR)
    meta = read(Path(job['folder']) / 'prepared.json')
    files = ['training.npz'] + (['sealed_test.npz'] if job['phase'] == 'evaluate' else [])
    categories = []
    for name in files:
        with np.load(Path(job['folder']) / name) as arrays:
            categories.append(arrays['categories'])
    categories = np.concatenate(categories)
    original_filter, captured = archived.run_state_model_particle_filter, {}

    def observe(**kwargs):
        if (captured or kwargs['filter_seed'] != job['seeds'][0] or kwargs['particle_count'] != job['particles']
                or kwargs['subject_id'] != meta['subject'] or kwargs['condition'] != meta['condition']):
            raise ValueError('Capture changed filter identity')
        collector = observer.ExecutionStateCollector(kwargs['choices'], categories[:len(kwargs['choices'])],
                                                       job['target_rules'])
        oral = None
        if job.get('capture_oral', False):
            from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
            from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
            oral_module = study.base.load('src.Bayesian_state.evaluation.oral._execution_capture', ORAL)
            space = build_continuous_hypothesis_space(4, 2) if meta['condition'] == 1 else build_relational_0923_space()
            oral = oral_module.DecisionBasisCollector(space, kwargs['choices'])

        def collect(trial, phase, snapshot):
            collector(trial, phase, snapshot)
            if oral is not None:
                oral(trial, phase, snapshot)

        result = backend.run_state_model_particle_filter(**kwargs, state_observer=collect, observe_decision_basis=True)
        arrays = collector.arrays()
        predicted = result.observation_probabilities['prior_t']
        np.testing.assert_allclose(arrays['prediction'], predicted, atol=1e-14, rtol=0)
        captured['observer_prediction_roundoff'] = dict(
            execution=float(np.max(np.abs(arrays['prediction']-predicted))))
        arrays['prediction'] = predicted.copy()
        if oral is not None:
            oral_arrays = oral.arrays()
            np.testing.assert_allclose(oral_arrays['prediction'], predicted, atol=1e-14, rtol=0)
            captured['observer_prediction_roundoff']['oral'] = float(
                np.max(np.abs(oral_arrays['prediction']-predicted)))
            # The observer repeats the weighted sum; the filter additionally
            # normalizes its prediction. Bind the saved oral prediction to
            # that exact filter output after checking roundoff equivalence.
            # Joint component masses keep their original accumulation.
            oral_arrays['prediction'] = predicted.copy()
            arrays.update({'oral_' + k: v for k, v in oral_arrays.items()})
            captured['catalogue'] = dict(catalogue_version=oral.space.version,
                                        catalogue_signature=oral.space.signature)
        captured.update(arrays=arrays)
        return result

    archived.run_state_model_particle_filter = observe
    try:
        study.worker(file)
    finally:
        archived.run_state_model_particle_filter = original_filter
    if not captured or job['capture_sources'] != sources():
        raise ValueError('Incomplete or changing capture')
    arrays = captured['arrays']
    score = Path(job['target']).with_suffix('.npz')
    with np.load(score) as saved:
        observed = arrays['prediction'][np.arange(len(arrays['choices'])), arrays['choices'] - 1]
        np.testing.assert_array_equal(observed, saved['observed_probability'][0])
    replay = job.get('original_score')
    if replay:
        reference = Path(replay)
        if digest(reference) != job['original_score_sha256']:
            raise ValueError('Original behavioral score changed')
        with np.load(reference) as saved:
            index, = np.flatnonzero(saved['seeds'] == job['seeds'][0])
            np.testing.assert_array_equal(observed, saved['observed_probability'][index])
            correct = arrays['prediction'][np.arange(len(arrays['categories'])), arrays['categories'] - 1]
            np.testing.assert_array_equal(correct, saved['correct_probability'][index])
            if 'probabilities' in saved:
                np.testing.assert_array_equal(arrays['prediction'], saved['probabilities'][index])
                np.testing.assert_array_equal(arrays['choices'], saved['choices'])
                np.testing.assert_array_equal(arrays['categories'], saved['categories'])
    target = Path(job['state_capture_target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays, seeds=job['seeds'])
    write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'], condition=meta['condition'],
        model=job['execution_mode'], candidate=job['candidate'], particles=job['particles'],
        seeds=job['seeds'], target_rules=job['target_rules'], trials=len(arrays['choices']), smoke=job['smoke'],
        training_trials=meta['training_trials'], original_job=str(file.resolve()),
        original_job_sha256=digest(file), filters_run=1,
        timing='predictive and choice-conditioned filtered; before current feedback; no future choices',
        source_order=['target_absent', 'target_available_other_rule', 'target_rule'],
        state_interpretation='latent state distributions; MIX execution columns are rule contributions',
        capture_oral=job.get('capture_oral', False), oral_used_for_inference=False,
        original_score=replay, exact_saved_filter_replay=bool(replay),
        array_sha256=digest(target.with_suffix('.npz')), score_sha256=digest(score),
        observer_prediction_roundoff=captured['observer_prediction_roundoff'],
        capture_sources=sources(), study_sources=job['source_sha256'], **captured.get('catalogue', {})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    worker(parser.parse_args().worker)
