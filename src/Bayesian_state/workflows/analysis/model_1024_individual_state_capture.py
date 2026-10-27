"""Capture all active states under the frozen Model 1024 behavioral runtime.

Internal entry: ``--worker JOB``. The observer is passive; each job verifies
the shared state aggregates and the unchanged behavioral score.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
BASE = 'src/Bayesian_state/workflows/analysis/model_1024_marginal_execution.py'
HELPER = 'src/Bayesian_state/evaluation/individual_state_inference.py'
ORAL = 'src/Bayesian_state/evaluation/oral/decision_basis.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load('individual_state_base', BASE)
capture, study = base.capture, base.study
read, write, digest = study.read, study.write, study.digest
load('src.Bayesian_state.evaluation.marginal_execution', base.HELPER)
load('src.Bayesian_state.evaluation.oral.decision_basis', ORAL)
helper = load('src.Bayesian_state.evaluation.individual_state_inference', HELPER)


def sources() -> dict:
    return dict(base.sources(), **{p: digest(ROOT/p) for p in (ENTRY, HELPER, ORAL)})


def worker(file: Path) -> None:
    job = read(file)
    if job['individual_state_sources'] != sources() or job.get('capture_oral'):
        raise ValueError('Require registered passive state capture')
    limit = int(job.get('cpu_limit_seconds', 7200))
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit+5))
    started = time.monotonic()
    original_load = capture.study.base.load
    audit = {}

    def instrument(name, path):
        module = original_load(name, path)
        if path == base.PF:
            original_filter = module.run_state_model_particle_filter

            def observe(**kwargs):
                if job['execution_mode'] == 'MIX':
                    base.validate_carrier(kwargs)
                from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
                from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
                space = build_continuous_hypothesis_space(4, 2) if kwargs['condition'] == 1 else build_relational_0923_space()
                capacity = kwargs['engine_config']['modules']['hypo_transitions_mod']['kwargs']['max_capacity']
                collector = helper.IndividualStateCollector(space, kwargs['choices'], capacity)
                existing = kwargs.pop('state_observer')

                def both(t, phase, snapshot):
                    existing(t, phase, snapshot)
                    collector(t, phase, snapshot)

                result = original_filter(**kwargs, state_observer=both)
                values = collector.arrays()
                parents = base.ancestry.resampling_parents(result.resampling_log, len(kwargs['choices']), job['particles'])
                summary = helper.summarize(values, parents, module._trace_ancestral_indices, job['target_rules'])
                target = Path(job['individual_target']); target.parent.mkdir(parents=True, exist_ok=True)
                with target.with_suffix('.npz').open('xb') as stream:
                    np.savez_compressed(stream, **{'particle_'+k: v for k, v in values.items()
                        if k not in ('filtered_weights', 'component_counts')},
                        filtered_weights=values['filtered_weights'], component_counts=values['component_counts'],
                        parents=parents, **summary)
                audit.update(trials=len(kwargs['choices']), rules=len(space), catalogue_signature=space.signature,
                    catalogue_version=space.version, minimum_particle_choice_mass=collector.minimum_choice_mass,
                    resampling_steps=int(np.count_nonzero(result.resampled)),
                    array_sha256=digest(target.with_suffix('.npz')))
                return result

            module.run_state_model_particle_filter = observe
        return module

    capture.study.base.load = instrument
    try:
        capture.worker(file)
    finally:
        capture.study.base.load = original_load
    with np.load(job['individual_target']+'.npz') as own, np.load(job['state_capture_target']+'.npz') as old:
        for key in ('active', 'belief', 'target_present', 'target_belief', 'target_high_belief'):
            np.testing.assert_allclose(own['filtered_'+key], old['filtered_'+key], atol=3e-12, rtol=0)
        np.testing.assert_allclose(own['filtered_source'], old['conditional_decision_sources'], atol=3e-12, rtol=0)
    if job['individual_state_sources'] != sources():
        raise ValueError('Capture sources changed during computation')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(Path(job['individual_target']).with_suffix('.json'), dict(status='complete', **audit,
        subject=job['subject'], candidate=job['candidate'], particles=job['particles'], seeds=job['seeds'],
        timing='after current choice, before current feedback',
        conditioning=['filtered: choices through t', 'full: complete observed sequence'],
        parameter_conditioning='one fixed candidate; candidates are kept separate',
        oral_used_for_inference=False, state_sources=sources(),
        cpu_seconds=usage.ru_utime+usage.ru_stime, wall_seconds=time.monotonic()-started,
        max_rss_kib=usage.ru_maxrss, job_sha256=digest(file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    worker(parser.parse_args().worker)
