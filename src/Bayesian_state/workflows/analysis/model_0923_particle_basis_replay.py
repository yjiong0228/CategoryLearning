"""Add particle-level decision records to one completed, frozen state replay.

Run --source-job ORIGINAL_STATE_JOB --output NEW_DIRECTORY. The source state
arrays must already exist. The exact archived-runtime worker, parameters,
seed, particle count and trial limit are retained. A process-local loader
extension targets only the original decision collector; all established arrays
must reproduce the source bit for bit. This single-filter diagnostic does not
constitute a complete cohort oral evaluation or fit new parameters.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COLLECTOR = 'src/Bayesian_state/evaluation/oral/decision_basis.py'
PARTICLE = 'src/Bayesian_state/evaluation/oral/particle_basis.py'
ADAPTERS = {
    'development_cohort_belief_refit': 'model_0923_cohort_belief_states.py',
    'development_cohort_dominant_execution_refit': 'model_0923_cohort_execution_states.py',
    'development_separate_belief_refit': 'model_0923_separate_belief_refit_states.py',
    'development_dominant_execution_refit': 'model_0923_dominant_execution_refit_states.py',
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def particle_collector_loading(factory):
    """Extend only the dynamically loaded collector in this isolated worker."""
    original = importlib.util.spec_from_file_location
    installed = []

    def spec_from_file_location(name, location=None, *args, **kwargs):
        spec = original(name, location, *args, **kwargs)
        if location is not None and Path(location).resolve() == ROOT / COLLECTOR:
            execute = spec.loader.exec_module

            def execute_collector(module):
                execute(module)
                module.DecisionBasisCollector = factory(module.DecisionBasisCollector)
                installed.append(name)

            spec.loader.exec_module = execute_collector
        return spec

    importlib.util.spec_from_file_location = spec_from_file_location
    try:
        yield installed
    finally:
        importlib.util.spec_from_file_location = original


def worker(file: Path) -> None:
    job = read(file)
    source_file = Path(job['particle_basis_source_job'])
    if digest(source_file) != job['particle_basis_source_job_sha256']:
        raise ValueError('Original state job changed')
    original = read(source_file)
    additions = ('particle_basis_source_job', 'particle_basis_source_job_sha256', 'particle_basis_inputs_sha256')
    restored = {key: value for key, value in job.items() if key not in additions}
    restored['target'], restored['replay_source_sha256'] = original['target'], original['replay_source_sha256']
    if restored != original or job['target'] == original['target']:
        raise ValueError('Particle capture changed the original state experiment')
    expected_hashes = dict(original['replay_source_sha256'], **{path: digest(ROOT / path) for path in (ENTRY, PARTICLE)})
    if job['replay_source_sha256'] != expected_hashes:
        raise ValueError('Particle collector source bindings differ')
    for path, expected in job['particle_basis_inputs_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Particle replay source changed: ' + path)
    adapter = ADAPTERS.get(job.get('evaluation_use'))
    if adapter is None:
        raise ValueError('Unsupported original state worker')
    extension = load_module('particle_basis_extension', ROOT / PARTICLE)
    source_arrays = Path(original['target']).with_suffix('.npz')
    source_record = read(Path(original['target']).with_suffix('.json'))
    if digest(source_arrays) != source_record['array_sha256']:
        raise ValueError('Original decision arrays changed')
    with particle_collector_loading(extension.with_particle_basis) as installed:
        module = load_module('src.Bayesian_state.workflows.analysis._particle_basis_original_worker',
                             Path(__file__).with_name(adapter))
        module.worker(file)
    if len(installed) != 1:
        raise ValueError('Require exactly one instrumented decision collector')
    target = Path(job['target'])
    record = read(target.with_suffix('.json'))
    keys = ('subject', 'condition', 'arm', 'trials', 'training_trials', 'candidate', 'particles', 'seeds',
            'timing', 'data_use', 'evaluation_use', 'structural_intervention', 'belief_intervention', 'execution_policy')
    if any(record.get(key) != source_record.get(key) for key in keys):
        raise ValueError('Particle capture changed the source state identity')
    original_elements = 0
    with np.load(source_arrays) as before, np.load(target.with_suffix('.npz')) as after:
        for key in before.files:
            np.testing.assert_array_equal(after[key], before[key], err_msg='Changed established state array: ' + key)
            original_elements += before[key].size
        extra = [key for key in after.files if key not in before.files]
        if not extra or any(not key.startswith(extension.PREFIX) for key in extra):
            raise ValueError('Unexpected particle snapshot schema')
        if int(after[extension.PREFIX + 'format_version']) != extension.VERSION:
            raise ValueError('Unexpected particle snapshot version')
        ids = after[extension.PREFIX + 'rule_ids']
        if ids.shape[:2] != (record['trials'], record['particles']):
            raise ValueError('Particle snapshots lost trial or particle identity')
        width = ids.shape[-1]
    for path, expected in job['particle_basis_inputs_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Particle replay input changed during execution')
    write(target.with_suffix('.particle_basis.json'), dict(status='complete', schema_version=extension.VERSION,
        original_elements_exact=original_elements, added_arrays=extra, max_active_slots=width,
        original_state_arrays=str(source_arrays), original_state_sha256=digest(source_arrays),
        array_sha256=digest(target.with_suffix('.npz')), collector_loads=installed,
        timing='after_choice_before_current_feedback; predictive_weights_and_response_conditioning_retained',
        data_use='behavior_only; no_oral_input', scope='single_original_filter_decision_snapshot'))


def run(source_job: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new particle-basis replay directory')
    source_job = source_job.resolve()
    original = read(source_job)
    if original.get('evaluation_use') not in ADAPTERS:
        raise ValueError('Unsupported original state worker')
    target = Path(original['target'])
    record = read(target.with_suffix('.json'))
    if (record.get('status') != 'complete' or record.get('exact_behavior_replay') is not True
            or record.get('data_use') != 'behavior_only; no_oral_input'
            or record.get('timing') != 'after_choice_before_current_feedback'
            or digest(target.with_suffix('.npz')) != record['array_sha256']):
        raise ValueError('Require a complete, exact, behavior-only source replay')
    paths = [source_job, target.with_suffix('.json'), target.with_suffix('.npz'), ROOT / ENTRY, ROOT / PARTICLE]
    inputs = {str(path.resolve()): digest(path) for path in paths}
    job = dict(original, target=str((output / 'states' / source_job.stem).resolve()),
        replay_source_sha256=dict(original['replay_source_sha256'], **{path: digest(ROOT / path) for path in (ENTRY, PARTICLE)}),
        particle_basis_source_job=str(source_job), particle_basis_source_job_sha256=digest(source_job),
        particle_basis_inputs_sha256=inputs)
    (output / 'jobs').mkdir(parents=True)
    file = output / 'jobs' / source_job.name
    write(file, job)
    launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                'sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")')
    with tempfile.TemporaryDirectory(prefix='particle_basis_replay_') as cache:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=cache, NUMBA_CACHE_DIR=cache)
        with (output / 'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-c', launcher, original['runtime'], str(ROOT / ENTRY),
                            '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    completed = read(Path(job['target']).with_suffix('.particle_basis.json'))
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError('Source changed during particle replay')
    result = dict(status='complete', source_job=str(source_job), inputs_sha256=inputs,
        worker=completed, scientific_scope='single_filter_readout_diagnostic',
        parameters_changed=False, oral_used=False, original_trial_limit=original['trial_limit'])
    write(output / 'completion.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-job', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.source_job and args.output:
        done = run(args.source_job, args.output)
        print({key: done['worker'][key] for key in ('status', 'original_elements_exact', 'max_active_slots')})
    else:
        parser.error('Use --source-job and a new --output')
