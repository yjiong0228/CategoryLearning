"""Replay frozen admission fits and export report-independent decision bases.

Run with --config CONFIG --output NEW_DIRECTORY. --smoke uses the first subject,
arm and saved seed, sixteen trials and one worker. Original particles and seeds
are retained to check every replayed choice probability against saved results.
Workers import the archived cognitive runtime and the existing experimental
adapters; the current shared PF adds observational capture only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
ENTRY = 'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_replay.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
COLLECTOR = 'src/Bayesian_state/evaluation/oral/decision_basis.py'
REFIT = 'src/Bayesian_state/workflows/analysis/model_0923_episodic_refit.py'
PROPOSAL = 'src/Bayesian_state/model/modules/hypothesis_transition/episodic_proposal.py'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources(job: dict) -> None:
    for name, expected in {**job['source_sha256'], **job['replay_source_sha256']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Executing source changed: {name}')
    if digest(Path(job['runtime']) / PF) != job['archived_pf_sha256']:
        raise ValueError('Archived particle filter changed')
    if digest(Path(job['reference'])) != job['reference_sha256']:
        raise ValueError('Original probability reference changed')


def worker(file: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    job = read(file)
    verify_sources(job)
    refit = load('decision_basis_refit', REFIT)
    v = refit.base.validation()
    workflow = v.intervention_workflow()
    workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', refit.base.FEEDBACK)
    precision = refit.base.load('src.Bayesian_state.model.modules.precision_interventions', refit.base.PRECISION)
    guard = refit.base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', refit.GUARD)
    history = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission', refit.HELPER)
    capture = load('src.Bayesian_state.inference.backends._decision_basis_capture', PF)
    collect = load('src.Bayesian_state.evaluation.oral._decision_basis_capture', COLLECTOR)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
    from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    folder = Path(job['folder'])
    meta = read(folder / 'prepared.json')
    if digest(folder / 'engine.json') != meta['engine_sha256']:
        raise ValueError('Prepared engine changed')
    for name, expected in meta['frozen_cognitive_source_sha256'].items():
        if digest(Path(job['runtime']) / name) != expected:
            raise ValueError(f'Archived cognitive source changed: {name}')
    engine = read(folder / 'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        if digest(folder / name) != meta['array_sha256'][name]:
            raise ValueError('Prepared behavior changed')
        with np.load(folder / name) as saved:
            parts.append({key: saved[key] for key in ('stimulus', 'choices', 'feedback')})
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    n = min(job['trial_limit'], len(arrays['choices'])) if job['trial_limit'] else len(arrays['choices'])
    arrays = {key: value[:n] for key, value in arrays.items()}
    space = (build_continuous_hypothesis_space(4, 2) if meta['condition'] == 1 else build_relational_0923_space())
    with np.load(job['reference']) as saved:
        expected = saved['probabilities'][0, :n].copy()
        np.testing.assert_array_equal(saved['seeds'], job['seeds'])
        np.testing.assert_array_equal(saved['choices'][:n], arrays['choices'])
    if len(job['seeds']) != 1:
        raise ValueError('One independent filter seed per replay job')
    collector = collect.DecisionBasisCollector(space, arrays['choices'])
    start, cpu = time.monotonic(), time.process_time()
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        feedback_fixes = stack.enter_context(feedback.retention_roundoff_guard(cls))
        snapshot_fixes = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        arm = job['arm_settings']
        stack.enter_context(precision.precision_intervention(BetaModule, mode=arm['precision'], upper=arm['upper']))
        recall_fixes = []
        if job['model'] != 'BASE':
            mode = 'POOL' if job['model'] == 'SEARCH' else job['model']
            recall_fixes = stack.enter_context(history.episodic_admission(cls, mode=mode))
        if job['model'] == 'SEARCH':
            proposal = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal', PROPOSAL)
            stack.enter_context(proposal.episodic_proposal(cls, proposal=True))
        result = capture.run_state_model_particle_filter(
            engine_config=engine, subject_id=meta['subject'], condition=meta['condition'],
            stimulus=arrays['stimulus'], choices=arrays['choices'], feedback=arrays['feedback'],
            particle_count=job['particles'], particle_blocks=1, filter_seed=job['seeds'][0],
            choice_readout_power=1., processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'],
            state_observer=collector, observe_decision_basis=True)
    predictions = result.observation_probabilities['prior_t']
    np.testing.assert_array_equal(predictions, expected, err_msg='Saved behavioral replay differs')
    recorded = collector.arrays()
    # The observer sums weighted particle predictions; the filter additionally
    # normalizes that vector. Their difference is floating-point roundoff only.
    aggregation_roundoff = float(np.max(np.abs(recorded['prediction'] - predictions)))
    np.testing.assert_allclose(recorded['prediction'], predictions, atol=1e-14, rtol=0)
    recorded['prediction'] = predictions
    np.testing.assert_allclose(recorded['choice_component_joint_mass'].sum(axis=(1, 2)),
                               expected[np.arange(n), arrays['choices'] - 1], atol=1e-12, rtol=0)
    verify_sources(job)
    target = Path(job['target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **recorded, seeds=job['seeds'])
    write(target.with_suffix('.json'), dict(
        status='complete', subject=meta['subject'], condition=meta['condition'], arm=job['model'],
        particles=job['particles'], seeds=job['seeds'], trials=n, training_trials=meta['training_trials'],
        candidate=job['candidate'], exact_behavior_replay=True, aggregation_roundoff_max=aggregation_roundoff,
        catalogue_version=space.version,
        catalogue_signature=space.signature, component_policy='nearest_internal_stimulus_uniform_ties',
        timing='after_choice_before_current_feedback', data_use='behavior_only; no_oral_input',
        seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
        numeric_pools=threadpool_info(), array_sha256=digest(target.with_suffix('.npz')),
        feedback_corrections=feedback_fixes, snapshot_corrections=snapshot_fixes, recall_corrections=recall_fixes))


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, list[dict]]:
    import yaml
    settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError('Use a new decision-basis output directory')
    if settings['sample_role'] != 'development' or settings['workers'] < 1:
        raise ValueError('Expected a development protocol with positive worker budget')
    source = ROOT / settings['source']
    if read(source / 'completion.json')['status'] != 'complete':
        raise ValueError('Original evaluation must be complete')
    hashes = {name: digest(ROOT / name) for name in (ENTRY, PF, COLLECTOR, REFIT)}
    if 'SEARCH' in settings['arms']:
        hashes[PROPOSAL] = digest(ROOT / PROPOSAL)
    jobs = []
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    arms = settings['arms'][:1] if smoke else settings['arms']
    for subject in subjects:
        for arm in arms:
            files = sorted((source / 'jobs').glob(f'evaluate_S{subject}_{arm}_*_C*.json'))
            if len(files) != settings['seeds_per_arm']:
                raise ValueError(f'Expected all saved independent seeds for {subject} {arm}')
            for file in files[:1] if smoke else files:
                job = read(file)
                reference = Path(job['target']).with_suffix('.npz')
                record = read(reference.with_suffix('.json'))
                if digest(reference) != record['predictions_sha256']:
                    raise ValueError('Original evaluation array changed')
                job.update(reference=str(reference), reference_sha256=digest(reference),
                           original_job=str(file.resolve()), original_job_sha256=digest(file),
                           replay_source_sha256=hashes, archived_pf_sha256=digest(Path(job['runtime']) / PF),
                           trial_limit=16 if smoke else 0,
                           target=str((output / 'states' / job['name']).resolve()))
                verify_sources(job)
                jobs.append(job)
    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    for relative in hashes:
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(config, output / 'protocol.yaml')
    workers = min(1 if smoke else settings['workers'], len(os.sched_getaffinity(0)), len(jobs))
    protocol = dict(settings=settings, smoke=smoke, workers=workers, jobs=len(jobs),
                    source_sha256=hashes, config_sha256=digest(config),
                    python=sys.version,
                    versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
                    original_completion_sha256=digest(source / 'completion.json'),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    role='fixed_parameter_replay_no_oral_input')
    write(output / 'protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool) -> None:
    protocol, jobs = prepare(config, output, smoke)
    print(f"{len(jobs)} saved filter seeds, {protocol['workers']} workers, one numeric thread", flush=True)
    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json')
        write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='decision_basis_worker_') as cache:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                                         '--worker', str(file.resolve())], cwd=ROOT, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Replay worker failed: {file.with_suffix('.log')}")
        record = read(Path(job['target']).with_suffix('.json'))
        if digest(Path(job['target']).with_suffix('.npz')) != record['array_sha256']:
            raise ValueError('Replay artifact changed')
        return record
    records = []
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            records.append(future.result())
            print(f"{len(records)}/{len(jobs)} exact replays complete", flush=True)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(records),
          exact_behavior_replays=sum(r['exact_behavior_replay'] for r in records),
          cpu_seconds=sum(r['cpu_seconds'] for r in records),
          records=[dict(subject=r['subject'], arm=r['arm'], seed=r['seeds'][0], trials=r['trials']) for r in records]))


def main() -> None:
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
        parser.error('Supply --config and --output')


if __name__ == '__main__':
    main()
