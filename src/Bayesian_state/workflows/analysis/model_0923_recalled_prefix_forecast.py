"""Continue frozen POOL/SEARCH fits from behavior-conditioned prefix states.

Run --config CONFIG --output NEW_DIRECTORY; --smoke runs each configured arm
on eight training and eight future trials, with four particles and two branches.
Scientific runs preserve every saved filter seed and particle count. Future
responses and feedback are generated through the shared model lifecycle. Oral
reports are not inputs. The original evaluation, candidate and source hashes
are retained so these forecasts can be compared with task-start generation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
# Workers import the archived cognitive package. Load this current orchestration
# helper by file so an old runtime need not contain the new analysis modules.
_spec = importlib.util.spec_from_file_location('recalled_prefix_replay', ROOT /
    'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_replay.py')
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)
ENTRY = 'src/Bayesian_state/workflows/analysis/model_0923_recalled_prefix_forecast.py'
HELPER = 'src/Bayesian_state/simulation/filtered_continuation.py'
COMBINER = 'src/Bayesian_state/workflows/analysis/model_0923_h_prefix_forecast.py'


def forecast_inputs(training: dict, future: dict, smoke: bool) -> tuple[dict, dict]:
    """Expose observed responses only in the prefix, and task inputs afterward."""
    if smoke:
        if len(training['choices']) < 16:
            raise ValueError('Smoke check needs sixteen training trials')
        return ({key: training[key][:8] for key in ('stimulus', 'choices', 'feedback')},
                {key: training[key][8:16] for key in ('stimulus', 'categories')})
    return ({key: training[key] for key in ('stimulus', 'choices', 'feedback')},
            {key: future[key] for key in ('stimulus', 'categories')})


def branch_seeds(settings: dict, subject: int, filter_seed: int) -> list[int]:
    """Share future random streams across arms, preserving independent filters."""
    from src.Bayesian_state.utils.seeding import stable_seed
    return [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                             subject=int(subject), filter_seed=int(filter_seed), branch=index,
                             role='autonomous_prefix_continuation'))
            for index in range(settings['branches_per_filter'])]


def worker(file: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    job = replay.read(file)
    replay.verify_sources(job)
    refit = replay.load('recalled_prefix_refit', replay.REFIT)
    v = refit.base.validation()
    workflow = v.intervention_workflow()
    workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', refit.base.FEEDBACK)
    precision = refit.base.load('src.Bayesian_state.model.modules.precision_interventions', refit.base.PRECISION)
    guard = refit.base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', refit.GUARD)
    history = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission', refit.HELPER)
    helper = replay.load('src.Bayesian_state.simulation.filtered_continuation', HELPER)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    folder = Path(job['folder'])
    meta = replay.read(folder / 'prepared.json')
    if replay.digest(folder / 'engine.json') != meta['engine_sha256']:
        raise ValueError('Prepared engine changed')
    for name, expected in meta['frozen_cognitive_source_sha256'].items():
        if replay.digest(Path(job['runtime']) / name) != expected:
            raise ValueError(f'Archived cognitive source changed: {name}')
    engine = replay.read(folder / 'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    for name in ('training.npz', 'sealed_test.npz'):
        if replay.digest(folder / name) != meta['array_sha256'][name]:
            raise ValueError('Prepared behavior changed')
    with np.load(folder / 'training.npz') as training, np.load(folder / 'sealed_test.npz') as future:
        prefix, future = forecast_inputs(training, future, job['smoke'])
    kwargs = dict(engine_config=engine, subject_id=meta['subject'], condition=meta['condition'],
                  **prefix, particle_count=job['particles'], particle_blocks=1,
                  filter_seed=job['filter_seed'], choice_readout_power=1.,
                  processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
    start, cpu = time.monotonic(), time.process_time()
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        fixes = stack.enter_context(feedback.retention_roundoff_guard(cls))
        pairing_fixes = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        arm = job['arm_settings']
        stack.enter_context(precision.precision_intervention(BetaModule, mode=arm['precision'], upper=arm['upper']))
        recall_fixes = stack.enter_context(history.episodic_admission(cls, mode='POOL'))
        if job['model'] == 'SEARCH':
            proposal = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal', replay.PROPOSAL)
            stack.enter_context(proposal.episodic_proposal(cls, proposal=True))
        models, result = helper.capture_prefix(**kwargs)
        probabilities = result.observation_probabilities['prior_t']
        if job['smoke']:
            independent = run_state_model_particle_filter(**kwargs)
            expected = independent.observation_probabilities['prior_t']
            np.testing.assert_array_equal(result.final_weights, independent.final_weights)
        else:
            with np.load(job['reference']) as saved:
                np.testing.assert_array_equal(saved['seeds'], [job['filter_seed']])
                np.testing.assert_array_equal(saved['choices'][:len(prefix['choices'])], prefix['choices'])
                expected = saved['probabilities'][0, :len(prefix['choices'])]
        np.testing.assert_array_equal(probabilities, expected, err_msg='Prefix differs from unmodified filter')
        seeds = branch_seeds(job['settings'], meta['subject'], job['filter_seed'])
        generate = dict(models=models, weights=result.final_weights, observed_choices=prefix['choices'],
                        observed_feedback=prefix['feedback'], **future, condition=meta['condition'])
        arrays = helper.generate_continuations(**generate, branch_seeds=seeds)
        if job['smoke']:
            # Reversing branch order tests restoration of episodic history,
            # proposal caches and random streams, including shared ancestors.
            again = helper.generate_continuations(**generate, branch_seeds=list(reversed(seeds)))
            for key, values in arrays.items():
                other = again[key] if key in ('terminal_weights', 'categories') else again[key][::-1]
                np.testing.assert_array_equal(values, other, err_msg=f'Branch order changed {key}')
    replay.verify_sources(job)
    arrays['prefix_observed_probability'] = expected[np.arange(len(prefix['choices'])), prefix['choices'] - 1]
    arrays['filter_seed'] = np.asarray(job['filter_seed'])
    target = Path(job['target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    replay.write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'], condition=meta['condition'],
        model=job['model'], candidate=job['candidate'], filter_seed=job['filter_seed'], particles=job['particles'],
        paths=len(seeds), training_trials=len(prefix['choices']), forecast_trials=len(future['categories']),
        prefix_exact=True, branch_order_check=job['smoke'], reference_type='separate_filter' if job['smoke'] else 'saved_filter',
        feedback_corrections=fixes, pairing_corrections=pairing_fixes, recall_corrections=recall_fixes,
        seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, numeric_pools=threadpool_info(),
        predictions_sha256=replay.digest(target.with_suffix('.npz'))))


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, list[dict]]:
    import yaml
    settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError('Use a new prefix forecast directory')
    if (settings['sample_role'] != 'development' or set(settings['arms']) != {'POOL', 'SEARCH'}
            or any(settings[name] < 1 for name in ('workers', 'filters', 'branches_per_filter'))):
        raise ValueError('Need a positive-budget development comparison of POOL and SEARCH')
    source = ROOT / settings['source']
    original = replay.read(source / 'protocol.json')
    completed = replay.read(source / 'completion.json')
    fit = Path(original['fit'])
    nomination = replay.read(fit / 'nominees.json')
    if (completed['status'] != 'complete' or nomination['smoke']
            or nomination['status'] != 'frozen_before_test'
            or original['nomination_sha256'] != replay.digest(fit / 'nominees.json')
            or completed['nomination_sha256'] != original['nomination_sha256']):
        raise ValueError('Need completed predictions from frozen scientific nominees')
    nominees = {(row['subject'], row['model']): row['candidate'] for row in nomination['nominees']}
    hashes = {name: replay.digest(ROOT / name) for name in
              (ENTRY, HELPER, COMBINER, replay.ENTRY, replay.REFIT, replay.PROPOSAL)}
    jobs = []
    effective = dict(settings)
    if smoke:
        effective.update(filters=1, branches_per_filter=2, workers=1)
    for subject in settings['subjects']:
        for arm in settings['arms']:
            files = sorted((source / 'jobs').glob(f'evaluate_S{subject}_{arm}_*_C*.json'))
            if len(files) != settings['filters']:
                raise ValueError(f'Expected all original filter seeds: {subject} {arm}')
            for file in files[:effective['filters']]:
                job = replay.read(file)
                if job['candidate'] != nominees[subject, arm] or len(job['seeds']) != 1:
                    raise ValueError('Saved worker differs from the frozen nominee or seed protocol')
                reference = Path(job['target']).with_suffix('.npz')
                if replay.digest(reference) != replay.read(reference.with_suffix('.json'))['predictions_sha256']:
                    raise ValueError('Original probability reference changed')
                job.update(smoke=smoke, settings=effective, filter_seed=job['seeds'][0],
                    particles=4 if smoke else job['particles'], reference=str(reference),
                    reference_sha256=replay.digest(reference), original_job=str(file.resolve()),
                    original_job_sha256=replay.digest(file), replay_source_sha256=hashes,
                    archived_pf_sha256=replay.digest(Path(job['runtime']) / replay.PF),
                    target=str((output / 'chunks' / job['name']).resolve()))
                replay.verify_sources(job)
                jobs.append(job)
    workers = min(effective['workers'], len(os.sched_getaffinity(0)), len(jobs))
    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    for relative in hashes:
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(config, output / 'protocol.yaml')
    protocol = dict(settings=effective, smoke=smoke, workers=workers, jobs=len(jobs),
        source_sha256=hashes, config_sha256=replay.digest(config),
        input_sha256={str(source / name): replay.digest(source / name) for name in ('protocol.json', 'completion.json')},
        fit=str(fit), evaluation=str(source.resolve()), nomination_sha256=original['nomination_sha256'],
        python=sys.version, versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        data_use_status='development', future_inputs='recorded stimulus schedule and task categories',
        conditioning='observed training prefix; own choices and task feedback throughout future',
        uncertainty_unit='independent filter seed; branches share the terminal particle distribution')
    replay.write(output / 'forecast_protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool) -> None:
    protocol, jobs = prepare(config, output, smoke)
    print(f"Running {len(jobs)} filters with {protocol['workers']} workers; one numeric thread each", flush=True)

    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json')
        replay.write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='recalled_prefix_') as cache, file.with_suffix('.log').open('x') as stream:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                            '--worker', str(file.resolve())], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for index, future in enumerate(as_completed([pool.submit(launch, job) for job in jobs]), 1):
            future.result()
            print(f'{index}/{len(jobs)} prefix forecasts complete', flush=True)
    combine = replay.load('recalled_prefix_combiner', COMBINER).combine
    groups = {}
    for job in jobs:
        subject = replay.read(Path(job['folder']) / 'prepared.json')['subject']
        groups.setdefault((subject, job['model']), []).append(job)
    rows = [combine(replay, group, output / 'forecast' / f'subject_{subject}' / arm / group[0]['candidate']['id'])
            for (subject, arm), group in groups.items()]
    replay.write(output / 'forecast.json', dict(status='complete', smoke=smoke, rows=rows,
        seconds=time.monotonic()-start, filters=len(jobs), paths=sum(row['paths'] for row in rows)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
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
