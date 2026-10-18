"""Diagnose saved autonomous errors without changing fitted mechanisms.

Run --source COMPLETED_RECALLED_EVALUATION --output NEW_DIRECTORY. The archived
perception and rule emission code reconstruct the target rule's predictions.
One complete path per group must exactly reproduce saved choices, probabilities,
target state and reconstructed perceptions. --smoke checks two paths, 16 trials,
one group and one worker. Output errors describe fixed saved states; forcing an
existing target to supply the response is an instantaneous diagnostic, not a new
learning trajectory. Oral reports are not read. PNG plotting is separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
METRICS = 'src/Bayesian_state/evaluation/model_0923_readout_channels.py'


def helpers():
    spec = importlib.util.spec_from_file_location('readout_channel_refit', ROOT /
        'src/Bayesian_state/workflows/analysis/model_0923_recalled_search_refit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def diagnose(job: dict, workflow, recalled) -> None:
    """Reconstruct independent perception noise and use the archived emission."""
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.perception import PerceptionModule
    from src.Bayesian_state.model.modules.readout.choice import normalize_probability_vector
    from threadpoolctl import threadpool_info

    v = recalled.base.validation()
    inputs = recalled.base.load('readout_channel_inputs', recalled.refit.INPUTS)
    metrics = recalled.base.load('readout_channel_metrics', METRICS)
    engine, meta, arrays = inputs.inputs(job, v)
    source = Path(job['reference'])
    if v.digest(source) != job['reference_sha256']:
        raise ValueError('Saved autonomous reference changed')
    if v.digest(Path(job['original_job'])) != job['original_job_sha256']:
        raise ValueError('Original autonomous job changed')
    with np.load(source) as saved:
        indices = [int(np.flatnonzero(saved['seeds'] == seed)[0]) for seed in job['seeds']]
        n = 16 if job['smoke'] else len(saved['categories'])
        data = {key: saved[key][indices, :n] for key in
                ('choices', 'feedback', 'probabilities', 'correct_probability',
                 'present', 'contribution', 'beta')}
        np.testing.assert_array_equal(saved['categories'][:n], arrays['categories'][:n])
    physical, categories = arrays['stimulus'][:n], arrays['categories'][:n]
    template = StateModel(engine, context=ModelContext(subject_id=meta['subject'],
        condition=meta['condition'], processed_data_dir=meta['paths']['processed_dir'],
        dataset_paths=meta['paths']))
    readout = engine.get('choice_readout', {}).get('kwargs', {})
    if (readout.get('method') != 'expectation' or readout.get('power') != 1
            or readout.get('strategy_confidence_gain', 0) != 0
            or readout.get('rule_commitment_confidence_gain', 0) != 0
            or engine.get('output_noise', {}).get('kwargs', {}).get('enabled') is not False
            or template.engine.get_module(ModuleRole.MAPPING) is not None):
        raise ValueError('Require the saved fixed-category linear choice mixture')
    if v.digest(Path(job['audit'])) != job['audit_sha256']:
        raise ValueError('Target geometry audit changed')
    target_rule = v.read(Path(job['audit']))['target']
    q = np.full(data['present'].shape, np.nan)
    first_perceived = None
    state = np.random.get_state()
    try:
        for row, seed in enumerate(job['seeds']):
            perception = PerceptionModule(template.engine,
                **deepcopy(engine['modules']['perception_mod'].get('kwargs', {})))
            np.random.seed(int(seed))
            perceived = []
            for stimulus in physical:
                template.engine.observation = (stimulus, 1, 1.)
                perceived.append(perception.process(stimu=stimulus))
            perceived = np.asarray(perceived)
            if row == 0:
                first_perceived = perceived.copy()
            for trial in np.flatnonzero(data['present'][row]):
                probability = template.partition_model.get_category_probabilities(
                    hypo=target_rule, data=([perceived[trial]], [1], [1.]),
                    beta=float(data['beta'][row, trial]), distance_mode=template.engine.distance_mode)
                q[row, trial] = normalize_probability_vector(probability[:, 0], strict=True)[categories[trial]-1]
    finally:
        np.random.set_state(state)

    # The full first path checks that no other module consumes the perception
    # stream, including late trials after several changes in active rules.
    diagnosis = workflow.cohort().load_diagnosis()
    live = []

    class Recorder:
        next_path = 0

        def before(self, model, choice, outcome):
            h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            live.append((int(target_rule in h.active), float(h.execution_probabilities[target_rule]),
                float(model._pending_trial.beta[target_rule]) if target_rule in h.active else 0.))

        def after(self, *args):
            pass

    with diagnosis.record_lifecycle(Recorder()):
        replay = diagnosis.autonomous.run_autonomous_category_learning(engine_config=engine,
            subject_id=meta['subject'], condition=meta['condition'], stimulus=physical,
            categories=categories, trajectory_seed=job['seeds'][0],
            processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths']).trajectory
    np.testing.assert_array_equal(replay.perceived_stimulus, first_perceived)
    for name, actual in [('choices', replay.choices), ('feedback', replay.feedback),
                         ('probabilities', replay.observed_probabilities)]:
        np.testing.assert_array_equal(actual, data[name][0])
    for column, name in enumerate(('present', 'contribution', 'beta')):
        np.testing.assert_array_equal(np.asarray(live)[:, column], data[name][0])
    channels = metrics.error_channels(data['present'], data['contribution'], q, data['correct_probability'])
    target = Path(job['target'])
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **channels, target_correct=q, seeds=job['seeds'])
    rows = []
    for label, begin, end in [('whole', 0, n), ('late128', max(0, n-128), n)]:
        values = dict(channels, present=data['present'], contribution=data['contribution'],
                      expected_accuracy=data['correct_probability'],
                      realized_accuracy=(data['choices'] == categories[None, :]).astype(float))
        for name, array in values.items():
            for seed, value in zip(job['seeds'], array[:, begin:end].mean(axis=1)):
                rows.append(dict(subject=meta['subject'], model=job['model'], seed=seed,
                    window=label, first_trial=begin+1, last_trial=end, metric=name, value=float(value)))
    pd.DataFrame(rows).to_csv(target.with_suffix('.csv'), index=False)
    v.write(target.with_suffix('.json'), dict(subject=meta['subject'], model=job['model'],
        candidate=job['candidate']['id'], paths=len(job['seeds']), trials=n, smoke=job['smoke'],
        target_rule=target_rule, exact_replay=True, exact_perception=True,
        replay_seed=job['seeds'][0], numeric_pools=threadpool_info(),
        maximum_decomposition_error=float(np.max(np.abs(channels['expected_error'] -
            sum(channels[k] for k in ('missing_target_error', 'target_emission_error', 'other_rules_error'))))),
        array_sha256=v.digest(target.with_suffix('.npz')), table_sha256=v.digest(target.with_suffix('.csv'))))


def run(source: Path, output: Path, smoke: bool) -> None:
    recalled = helpers()
    v = recalled.base.validation()
    if output.exists():
        raise FileExistsError('Use a new readout diagnosis directory')
    if v.read(source / 'completion.json')['status'] != 'complete':
        raise ValueError('Need a completed autonomous evaluation')
    records = v.read(source / 'autonomous/evaluation.json')['rows']
    if smoke:
        records = records[:1]
    jobs, input_hashes = [], {}
    for record in records:
        subject, arm, point = record['subject'], record['model'], record['candidate']['id']
        original = sorted((source / 'jobs').glob(f'AUTO_evaluate_S{subject}_{arm}_{point}_C*.json'))[0]
        job = v.read(original)
        bank = source / f'autonomous/evaluate/subject_{subject}/{arm}/{point}.npz'
        if v.digest(bank) != record['predictions_sha256']:
            raise ValueError('Autonomous bank changed')
        with np.load(bank) as saved:
            seeds = saved['seeds'].tolist()
            if len(set(seeds)) != len(seeds) or len(seeds) != record['paths']:
                raise ValueError('Invalid path seed bank')
        name = f'S{subject}_{arm}'
        job.update(name=name, seeds=seeds[:2] if smoke else seeds, smoke=smoke,
            reference=str(bank.resolve()), reference_sha256=v.digest(bank),
            original_job=str(original.resolve()), original_job_sha256=v.digest(original),
            target=str((output / name).resolve()))
        job['source_sha256'].update({path: v.digest(ROOT / path) for path in (ENTRY, METRICS)})
        jobs.append(job)
        for path in (bank, original, source / 'protocol.json', source / 'completion.json',
                     source / 'autonomous/evaluation.json'):
            input_hashes[str(path.resolve())] = v.digest(path)
    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    v.write(output / 'protocol.json', dict(source=str(source.resolve()), smoke=smoke,
        input_sha256=input_hashes, source_sha256=jobs[0]['source_sha256'], jobs=len(jobs),
        data_use='development fixed-state error allocation; no fitting; oral excluded',
        uncertainty='independent whole generated paths; Monte Carlo standard error',
        limit='instantaneous target replacement holds states fixed; not a generated learning curve'))

    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json')
        v.write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory(prefix='readout_channels_') as cache:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                    '--worker', str(file.resolve())], env=env, cwd=ROOT, check=True,
                    stdout=log, stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(max_workers=min(len(jobs), len(os.sched_getaffinity(0)))) as pool:
        list(pool.map(launch, jobs))
    frame = pd.concat([pd.read_csv(Path(job['target']).with_suffix('.csv')) for job in jobs])
    frame.to_csv(output / 'path_windows.csv', index=False)
    grouped = frame.groupby(['subject', 'model', 'window', 'first_trial', 'last_trial', 'metric']).value
    summary = grouped.agg(['mean', 'std', 'count']).reset_index()
    summary['mc_se'] = summary['std'] / np.sqrt(summary['count'])
    summary.to_csv(output / 'summary.csv', index=False)
    for path, expected in input_hashes.items():
        if v.digest(Path(path)) != expected:
            raise ValueError('Reference changed during diagnosis')
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, groups=len(jobs),
        exact_replay_groups=len(jobs), exact_perception_groups=len(jobs),
        outputs_sha256={str(path.relative_to(output)): v.digest(path)
                        for path in output.iterdir() if path.suffix in ('.npz', '.csv', '.json')}))
    print(f'{len(jobs)} groups complete; exact replay and perception checks passed', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        recalled = helpers()
        recalled.refit.autonomous = lambda job, workflow: diagnose(job, workflow, recalled)
        recalled.worker(args.worker)
    elif args.source and args.output:
        run(args.source, args.output, args.smoke)
    else:
        parser.error('Supply --source and --output')


if __name__ == '__main__':
    main()
