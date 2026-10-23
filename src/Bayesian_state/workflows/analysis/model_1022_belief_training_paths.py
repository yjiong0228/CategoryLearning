"""Diagnose training likelihood versus autonomous learning for frozen finalists.

Run --config YAML --output NEW [--smoke]. All declared mechanisms evaluate the
same confirmed parameter banks with equal path budgets and common seeds.
Only training.npz enters generation and scoring. Parameters remain unchanged;
this diagnostic does not nominate a replacement. --smoke checks eight-trial
prefixes against saved full autonomous paths at the original nominees.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PILOT = 'src/Bayesian_state/workflows/analysis/model_1022_belief_comparison.py'
GENERATION = 'src/Bayesian_state/workflows/analysis/model_0923_h_objective_comparison.py'
METRICS = 'src/Bayesian_state/metrics/autonomous_fit.py'
spec = importlib.util.spec_from_file_location('belief_training_paths_pilot', ROOT / PILOT)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
read, write, digest = pilot.read, pilot.write, pilot.digest
generation = pilot.driver.base.load('belief_training_generation', GENERATION)
metrics = pilot.driver.base.load('belief_training_metrics', METRICS)


def sources() -> dict:
    return dict(pilot.sources(pilot.driver.io), **{name: digest(ROOT / name)
                for name in (ENTRY, GENERATION, METRICS)})


def assert_prefix(reference: Path, generated: Path, expected_sha: str) -> None:
    """Changing from full-path observation to prefix scoring must preserve choices."""
    if digest(reference) != expected_sha:
        raise ValueError('Original autonomous reference changed')
    with np.load(reference) as original, np.load(generated) as current:
        n = current['choices'].shape[1]
        indices = [int(np.flatnonzero(original['seeds'] == seed).item()) for seed in current['seeds']]
        np.testing.assert_array_equal(current['choices'], original['choices'][indices, :n])
        np.testing.assert_array_equal(current['categories'], original['categories'][:n])


def worker(file: Path) -> None:
    job = read(file)
    original = pilot.driver.refit.autonomous
    # Reuse the training-only generator inside the exact same mechanism and
    # precision contexts used for the existing full-sequence simulations.
    pilot.driver.refit.autonomous = lambda job, workflow: generation.generation_score(job, pilot.driver.io)
    try:
        pilot.worker(file)
    finally:
        pilot.driver.refit.autonomous = original
    if job.get('prefix_reference'):
        assert_prefix(Path(job['prefix_reference']), Path(job['target']).with_suffix('.npz'),
                      job['prefix_reference_sha256'])


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    original = pilot.ENTRY
    try:
        pilot.ENTRY = ENTRY
        pilot.execute(jobs, output, budget)
    finally:
        pilot.ENTRY = original


def combine(case: dict, features_policy: dict, inputs: dict) -> dict:
    """Score pooled paths, preserving category labels, seeds and the full budget."""
    folder = Path(case['folder'])
    meta = read(folder / 'prepared.json')
    if meta['subject'] != case['subject']:
        raise ValueError('Training participant changed')
    training = folder / 'training.npz'
    if digest(training) != meta['array_sha256']['training.npz']:
        raise ValueError('Training data changed')
    inputs[str(training)] = digest(training)
    with np.load(training) as z:
        # The selection smoke fixture contains four training trials; keep its
        # boundary while also supporting the usual eight-trial prefix check.
        n = min(8, meta['training_trials']) if case['smoke'] else meta['training_trials']
        categories, choices = z['categories'][:n], z['choices'][:n]
    kwargs = dict(n_categories=2 if meta['condition'] == 1 else 4, **features_policy)
    empirical = metrics.behavior_features(choices[None, :], categories, **kwargs)[0]
    parts, seeds = [], []
    for target in case['targets']:
        target = Path(target)
        score = read(target.with_suffix('.json'))
        if (score['subject'] != case['subject'] or score['condition'] != meta['condition']
                or score['candidate']['id'] != case['candidate']['id']
                or score['candidate']['hyperparams'] != case['candidate']['hyperparams']
                or score['data_boundary'] != 'training_only' or score['trials'] != n
                or digest(target.with_suffix('.npz')) != score['predictions_sha256']):
            raise ValueError('Training generation identity, boundary or file changed')
        inputs[str(target.with_suffix('.json'))] = digest(target.with_suffix('.json'))
        inputs[str(target.with_suffix('.npz'))] = score['predictions_sha256']
        with np.load(target.with_suffix('.npz')) as z:
            np.testing.assert_array_equal(z['categories'], categories)
            np.testing.assert_array_equal(z['observed_features'], empirical)
            recalculated = metrics.behavior_features(z['choices'], categories, **kwargs)
            np.testing.assert_array_equal(z['features'], recalculated)
            if len(recalculated) != score['paths']:
                raise ValueError('Generation path count changed')
            parts.append(recalculated)
            seeds.extend(z['seeds'].tolist())
    if seeds != case['seeds'] or len(set(seeds)) != len(seeds):
        raise ValueError('Generation seeds or path budget changed')
    score = metrics.energy_u(np.concatenate(parts), empirical)
    return dict(subject=case['subject'], arm=case['arm'], candidate=case['candidate']['id'],
                original_nominee=case['original_nominee'], training_trials=n,
                confirmation_nll=case['candidate']['mean_nll'], training_energy=score['value'],
                training_energy_mcse=score['mc_se'], paths=len(seeds))


def run(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.utils.seeding import stable_seed

    if output.exists():
        raise FileExistsError('Use a new training-path diagnosis directory')
    policy = yaml.safe_load(config.read_text())
    if (policy['scope'] != 'training_finalist_generation_diagnosis'
            or not pilot.valid_variants(policy['variants'])
            or policy['paths'] < 3 or policy['chunk'] < 3 or not 1 <= policy['parallel_budget'] <= 128):
        raise ValueError('Require the declared matched-arm training diagnosis')
    study = ROOT / policy['source_study']
    protocol, nomination, done = [read(study / name) for name in ('protocol.json', 'nominees.json', 'completion.json')]
    if (nomination['smoke'] or nomination['status'] != 'frozen_before_test' or nomination['later_choices_used']
            or nomination['oral_used'] or done['status'] != 'complete'
            or read(study / 'summary/completion.json')['nominee_sha256'] != digest(study / 'nominees.json')):
        raise ValueError('Require a completed comparison with intact training-only nominations')
    selected = [policy['smoke_subject']] if smoke else policy['subjects']
    if not set(selected) <= set(protocol['settings']['subjects']):
        raise ValueError('Requested participants are absent from the source study')
    inputs = {str((study / name).resolve()): digest(study / name)
              for name in ('protocol.json', 'nominees.json', 'completion.json', 'autonomous_jobs.json')}
    templates = read(study / 'autonomous_jobs.json')['jobs']
    cases, jobs = [], []
    for sid in selected:
        rows = {r['arm']: r for r in nomination['nominees'] if r['subject'] == sid}
        if set(rows) != set(policy['variants']):
            raise ValueError('Source participant is missing a declared arm')
        banks = [{r['id']: r['hyperparams'] for r in rows[arm]['finalists']} for arm in policy['variants']]
        if any(bank != banks[0] for bank in banks[1:]):
            raise ValueError('Confirmed parameter banks differ across mechanisms')
        seeds = [stable_seed(dict(protocol=policy['protocol'], base=policy['seed'], subject=sid,
                                  role='training_autonomous', repeat=i)) for i in range(policy['paths'])]
        for arm in policy['variants']:
            template = next(j for j in templates if j['subject'] == sid and j['model'] == arm)
            folder = Path(template['folder'])
            meta = read(folder / 'prepared.json')
            if meta['subject'] != sid:
                raise ValueError('Training template participant changed')
            for name in ('prepared.json', 'engine.json', 'training.npz'):
                inputs[str(folder / name)] = digest(folder / name)
            bank = [rows[arm]['candidate']] if smoke else rows[arm]['finalists']
            for point in bank:
                score_path = Path(point['score_file'])
                score = read(score_path)
                probabilities = score_path.with_suffix('.npz')
                if (score['subject'] != sid or score['candidate']['hyperparams'] != point['hyperparams']
                        or digest(probabilities) != score['predictions_sha256']):
                    raise ValueError('Confirmed training score changed')
                inputs[str(score_path)] = digest(score_path)
                inputs[str(probabilities)] = score['predictions_sha256']
                with np.load(probabilities) as z:
                    p = z['observed_probability']
                if p.shape[1] != meta['training_trials'] or not np.isfinite(p).all() or np.any(p <= 0):
                    raise ValueError('Invalid confirmed training probability array')
                np.testing.assert_allclose(-np.log(p.mean(0)).mean(), point['mean_nll'], atol=1e-14, rtol=0)
                path_seeds = template['seeds'] if smoke else seeds
                if len(path_seeds) < 3:
                    raise ValueError('At least three prefix paths are required')
                targets = []
                for i in range(0, len(path_seeds), policy['chunk']):
                    chunk = path_seeds[i:i+policy['chunk']]
                    if len(chunk) < 3:
                        raise ValueError('Each generation chunk requires at least three paths')
                    name = f'training_S{sid}_{arm}_{point["id"]}_C{i:03d}'
                    target = str((output / 'generated' / name).resolve())
                    job = dict(phase='autonomous', name=name, subject=sid, case_id=sid, model=arm,
                        candidate=point, mechanism=deepcopy(template['mechanism']), folder=template['folder'],
                        runtime=template['runtime'], arm_settings=template['arm_settings'], features=policy['features'],
                        seeds=chunk, smoke=smoke, source_sha256=sources(), target=target)
                    if smoke:
                        reference = Path(template['target']).with_suffix('.npz')
                        job.update(prefix_reference=str(reference), prefix_reference_sha256=digest(reference))
                        inputs[str(reference)] = digest(reference)
                    jobs.append(job)
                    targets.append(target)
                cases.append(dict(subject=sid, arm=arm, candidate=point, folder=template['folder'],
                    original_nominee=point['id'] == rows[arm]['candidate']['id'], smoke=smoke,
                    targets=targets, seeds=path_seeds))
    output.mkdir(parents=True)
    pilot.driver.base.archive(output, config, sources())
    write(output / 'protocol.json', dict(policy=policy, smoke=smoke, source_sha256=sources(),
        python=sys.version, versions={name: importlib.metadata.version(name)
                                     for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        inputs_sha256=inputs, decision='diagnosis only; keep every frozen nominee', later_choices_read=False,
        oral_used=False, generation='training stimuli with internally generated choices and feedback'))
    write(output / 'cases.json', dict(cases=cases))
    execute(jobs, output, 1 if smoke else policy['parallel_budget'])
    table = pd.DataFrame([combine(case, policy['features'], inputs) for case in cases])
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Training diagnosis input changed: ' + file)
    table.to_csv(output / 'summary.csv', index=False)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        paths=int(table.paths.sum()), nominees_changed=False, later_choices_read=False, oral_used=False,
        inputs_sha256=inputs, source_sha256=sources(), summary_sha256=digest(output / 'summary.csv'),
        exact_prefix_replays=len(cases) if smoke else 0))
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Supply --config and --output, or --worker')
