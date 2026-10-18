"""Behavior-only test of fixed versus learned geometry/response associations.

Run --config CONFIG --output NEW_DIRECTORY, optionally --smoke. Four arms cross
fixed/learned associations with shared/stable conditional retention. All use the
same 152 geometries, physical stimuli, fixed precision grid and exact rule-memory
update. Training pre-choice NLL selects a point; nominees are saved before later
choices are opened. Later feedback continues learning at those fixed parameters.
This bounded representation diagnostic has no H search/capacity, perceptual
noise, adaptive precision, oral input or autonomous generation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from ...model.association_learner import FixedCatalogueAssociationLearner
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from ...hypothesis_space.spaces.structural_0923 import RELATIONAL_EXTENSION
from .model_0923_episodic_adaptive_task import SharedWorkerBudget

ROOT = Path(__file__).resolve().parents[4]
MODULE = 'src.Bayesian_state.workflows.analysis.model_0923_association_diagnostic'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def save_arrays(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle: np.savez_compressed(handle, **arrays)
    return digest(path)


def verify(hashes: dict[str, str]) -> None:
    for name, sha in hashes.items():
        if digest(ROOT/name) != sha: raise ValueError(f'Source or input changed: {name}')


def source_hashes() -> dict[str, str]:
    files = list((ROOT/'src/Bayesian_state/hypothesis_space').rglob('*.py'))
    files += [Path(__file__), *(ROOT/name for name in (
        'src/Bayesian_state/model/association_learner.py',
        'src/Bayesian_state/model/modules/memory.py',
        'src/Bayesian_state/model/modules/base_module.py',
        'src/Bayesian_state/utils/numeric.py',
        'src/Bayesian_state/workflows/analysis/model_0923_episodic_adaptive_task.py',
    ))]
    return {str(f.relative_to(ROOT)): digest(f) for f in files}


def validate_settings(settings: dict) -> None:
    if (type(settings['parallel_budget']) is not int or not 1 <= settings['parallel_budget'] <= 128
            or not settings['subjects'] or len(set(settings['subjects'])) != len(settings['subjects'])
            or any(type(s) is not int for s in settings['subjects'])):
        raise ValueError('Require unique subject IDs and a worker budget in 1..128.')
    for key in ('beta', 'rule_retention'):
        values = settings[key]
        if (not values or len(set(values)) != len(values) or not np.isfinite(values).all()
                or any(v < 0. for v in values) or (key == 'rule_retention' and any(v > 1. for v in values))):
            raise ValueError('Invalid or repeated grid values.')
    expected = {(learn, mode) for learn in (False, True) for mode in ('shared', 'stable')}
    arms = settings['arms']
    if (len(arms) != 4 or any(type(a['learn_associations']) is not bool for a in arms.values())
            or {(a['learn_associations'], a['association_retention']) for a in arms.values()} != expected):
        raise ValueError('Require the full fixed/learned by shared/stable factorial.')


def replay(
    geometry: np.ndarray, choices: np.ndarray, feedback: np.ndarray, *,
    gamma: float, arm: dict, condition: int, trace: bool = False,
) -> dict[str, np.ndarray]:
    """Score sequentially; current choice and feedback enter only after prediction."""
    q = np.asarray(geometry, dtype=float); y = np.asarray(choices); f = np.asarray(feedback)
    if (q.ndim != 3 or q.shape[0] < 1 or q.shape[2] != 4 or y.shape != (len(q),)
            or f.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(f).all()
            or np.any(y != y.astype(int)) or np.any((y < 1) | (y > 4)) or condition not in (2, 3)):
        raise ValueError('Require finite aligned trial arrays and four-choice condition 2 or 3.')
    mode = arm['association_retention']
    if mode not in ('shared', 'stable'): raise ValueError('Unknown conditional retention mode.')
    learner = FixedCatalogueAssociationLearner(q.shape[1], rule_retention=gamma,
        association_retention=gamma if mode == 'shared' else 1.,
        learn_associations=arm['learn_associations'], hierarchical=condition == 3)
    predictions = np.empty((len(q), 4)); weights = []
    for t in range(len(q)):
        predictions[t] = learner.predict(q[t])
        if trace: weights.append(learner.rule_weights)
        learner.update(q[t], int(y[t]), float(f[t]))
    result = dict(probabilities=predictions, observed_probability=predictions[np.arange(len(q)), y.astype(int)-1])
    if trace:
        result.update(rule_weights=np.asarray(weights), final_associations=learner.association_weights,
                      final_rule_weights=learner.rule_weights)
    return result


def geometry_arrays(stimuli: np.ndarray, betas: list[float]) -> np.ndarray:
    """Reuse the maintained 152-rule geometry and observation probabilities."""
    partition = ContinuousPartition(4, 4, structural_extension=RELATIONAL_EXTENSION)
    count = len(partition.hypothesis_space)
    if count != 152: raise ValueError('Diagnostic expects the declared 152 geometries.')
    output = np.empty((len(betas), len(stimuli), count, 4))
    # Adjacent precision evaluations reuse the same rule's exact distance cache.
    for h in range(count):
        for b, beta in enumerate(betas):
            output[b, :, h] = partition.get_category_probabilities(
                h, (stimuli,), beta, distance_mode='boundary').T
    return output


def worker(file: Path) -> None:
    job = read(file); verify(job['sources']); verify(job['inputs'])
    target = Path(job['target'])
    if job['phase'] == 'geometry':
        with np.load(job['training']) as z: x = z['stimulus'][:job['length']]
        result = {'geometry': geometry_arrays(x, job['betas'])}
    elif job['phase'] == 'training':
        with np.load(job['geometry']) as z: q = z['geometry'][job['beta_index']]
        with np.load(job['training']) as z: y, f = z['choices'][:len(q)], z['feedback'][:len(q)]
        result = replay(q, y, f, gamma=job['gamma'], arm=job['arm'], condition=job['condition'])
    elif job['phase'] == 'evaluation':
        with np.load(job['geometry']) as z: train_q = z['geometry'][job['beta_index']]
        with np.load(job['training']) as z: train = {k: z[k][:len(train_q)] for k in ('choices', 'feedback')}
        with np.load(job['test']) as z: test = {k: z[k][:job['test_length']] for k in ('stimulus', 'choices', 'feedback')}
        test_q = geometry_arrays(test['stimulus'], [job['beta']])[0]
        result = replay(np.concatenate([train_q, test_q]),
            np.concatenate([train['choices'], test['choices']]),
            np.concatenate([train['feedback'], test['feedback']]),
            gamma=job['gamma'], arm=job['arm'], condition=job['condition'], trace=True)
    else:
        raise ValueError('Unknown diagnostic phase.')
    verify(job['sources']); verify(job['inputs'])
    sha = save_arrays(target.with_suffix('.npz'), **result)
    record = dict(subject=job['subject'], phase=job['phase'], arrays_sha256=sha,
                  job_sha256=digest(file), source_sha256=job['sources'])
    if job['phase'] != 'geometry':
        p = result['observed_probability']
        if np.any(p <= 0.) or not np.isfinite(p).all(): raise ValueError('Nonpositive choice prediction.')
        record.update(arm_name=job['arm_name'], beta=job['beta'], gamma=job['gamma'],
                      trials=len(p), mean_nll=float(-np.log(p).mean()))
        if job['phase'] == 'evaluation':
            n = len(train_q)
            record.update(training_trials=n, test_trials=len(p)-n,
                training_nll=float(-np.log(p[:n]).mean()), test_nll=float(-np.log(p[n:]).mean()))
    write(target.with_suffix('.json'), record)


def execute(jobs: list[dict], output: Path, settings: dict, smoke: bool) -> None:
    budget = min(1 if smoke else settings['parallel_budget'], len(os.sched_getaffinity(0)))
    slots = SharedWorkerBudget(budget, None if smoke else ROOT/settings['resource_predecessor'])
    def launch(job: dict) -> None:
        file = output/'jobs'/(job['name']+'.json'); write(file, job)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with slots.slot(), file.with_suffix('.log').open('x') as log:
            result = subprocess.run([sys.executable, '-m', MODULE, '--worker', str(file)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')
    print(f'{jobs[0]["phase"]}: {len(jobs)} jobs, up to {min(budget, len(jobs))} workers', flush=True)
    with ThreadPoolExecutor(max_workers=min(budget, len(jobs))) as pool:
        for n, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            future.result()
            if n % 16 == 0 or n == len(jobs): print(f'{n}/{len(jobs)} complete', flush=True)


def run(config: Path, output: Path, smoke: bool) -> None:
    settings = yaml.safe_load(config.read_text()); validate_settings(settings)
    if output.exists(): raise FileExistsError('Use a new association diagnostic directory.')
    output = output.resolve(); data = ROOT/settings['data']; sources = source_hashes()
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    betas = settings['beta'][:1] if smoke else settings['beta']
    gammas = settings['rule_retention'][:2] if smoke else settings['rule_retention']
    folders = {s: data/f'subject_{s}' for s in subjects}; inputs = {}; meta = {}
    for s, folder in folders.items():
        item = read(folder/'prepared.json'); meta[s] = item
        if item['subject'] != s or item['condition'] not in (2, 3) or item['test_trials'] < 1:
            raise ValueError('Need matching prepared four-choice development data with a later segment.')
        for name in ('prepared.json', 'training.npz'):
            path = folder/name; inputs[str(path)] = digest(path)
        if inputs[str(folder/'training.npz')] != item['array_sha256']['training.npz']:
            raise ValueError('Training data changed.')
    output.mkdir(parents=True)
    for name in sources:
        dest = output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, dest)
    shutil.copy2(config, output/'config.yaml')
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=sources,
        input_sha256=inputs, beta=betas, rule_retention=gammas, subjects=subjects,
        selection='training pre-choice NLL; all exact minima retained; beta then gamma breaks ties',
        boundary='physical stimuli; choice and feedback after prediction; oral never read; nominees before later choices',
        model='152 simultaneous geometries; uniform geometry prior; fixed precision; no H or perception noise',
        python=sys.version, versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    common = dict(sources=sources, inputs=inputs)
    geometries = [dict(common, name=f'geometry_S{s}', phase='geometry', subject=s,
        training=str(folders[s]/'training.npz'), length=min(8, meta[s]['training_trials']) if smoke else meta[s]['training_trials'],
        betas=betas, target=str(output/'geometry'/f'S{s}')) for s in subjects]
    execute(geometries, output, settings, smoke)
    jobs = []
    for s in subjects:
        geom = output/'geometry'/f'S{s}.npz'; geom_hash = read(geom.with_suffix('.json'))['arrays_sha256']
        for arm_name, arm in settings['arms'].items():
            for bi, beta in enumerate(betas):
                for gi, gamma in enumerate(gammas):
                    name = f'S{s}_{arm_name}_B{bi}_G{gi}'
                    jobs.append(dict(common, inputs={**inputs, str(geom): geom_hash}, name=name, phase='training',
                        subject=s, condition=meta[s]['condition'], arm_name=arm_name, arm=arm, beta=beta,
                        beta_index=bi, gamma=gamma, training=str(folders[s]/'training.npz'),
                        geometry=str(geom), target=str(output/'training'/name)))
    execute(jobs, output, settings, smoke)
    scores = [read(Path(j['target']).with_suffix('.json')) for j in jobs]
    table = pd.DataFrame([{k: r[k] for k in ('subject', 'arm_name', 'beta', 'gamma', 'trials', 'mean_nll')} for r in scores])
    table.to_csv(output/'training_scores.csv', index=False)
    nominees = []; evaluations = []
    for (s, arm_name), group in table.groupby(['subject', 'arm_name'], sort=True):
        ordered = group.sort_values(['mean_nll', 'beta', 'gamma']); winner = ordered.iloc[0]
        ties = ordered.loc[ordered.mean_nll.eq(winner.mean_nll)]
        nominees.append(dict(subject=int(s), arm_name=arm_name, beta=float(winner.beta), gamma=float(winner.gamma),
                             training_nll=float(winner.mean_nll), tied_minima=ties[['beta', 'gamma']].to_dict('records')))
        original = next(j for j in jobs if j['subject'] == s and j['arm_name'] == arm_name
                        and j['beta'] == winner.beta and j['gamma'] == winner.gamma)
        evaluations.append(dict(original, phase='evaluation', name=f'evaluation_S{s}_{arm_name}',
            target=str(output/'evaluation'/f'S{s}_{arm_name}'), test=str(folders[s]/'sealed_test.npz'),
            test_length=min(8, meta[s]['test_trials']) if smoke else meta[s]['test_trials'],
            inputs={**original['inputs'], str(folders[s]/'sealed_test.npz'): meta[s]['array_sha256']['sealed_test.npz']}))
    write(output/'nominees.json', dict(status='frozen_before_later_choices', nominees=nominees,
        protocol_sha256=digest(output/'protocol.json'), training_scores_sha256=digest(output/'training_scores.csv')))
    execute(evaluations, output, settings, smoke)
    rows = [read(Path(j['target']).with_suffix('.json')) for j in evaluations]
    fields = ('subject', 'arm_name', 'beta', 'gamma', 'training_trials', 'test_trials', 'training_nll', 'test_nll', 'mean_nll')
    summary = pd.DataFrame([{k: row[k] for k in fields} for row in rows])
    for row in summary.itertuples():
        selected = next(n for n in nominees if n['subject'] == row.subject and n['arm_name'] == row.arm_name)
        np.testing.assert_allclose(row.training_nll, selected['training_nll'], rtol=0., atol=1e-14)
    summary.to_csv(output/'summary.csv', index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>类别关联学习的受控检验</title><h1>固定目录的类别关联学习</h1>'
        '<p>四方案共享152条几何规则、物理刺激与参数网格。参数仅由训练段选择预测确定；'
        '后段在固定参数下继续接收真实反馈。所有被试均为开发样本。</p>'
        '<p>本诊断保持全部规则可用，使用固定精度与信念平均读出。H搜索、容量、知觉噪声、'
        '口述预测和自主生成尚待接入。</p>'+summary.to_html(index=False)+'</html>')
    verify(sources); verify(inputs)
    write(output/'completion.json', dict(status='complete', smoke=smoke, training_points=len(jobs),
        evaluation_groups=len(evaluations), protocol_sha256=digest(output/'protocol.json'),
        nomination_sha256=digest(output/'nominees.json'),
        artifacts={p.name: digest(p) for p in (output/'training_scores.csv', output/'summary.csv', output/'index.html')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker is not None: worker(args.worker)
    elif args.config is None or args.output is None: parser.error('--config and --output are required')
    else: run(args.config, args.output, args.smoke)
