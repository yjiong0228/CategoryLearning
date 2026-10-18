"""Generate own-history paths from frozen association-diagnostic parameters.

Use --config CONFIG --output NEW_DIRECTORY, optionally --smoke. Every arm uses
all independent paths with equal weight. Common choice streams pair arms within
a participant. Task labels produce feedback after each sampled choice; observed
choices are read only for final trajectory scoring. This continues the fixed-
catalogue diagnostic, without H search/capacity or perceptual noise.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from . import model_0923_association_diagnostic as diagnostic
from .model_0923_episodic_adaptive_task import SharedWorkerBudget
from ...model.association_learner import FixedCatalogueAssociationLearner
from ...metrics.task import category_learning_feedback
from ...metrics.autonomous_fit import energy_u
from ...evaluation.model_0923_h_validation import block_paths, distribution_scores, jackknife_se
from ...evaluation.model_0923_h_generation_uncertainty import path_jackknife
from ...utils.seeding import stable_seed

ROOT = diagnostic.ROOT
MODULE = 'src.Bayesian_state.workflows.analysis.model_0923_association_autonomous'
read, write, digest, verify = diagnostic.read, diagnostic.write, diagnostic.digest, diagnostic.verify


def generate(
    geometry: np.ndarray, categories: np.ndarray, *, gamma: float, arm: dict,
    condition: int, seed: int,
) -> dict[str, np.ndarray]:
    """Sample choices, then let the existing task environment return feedback."""
    q = np.asarray(geometry, dtype=float); labels = np.asarray(categories)
    if (q.ndim != 3 or q.shape[0] < 1 or q.shape[2] != 4 or labels.shape != (len(q),)
            or not np.isfinite(labels).all() or np.any(labels != labels.astype(int))
            or np.any((labels < 1) | (labels > 4)) or condition not in (2, 3)
            or arm['association_retention'] not in ('shared', 'stable')):
        raise ValueError('Need aligned four-category stimuli and a valid diagnostic arm.')
    learner = FixedCatalogueAssociationLearner(q.shape[1], rule_retention=gamma,
        association_retention=gamma if arm['association_retention'] == 'shared' else 1.,
        learn_associations=arm['learn_associations'], hierarchical=condition == 3)
    rng = np.random.default_rng(seed)
    predictions = np.empty((len(q), 4)); choices = np.empty(len(q), dtype=np.int8)
    feedback = np.empty(len(q))
    for t in range(len(q)):
        predictions[t] = learner.predict(q[t])
        choices[t] = rng.choice(4, p=predictions[t]) + 1
        feedback[t] = category_learning_feedback(choices[t], labels[t], condition=condition)
        learner.update(q[t], int(choices[t]), float(feedback[t]))
    return dict(probabilities=predictions, choices=choices, feedback=feedback)


def block_scores(samples: np.ndarray, observed: np.ndarray) -> tuple[dict, dict]:
    """Retain old block scores and add U-statistic Energy without self-pairs."""
    metrics = distribution_scores(samples, observed)
    jack = path_jackknife(samples, observed)
    scale = np.sqrt(samples.shape[1]); x, y = samples/scale, observed/scale
    u = energy_u(x, y)
    metrics.update(energy_u=u['value'], energy_u_mc_se=u['mc_se'])
    jack['energy_u'] = np.array([energy_u(np.delete(x, i, axis=0), y)['value'] for i in range(len(x))])
    for name in ('energy', 'crps'): metrics[name+'_mc_se'] = jackknife_se(jack[name])
    np.testing.assert_allclose(jackknife_se(jack['energy_u']), u['mc_se'], rtol=1e-10, atol=1e-14)
    return metrics, jack


def worker(file: Path) -> None:
    job = read(file); verify(job['sources']); verify(job['inputs'])
    target = Path(job['target'])
    if job['phase'] == 'geometry':
        parts = []
        for path in job['data_files']:
            with np.load(path) as z: parts.append({k: z[k].copy() for k in ('stimulus', 'categories')})
        x = np.concatenate([p['stimulus'] for p in parts])[:job['length']]
        labels = np.concatenate([p['categories'] for p in parts])[:job['length']]
        arrays = dict(geometry=diagnostic.geometry_arrays(x, [job['beta']])[0], categories=labels)
    elif job['phase'] == 'autonomous':
        with np.load(job['geometry']) as z: q, labels = z['geometry'], z['categories']
        paths = [generate(q, labels, gamma=job['gamma'], arm=job['arm'],
                          condition=job['condition'], seed=seed) for seed in job['seeds']]
        arrays = {k: np.stack([r[k] for r in paths]) for k in paths[0]}
        arrays.update(seeds=np.asarray(job['seeds'], dtype=np.uint64), categories=labels)
    else:
        raise ValueError('Unknown generation phase.')
    verify(job['sources']); verify(job['inputs'])
    sha = diagnostic.save_arrays(target.with_suffix('.npz'), **arrays)
    write(target.with_suffix('.json'), dict(job_sha256=digest(file), arrays_sha256=sha,
        phase=job['phase'], subject=job['subject'], beta=job['beta'],
        arm_name=job.get('arm_name'), seeds=job.get('seeds')))


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
    print(f'{jobs[0]["phase"]}: {len(jobs)} jobs; up to {min(budget,len(jobs))} workers', flush=True)
    with ThreadPoolExecutor(max_workers=min(budget, len(jobs))) as pool:
        for n, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            future.result()
            if n % 16 == 0 or n == len(jobs): print(f'{n}/{len(jobs)} complete', flush=True)


def run(config: Path, output: Path, smoke: bool) -> None:
    settings = yaml.safe_load(config.read_text())
    if (any(type(settings[k]) is not int or settings[k] < 1 for k in ('paths', 'chunk', 'parallel_budget', 'block_width'))
            or settings['paths'] < 4 or settings['parallel_budget'] > 128):
        raise ValueError('Need at least four paths, positive chunk/width, and 1..128 workers.')
    if output.exists(): raise FileExistsError('Use a new autonomous association directory.')
    output = output.resolve(); source = ROOT/settings['source']
    done = read(source/'completion.json'); pro = read(source/'protocol.json'); frozen = read(source/'nominees.json')
    if (done['status'] != 'complete' or done['smoke'] or frozen['status'] != 'frozen_before_later_choices'
            or digest(source/'protocol.json') != done['protocol_sha256']
            or digest(source/'nominees.json') != done['nomination_sha256']):
        raise ValueError('Require complete scientific diagnostic with frozen training nominees.')
    verify(pro['source_sha256']); verify(pro['input_sha256'])
    sources = dict(pro['source_sha256'])
    for path in [Path(__file__), *(ROOT/n for n in (
        'src/Bayesian_state/metrics/task.py', 'src/Bayesian_state/metrics/autonomous_fit.py',
        'src/Bayesian_state/evaluation/model_0923_h_validation.py',
        'src/Bayesian_state/evaluation/model_0923_h_generation_uncertainty.py',
        'src/Bayesian_state/utils/seeding.py'))]: sources[str(path.relative_to(ROOT))] = digest(path)
    inputs = {str(source/n): digest(source/n) for n in ('completion.json', 'protocol.json', 'nominees.json')}
    nominees = frozen['nominees']
    if smoke: nominees = [n for n in nominees if n['subject'] == nominees[0]['subject']]
    data = ROOT/pro['settings']['data']; meta = {}; data_files = {}
    for s in sorted({n['subject'] for n in nominees}):
        folder = data/f'subject_{s}'; meta[s] = read(folder/'prepared.json')
        inputs[str(folder/'prepared.json')] = digest(folder/'prepared.json')
        data_files[s] = [str(folder/n) for n in ('training.npz', 'sealed_test.npz')]
        for name, path in zip(('training.npz', 'sealed_test.npz'), data_files[s]):
            inputs[path] = meta[s]['array_sha256'][name]
    verify(inputs); output.mkdir(parents=True)
    for name in sources:
        dest = output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, dest)
    shutil.copy2(config, output/'config.yaml')
    paths, chunk = (4, 4) if smoke else (settings['paths'], settings['chunk'])
    seed_banks = {s: [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], subject=s, path=i))
                     for i in range(paths)] for s in meta}
    if len({seed for bank in seed_banks.values() for seed in bank}) != paths*len(meta):
        raise ValueError('Independent path streams must be distinct across subjects.')
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=sources, input_sha256=inputs,
        parent_nomination_sha256=done['nomination_sha256'], nominees=nominees, seed_banks=seed_banks,
        paths_per_group=paths, chunk=chunk, source_versions=pro['versions'], python=sys.version,
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','PyYAML','numba')},
        behavior_boundary='full own-history from trial1; task categories produce feedback after own choice; no human choices in learning',
        scoring='equal-weight paths; RMS block-accuracy U Energy primary; previous empirical Energy/CRPS retained',
        scope='fixed catalogue representation diagnostic; all development participants; no H or oral input'))
    geometries = []; geometry_lookup = {}
    for index, (s, beta) in enumerate(sorted({(n['subject'], n['beta']) for n in nominees})):
        target = output/'geometry'/f'S{s}_B{index}'
        geometry_lookup[s, beta] = target.with_suffix('.npz')
        geometries.append(dict(name=f'geometry_S{s}_B{index}', phase='geometry', subject=s, beta=beta,
            length=min(16, meta[s]['trials']) if smoke else meta[s]['trials'], data_files=data_files[s],
            sources=sources, inputs=inputs, target=str(target)))
    execute(geometries, output, settings, smoke)
    jobs = []; groups = {}
    for n in nominees:
        s, arm_name = n['subject'], n['arm_name']; file = geometry_lookup[s, n['beta']]
        sha = read(file.with_suffix('.json'))['arrays_sha256']; groups[s, arm_name] = []
        for first in range(0, paths, chunk):
            name = f'S{s}_{arm_name}_C{first//chunk:03d}'
            job = dict(n, name=name, phase='autonomous', condition=meta[s]['condition'],
                arm=pro['settings']['arms'][arm_name], geometry=str(file), seeds=seed_banks[s][first:first+chunk],
                sources=sources, inputs={**inputs, str(file): sha}, target=str(output/'paths'/name))
            jobs.append(job); groups[s, arm_name].append(job)
    execute(jobs, output, settings, smoke)
    rows, contrast_rows, curve_rows, bank = [], [], [], {}
    artifact_inputs = dict(inputs)
    for (s, arm_name), members in groups.items():
        arrays = []
        for j in members:
            file = Path(j['target']).with_suffix('.npz'); record = read(file.with_suffix('.json'))
            if digest(file) != record['arrays_sha256']: raise ValueError('Generated arrays changed.')
            artifact_inputs[str(file)] = record['arrays_sha256']
            with np.load(file) as z: arrays.append({k:z[k].copy() for k in z.files})
        seeds = np.concatenate([a['seeds'] for a in arrays]); np.testing.assert_array_equal(seeds, seed_banks[s])
        labels = arrays[0]['categories']
        for a in arrays: np.testing.assert_array_equal(a['categories'], labels)
        generated = np.concatenate([a['choices'] for a in arrays]) == labels[None, :]
        choices = []
        for file in data_files[s]:
            with np.load(file) as z: choices.append(z['choices'])
        human = np.concatenate(choices)[:len(labels)] == labels
        cut = min(meta[s]['training_trials'], len(labels)); n = len(labels)
        windows = [('whole',0,n), ('training',0,cut)] + ([('test',cut,n)] if cut < n else [])
        for phase, first, last in windows:
            samples = block_paths(generated,first,last,settings['block_width'])
            observed = block_paths(human[None,:],first,last,settings['block_width'])[0]
            metrics, jack = block_scores(samples, observed); late = max(first,last-128)
            late_accuracy = generated[:,late:last].mean(axis=1); observed_late = human[late:last].mean()
            rows.append(dict(subject=s,arm_name=arm_name,phase=phase,paths=paths,**metrics,
                late_accuracy=float(late_accuracy.mean()), observed_late=float(observed_late),
                late_reaches_observed=float((late_accuracy>=observed_late).mean())))
            bank[s,arm_name,phase] = (metrics,jack)
        blocks = block_paths(generated,0,n,settings['block_width']); observed = block_paths(human[None,:],0,n,settings['block_width'])[0]
        for b in range(blocks.shape[1]):
            curve_rows.append(dict(subject=s,arm_name=arm_name,block=b+1,observed=float(observed[b]),
                mean=float(blocks[:,b].mean()),q10=float(np.quantile(blocks[:,b],.1)),q90=float(np.quantile(blocks[:,b],.9))))
    for (s, arm_name, phase), (metrics, jack) in bank.items():
        if not arm_name.startswith('LEARN_'): continue
        reference = 'FIXED_'+arm_name.split('_',1)[1]; other, other_jack = bank[s,reference,phase]
        row = dict(subject=s,arm_name=arm_name,reference=reference,phase=phase,paths=paths)
        for metric in ('energy_u','energy','crps'):
            row['delta_'+metric] = metrics[metric]-other[metric]
            row['delta_'+metric+'_mc_se'] = jackknife_se(jack[metric]-other_jack[metric])
        contrast_rows.append(row)
    pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(output/'contrasts.csv', index=False)
    pd.DataFrame(curve_rows).to_csv(output/'curves.csv', index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>类别关联方案的自主学习</title><h1>冻结参数后的自主路径检验</h1>'
        '<p>每组全部路径等权，从第一题开始接收自身作答对应的反馈。参数来自已完成的训练选择拟合。'
        'Energy U去除同一路径自身的距离项；数值越低，块正确率分布与被试越接近。'
        '标准误只描述有限路径的模拟波动。</p>'+pd.DataFrame(contrast_rows).to_html(index=False)
        +'<p>全几何目录始终可用，H搜索和容量尚待接入。所有数据均为开发样本。</p></html>')
    verify(sources); verify(artifact_inputs)
    write(output/'completion.json', dict(status='complete', smoke=smoke, groups=len(groups), paths_per_group=paths,
        protocol_sha256=digest(output/'protocol.json'), input_sha256=artifact_inputs,
        artifacts={f.name:digest(f) for f in (output/'scores.csv',output/'contrasts.csv',output/'curves.csv',output/'index.html')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker is not None: worker(args.worker)
    elif args.config is None or args.output is None: parser.error('--config and --output are required')
    else: run(args.config, args.output, args.smoke)
