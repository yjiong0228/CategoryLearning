"""Recover capacity and belief retention from existing SEARCH probe trajectories.

Run --config CONFIG --output NEW_DIRECTORY; --smoke scores one saved eight-trial
prefix with one worker, two particles and two seeds. The scientific run takes
disjoint seed ranks across generating points, then scores every declared grid
point using training stimuli and synthetic choices/feedback. Other parameters
remain known. Prepared inputs wait for the declared recovery completion before
scoring; the remaining state-replay workers reserve their share of 128 CPUs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

from . import model_0923_recalled_search_refit as refit
from .model_0923_recalled_capacity_memory_probe import MEMORY, CAPACITY, points
from .model_0923_h_objective_recovery import fixture
from .model_0923_h_training_audit import combine
from .model_0923_h_profiles import numerical_scores
from ...evaluation.model_0923_h_validation import jackknife_se
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]


def disjoint_seeds(bank: list[int], truth_index: int, replicates: int) -> list[int]:
    """Assign consecutive ranks of the sorted seed bank without using outcomes."""
    if (type(truth_index) is not int or truth_index < 0 or type(replicates) is not int
            or replicates < 1 or len(set(bank)) != len(bank)
            or any(type(seed) is not int or seed < 0 for seed in bank)):
        raise ValueError('Require unique seeds and positive replicate count')
    selected = sorted(bank)[truth_index*replicates:(truth_index+1)*replicates]
    if len(selected) != replicates:
        raise ValueError('Not enough independent archived seeds')
    return selected


def training_prefix(physical: dict, generated: dict, seed: int, length: int) -> dict:
    """Select the original training prefix; read behavior only from the saved path."""
    found = np.flatnonzero(generated['seeds'] == seed)
    if (len(found) != 1 or type(length) is not int or length < 1 or length > len(physical['categories'])
            or length > len(generated['categories'])):
        raise ValueError('Missing unique generating seed or valid training prefix')
    np.testing.assert_array_equal(generated['categories'][:length], physical['categories'][:length])
    index = int(found[0]); arrays = {key: physical[key][:length].copy() for key in ('stimulus', 'categories')}
    for key in ('choices', 'feedback'):
        arrays[key] = generated[key][index, :length].copy()
        if arrays[key].shape != (length,) or not np.isfinite(arrays[key]).all():
            raise ValueError('Incomplete synthetic behavior prefix')
    return arrays


def verified_resource_dependency(dependency: Path, predecessor: Path) -> dict:
    """Bind the finished recovery to the exact replay whose slots remain reserved."""
    v = refit.base.validation(); done = v.read(dependency)
    previous_protocol = dependency.parent/'protocol.json'
    previous = v.read(previous_protocol); replay = v.read(predecessor/'protocol.json')
    if (done['status'] != 'complete' or done['smoke']
            or v.digest(previous_protocol) != done['protocol_sha256']
            or Path(previous['resource_predecessor']).resolve() != predecessor.resolve()
            or replay['smoke'] or replay['role'] != 'fixed_parameter_replay_no_oral_input'):
        raise ValueError('Need the completed recovery and its declared scientific replay')
    return dict(dependency_sha256=v.digest(dependency),
                predecessor_protocol_sha256=v.digest(predecessor/'protocol.json'))


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new capacity-memory recovery directory')
    v = refit.base.validation(); settings = yaml.safe_load(config.read_text())
    for key in ('replicates', 'particles', 'seeds', 'parallel_budget'):
        if type(settings[key]) is not int or settings[key] < (2 if key == 'seeds' else 1):
            raise ValueError('Invalid recovery budget')
    source = ROOT/settings['source']; completion = v.read(source/'completion.json')
    if completion['status'] != 'complete' or completion['smoke']:
        raise ValueError('Need complete scientific capacity-memory probe paths')
    if v.digest(source/'summary.json') != completion['summary_sha256']:
        raise ValueError('Probe summary changed')
    source_arrays = v.read(source/'summary.json')['inputs']
    original_protocol = v.read(source/'protocol.json'); hashes = dict(original_protocol['source_sha256'])
    for file in (Path(__file__), ROOT/'src/Bayesian_state/workflows/analysis/model_0923_h_objective_recovery.py',
                 ROOT/'src/Bayesian_state/workflows/analysis/model_0923_h_profiles.py'):
        hashes[str(file.resolve().relative_to(ROOT))] = v.digest(file)
    for name, sha in hashes.items():
        if v.digest(ROOT/name) != sha:
            raise ValueError(f'Generating implementation changed: {name}')
    output.mkdir(parents=True); refit.base.archive(output, config, hashes)
    selected_subjects = list(settings['subjects'].items())[:1] if smoke else settings['subjects'].items()
    jobs, chunks, groups, membership, truths = [], [], {}, [], []
    inputs = {str(source/'completion.json'): v.digest(source/'completion.json'),
              str(source/'protocol.json'): v.digest(source/'protocol.json'),
              str(source/'summary.json'): v.digest(source/'summary.json')}
    generation_seeds = set(); scoring_seeds = set()
    for subject, spec in selected_subjects:
        original_jobs = [dict(v.read(f), source_job_file=str(f.resolve()))
                         for f in sorted((source/'jobs').glob(f'S{subject}_*.json'))]
        anchor = next(j for j in original_jobs if j['anchor'])
        original = Path(anchor['folder']); meta = v.read(original/'prepared.json')
        if (v.digest(original/'training.npz') != meta['array_sha256']['training.npz']
                or v.digest(original/'engine.json') != meta['engine_sha256']):
            raise ValueError('Original training stimuli or engine changed')
        inputs[str(original/'training.npz')] = meta['array_sha256']['training.npz']
        for file in (original/'prepared.json', original/'engine.json'):
            inputs[str(file)] = v.digest(file)
        with np.load(original/'training.npz') as saved:
            physical = {key: saved[key] for key in ('stimulus', 'categories')}
        grid = points(anchor['candidate']['hyperparams'], spec['memory'], spec['capacity'])
        for truth_index, truth in enumerate(grid[:1] if smoke else grid):
            sources = [j for j in original_jobs if j['candidate']['id'] == truth['id']]
            if not sources or any(j['candidate']['hyperparams'] != truth['hyperparams'] for j in sources):
                raise ValueError('Declared generating grid point absent from archived paths')
            seed_bank = [seed for j in sources for seed in j['seeds']]
            seeds = disjoint_seeds(seed_bank, truth_index, 1 if smoke else settings['replicates'])
            for repeat, seed in enumerate(seeds):
                if seed in generation_seeds:
                    raise ValueError('Generating streams must be distinct across cases')
                generation_seeds.add(seed)
                saved_job = next(j for j in sources if seed in j['seeds'])
                file = Path(saved_job['target']).with_suffix('.npz'); record = v.read(file.with_suffix('.json'))
                if (v.digest(file) != record['predictions_sha256']
                        or source_arrays[str(file.resolve())] != record['predictions_sha256']
                        or record['candidate']['hyperparams'] != truth['hyperparams']
                        or record['candidate']['id'] != truth['id'] or record['subject'] != subject
                        or record['model'] != 'SEARCH'):
                    raise ValueError('Archived synthetic path changed')
                inputs[str(file)] = record['predictions_sha256']
                for origin in (file.with_suffix('.json'), Path(saved_job['source_job_file'])):
                    inputs[str(origin)] = v.digest(origin)
                with np.load(file) as saved:
                    generated = {key: saved[key] for key in ('seeds', 'categories', 'choices', 'feedback')}
                np.testing.assert_array_equal(generated['seeds'], saved_job['seeds'])
                length = min(8, meta['training_trials']) if smoke else meta['training_trials']
                arrays = training_prefix(physical, generated, seed, length)
                case = f'S{subject}_T{truth_index:02d}_R{repeat:02d}'; folder = output/'synthetic_data'/case
                fixture(original, folder, arrays)
                truths.append(dict(case=case, subject=subject, repeat=repeat, generation_seed=seed,
                    candidate_id=truth['id'], hyperparams=truth['hyperparams'],
                    true_memory=truth['hyperparams'][MEMORY], true_capacity=truth['hyperparams'][CAPACITY],
                    source_array=str(file), source_array_sha256=inputs[str(file)], trials=length))
                filter_seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                    role='scoring', case=case, repeat=i)) for i in range(2 if smoke else settings['seeds'])]
                if len(set(filter_seeds)) != len(filter_seeds) or scoring_seeds.intersection(filter_seeds):
                    raise ValueError('Scoring streams must be distinct between cases')
                scoring_seeds.update(filter_seeds)
                for point in grid:
                    job = {key: deepcopy(anchor[key]) for key in ('model', 'mechanism', 'runtime', 'arm_settings')}
                    job.update(phase='training', name='score_'+case+'_'+point['id'], folder=str(folder.resolve()),
                        candidate=point, seeds=filter_seeds, particles=2 if smoke else settings['particles'],
                        source_sha256=hashes, smoke=smoke,
                        target=str((output/'scores'/case/point['id']).resolve()))
                    jobs.append(job); groups[job['name']] = []
                    for index, filter_seed in enumerate(filter_seeds):
                        chunk = deepcopy(job); suffix = f'_C{index:03d}'; target = Path(job['target'])
                        chunk.update(name=job['name']+suffix, seeds=[filter_seed],
                                     target=str(target.parent/'chunks'/(target.name+suffix)))
                        chunks.append(chunk); groups[job['name']].append(chunk)
                    membership.append(dict(case=case, subject=subject, repeat=repeat,
                        true_memory=truth['hyperparams'][MEMORY], true_capacity=truth['hyperparams'][CAPACITY],
                        memory=point['hyperparams'][MEMORY], capacity=point['hyperparams'][CAPACITY],
                        candidate_id=point['id'], score_file=job['target']+'.json'))
    if generation_seeds.intersection(scoring_seeds):
        raise ValueError('Generating and scoring streams overlap')
    v.write(output/'truths.json', dict(rows=truths))
    v.write(output/'score_membership.json', dict(rows=membership))
    v.write(output/'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        input_sha256=inputs, sequences=len(truths), score_points=len(jobs), seed_jobs=len(chunks),
        python=sys.version, versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        estimated_coordinates=[MEMORY, CAPACITY], nuisance_parameters='known generating values',
        seed_selection='disjoint consecutive sorted seed ranks per truth; no outcome selection',
        data_boundary='original training stimuli; archived autonomous choices and own feedback; no oral input',
        scope='conditional joint capacity-memory recovery at development-probe nuisance values'))
    v.write(output/'prepared.json', dict(status='prepared', sequences=len(truths), seed_jobs=len(chunks),
        protocol_sha256=v.digest(output/'protocol.json')))
    print(f'Prepared {len(truths)} synthetic prefixes and {len(chunks)} score jobs', flush=True)
    if not smoke:
        dependency = ROOT/settings['wait_for_completion']
        print(f'Waiting for {dependency.name} in {dependency.parent.name} before scoring', flush=True)
        while not dependency.exists():
            time.sleep(20)
        v.write(output/'scoring_started.json', verified_resource_dependency(
            dependency, ROOT/settings['resource_predecessor']))
    started = time.monotonic(); refit.execute(chunks, output, 1 if smoke else settings['parallel_budget'])
    for job in jobs:
        combine(job, groups[job['name']])
    for item in membership:
        record = v.read(Path(item['score_file'])); _, jack = numerical_scores(Path(item['score_file']))
        item.update(mean_nll=record['mean_nll'], mean_nll_mcse=jackknife_se(jack), trials=record['trials'])
    scores = pd.DataFrame(membership); recovered = []
    for case, group in scores.groupby('case', sort=True):
        ordered = group.sort_values(['mean_nll', 'candidate_id']); best, second = ordered.iloc[0], ordered.iloc[1]
        truth = group.loc[group.memory.eq(best.true_memory) & group.capacity.eq(best.true_capacity)].iloc[0]
        seeds, best_jack = numerical_scores(Path(best.score_file))
        for row in (second, truth):
            other, _ = numerical_scores(Path(row.score_file)); np.testing.assert_array_equal(seeds, other)
        _, runner_jack = numerical_scores(Path(second.score_file)); _, truth_jack = numerical_scores(Path(truth.score_file))
        tied = int(group.mean_nll.eq(best.mean_nll).sum()); joint = best.memory == best.true_memory and best.capacity == best.true_capacity
        recovered.append(dict(case=case, subject=best.subject, repeat=best['repeat'],
            true_memory=best.true_memory, true_capacity=best.true_capacity, memory=best.memory, capacity=best.capacity,
            memory_correct=bool(best.memory == best.true_memory), capacity_correct=bool(best.capacity == best.true_capacity),
            selected_joint_correct=bool(joint), unique_joint_correct=bool(joint and tied == 1), tied_minima=tied,
            truth_is_minimum=bool(truth.mean_nll == best.mean_nll), best_nll=best.mean_nll,
            runner_up_gap=second.mean_nll-best.mean_nll, runner_up_gap_mcse=jackknife_se(runner_jack-best_jack),
            truth_gap=truth.mean_nll-best.mean_nll, truth_gap_mcse=jackknife_se(truth_jack-best_jack)))
    scores.to_csv(output/'scores.csv', index=False)
    pd.DataFrame(recovered).to_csv(output/'recovered_capacity_memory.csv', index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>容量与信念保持的联合恢复</title><h1>已知其余参数时的联合恢复</h1>'
        '<p>每条行为序列来自已有SEARCH自主轨迹的训练前缀。四个生成组合使用互不重复的随机种子，'
        '每条序列在完整四点网格上评分。其他参数固定为生成值。</p>'
        '<p>同分数量、生成点是否处于最低值集合及唯一恢复结果分别保留。差值的模拟标准误由共同'
        '滤波种子逐个删除计算，仅描述数值评分波动。该检查评价当前有限网格的条件可辨识性。</p>'
        + pd.DataFrame(recovered).to_html(index=False)
        + '<p><a href="scores.csv">全部网格分数</a> · <a href="truths.json">生成参数与来源</a></p></html>')
    for name, sha in hashes.items():
        if v.digest(ROOT/name) != sha:
            raise ValueError(f'Recovery source changed: {name}')
    v.write(output/'completion.json', dict(status='complete', smoke=smoke, seconds=time.monotonic()-started,
        sequences=len(truths), score_points=len(jobs), seed_jobs=len(chunks), estimated_coordinates=2,
        nuisance_parameters_known=True, protocol_sha256=v.digest(output/'protocol.json'),
        artifacts={p.name: v.digest(p) for p in [*output.glob('*.csv'), output/'index.html']},
        interpretation='coarse conditional capacity-memory recovery; other coordinates known; no all-parameter reliability claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); run(args.config, args.output, args.smoke)
