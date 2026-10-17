"""Coarse two-memory recovery with other SEARCH parameters known.

Generate independent synthetic training prefixes at the crossed declared
online-belief and episodic-memory levels. Score every grid point with common
particle seeds; no optimizer eliminates the truth. This is a conditional
identifiability diagnostic, not recovery of all individual parameters.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import model_0923_recalled_search_refit as refit
from .model_0923_h_objective_recovery import fixture
from .model_0923_h_training_audit import combine
from .model_0923_h_profiles import numerical_scores
from ...evaluation.model_0923_h_validation import jackknife_se
from ...optimization.search.adaptive_proposals import candidate
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
MEMORY = refit.base.MEMORY
DECAY = refit.base.DECAY


def grid_points(anchor: dict, memory: list[float], decay: list[float]) -> list[dict]:
    """Change exactly the two specified coordinates, including every truth."""
    if (len(memory) < 2 or len(decay) < 2 or len(set(memory)) != len(memory)
            or len(set(decay)) != len(decay) or any(not np.isfinite(x) or not 0 <= x <= 1 for x in memory + decay)):
        raise ValueError('Need distinct finite retention levels in [0,1]')
    return [candidate(dict(anchor, **{MEMORY: gamma, DECAY: delta}), 'complete_two_memory_grid')
            for gamma, delta in product(memory, decay)]


def run(fit: Path, config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new memory-recovery destination')
    v = refit.base.validation(); frozen = v.read(fit / 'nominees.json')
    if frozen['status'] != 'frozen_before_test' or (frozen['smoke'] and not smoke):
        raise ValueError('Need frozen SEARCH nominees')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Fitted implementation changed: {name}')
    settings = yaml.safe_load(config.read_text())
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    rows = [next(r for r in frozen['nominees'] if r['subject'] == s and r['model'] == 'SEARCH') for s in subjects]
    data = Path(frozen['data']); preparation = v.read(data / 'preparation_manifest.json')
    runtime = str((ROOT / preparation['settings']['source_fit'] / 'source').resolve())
    hashes = dict(frozen['source_sha256'])
    for module in (Path(__file__), ROOT / 'src/Bayesian_state/workflows/analysis/model_0923_h_objective_recovery.py',
                   ROOT / 'src/Bayesian_state/workflows/analysis/model_0923_h_profiles.py'):
        hashes[str(module.resolve().relative_to(ROOT))] = v.digest(module)
    output.mkdir(parents=True)
    refit.base.archive(output, config, hashes)
    # Reuse the validated dispatcher. Production recovery starts after the
    # preceding refit/evaluation queues, so only the old replay can reserve slots.
    execution = dict(frozen['settings'], parallel_budget=1 if smoke else settings['parallel_budget'])
    v.write(output / 'protocol.json', dict(fit=str(fit.resolve()), nomination_sha256=v.digest(fit / 'nominees.json'),
        settings=execution, recovery=settings, smoke=smoke, source_sha256=hashes,
        scope='two_memory_grid_recovery_with_all_other_parameters_known',
        data_boundary='original training stimuli only; synthetic choices and own task feedback',
        estimated_coordinates=[MEMORY, DECAY], other_coordinates='fixed to generating values'))
    generated, truths, physical_data, all_grids = [], [], {}, {}
    for row in rows:
        subject = row['subject']; original = data / f'subject_{subject}'
        meta = v.read(original / 'prepared.json')
        if v.digest(original / 'training.npz') != meta['array_sha256']['training.npz']:
            raise ValueError('Original training stimuli changed')
        with np.load(original / 'training.npz') as saved:
            physical = {key: saved[key][:16] if smoke else saved[key] for key in ('stimulus', 'categories')}
        physical_data[subject] = physical
        point_grid = grid_points(row['candidate']['hyperparams'], settings['memory_values'], settings['decay_values'])
        all_grids[subject] = point_grid
        for index, point in enumerate(point_grid[:1] if smoke else point_grid):
            label = f'S{subject}_T{index:02d}'
            folder = output / 'generation_data' / label
            n = len(physical['categories'])
            # The autonomous engine reads only stimulus and categories. These
            # placeholders keep the prepared schema without importing choices.
            fixture(original, folder, dict(physical, choices=np.ones(n, dtype=int), feedback=np.ones(n)))
            seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], role='generation',
                subject=subject, truth=index, repeat=i)) for i in range(1 if smoke else settings['replicates'])]
            job = dict(phase='autonomous', name='generate_' + label, folder=str(folder.resolve()),
                model='SEARCH', mechanism=frozen['settings']['mechanism'], candidate=point,
                runtime=runtime, seeds=seeds, particles=2 if smoke else settings['particles'],
                source_sha256=hashes, smoke=smoke, arm_settings=frozen['settings']['arms']['SEARCH'],
                target=str((output / 'generated' / label).resolve()))
            generated.append(job)
            truths.append(dict(subject=subject, truth_index=index, label=label, hyperparams=point['hyperparams'],
                true_memory=point['hyperparams'][MEMORY], true_decay=point['hyperparams'][DECAY], seeds=seeds))
    v.write(output / 'truths.json', dict(rows=truths))
    refit.execute(generated, output, execution['parallel_budget'])
    jobs, chunks, groups, membership = [], [], {}, []
    for gen, truth in zip(generated, truths):
        target = Path(gen['target']); record = v.read(target.with_suffix('.json'))
        if v.digest(target.with_suffix('.npz')) != record['predictions_sha256']:
            raise ValueError('Synthetic sequence changed')
        with np.load(target.with_suffix('.npz')) as saved:
            np.testing.assert_array_equal(saved['seeds'], gen['seeds'])
            np.testing.assert_array_equal(saved['categories'], physical_data[truth['subject']]['categories'])
            choices, feedback = saved['choices'], saved['feedback']
        for repeat in range(len(choices)):
            case = truth['label'] + f'_R{repeat:02d}'
            folder = output / 'synthetic_data' / case
            fixture(Path(gen['folder']), folder, dict(physical_data[truth['subject']], choices=choices[repeat], feedback=feedback[repeat]))
            seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], role='scoring',
                case=case, repeat=i)) for i in range(2 if smoke else settings['seeds'])]
            for point in all_grids[truth['subject']]:
                job = dict(gen, phase='training', name='score_' + case + '_' + point['id'],
                    folder=str(folder.resolve()), candidate=point, seeds=seeds,
                    target=str((output / 'scores' / case / point['id']).resolve()))
                jobs.append(job); groups[job['name']] = []
                for index, seed in enumerate(seeds):
                    chunk = deepcopy(job); target = Path(job['target']); suffix = f'_C{index:03d}'
                    chunk.update(name=job['name'] + suffix, seeds=[seed],
                        target=str(target.parent / 'chunks' / (target.name + suffix)))
                    chunks.append(chunk); groups[job['name']].append(chunk)
                membership.append(dict(case=case, subject=truth['subject'], repeat=repeat,
                    true_memory=truth['true_memory'], true_decay=truth['true_decay'],
                    memory=point['hyperparams'][MEMORY], decay=point['hyperparams'][DECAY],
                    score_file=job['target'] + '.json', candidate_id=point['id']))
    v.write(output / 'score_membership.json', dict(rows=membership))
    lengths = {j['folder']: v.read(Path(j['folder']) / 'prepared.json')['training_trials'] for j in chunks}
    chunks.sort(key=lambda j: -lengths[j['folder']])
    refit.execute(chunks, output, execution['parallel_budget'])
    for job in jobs:
        combine(job, groups[job['name']])
    for item in membership:
        record = v.read(Path(item['score_file'])); _, jackknife = numerical_scores(Path(item['score_file']))
        item.update(mean_nll=record['mean_nll'], mean_nll_mcse=jackknife_se(jackknife), trials=record['trials'])
    scores = pd.DataFrame(membership); estimates = []
    for case, group in scores.groupby('case', sort=True):
        ordered = group.sort_values(['mean_nll', 'candidate_id'])
        best, second = ordered.iloc[0], ordered.iloc[1]
        seeds, first_jack = numerical_scores(Path(best.score_file))
        other_seeds, second_jack = numerical_scores(Path(second.score_file))
        np.testing.assert_array_equal(seeds, other_seeds)
        estimates.append(dict(case=case, subject=int(best.subject), repeat=int(best['repeat']),
            true_memory=best.true_memory, true_decay=best.true_decay, memory=best.memory, decay=best.decay,
            memory_correct=bool(best.memory == best.true_memory), decay_correct=bool(best.decay == best.true_decay),
            joint_correct=bool(best.memory == best.true_memory and best.decay == best.true_decay),
            best_nll=best.mean_nll, runner_up_gap=second.mean_nll-best.mean_nll,
            runner_up_gap_mcse=jackknife_se(second_jack-first_jack)))
    scores.to_csv(output / 'scores.csv', index=False)
    pd.DataFrame(estimates).to_csv(output / 'recovered_memory.csv', index=False)
    for name, sha in hashes.items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Recovery source changed: {name}')
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, sequences=len(estimates),
        grid_points=len(jobs), seed_jobs=len(chunks), estimated_coordinates=2,
        nuisance_parameters_known=True, protocol_sha256=v.digest(output / 'protocol.json'),
        interpretation='coarse conditional recovery pilot; independent generated sequences; no joint individual-parameter reliability claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); run(args.fit, args.config, args.output, args.smoke)
