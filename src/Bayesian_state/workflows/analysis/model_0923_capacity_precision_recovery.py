"""Recover active capacity and precision storage from existing synthetic paths.

Use --config CONFIG --output NEW_DIRECTORY. Each of four generating cells
(capacity 2/4, RESET/RETAIN) contributes four disjoint archived seed ranks.
Every synthetic training prefix is scored at all four cells, with the remaining
parameters known. Four filter seeds are paired across candidates within a case.
One worker scores all seeds for its candidate; 16 cases give 64 ready jobs.
--smoke uses one eight-trial case, two particles, two seeds and one worker.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/reentry_precision.py'
CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'


def helpers():
    spec = importlib.util.spec_from_file_location('capacity_precision_recovery_base', ROOT/
        'src/Bayesian_state/workflows/analysis/model_0923_recalled_search_refit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    """Use the original training scorer under the declared precision context."""
    recalled = helpers()
    original_validation = recalled.base.validation

    def validation():
        v = original_validation()
        original_score = v.score

        def score(job, workflow):
            if (job['phase'] != 'training' or job['model'] != 'SEARCH'
                    or type(job.get('retain_precision')) is not bool):
                raise ValueError('Recovery requires an explicit SEARCH training mechanism')
            module = recalled.base.load(
                'src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision', MECHANISM)
            cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search']
            with module.reentry_precision(cls.MechanisticRuleSearchHypothesisTransitionModule,
                                          enabled=job['retain_precision']) as events:
                original_score(job, workflow)
            v.write(Path(job['target']).with_suffix('.precision.json'), dict(
                retain_precision=job['retain_precision'], particle_transitions=len(events),
                particle_restores=sum(len(e['precision_restored_rules']) for e in events),
                role='computational particle events'))

        v.score = score
        return v

    recalled.base.validation = validation
    recalled.worker(file)


def crossed_cells(points: list[dict]) -> list[dict]:
    """Vary only capacity and the storage mechanism, retaining both true points."""
    if len(points) != 2 or len({p['id'] for p in points}) != 2:
        raise ValueError('Need two distinct archived capacity points')
    points = sorted(points, key=lambda row: row['hyperparams'][CAPACITY])
    if [p['hyperparams'][CAPACITY] for p in points] != [2, 4]:
        raise ValueError('Require the declared capacities 2 and 4')
    nuisance = [{key: value for key, value in p['hyperparams'].items() if key != CAPACITY} for p in points]
    if nuisance[0] != nuisance[1]:
        raise ValueError('Other generating parameters must be identical')
    return [dict(candidate=deepcopy(point), capacity=point['hyperparams'][CAPACITY],
                 retain_precision=retain, variant='RETAIN' if retain else 'RESET',
                 label=f'K{point["hyperparams"][CAPACITY]}_'+('RETAIN' if retain else 'RESET'))
            for point in points for retain in (False, True)]


def selected_estimates(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep exact ties and paired scoring uncertainty separate from recovery."""
    from src.Bayesian_state.workflows.analysis.model_0923_h_profiles import numerical_scores
    from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se
    rows = []
    for case, group in scores.groupby('case', sort=True):
        if len(group) != 4 or set(group.cell) != {'K2_RESET', 'K2_RETAIN', 'K4_RESET', 'K4_RETAIN'}:
            raise ValueError('Recovery requires every crossed candidate in each case')
        ordered = group.sort_values(['mean_nll', 'cell'])
        best, runner = ordered.iloc[0], ordered.iloc[1]
        truths = group[group.cell == best.true_cell]
        if len(truths) != 1 or group.true_cell.nunique() != 1:
            raise ValueError('Missing unique generating cell')
        truth = truths.iloc[0]
        bank = {row.cell: numerical_scores(Path(row.score_file)) for row in ordered.itertuples()}
        seeds, best_jack = bank[best.cell]
        for own_seeds, _ in bank.values():
            np.testing.assert_array_equal(seeds, own_seeds)
        tied = int(group.mean_nll.eq(best.mean_nll).sum())
        joint = best.cell == best.true_cell
        rows.append(dict(case=case, true_cell=best.true_cell, selected_cell=best.cell,
            true_capacity=int(best.true_capacity), capacity=int(best.capacity),
            true_variant=best.true_variant, variant=best.variant,
            capacity_correct=bool(best.capacity == best.true_capacity),
            mechanism_correct=bool(best.variant == best.true_variant),
            selected_joint_correct=bool(joint), unique_joint_correct=bool(joint and tied == 1),
            tied_minima=tied, truth_is_minimum=bool(truth.mean_nll == best.mean_nll),
            best_nll=best.mean_nll, runner_up_gap=runner.mean_nll-best.mean_nll,
            runner_up_gap_mcse=jackknife_se(bank[runner.cell][1]-best_jack),
            truth_gap=truth.mean_nll-best.mean_nll,
            truth_gap_mcse=jackknife_se(bank[truth.cell][1]-best_jack)))
    return pd.DataFrame(rows)


def run(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.workflows.analysis.model_0923_capacity_memory_recovery import disjoint_seeds, training_prefix
    from src.Bayesian_state.workflows.analysis.model_0923_h_objective_recovery import fixture
    from src.Bayesian_state.workflows.analysis.model_0923_h_profiles import numerical_scores
    from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se
    from src.Bayesian_state.metrics.task import category_learning_feedback
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new capacity/precision recovery directory')
    recalled = helpers()
    v = recalled.base.validation()
    settings = yaml.safe_load(config.read_text())
    if (settings['replicates'] != 4 or settings['seeds'] != 4 or settings['particles'] != 256
            or not 1 <= settings['parallel_budget'] <= 128 or settings['subject'] != 208):
        raise ValueError('Require the bounded S208 recovery budget')
    source = ROOT/settings['source']
    done, protocol = (v.read(source/name) for name in ('completion.json', 'protocol.json'))
    if done['status'] != 'complete' or done['smoke'] or protocol['smoke']:
        raise ValueError('Need the completed scientific precision-retention probe')
    hashes = dict(protocol['source_sha256'])
    for name in (ENTRY, 'src/Bayesian_state/workflows/analysis/model_0923_capacity_memory_recovery.py',
                 'src/Bayesian_state/workflows/analysis/model_0923_h_objective_recovery.py',
                 'src/Bayesian_state/workflows/analysis/model_0923_h_profiles.py',
                 'src/Bayesian_state/evaluation/model_0923_h_validation.py'):
        hashes[name] = v.digest(ROOT/name)
    for name, sha in hashes.items():
        if v.digest(ROOT/name) != sha:
            raise ValueError('Generating or scoring implementation changed: '+name)
    inputs = {str(p.resolve()): v.digest(p) for p in (config, source/'completion.json', source/'protocol.json')}
    # Bind all saved probe artifacts before choosing seed ranks; outcomes never
    # enter the deterministic selection of paths.
    for name, sha in dict(protocol['input_sha256'], **done['input_sha256']).items():
        if v.digest(Path(name)) != sha:
            raise ValueError('Archived generating input changed')
        inputs[name] = sha
    templates = []
    for file in sorted((source/'jobs').glob('RETAIN_*.json')):
        job = v.read(file)
        if job['retain_precision'] is not True or job['model'] != 'SEARCH':
            raise ValueError('Unexpected generating mechanism')
        job['_source_job'] = str(file.resolve())
        templates.append(job)
        inputs[str(file.resolve())] = v.digest(file)
    bank = {}
    for job in templates:
        point = job['candidate']
        if point['id'] in bank and bank[point['id']]['hyperparams'] != point['hyperparams']:
            raise ValueError('Generating candidate identity changed')
        bank[point['id']] = point
    cells = crossed_cells(list(bank.values()))
    output.mkdir(parents=True)
    recalled.base.archive(output, config, hashes)
    jobs, membership, truths = [], [], []
    generation_seeds, scoring_seeds = set(), set()
    for truth_index, truth in enumerate(cells[:1] if smoke else cells):
        parts = [j for j in templates if j['candidate']['id'] == truth['candidate']['id']]
        candidates = [seed for job in parts for seed in job['seeds']]
        selected = disjoint_seeds(candidates, truth_index, 1 if smoke else settings['replicates'])
        for repeat, seed in enumerate(selected):
            if seed in generation_seeds:
                raise ValueError('Generating streams overlap across recovery cases')
            generation_seeds.add(seed)
            original = next(job for job in parts if seed in job['seeds'])
            folder = Path(original['folder'])
            meta = v.read(folder/'prepared.json')
            if (meta['subject'] != settings['subject'] or meta['condition'] != 2
                    or v.digest(folder/'training.npz') != meta['array_sha256']['training.npz']
                    or v.digest(folder/'engine.json') != meta['engine_sha256']):
                raise ValueError('Original task identity, stimuli or engine changed')
            for name in ('prepared.json', 'engine.json', 'training.npz'):
                inputs[str((folder/name).resolve())] = v.digest(folder/name)
            path = Path(original['target']).with_suffix('.npz') if truth['retain_precision'] else Path(original['reference'])
            record = v.read(path.with_suffix('.json'))
            if (v.digest(path) != record['predictions_sha256']
                    or record['candidate']['hyperparams'] != truth['candidate']['hyperparams']):
                raise ValueError('Synthetic generating path changed')
            expected = done['input_sha256'].get(str(path.resolve())) if truth['retain_precision'] else original['reference_sha256']
            if v.digest(path) != expected:
                raise ValueError('Synthetic path no longer matches the frozen probe')
            inputs.update({str(p.resolve()): v.digest(p) for p in (path, path.with_suffix('.json'))})
            with np.load(folder/'training.npz') as z:
                physical = {name: z[name] for name in ('stimulus', 'categories')}
            with np.load(path) as z:
                generated = {name: z[name] for name in ('seeds', 'categories', 'choices', 'feedback')}
            n = 8 if smoke else meta['training_trials']
            arrays = training_prefix(physical, generated, seed, n)
            expected_feedback = [category_learning_feedback(int(c), int(y), condition=2)
                                 for c, y in zip(arrays['choices'], arrays['categories'])]
            np.testing.assert_array_equal(arrays['feedback'], expected_feedback)
            case = f'S208_T{truth_index:02d}_R{repeat:02d}'
            destination = output/'synthetic_data'/case
            fixture(folder, destination, arrays)
            truths.append(dict(case=case, true_cell=truth['label'], generation_seed=seed,
                capacity=truth['capacity'], variant=truth['variant'], candidate=truth['candidate'],
                source_array=str(path.resolve()), source_array_sha256=v.digest(path), trials=n))
            seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                role='scoring', case=case, repeat=i)) for i in range(2 if smoke else settings['seeds'])]
            if len(set(seeds)) != len(seeds) or scoring_seeds.intersection(seeds):
                raise ValueError('Scoring streams overlap between cases')
            scoring_seeds.update(seeds)
            for cell in cells:
                target = output/'scores'/case/cell['label']
                job = {key: deepcopy(original[key]) for key in ('model', 'mechanism', 'runtime', 'arm_settings')}
                job.update(phase='training', name='score_'+case+'_'+cell['label'], candidate=cell['candidate'],
                    folder=str(destination.resolve()), seeds=seeds, particles=2 if smoke else settings['particles'],
                    source_sha256=hashes, smoke=smoke, retain_precision=cell['retain_precision'],
                    target=str(target.resolve()))
                jobs.append(job)
                membership.append(dict(case=case, true_cell=truth['label'], true_capacity=truth['capacity'],
                    true_variant=truth['variant'], cell=cell['label'], capacity=cell['capacity'],
                    variant=cell['variant'], candidate_id=cell['candidate']['id'], score_file=str(target.resolve())+'.json'))
    if generation_seeds.intersection(scoring_seeds):
        raise ValueError('Generating and scoring seeds overlap')
    v.write(output/'truths.json', dict(rows=truths))
    v.write(output/'score_membership.json', dict(rows=membership))
    v.write(output/'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        input_sha256=inputs, jobs=len(jobs), sequences=len(truths), filter_runs=sum(len(j['seeds']) for j in jobs),
        seed_selection='disjoint consecutive sorted generating-seed ranks; no outcome-based selection',
        data_use='original training stimuli; synthetic choices and own task feedback; no oral input',
        scope='coarse crossed capacity/storage recovery; all remaining parameters known',
        worker_unit='one case and candidate, all paired filter seeds; one numeric thread',
        python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    print(f'{len(truths)} synthetic prefixes; {len(jobs)} candidate jobs; one worker per candidate', flush=True)
    recalled.ENTRY = ENTRY
    recalled.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    for row, job in zip(membership, jobs):
        file = Path(row['score_file'])
        record, precision = v.read(file), v.read(file.with_suffix('.precision.json'))
        if precision['retain_precision'] != job['retain_precision']:
            raise ValueError('Scoring storage mechanism differs from the declared candidate')
        if any(p['num_threads'] != 1 for p in record['numeric_pools']):
            raise ValueError('Nested numeric worker threads')
        seeds, jack = numerical_scores(file)
        np.testing.assert_array_equal(seeds, job['seeds'])
        row.update(mean_nll=record['mean_nll'], mean_nll_mcse=jackknife_se(jack), trials=record['trials'])
    scores = pd.DataFrame(membership)
    recovered = selected_estimates(scores)
    scores.to_csv(output/'scores.csv', index=False)
    recovered.to_csv(output/'recovered.csv', index=False)
    for name, sha in dict(inputs, **{str(ROOT/name): sha for name, sha in hashes.items()}).items():
        if v.digest(Path(name)) != sha:
            raise ValueError('Recovery input or implementation changed during scoring')
    v.write(output/'completion.json', dict(status='complete', smoke=smoke, sequences=len(truths),
        jobs=len(jobs), filter_runs=sum(len(j['seeds']) for j in jobs), nuisance_parameters_known=True,
        protocol_sha256=v.digest(output/'protocol.json'), artifacts={p.name:v.digest(p) for p in output.glob('*.csv')},
        interpretation='conditional capacity and mechanism recovery at two existing parameter points; joint parameter uncertainty remains'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Supply --config and --output, or --worker')
