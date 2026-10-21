"""Summarize frozen COMPACT versus REFERENCE fits without changing score definitions.

Use --fit FIT --evaluation EVALUATION --output NEW. Whole independent filters
and complete autonomous paths supply Monte Carlo errors. Development subjects
are equally weighted; these errors do not measure population or parameter
uncertainty. Oral and parameter recovery are reported in subsequent analyses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import (
    block_paths, checked_arrays, digest, distribution_scores, jackknife_scores,
    jackknife_se, load, nll,
)
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_0923_h_validation import write


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new simplification summary directory')
    frozen = load(fit / 'nominees.json')
    config = frozen['settings']
    if (frozen['smoke'] or frozen['status'] != 'frozen_before_test'
            or load(evaluation / 'completion.json')['status'] != 'complete'
            or load(evaluation / 'protocol.json')['nomination_sha256'] != digest(fit / 'nominees.json')):
        raise ValueError('Require a completed scientific evaluation of frozen parameters')
    expected = {(s, m) for s in config['subjects'] for m in config['arms']}
    nominees = {(r['subject'], r['model']): r for r in frozen['nominees']}
    if set(nominees) != expected or len(nominees) != len(frozen['nominees']):
        raise ValueError('Missing or duplicate nominees')
    roles = {}
    for role in ('conditional', 'autonomous'):
        entries = load(evaluation / role / 'evaluation.json')['rows']
        roles[role] = {(r['subject'], r['model']): r for r in entries}
        if set(roles[role]) != expected or len(entries) != len(expected):
            raise ValueError('Missing or duplicate evaluation cells')
    inputs = {str(p.resolve()): digest(p) for p in (
        fit / 'nominees.json', fit / 'protocol.json', evaluation / 'protocol.json',
        evaluation / 'completion.json', Path(__file__),
        Path(__file__).with_name('model_0923_h_validation.py'),
        Path(__file__).with_name('model_0923_h_generation_uncertainty.py'))}
    scores, curves, parameters, finalists, budgets = [], [], [], [], []
    jackknives, seed_banks = {}, {}
    for key, nominee in nominees.items():
        subject, arm = key
        cr, ar = [roles[r][key] for r in ('conditional', 'autonomous')]
        for record in (cr, ar):
            if record['candidate']['hyperparams'] != nominee['candidate']['hyperparams']:
                raise ValueError('Evaluation differs from the frozen parameter vector')
        c, a = [checked_arrays(evaluation / role, row) for role, row in [('conditional', cr), ('autonomous', ar)]]
        for role, row in [('conditional', cr), ('autonomous', ar)]:
            file = evaluation / role / 'evaluate' / f'subject_{subject}' / arm / (row['candidate']['id'] + '.npz')
            inputs[str(file.resolve())] = digest(file)
        np.testing.assert_array_equal(c['categories'], a['categories'])
        np.testing.assert_allclose(c['probabilities'].sum(-1), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['probabilities'].sum(-1), 1., atol=1e-12, rtol=0)
        length, cut = len(c['choices']), cr['training_trials']
        human = c['choices'] == c['categories']
        generated = a['choices'] == a['categories'][None, :]
        if (generated.shape != (config['autonomous_paths'], length)
                or c['observed_probability'].shape != (config['budgets']['evaluate']['seeds'], length)):
            raise ValueError('Incomplete path or filter bank')
        tag = dict(subject=subject, model=arm, condition=cr['condition'])
        for phase, first, last in [('training', 0, cut), ('test', cut, length), ('whole', 0, length)]:
            probability = c['observed_probability'][:, first:last]
            samples = block_paths(generated, first, last, config['features']['block_width'])
            observed = block_paths(human[None, :], first, last, config['features']['block_width'])[0]
            jk = dict(choice_nll=jackknife_scores(probability), **path_jackknife(samples, observed))
            jackknives[subject, arm, phase] = jk
            seed_banks[subject, arm, phase] = c['seeds'], a['seeds']
            late = max(first, last - 128)
            scores.append(dict(tag, phase=phase, trials=last-first, training_trials=cut,
                choice_nll=nll(probability), choice_nll_mc_se=jackknife_se(jk['choice_nll']),
                **distribution_scores(samples, observed), energy_mc_se=jackknife_se(jk['energy']),
                late_accuracy=float(generated[:, late:last].mean()), observed_late=float(human[late:last].mean()),
                target_ever=float(a['present'][:, :last].any(1).mean()),
                target_late_present=float(a['present'][:, late:last].mean()),
                target_late_contribution=float(a['contribution'][:, late:last].mean()),
                mean_search=float(a['search'][:, first:last].mean()),
                target_capacity_drops=float(a['target_capacity_dropped'][:, first:last].sum(1).mean()),
                target_autonomous_exits=float(a['target_exited'][:, first:last].sum(1).mean())))
            if phase == 'whole':
                low, high = np.quantile(samples, [.1, .9], axis=0)
                conditional = block_paths(c['correct_probability'], 0, length, config['features']['block_width']).mean(0)
                for i in range(len(observed)):
                    curves.append(dict(tag, trial=min((i+1)*config['features']['block_width'], length),
                        training_trials=cut, observed=observed[i], autonomous=samples[:, i].mean(),
                        lower=low[i], upper=high[i], conditional=conditional[i]))
        parameters.extend(dict(tag, parameter=k, value=v) for k, v in nominee['candidate']['hyperparams'].items())
        for point in nominee['finalists']:
            file = Path(point['score_file'])
            row = load(file)
            if row['predictions_sha256'] != digest(file.with_suffix('.npz')):
                raise ValueError('Confirmation artifact changed')
            with np.load(file.with_suffix('.npz')) as z:
                probability, seeds = z['observed_probability'], z['seeds']
                if set(seeds) & set(c['seeds']):
                    raise ValueError('Confirmation and evaluation seeds overlap')
                finalists.append(dict(tag, candidate=point['id'], selected=point['id'] == nominee['candidate']['id'],
                    reference=point['id'] == nominee['reference_id'], nll=nll(probability),
                    mc_se=jackknife_se(jackknife_scores(probability))))
    score_table = pd.DataFrame(scores)
    contrasts = []
    for subject in config['subjects']:
        for phase in ('training', 'test', 'whole'):
            subset = score_table[(score_table.subject == subject) & (score_table.phase == phase)].set_index('model')
            for index in (0, 1):
                np.testing.assert_array_equal(seed_banks[subject, 'REFERENCE', phase][index],
                                              seed_banks[subject, 'COMPACT', phase][index])
            row = dict(subject=subject, phase=phase)
            for metric in ('choice_nll', 'energy', 'crps'):
                row['delta_' + metric] = float(subset.loc['COMPACT', metric] - subset.loc['REFERENCE', metric])
                row['delta_' + metric + '_mc_se'] = jackknife_se(
                    jackknives[subject, 'COMPACT', phase][metric] - jackknives[subject, 'REFERENCE', phase][metric])
            contrasts.append(row)
    contrast_table = pd.DataFrame(contrasts)
    aggregate = []
    for phase, group in contrast_table.groupby('phase', sort=False):
        row = dict(phase=phase, subjects=len(group))
        for metric in ('choice_nll', 'energy', 'crps'):
            row['delta_' + metric] = float(group['delta_' + metric].mean())
            row['delta_' + metric + '_mc_se'] = float(np.linalg.norm(group['delta_' + metric + '_mc_se']) / len(group))
        aggregate.append(row)
    for stage in ('screen', 'refine', 'selection', 'confirmation'):
        for file in (fit / stage).glob('subject_*/*/chunks/*.json'):
            if file.name.endswith('.mechanism.json'):
                continue
            row = load(file)
            if any(p['num_threads'] != 1 for p in row['numeric_pools']):
                raise ValueError('Nested numeric parallelism')
            budgets.append(dict(subject=row['subject'], model=row['model'], stage=stage,
                filters=len(row['seeds']), particles=row['particles'],
                state_updates=len(row['seeds']) * row['particles'] * row['trials'], cpu_seconds=row['cpu_seconds']))
    if {(r['subject'], r['model'], r['stage']) for r in budgets} != {
            (s, m, stage) for s, m in expected for stage in ('screen', 'refine', 'selection', 'confirmation')}:
        raise ValueError('Missing executed training budgets')
    output.mkdir(parents=True)
    for name, table in dict(scores=score_table, contrasts=contrast_table, aggregate=pd.DataFrame(aggregate),
            curves=pd.DataFrame(curves), parameters=pd.DataFrame(parameters), finalists=pd.DataFrame(finalists),
            compute_budgets=pd.DataFrame(budgets)).items():
        table.to_csv(output / (name + '.csv'), index=False)
    test = next(row for row in aggregate if row['phase'] == 'test')
    write(output / 'completion.json', dict(status='complete', fit=str(fit.resolve()), evaluation=str(evaluation.resolve()),
        inputs_sha256=inputs, subjects=config['subjects'], development_sample=True,
        equal_subject_test_contrast=test, working_tolerances=dict(choice_nll=.02, energy=.03),
        within_behavior_working_tolerances=test['delta_choice_nll'] <= .02 and test['delta_energy'] <= .03,
        interpretation='Saved parameter winners; Monte Carlo error conditional on parameters, not population equivalence',
        oral_and_recovery='pending separate analysis',
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.fit, args.evaluation, args.output)
