"""Compare choice likelihood and autonomous learning across estimation scopes.

Use --scope SCOPE_FIT --generation GENERATION --output NEW_DIRECTORY. All
choice comparisons use the scope experiment's common confirmation filters,
including the original parameter points. All shared confirmation candidates
also compare mechanisms at exactly matched parameters. Autonomous differences pair complete
original trajectory seeds. Original/prefix/full are parameter-selection labels;
prefix/later/whole are chronological scoring windows. Every comparison is a
development diagnostic. --smoke permits explicitly marked pipeline fixtures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import (
    block_paths, digest, distribution_scores, inactive_parameter_channels,
    jackknife_scores, jackknife_se, load, nll,
)
from .model_0923_h_generation_uncertainty import path_jackknife
from .model_0923_episodic_refit import active_precision_mean, loss_parts
from ..metrics.task import category_learning_feedback
from ..workflows.analysis.model_0923_h_validation import write


def verified_arrays(file: Path, expected: str) -> dict:
    if digest(file) != expected:
        raise ValueError(f'Prediction array changed: {file}')
    with np.load(file) as saved:
        return {key: saved[key] for key in saved.files}


def validate_generation(arrays: dict, record: dict) -> None:
    """Check task feedback and target-rule diagnostics over the complete bank."""
    p = arrays['probabilities']
    shape = (len(arrays['seeds']), len(arrays['categories']))
    if (p.shape[:2] != shape or not np.isfinite(p).all() or np.any(p < 0)
            or len(set(arrays['seeds'].tolist())) != shape[0]):
        raise ValueError('Invalid autonomous probabilities or repeated paths')
    np.testing.assert_allclose(p.sum(axis=-1), 1., atol=1e-12, rtol=0)
    for name in ('choices', 'feedback', 'present', 'contribution', 'beta', 'active_count',
                 'search', 'target_exited', 'target_capacity_dropped'):
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f'Invalid autonomous process array: {name}')
    if np.any((arrays['choices'] < 1) | (arrays['choices'] > p.shape[2])):
        raise ValueError('Generated response outside the task categories')
    expected = np.array([[category_learning_feedback(int(choice), int(category), condition=record['condition'])
                          for choice, category in zip(path, arrays['categories'])] for path in arrays['choices']])
    np.testing.assert_array_equal(arrays['feedback'], expected)
    correct = np.take_along_axis(p, (arrays['categories']-1)[None, :, None], axis=2)[..., 0]
    np.testing.assert_array_equal(arrays['correct_probability'], correct)
    if (not np.isin(arrays['present'], [0, 1]).all() or np.any(arrays['contribution'] < 0)
            or np.any(arrays['contribution'] > 1+1e-12)
            or np.any(arrays['contribution'][arrays['present'] == 0] != 0)):
        raise ValueError('Target-rule contribution disagrees with presence')
    capacity = record['candidate']['hyperparams']['engine.modules.hypo_transitions_mod.kwargs.max_capacity']
    if np.any((arrays['active_count'] < 1) | (arrays['active_count'] > capacity)):
        raise ValueError('Active-rule count exceeds the candidate capacity')


def paired_difference(target: dict, reference: dict) -> dict:
    """Pair filters for choice scores and full paths for generation scores."""
    np.testing.assert_array_equal(target['filter_seeds'], reference['filter_seeds'])
    np.testing.assert_array_equal(target['path_seeds'], reference['path_seeds'])
    result = {}
    for metric in ('choice_nll', 'energy', 'crps'):
        result['delta_' + metric] = target['metrics'][metric] - reference['metrics'][metric]
        result['delta_' + metric + '_mc_se'] = jackknife_se(target['jackknife'][metric] - reference['jackknife'][metric])
    return result


def matched_choice_scores(confirmed: dict) -> tuple[list[dict], list[dict], dict]:
    """Separate mechanism effects from parameter changes on the shared finalist bank."""
    cache, inputs, scores, contrasts = {}, {}, [], []
    for (subject, model, identity), point in confirmed.items():
        file = Path(point['score_file']); record = load(file)
        array = file.with_suffix('.npz')
        data = verified_arrays(array, record['predictions_sha256'])
        if ((record['subject'], record['model'], record['candidate']['id']) != (subject, model, identity)
                or record['candidate']['hyperparams'] != point['hyperparams']
                or record['seeds'] != data['seeds'].tolist()
                or record['particles'] != point['particles']):
            raise ValueError('Shared confirmation identity or numerical budget changed')
        n, cut = len(data['choices']), record['training_trials']
        p = data['probabilities']
        if (p.shape[:2] != (len(data['seeds']), n) or not 0 < cut < n or len(data['seeds']) < 2
                or len(set(data['seeds'].tolist())) != len(data['seeds'])
                or not np.isfinite(p).all() or np.any(p < 0)):
            raise ValueError('Need valid full sequences and independent confirmation filters')
        np.testing.assert_allclose(p.sum(axis=-1), 1., atol=1e-12, rtol=0)
        actual = np.take_along_axis(p, (data['choices']-1)[None, :, None], axis=2)[..., 0]
        np.testing.assert_array_equal(actual, data['observed_probability'])
        estimates = {}
        for phase, first, last, field in [('prefix', 0, cut, 'prefix'), ('later', cut, n, 'later'),
                                        ('whole', 0, n, 'full')]:
            probability = actual[:, first:last]
            value, deletions = nll(probability), jackknife_scores(probability)
            np.testing.assert_allclose(value, point[field + '_nll'], atol=1e-12, rtol=0)
            estimates[phase] = (value, deletions)
            scores.append(dict(subject=subject, model=model, candidate_id=identity, phase=phase,
                trials=last-first, particles=record['particles'], filter_seeds=len(data['seeds']),
                choice_nll=value, choice_nll_mc_se=jackknife_se(deletions)))
        cache[subject, model, identity] = (data, record, estimates)
        inputs.update({str(file): digest(file), str(array): digest(array)})
    identities = {(subject, identity) for subject, _, identity in confirmed}
    if set(confirmed) != {(subject, arm, identity) for subject, identity in identities for arm in ('POOL', 'SEARCH')}:
        raise ValueError('Mechanism comparison needs the identical confirmation candidate bank')
    for subject, identity in sorted(identities):
        pool, reference, baseline = cache[subject, 'POOL', identity]
        search, target, estimate = cache[subject, 'SEARCH', identity]
        if (reference['candidate']['hyperparams'] != target['candidate']['hyperparams']
                or reference['particles'] != target['particles']
                or reference['training_trials'] != target['training_trials']):
            raise ValueError('Matched mechanisms need the same parameters, budget and trial boundary')
        for field in ('seeds', 'choices', 'categories'):
            np.testing.assert_array_equal(pool[field], search[field])
        for phase in ('prefix', 'later', 'whole'):
            contrasts.append(dict(subject=subject, candidate_id=identity, phase=phase,
                target_model='SEARCH', reference_model='POOL',
                delta_choice_nll=estimate[phase][0]-baseline[phase][0],
                delta_choice_nll_mc_se=jackknife_se(estimate[phase][1]-baseline[phase][1])))
    return scores, contrasts, inputs


def summarize(scope: Path, generation: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new estimation-scope summary destination')
    frozen = load(scope / 'nominees.json')
    g = load(generation / 'generation.json')
    g_protocol = load(generation / 'protocol.json')
    if (frozen['status'] != 'frozen_for_retrospective_diagnostics' or g['status'] != 'complete'
            or frozen['smoke'] != smoke or g['smoke'] != smoke
            or g_protocol['nomination_sha256'] != digest(scope / 'nominees.json')):
        raise ValueError('Need matching completed scope and generation records; fixtures require --smoke')
    if g['bindings'] != g_protocol['bindings']:
        raise ValueError('Generation scope bindings changed')
    selected_bindings = {(row['subject'], row['model'], row['objective']): row['candidate']['id']
                         for row in frozen['nominees']}
    for binding in g['bindings']:
        if binding['objective'] != 'original' and selected_bindings.get(
                (binding['subject'], binding['model'], binding['objective'])) != binding['candidate_id']:
            raise ValueError('Generation uses another scope-selected parameter point')
    confirmation = load(scope / 'confirmation_completion.json')
    confirmed = {(group['subject'], group['model'], row['id']): row
                 for group in confirmation['groups'] for row in group['rows']}
    generated = {(row['subject'], row['model'], row['candidate_id']): row for row in g['rows']}
    if len(generated) != len(g['rows']):
        raise ValueError('Duplicate scientific point banks')
    files = [scope / name for name in ('nominees.json', 'protocol.json', 'completion.json', 'confirmation_completion.json')]
    files += [generation / name for name in ('protocol.json', 'generation.json')]
    inputs = {str(file): digest(file) for file in files}
    candidate_scores, matched_mechanisms, candidate_inputs = matched_choice_scores(confirmed)
    inputs.update(candidate_inputs)
    cache = {}
    for key, row in generated.items():
        reference = confirmed[key]
        score_file = Path(reference['score_file'])
        c_record = load(score_file)
        if ((c_record['subject'], c_record['model'], c_record['candidate']['id']) != key
                or c_record['candidate']['hyperparams'] != row['candidate']['hyperparams']):
            raise ValueError('Choice and generation parameters differ')
        c_file = score_file.with_suffix('.npz')
        a_file = Path(row['target']).with_suffix('.npz')
        c = verified_arrays(c_file, c_record['predictions_sha256'])
        a = verified_arrays(a_file, row['predictions_sha256'])
        for file in (score_file, c_file, a_file):inputs[str(file)] = digest(file)
        validate_generation(a, row)
        n, cut = len(c['choices']), c_record['training_trials']
        if not 0 < cut < n or (len(a['categories']) != n and not smoke):
            raise ValueError('Choice and autonomous trial ranges differ')
        np.testing.assert_array_equal(c['categories'], a['categories'][:n])
        if not np.isfinite(c['probabilities']).all() or np.any(c['probabilities'] < 0):
            raise ValueError('Invalid conditional probabilities')
        np.testing.assert_allclose(c['probabilities'].sum(axis=-1), 1., atol=1e-12, rtol=0)
        indices = (c['choices']-1)[None, :, None]
        np.testing.assert_array_equal(np.take_along_axis(c['probabilities'], indices, axis=2)[..., 0], c['observed_probability'])
        np.testing.assert_array_equal(np.take_along_axis(c['probabilities'], (c['categories']-1)[None, :, None], axis=2)[..., 0], c['correct_probability'])
        np.testing.assert_allclose(nll(c['observed_probability']), reference['full_nll'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(nll(c['observed_probability'][:, :cut]), reference['prefix_nll'], atol=1e-12, rtol=0)
        cache[key] = (c, a, c_record, row)
    scores, curves, paths, parameters, inactive, numeric = [], [], [], [], [], {}
    width = frozen['settings']['features']['block_width']
    for binding in g['bindings']:
        subject, model, objective = binding['subject'], binding['model'], binding['objective']
        identity = binding['candidate_id']
        c, a, c_record, a_record = cache[subject, model, identity]
        tag = dict(subject=subject, model=model, objective=objective, candidate_id=identity)
        n, cut = len(c['choices']), c_record['training_trials']
        observed = c['choices'] == c['categories']
        realized = a['choices'][:, :n] == c['categories'][None, :]
        point = a_record['candidate']['hyperparams']
        parameters.extend(dict(tag, parameter=name, value=value) for name, value in point.items())
        closed = inactive_parameter_channels(dict(nominees=[dict(subject=subject, model=model, candidate=dict(hyperparams=point))],
            settings=dict(models={model: frozen['settings']['mechanism']})))
        inactive.extend(dict(row, objective=objective, candidate_id=identity) for row in closed.to_dict('records'))
        for phase, first, last in [('prefix', 0, cut), ('later', cut, n), ('whole', 0, n)]:
            probability = c['observed_probability'][:, first:last]
            sampled_blocks = block_paths(realized, first, last, width)
            human_blocks = block_paths(observed[None, :], first, last, width)[0]
            metric = dict(choice_nll=nll(probability), **distribution_scores(sampled_blocks, human_blocks))
            jackknife = dict(choice_nll=jackknife_scores(probability), **path_jackknife(sampled_blocks, human_blocks))
            numeric[subject, model, objective, phase] = dict(metrics=metric, jackknife=jackknife,
                filter_seeds=c['seeds'], path_seeds=a['seeds'])
            late = max(first, last-128)
            late_paths = realized[:, late:last].mean(axis=1)
            late_human = float(observed[late:last].mean())
            scores.append(dict(tag, phase=phase, trials=last-first, condition=a_record['condition'],
                particles=c_record['particles'], filter_seeds=len(c['seeds']), paths=len(a['seeds']),
                **metric, **{key+'_mc_se': jackknife_se(values) for key, values in jackknife.items()},
                **loss_parts(probability, observed[first:last]),
                late_accuracy=float(late_paths.mean()), late_accuracy_mc_se=float(late_paths.std(ddof=1)/np.sqrt(len(late_paths))),
                observed_late_accuracy=late_human, paths_reaching_observed_late_accuracy=int((late_paths >= late_human).sum()),
                target_present_any_in_phase=float(a['present'][:, first:last].any(axis=1).mean()),
                target_late_present=float(a['present'][:, late:last].mean()),
                target_late_contribution=float(a['contribution'][:, late:last].mean()),
                target_late_beta_when_present=active_precision_mean(a['beta'][:, late:last], a['present'][:, late:last])))
        sampled_blocks = block_paths(realized, 0, n, width)
        probability_blocks = block_paths(a['correct_probability'], 0, n, width)
        conditional_blocks = block_paths(c['correct_probability'], 0, n, width).mean(axis=0)
        human_blocks = block_paths(observed[None, :], 0, n, width)[0]
        bounds = np.quantile(sampled_blocks, [.1, .9], axis=0)
        for index, first in enumerate(range(0, n, width)):
            curves.append(dict(tag, first_trial=first+1, last_trial=min(n, first+width), prefix_trials=cut,
                observed=human_blocks[index], conditional=conditional_blocks[index],
                generated=sampled_blocks[:, index].mean(), generated_q10=bounds[0, index], generated_q90=bounds[1, index],
                probability_mean=probability_blocks[:, index].mean()))
        late = max(0, n-128)
        for index, seed in enumerate(a['seeds']):
            present = np.flatnonzero(a['present'][index, :n])
            paths.append(dict(tag, seed=int(seed), late_accuracy=float(realized[index, late:n].mean()),
                first_target_trial=int(present[0]+1) if len(present) else np.nan,
                target_late_present=float(a['present'][index, late:n].mean()),
                target_late_contribution=float(a['contribution'][index, late:n].mean()),
                target_exits=int(a['target_exited'][index, :n].sum()),
                target_capacity_drops=int(a['target_capacity_dropped'][index, :n].sum())))
    contrasts = []
    for subject in sorted({row['subject'] for row in g['bindings']}):
        comparisons = [(model, target, model, reference) for model in ('POOL', 'SEARCH')
                       for target, reference in [('prefix', 'original'), ('full', 'original'), ('full', 'prefix')]]
        comparisons += [('SEARCH', objective, 'POOL', objective) for objective in ('original', 'prefix', 'full')]
        for target_model, target_scope, reference_model, reference_scope in comparisons:
            for phase in ('prefix', 'later', 'whole'):
                target = numeric[subject, target_model, target_scope, phase]
                reference = numeric[subject, reference_model, reference_scope, phase]
                contrasts.append(dict(subject=subject, phase=phase, target_model=target_model, target_scope=target_scope,
                    reference_model=reference_model, reference_scope=reference_scope, **paired_difference(target, reference)))
    output.mkdir(parents=True)
    for name, rows in [('scores', scores), ('curves', curves), ('paths', paths), ('parameters', parameters),
                       ('contrasts', contrasts), ('inactive_parameters', inactive),
                       ('candidate_scores', candidate_scores), ('matched_mechanisms', matched_mechanisms)]:
        frame = pd.DataFrame(rows)
        if name == 'inactive_parameters' and frame.empty:
            frame = pd.DataFrame(columns=['subject','model','objective','candidate_id','parameter','value','reason'])
        frame.to_csv(output / (name + '.csv'), index=False)
    write(output / 'manifest.json', dict(status='complete', smoke=smoke, scope=str(scope.resolve()),
        generation=str(generation.resolve()), source_sha256=digest(Path(__file__)), inputs_sha256=inputs,
        output_sha256={file.name: digest(file) for file in output.glob('*.csv')},
        metric_sources_sha256={name: digest(Path(__file__).with_name(name)) for name in
            ('model_0923_h_validation.py', 'model_0923_h_generation_uncertainty.py', 'model_0923_episodic_refit.py')},
        data_use='development; scope selection includes full behavior; later window is chronological, not heldout',
        uncertainty='fixed-parameter simulation error: paired complete confirmation filters and paired complete autonomous paths',
        identity='Repeated objective labels for the same parameter point share one path bank and one filter bank',
        original_parameters='rescored on the same confirmation seeds as prefix/full nominees',
        matched_mechanisms='SEARCH minus POOL at each identical shared confirmation parameter point; common filter seeds',
        block_width=width, smoke_generation_prefix_only=smoke))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scope', 'generation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.scope, args.generation, args.output, args.smoke)


if __name__ == '__main__':
    main()
