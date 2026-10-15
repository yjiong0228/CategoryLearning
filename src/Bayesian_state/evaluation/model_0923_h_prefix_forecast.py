"""Compare task-start generation with autonomous training-prefix forecasts.

All scores concern the same future trials on the recorded stimulus schedule.
The filtered-prefix bank uses complete filter seeds as Monte Carlo clusters;
the task-start bank uses independent complete paths. Their simulation streams
are separate. Parameters remain fixed, and these participants are development
data. Conditional prediction is retained as a mean-curve diagnostic only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import load, digest, checked_arrays, block_paths, distribution_scores, jackknife_se
from .model_0923_h_generation_uncertainty import path_jackknife


def filter_cluster_jackknife(samples: np.ndarray, observed: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Delete every branch of one independently run filter at a time."""
    labels = np.asarray(labels)
    if labels.shape != (len(samples),):raise ValueError('Need one filter label per branch')
    clusters, counts = np.unique(labels, return_counts=True)
    if len(clusters) < 2 or np.any(counts != counts[0]):
        raise ValueError('Need multiple filters with equal numbers of branches')
    scores = [distribution_scores(samples[labels != label], observed) for label in clusters]
    return clusters, {metric: np.array([row[metric] for row in scores]) for metric in ('energy', 'crps')}


def summarize(summary: Path, forecast: Path, output: Path) -> None:
    import json
    if output.exists():raise FileExistsError('Use a new forecast-comparison destination')
    original = load(summary / 'summary_manifest.json'); protocol = load(forecast / 'forecast_protocol.json')
    completed = load(forecast / 'forecast.json')
    if completed['status'] != 'complete' or completed['smoke'] or protocol['smoke']:
        raise ValueError('Need a complete scientific forecast')
    if protocol['nomination_sha256'] != original['nomination_sha256']:
        raise ValueError('Forecast and original predictions use different fits')
    conditional, autonomous = Path(original['conditional']), Path(original['autonomous'])
    references = [{(row['subject'], row['model']): row for row in load(folder / 'evaluation.json')['rows']}
                  for folder in (conditional, autonomous)]
    rows = {(row['subject'], row['model']): row for row in completed['rows']}
    if set(rows) != set(references[0]) or set(rows) != set(references[1]):
        raise ValueError('Forecast model/participant coverage differs')
    results, blocks, model_contrasts, start_contrasts, numeric = [], [], [], [], {}
    hashes = {}
    for key, row in rows.items():
        c = checked_arrays(conditional, references[0][key]); a = checked_arrays(autonomous, references[1][key])
        file = Path(row['target']).with_suffix('.npz')
        if digest(file) != row['predictions_sha256']:raise ValueError('Forecast paths changed')
        hashes[str(file)] = row['predictions_sha256']
        with np.load(file) as saved:f = {name: saved[name] for name in saved.files}
        cut = references[0][key]['training_trials']; end = len(c['choices'])
        np.testing.assert_array_equal(a['categories'], c['categories'])
        np.testing.assert_array_equal(f['categories'], c['categories'][cut:])
        observed = block_paths((c['choices'] == c['categories'])[None, :], cut, end)[0]
        conditional_curve = block_paths(c['correct_probability'], cut, end).mean(axis=0)
        tag = dict(subject=key[0], model=key[1], condition=references[0][key]['condition'])
        for start in ('task_start', 'filtered_prefix'):
            if start == 'task_start':
                samples = block_paths((a['choices'] == a['categories'][None, :]).astype(float), cut, end)
                probability = block_paths(a['correct_probability'], cut, end)
                units, jackknife = a['seeds'], path_jackknife(samples, observed)
                branch_seeds = a['seeds']
            else:
                samples = block_paths((f['choices'] == f['categories'][None, :]).astype(float), 0, end - cut)
                correct = np.take_along_axis(f['probabilities'], (f['categories'] - 1)[None, :, None], axis=2)[..., 0]
                probability = block_paths(correct, 0, end - cut)
                units, jackknife = filter_cluster_jackknife(samples, observed, f['branch_filter_seed'])
                np.testing.assert_array_equal(units, np.sort(f['filter_seed']))
                branch_seeds = f['branch_seeds']
            metrics = distribution_scores(samples, observed)
            numeric[(key[0], key[1], start)] = dict(metrics=metrics, jackknife=jackknife, units=units, branch_seeds=branch_seeds)
            result = dict(tag, start=start, paths=len(samples), independent_units=len(units), training_trials=cut, future_trials=end-cut,
                probability_mean_mae=float(abs(probability.mean(axis=0)-observed).mean()),
                conditional_mean_mae=float(abs(conditional_curve-observed).mean()), **metrics)
            for metric in ('energy', 'crps'):result[metric + '_mc_se'] = jackknife_se(jackknife[metric])
            if start == 'filtered_prefix':
                result.update(mean_search_probability=float(f['search_probability'].mean()),
                    late_search_probability=float(f['search_probability'][:, -64:].mean()),
                    mean_active_rules=float((f['rule'] >= 0).sum(axis=-1).mean()))
            results.append(result)
            lower, upper = np.quantile(samples, [.1, .9], axis=0)
            for index, first in enumerate(range(cut, end, 32)):
                blocks.append(dict(tag, start=start, first_trial=first+1, last_trial=min(first+32, end),
                    observed=observed[index], conditional_mean=conditional_curve[index],
                    probability_mean=probability[:, index].mean(), realized_mean=samples[:, index].mean(),
                    realized_q10=lower[index], realized_q90=upper[index]))
    for (subject, model, start), row in numeric.items():
        baseline = numeric[(subject, 'B', start)]
        np.testing.assert_array_equal(row['units'], baseline['units'])
        np.testing.assert_array_equal(row['branch_seeds'], baseline['branch_seeds'])
        contrast = dict(subject=subject, model=model, reference='B', start=start, independent_units=len(row['units']))
        for metric in ('energy', 'crps'):
            contrast['delta_' + metric] = row['metrics'][metric] - baseline['metrics'][metric]
            contrast['delta_' + metric + '_mc_se'] = jackknife_se(row['jackknife'][metric] - baseline['jackknife'][metric])
        model_contrasts.append(contrast)
        if start == 'filtered_prefix':
            initial = numeric[(subject, model, 'task_start')]
            contrast = dict(subject=subject, model=model, target='filtered_prefix', reference='task_start')
            for metric in ('energy', 'crps'):
                contrast['delta_' + metric] = row['metrics'][metric] - initial['metrics'][metric]
                contrast['independent_mc_se_' + metric] = float(np.hypot(jackknife_se(row['jackknife'][metric]), jackknife_se(initial['jackknife'][metric])))
            start_contrasts.append(contrast)
    frame = pd.DataFrame(model_contrasts); aggregate = []
    for (model, start), group in frame.groupby(['model', 'start'], sort=False):
        result = dict(model=model, start=start, participants=len(group))
        for metric in ('energy', 'crps'):
            result['delta_' + metric] = float(group['delta_' + metric].mean())
            result['delta_' + metric + '_mc_se'] = float(np.linalg.norm(group['delta_' + metric + '_mc_se']) / len(group))
        aggregate.append(result)
    output.mkdir(parents=True)
    for name, values in [('scores', results), ('blocks', blocks), ('model_contrasts', model_contrasts),
                         ('start_contrasts', start_contrasts), ('aggregate_model_contrasts', aggregate)]:
        pd.DataFrame(values).to_csv(output / (name + '.csv'), index=False)
    with (output / 'comparison_manifest.json').open('x') as handle:
        json.dump(dict(status='complete', summary=str(summary.resolve()), forecast=str(forecast.resolve()),
            nomination_sha256=protocol['nomination_sha256'], source_sha256=digest(Path(__file__)),
            inputs_sha256={str(summary/'summary_manifest.json'):digest(summary/'summary_manifest.json'),
                str(forecast/'forecast_protocol.json'):digest(forecast/'forecast_protocol.json'),
                str(forecast/'forecast.json'):digest(forecast/'forecast.json'), **hashes},
            conditioning='task_start: fitted parameters only; filtered_prefix: fitted parameters and observed training prefix',
            uncertainty='fixed-parameter simulation error; paired complete filters for prefix forecasts, complete paths for task-start generation',
            boundary='development participants on recorded future stimulus schedule; future choices do not update forecasts'), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'forecast', 'output'):parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); summarize(args.summary, args.forecast, args.output)


if __name__ == '__main__':main()
