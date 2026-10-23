"""Compare fixed-parameter task-start generation and prefix-state continuation.

Run --forecast COMPLETE_FORECAST --output NEW. Every archived candidate and
generated branch is retained. Scores reuse the established 32-trial summaries;
Monte Carlo uncertainty clusters every descendant of the same prefix filter.
Endpoint summaries integrate the full weighted cloud, and precision/pairing
summaries condition on rule presence. No fitting or parameter selection occurs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflows.analysis.model_1023_prefix_forecast import read, write, digest, verify
from .model_0923_h_validation import checked_arrays, block_paths, distribution_scores, jackknife_se
from .model_0923_h_prefix_forecast import filter_cluster_jackknife, clustered_mean_se
from .model_0923_h_generation_uncertainty import path_jackknife


def assemble(jobs: list[dict]) -> tuple[dict, dict]:
    """Keep full endpoint clouds separate from the sampled future branches."""
    chunks, bindings = [], {}
    for job in jobs:
        file = Path(job['target']).with_suffix('.npz')
        record = read(file.with_suffix('.json'))
        if (record['status'] != 'complete' or not record['prefix_exact']
                or record['filter_seed'] != job['seeds'][0]):
            raise ValueError('Incomplete or mismatched prefix chunk')
        bindings[str(file)] = record['predictions_sha256']
        bindings[str(file.with_suffix('.json'))] = digest(file.with_suffix('.json'))
        verify(bindings)
        with np.load(file) as z:
            chunks.append({k: z[k] for k in z.files})
    result = {}
    for key in chunks[0]:
        if key == 'categories':
            for c in chunks[1:]:
                np.testing.assert_array_equal(c[key], chunks[0][key])
            result[key] = chunks[0][key]
        elif key.startswith('endpoint_') or key in ('terminal_weights', 'prefix_probabilities', 'filter_seed'):
            result[key] = np.stack([c[key] for c in chunks])
        else:
            result[key] = np.concatenate([c[key] for c in chunks])
    result['branch_filter_seed'] = np.concatenate([np.repeat(c['filter_seed'], len(c['branch_seeds'])) for c in chunks])
    if len(np.unique(result['branch_seeds'])) != len(result['branch_seeds']):
        raise ValueError('Repeated branch seed')
    return result, bindings


def conditional_mean(values: np.ndarray, present: np.ndarray) -> float:
    """Zero-encoded absent states contribute neither values nor denominator."""
    return float(values[present].mean()) if present.any() else float('nan')


def summarize(forecast: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new continuation summary directory')
    protocol, done = read(forecast/'protocol.json'), read(forecast/'completion.json')
    if done['status'] != 'complete' or done['smoke'] or not done['prefix_exact']:
        raise ValueError('Require complete scientific prefix forecasts')
    if digest(forecast/'protocol.json') != done['protocol_sha256']:
        raise ValueError('Forecast protocol changed')
    verify(protocol['inputs_sha256']); verify(protocol['source_sha256'])
    output.mkdir(parents=True)
    (output/'arrays').mkdir()
    inputs = {str((forecast/k).resolve()): digest(forecast/k) for k in ('protocol.json', 'completion.json')}
    inputs[str(Path(__file__).resolve())] = digest(Path(__file__))
    settings = protocol['settings']
    jobs = [read(p) for p in sorted((forecast/'jobs').glob('*.json'))]
    rows, curves, paths, states, endpoints, path_metrics, contrasts = [], [], [], [], [], [], []
    uncertainties = {}
    for case in protocol['cases']:
        own_jobs = [j for j in jobs if j['case_id'] == case['case_id']]
        if len(own_jobs) != settings['filters']:
            raise ValueError('Missing filters')
        f, bindings = assemble(own_jobs); inputs.update(bindings)
        own = Path(case['evaluation'])
        banks, records = {}, {}
        for role in ('conditional', 'autonomous'):
            record, = read(own/role/'evaluation.json')['rows']
            if (record['candidate']['id'] != case['candidate']
                    or record['candidate']['hyperparams'] != case['candidate_hyperparams']):
                raise ValueError('Candidate parameters changed')
            banks[role] = checked_arrays(own/role, record)
            records[role] = record
            file = own/role/'evaluate'/f'subject_{case["subject"]}'/'COMPACT'/(case['candidate']+'.npz')
            inputs[str(file)] = record['predictions_sha256']
            inputs[str(own/role/'evaluation.json')] = digest(own/role/'evaluation.json')
        c, a = banks['conditional'], banks['autonomous']
        cut = records['conditional']['training_trials']; end = len(c['choices']); length = end-cut
        if cut % settings['block_width'] or length % settings['block_width']:
            raise ValueError('Keep original complete block boundaries')
        np.testing.assert_array_equal(f['categories'], c['categories'][cut:])
        np.testing.assert_array_equal(a['categories'], c['categories'])
        np.testing.assert_array_equal(f['filter_seed'], c['seeds'])
        np.testing.assert_array_equal(f['prefix_probabilities'], c['probabilities'][:, :cut])
        if f['choices'].shape != (128, length):
            raise ValueError('Unexpected branch/trial budget')
        with (output/'arrays'/(case['case_id']+'.npz')).open('xb') as stream:
            np.savez_compressed(stream, **f)
        tag = dict(subject=case['subject'], condition=case['condition'], candidate=case['candidate'],
            roles='|'.join(case['roles']), selected='selected' in case['roles'], reference='reference' in case['roles'],
            cohort=case['cohort'], training_trials=cut)
        for i, seed in enumerate(f['filter_seed']):
            w = f['terminal_weights'][i]; present = f['endpoint_present'][i].astype(bool)
            mass = float(w @ present)
            value = dict(tag, filter_seed=int(seed), target_present=mass,
                target_belief=float(w @ f['endpoint_belief'][i]),
                active_count=float(w @ f['endpoint_active_count'][i]),
                precision_given_presence=float(w @ f['endpoint_beta'][i]/mass) if mass else np.nan,
                effective_particles=float(1/np.sum(w*w)))
            for k in range(3):
                value[f'target_pairing_{k}'] = float(w @ f['endpoint_target_pairing'][i,:,k]/mass) if mass and case['condition']==3 else np.nan
            endpoints.append(value)
        actual = c['choices'][cut:] == f['categories']
        observed = block_paths(actual[None, :], 0, length, settings['block_width'])[0]
        conditional = block_paths(c['correct_probability'], cut, end, settings['block_width']).mean(0)
        late = min(settings['late_trials'], length); obs_late = float(actual[-late:].mean())
        for start in ('task_start', 'filtered_prefix'):
            if start == 'task_start':
                choice = a['choices'][:,cut:]; probability = a['probabilities'][:,cut:]
                labels = seeds = a['seeds']
                trace = {k:a[k][:,cut:] for k in ('present','contribution','beta','search','active_count','target_capacity_dropped')}
            else:
                choice, probability = f['choices'], f['probabilities']
                seeds, labels = f['branch_seeds'], f['branch_filter_seed']
                trace = {k:f['diagnostic_'+k] for k in ('present','contribution','beta','search','active_count','target_capacity_dropped')}
            if not np.isfinite(probability).all() or np.any(probability < 0):
                raise ValueError('Invalid generated choice probabilities')
            np.testing.assert_allclose(probability.sum(-1), 1., atol=1e-12, rtol=0)
            np.testing.assert_array_equal(trace['beta'][~trace['present'].astype(bool)], 0.)
            generated = choice == f['categories'][None,:]
            correct = np.take_along_axis(probability, (f['categories']-1)[None,:,None], axis=2)[...,0]
            samples = block_paths(generated, 0, length, settings['block_width'])
            scores = distribution_scores(samples, observed)
            if start == 'task_start':
                jk = path_jackknife(samples, observed)
            else:
                _, jk = filter_cluster_jackknife(samples, observed, labels)
            uncertainties[case['case_id'],start] = jk
            late_paths = generated[:,-late:].mean(1)
            q = np.quantile(late_paths, [.1,.5,.9])
            row = dict(tag, start=start, paths=len(seeds), independent_units=len(np.unique(labels)),
                **scores, late_observed=obs_late, late_accuracy=float(late_paths.mean()),
                late_accuracy_mc_se=clustered_mean_se(late_paths, labels),
                late_q10=q[0], late_q50=q[1], late_q90=q[2],
                paths_reaching_observed=int((late_paths>=obs_late).sum()),
                observed_midrank=float((late_paths<obs_late).mean()+.5*(late_paths==obs_late).mean()),
                late_correct_probability=float(correct[:,-late:].mean()),
                first_target_present=float(trace['present'][:,0].mean()),
                future_target_ever=float(trace['present'].any(1).mean()))
            for metric in ('energy','crps'):
                row[metric+'_mc_se'] = jackknife_se(jk[metric])
            for k in ('present','contribution','search','active_count','target_capacity_dropped'):
                row['late_'+k] = float(trace[k][:,-late:].mean())
            row['late_precision_given_presence'] = conditional_mean(trace['beta'][:,-late:],trace['present'][:,-late:].astype(bool))
            initial = trace['present'][:,0].astype(bool)
            row['loss_any_given_first_present'] = float((~trace['present'][initial].astype(bool)).any(1).mean()) if initial.any() else np.nan
            for index, first in enumerate(range(0,length,settings['block_width'])):
                last = min(first+settings['block_width'],length); sl = slice(first,last)
                tick = cut+last; mask = trace['present'][:,sl].astype(bool)
                curves.append(dict(tag,start=start,trial=tick,observed=observed[index],conditional=conditional[index],
                    mean=samples[:,index].mean(),q10=np.quantile(samples[:,index],.1),q90=np.quantile(samples[:,index],.9)))
                paths.extend(dict(tag,start=start,trial=tick,seed=int(seed),filter_seed=int(label),accuracy=value)
                             for seed,label,value in zip(seeds,labels,samples[:,index]))
                state = dict(tag,start=start,trial=tick,precision_given_presence=conditional_mean(trace['beta'][:,sl],mask))
                state.update({k:float(trace[k][:,sl].mean()) for k in trace if k!='beta'})
                if start == 'filtered_prefix':
                    target_correct = np.take_along_axis(f['diagnostic_target_probabilities'][:,sl],
                        (f['categories'][sl]-1)[None,:,None],axis=2)[...,0]
                    weighted_target = trace['contribution'][:,sl]*target_correct
                    state['target_correct_given_presence'] = conditional_mean(target_correct,mask)
                    state['target_correct_mass'] = float(weighted_target.mean())
                    state['other_correct_mass'] = float((correct[:,sl]-weighted_target).mean())
                    for k in range(3):
                        state[f'target_pairing_{k}'] = conditional_mean(f['diagnostic_target_pairing'][:,sl,k],mask) if case['condition']==3 else np.nan
                        state[f'pairing_marginal_{k}'] = float(f['diagnostic_pairing_marginal'][:,sl,k].mean()) if case['condition']==3 else np.nan
                states.append(state)
            if start == 'filtered_prefix':
                tail = f['diagnostic_target_probabilities'][:,-late:]
                target_correct = np.take_along_axis(tail,(f['categories'][-late:]-1)[None,:,None],axis=2)[...,0]
                row['late_target_correct_given_presence'] = conditional_mean(target_correct,trace['present'][:,-late:].astype(bool))
                row['late_target_correct_mass'] = float((trace['contribution'][:,-late:]*target_correct).mean())
                row['late_other_correct_mass'] = row['late_correct_probability']-row['late_target_correct_mass']
            rows.append(row)
            path_metrics.extend(dict(tag,start=start,seed=int(seed),filter_seed=int(label),late_accuracy=float(value),
                late_present=float(trace['present'][i,-late:].mean()),late_contribution=float(trace['contribution'][i,-late:].mean()))
                for i,(seed,label,value) in enumerate(zip(seeds,labels,late_paths)))
        pair = rows[-2:]
        contrast = dict(tag)
        for metric in ('energy','crps','late_accuracy'):
            contrast['delta_'+metric] = pair[1][metric]-pair[0][metric]
            contrast['delta_'+metric+'_mc_se'] = float(np.hypot(pair[0][metric+'_mc_se'],pair[1][metric+'_mc_se']))
        contrasts.append(contrast)
    for name,data in [('scores',rows),('curves',curves),('path_blocks',paths),('states',states),
                      ('endpoints',endpoints),('path_metrics',path_metrics),('start_contrasts',contrasts)]:
        pd.DataFrame(data).to_csv(output/(name+'.csv'),index=False)
    write(output/'completion.json',dict(status='complete',forecast=str(forecast.resolve()),
        participants=len({c['subject'] for c in protocol['cases']}),candidates=len(protocol['cases']),
        forecast_paths=done['paths'],inputs_sha256=inputs,
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()},
        scope='Development diagnostics at fixed parameters; candidates are not a parameter confidence region',
        uncertainty='MC error only; full filter clusters for continuation, independent paths for task-start',
        precision='Mean conditional on target presence, with NaN for no active observations',
        pairing='Conditional pairing states 0/1/2 in model order; unavailable for condition 2'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); summarize(args.forecast,args.output)
