"""Audit joint admission refits and summarize all frozen paths and filter seeds.

Scores reuse the established trial NLL and empirical Energy definitions.
Whole-filter/whole-path jackknife errors measure Monte Carlo resolution at
fixed parameters. Each participant remains a development participant.
--reference-model defaults to BASE; use POOL for recalled-search comparisons.
The saved parameter_spaces.json defines the scored support. An additionally
archived parameter_support.yaml is recorded when supplied by the fit workflow.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .model_0923_h_validation import (load, digest, checked_arrays, nll,
    jackknife_scores, jackknife_se, block_paths, distribution_scores, inactive_parameter_channels)
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_0923_h_validation import write


def loss_parts(probability: np.ndarray, correct: np.ndarray) -> dict:
    """Correct/error contributions use the same full-segment denominator."""
    probability = np.asarray(probability)
    correct = np.asarray(correct, dtype=bool)
    if probability.ndim != 2 or probability.shape[1] != len(correct) or not len(correct):
        raise ValueError('Mismatched probability and correctness arrays')
    losses = -np.log(np.maximum(probability.mean(axis=0), np.finfo(float).tiny))
    return dict(correct_loss=float(losses[correct].sum()/len(correct)),
                error_loss=float(losses[~correct].sum()/len(correct)))


def active_precision_mean(beta: np.ndarray, present: np.ndarray) -> float:
    """Average over active target-rule trials; absence has no measured beta."""
    beta, present = np.asarray(beta), np.asarray(present, dtype=bool)
    if beta.shape != present.shape: raise ValueError('Mismatched beta and presence arrays')
    return float(beta[present].mean()) if present.any() else np.nan


def summarize(fit: Path, evaluation: Path, output: Path, reference_model: str = 'BASE') -> None:
    if output.exists(): raise FileExistsError('Use a new episodic-refit summary directory')
    frozen = load(fit/'nominees.json'); settings = frozen['settings']
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need scientific frozen fits')
    if load(evaluation/'completion.json')['status'] != 'complete': raise ValueError('Evaluation incomplete')
    if load(evaluation/'protocol.json')['nomination_sha256'] != digest(fit/'nominees.json'):
        raise ValueError('Evaluation used different nominees')
    expected = {(r['subject'], r['model']): r for r in frozen['nominees']}
    if len(expected) != len(frozen['nominees']) or set(expected) != {
            (s,m) for s in settings['subjects'] for m in settings['arms']}:
        raise ValueError('Incomplete fitted comparison')
    if reference_model not in settings['arms']:
        raise ValueError('Reference model must be one of the fitted arms')
    spaces = load(fit/'parameter_spaces.json')
    if set(spaces) != set(settings['arms']):
        raise ValueError('Missing archived parameter support')
    for row in frozen['nominees']:
        point, space = row['candidate']['hyperparams'], spaces[row['model']]
        if set(point) != set(space) or any(point[k] not in values for k,values in space.items()):
            raise ValueError('Nominee outside archived parameter support')
    inputs = {}
    def record(path): inputs[str(path)] = digest(path)
    metric_sources = [Path(__file__).with_name(name) for name in
        ('model_0923_h_validation.py','model_0923_h_generation_uncertainty.py')]
    for path in (fit/'nominees.json', fit/'protocol.json', fit/'parameter_spaces.json',
                 evaluation/'protocol.json', evaluation/'completion.json',
                 *metric_sources):
        record(path)
    if (fit/'parameter_support.yaml').exists():
        record(fit/'parameter_support.yaml')
    roles = {}
    for role in ('conditional', 'autonomous'):
        path = evaluation/role/'evaluation.json'; record(path)
        rows = load(path)['rows']; roles[role] = {(r['subject'],r['model']):r for r in rows}
        if len(rows) != len(expected) or set(roles[role]) != set(expected): raise ValueError('Missing evaluation cells')
    scores, curves, paths, parameters, finalists, blocks = [], [], [], [], [], []
    jk_bank, seed_bank, training_bank, confirmations = {}, {}, {}, {}
    width = settings['features']['block_width']
    for key, nominee in expected.items():
        subject, arm = key; tag = dict(subject=subject, model=arm)
        c_row, a_row = [roles[role][key] for role in ('conditional','autonomous')]
        if c_row['candidate']['hyperparams'] != a_row['candidate']['hyperparams'] or c_row['candidate']['hyperparams'] != nominee['candidate']['hyperparams']:
            raise ValueError('Evaluation parameters differ from frozen nomination')
        c, a = [checked_arrays(evaluation/role, row) for role,row in [('conditional',c_row),('autonomous',a_row)]]
        for role,row in [('conditional',c_row),('autonomous',a_row)]:
            record(evaluation/role/'evaluate'/f'subject_{subject}'/arm/(row['candidate']['id']+'.npz'))
        np.testing.assert_array_equal(c['categories'], a['categories'])
        np.testing.assert_allclose(c['probabilities'].sum(axis=-1), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['probabilities'].sum(axis=-1), 1., atol=1e-12, rtol=0)
        n, cut = len(c['choices']), c_row['training_trials']
        human = c['choices'] == c['categories']; generated = a['choices'] == a['categories'][None,:]
        if generated.shape != (settings['autonomous_paths'], n): raise ValueError('Missing autonomous paths')
        if c['observed_probability'].shape != (settings['budgets']['evaluate']['seeds'], n): raise ValueError('Missing filter seeds')
        parameters.extend(dict(tag, parameter=k, value=value) for k,value in nominee['candidate']['hyperparams'].items())
        for phase, first, last in [('training',0,cut),('test',cut,n),('whole',0,n)]:
            p = c['observed_probability'][:,first:last]
            samples = block_paths(generated, first, last, width)
            observed = block_paths(human[None,:], first, last, width)[0]
            jk = dict(choice_nll=jackknife_scores(p), **path_jackknife(samples, observed))
            jk_bank[subject,arm,phase] = jk; seed_bank[subject,arm,phase] = (c['seeds'],a['seeds'])
            late = max(first,last-128)
            late_accuracy = generated[:,late:last].mean(axis=1)
            observed_late = float(human[late:last].mean())
            scores.append(dict(tag,phase=phase,condition=c_row['condition'],trials=last-first,
                choice_nll=nll(p), choice_nll_mc_se=jackknife_se(jk['choice_nll']),
                energy_mc_se=jackknife_se(jk['energy']), crps_mc_se=jackknife_se(jk['crps']),
                **distribution_scores(samples,observed), **loss_parts(p,human[first:last]),
                late_accuracy=float(late_accuracy.mean()), observed_late=observed_late,
                late_reaches_observed=float((late_accuracy >= observed_late).mean()),
                target_ever=float(a['present'][:,:last].any(axis=1).mean()),
                target_proposed=float((a['proposed'][:,:last]==a_row['target_rule']).any(axis=1).mean()),
                max_active_count=int(a['active_count'][:,first:last].max()),
                target_late_present=float(a['present'][:,late:last].mean()),
                target_late_contribution=float(a['contribution'][:,late:last].mean()),
                target_late_beta_when_present=active_precision_mean(a['beta'][:,late:last],a['present'][:,late:last]),
                particles=c_row['particles'],filter_seeds=len(c['seeds']),paths=len(a['seeds'])))
            if phase == 'training': training_bank[key] = (p,c['seeds'])
            if phase != 'whole': continue
            quantiles = np.quantile(samples,[.1,.9],axis=0)
            conditional = block_paths(c['correct_probability'],0,n,width).mean(axis=0)
            for b in range(samples.shape[1]):
                first_b, last_b = b*width, min((b+1)*width,n)
                curves.append(dict(tag, trial=last_b, training_trials=cut, observed=observed[b],
                    autonomous=float(samples[:,b].mean()), lower=quantiles[0,b], upper=quantiles[1,b],
                    conditional=conditional[b], **loss_parts(c['observed_probability'][:,first_b:last_b],human[first_b:last_b])))
                blocks.extend(dict(tag,seed=int(seed),trial=last_b,accuracy=float(samples[i,b]))
                              for i,seed in enumerate(a['seeds']))
            for i,seed in enumerate(a['seeds']):
                indices = np.flatnonzero(a['present'][i])
                paths.append(dict(tag,seed=int(seed),late_accuracy=float(generated[i,late:last].mean()),
                    target_ever=bool(len(indices)),first_target=int(indices[0]+1) if len(indices) else np.nan,
                    max_active_count=int(a['active_count'][i].max()),
                    target_late_present=float(a['present'][i,late:last].mean()),
                    target_late_contribution=float(a['contribution'][i,late:last].mean()),
                    target_late_beta_when_present=active_precision_mean(a['beta'][i,late:last],a['present'][i,late:last]),
                    target_exits=int(a['target_exited'][i].sum()),target_capacity_drops=int(a['target_capacity_dropped'][i].sum()),
                    search_events=int(a['search'][i].sum()),rejections=int(a['rejected'][i].sum()),
                    target_proposals=int((a['proposed'][i]==a_row['target_rule']).sum())))
        saved = {}
        for row in nominee['finalists']:
            file = Path(row['score_file']); r = load(file)
            if digest(file.with_suffix('.npz')) != r['predictions_sha256']: raise ValueError('Finalist arrays changed')
            record(file); record(file.with_suffix('.npz'))
            with np.load(file.with_suffix('.npz')) as z: p,seeds = z['observed_probability'],z['seeds']
            low = fit/'selection'/f'subject_{subject}'/arm/(row['id']+'.npz'); record(low)
            with np.load(low) as z: low_p,low_seeds = z['observed_probability'],z['seeds']
            if set(seeds)&set(low_seeds) or set(seeds)&set(c['seeds']): raise ValueError('Independent stage seeds overlap')
            saved[row['id']] = p,seeds
            finalists.append(dict(tag,candidate=row['id'],selected=row['id']==nominee['candidate']['id'],
                reference=row['id']==nominee['reference_id'],nll=nll(p),mc_se=jackknife_se(jackknife_scores(p)),
                selection_nll=nll(low_p),selection_mc_se=jackknife_se(jackknife_scores(low_p))))
        best,seeds = saved[nominee['candidate']['id']]; reference, ref_seeds = saved[nominee['reference_id']]
        np.testing.assert_array_equal(seeds,ref_seeds)
        confirmations[key] = best,seeds
        for row in finalists:
            if (row['subject'],row['model']) != key: continue
            p, own_seeds = saved[row['candidate']]; np.testing.assert_array_equal(seeds,own_seeds)
            row.update(delta_to_selected=nll(p)-nll(best), delta_to_selected_mc_se=jackknife_se(jackknife_scores(p)-jackknife_scores(best)))
    frame = pd.DataFrame(scores); contrasts = []
    indexed = frame.set_index(['subject','model','phase'])
    for row in frame.itertuples():
        key = row.subject,row.model,row.phase; baseline = row.subject,reference_model,row.phase
        ref = indexed.loc[baseline]
        for a,b in zip(seed_bank[key],seed_bank[baseline]): np.testing.assert_array_equal(a,b)
        result = dict(subject=row.subject, model=row.model, phase=row.phase)
        for metric in ('choice_nll','energy','crps'):
            result['delta_'+metric] = getattr(row,metric)-ref[metric]
            result['delta_'+metric+'_mc_se'] = jackknife_se(jk_bank[key][metric]-jk_bank[baseline][metric])
        for metric in ('correct_loss','error_loss','late_accuracy','target_ever','target_late_present','target_late_contribution'):
            result['delta_'+metric] = getattr(row,metric)-ref[metric]
        contrasts.append(result)
    repeatability = []
    for key,(p,seeds) in confirmations.items():
        repeated,repeated_seeds = training_bank[key]
        if set(seeds)&set(repeated_seeds): raise ValueError('Confirmation reused evaluation seeds')
        delta = nll(repeated)-nll(p)
        se = np.hypot(jackknife_se(jackknife_scores(p)),jackknife_se(jackknife_scores(repeated)))
        repeatability.append(dict(subject=key[0],model=key[1],confirmation_nll=nll(p),repeated_nll=nll(repeated),
            delta_nll=delta,independent_mc_se=se,outside_two_mc_se=abs(delta)>2*se))
    candidates, budgets = [], []
    for stage in ('screen','refine','exchange','selection','confirmation'):
        file = fit/(stage+'_completion.json'); record(file); completed = load(file)
        if completed['status'] != 'complete': raise ValueError('Incomplete search stage')
        for group in completed['groups']:
            subject,arm = group['subject'],group['model']
            for row in group['rows']:
                file = Path(row['score_file']); r = load(file)
                if digest(file.with_suffix('.npz')) != r['predictions_sha256']: raise ValueError('Candidate arrays changed')
                record(file.with_suffix('.npz'))
                candidates.append(dict(subject=subject,model=arm,stage=stage,candidate=row['id'],nll=row['mean_nll'],
                    source=';'.join(row['sources']),**row['hyperparams']))
        for file in (fit/stage).glob('subject_*/*/chunks/*.json'):
            if file.name.endswith('.experiment.json'): continue
            r = load(file)
            if any(pool['num_threads'] != 1 for pool in r['numeric_pools']): raise ValueError('Nested numeric threads')
            budgets.append(dict(subject=r['subject'],model=r['model'],stage=stage,
                state_updates=r['particles']*len(r['seeds'])*r['trials'],filter_runs=len(r['seeds']),cpu_seconds=r['cpu_seconds']))
    budget = pd.DataFrame(budgets).groupby(['subject','model','stage'],as_index=False).sum()
    if not (budget.groupby(['subject','stage']).state_updates.nunique()==1).all(): raise ValueError('Unequal per-subject budgets')
    candidate_frame = pd.DataFrame(candidates)
    for subject,group in candidate_frame[candidate_frame.stage.isin(['screen','exchange'])].groupby('subject'):
        for stage, own in group.groupby('stage'):
            banks = own.groupby('model').candidate.apply(set)
            if any(bank != banks.iloc[0] for bank in banks): raise ValueError('Common candidate banks differ')
    contrast_frame = pd.DataFrame(contrasts); cohort = []
    for (phase,arm),group in frame.groupby(['phase','model']):
        group = group.sort_values('subject')
        d = contrast_frame[(contrast_frame.phase==phase)&(contrast_frame.model==arm)].sort_values('subject')
        np.testing.assert_array_equal(group.subject, d.subject)
        weights = group.trials.to_numpy()
        cohort.append(dict(phase=phase,model=arm,participants=len(group),choice_nll=group.choice_nll.mean(),
            energy=group.energy.mean(),delta_choice_nll=d.delta_choice_nll.mean(),delta_energy=d.delta_energy.mean(),
            delta_choice_nll_mc_se=np.linalg.norm(d.delta_choice_nll_mc_se)/len(group),
            delta_energy_mc_se=np.linalg.norm(d.delta_energy_mc_se)/len(group),
            trial_weighted_choice_nll=np.average(group.choice_nll,weights=weights),
            delta_trial_weighted_choice_nll=np.average(d.delta_choice_nll,weights=weights),
            delta_trial_weighted_choice_nll_mc_se=np.linalg.norm(weights*d.delta_choice_nll_mc_se)/weights.sum()))
    pairwise = []
    for phase in ('training','test','whole'):
        for reference,arm in combinations(settings['arms'],2):
            row = dict(phase=phase,reference=reference,model=arm,participants=len(settings['subjects']))
            for metric in ('choice_nll','energy','crps'):
                differences,errors = [],[]
                for subject in settings['subjects']:
                    key,baseline = (subject,arm,phase),(subject,reference,phase)
                    differences.append(indexed.loc[key,metric]-indexed.loc[baseline,metric])
                    errors.append(jackknife_se(jk_bank[key][metric]-jk_bank[baseline][metric]))
                row['delta_'+metric] = np.mean(differences)
                row['delta_'+metric+'_mc_se'] = np.linalg.norm(errors)/len(errors)
            pairwise.append(row)
    channels = dict(frozen,settings=dict(settings,models={arm:settings['mechanism'] for arm in settings['arms']}))
    tables = dict(scores=frame,contrasts=contrast_frame,cohort_scores=pd.DataFrame(cohort),
        cohort_pairwise=pd.DataFrame(pairwise),curves=pd.DataFrame(curves),
        paths=pd.DataFrame(paths),path_blocks=pd.DataFrame(blocks),parameters=pd.DataFrame(parameters),
        training_finalists=pd.DataFrame(finalists),training_repeatability=pd.DataFrame(repeatability),
        candidates=candidate_frame,budgets=budget,inactive_parameters=inactive_parameter_channels(channels))
    output.mkdir(parents=True)
    for name,table in tables.items(): table.to_csv(output/(name+'.csv'),index=False)
    shutil.copy2(__file__,output/'analysis_source.py')
    (output/'metric_sources').mkdir()
    for path in metric_sources: shutil.copy2(path,output/'metric_sources'/path.name)
    write(output/'manifest.json',dict(status='complete',fit=str(fit.resolve()),evaluation=str(evaluation.resolve()),
        source_sha256=digest(Path(__file__)),inputs_sha256=inputs,
        metric_sources_sha256={p.name:digest(p) for p in metric_sources},
        summary_sha256={name+'.csv':digest(output/(name+'.csv')) for name in tables},
        participants=len(settings['subjects']),arms=list(settings['arms']),block_width=width,reference_model=reference_model,
        data_use='all participants development; training-only estimation; chronological later-trial evaluation',
        uncertainty='whole-filter/whole-path jackknife Monte Carlo standard errors at frozen parameters'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit','evaluation','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-model',default='BASE')
    args = parser.parse_args(); summarize(args.fit,args.evaluation,args.output,args.reference_model)
