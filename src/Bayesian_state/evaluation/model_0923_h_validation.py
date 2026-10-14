"""Score frozen H comparisons using conditional choices and autonomous paths.

Probability scores use pre-choice predictions. Autonomous scores use complete
sampled choice paths, including observation variability. Each participant keeps
their chronological boundary; all summaries retain every configured model.
Energy and CRPS use the equal-weight empirical predictive distribution (V-statistic).
Gneiting & Raftery (2007), doi:10.1198/016214506000001437, equations 21, 22, 43.
This module creates new summaries and never changes the evaluated fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def nll(probability: np.ndarray) -> float:
    return float(-np.log(np.maximum(probability.mean(axis=0),np.finfo(float).tiny)).mean())


def jackknife_scores(probability: np.ndarray) -> np.ndarray:
    """Delete one independent filter seed at a time; retain all trial dependence."""
    if probability.ndim!=2 or probability.shape[0]<2:raise ValueError('Need multiple complete filter seeds')
    return np.array([nll(np.delete(probability,i,axis=0)) for i in range(len(probability))])


def jackknife_se(values: np.ndarray) -> float:
    return float(np.sqrt((len(values)-1)/len(values)*np.square(values-values.mean()).sum()))


def block_paths(values: np.ndarray, first: int, last: int, width: int=32) -> np.ndarray:
    """Return path-by-block means; trailing blocks retain their actual length."""
    if values.ndim!=2 or not 0<=first<last<=values.shape[1] or width<1:raise ValueError('Invalid path block boundaries')
    return np.array([values[:,a:min(a+width,last)].mean(axis=1) for a in range(first,last,width)]).T


def distribution_scores(samples: np.ndarray, observed: np.ndarray, alpha: float=.2) -> dict:
    """Lower is better for energy, CRPS and interval score; width needs coverage."""
    x,y=np.asarray(samples,float),np.asarray(observed,float)
    if x.ndim!=2 or y.shape!=(x.shape[1],) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need finite path-by-block samples and aligned observed blocks')
    if not 0<alpha<1:raise ValueError('Interval alpha must be in (0,1)')
    energy=(np.linalg.norm(x-y,axis=1).mean()-.5*cdist(x,x).mean())/np.sqrt(len(y))
    crps=np.abs(x-y).mean(axis=0)-.5*np.abs(x[:,None,:]-x[None,:,:]).mean(axis=(0,1))
    lower,upper=np.quantile(x,[alpha/2,1-alpha/2],axis=0)
    interval=upper-lower+2/alpha*(np.maximum(lower-y,0)+np.maximum(y-upper,0))
    return dict(energy=float(energy),crps=float(crps.mean()),interval_score=float(interval.mean()),
        interval_coverage=float(((y>=lower)&(y<=upper)).mean()),interval_width=float((upper-lower).mean()),
        realized_mean_mae=float(abs(x.mean(axis=0)-y).mean()))


def checked_arrays(source: Path, row: dict) -> dict:
    file=source/'evaluate'/f'subject_{row["subject"]}'/row['model']/(row['candidate']['id']+'.npz')
    if digest(file)!=row['predictions_sha256']:raise ValueError('Evaluated predictions changed')
    with np.load(file) as z:return {k:z[k] for k in z.files}


def aggregate_choices(test: pd.DataFrame, contrasts: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Retain both participant-equal and trial-weighted held-out choice scores."""
    aggregate=[]
    for model in models:
        group=test[test.model==model].sort_values('subject')
        baseline=test[test.model=='B'].sort_values('subject')
        np.testing.assert_array_equal(group.subject,baseline.subject)
        np.testing.assert_array_equal(group.trials,baseline.trials)
        difference=group.choice_nll.to_numpy()-baseline.choice_nll.to_numpy()
        numerical=contrasts[contrasts.model==model].sort_values('subject').choice_nll_mc_se.to_numpy() if model!='B' else np.zeros(len(group))
        lengths=group.trials.to_numpy()
        aggregate.append(dict(model=model,participants=len(group),test_trials=int(lengths.sum()),
            participant_mean_nll=float(group.choice_nll.mean()),
            trial_weighted_nll=float(np.average(group.choice_nll,weights=lengths)),
            total_negative_log_score=float((group.choice_nll*lengths).sum()),
            delta_participant_mean_nll=float(difference.mean()),
            delta_total_negative_log_score=float(difference@lengths),
            delta_participant_mc_se=float(np.linalg.norm(numerical)/len(group)),
            delta_total_mc_se=float(np.linalg.norm(numerical*lengths)),
            participants_lower_nll=int((difference<0).sum())))
    return pd.DataFrame(aggregate)


def score_screen(source: Path, output: Path) -> None:
    """Apply the same distribution scores to complete fixed-parameter screens."""
    if output.exists():raise FileExistsError('Use a new screen score destination')
    manifest=load(source/'simulation_manifest.json');rows=[];hashes={}
    if manifest['status']!='complete' or manifest['smoke']:raise ValueError('Need a full completed screen')
    for record in manifest['records']:
        observed=pd.read_csv(source/record['id']/'observed_trials.csv')
        y=(observed.choice==observed.category).to_numpy()
        target=block_paths(y[None,:],0,len(y))[0]
        for model in manifest['settings']['variants']:
            file=source/record['id']/model/'correct_paths.npz';hashes[str(file.relative_to(source))]=digest(file)
            with np.load(file) as z:
                realized=z['realized'].T;probability=z['correct_probability'].T
            groups=[('all',slice(None))]
            if len(realized)>=64:groups.extend([('development_seeds',slice(0,32)),('independent_seeds',slice(32,64))])
            for group,index in groups:
                r=block_paths(realized[index],0,len(y));p=block_paths(probability[index],0,len(y))
                rows.append(dict(subject=record['subject'],model=model,seed_group=group,paths=len(r),
                    probability_mean_mae=float(abs(p.mean(axis=0)-target).mean()),**distribution_scores(r,target)))
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'scores.csv',index=False)
    with (output/'scores_manifest.json').open('x') as f:json.dump(dict(status='complete',source=str(source.resolve()),
        source_sha256=digest(Path(__file__)),manifest_sha256=digest(source/'simulation_manifest.json'),
        artifacts_sha256=hashes,boundary='development participants with original whole-sequence fitted parameters'),f,indent=2)


def summarize(fit: Path, conditional: Path, autonomous: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new validation summary destination')
    frozen=load(fit/'nominees.json');nomination_hash=digest(fit/'nominees.json')
    for source in (conditional,autonomous):
        if load(source/'evaluation_protocol.json')['nomination_sha256']!=nomination_hash:
            raise ValueError('Evaluation was produced from a different frozen fit')
        if load(source/'evaluation.json')['status']!='complete':raise ValueError('Evaluation is incomplete')
    expected={(r['subject'],r['model']):r for r in frozen['nominees']}
    inputs=[]
    for source in (conditional,autonomous):
        rows=load(source/'evaluation.json')['rows'];indexed={(r['subject'],r['model']):r for r in rows}
        if len(indexed)!=len(rows) or set(indexed)!=set(expected):raise ValueError('Evaluation model coverage differs')
        inputs.append(indexed)
    rows,blocks,parameters,contrasts=[],[],[],[];seed_scores={};seeds={}
    for key,nominee in expected.items():
        cr,ar=inputs[0][key],inputs[1][key]
        if any(r['candidate']['hyperparams']!=nominee['candidate']['hyperparams'] for r in (cr,ar)):
            raise ValueError('Evaluation parameters differ from frozen nominees')
        c,a=checked_arrays(conditional,cr),checked_arrays(autonomous,ar)
        np.testing.assert_array_equal(c['categories'],a['categories'])
        trials=len(c['choices']);cut=cr['training_trials']
        if a['choices'].shape[1]!=trials or c['probabilities'].shape[1]!=trials:raise ValueError('Trial order mismatch')
        observed=(c['choices']==c['categories']).astype(float)
        realized=(a['choices']==a['categories'][None,:]).astype(float)
        tag=dict(subject=key[0],model=key[1],condition=cr['condition'])
        p=c['probabilities'].mean(axis=0)
        np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12,rtol=0)
        onehot=np.eye(p.shape[1])[c['choices']-1]
        seed_scores[key]=jackknife_scores(c['observed_probability'][:,cut:]);seeds[key]=c['seeds']
        for phase,first,last in [('training',0,cut),('test',cut,trials),('whole',0,trials)]:
            y=block_paths(observed[None,:],first,last)[0]
            generated=block_paths(realized,first,last)
            probability=block_paths(a['correct_probability'],first,last)
            conditional_blocks=block_paths(c['correct_probability'],first,last)
            metrics=distribution_scores(generated,y)
            row=dict(tag,phase=phase,trials=last-first,blocks=len(y),paths=len(realized),
                filter_seeds=len(c['seeds']),particles=cr['particles'],
                choice_nll=nll(c['observed_probability'][:,first:last]),
                choice_nll_mc_se=jackknife_se(jackknife_scores(c['observed_probability'][:,first:last])),
                choice_brier=float(np.square(p[first:last]-onehot[first:last]).sum(axis=1).mean()),
                probability_mean_mae=float(abs(probability.mean(axis=0)-y).mean()),
                conditional_mean_mae=float(abs(conditional_blocks.mean(axis=0)-y).mean()),
                observed_correct=float(observed[first:last].mean()),**metrics)
            rows.append(row)
            if phase!='whole':
                low,high=np.quantile(generated,[.1,.9],axis=0)
                for j,start in enumerate(range(first,last,32)):
                    blocks.append(dict(tag,phase=phase,first_trial=start+1,last_trial=min(start+32,last),
                        observed=y[j],autonomous_probability_mean=probability[:,j].mean(),
                        conditional_probability_mean=conditional_blocks[:,j].mean(),
                        realized_mean=generated[:,j].mean(),realized_q10=low[j],realized_q90=high[j]))
        for name,value in nominee['candidate']['hyperparams'].items():
            parameters.append(dict(tag,parameter=name,value=value))
    frame=pd.DataFrame(rows)
    for (subject,model),scores in seed_scores.items():
        if model=='B':continue
        np.testing.assert_array_equal(seeds[(subject,model)],seeds[(subject,'B')])
        left=frame[(frame.subject==subject)&(frame.model==model)&(frame.phase=='test')].iloc[0]
        right=frame[(frame.subject==subject)&(frame.model=='B')&(frame.phase=='test')].iloc[0]
        contrast=dict(subject=subject,model=model,reference='B',choice_nll_mc_se=jackknife_se(scores-seed_scores[(subject,'B')]))
        for metric in ['choice_nll','choice_brier','energy','crps','interval_score','interval_coverage',
                       'interval_width','probability_mean_mae','conditional_mean_mae']:
            contrast['delta_'+metric]=float(left[metric]-right[metric])
        contrasts.append(contrast)
    output.mkdir(parents=True)
    test=frame[frame.phase=='test']
    contrast_frame=pd.DataFrame(contrasts)
    aggregate=aggregate_choices(test,contrast_frame,list(frozen['settings']['models']))
    for name,data in [('scores',frame),('blocks',pd.DataFrame(blocks)),('parameters',pd.DataFrame(parameters)),
                      ('test_contrasts',contrast_frame),('aggregate_choice_scores',aggregate)]:data.to_csv(output/(name+'.csv'),index=False)
    manifest=dict(status='complete',fit=str(fit.resolve()),conditional=str(conditional.resolve()),autonomous=str(autonomous.resolve()),
        subjects=sorted(set(k[0] for k in expected)),models=list(frozen['settings']['models']),
        nomination_sha256=nomination_hash,source_sha256=digest(Path(__file__)),
        inputs_sha256={str(p):digest(p) for p in [fit/'nominees.json',conditional/'evaluation.json',autonomous/'evaluation.json']},
        scientific_status='six_preselected_participants_bounded_prefix_fit_with_sequential_test_and_from_start_generation',
        uncertainty='filter-seed jackknife measures numerical uncertainty; parameters remain point estimates',
        distribution_score='empirical equal-weight sampled choice paths; RMS-scaled energy; mean block CRPS; central 80 percent interval')
    with (output/'summary_manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','conditional','autonomous','screen'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.screen:score_screen(args.screen,args.output)
    elif all([args.fit,args.conditional,args.autonomous]):summarize(args.fit,args.conditional,args.autonomous,args.output)
    else:p.error('Supply --screen or all of --fit, --conditional and --autonomous')


if __name__=='__main__':main()
