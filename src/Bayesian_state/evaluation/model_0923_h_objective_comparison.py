"""Evaluate frozen objective fits on shared autonomous behavior features.

The reported held-out energy uses the same predeclared feature definition as
training, applied only to future trials. Whole-path jackknife comparisons use
matched independent generation seeds. Training confirmations and actual fit
costs are reported separately from future behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import load,digest,checked_arrays,nll,jackknife_scores,jackknife_se
from ..metrics.autonomous_fit import behavior_features,energy_u


def feature_jackknife(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Delete independent complete paths, keeping each path's joint features."""
    return np.array([energy_u(np.delete(samples,i,axis=0),target)['value'] for i in range(len(samples))])


def summarize(fit: Path, conditional: Path, autonomous: Path, output: Path) -> None:
    import json
    if output.exists():raise FileExistsError('Use a new objective evaluation destination')
    frozen=load(fit/'nominees.json');nomination_sha=digest(fit/'nominees.json')
    for folder in (conditional,autonomous):
        if load(folder/'evaluation_protocol.json')['nomination_sha256']!=nomination_sha:
            raise ValueError('Prediction parameters were not frozen together')
        if load(folder/'evaluation.json')['status']!='complete':raise ValueError('Incomplete prediction')
    inputs={};groups=[]
    for folder in (conditional,autonomous):
        rows=load(folder/'evaluation.json')['rows'];groups.append({(r['subject'],r['model']):r for r in rows})
    expected={(r['subject'],r['model']) for r in frozen['nominees']}
    if any(set(g)!=expected for g in groups):raise ValueError('Prediction coverage differs')
    results=[];numeric={};confirmations=[];parameters=[]
    for nominee in frozen['nominees']:
        key=nominee['subject'],nominee['model'];cr,ar=groups[0][key],groups[1][key]
        c,a=checked_arrays(conditional,cr),checked_arrays(autonomous,ar)
        for folder,row in [(conditional,cr),(autonomous,ar)]:
            path=folder/'evaluate'/f'subject_{key[0]}'/key[1]/(row['candidate']['id']+'.npz')
            inputs[str(path)]=digest(path)
        if cr['candidate']['hyperparams']!=nominee['candidate']['hyperparams'] or ar['candidate']['hyperparams']!=nominee['candidate']['hyperparams']:
            raise ValueError('Prediction point differs')
        np.testing.assert_array_equal(c['categories'],a['categories'])
        cut=cr['training_trials'];end=len(c['choices']);kwargs=dict(n_categories=2 if cr['condition']==1 else 4,**frozen['settings']['features'])
        for phase,first,last in [('training',0,cut),('test',cut,end),('whole',0,end)]:
            categories=c['categories'][first:last]
            target=behavior_features(c['choices'][None,first:last],categories,**kwargs)[0]
            samples=behavior_features(a['choices'][:,first:last],categories,**kwargs)
            score=energy_u(samples,target)
            results.append(dict(subject=key[0],model=key[1],condition=cr['condition'],phase=phase,
                trials=last-first,paths=len(samples),feature_dimensions=len(target),energy_u=score['value'],mc_se=score['mc_se']))
            if phase=='test':numeric[key]=dict(score=score['value'],jackknife=feature_jackknife(samples,target),seeds=a['seeds'])
        for arm in ('NLL','GEN'):
            path=fit/'confirmation'/f'subject_{key[0]}'/arm/(nominee['candidate']['id']+'.json')
            row=load(path);inputs[str(path)]=digest(path)
            if digest(path.with_suffix('.npz'))!=row['predictions_sha256']:raise ValueError('Confirmation changed')
            with np.load(path.with_suffix('.npz')) as saved:
                if arm=='NLL':
                    value=nll(saved['observed_probability']);se=jackknife_se(jackknife_scores(saved['observed_probability']))
                    np.testing.assert_allclose(value,row['mean_nll'],atol=1e-14,rtol=0)
                else:
                    score=energy_u(saved['features'],saved['observed_features']);value,se=score['value'],score['mc_se']
                    np.testing.assert_allclose(value,row['objective_score'],atol=1e-14,rtol=0)
            confirmations.append(dict(subject=key[0],selected_by=key[1],evaluated_by=arm,
                candidate=nominee['candidate']['id'],training_score=value,training_score_mc_se=se))
        for parameter,value in nominee['candidate']['hyperparams'].items():
            parameters.append(dict(subject=key[0],model=key[1],parameter=parameter,value=value))
    contrasts=[]
    for subject in sorted({s for s,_ in expected}):
        left,right=numeric[(subject,'GEN')],numeric[(subject,'NLL')]
        np.testing.assert_array_equal(left['seeds'],right['seeds'])
        contrasts.append(dict(subject=subject,model='GEN',reference='NLL',
            delta_energy_u=left['score']-right['score'],delta_mc_se=jackknife_se(left['jackknife']-right['jackknife'])))
    costs=[]
    for file in sorted(fit.glob('*/subject_*/*/*.json')):
        row=load(file);arm=row['model'];stage=file.relative_to(fit).parts[0]
        paths=row['paths'] if arm=='GEN' else row['particles']*len(row['seeds'])
        if any(pool['num_threads']!=1 for pool in row['numeric_pools']):raise ValueError('Numeric thread budget exceeded')
        costs.append(dict(subject=int(file.parent.parent.name.removeprefix('subject_')),model=arm,stage=stage,
            candidate=row['candidate']['id'],state_updates=row['trials']*paths,
            cpu_seconds=row['cpu_seconds'],wall_seconds=row['seconds']))
    cost=pd.DataFrame(costs);budget=cost.groupby(['subject','stage','model']).state_updates.sum().unstack('model')
    np.testing.assert_array_equal(budget['NLL'],budget['GEN'])
    frame=pd.DataFrame(results);contrast=pd.DataFrame(contrasts)
    aggregate=frame.query("phase == 'test'").groupby('model',sort=False).agg(
        participants=('subject','size'),mean_energy_u=('energy_u','mean')).reset_index()
    aggregate['delta_from_nll']=aggregate.mean_energy_u-aggregate.loc[aggregate.model=='NLL','mean_energy_u'].item()
    aggregate['delta_mc_se']=np.where(aggregate.model=='GEN',np.linalg.norm(contrast.delta_mc_se)/len(contrast),0.)
    output.mkdir(parents=True)
    for name,data in [('feature_scores',frame),('feature_contrasts',contrast),('feature_aggregate',aggregate),
                      ('training_confirmations',pd.DataFrame(confirmations)),('parameters',pd.DataFrame(parameters)),('fit_costs',cost)]:
        data.to_csv(output/(name+'.csv'),index=False)
    with (output/'completion.json').open('x') as handle:
        json.dump(dict(status='complete',nomination_sha256=nomination_sha,source_sha256=digest(Path(__file__)),
            inputs_sha256=inputs,feature_specification=frozen['settings']['features'],participants=sorted({s for s,_ in expected}),
            paths_per_point=len(next(iter(numeric.values()))['seeds']),paired_generation_seeds=True,
            equal_fitting_state_updates=True,matched_subject_stage_budgets=len(budget),
            data_use_status=frozen['settings']['data_use_status'],uncertainty='fixed-parameter whole-path Monte Carlo error',
            scope='Comparison of predeclared fitting objectives under a fixed ECM_S mechanism'),handle,indent=2)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','conditional','autonomous','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();summarize(args.fit,args.conditional,args.autonomous,args.output)


if __name__=='__main__':main()
