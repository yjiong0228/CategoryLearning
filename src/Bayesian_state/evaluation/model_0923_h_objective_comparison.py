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
    from scipy.spatial.distance import cdist
    x,y=np.asarray(samples,float),np.asarray(target,float)
    if x.ndim!=2 or len(x)<3 or y.shape!=(x.shape[1],) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need at least three finite feature paths and an aligned target')
    n=len(x);distances=cdist(x,x);target_distances=np.linalg.norm(x-y,axis=1)
    pairs=distances.sum()/2
    return ((target_distances.sum()-target_distances)/(n-1)
            -(pairs-distances.sum(axis=1))/((n-1)*(n-2)))


def summarize(fit: Path, conditional: Path, autonomous: Path, output: Path) -> None:
    import json
    import shutil
    if output.exists():raise FileExistsError('Use a new objective evaluation destination')
    frozen=load(fit/'nominees.json');nomination_sha=digest(fit/'nominees.json')
    for folder in (conditional,autonomous):
        if load(folder/'evaluation_protocol.json')['nomination_sha256']!=nomination_sha:
            raise ValueError('Prediction parameters were not frozen together')
        if load(folder/'evaluation.json')['status']!='complete':raise ValueError('Incomplete prediction')
    inputs={};groups=[]
    for folder in (conditional,autonomous):
        rows=load(folder/'evaluation.json')['rows'];groups.append({(r['subject'],r['model']):r for r in rows})
        if len(groups[-1])!=len(rows):raise ValueError('Duplicate prediction group')
    nominees={(r['subject'],r['model']):r for r in frozen['nominees']};expected=set(nominees)
    if len(nominees)!=len(frozen['nominees']):raise ValueError('Duplicate frozen nominee')
    if any(set(g)!=expected for g in groups):raise ValueError('Prediction coverage differs')
    results=[];numeric={};confirmations=[];parameters=[];training_numeric={}
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
                    deleted=jackknife_scores(saved['observed_probability'])
                    np.testing.assert_allclose(value,row['mean_nll'],atol=1e-14,rtol=0)
                else:
                    score=energy_u(saved['features'],saved['observed_features']);value,se=score['value'],score['mc_se']
                    deleted=feature_jackknife(saved['features'],saved['observed_features'])
                    np.testing.assert_allclose(value,row['objective_score'],atol=1e-14,rtol=0)
                training_numeric[(key[0],key[1],arm)]=dict(value=value,deleted=deleted,seeds=saved['seeds'])
            confirmations.append(dict(subject=key[0],selected_by=key[1],evaluated_by=arm,
                candidate=nominee['candidate']['id'],training_score=value,training_score_mc_se=se))
        for parameter,value in nominee['candidate']['hyperparams'].items():
            parameters.append(dict(subject=key[0],model=key[1],parameter=parameter,value=value))
    contrasts=[];training_contrasts=[]
    for subject in sorted({s for s,_ in expected}):
        left,right=numeric[(subject,'GEN')],numeric[(subject,'NLL')]
        np.testing.assert_array_equal(left['seeds'],right['seeds'])
        contrasts.append(dict(subject=subject,model='GEN',reference='NLL',
            delta_energy_u=left['score']-right['score'],delta_mc_se=jackknife_se(left['jackknife']-right['jackknife'])))
        for arm in ('NLL','GEN'):
            left=training_numeric[(subject,'GEN',arm)];right=training_numeric[(subject,'NLL',arm)]
            np.testing.assert_array_equal(left['seeds'],right['seeds'])
            delta=left['value']-right['value']
            if (arm=='NLL' and delta < -1e-12) or (arm=='GEN' and delta > 1e-12):
                raise ValueError('Frozen nominee does not minimize its own confirmation objective')
            training_contrasts.append(dict(subject=subject,evaluated_by=arm,
                gen_selected_minus_nll_selected=delta,paired_mc_se=jackknife_se(left['deleted']-right['deleted'])))
    costs=[];finalists=[]
    for file in sorted(fit.glob('*/subject_*/*/*.json')):
        row=load(file);arm=row['model'];stage=file.relative_to(fit).parts[0]
        paths=row['paths'] if arm=='GEN' else row['particles']*len(row['seeds'])
        if any(pool['num_threads']!=1 for pool in row['numeric_pools']):raise ValueError('Numeric thread budget exceeded')
        costs.append(dict(subject=int(file.parent.parent.name.removeprefix('subject_')),model=arm,stage=stage,
            candidate=row['candidate']['id'],state_updates=row['trials']*paths,
            cpu_seconds=row['cpu_seconds'],wall_seconds=row['seconds']))
        if stage=='confirmation':
            subject=costs[-1]['subject'];reference=training_numeric[(subject,arm,arm)]
            point=nominees[(subject,arm)]['candidate'];inputs[str(file)]=digest(file)
            if digest(file.with_suffix('.npz'))!=row['predictions_sha256']:raise ValueError('Finalist predictions changed')
            with np.load(file.with_suffix('.npz')) as saved:
                np.testing.assert_array_equal(saved['seeds'],reference['seeds'])
                if arm=='NLL':value=nll(saved['observed_probability']);deleted=jackknife_scores(saved['observed_probability'])
                else:value=energy_u(saved['features'],saved['observed_features'])['value'];deleted=feature_jackknife(saved['features'],saved['observed_features'])
            if value < reference['value']-1e-12:raise ValueError('A confirmed candidate improves on the frozen nominee')
            finalists.append(dict(subject=subject,model=arm,candidate=row['candidate']['id'],
                selected=row['candidate']['id']==point['id'],objective_score=value,mc_se=jackknife_se(deleted),
                delta_from_selected=value-reference['value'],paired_delta_mc_se=jackknife_se(deleted-reference['deleted']),
                different_coordinates=sum(parameter_value!=point['hyperparams'][key] for key,parameter_value in row['candidate']['hyperparams'].items())))
    cost=pd.DataFrame(costs);budget=cost.groupby(['subject','stage','model']).state_updates.sum().unstack('model')
    np.testing.assert_array_equal(budget['NLL'],budget['GEN'])
    frame=pd.DataFrame(results);contrast=pd.DataFrame(contrasts)
    aggregate=frame.query("phase == 'test'").groupby('model',sort=False).agg(
        participants=('subject','size'),mean_energy_u=('energy_u','mean')).reset_index()
    aggregate['delta_from_nll']=aggregate.mean_energy_u-aggregate.loc[aggregate.model=='NLL','mean_energy_u'].item()
    aggregate['delta_mc_se']=np.where(aggregate.model=='GEN',np.linalg.norm(contrast.delta_mc_se)/len(contrast),0.)
    output.mkdir(parents=True)
    shutil.copy2(__file__,output/'analysis_source.py')
    for name,data in [('feature_scores',frame),('feature_contrasts',contrast),('feature_aggregate',aggregate),
                      ('training_confirmations',pd.DataFrame(confirmations)),('training_contrasts',pd.DataFrame(training_contrasts)),
                      ('confirmation_candidates',pd.DataFrame(finalists)),('parameters',pd.DataFrame(parameters)),('fit_costs',cost)]:
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
