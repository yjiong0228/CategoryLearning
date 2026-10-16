"""Evaluate frozen retention-policy fits, separating development and new cases.

Scores use the full saved filter and autonomous path banks. Choice NLL and
autonomous feature energy retain their established definitions. Paired errors
delete complete filters or complete paths; participant means are descriptive.
Outputs must be new. This analysis never chooses or changes fitted parameters.
The original FREE reference and a separately frozen pooled-search reference
are reported in distinct tables. The latter is read from the combined
evaluation workflow's training_pool_reference.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .model_0923_h_validation import (load,digest,checked_arrays,nll,jackknife_scores,
    jackknife_se,block_paths,distribution_scores,inactive_parameter_channels)
from .model_0923_h_objective_comparison import feature_jackknife
from ..metrics.autonomous_fit import behavior_features,energy_u
from ..workflows.analysis.model_0923_h_validation import write


def cohort_for(subject: int, settings: dict) -> str:
    if subject in settings['development_subjects']:return 'development'
    if subject in settings['new_validation_subjects']:return 'new_validation'
    raise ValueError('Unregistered participant')


def retention_constraint_accepts(point: dict, model: str) -> bool:
    """Check a FREE-space point against a policy's retention restriction."""
    decay=point['engine.modules.hypo_transitions_mod.kwargs.intervention.decay']
    memory=point['engine.modules.memory_mod.kwargs.gamma']
    if model=='FREE':return True
    if model=='CUMULATIVE':return decay==1.
    if model=='SHARED':return decay==memory
    raise ValueError('Unknown retention policy')


def aggregate_contrasts(scores: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (cohort,phase,model),data in scores.groupby(['cohort','phase','model']):
        diff=contrasts[(contrasts.cohort==cohort)&(contrasts.phase==phase)&(contrasts.model==model)]
        assert len(diff)==len(data)
        row=dict(cohort=cohort,phase=phase,model=model,subjects=len(data),trials=int(data.trials.sum()))
        for metric in ('choice_nll','feature_energy'):
            row[metric]=float(data[metric].mean())
            row['delta_'+metric]=float(diff['delta_'+metric].mean())
            row['delta_'+metric+'_mc_se']=float(np.linalg.norm(diff[metric+'_paired_mc_se'])/len(data))
            row['improved_'+metric]=int((diff['delta_'+metric]<0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_prefix_cohorts(scores: pd.DataFrame, contrasts: pd.DataFrame,
                             settings: dict) -> pd.DataFrame:
    """Describe prefix forecasts separately for the prespecified participant groups."""
    keys=['subject','model','start']
    if scores.duplicated(keys).any() or contrasts.duplicated(keys).any():
        raise ValueError('Each participant/model/start must appear once')
    joined=scores.merge(contrasts,on=keys,validate='one_to_one',suffixes=('','_contrast'),indicator=True)
    if len(joined)!=len(scores) or len(joined)!=len(contrasts):
        raise ValueError('Prefix scores and contrasts must cover the same cases')
    joined['cohort']=[cohort_for(int(s),settings) for s in joined.subject]
    rows=[]
    for (cohort,model,start),group in joined.groupby(['cohort','model','start']):
        row=dict(cohort=cohort,model=model,start=start,subjects=len(group))
        for metric in ('energy','crps'):
            row[metric]=float(group[metric].mean())
            row['delta_'+metric]=float(group['delta_'+metric].mean())
            row['delta_'+metric+'_mc_se']=float(np.linalg.norm(group['delta_'+metric+'_mc_se'])/len(group))
            row['improved_'+metric]=int((group['delta_'+metric]<0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def training_diagnostics(fit: Path, frozen: dict, output: Path) -> None:
    rows=[];bank={};precision=[]
    for nominee in frozen['nominees']:
        subject,model=nominee['subject'],nominee['model']
        for index,point in enumerate(nominee['finalists']):
            f=Path(point['score_file']);record=load(f)
            assert digest(f.with_suffix('.npz'))==record['predictions_sha256']
            with np.load(f.with_suffix('.npz')) as a:p=a['observed_probability'];seeds=a['seeds']
            rows.append(dict(subject=subject,cohort=cohort_for(subject,frozen['settings']),model=model,
                candidate=point['id'],selected=index==0,mean_nll=nll(p),mc_se=jackknife_se(jackknife_scores(p)),
                particles=record['particles'],seeds=len(seeds)))
            earlier=fit/'selection'/f'subject_{subject}'/model/(point['id']+'.json')
            low=load(earlier)
            assert digest(earlier.with_suffix('.npz'))==low['predictions_sha256']
            with np.load(earlier.with_suffix('.npz')) as a:
                low_p=a['observed_probability'];low_seeds=a['seeds']
            assert not set(seeds)&set(low_seeds)
            precision.append(dict(subject=subject,model=model,candidate=point['id'],selected=index==0,
                low_particles=low['particles'],high_particles=record['particles'],
                low_seeds=len(low_seeds),high_seeds=len(seeds),low_nll=nll(low_p),high_nll=nll(p),
                delta_nll=nll(p)-nll(low_p),independent_mc_se=float(np.hypot(
                    jackknife_se(jackknife_scores(p)),jackknife_se(jackknife_scores(low_p))))))
            if index==0:bank[subject,model]=(p,seeds)
    pd.DataFrame(rows).to_csv(output/'training_finalists.csv',index=False)
    differences=pd.DataFrame(frozen['numerical_finalists'])
    selected={(r['subject'],r['model']):r['candidate']['id'] for r in frozen['nominees']}
    differences['selected']=[row.candidate==selected[row.subject,row.model] for row in differences.itertuples()]
    differences.to_csv(output/'training_finalist_differences.csv',index=False)
    pd.DataFrame(precision).to_csv(output/'training_precision_change.csv',index=False)
    regret=[]
    free_points={r['subject']:r['candidate']['hyperparams'] for r in frozen['nominees'] if r['model']=='FREE'}
    for (subject,model),(p,seeds) in bank.items():
        base,base_seeds=bank[subject,'FREE'];np.testing.assert_array_equal(seeds,base_seeds)
        regret.append(dict(subject=subject,cohort=cohort_for(subject,frozen['settings']),model=model,
            delta_nll_from_free=nll(p)-nll(base),
            paired_mc_se=jackknife_se(jackknife_scores(p)-jackknife_scores(base)),
            free_can_represent_point=True,
            free_reference_feasible_in_model=retention_constraint_accepts(free_points[subject],model)))
    pd.DataFrame(regret).to_csv(output/'nested_search_diagnostic.csv',index=False)
    # Actual scientific work, excluding aggregate files that pool the chunks.
    budget=[]
    for name in ('screen','refine1','refine2','selection','confirmation'):
        for file in sorted((fit/name).glob('subject_*/*/chunks/*.json')):
            r=load(file)
            assert all(pool['num_threads']==1 for pool in r['numeric_pools'])
            budget.append(dict(subject=r['subject'],model=r['model'],stage=name,
                particles=r['particles'],seeds=len(r['seeds']),trials=r['trials'],
                state_updates=r['particles']*len(r['seeds'])*r['trials'],cpu_seconds=r['cpu_seconds']))
    b=pd.DataFrame(budget)
    grouped=b.groupby(['subject','model','stage'],as_index=False).agg(jobs=('seeds','size'),
        state_updates=('state_updates','sum'),cpu_seconds=('cpu_seconds','sum'))
    assert (grouped.groupby(['subject','stage']).state_updates.nunique()==1).all()
    grouped.to_csv(output/'training_budgets.csv',index=False)


def summarize(fit: Path, conditional: Path, autonomous: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new retention-policy summary')
    frozen=load(fit/'nominees.json');settings=frozen['settings'];nomination_hash=digest(fit/'nominees.json')
    assert not frozen['smoke'] and frozen['status']=='frozen_before_test'
    inputs={str(fit/'nominees.json'):nomination_hash};evaluations=[]
    pool_file=conditional.parent/'training_pool_reference.json';pool=load(pool_file)
    assert pool['status']=='frozen_before_test' and pool['nomination_sha256']==nomination_hash
    pooled_reference={r['subject']:r['source_model'] for r in pool['rows']}
    pooled_points={r['subject']:r['equivalent_free_hyperparams'] for r in pool['rows']}
    inputs[str(pool_file)]=digest(pool_file)
    expected={(r['subject'],r['model']):r for r in frozen['nominees']}
    for source in (conditional,autonomous):
        assert load(source/'evaluation_protocol.json')['nomination_sha256']==nomination_hash
        record=load(source/'evaluation.json');assert record['status']=='complete'
        indexed={(r['subject'],r['model']):r for r in record['rows']}
        assert len(indexed)==len(record['rows']) and set(indexed)==set(expected)
        inputs[str(source/'evaluation.json')]=digest(source/'evaluation.json');evaluations.append(indexed)
    scores=[];curves=[];parameters=[];paths=[];repeated={};seeds={};details=[]
    for key,nominee in expected.items():
        cr,ar=evaluations[0][key],evaluations[1][key]
        for source,row in [(conditional,cr),(autonomous,ar)]:
            assert row['candidate']['hyperparams']==nominee['candidate']['hyperparams']
            inputs[str(source/'evaluate'/f'subject_{key[0]}'/key[1]/(row['candidate']['id']+'.npz'))]=row['predictions_sha256']
        c,a=checked_arrays(conditional,cr),checked_arrays(autonomous,ar)
        np.testing.assert_array_equal(c['categories'],a['categories'])
        cut=cr['training_trials'];trials=len(c['choices']);actual=c['choices']==c['categories']
        generated=a['choices']==a['categories'][None,:]
        assert a['choices'].shape==(settings['autonomous_paths'],trials)
        assert c['observed_probability'].shape==(settings['budgets']['evaluate']['seeds'],trials)
        tag=dict(subject=key[0],model=key[1],condition=cr['condition'],cohort=cohort_for(key[0],settings))
        for phase,start,stop in [('training',0,cut),('test',cut,trials),('whole',0,trials)]:
            p=c['observed_probability'][:,start:stop]
            feature_kwargs=dict(n_categories=2 if cr['condition']==1 else 4,**settings['features'])
            target=behavior_features(c['choices'][None,start:stop],c['categories'][start:stop],**feature_kwargs)[0]
            features=behavior_features(a['choices'][:,start:stop],a['categories'][start:stop],**feature_kwargs)
            energy=energy_u(features,target)
            blocks=block_paths(generated,start,stop);observed=block_paths(actual[None,:],start,stop)[0]
            metrics=distribution_scores(blocks,observed);low,high=np.quantile(blocks,[.1,.9],axis=0)
            late=min(128,stop-start);late_values=generated[:,stop-late:stop].mean(axis=1)
            row=dict(tag,phase=phase,trials=stop-start,particles=cr['particles'],filter_seeds=len(c['seeds']),
                paths=len(a['seeds']),choice_nll=nll(p),choice_nll_mc_se=jackknife_se(jackknife_scores(p)),
                feature_energy=energy['value'],feature_energy_mc_se=energy['mc_se'],
                observed_late_accuracy=float(actual[stop-late:stop].mean()),late_window=late,
                autonomous_late_accuracy=float(late_values.mean()),
                reaches_observed_late=float(np.mean(late_values>=actual[stop-late:stop].mean())),**metrics)
            scores.append(row)
            repeated[key[0],key[1],phase]=dict(choice_nll=jackknife_scores(p),feature_energy=feature_jackknife(features,target))
            seeds[key[0],key[1],phase]=(c['seeds'],a['seeds'])
            if phase=='whole':
                conditional_mean=block_paths(c['correct_probability'],start,stop).mean(axis=0)
                for b in range(blocks.shape[1]):
                    curves.append(dict(tag,block=b+1,first_trial=b*32+1,last_trial=min((b+1)*32,trials),
                        training_trials=cut,observed=observed[b],conditional=conditional_mean[b],
                        autonomous=blocks[:,b].mean(),q10=low[b],q90=high[b]))
                for i,seed in enumerate(a['seeds']):
                    paths.append(dict(tag,seed=int(seed),late_accuracy=late_values[i],observed_late_accuracy=row['observed_late_accuracy']))
        point=nominee['candidate']['hyperparams'];decay=settings['models'][key[1]]['decay']
        effective=point.get('engine.modules.hypo_transitions_mod.kwargs.intervention.decay',decay)
        if effective=='memory':effective=point['engine.modules.memory_mod.kwargs.gamma']
        details.append(dict(tag,effective_evidence_decay=effective,memory_gamma=point['engine.modules.memory_mod.kwargs.gamma'],
            evidence_policy={'FREE':'estimated','CUMULATIVE':'cumulative','SHARED':'memory'}[key[1]],
            confirmation_nll=nominee['candidate']['mean_nll'],
            evaluation_training_nll=nll(c['observed_probability'][:,:cut]),
            capacity=point['engine.modules.hypo_transitions_mod.kwargs.max_capacity'],
            retention_channel_closed=point['engine.modules.hypo_transitions_mod.kwargs.max_capacity']==1))
        parameters.extend(dict(tag,parameter=k,value=value) for k,value in point.items())
    frame=pd.DataFrame(scores);contrasts=[];pooled_contrasts=[]
    for row in frame.itertuples():
        key=row.subject,row.model,row.phase;base_key=row.subject,'FREE',row.phase
        for left,right in zip(seeds[key],seeds[base_key]):np.testing.assert_array_equal(left,right)
        base=frame[(frame.subject==row.subject)&(frame.model=='FREE')&(frame.phase==row.phase)].iloc[0]
        d=dict(subject=row.subject,model=row.model,phase=row.phase,cohort=row.cohort)
        for metric in ('choice_nll','feature_energy'):
            d['delta_'+metric]=getattr(row,metric)-base[metric]
            d[metric+'_paired_mc_se']=jackknife_se(repeated[key][metric]-repeated[base_key][metric])
        contrasts.append(d)
        source_model=pooled_reference[row.subject];base_key=row.subject,source_model,row.phase
        base=frame[(frame.subject==row.subject)&(frame.model==source_model)&(frame.phase==row.phase)].iloc[0]
        d=dict(subject=row.subject,model=row.model,phase=row.phase,cohort=row.cohort,reference_source_model=source_model,
            pooled_reference_feasible_in_model=retention_constraint_accepts(pooled_points[row.subject],row.model))
        for metric in ('choice_nll','feature_energy'):
            d['delta_'+metric]=getattr(row,metric)-base[metric]
            d[metric+'_paired_mc_se']=jackknife_se(repeated[key][metric]-repeated[base_key][metric])
        pooled_contrasts.append(d)
    contrast=pd.DataFrame(contrasts);output.mkdir(parents=True)
    pooled_contrast=pd.DataFrame(pooled_contrasts)
    for name,table in [('scores',frame),('contrasts',contrast),('aggregate',aggregate_contrasts(frame,contrast)),
        ('curves',pd.DataFrame(curves)),('paths',pd.DataFrame(paths)),('parameters',pd.DataFrame(parameters)),
        ('retention_parameters',pd.DataFrame(details)),('inactive_parameters',inactive_parameter_channels(frozen))]:
        table.to_csv(output/(name+'.csv'),index=False)
    pooled_contrast.to_csv(output/'pooled_search_contrasts.csv',index=False)
    aggregate_contrasts(frame,pooled_contrast).to_csv(output/'pooled_search_aggregate.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k!='equivalent_free_hyperparams'} for row in pool['rows']]).to_csv(
        output/'pooled_training_reference.csv',index=False)
    training_diagnostics(fit,frozen,output);shutil.copy2(__file__,output/'analysis_source.py')
    write(output/'completion.json',dict(status='complete',subjects=sorted({k[0] for k in expected}),
        models=list(settings['models']),settings=settings,nomination_sha256=nomination_hash,
        inputs_sha256=inputs,source_sha256=digest(Path(__file__)),
        data_use='Old six participants are development; three new H-study participants are reported separately.',
        limitations='Bounded search, fixed stimulus replay, retrospective prefix boundary, point estimates, no population confidence intervals.'))
    print(aggregate_contrasts(frame,contrast).query("phase=='test'").to_string(index=False))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','conditional','autonomous','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summarize(a.fit,a.conditional,a.autonomous,a.output)


if __name__=='__main__':main()
