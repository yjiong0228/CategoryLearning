"""Score behavior-only particle resolution and known-state execution references.

Use --run COMPLETE --output NEW. All fixed and enriched cases remain separate.
Reference-state information is explicit; it is not an achievable recall ceiling
for behavior-only filtering. Targeted event windows retain their full prefixes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_1024_behavior_recovery as study
from . import model_1023_execution_recovery_score as shared
from .model_1024_state_oral_summary import error_scores, window_states, window_truth
from .model_1022_belief_summary import nll, jackknife_se

read, write, digest = study.read, study.write, study.digest


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, bool)
    if mask.ndim != 1:
        raise ValueError('Require one ordered event mask')
    difference = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(difference==1).tolist(), np.flatnonzero(difference==-1).tolist()))


def core(part: dict, n: int) -> dict:
    return {k: v if k in ('seeds','target_rules') else v[:n]
            for k,v in part.items() if not k.startswith('monitor_')}


def load_raw(row: dict) -> dict:
    truth = row['truth']; file = Path(truth['states']).with_suffix('.npz')
    if digest(file) != truth['states_sha256']:
        raise ValueError('Generating truth changed')
    with np.load(file) as z:
        return {k:z[k].copy() for k in z.files}


def reference_state(reference: dict, raw: dict, targets: list[int], rules: int) -> dict:
    """Known continuous state; current execution remains inferred from choice."""
    n = len(reference['choices']); truth = shared.truth_states(raw,n,rules,np.array(targets))
    execution = np.zeros((n,rules)); prior = np.zeros_like(execution)
    ids = raw['rule'][:n]; t,k = np.nonzero(ids>=0)
    execution[t,ids[t,k]] = reference['execution_posterior'][t,k]
    prior[t,ids[t,k]] = reference['execution_prior'][t,k]
    np.testing.assert_allclose(execution.sum(1),1.,atol=1e-12)
    np.testing.assert_allclose(prior.sum(1),1.,atol=1e-12)
    return dict(predictive_belief=truth['belief'], target_rules=np.array(targets),
        choices=reference['choices'], categories=reference['categories'],
        conditional_decision_sources=reference['conditional_decision_sources'],
        filtered_execution=execution, predictive_execution=prior)


def reference_metrics(state: dict, raw: dict) -> list[dict]:
    rows,_ = error_scores(state,raw)
    wrong = state['choices'] != state['categories']; p=state['conditional_decision_sources'][wrong]
    truth=shared.truth_states(raw,len(wrong),state['predictive_belief'].shape[1],state['target_rules'])
    labels=np.where(~truth['present'],0,np.where(truth['target_execution']>.5,2,1))[wrong]
    if len(labels):
        rows.append(dict(timing='filtered',state='error_source',trials=len(labels),
            accuracy=float((p.argmax(1)==labels).mean()),
            brier=float(np.square(p-np.eye(3)[labels]).sum(1).mean()),
            log_loss=float(-np.log(np.maximum(p[np.arange(len(labels)),labels],1e-12)).mean())))
    for timing in ('predictive','filtered'):
        p=state[timing+'_execution'];identity=raw['executed'][:len(p)].astype(int)
        chosen=p[np.arange(len(p)),identity]
        rows.append(dict(timing=timing,state='exact_execution_identity',trials=len(p),
            accuracy=float((p.argmax(1)==identity).mean()),log_loss=float(-np.log(np.maximum(chosen,1e-12)).mean())))
    return rows


def score_state(parts: list[dict],raw: dict,start: int,stop: int) -> list[dict]:
    pooled=shared.pool_states([window_states(core(p,len(p['choices'])),start,stop) for p in parts])
    truth=window_truth(raw,start,stop)
    rows,_=shared.state_metrics(pooled,truth,'STICKY');extra,_=error_scores(pooled,truth)
    return rows+extra


def episode_scores(state: dict, raw: dict, full_n: int) -> list[dict]:
    """Score threshold attainment within true spells, with censoring explicit."""
    targets=state['target_rules'];present=np.isin(raw['rule'],targets).any(1)
    n=len(state['choices']);result=[]
    for index,(a,b) in enumerate(runs(present)):
        if a>=n:continue
        stop=min(b,n);p=state['filtered_target_present'][a:stop];detected=np.flatnonzero(p>=.5)
        result.append(dict(episode=index+1,entry_trial=a+1,last_present_trial=b,exit_trial=b+1 if b<full_n else None,
            true_length=b-a,observed_length=stop-a,right_censored=b>n or b==full_n,
            detected=bool(len(detected)),first_detection_trial=a+int(detected[0])+1 if len(detected) else None,
            detection_delay=int(detected[0]) if len(detected) else None,
            detection_definition='first probability >= 0.5 within true spell; not an inferred entry transition',
            mean_presence_probability=float(p.mean()),max_presence_probability=float(p.max()),
            false_negative_fraction=float((p<.5).mean())))
    return result


def summarize(run: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new behavioral summary directory')
    protocol=read(run/'protocol.json');done=read(run/'completion.json')
    prepared=Path(protocol['prepared']);preparation=read(prepared/'protocol.json')
    if (done['status']!='complete' or done['protocol_sha256']!=digest(run/'protocol.json')
            or preparation['sources']!=study.sources() or protocol['preparation_sha256']!=digest(prepared/'protocol.json')):
        raise ValueError('Require complete unchanged behavior-only study')
    output.mkdir(parents=True);smoke=protocol['smoke'];jobs=protocol['jobs']
    source=study.ROOT/preparation['policy']['inference'];old=read(source/'protocol.json')
    old_cases={c['definition']['case_id']:c for c in old['cases']}
    metrics=[];information=[];trials=[];episodes=[];functional=[];bindings={}
    for row in preparation['oracle']:
        cid=row['case_id'];job=next(j for j in jobs if j['kind']=='oracle' and j['recovery_case']==cid)
        stem=Path(job['target']);reference=shared.checked_arrays(stem);meta=read(stem.with_suffix('.json'))
        if not meta['exact_generation_replay'] or meta['candidate']['id']!=row['candidate']['id']:
            raise ValueError('Reference lost its generating identity')
        raw=load_raw(row);n=len(reference['choices']);c=old_cases[cid];stems=c['states'][row['candidate']['id']]
        parts=[core(shared.checked_arrays(Path(f),'array_sha256'),n) for f in stems]
        for part in parts:
            np.testing.assert_array_equal(part['choices'],reference['choices'])
        pooled=shared.pool_states(parts);rules=pooled['predictive_belief'].shape[1]
        known=reference_state(reference,raw,row['target_rules'],rules)
        base=dict(case_id=cid,subject=row['subject'],scope=row['scope'])
        truth_state=shared.truth_states(raw,n,rules,np.array(row['target_rules']))
        functional.append(dict(**base,trials=n,
            true_presence_trials=int(truth_state['present'].sum()),
            filtered_presence_probability_sum=float(pooled['filtered_target_present'].sum()),
            presence_fraction_error=float(pooled['filtered_target_present'].mean()-truth_state['present'].mean()),
            true_target_execution_trials=float(truth_state['target_execution'].sum()),
            filtered_target_execution_probability_sum=float(pooled['filtered_target_execution'].sum()),
            true_mean_target_belief=float(truth_state['target_belief'].mean()),
            filtered_mean_target_belief=float(pooled['filtered_target_belief'].mean())))
        windows=[('whole',0,n)] if smoke else [('whole',0,n),('training',0,min(n,c['definition']['training_trials'])),('test',min(n,c['definition']['training_trials']),n)]
        for name,a,b in windows:
            if a==b:continue
            for role,state in [('behavior_filter',pooled),('known_state_reference',known)]:
                own=window_states(state,a,b);own_raw=window_truth(raw,a,b)
                result=reference_metrics(own,own_raw)
                metrics.extend(dict(**base,role=role,window=name,**r) for r in result)
            information.append(dict(**base,window=name,trials=b-a,
                mean_information_nats=float(reference['information'][a:b].mean()),
                mean_optimal_identity_accuracy=float(reference['optimal_accuracy'][a:b].mean()),
                mean_prior_optimal_accuracy=float(reference['prior_optimal_accuracy'][a:b].mean()),
                mean_choice_information_accuracy_gain=float((reference['optimal_accuracy'][a:b]-reference['prior_optimal_accuracy'][a:b]).mean())))
        labels=np.where(~reference['present'],0,np.where(np.isin(reference['true_executed'],row['target_rules']),2,1))
        for t in range(n):
            record=dict(**base,trial=t+1,wrong=bool(reference['choices'][t]!=reference['categories'][t]),
                true_present=bool(reference['present'][t]),true_error_source=int(labels[t]),
                behavior_presence=pooled['filtered_target_present'][t],information_nats=reference['information'][t])
            for role,state in [('behavior',pooled),('known_state',known)]:
                for k in range(3):record[role+'_source_'+str(k)]=state['conditional_decision_sources'][t,k]
            trials.append(record)
        episodes.extend(dict(**base,role='behavior_filter',particles=512,seeds=4,**r)
                        for r in episode_scores(pooled,raw,len(raw['choices'])))
        bindings[str(stem.with_suffix('.npz'))]=meta['predictions_sha256']
    numerical=[];drifts=[];monitor=[];ncurves=[];nll_rows=[]
    for row in preparation['numerical']:
        cid=row['case_id'];raw=load_raw(row);n=8 if smoke else row['prefix_trials']
        a,b=(0,8) if smoke else (row['window_start'],row['window_stop']);seeds=row['seeds'][:1] if smoke else row['seeds']
        levels=[2] if smoke else preparation['policy']['particles'];by_level={}
        base=dict(case_id=cid,subject=row['subject'],scope=row['scope'],window_start=a+1,window_stop=b,prefix_trials=n)
        for level in levels:
            parts=[]
            for seed in seeds:
                ref=next((r for r in row['references'] if r['particles']==level and r['seed']==seed),None)
                if ref:
                    stem=Path(ref['state'])
                    if digest(stem.with_suffix('.npz'))!=ref['state_sha256']:raise ValueError('Saved state changed')
                    control=next((j for j in jobs if j['kind']=='prefix_control' and j['recovery_case']==cid
                                  and j['particles']==level and j['seeds']==[seed]),None)
                    if control:
                        stem=Path(control['state_capture_target'])
                else:
                    job=next(j for j in jobs if j['kind']=='filter' and j['recovery_case']==cid and j['particles']==level and j['seeds']==[seed])
                    stem=Path(job['state_capture_target'])
                part=shared.checked_arrays(stem,'array_sha256');meta=read(stem.with_suffix('.json'))
                if meta['candidate']['id']!=row['candidate']['id'] or meta['seeds']!=[seed]:raise ValueError('Numerical identity changed')
                np.testing.assert_array_equal(part['choices'][:n],raw['choices'][:n])
                if 'monitor_filtered_ess' in part:
                    true=np.isin(raw['rule'][:n],row['target_rules']).any(1)[a:b]
                    count=part['monitor_filtered_target_particles'][a:b]
                    monitor.append(dict(**base,particles=level,seed=seed,
                        true_present_trials=int(true.sum()),mean_ess_fraction=float((part['monitor_filtered_ess'][a:b]/level).mean()),
                        min_ess_fraction=float((part['monitor_filtered_ess'][a:b]/level).min()),
                        mean_target_particles=float(count.mean()),zero_target_particles_when_true=int(((count==0)&true).sum()),
                        mean_target_ess=float(part['monitor_filtered_target_ess'][a:b].mean())))
                parts.append(core(part,n));bindings[str(stem.with_suffix('.npz'))]=meta['array_sha256']
            by_level[level]=parts;state=shared.pool_states(parts)
            rows=score_state(parts,raw,a,b)
            deleted=[score_state(parts[:i]+parts[i+1:],raw,a,b) for i in range(len(parts))] if len(parts)>1 else []
            for r in rows:
                for field in ('brier','log_loss','accuracy','hard_recall','true_event_probability'):
                    if field in r and deleted:
                        values=[next(x for x in v if (x['timing'],x['state'])==(r['timing'],r['state']))[field] for v in deleted]
                        r[field+'_mcse']=jackknife_se(np.array(values)) if np.isfinite(values).all() else np.nan
                numerical.append(dict(**base,particles=level,filter_seeds=len(parts),**r))
            probabilities=np.stack([p['prediction'][np.arange(n),p['choices']-1] for p in parts])
            nll_rows.append(dict(**base,particles=level,prefix_nll=nll(probabilities),window_nll=nll(probabilities[:,a:b])))
            episodes.extend(dict(case_id=cid,subject=row['subject'],scope=row['scope'],role='numerical_prefix',particles=level,seeds=len(parts),**r)
                            for r in episode_scores(state,raw,len(raw['choices'])))
            for t in range(n):
                record=dict(**base,particles=level,trial=t+1,in_window=a<=t<b,
                    true_present=bool(np.isin(raw['rule'][t],row['target_rules']).any()),
                    presence=state['filtered_target_present'][t],source_other=state['conditional_decision_sources'][t,1])
                ncurves.append(record)
        for lo,hi in zip(levels[:-1],levels[1:]):
            low=shared.pool_states(by_level[lo]);high=shared.pool_states(by_level[hi])
            for field in ('filtered_target_present','filtered_target_execution','conditional_decision_sources'):
                difference=np.abs(high[field][a:b]-low[field][a:b])
                drifts.append(dict(**base,particles_from=lo,particles_to=hi,field=field,
                    mean_absolute_change=float(difference.mean()),max_absolute_change=float(difference.max())))
    frames=dict(reference_metrics=metrics,reference_information=information,reference_trials=trials,functional_recovery=functional,
        state_episodes=episodes,numerical_metrics=numerical,numerical_drift=drifts,
        particle_support=monitor,numerical_curves=ncurves,numerical_likelihood=nll_rows)
    for name,rows in frames.items():pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    calibration=[]
    fixed=pd.DataFrame(trials).loc[lambda x:x.scope.eq('fixed')&x.wrong]
    for role in ('behavior','known_state'):
        for k in range(3):
            p=fixed[role+'_source_'+str(k)].to_numpy();y=fixed.true_error_source.eq(k).to_numpy()
            bins=np.clip((p*10).astype(int),0,9)
            for index in range(10):
                mask=bins==index
                calibration.append(dict(role=role,error_source=k,bin=index,trials=int(mask.sum()),
                    cases=int(fixed.loc[mask,'case_id'].nunique()),positive_events=int(y[mask].sum()),
                    mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
                    observed_fraction=float(y[mask].mean()) if mask.any() else np.nan))
    pd.DataFrame(calibration).to_csv(output/'reference_calibration.csv',index=False)
    audits=[]
    for file in sorted((run/'jobs').glob('*.json')):
        job=read(file);stem=Path(job['target']);meta=read(stem.with_suffix('.json'))
        if digest(stem.with_suffix('.npz'))!=meta['predictions_sha256']:raise ValueError('Saved numeric output changed')
        if any(pool.get('num_threads',1)!=1 for pool in meta['numeric_pools']):raise ValueError('Numerical worker has multiple threads')
        mechanism=read(stem.with_suffix('.mechanism.json'))
        if mechanism['oral_used_for_fitting']:raise ValueError('Oral observations entered fitting')
        expected=meta['trials']*(1 if job['kind']=='oracle' else meta['particles'])
        if mechanism['audit']['transitions']!=expected or mechanism['audit']['feedback_updates']!=expected:
            raise ValueError('Mechanism update counts changed')
        audits.append(dict(name=job['name'],kind=job['kind'],transitions=expected,seconds=meta['seconds'],cpu_seconds=meta['cpu_seconds']))
    pd.DataFrame(audits).to_csv(output/'numeric_audit.csv',index=False)
    write(output/'completion.json',dict(status='complete',smoke=smoke,oracle_cases=len(preparation['oracle']),
        numerical_cases=len(preparation['numerical']),jobs=done['jobs'],oral_status='deferred_pending_user_manual_review',
        source_sha256=digest(Path(__file__)),run_completion_sha256=digest(run/'completion.json'),
        input_sha256=bindings,artifacts_sha256={f.name:digest(f) for f in output.glob('*.csv')},
        reference_scope='exact conditioning on known pre-execution state; not a behavior-only recovery ceiling',
        functional_scope='sum of online filtering probabilities with trial-specific histories; descriptive occupancy comparison',
        episode_detection_scope='first probability >= 0.5 within a true spell; persistence scored separately',
        numerical_scope='targeted development windows; same true parameters and paired filter seeds'))
    print('Behavior-only references, episodes, particle support and numerical comparisons checked',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summarize(args.run,args.output)
