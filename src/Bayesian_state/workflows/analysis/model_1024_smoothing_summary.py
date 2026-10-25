"""Score known-parameter filtering and retrospective genealogical smoothing.

Use --run COMPLETE --benchmark COMPLETE --output NEW. Independent normalized
filter estimates are averaged equally; common-endpoint path weights retain
their filter identity. Oral observations are deferred.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_1024_smoothing as study
from . import model_1023_execution_recovery_score as scores
from .model_1024_behavior_recovery_summary import runs
from ...inference.backends.particle_filter import _weighted_quantiles

read, write, digest = study.read, study.write, study.digest
MODES=('filtered','lag','prefix','full')


def truth_events(raw: dict, targets: list[int], n: int) -> dict:
    present=np.isin(raw['rule'][:n],targets).any(axis=1)
    belief=(raw['belief'][:n]*np.isin(raw['rule'][:n],targets)).sum(axis=1)
    execution=np.isin(raw['executed'][:n],targets)
    return dict(present=present,target_belief=belief,high_belief=present & (belief>=.5),
        target_execution=execution,source=np.where(~present,0,np.where(execution,2,1)),
        wrong=raw['choices'][:n]!=raw['categories'][:n])


def metric_rows(state: dict, truth: dict) -> list[dict]:
    rows=[]
    for key in ('present','high_belief','target_execution'):
        rows.append(dict(state=key,**scores.binary_scores(state[key],truth[key])))
    rows.append(dict(state='target_belief',trials=len(truth['target_belief']),
        rmse=float(np.sqrt(np.square(state['target_belief']-truth['target_belief']).mean()))))
    wrong=truth['wrong'];labels=truth['source'][wrong];p=state['source'][wrong]
    if len(labels):
        study.helper.probabilities(p)
        selected=p.argmax(axis=1);positive=labels==1;chosen=selected==1
        rows.append(dict(state='error_available_other',**scores.binary_scores(p[:,1],positive),
            hard_true_positive=int((chosen&positive).sum()),hard_false_positive=int((chosen&~positive).sum()),
            hard_recall=float(chosen[positive].mean()) if positive.any() else np.nan,
            hard_precision=float(positive[chosen].mean()) if chosen.any() else np.nan,
            true_event_probability=float(p[positive,1].mean()) if positive.any() else np.nan))
        rows.append(dict(state='error_source',trials=len(labels),accuracy=float((selected==labels).mean()),
            brier=float(np.square(p-np.eye(3)[labels]).sum(axis=1).mean()),
            log_loss=float(-np.log(np.maximum(p[np.arange(len(labels)),labels],1e-12)).mean())))
    return rows


def coherent_features(present: np.ndarray, execution: np.ndarray, belief: np.ndarray) -> dict:
    """One row per coherent terminal path; no-entry uses endpoint+1 censor code."""
    present=np.asarray(present,bool)
    if present.ndim!=2 or execution.shape!=present.shape or belief.shape!=present.shape or len(present)<1:
        raise ValueError('Require aligned trial by path arrays')
    n,paths=present.shape;current=np.zeros(paths,int);longest=current.copy();entries=current.copy()
    for t in range(n):
        current=np.where(present[t],current+1,0);longest=np.maximum(longest,current)
        entries+=present[t] & (~present[t-1] if t else True)
    ever=present.any(axis=0)
    return dict(presence_trials=present.sum(axis=0),target_execution_trials=execution.sum(axis=0),
        mean_target_belief=belief.mean(axis=0),longest_presence=longest,entry_count=entries,
        first_entry_trial=np.where(ever,present.argmax(axis=0)+1,n+1),no_entry=~ever)


def summarize(run: Path, benchmark: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new smoothing summary directory')
    protocol=read(run/'protocol.json');done=read(run/'completion.json');bench=read(benchmark/'completion.json')
    if (done['status']!='complete' or done['smoke'] or protocol['sources']!=study.sources()
            or done['protocol_sha256']!=digest(run/'protocol.json') or bench['status']!='complete'):
        raise ValueError('Require unchanged formal smoothing and exact benchmark results')
    for file,sha in protocol['inputs'].items():
        if digest(Path(file))!=sha:raise ValueError('Registered source protocol changed')
    for file,sha in bench['files'].items():
        if digest(benchmark/file)!=sha:raise ValueError('Exact benchmark artifact changed')
    output.mkdir(parents=True);metrics=[];seed_metrics=[];curves=[];episodes=[];path_rows=[];functionals=[];numeric=[];bindings={}
    for case in protocol['cases']:
        cid=case['case_id'];truth=case['truth'];file=Path(truth['states']).with_suffix('.npz')
        if digest(file)!=truth['states_sha256']:raise ValueError('Generating truth changed')
        with np.load(file) as z:raw={k:z[k].copy() for k in z.files}
        jobs=[j for j in protocol['jobs'] if j['recovery_case']==cid];parts=[]
        n=len(raw['choices']);events=truth_events(raw,case['target_rules'],n)
        base=dict(case_id=cid,subject=case['subject'],scope=case['scope'])
        for job in jobs:
            stem=Path(job['genealogy_target']);meta=read(stem.with_suffix('.json'))
            part=scores.checked_arrays(stem,'array_sha256');parts.append(part)
            if meta['trials']!=n or not meta['original_score_replay'] or not meta['original_state_replay']:
                raise ValueError('Incomplete full-path replay')
            score=read(Path(job['target']).with_suffix('.json'))
            if any(pool.get('num_threads',1)!=1 for pool in score['numeric_pools']):
                raise ValueError('Numerical worker exceeded one numeric thread')
            audit_file=Path(job['target']+'.mechanism.json');audit=read(audit_file)['audit']
            expected=n*job['particles']
            if (audit['transitions']!=expected or audit['feedback_updates']!=expected
                    or audit['error_entry']['execution']['trials']!=expected):
                raise ValueError('Shared scientific mechanism update count changed')
            numeric.append(dict(**base,seed=job['seeds'][0],trials=n,particles=job['particles'],
                seconds=score['seconds'],cpu_seconds=score['cpu_seconds'],
                updates=n*job['particles'],resampling_steps=meta['resampling_steps'],
                last_trial_resampled=meta['last_trial_resampled']))
            bindings[str(stem.with_suffix('.npz'))]=meta['array_sha256']
            bindings[str(audit_file)]=digest(audit_file)
            for mode in ('prefix','full'):
                indices=part[mode+'_indices'];end=indices.shape[1];w=part['filtered_weights'][end-1]
                vals={key:study.helper.trace_values(part['particle_'+key][:end],indices)
                      for key in ('present','target_execution','target_belief')}
                features=coherent_features(vals['present'],vals['target_execution'],vals['target_belief'])
                actual_features=coherent_features(events['present'][:end,None],events['target_execution'][:end,None],
                    events['target_belief'][:end,None])
                own=dict(**base,mode=mode,seed=job['seeds'][0],endpoint=end)
                path_rows.extend(dict(**own,terminal_slot=i,weight=float(w[i]),
                    **{k:float(v[i]) for k,v in features.items()}) for i in range(len(w)))
                # A strict overlap quantity for the first true episode under
                # this endpoint. It describes a complete event in each path.
                first=runs(events['present'][:end])
                covers=np.all(vals['present'][first[0][0]:first[0][1]],axis=0) if first else np.zeros(len(w),bool)
                full_first=runs(events['present'])
                episode_censored=bool(full_first and (full_first[0][1]>end or full_first[0][1]==n))
                entered=~features['no_entry'];entered_mass=float(w@entered)
                for name,values in features.items():
                    actual=actual_features[name][0]
                    q=_weighted_quantiles(values,w,[.05,.5,.95])
                    functionals.append(dict(**own,function=name,true_value=float(actual),
                        posterior_mean=float(w@values),q05=float(q[0]),q50=float(q[1]),q95=float(q[2]),
                        truth_in_90_interval=bool(q[0]<=actual<=q[2]),
                        no_entry_mass=float(w@features['no_entry']),
                        first_true_episode_covered_mass=float(w@covers) if first else np.nan,
                        first_true_episode_observed_length=first[0][1]-first[0][0] if first else 0,
                        first_true_episode_right_censored=episode_censored,
                        conditional_first_entry_mean=float(w[entered]@features['first_entry_trial'][entered]/entered_mass) if entered_mass else np.nan,
                        unique_presence_paths=int(len(np.unique(vals['present'].T,axis=0)))))
        for mode in MODES:
            end=case['prefix_trials'] if mode=='prefix' else n
            pool={k:np.mean([p[mode+'_'+k] for p in parts],axis=0)
                  for k in ('present','high_belief','target_belief','target_execution','source')}
            sd=np.std([p[mode+'_present'] for p in parts],axis=0,ddof=1)
            windows=[('entry_window',case['window_start'],case['window_stop']),('available_span',0,end)]
            for window,a,b in windows:
                own={k:v[a:b] for k,v in pool.items()};known={k:v[a:b] for k,v in events.items()}
                for row in metric_rows(own,known):
                    metrics.append(dict(**base,mode=mode,window=window,endpoint=end,**row))
                for job,part in zip(jobs,parts):
                    one={k:part[mode+'_'+k][a:b] for k in pool}
                    for row in metric_rows(one,known):
                        seed_metrics.append(dict(**base,mode=mode,window=window,seed=job['seeds'][0],endpoint=end,**row))
            for t in range(end):
                ancestors=[p[mode+'_unique_ancestors'][t] for p in parts] if mode!='filtered' else []
                ess=[p[mode+'_ancestral_ess'][t] for p in parts] if mode!='filtered' else []
                curves.append(dict(**base,mode=mode,trial=t+1,
                    observation_endpoint=(min(n,t+protocol['policy']['fixed_lag']+1) if mode=='lag' else t+1 if mode=='filtered' else end),
                    in_entry_window=case['window_start']<=t<case['window_stop'],
                    true_present=bool(events['present'][t]),true_high_belief=bool(events['high_belief'][t]),
                    true_target_execution=bool(events['target_execution'][t]),
                    presence=float(pool['present'][t]),high_belief=float(pool['high_belief'][t]),
                    target_execution=float(pool['target_execution'][t]),available_other=float(pool['source'][t,1]),
                    presence_seed_sd=float(sd[t]),
                    mean_unique_ancestors=float(np.mean(ancestors)) if ancestors else np.nan,
                    min_unique_ancestors=int(min(ancestors)) if ancestors else np.nan,
                    mean_ancestral_ess=float(np.mean(ess)) if ess else np.nan,
                    min_ancestral_ess=float(min(ess)) if ess else np.nan))
            for i,(a,b) in enumerate(runs(events['present'])):
                if a>=end:continue
                stop=min(end,b);p=pool['present'][a:stop];hit=np.flatnonzero(p>=.5)
                episodes.append(dict(**base,mode=mode,episode=i+1,entry_trial=a+1,true_length=b-a,
                    observed_length=stop-a,right_censored=b>end or b==n,
                    first_threshold_trial=a+int(hit[0])+1 if len(hit) else None,
                    threshold_delay=int(hit[0]) if len(hit) else None,detected=bool(len(hit)),
                    mean_probability=float(p.mean()),max_probability=float(p.max()),
                    false_negative_fraction=float((p<.5).mean()),
                    definition='first probability >=0.5 within the true episode; not inferred entry time'))
    frames=dict(metrics=metrics,seed_metrics=seed_metrics,curves=curves,episodes=episodes,
        path_features=path_rows,functionals=functionals,numerical_audit=numeric)
    for name,rows in frames.items():pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    pd.read_csv(benchmark/'metrics.csv').to_csv(output/'benchmark_metrics.csv',index=False)
    pd.read_csv(benchmark/'curves.csv').to_csv(output/'benchmark_curves.csv',index=False)
    write(output/'completion.json',dict(status='complete',smoke=False,cases=len(protocol['cases']),filters=len(protocol['jobs']),
        source_sha256=digest(Path(__file__)),run_completion_sha256=digest(run/'completion.json'),
        analysis_sources={path:digest(study.ROOT/path) for path in (
            str(Path(__file__).resolve().relative_to(study.ROOT)),
            'src/Bayesian_state/workflows/analysis/model_1023_execution_recovery_score.py',
            'src/Bayesian_state/workflows/analysis/model_1024_behavior_recovery_summary.py',
            study.HELPER,study.PF)},
        benchmark_completion_sha256=digest(benchmark/'completion.json'),
        artifacts_sha256={f.name:digest(f) for f in sorted(output.glob('*.csv'))},inputs_sha256=bindings,
        pooling='equal average of independent normalized particle approximations; not parameter averaging',
        path_weighting='each terminal path has its terminal filtered weight within its own filter',
        no_entry_code='first_entry_trial = endpoint+1; no_entry identifies right censoring',
        oral_used=False,conditioning='known generating parameters; filtering and retrospective smoothing separately labeled'))
    print('Smoothing recovery, coherent path features and ancestry diversity scored',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summarize(args.run,args.benchmark,args.output)
