"""Compare sampled and analytically marginalized zero-inertia execution.

Use --run COMPLETE --output NEW. Every posterior timing, particle level,
case and numerical seed remains explicit. Oral data are deferred.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_1024_marginal_execution as study
from . import model_1023_execution_recovery_score as checked
from .model_1024_smoothing_summary import truth_events,metric_rows,coherent_features
from .model_1024_behavior_recovery_summary import runs
from ...inference.backends.particle_filter import _weighted_quantiles

read,write,digest=study.read,study.write,study.digest
MODES=('filtered','lag','prefix','full')


def count_moments(per_trial: np.ndarray,weights: np.ndarray,*,integrated: bool) -> tuple[float,float,float]:
    """Return count mean, total variance and conditional variance component."""
    per_trial=np.asarray(per_trial,float);weights=study.ancestry.probabilities(weights)
    if (per_trial.ndim!=2 or per_trial.shape[1]!=len(weights) or not np.isfinite(per_trial).all()
            or np.any((per_trial<0)|(per_trial>1)) or not integrated and not np.isin(per_trial,[0.,1.]).all()):
        raise ValueError('Require coherent paths of probabilities or sampled binary execution')
    counts=per_trial.sum(axis=0);within=(per_trial*(1.-per_trial)).sum(axis=0) if integrated else np.zeros(len(weights))
    mean=float(weights@counts);component=float(weights@within)
    return mean,float(weights@np.square(counts-mean)+component),component


def summarize(run: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new marginalized execution summary directory')
    protocol=read(run/'protocol.json');done=read(run/'completion.json')
    if (done['status']!='complete' or done['smoke'] or protocol['sources']!=study.sources()
            or done['protocol_sha256']!=digest(run/'protocol.json')):
        raise ValueError('Require a complete unchanged formal run')
    previous=study.ROOT/protocol['policy']['previous_run']
    if digest(previous/'protocol.json')!=protocol['previous_protocol_sha256']:
        raise ValueError('Previous recovery protocol changed')
    output.mkdir(parents=True)
    tables={k:[] for k in ('metrics','seed_metrics','curves','episodes','functionals',
        'execution_count_moments','path_features','likelihood','numerical_audit','resolution_drift')}
    bindings={}
    for case in protocol['cases']:
        cid=case['case_id'];truth=case['truth'];file=Path(truth['states']).with_suffix('.npz')
        if digest(file)!=truth['states_sha256']:raise ValueError('Generating truth changed')
        with np.load(file) as z:raw={k:z[k].copy() for k in z.files}
        n=len(raw['choices']);events=truth_events(raw,case['target_rules'],n)
        base=dict(case_id=cid,subject=case['subject'],scope=case['scope'])
        groups={('sampled',512):case['baseline_jobs']}
        for job in protocol['jobs']:
            if job['recovery_case']==cid:groups.setdefault(('marginal',job['particles']),[]).append(job)
        pooled={}
        for (method,particles),jobs in groups.items():
            parts=[]
            for job in jobs:
                stem=Path(job['genealogy_target'] if method=='sampled' else job['marginal_target'])
                meta=read(stem.with_suffix('.json'));part=checked.checked_arrays(stem,'array_sha256');parts.append(part)
                bindings[str(stem.with_suffix('.npz'))]=meta['array_sha256']
                score_file=Path(job['target']);score=read(score_file.with_suffix('.json'))
                probability=checked.checked_arrays(score_file)['observed_probability'][0]
                if len(probability)!=n or any(x.get('num_threads',1)!=1 for x in score['numeric_pools']):
                    raise ValueError('Changed filter length or numerical thread budget')
                audit_file=Path(job['target']+'.mechanism.json');audit=read(audit_file)['audit'];expected=n*particles
                executed=audit['error_entry']['execution']['trials']
                if (audit['transitions']!=expected or audit['feedback_updates']!=expected
                        or executed!=(expected if method=='sampled' else 0)):
                    raise ValueError('Scientific state update count or carrier identity changed')
                if method=='marginal' and meta['minimum_particle_choice_mass']<1e-12:
                    raise ValueError('Likelihood floor requires a separate conditioning audit')
                tables['numerical_audit'].append(dict(**base,method=method,particles=particles,seed=job['seeds'][0],
                    trials=n,updates=expected,seconds=score['seconds'],cpu_seconds=score['cpu_seconds'],
                    resampling_steps=meta['resampling_steps'],last_trial_resampled=meta['last_trial_resampled'],
                    minimum_choice_mass=meta.get('minimum_particle_choice_mass',np.nan)))
                bindings[str(audit_file)]=digest(audit_file)
                for window,a,b in [('entry_window',case['window_start'],case['window_stop']),('whole',0,n)]:
                    tables['likelihood'].append(dict(**base,method=method,particles=particles,seed=job['seeds'][0],
                        window=window,trials=b-a,nll=float(-np.log(np.maximum(probability[a:b],1e-12)).mean())))
                for mode in ('prefix','full'):
                    indices=part[mode+'_indices'];end=indices.shape[1];w=part['filtered_weights'][end-1]
                    selected={k:study.ancestry.trace_values(part['particle_'+k][:end],indices)
                        for k in ('present','target_execution','target_belief')}
                    features=coherent_features(selected['present'],selected['target_execution'],selected['target_belief'])
                    actual=coherent_features(events['present'][:end,None],events['target_execution'][:end,None],events['target_belief'][:end,None])
                    mean_count=features.pop('target_execution_trials');truth_count=float(actual.pop('target_execution_trials')[0])
                    conditional_variance=(selected['target_execution']*(1.-selected['target_execution'])).sum(0) if method=='marginal' else np.zeros(len(w))
                    mean,variance,within=count_moments(selected['target_execution'],w,integrated=method=='marginal')
                    own=dict(**base,method=method,particles=particles,mode=mode,seed=job['seeds'][0],endpoint=end)
                    tables['execution_count_moments'].append(dict(**own,true_value=truth_count,posterior_mean=mean,
                        posterior_variance=variance,mean_conditional_variance=within,
                        interpretation='conditional independent execution draws summed exactly within each continuous-state path' if method=='marginal' else 'sampled execution histories'))
                    tables['path_features'].extend(dict(**own,terminal_slot=i,weight=float(w[i]),
                        conditional_execution_count_mean=float(mean_count[i]),conditional_execution_count_variance=float(conditional_variance[i]),
                        **{k:float(v[i]) for k,v in features.items()}) for i in range(len(w)))
                    for key,values in features.items():
                        q=_weighted_quantiles(values,w,[.05,.5,.95]);true=float(actual[key][0])
                        tables['functionals'].append(dict(**own,function=key,true_value=true,posterior_mean=float(w@values),
                            q05=float(q[0]),q50=float(q[1]),q95=float(q[2]),truth_in_90_interval=bool(q[0]<=true<=q[2]),
                            no_entry_mass=float(w@features['no_entry']),
                            unique_presence_paths=int(len(np.unique(selected['present'].T,axis=0)))))
            for mode in MODES:
                end=case['prefix_trials'] if mode=='prefix' else n
                means={k:np.mean([part[mode+'_'+k] for part in parts],axis=0)
                       for k in ('present','high_belief','target_belief','target_execution','source')}
                pooled[method,particles,mode]=means
                sd=np.std([part[mode+'_present'] for part in parts],axis=0,ddof=1)
                ownbase=dict(**base,method=method,particles=particles,mode=mode)
                for window,a,b in [('entry_window',case['window_start'],case['window_stop']),('available_span',0,end)]:
                    known={k:v[a:b] for k,v in events.items()}
                    tables['metrics'].extend(dict(**ownbase,window=window,endpoint=end,**row)
                        for row in metric_rows({k:v[a:b] for k,v in means.items()},known))
                    for job,part in zip(jobs,parts):
                        one={k:part[mode+'_'+k][a:b] for k in means}
                        tables['seed_metrics'].extend(dict(**ownbase,window=window,endpoint=end,seed=job['seeds'][0],**row)
                            for row in metric_rows(one,known))
                for t in range(end):
                    ess=[part[mode+'_ancestral_ess'][t] for part in parts] if mode!='filtered' else []
                    counts=[part[mode+'_unique_ancestors'][t] for part in parts] if mode!='filtered' else []
                    tables['curves'].append(dict(**ownbase,trial=t+1,
                        observation_endpoint=min(n,t+protocol['policy']['fixed_lag']+1) if mode=='lag' else t+1 if mode=='filtered' else end,
                        in_entry_window=case['window_start']<=t<case['window_stop'],true_present=bool(events['present'][t]),
                        presence=float(means['present'][t]),target_execution=float(means['target_execution'][t]),
                        available_other=float(means['source'][t,1]),presence_seed_sd=float(sd[t]),
                        mean_ancestral_ess=float(np.mean(ess)) if ess else np.nan,
                        min_ancestral_ess=float(min(ess)) if ess else np.nan,
                        mean_unique_ancestors=float(np.mean(counts)) if counts else np.nan))
                for index,(a,b) in enumerate(runs(events['present'])):
                    if a>=end:continue
                    p=means['present'][a:min(b,end)];hit=np.flatnonzero(p>=.5)
                    tables['episodes'].append(dict(**ownbase,episode=index+1,entry_trial=a+1,true_length=b-a,
                        observed_length=len(p),right_censored=b>end or b==n,detected=bool(len(hit)),
                        threshold_delay=int(hit[0]) if len(hit) else None,
                        mean_probability=float(p.mean()),max_probability=float(p.max()),false_negative_fraction=float((p<.5).mean()),
                        definition='first probability >=0.5 within the true episode; not inferred entry time'))
        if ('marginal',1024) in groups:
            a,b=case['window_start'],case['window_stop']
            for mode in MODES:
                for key in ('present','high_belief','target_execution','source'):
                    difference=np.abs(pooled['marginal',512,mode][key][a:b]-pooled['marginal',1024,mode][key][a:b])
                    tables['resolution_drift'].append(dict(**base,mode=mode,field=key,
                        mean_absolute_change=float(difference.mean()),max_absolute_change=float(difference.max())))
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    source_paths=[str(Path(__file__).resolve().relative_to(study.ROOT)),study.HELPER,study.ANCESTRY,
        'src/Bayesian_state/workflows/analysis/model_1024_smoothing_summary.py',
        'src/Bayesian_state/workflows/analysis/model_1023_execution_recovery_score.py']
    write(output/'completion.json',dict(status='complete',smoke=False,cases=len(protocol['cases']),filters=len(protocol['jobs']),
        run_completion_sha256=digest(run/'completion.json'),analysis_sources={p:digest(study.ROOT/p) for p in source_paths},
        artifacts_sha256={p.name:digest(p) for p in sorted(output.glob('*.csv'))},inputs_sha256=bindings,
        pooling='equal average of independent normalized filter estimates',
        no_entry_code='first_entry_trial=endpoint+1; no_entry identifies censoring',
        execution_counts='mean and total variance include within-path conditional execution uncertainty',
        parameter_refitting=False,oral_used=False))
    print('Sampled and marginalized execution recovery comparison complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();summarize(args.run,args.output)
