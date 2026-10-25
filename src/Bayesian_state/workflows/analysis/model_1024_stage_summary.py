"""Score registered state validation without selecting successful trajectories.

Use --phase history|panel|sensitivity --prepared REGISTERED --run COMPLETE
--output NEW. History writes its first diagnostic beside the completed run;
that frozen diagnostic determines the single permitted follow-up batch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_1024_state_validation_stage as stage
from . import model_1024_smoothing_summary as scoring
from ...evaluation import independent_particle_systems as bank
from ...evaluation import trajectory_calibration as calibration
from ...evaluation import particle_smoothing as helper
from ...inference.backends.particle_filter import _weighted_quantiles

read,write,digest=stage.read,stage.write,stage.digest
KEYS=('present','high_belief','target_belief','target_execution','source')
MODES=('filtered','lag','prefix','full')


def complete(run: Path) -> dict:
    p=read(run/'protocol.json');d=read(run/'completion.json')
    if d['status']!='complete' or d.get('smoke') or d['protocol_sha256']!=digest(run/'protocol.json'):
        raise ValueError('Require complete unchanged formal run')
    if p['worker_sources']!=stage.worker_tools.sources():raise ValueError('Scientific worker sources changed')
    return p


def capture(job: dict) -> tuple[dict,np.ndarray,dict]:
    stem=Path(job.get('marginal_target',job.get('genealogy_target')))
    part=stage.checked.checked_arrays(stem,'array_sha256');meta=read(stem.with_suffix('.json'))
    helper.probabilities(part['filtered_weights'])
    for mode in MODES:
        source=helper.probabilities(part[mode+'_source']);present=part[mode+'_present'];execution=part[mode+'_target_execution']
        for key in ('present','high_belief','target_belief','target_execution'):
            value=part[mode+'_'+key]
            if not np.isfinite(value).all() or np.any(value < -1e-10) or np.any(value>1+1e-10):raise ValueError('Invalid state probability')
        np.testing.assert_allclose(source[:,0],1-present,atol=1e-10,rtol=0)
        np.testing.assert_allclose(source[:,1],present-execution,atol=1e-10,rtol=0)
        np.testing.assert_allclose(source[:,2],execution,atol=1e-10,rtol=0)
        if np.any(part[mode+'_high_belief']>present+1e-10):raise ValueError('High belief outside workspace')
    arrays=stage.checked.checked_arrays(Path(job['target']));score=read(Path(job['target']).with_suffix('.json'))
    probability=arrays['observed_probability'][0]
    if probability.shape!=(meta['trials'],) or np.any(probability<1e-12):raise ValueError('Evidence floor active or incompatible length')
    if any(pool.get('num_threads',1)!=1 for pool in score['numeric_pools']):raise ValueError('Numeric thread budget exceeded')
    if meta.get('minimum_particle_choice_mass',1.)<1e-12:raise ValueError('Particle likelihood floor active')
    audit=read(Path(job['target']+'.mechanism.json'))['audit'];expected=len(probability)*job['particles']
    executed=0 if job['execution_mode']=='MIX' else expected
    if (audit['transitions']!=expected or audit['feedback_updates']!=expected
            or audit['error_entry']['execution']['trials']!=executed):raise ValueError('Shared update count mismatch')
    return part,np.cumsum(np.log(probability)),dict(name=job['name'],particles=job['particles'],
        trials=len(probability),updates=expected,seconds=score['seconds'],cpu_seconds=score['cpu_seconds'],
        array_sha256=meta['array_sha256'],capture=str(stem),resampling_steps=meta['resampling_steps'])


def raw_truth(case: dict) -> dict:
    item=case['truth'];stem=Path(item['states'] if isinstance(item,dict) else item)
    sha=item['states_sha256'] if isinstance(item,dict) else case['truth_sha256']
    if digest(stem.with_suffix('.npz'))!=sha:raise ValueError('Simulation truth changed')
    with np.load(stem.with_suffix('.npz')) as z:return {k:z[k].copy() for k in z.files}


def endpoints(mode: str,n: int,prefix: int,lag: int) -> np.ndarray:
    if mode=='filtered':return np.arange(1,n+1)
    if mode=='lag':return np.minimum(n,np.arange(1,n+1)+lag)
    if mode=='prefix':return np.full(prefix,prefix)
    if mode=='full':return np.full(n,n)
    raise ValueError(mode)


def pooled(parts: list[dict],logz: np.ndarray,mode: str,prefix: int,lag: int,weighted: bool) -> dict:
    n=logz.shape[1];ends=endpoints(mode,n,prefix,lag)
    result={key:(bank.pool_curves(np.array([a[mode+'_'+key] for a in parts]),logz,ends) if weighted
                 else np.mean([a[mode+'_'+key] for a in parts],axis=0)) for key in KEYS}
    helper.probabilities(result['source']);return result


def path_features(part: dict,mode: str) -> tuple[dict,np.ndarray]:
    indices=part[mode+'_indices'];n=indices.shape[1]
    values={key:helper.trace_values(part['particle_'+key][:n],indices)
            for key in ('present','target_execution','target_belief')}
    return scoring.coherent_features(values['present'],values['target_execution'],values['target_belief']),part['filtered_weights'][n-1]


def functional_rows(parts: list[dict],logz: np.ndarray,events: dict,mode: str,weighted: bool) -> list[dict]:
    end=parts[0][mode+'_indices'].shape[1];outer=bank.evidence_weights(logz[:,end-1]) if weighted else np.full(len(parts),1/len(parts))
    features=[];weights=[]
    for p,a in zip(parts,outer):
        f,w=path_features(p,mode);features.append(f);weights.append(a*w)
    w=np.concatenate(weights);helper.probabilities(w)
    actual=scoring.coherent_features(events['present'][:end,None],events['target_execution'][:end,None],events['target_belief'][:end,None])
    rows=[]
    for name in features[0]:
        values=np.concatenate([f[name] for f in features]);q=_weighted_quantiles(values,w,[.05,.5,.95]);truth=float(actual[name][0])
        # A fractional expected execution count retains its mean. Its quantiles
        # do not represent discrete execution-count uncertainty after integration.
        integrated=name=='target_execution_trials' and any(np.any(p['particle_target_execution']%1) for p in parts)
        within=0.
        if integrated:
            for p,a in zip(parts,outer):
                indices=p[mode+'_indices'];qexec=helper.trace_values(p['particle_target_execution'][:end],indices)
                within+=a*float(p['filtered_weights'][end-1]@np.sum(qexec*(1-qexec),axis=0))
        mean=float(w@values);variance=float(w@np.square(values-mean)+within)
        rows.append(dict(function=name,endpoint=end,true_value=truth,mean=mean,variance=variance,within_execution_variance=within,
            q05=np.nan if integrated else float(q[0]),q50=np.nan if integrated else float(q[1]),q95=np.nan if integrated else float(q[2]),
            truth_in_90_interval=np.nan if integrated else bool(q[0]<=truth<=q[2]),
            interval_width=np.nan if integrated else float(q[2]-q[0]),
            no_entry_mass=float(w@np.concatenate([f['no_entry'] for f in features])),
            conditioning='common complete observation endpoint; known parameters',integrated_execution=integrated))
    return rows


def history(prepared: Path,run: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new history summary')
    r=stage.registered(prepared);p=complete(run);jobs=list(p['jobs'])
    cases=p['cases']
    if p['previous']:
        previous=complete(Path(p['previous']));jobs=previous['jobs']+jobs;cases=previous['cases']
    output.mkdir(parents=True);metrics=[];curves=[];functionals=[];numeric=[];diagnostics=[]
    for case in cases:
        cid=case['case_id'];own=[j for j in jobs if j['recovery_case']==cid];captured=[capture(j) for j in own]
        parts=[a[0] for a in captured];logz=np.array([a[1] for a in captured]);numeric.extend(dict(case_id=cid,**a[2]) for a in captured)
        events=scoring.truth_events(raw_truth(case),case['target_rules'],logz.shape[1]);prefix=case['prefix_trials'];n=logz.shape[1]
        a,b=case['window_start'],case['window_stop'];weights=bank.evidence_weights(logz[:,-1]);d=bank.bank_diagnostics(weights)
        middle=len(parts)//2
        halves=[pooled(parts[s],logz[s],'full',prefix,32,True)['present'] for s in (slice(0,middle),slice(middle,None))]
        half_mad=float(np.abs(halves[0][a:b]-halves[1][a:b]).mean())
        half_q95=float(np.quantile(np.abs(halves[0][a:b]-halves[1][a:b]),.95))
        whole_delta=np.abs(halves[0]-halves[1])
        half_features=[{v['function']:v for v in functional_rows(parts[s],logz[s],events,'full',True)}
            for s in (slice(0,middle),slice(middle,None))]
        follow=(float(d['ess'])<8 or float(d['maximum'])>.25 or half_mad>.05)
        diagnostics.append(dict(case_id=cid,systems=len(parts),outer_ess=float(d['ess']),maximum_weight=float(d['maximum']),
            half_bank_window_mad=half_mad,followup_required=bool(follow and not p['previous']),
            half_bank_window_q95=half_q95,half_bank_whole_mad=float(whole_delta.mean()),half_bank_whole_q95=float(np.quantile(whole_delta,.95)),
            half_presence_fraction_difference=abs(half_features[0]['presence_trials']['mean']-half_features[1]['presence_trials']['mean'])/n,
            half_first_entry_median_difference=abs(half_features[0]['first_entry_trial']['q50']-half_features[1]['first_entry_trial']['q50']),
            half_no_entry_mass_difference=abs(half_features[0]['first_entry_trial']['no_entry_mass']-half_features[1]['first_entry_trial']['no_entry_mass']),
            numerical_gate_pass=bool(float(d['ess'])>=8 and float(d['maximum'])<=.25 and half_mad<=.025 and half_q95<=.10
                and whole_delta.mean()<=.025 and np.quantile(whole_delta,.95)<=.10),
            single_followup_used=bool(len(parts)>p['policy']['long_history']['initial_systems']),
            log_evidence_sd=float(logz[:,-1].std(ddof=1))))
        for mode in MODES:
            end=prefix if mode=='prefix' else n;known={k:v[:end] for k,v in events.items()}
            for method,weighted in [('equal',False),('evidence',True)]:
                mean=pooled(parts,logz,mode,prefix,32,weighted)
                for scope,start,stop in [('whole',0,end),('entry_window',a,b)]:
                    for row in scoring.metric_rows({k:v[start:stop] for k,v in mean.items()},{k:v[start:stop] for k,v in known.items()}):
                        metrics.append(dict(case_id=cid,mode=mode,method=method,scope=scope,systems=len(parts),**row))
                for t in range(end):
                    curves.append(dict(case_id=cid,mode=mode,method=method,trial=t+1,present=mean['present'][t],
                        high_belief=mean['high_belief'][t],execution=mean['target_execution'][t],source1=mean['source'][t,1],
                        true_present=bool(events['present'][t]),true_execution=bool(events['target_execution'][t]),
                        in_entry_window=a<=t<b))
                if mode in ('prefix','full'):
                    functionals.extend(dict(case_id=cid,mode=mode,method=method,systems=len(parts),**row)
                        for row in functional_rows(parts,logz,events,mode,weighted))
        with (output/(cid+'_outer.npz')).open('xb') as stream:np.savez_compressed(stream,log_evidence=logz,full_weights=weights)
    for name,rows in [('metrics',metrics),('curves',curves),('functionals',functionals),('numerical_audit',numeric)]:pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    write(output/'bank_diagnostics.json',dict(status='complete',cases=diagnostics,run=str(run.resolve()),run_sha256=digest(run/'protocol.json')))
    decision=run/'bank_diagnostics.json'
    if decision.exists():raise FileExistsError('Existing follow-up decision must be retained')
    write(decision,read(output/'bank_diagnostics.json'))
    finish(output,dict(run=str(run.resolve()),prepared=str(prepared.resolve()),systems=len(jobs),oral_used=False))


def reference(context: dict,raw: dict) -> tuple[dict,dict]:
    chunks=[stage.checked.checked_arrays(Path(j['target'])) for j in context['reference_jobs']]
    for chunk in chunks:
        np.testing.assert_array_equal(chunk['categories'],raw['categories'])
        helper.probabilities(chunk['probabilities'])
    keys=('rule','belief','executed','probabilities');bankraw={k:np.concatenate([c[k] for c in chunks]) for k in keys}
    target=np.isin(bankraw['rule'],context['target_rules']);present=target.any(axis=2)
    belief=(bankraw['belief']*target).sum(axis=2);execution=np.isin(bankraw['executed'],context['target_rules'])
    source=np.stack([~present,present&~execution,execution],axis=2)
    values=dict(present=present,high_belief=present&(belief>=.5),target_belief=belief,target_execution=execution,source=source)
    n=len(raw['choices']);likelihood=bankraw['probabilities'][:,np.arange(n),raw['choices']-1]
    mass=likelihood/likelihood.sum(axis=0,keepdims=True);helper.probabilities(mass,axis=0)
    conditional={k:np.einsum('pt,pt...->t...',mass,v) for k,v in values.items()}
    unconditional={k:v.mean(axis=0) for k,v in values.items()}
    halves=[]
    for s in (slice(0,64),slice(64,None)):
        w=likelihood[s];w=w/w.sum(axis=0,keepdims=True)
        halves.append({key:np.einsum('pt,pt...->t...',w,v[s]) for key,v in values.items()})
    half_difference=np.abs(halves[0]['present']-halves[1]['present'])
    diag=dict(minimum_ess=float(np.min(1/np.square(mass).sum(axis=0))),mean_ess=float(np.mean(1/np.square(mass).sum(axis=0))),
        half_presence_mad=float(half_difference.mean()),half_presence_q95=float(np.quantile(half_difference,.95)),
        half_execution_mad=float(np.abs(halves[0]['target_execution']-halves[1]['target_execution']).mean()),
        half_high_belief_mad=float(np.abs(halves[0]['high_belief']-halves[1]['high_belief']).mean()),
        half_source1_mad=float(np.abs(halves[0]['source'][:,1]-halves[1]['source'][:,1]).mean()))
    return dict(current_choice=conditional,unconditional=unconditional),diag


def finish(output: Path,extra: dict) -> None:
    write(output/'completion.json',dict(status='complete',**extra,
        source_sha256=digest(Path(__file__)),files={f.name:digest(f) for f in output.iterdir() if f.is_file()}))


def panel(prepared: Path,run: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new panel summary')
    r=stage.registered(prepared);p=complete(run);policy=r['policy'];output.mkdir(parents=True)
    metrics=[];numeric=[];sensitivity=[];functional=[];reference_diagnostics=[];records=[]
    for c in p['cases']:
        raw=raw_truth(c);events=scoring.truth_events(raw,c['targets'],c['trials']);context=next(a for a in r['contexts'] if a['context_id']==c['context_id'])
        baselines,diag=reference(context,raw);reference_diagnostics.append(dict(case_id=c['case_id'],context_id=c['context_id'],**diag))
        base=dict(case_id=c['case_id'],context_id=c['context_id'],subject=c['subject'],condition=c['condition'])
        levels=sorted(set(j['particles'] for j in p['jobs'] if j['recovery_case']==c['case_id']));means={}
        for level in levels:
            own=[j for j in p['jobs'] if j['recovery_case']==c['case_id'] and j['particles']==level]
            cap=[capture(j) for j in own];parts=[a[0] for a in cap];z=np.array([a[1] for a in cap]);numeric.extend(dict(**base,**a[2]) for a in cap)
            for mode in MODES:
                mean=pooled(parts,z,mode,own[0]['prefix_trials'],policy['fixed_lag'],False);end=len(mean['present'])
                means[level,mode]=mean;truth={k:v[:end] for k,v in events.items()}
                for row in scoring.metric_rows(mean,truth):metrics.append(dict(**base,particles=level,mode=mode,method='two_filter_equal_mean',**row))
                for key in ('present','high_belief','target_execution','source','source_0','source_1','source_2'):
                    field='source' if key.startswith('source_') else key
                    left,right=parts[0][mode+'_'+field],parts[1][mode+'_'+field]
                    if key.startswith('source_'):left,right=left[:,int(key[-1])],right[:,int(key[-1])]
                    delta=np.abs(left-right)
                    sensitivity.append(dict(**base,particles=level,mode=mode,comparison='independent_filter_seeds',state=key,
                        mad=float(delta.mean()),q95=float(np.quantile(delta,.95))))
                if mode in ('prefix','full'):
                    functional.extend(dict(**base,particles=level,mode=mode,**row) for row in functional_rows(parts,z,events,mode,False))
                if level==512:records.append(dict(**base,mode=mode,prediction=mean,truth=truth,
                    baselines={label:{key:v[:end] for key,v in value.items()} for label,value in baselines.items()}))
        if 1024 in levels:
            for mode in MODES:
                for key in ('present','high_belief','target_execution','source','source_0','source_1','source_2'):
                    field='source' if key.startswith('source_') else key
                    left,right=means[512,mode][field],means[1024,mode][field]
                    if key.startswith('source_'):left,right=left[:,int(key[-1])],right[:,int(key[-1])]
                    delta=np.abs(left-right);sensitivity.append(dict(**base,particles=1024,mode=mode,
                        comparison='512_vs_1024_two_seed_means',state=key,mad=float(delta.mean()),q95=float(np.quantile(delta,.95))))
        for label,value in baselines.items():
            for row in scoring.metric_rows(value,events):metrics.append(dict(**base,particles=0,mode='reference',method=label,**row))
    tables=dict(metrics=metrics,numerical_audit=numeric,numerical_stability=sensitivity,functionals=functional,reference_diagnostics=reference_diagnostics)
    tables.update(calibration_tables(records,policy))
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    finish(output,dict(run=str(run.resolve()),run_protocol_sha256=digest(run/'protocol.json'),prepared=str(prepared.resolve()),
        cases=len(p['cases']),filters=len(p['jobs']),pooling='two independent normalized PF estimates averaged equally',
        bootstrap='whole independent generating paths within each parameter context',oral_used=False))


def calibration_tables(records: list[dict],policy: dict) -> dict:
    bins=[];proper=[];labels=[];coverage=[];a=policy['acceptance']
    for mode in MODES:
        all_rows=[r for r in records if r['mode']==mode]
        scopes=['all']+['condition_'+str(c) for c in sorted(set(r['condition'] for r in all_rows))]
        scopes+=['context_'+c for c in sorted(set(r['context_id'] for r in all_rows))]
        for scope in scopes:
            rows=(all_rows if scope=='all' else [r for r in all_rows if r['context_id']==scope.removeprefix('context_')]
                  if scope.startswith('context_') else [r for r in all_rows if r['condition']==int(scope.split('_')[-1])])
            counts=calibration.bootstrap_counts([r['context_id'] for r in rows],policy['bootstrap_repeats'],policy['bootstrap_seed'])
            error_index={'error_absent':0,'error_available_other':1,'error_executed':2}
            for key in ('present','high_belief','target_execution',*error_index):
                prediction=[];truth=[];reference=[]
                for row in rows:
                    mask=row['truth']['wrong'] if key in error_index else np.ones(len(row['truth']['present']),bool)
                    if key in error_index:
                        index=error_index[key]
                        prediction.append(row['prediction']['source'][mask,index]);truth.append(row['truth']['source'][mask]==index)
                        reference.append(row['baselines']['current_choice']['source'][mask,index])
                    else:
                        prediction.append(row['prediction'][key][mask]);truth.append(row['truth'][key][mask]);reference.append(row['baselines']['current_choice'][key][mask])
                base=dict(mode=mode,scope=scope,state=key,independent_paths=len(rows))
                c=calibration.calibration(prediction,truth,counts,policy['probability_bins'],a['minimum_bin_paths'],a['minimum_bin_trials'],a['calibration_practical_error'])
                bins.extend(dict(**base,**item) for item in c)
                den=np.array([len(y) for y in truth]);positive_paths=sum(bool(y.any()) for y in truth);negative_paths=sum(bool((~y).any()) for y in truth)
                coverage.append(dict(**base,positive_paths=positive_paths,negative_paths=negative_paths,positive_events=sum(y.sum() for y in truth),
                    trials=int(den.sum()),state_coverage=positive_paths>=a['minimum_positive_paths'] and negative_paths>=a['minimum_negative_paths'],
                    rare_event_coverage=positive_paths>=a['minimum_rare_event_paths'] and sum(y.sum() for y in truth)>=a['minimum_rare_events']))
                for score in ('brier','log_loss'):
                    sums=[];diffs=[]
                    for prob,ref,y in zip(prediction,reference,truth):
                        def loss(v):
                            if score=='brier':return np.square(v-y)
                            v=np.clip(v,1e-12,1-1e-12)
                            return -(y*np.log(v)+(1-y)*np.log1p(-v))
                        sums.append(loss(prob).sum());diffs.append((loss(prob)-loss(ref)).sum())
                    proper.append(dict(**base,score=score,**calibration.ratio_interval(sums,den,counts),
                        **{'reference_difference_'+k:v for k,v in calibration.ratio_interval(diffs,den,counts).items()}))
                for label,selectors in [('clear_all',[(v>=.8)|(v<=.2) for v in prediction]),
                                        ('clear_positive',[v>=.8 for v in prediction]),('clear_negative',[v<=.2 for v in prediction])]:
                    numerators=[];denominators=[];recalled=[];total=[]
                    for v,y,s in zip(prediction,truth,selectors):
                        numerators.append(int(np.sum(((v>=.5)==y)&s)));denominators.append(int(s.sum()))
                        target=~y if label=='clear_negative' else y
                        selected_event=v<=.2 if label=='clear_negative' else v>=.8
                        recalled.append(int(np.sum(selected_event&target)));total.append(int(target.sum()))
                    precision=calibration.ratio_interval(numerators,denominators,counts);report_fraction=calibration.ratio_interval(denominators,den,counts)
                    recall=calibration.ratio_interval(recalled,total,counts)
                    labels.append(dict(**base,label=label,**{'precision_'+k:v for k,v in precision.items()},
                        **{'coverage_'+k:v for k,v in report_fraction.items()},**{'recall_'+k:v for k,v in recall.items()}))
            # Three-way execution-error labels use one fixed high-confidence rule.
            ps=[r['prediction']['source'][r['truth']['wrong']] for r in rows];ys=[r['truth']['source'][r['truth']['wrong']] for r in rows]
            chosen=[p.max(axis=1)>=.8 for p in ps];den=np.array([len(y) for y in ys]);sel=[int(m.sum()) for m in chosen]
            correct=[int(np.sum((p.argmax(axis=1)==y)&m)) for p,y,m in zip(ps,ys,chosen)]
            labels.append(dict(mode=mode,scope=scope,state='error_source',independent_paths=len(rows),label='clear_three_class',
                **{'precision_'+k:v for k,v in calibration.ratio_interval(correct,sel,counts).items()},
                **{'coverage_'+k:v for k,v in calibration.ratio_interval(sel,den,counts).items()}))
    return dict(calibration=bins,proper_scores=proper,clear_labels=labels,event_coverage=coverage)


def sensitivity(prepared: Path,output: Path):
    if output.exists():raise FileExistsError('Use a new sensitivity summary')
    r=stage.registered(prepared);source=stage.ROOT/r['policy']['source_inference'];p=read(source/'protocol.json');output.mkdir(parents=True);rows=[]
    for case in p['cases']:
        if case['definition']['scope']!='fixed':continue
        wrong=raw_truth(dict(truth=case['truth']))
        wrong=wrong['choices']!=wrong['categories']
        role_states={}
        for role,candidate in case['roles'].items():
            if candidate['id'] not in case['states']:
                rows.append(dict(case_id=case['definition']['case_id'],subject=case['definition']['subject'],role=role,
                    state='all',status='no_saved_state_capture',conditioning='fixed point not captured in the original study'))
                continue
            parts=[stage.checked.checked_arrays(Path(s),'array_sha256') for s in case['states'][candidate['id']]]
            role_states[role]={k:np.mean([a['filtered_'+k] for a in parts],axis=0) for k in ('target_present','target_high_belief','target_belief')}
            role_states[role]['target_execution']=np.mean([a['conditional_decision_sources'][:,2] for a in parts],axis=0)
            role_states[role]['available_other_wrong']=np.mean([a['conditional_decision_sources'][wrong,1] for a in parts],axis=0)
        for role in ('adopted','nll_reference'):
            if role not in role_states:continue
            for key in role_states['true']:
                d=np.abs(role_states[role][key]-role_states['true'][key]);rows.append(dict(case_id=case['definition']['case_id'],
                    subject=case['definition']['subject'],role=role,state=key,mad=float(d.mean()),q95=float(np.quantile(d,.95)),maximum=float(d.max()),
                    same_candidate=case['roles'][role]['id']==case['roles']['true']['id'],
                    status='available',conditioning='fixed candidate point comparison; no parameter posterior integration'))
    pd.DataFrame(rows).to_csv(output/'point_sensitivity.csv',index=False)
    finish(output,dict(prepared=str(prepared.resolve()),source_protocol=str(source/'protocol.json'),source_protocol_sha256=digest(source/'protocol.json'),oral_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('history','panel','sensitivity'),required=True)
    parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--run',type=Path);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.phase=='sensitivity':sensitivity(a.prepared,a.output)
    elif a.phase=='history' and a.run:history(a.prepared,a.run,a.output)
    elif a.phase=='panel' and a.run:panel(a.prepared,a.run,a.output)
    else:parser.error('This phase requires --run')
