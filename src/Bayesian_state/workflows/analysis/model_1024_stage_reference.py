"""Exact references for the registered state-validation stage.

Use --prepared REGISTERED --output NEW [--smoke]. Finite-state runs use
independent fresh PF proposals. Shared-model conditional execution reuses
saved generator states and emissions, with that additional conditioning explicit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import model_1024_state_validation_stage as stage
from . import model_1024_smoothing as old
from .model_1024_smoothing_summary import truth_events
from ...evaluation import independent_particle_systems as bank
from ...evaluation import particle_smoothing as helper
from ...inference.backends import particle_filter as backend

read,write,digest=stage.read,stage.write,stage.digest


def diagnostic_pf(initial,transitions,likelihood,particles,seed):
    """Finite-state bootstrap PF with its actual evidence estimate."""
    rng=np.random.default_rng(seed);n,states=likelihood.shape
    x=rng.choice(states,size=particles,p=initial);w=np.full(particles,1/particles)
    parents=np.tile(np.arange(particles),(n,1));history=[];logz=0.
    for t in range(n):
        if t:x=np.sum(rng.random(particles)[:,None]>transitions[t-1,x].cumsum(axis=1),axis=1)
        history.append(x.copy());mass=w*likelihood[t,x];scale=mass.sum()
        if scale<=0:raise ValueError('Impossible finite-state observation')
        logz+=np.log(scale);w=mass/scale
        if t==n-1:terminal=w.copy()
        if 1/np.square(w).sum()<.5*particles:
            a=backend.systematic_resample(w,float(rng.random()));parents[t]=a;x=x[a];w.fill(1/particles)
    paths=helper.trace_values(np.array(history),backend._trace_ancestral_indices(parents))
    smooth=np.stack([np.bincount(row,weights=terminal,minlength=states) for row in paths])
    helper.probabilities(smooth)
    return smooth,float(logz)


def reference_bank(job):
    n,particles,index,p,initial,transitions,likelihood,exact,smoke=job
    count=8 if smoke else p['systems_per_bank'];curves=[];logz=[];began=time.process_time()
    for j in range(count):
        s=stage.seed(p['seed_base'],f'T{n}P{particles}B{index}',j)
        curve,z=diagnostic_pf(initial,transitions,likelihood,particles,s);curves.append(curve);logz.append(z)
    curves=np.array(curves);logz=np.array(logz);weights=bank.evidence_weights(logz)
    rng=np.random.default_rng(stage.seed(p['seed_base'],'pimh'+f'{n}_{particles}',index))
    chain=bank.pimh_indices(logz,rng.uniform(np.nextafter(0.,1.),1.,count-1))
    burn=2 if smoke else p['pimh_burn_in'];indices=chain[burn:]
    means=dict(equal=curves.mean(axis=0),evidence=np.einsum('j,jtk->tk',weights,curves),pimh=curves[indices].mean(axis=0))
    diagnostic=bank.bank_diagnostics(weights)
    rows=[dict(length=n,particles=particles,bank=index,method=method,systems=count,
        mae=float(np.abs(mean-exact).mean()),maximum=float(np.abs(mean-exact).max()),
        outer_ess=float(diagnostic['ess']),max_weight=float(diagnostic['maximum']),
        chain_acceptance=float(np.mean(chain[1:]!=chain[:-1])),cpu_seconds=time.process_time()-began)
        for method,mean in means.items()]
    return rows,dict(**means,log_evidence=logz,weights=weights,chain=chain)


def shared_references(r,output):
    p=r['policy'];oracle=read(stage.ROOT/p['source_oracles']/'protocol.json');rows=[];bindings={}
    for cid in p['benchmark']['shared_conditional_cases']:
        job=next(j for j in oracle['jobs'] if j['kind']=='oracle' and j['recovery_case']==cid)
        z=stage.checked.checked_arrays(Path(job['target']))
        truth_file=Path(job['truth_states']).with_suffix('.npz')
        if digest(truth_file)!=job['truth_sha256']:raise ValueError('Conditional reference truth changed')
        with np.load(truth_file) as a:raw={k:a[k].copy() for k in a.files}
        n=len(z['choices']);ids=[];belief=[];ell=[];local=[];prior_error=[]
        kappa=job['candidate']['hyperparams'][stage.worker_tools.study.KAPPA]
        for t in range(n):
            mask=z['rule'][t]>=0;ids.append(z['rule'][t,mask]);belief.append(raw['belief'][t,mask])
            ell.append(z['rule_choice_probabilities'][t,mask,int(z['choices'][t])-1])
            local.append(z['execution_posterior'][t,mask])
            if t:
                kernel=bank.execution_kernel(np.array([z['true_executed'][t-1]]),ids[t],belief[t],kappa)[0]
                prior_error.append(np.max(np.abs(kernel-z['execution_prior'][t,mask])))
        result=bank.conditional_execution(ids,belief,ell,kappa)
        if max(prior_error,default=0)>1e-10:raise ValueError('Conditional kernel differs from shared generator')
        target=np.array([np.isin(rule,job['target_rules']) for rule in ids],dtype=object)
        executed={mode:np.array([float(prob[np.asarray(hit,bool)].sum()) for prob,hit in zip(result[mode],target)])
                  for mode in ('filtered','smoothed')}
        if kappa==0:
            for actual,expected in zip(result['smoothed'],local):np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
        events=truth_events(raw,job['target_rules'],n);wrong=events['wrong'];source=np.stack([
            1-events['present'],events['present']-executed['smoothed'],executed['smoothed']],axis=1)
        helper.probabilities(source)
        row=dict(case_id=cid,kappa=kappa,trials=n,kernel_max_error=float(max(prior_error,default=0)),
            full_vs_current_choice_mad=float(np.mean([np.abs(a-b).mean() for a,b in zip(result['smoothed'],local)])),
            execution_brier=float(np.square(executed['smoothed']-events['target_execution']).mean()),
            error_source_accuracy=float((source[wrong].argmax(axis=1)==events['source'][wrong]).mean()),
            conditioning='complete true other-state path plus all choices; known parameters')
        rows.append(row)
        with (output/(cid+'_conditional.npz')).open('xb') as stream:
            np.savez_compressed(stream,filtered_execution=executed['filtered'],smoothed_execution=executed['smoothed'],
                source=source,true_execution=events['target_execution'],true_source=events['source'],wrong=wrong)
        bindings[str(truth_file)]=job['truth_sha256'];bindings[job['target']+'.npz']=digest(Path(job['target']).with_suffix('.npz'))
    pd.DataFrame(rows).to_csv(output/'shared_conditional.csv',index=False)
    return bindings


def run(prepared: Path,output: Path,smoke: bool):
    if output.exists():raise FileExistsError('Use a new reference directory')
    r=stage.registered(prepared);p=r['policy']['benchmark'];output.mkdir(parents=True);started=time.monotonic()
    jobs=[];source=stage.ROOT/p['source'] if 'source' in p else stage.ROOT/r['policy']['benchmark']['source']
    for n in p['lengths']:
        with np.load(source/f'exact_T{n}.npz') as z:a={k:z[k].copy() for k in z.files}
        exact=helper.exact_hmm(a['initial'],a['transitions'],a['likelihood'])
        np.testing.assert_allclose(exact['smoothed'],a['smoothed'],atol=1e-10,rtol=0)
        if n==8:
            brute=old.brute_hmm(a['initial'],a['transitions'],a['likelihood'])
            np.testing.assert_allclose(exact['smoothed'],brute['smoothed'],atol=1e-10,rtol=0)
        for particles in p['particles']:
            for index in range(1 if smoke else p['banks']):jobs.append((n,particles,index,p,a['initial'],a['transitions'],a['likelihood'],a['smoothed'],smoke))
    # Four independent banks are the ready diagnostic jobs; shared-model jobs
    # have their own single process-parallel layer and one numeric thread each.
    write(output/'protocol.json',dict(prepared=str(prepared.resolve()),prepared_sha256=digest(prepared/'protocol.json'),
        policy=p,smoke=smoke,jobs=[dict(length=j[0],particles=j[1],bank=j[2]) for j in jobs],oral_used=False))
    rows=[]
    with ProcessPoolExecutor(max_workers=min(4,len(jobs))) as pool:
        for job,(metrics,arrays) in zip(jobs,pool.map(reference_bank,jobs)):
            rows.extend(metrics)
            with (output/f'T{job[0]}_P{job[1]}_B{job[2]}.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    bindings=shared_references(r,output)
    pd.DataFrame(rows).to_csv(output/'finite_metrics.csv',index=False)
    write(output/'completion.json',dict(status='complete',smoke=smoke,seconds=time.monotonic()-started,inputs=bindings,
        sources={str(Path(__file__).relative_to(stage.ROOT)):digest(Path(__file__)),str(Path(bank.__file__).relative_to(stage.ROOT)):digest(Path(bank.__file__))},
        files={f.name:digest(f) for f in output.iterdir() if f.is_file()},oral_used=False))
    print('Finite and shared-model conditional references complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    run(a.prepared,a.output,a.smoke)
