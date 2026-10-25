"""Audit all paired continuation branches and summarize fixed oracle contrasts.

Use --source COMPLETE --output NEW. Scores retain every path. Numerical errors
cluster descendants by prefix filter; task-start references retain all 128
independent paths. Both ordinary U and cross-filter Energy are reported.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .model_1024_continuation_diagnosis import read, write, digest, verify, sources
from .model_1024_trajectory_diagnosis import complete_followup
from .model_1023_execution_summary import trajectory_diagnostics
from .model_1022_belief_summary import jackknife_se
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...metrics.task import category_learning_feedback

CONTRASTS = [('RETAIN','BASE'),('DISCOVER','BASE'),('AVAILABLE','DISCOVER'),('AVAILABLE','BASE'),
    ('PRECISION','BASE'),('AVAILABLE_PRECISION','AVAILABLE'),('AVAILABLE_EXECUTE','AVAILABLE'),
    ('AVAILABLE_EXECUTE_PRECISION','AVAILABLE_EXECUTE'),('AVAILABLE_EXECUTE_PRECISION','AVAILABLE_PRECISION')]


def clustered_se(values: np.ndarray) -> float:
    """First axis contains independent filters; all descendants stay together."""
    x=np.asarray(values,float)
    means=x.reshape(len(x),-1).mean(1)
    return float(means.std(ddof=1)/np.sqrt(len(means)))


def cross_filter_energy(samples: np.ndarray, truth: np.ndarray, labels: np.ndarray) -> float:
    """Exclude every within-filter pair from the ensemble-distance term."""
    x,y=np.asarray(samples,float),np.asarray(truth,float)
    labels=np.asarray(labels)
    if labels.shape!=(len(x),) or len(np.unique(labels))<2:
        raise ValueError('Cross-filter Energy requires two independent filters')
    mask=labels[:,None]!=labels[None,:]
    return float(np.linalg.norm(x-y,axis=1).mean()-.5*cdist(x,x)[mask].mean())


def energies(samples: np.ndarray, truth: np.ndarray, labels: np.ndarray | None):
    value=energy_u(samples,truth)['value']
    if labels is None:
        deleted=np.array([energy_u(np.delete(samples,i,axis=0),truth)['value'] for i in range(len(samples))])
        return value,jackknife_se(deleted),deleted,value
    units=np.unique(labels)
    deleted=np.array([energy_u(samples[labels!=u],truth)['value'] for u in units])
    return value,jackknife_se(deleted),deleted,cross_filter_energy(samples,truth,labels)


def load_case(case: dict) -> tuple[dict, dict]:
    chunks=[]; hashes={}
    for job in case['jobs']:
        stem=Path(job['target']); record=read(stem.with_suffix('.json'))
        for suffix,key in (('.npz','predictions_sha256'),('.pkl.gz','snapshot_sha256')):
            p=stem.with_suffix(suffix)
            if digest(p)!=record[key]: raise ValueError('Changed continuation artifact')
            hashes[str(p)]=record[key]
        hashes[str(stem.with_suffix('.json'))]=digest(stem.with_suffix('.json'))
        if (not record['prefix_exact'] or record['case_id']!=case['case_id'] or record['smoke']
                or record['candidate']!=case['candidate'] or record['filter_seed']!=job['seeds'][0]):
            raise ValueError('Continuation identity or timing changed')
        if any(p.get('num_threads',1)!=1 for p in record['numeric_pools']):
            raise ValueError('Numerical worker used multiple threads')
        with np.load(stem.with_suffix('.npz')) as z: chunks.append({k:z[k] for k in z.files})
    constant=('categories','arms')
    result={k:chunks[0][k] if k in constant else np.stack([c[k] for c in chunks]) for k in chunks[0]}
    for c in chunks[1:]:
        for k in constant: np.testing.assert_array_equal(c[k],result[k])
    return result,hashes


def task_start(case: dict, hashes: dict) -> dict:
    chunks=[]
    for j in case['autonomous_jobs']:
        stem=Path(j['target']); record=read(stem.with_suffix('.json'))
        if digest(stem.with_suffix('.npz'))!=record['predictions_sha256'] or record['candidate']!=case['candidate']:
            raise ValueError('Task-start reference changed')
        hashes[str(stem.with_suffix('.npz'))]=record['predictions_sha256']
        hashes[str(stem.with_suffix('.json'))]=digest(stem.with_suffix('.json'))
        with np.load(stem.with_suffix('.npz')) as z:
            np.testing.assert_array_equal(z['seeds'],j['seeds'])
            chunks.append({k:z[k] for k in ('choices','categories','probabilities')})
    return {k:chunks[0][k] if k=='categories' else np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def actual(case: dict) -> tuple[np.ndarray,np.ndarray,int]:
    folder=Path(case['folder']); meta=read(folder/'prepared.json')
    verify({str(folder/k):v for k,v in meta['array_sha256'].items()})
    if digest(folder/'engine.json')!=meta['engine_sha256']:
        raise ValueError('Prepared engine changed')
    with np.load(folder/'training.npz') as a,np.load(folder/'sealed_test.npz') as b:
        choice=np.concatenate([a['choices'],b['choices']])[case['cut']:]
        category=np.concatenate([a['categories'],b['categories']])[case['cut']:]
    return choice,category,meta['condition']


def events(choices: np.ndarray, categories: np.ndarray, observed: np.ndarray, n_categories: int):
    jump=complete_followup(choices,categories,32)
    target=complete_followup(observed[None,:],categories,32)
    joint=((jump['complete_gain']>=target['complete_gain'][0]) &
           (jump['complete_maintenance']>=target['complete_maintenance'][0]))
    result=dict(jump_joint=joint,jump_gain=jump['complete_gain'],jump_maintenance=jump['complete_maintenance'])
    truth=dict(gain=float(target['complete_gain'][0]),maintenance=float(target['complete_maintenance'][0]))
    if len(categories)>=128:
        low=trajectory_diagnostics(choices,categories,n_categories=n_categories)
        real=trajectory_diagnostics(observed[None,:],categories,n_categories=n_categories)
        result['low_joint']=((low['minimum_window_accuracy']<=real['minimum_window_accuracy'][0]) &
            (low['minimum_window_accuracy_minus_marginal_chance']<=real['minimum_window_accuracy_minus_marginal_chance'][0]))
        result['minimum128']=low['minimum_window_accuracy']
        truth.update(minimum128=float(real['minimum_window_accuracy'][0]),
            marginal_gap=float(real['minimum_window_accuracy_minus_marginal_chance'][0]))
    return result,truth


def summarize(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new summary directory')
    protocol=read(source/'protocol.json'); done=read(source/'completion.json')
    if done['status']!='complete' or done['smoke'] or done['protocol_sha256']!=digest(source/'protocol.json'):
        raise ValueError('Require completed formal continuation run')
    if protocol['source_sha256']!=sources(): raise ValueError('Generation source changed')
    verify(protocol['inputs_sha256'])
    output.mkdir(parents=True)
    policy=protocol['policy']; width=policy['block_width']; late=policy['late_trials']
    rows=[]; paths=[]; curves=[]; endpoints=[]; contrasts=[]; truth_rows=[]; reference_rows=[]
    hashes={str((source/'protocol.json').resolve()):digest(source/'protocol.json'),
        str((source/'completion.json').resolve()):digest(source/'completion.json')}
    audit_trials=0; per_case={}
    for case in protocol['cases']:
        bank,bindings=load_case(case); hashes.update(bindings)
        observed,categories,condition=actual(case)
        np.testing.assert_array_equal(categories,bank['categories'])
        ncat=2 if condition==1 else 4
        arms=list(bank['arms']); nf,na,nb,nt=bank['choices'].shape
        if (nf,na,nb,nt)!=(policy['filters'],len(arms),policy['branches_per_filter'],len(categories)):
            raise ValueError('Wrong registered path budget')
        labels=np.repeat(np.arange(nf),nb)
        tag={k:case[k] for k in ('case_id','subject','mode','cut','cut_label','is_original','is_selected')}
        tag['candidate']=case['candidate']['id']; tag['condition']=condition
        obs_correct=observed==categories
        real_full=behavior_features(observed[None,:],categories,n_categories=ncat,block_width=width)[0]
        real_late=behavior_features(observed[None,-late:],categories[-late:],n_categories=ncat,block_width=width)[0]
        truth_rows.append(dict(tag,observed_late=obs_correct[-late:].mean(),observed_full=obs_correct.mean(),trials=nt))
        for f in range(nf):
            w=bank['terminal_weights'][f]; present=bank['endpoint_present'][f]
            if not np.isfinite(w).all() or np.any(w<0): raise ValueError('Invalid terminal weights')
            np.testing.assert_allclose(w.sum(),1.,atol=1e-12,rtol=0)
            mass=float(w@present)
            endpoints.append(dict(tag,filter=f,present=mass,effective_particles=float(1/np.square(w).sum()),
                precision_given_presence=float(w@bank['endpoint_beta'][f]/mass) if mass else np.nan,
                sampled_present=float(present[bank['ancestors'][f]].mean())))
        own={}
        for a,arm in enumerate(arms):
            choice=bank['choices'][:,a]; p=bank['probabilities'][:,a]; feedback=bank['feedback'][:,a]
            if not np.isfinite(p).all() or np.any(p<0): raise ValueError('Invalid predictive probability')
            np.testing.assert_allclose(p.sum(-1),1.,atol=1e-12,rtol=0)
            expected=np.asarray([category_learning_feedback(int(c),int(k),condition=condition)
                for c,k in zip(choice.ravel(),np.tile(categories,nf*nb))]).reshape(choice.shape)
            np.testing.assert_array_equal(feedback,expected)
            present=bank['diagnostic_present'][:,a].astype(bool)
            beta=bank['diagnostic_beta'][:,a]
            contribution=bank['diagnostic_contribution'][:,a]
            target_p=bank['diagnostic_target_probabilities'][:,a]
            rules=bank['rule'][:,a]; weights=bank['contribution'][:,a]
            ordered=np.sort(rules,axis=-1)
            if np.any((ordered[...,1:]==ordered[...,:-1]) & (ordered[...,1:]>=0)):
                raise ValueError('Duplicate active rule')
            np.testing.assert_array_equal(present,(rules==case['target_rule']).any(-1))
            if not np.isfinite(weights).all() or np.any(weights<0) or np.any(weights[rules<0]):
                raise ValueError('Invalid active-rule weights')
            np.testing.assert_allclose(weights.sum(-1),1.,atol=1e-12,rtol=0)
            if condition==3:
                q=bank['diagnostic_target_pairing'][:,a]
                marginal=bank['diagnostic_pairing_marginal'][:,a]
                if not np.isfinite(q).all() or np.any(q<0) or not np.isfinite(marginal).all() or np.any(marginal<0):
                    raise ValueError('Invalid conditional pairing beliefs')
                np.testing.assert_allclose(q.sum(-1),present,atol=1e-12,rtol=0)
                np.testing.assert_allclose(marginal.sum(-1),1.,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(beta[~present],0.)
            if np.any(contribution<0) or np.any(contribution>1) or np.any(contribution[~present]):
                raise ValueError('Invalid target contribution')
            if arm.startswith('AVAILABLE') and not present.all(): raise ValueError('Oracle availability failed')
            if 'PRECISION' in arm: np.testing.assert_array_equal(beta[present],25.)
            if 'EXECUTE' in arm:
                np.testing.assert_array_equal(contribution,1.)
                np.testing.assert_allclose(p,target_p,atol=1e-12,rtol=0)
            correct=choice==categories; pc=np.take_along_axis(p,np.broadcast_to(categories-1,choice.shape)[...,None],-1)[...,0]
            e,truth=events(choice.reshape(-1,nt),categories,observed,ncat)
            features=behavior_features(choice.reshape(-1,nt),categories,n_categories=ncat,block_width=width)
            tail=behavior_features(choice[:,:,-late:].reshape(-1,late),categories[-late:],n_categories=ncat,block_width=width)
            energy,se,jk,cross=energies(features,real_full,labels)
            le,lse,ljk,lcross=energies(tail,real_late,labels)
            late_accuracy=correct[:,:,-late:].mean(-1)
            reach=late_accuracy>=obs_correct[-late:].mean()
            late_mask=present[:,:,-late:]
            ever=np.maximum.accumulate(present,axis=-1)
            risk=ever.sum(-1); lost=(ever & ~present).sum(-1)
            row=dict(tag,arm=arm,paths=nf*nb,filters=nf,energy=energy,energy_mc_se=se,cross_filter_energy=cross,
                late_energy=le,late_energy_mc_se=lse,late_cross_filter_energy=lcross,
                late_accuracy=float(late_accuracy.mean()),late_accuracy_mc_se=clustered_se(late_accuracy),
                late_q10=float(np.quantile(late_accuracy,.1)),late_q90=float(np.quantile(late_accuracy,.9)),
                observed_late=float(obs_correct[-late:].mean()),reach_count=int(reach.sum()),reach_probability=float(reach.mean()),
                reach_probability_mc_se=clustered_se(reach),late_correct_probability=float(pc[:,:,-late:].mean()),
                late_present=float(late_mask.mean()),late_contribution=float(contribution[:,:,-late:].mean()),
                late_precision_given_presence=float(beta[:,:,-late:][late_mask].mean()) if late_mask.any() else np.nan,
                first_present=float(present[:,:,0].mean()),ever_present=float(present.any(-1).mean()),
                absent_after_entry=float(np.divide(lost,risk,out=np.zeros_like(lost,dtype=float),where=risk>0)[risk>0].mean()) if (risk>0).any() else np.nan,
                jump_joint_count=int(e['jump_joint'].sum()),jump_joint_probability=float(e['jump_joint'].mean()),
                low_joint_count=int(e['low_joint'].sum()) if 'low_joint' in e else np.nan,
                low_joint_probability=float(e['low_joint'].mean()) if 'low_joint' in e else np.nan,
                observed_jump_gain=truth['gain'],observed_maintenance=truth['maintenance'],
                observed_minimum128=truth.get('minimum128',np.nan),observed_marginal_gap=truth.get('marginal_gap',np.nan),
                oracle_insertions=int(bank['diagnostic_inserted'][:,a].sum()),
                protected_proposals=int(bank['diagnostic_protected'][:,a].sum()))
            # Exact disjoint error classes apply to discrete execution arms.
            executed=bank['diagnostic_executed'][:,a]
            error=~correct
            if case['mode']!='MIX' or 'EXECUTE' in arm:
                absent=error & ~present; other=error & present & (executed!=case['target_rule'])
                target=error & (executed==case['target_rule'])
                np.testing.assert_array_equal(absent.astype(int)+other+target,error.astype(int))
                row.update(error_absent=int(absent.sum()),error_present_other=int(other.sum()),error_target_executed=int(target.sum()))
            rows.append(row)
            own[arm]=dict(row=row,jk=jk,late_jk=ljk,late_accuracy=late_accuracy,
                present=late_mask.mean(-1),contribution=contribution[:,:,-late:].mean(-1))
            for i in range(nf):
                for b in range(nb):
                    flat=i*nb+b
                    paths.append(dict(tag,arm=arm,filter=i,seed=int(bank['branch_seeds'][i,b]),
                        ancestor=int(bank['ancestors'][i,b]),late_accuracy=float(late_accuracy[i,b]),
                        late_present=float(late_mask[i,b].mean()),late_contribution=float(contribution[i,b,-late:].mean()),
                        jump_joint=bool(e['jump_joint'][flat]),jump_gain=float(e['jump_gain'][flat]),
                        maintenance=float(e['jump_maintenance'][flat]),
                        low_joint=bool(e['low_joint'][flat]) if 'low_joint' in e else None))
            for left in range(0,nt,width):
                sl=slice(left,min(left+width,nt)); values=correct[:,:,sl].mean(-1)
                mask=present[:,:,sl]
                curves.append(dict(tag,arm=arm,trial=case['cut']+sl.stop,observed=float(obs_correct[sl].mean()),
                    mean=float(values.mean()),mc_se=clustered_se(values),q10=float(np.quantile(values,.1)),q90=float(np.quantile(values,.9)),
                    present=float(mask.mean()),contribution=float(contribution[:,:,sl].mean()),
                    precision=float(beta[:,:,sl][mask].mean()) if mask.any() else np.nan))
            audit_trials+=choice.size
        for left,right in CONTRASTS:
            a,b=own[left],own[right]
            row=dict(tag,left=left,right=right)
            for metric,key in (('energy','jk'),('late_energy','late_jk')):
                row['delta_'+metric]=a['row'][metric]-b['row'][metric]
                row['delta_'+metric+'_mc_se']=jackknife_se(a[key]-b[key])
            for metric in ('late_accuracy','present','contribution'):
                value=a[metric]-b[metric]
                row['delta_'+metric]=float(value.mean()); row['delta_'+metric+'_mc_se']=clustered_se(value)
            contrasts.append(row)
        per_case[case['case_id']]=own
        start=task_start(case,hashes); np.testing.assert_array_equal(start['categories'][case['cut']:],categories)
        choice=start['choices'][:,case['cut']:]; corr=choice==categories
        ef=behavior_features(choice,categories,n_categories=ncat,block_width=width)
        tf=behavior_features(choice[:,-late:],categories[-late:],n_categories=ncat,block_width=width)
        en,ese,_,_=energies(ef,real_full,None); ln,lse,_,_=energies(tf,real_late,None)
        reference_rows.append(dict(tag,paths=len(choice),energy=en,energy_mc_se=ese,late_energy=ln,late_energy_mc_se=lse,
            late_accuracy=corr[:,-late:].mean(),late_accuracy_mc_se=corr[:,-late:].mean(1).std(ddof=1)/np.sqrt(len(choice)),
            observed_late=obs_correct[-late:].mean()))
    for name,data in [('scores',rows),('path_metrics',paths),('curves',curves),('endpoints',endpoints),
                      ('contrasts',contrasts),('observed',truth_rows),('task_start',reference_rows)]:
        pd.DataFrame(data).to_csv(output/(name+'.csv'),index=False)
    write(output/'completion.json',dict(status='complete',source=str(source.resolve()),cases=len(protocol['cases']),
        paths=len(paths),audited_generated_trials=audit_trials,inputs_sha256=hashes,
        summary_source_sha256=digest(Path(__file__)),
        uncertainty='Six independent prefix filters; all sixteen descendants deleted together for Energy jackknife',
        cross_filter_energy='Sensitivity excluding every within-filter pair from the ensemble-distance term',
        endpoint_belief='Legacy endpoint_belief is the stored prior; summaries use endpoint presence and precision only',
        scope='Fixed-parameter development oracle diagnosis; no refitting, recovery or oral validation',
        artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); summarize(args.source,args.output)
