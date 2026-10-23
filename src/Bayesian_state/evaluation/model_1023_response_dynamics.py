"""Score complete paired dynamic probes, retaining every frozen parameter point.

Use --study COMPLETE --output NEW. Full autonomous learning and prefix-based
future generation are scored separately. Paired simulation errors use shared
trajectory seeds or prefix-filter clusters; they do not measure human sampling
uncertainty. The registered gate only authorizes the next development comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflows.analysis.model_1023_response_dynamics import read,write,digest,verify
from ..metrics.task import category_learning_feedback
from .model_0923_h_validation import checked_arrays,block_paths,distribution_scores,jackknife_se
from .model_0923_h_prefix_forecast import filter_cluster_jackknife,clustered_mean_se
from .model_0923_h_generation_uncertainty import path_jackknife
from .model_1023_prefix_forecast import conditional_mean


def assemble(jobs: list[dict], role: str) -> tuple[dict,dict]:
    """Concatenate paths, while keeping prefix clouds on a separate filter axis."""
    chunks,inputs=[],{}
    for job in jobs:
        file=Path(job['target']).with_suffix('.npz');record=read(file.with_suffix('.json'))
        inputs[str(file)]=record['predictions_sha256']
        inputs[str(file.with_suffix('.json'))]=digest(file.with_suffix('.json'))
        verify({str(file):record['predictions_sha256']})
        if record['candidate']!=job['candidate']:
            raise ValueError('Chunk candidate changed')
        if role=='prefix' and (record['filter_seed']!=job['seeds'][0] or record.get('status')!='complete'):
            raise ValueError('Incomplete or mismatched prefix filter')
        with np.load(file) as z:chunks.append({k:z[k] for k in z.files})
    if not chunks:raise ValueError('Missing generated chunks')
    keys=set(chunks[0]);result={}
    if any(set(c)!=keys for c in chunks):raise ValueError('Chunk array schemas differ')
    for k in keys:
        if k=='categories':
            for c in chunks[1:]:np.testing.assert_array_equal(c[k],chunks[0][k])
            result[k]=chunks[0][k]
        elif role=='prefix' and (k.startswith('endpoint_') or k in ('terminal_weights','prefix_probabilities','filter_seed')):
            result[k]=np.stack([c[k] for c in chunks])
        else:result[k]=np.concatenate([c[k] for c in chunks])
    if role=='prefix':
        result['branch_filter_seed']=np.concatenate([np.repeat(c['filter_seed'],len(c['branch_seeds'])) for c in chunks])
    seeds=result['branch_seeds' if role=='prefix' else 'seeds']
    if len(np.unique(seeds))!=len(seeds):raise ValueError('Repeated generated path seed')
    return result,inputs


def summarize(study: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new dynamic summary directory')
    protocol,done=read(study/'protocol.json'),read(study/'completion.json')
    if done['status']!='complete' or done['smoke'] or digest(study/'protocol.json')!=done['protocol_sha256']:
        raise ValueError('Require a complete scientific dynamic probe')
    verify(protocol['source_sha256']);verify(protocol['inputs_sha256'])
    inputs={str(p):digest(p) for p in (study/'protocol.json',study/'completion.json',Path(__file__).resolve())}
    jobs=[read(f) for f in sorted((study/'jobs').glob('*.json'))]
    if len(jobs)!=done['jobs']:raise ValueError('Incomplete job manifest')
    output.mkdir(parents=True);(output/'arrays').mkdir()
    scores,curves,paths,states,contrasts=[],[],[],[],[]
    tested_trials=0
    width=protocol['settings']['block_width'];late=protocol['settings']['late_trials']
    for case in protocol['cases']:
        own=Path(case['evaluation']);bank={};records={}
        for role in ('conditional','autonomous'):
            record,=read(own/role/'evaluation.json')['rows']
            if record['candidate']['id']!=case['candidate'] or record['candidate']['hyperparams']!=case['candidate_hyperparams']:
                raise ValueError('Reference candidate changed')
            bank[role]=checked_arrays(own/role,record);records[role]=record
            inputs[str(own/role/'evaluation.json')]=digest(own/role/'evaluation.json')
            inputs[str(own/role/'evaluate'/f"subject_{case['subject']}"/'COMPACT'/(case['candidate']+'.npz'))]=record['predictions_sha256']
        cut=records['conditional']['training_trials'];conditional=bank['conditional']
        tag=dict(subject=case['subject'],condition=case['condition'],candidate=case['candidate'],roles='|'.join(case['roles']),
                 selected='selected' in case['roles'],cohort=case['cohort'],training_trials=cut)
        case_jobs=[j for j in jobs if j['case_id']==case['case_id']]
        saved={};jk={}
        for role in ('task_start','prefix'):
            first=0 if role=='task_start' else cut
            actual=conditional['choices'][first:]==conditional['categories'][first:]
            observed=block_paths(actual[None,:],0,len(actual),width)[0]
            for exponent in (1,2):
                if exponent==1 and role=='task_start':arrays=bank['autonomous']
                else:
                    group=[j for j in case_jobs if j['dynamic_role']==role and j['response_exponent']==exponent]
                    if exponent==1 and role=='prefix' and case['baseline_prefix_jobs']:
                        group=[read(Path(f)) for f in case['baseline_prefix_jobs']]
                    arrays,bindings=assemble(group,role);inputs.update(bindings)
                np.testing.assert_array_equal(arrays['categories'],conditional['categories'][first:])
                seeds=arrays['seeds' if role=='task_start' else 'branch_seeds']
                labels=seeds if role=='task_start' else arrays['branch_filter_seed']
                if arrays['choices'].shape!=(128,len(actual)):raise ValueError('Wrong generation budget')
                p=arrays['probabilities']
                if not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid choice probabilities')
                np.testing.assert_allclose(p.sum(-1),1.,atol=1e-12,rtol=0)
                expected=np.array([[category_learning_feedback(int(a),int(c),condition=case['condition'])
                    for a,c in zip(path,arrays['categories'])] for path in arrays['choices']])
                np.testing.assert_array_equal(arrays['feedback'],expected)
                tested_trials+=arrays['choices'].size
                if role=='prefix':
                    np.testing.assert_array_equal(arrays['filter_seed'],conditional['seeds'])
                    if exponent==1:np.testing.assert_array_equal(arrays['prefix_probabilities'],conditional['probabilities'][:,:cut])
                    if exponent==2:
                        raw=arrays['diagnostic_cognitive_probabilities'];squared=raw**2;squared/=squared.sum(-1,keepdims=True)
                        np.testing.assert_allclose(p,squared,atol=1e-12,rtol=0)
                generated=arrays['choices']==arrays['categories'][None,:]
                samples=block_paths(generated,0,len(actual),width)
                metrics=distribution_scores(samples,observed)
                jack=(path_jackknife(samples,observed) if role=='task_start'
                      else filter_cluster_jackknife(samples,observed,labels)[1])
                jk[role,exponent]=jack
                trace={k:arrays[k if role=='task_start' else 'diagnostic_'+k]
                       for k in ('present','contribution','beta','search','active_count')}
                np.testing.assert_array_equal(trace['beta'][~trace['present'].astype(bool)],0.)
                late_paths=generated[:,-late:].mean(1);observed_late=float(actual[-late:].mean())
                row=dict(tag,start=role,exponent=exponent,**metrics,late_observed=observed_late,
                    late_accuracy=float(late_paths.mean()),late_q10=float(np.quantile(late_paths,.1)),
                    late_q90=float(np.quantile(late_paths,.9)),late_accuracy_mc_se=clustered_mean_se(late_paths,labels),
                    paths_reaching_observed=int((late_paths>=observed_late).sum()),
                    late_abs_mean_error=abs(late_paths.mean()-observed_late),
                    late_precision_given_presence=conditional_mean(trace['beta'][:,-late:],trace['present'][:,-late:].astype(bool)))
                for k in ('energy','crps'):row[k+'_mc_se']=jackknife_se(jack[k])
                for k in ('present','contribution','search','active_count'):row['late_'+k]=float(trace[k][:,-late:].mean())
                if role=='task_start':
                    for name,sl in [('early',slice(None,128)),('training',slice(None,cut)),('future',slice(cut,None))]:
                        local=block_paths(generated[:,sl],0,generated[:,sl].shape[1],width)
                        obs=block_paths(actual[None,sl],0,actual[sl].size,width)[0]
                        row[name+'_energy']=distribution_scores(local,obs)['energy']
                        row[name+'_accuracy']=float(generated[:,sl].mean())
                        row[name+'_observed']=float(actual[sl].mean())
                scores.append(row)
                saved[role,exponent]=(arrays,seeds,labels,late_paths,row)
                with (output/'arrays'/f"{case['case_id']}_{role}_Q{exponent}.npz").open('xb') as stream:
                    np.savez_compressed(stream,**arrays)
                for i,start in enumerate(range(0,len(actual),width)):
                    sl=slice(start,start+width);tick=first+start+width
                    curves.append(dict(tag,start=role,exponent=exponent,trial=tick,observed=observed[i],mean=samples[:,i].mean(),
                        q10=np.quantile(samples[:,i],.1),q90=np.quantile(samples[:,i],.9)))
                    paths.extend(dict(tag,start=role,exponent=exponent,trial=tick,seed=int(s),cluster=int(l),accuracy=float(v))
                        for s,l,v in zip(seeds,labels,samples[:,i]))
                    state=dict(tag,start=role,exponent=exponent,trial=tick,
                        precision_given_presence=conditional_mean(trace['beta'][:,sl],trace['present'][:,sl].astype(bool)))
                    state.update({k:float(v[:,sl].mean()) for k,v in trace.items() if k!='beta'});states.append(state)
            a,b=saved[role,1],saved[role,2]
            np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
            row=dict(tag,start=role)
            for metric in ('energy','crps'):
                row['delta_'+metric]=b[4][metric]-a[4][metric]
                row['delta_'+metric+'_mc_se']=jackknife_se(jk[role,2][metric]-jk[role,1][metric])
            row.update(delta_late_accuracy=float((b[3]-a[3]).mean()),
                       delta_late_accuracy_mc_se=clustered_mean_se(b[3]-a[3],a[2]),
                       delta_late_abs_mean_error=b[4]['late_abs_mean_error']-a[4]['late_abs_mean_error'])
            contrasts.append(row)
    for name,values in [('scores',scores),('curves',curves),('path_blocks',paths),('states',states),('contrasts',contrasts)]:
        pd.DataFrame(values).to_csv(output/(name+'.csv'),index=False)
    selected=[r for r in contrasts if r['selected'] and r['start']=='task_start']
    improving=[r['subject'] for r in selected if r['delta_energy']<0]
    gate=len(improving)>=2 and bool(set(improving)&{205,306,309})
    write(output/'decision.json',dict(proceed_to_equal_budget_refit=gate,improving_selected_subjects=improving,
        rule='At least two selected subjects improve full-learning Energy score, including 205,306 or309',
        interpretation='Development resource gate; mechanism efficacy remains under evaluation',selected_contrasts=selected))
    verify(inputs)
    write(output/'completion.json',dict(status='complete',study=str(study.resolve()),participants=6,candidates=18,
        feedback_trials_checked=tested_trials,inputs_sha256=inputs,
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()},
        uncertainty='Paired simulation error; trajectories or prefix-filter clusters',
        rule_contribution='Untransformed cognitive mixture weights; category power is applied after mixing'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summarize(args.study,args.output)
