"""Summarize every frozen-parameter episodic-admission path and filter seed.

Whole-path / whole-filter jackknife errors describe Monte Carlo resolution.
Training and autonomous outcomes are separate development diagnostics. The
summary accepts optional score runs with disjoint filter seed sets, retaining
their own rows and comparing differences against independent Monte Carlo errors.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import (load,digest,nll,block_paths,distribution_scores,
                                     jackknife_scores,jackknife_se)
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_0923_h_validation import write


def compare_score_runs(rows: list[dict], seed_sets: dict) -> list[dict]:
    """Compare independent numerical runs at the same frozen parameter point."""
    comparisons = []
    for (subject,model),group in pd.DataFrame(rows).groupby(['subject','model'],sort=False):
        for first,second in combinations(group.to_dict('records'),2):
            keys = [(r['run'],subject,model) for r in (first,second)]
            if set(seed_sets[keys[0]]) & set(seed_sets[keys[1]]):
                raise ValueError('Run comparison requires independent filter seed sets')
            row = dict(subject=subject,model=model,reference_run=first['run'],comparison_run=second['run'],
                reference_particles=first['particles'],comparison_particles=second['particles'],
                reference_seeds=first['seeds'],comparison_seeds=second['seeds'])
            for metric,error in [('choice_nll','mc_se'),('delta_nll','delta_nll_mc_se')]:
                difference = second[metric]-first[metric]
                se = float(np.hypot(first[error],second[error]))
                row[metric+'_difference'] = difference
                row[metric+'_difference_mc_se'] = se
                row[metric+'_difference_in_mc_se'] = difference/se if se else np.nan
            comparisons.append(row)
    return comparisons


def read_run(folder: Path) -> tuple[dict,dict,dict]:
    protocol = load(folder/'protocol.json'); completion = load(folder/'completion.json')
    if (completion['status'] != 'complete' or protocol['smoke']
            or completion['protocol_sha256'] != digest(folder/'protocol.json')):
        raise ValueError('Need a complete scientific run with unchanged protocol')
    if (completion['jobs'] != len(protocol['jobs'])
            or len({j['name'] for j in protocol['jobs']}) != len(protocol['jobs'])):
        raise ValueError('Incorrect job count or duplicate job names')
    files = {str(folder/'protocol.json'):digest(folder/'protocol.json')}; groups = {}
    for job in protocol['jobs']:
        target = Path(job['target']); row = load(target.with_suffix('.json'))
        if digest(target.with_suffix('.npz')) != row['predictions_sha256']:
            raise ValueError('Scientific prediction array changed')
        if row['candidate']['hyperparams'] != job['candidate']['hyperparams']:
            raise ValueError('Parameters differ from the protocol')
        files[str(target.with_suffix('.npz'))] = row['predictions_sha256']
        with np.load(target.with_suffix('.npz')) as z: arrays = {k:z[k] for k in z.files}
        np.testing.assert_array_equal(arrays['seeds'],job['seeds'])
        for key in ('probabilities','correct_probability','observed_probability'):
            if key in arrays and (not np.isfinite(arrays[key]).all() or np.any(arrays[key]<0)
                                  or np.any(arrays[key]>1+1e-12)):
                raise ValueError('Invalid saved probability')
        if 'probabilities' in arrays:
            np.testing.assert_allclose(arrays['probabilities'].sum(axis=-1),1.,atol=1e-12,rtol=0)
        groups.setdefault((job['subject'],job['model']),[]).append((arrays,job,row))
    expected = {(s,m) for s in protocol['settings']['subjects'] for m in protocol['settings']['arms']}
    if set(groups) != expected:raise ValueError('Run has missing or extra comparison cells')
    result = {}
    for key,chunks in groups.items():
        for _,job,row in chunks:
            for name in ('candidate','particles','mechanism','precision'):
                if job[name] != chunks[0][1][name]:raise ValueError('Within-cell parameters or budget changed')
        arrays = {k:(chunks[0][0][k] if k in ('categories','oracle') else
                     np.concatenate([c[0][k] for c in chunks],axis=0)) for k in chunks[0][0]}
        for name in ('categories','oracle'):
            if name in arrays:
                for chunk,_,_ in chunks:np.testing.assert_array_equal(chunk[name],arrays[name])
        if len(set(arrays['seeds'])) != len(arrays['seeds']):raise ValueError('Duplicate seeds in a cell')
        expected_count = protocol['settings']['paths'] if protocol['stage']=='autonomous' else len(protocol['settings']['score']['seeds'])
        if len(arrays['seeds']) != expected_count:raise ValueError('Incorrect replicate count')
        order = np.argsort(arrays['seeds'])
        for name in arrays:
            if name not in ('categories','oracle'):arrays[name]=arrays[name][order]
        result[key] = (arrays,chunks[0][1],chunks[0][2])
    return protocol,result,files


def summarize(autonomous: Path, score: list[Path], output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new episodic summary directory')
    protocol,cells,inputs = read_run(autonomous)
    if protocol['stage'] != 'autonomous':raise ValueError('Need autonomous paths')
    settings = protocol['settings']; width = settings['features']['block_width']
    rows,curves,path_rows,blocks,contrasts = [],[],[],[],[]
    banks = {};humans = {}
    for (s,m),(a,job,record) in cells.items():
        folder = Path(job['folder']);meta=load(folder/'prepared.json')
        parts=[]
        for name in ('training.npz','sealed_test.npz'):
            if digest(folder/name) != meta['array_sha256'][name]:raise ValueError('Behavior data changed')
            inputs[str(folder/name)] = meta['array_sha256'][name]
            with np.load(folder/name) as z:parts.append((z['choices'],z['categories']))
        human_choice=np.concatenate([p[0] for p in parts]);categories=np.concatenate([p[1] for p in parts])
        np.testing.assert_array_equal(a['categories'],categories)
        humans[s]=(human_choice,categories,meta)
        generated=a['choices']==categories[None,:];human=human_choice==categories
        paths,n=generated.shape;target=int(a['oracle']);last=min(n,128)
        if n != meta['trials']:raise ValueError('Path task length changed')
        proposed=a['proposed']==target; rejected=proposed & (a['rejected']>0)
        # Paired path vectors retain the identity of complete simulations.
        vectors=dict(late_accuracy=generated[:,-last:].mean(axis=1),
            discovery=a['present'].max(axis=1),late_presence=a['present'][:,-last:].mean(axis=1),
            late_contribution=a['contribution'][:,-last:].mean(axis=1),
            target_proposals=proposed.sum(axis=1),target_rejections=rejected.sum(axis=1),
            target_exits=a['target_exited'].sum(axis=1),target_capacity_drops=a['target_capacity_dropped'].sum(axis=1))
        samples=block_paths(generated,0,n,width);observed=block_paths(human[None,:],0,n,width)[0]
        metrics=distribution_scores(samples,observed);jk=path_jackknife(samples,observed)
        lower,upper=np.quantile(samples,[.1,.9],axis=0)
        row=dict(subject=s,model=m,condition=meta['condition'],paths=paths,trials=n,
            training_trials=meta['training_trials'],oracle=target,observed_late=float(human[-last:].mean()),
            early_accuracy=float(generated[:,:last].mean()),observed_early=float(human[:last].mean()),
            paths_at_human_late=int((vectors['late_accuracy']>=human[-last:].mean()).sum()),
            active_count=float(a['active_count'].mean()),**metrics)
        for name,values in vectors.items():
            row[name]=float(values.mean());row[name+'_mc_se']=float(values.std(ddof=1)/np.sqrt(paths))
        for name in jk:row[name+'_mc_se']=jackknife_se(jk[name])
        rows.append(row);banks[s,m]=(a['seeds'],jk,vectors)
        present=block_paths(a['present'],0,n,width).mean(axis=0)
        contribution=block_paths(a['contribution'],0,n,width).mean(axis=0)
        for b in range(samples.shape[1]):
            curves.append(dict(subject=s,model=m,trial=min((b+1)*width,n),training_trials=meta['training_trials'],
                observed=observed[b],autonomous=float(samples[:,b].mean()),lower=lower[b],upper=upper[b],
                presence=present[b],contribution=contribution[b]))
        for i,seed in enumerate(a['seeds']):
            path_rows.append(dict(subject=s,model=m,seed=int(seed),observed_late=row['observed_late'],
                                  **{k:float(v[i]) for k,v in vectors.items()}))
            blocks.extend(dict(subject=s,model=m,seed=int(seed),trial=min((b+1)*width,n),accuracy=float(value))
                          for b,value in enumerate(samples[i]))
    frame=pd.DataFrame(rows);index=frame.set_index(['subject','model'])
    for (s,m),(seeds,jk,vectors) in banks.items():
        baseline=index.loc[s,'BASE'];base_seeds,base_jk,base_vectors=banks[s,'BASE']
        np.testing.assert_array_equal(seeds,base_seeds)
        r=dict(subject=s,model=m)
        for name in jk:
            r['delta_'+name]=index.loc[s,m][name]-baseline[name]
            r['delta_'+name+'_mc_se']=jackknife_se(jk[name]-base_jk[name])
        for name,values in vectors.items():
            diff=values-base_vectors[name];r['delta_'+name]=float(diff.mean())
            r['delta_'+name+'_mc_se']=float(diff.std(ddof=1)/np.sqrt(len(diff)))
        contrasts.append(r)
    planned=[('BASE','LAST'),('LAST','NEW'),('NEW','POOL'),('POOL','DRAW')]
    mechanism_contrasts=[]
    for s in settings['subjects']:
        for reference,alternative in planned:
            if (s,reference) not in banks or (s,alternative) not in banks:continue
            seeds,jk,values=banks[s,alternative];bs,bjk,bvalues=banks[s,reference]
            np.testing.assert_array_equal(seeds,bs)
            r=dict(subject=s,reference=reference,alternative=alternative)
            for metric in jk:
                r['delta_'+metric]=index.loc[s,alternative][metric]-index.loc[s,reference][metric]
                r['delta_'+metric+'_mc_se']=jackknife_se(jk[metric]-bjk[metric])
            for metric in values:
                diff=values[metric]-bvalues[metric]
                r['delta_'+metric]=float(diff.mean())
                r['delta_'+metric+'_mc_se']=float(diff.std(ddof=1)/np.sqrt(len(diff)))
            mechanism_contrasts.append(r)
    score_rows,filter_rows,score_curves,training_mechanisms = [],[],[],[]
    outcome_rows,block_rows=[] ,[]
    score_seeds = {}
    for directory in score:
        sp,sc,files=read_run(directory);inputs.update(files)
        if sp['stage']!='score' or sp['nomination_sha256']!=protocol['nomination_sha256']:
            raise ValueError('Score and path runs have different fitted parameters')
        for name,sha in sp['source_sha256'].items():
            if protocol['source_sha256'].get(name)!=sha:raise ValueError('Score and path implementations differ')
        for (s,m),(a,j,r) in sc.items():
            key = (directory.name,s,m)
            if key in score_seeds:raise ValueError('Score run names must be unique')
            score_seeds[key] = a['seeds'].tolist()
            base,bj,br=sc[s,'BASE'];np.testing.assert_array_equal(a['seeds'],base['seeds'])
            if j['candidate']['hyperparams']!=cells[s,m][1]['candidate']['hyperparams']:
                raise ValueError('Frozen parameters differ between scores and paths')
            count=humans[s][2]['training_trials'];p=a['observed_probability'];bp=base['observed_probability']
            if p.shape[1]!=count or r['particles']!=br['particles']:raise ValueError('Training boundary or budget mismatch')
            sj=jackknife_scores(p);bj=jackknife_scores(bp)
            score_rows.append(dict(subject=s,model=m,run=directory.name,particles=r['particles'],
                seeds=len(a['seeds']),trials=count,choice_nll=nll(p),mc_se=jackknife_se(sj),
                delta_nll=nll(p)-nll(bp),delta_nll_mc_se=jackknife_se(sj-bj)))
            filter_rows.extend(dict(subject=s,model=m,run=directory.name,seed=int(seed),nll=nll(p[i:i+1]))
                               for i,seed in enumerate(a['seeds']))
            correct=block_paths(a['correct_probability'],0,count,width).mean(axis=0)
            score_curves.extend(dict(subject=s,model=m,run=directory.name,trial=min((b+1)*width,count),
                                     correct_probability=float(value)) for b,value in enumerate(correct))
            human_correct=humans[s][0][:count]==humans[s][1][:count]
            for outcome,mask in [('correct',human_correct),('incorrect',~human_correct)]:
                n_outcome=int(mask.sum());fraction=n_outcome/count
                delta=nll(p[:,mask])-nll(bp[:,mask]) if n_outcome else np.nan
                se=jackknife_se(jackknife_scores(p[:,mask])-jackknife_scores(bp[:,mask])) if n_outcome else np.nan
                outcome_rows.append(dict(subject=s,model=m,run=directory.name,outcome=outcome,count=n_outcome,
                    fraction=fraction,choice_nll=nll(p[:,mask]) if n_outcome else np.nan,delta_nll=delta,
                    delta_nll_mc_se=se,contribution=delta*fraction if n_outcome else 0.,
                    contribution_mc_se=se*fraction if n_outcome else 0.))
            for first in range(0,count,width):
                last=min(first+width,count);part=slice(first,last)
                block_rows.append(dict(subject=s,model=m,run=directory.name,trial=last,count=last-first,
                    observed_accuracy=float(human_correct[part].mean()),choice_nll=nll(p[:,part]),
                    delta_nll=nll(p[:,part])-nll(bp[:,part]),
                    delta_nll_mc_se=jackknife_se(jackknife_scores(p[:,part])-jackknife_scores(bp[:,part]))))
        for s in sp['settings']['subjects']:
            for reference,alternative in planned:
                if (s,reference) not in sc or (s,alternative) not in sc:continue
                a,_,_=sc[s,alternative];b,_,_=sc[s,reference]
                np.testing.assert_array_equal(a['seeds'],b['seeds'])
                p,bp=a['observed_probability'],b['observed_probability']
                training_mechanisms.append(dict(subject=s,reference=reference,alternative=alternative,run=directory.name,
                    delta_nll=nll(p)-nll(bp),delta_nll_mc_se=jackknife_se(jackknife_scores(p)-jackknife_scores(bp))))
    cohort=[];contrast_frame=pd.DataFrame(contrasts)
    for m in settings['arms']:
        group=frame[frame.model==m];dif=contrast_frame[contrast_frame.model==m]
        row=dict(model=m,subjects=len(group))
        for metric in ('energy','crps','late_accuracy','discovery','late_presence'):
            row[metric]=float(group[metric].mean());row['delta_'+metric]=float(dif['delta_'+metric].mean())
            row['delta_'+metric+'_mc_se']=float(np.linalg.norm(dif['delta_'+metric+'_mc_se'])/len(group))
        cohort.append(row)
    output.mkdir(parents=True)
    tables=dict(scores=frame,contrasts=contrast_frame,cohort=pd.DataFrame(cohort),curves=pd.DataFrame(curves),
                paths=pd.DataFrame(path_rows),path_blocks=pd.DataFrame(blocks),
                mechanism_contrasts=pd.DataFrame(mechanism_contrasts))
    if score:
        tables.update(training=pd.DataFrame(score_rows),filters=pd.DataFrame(filter_rows),
                      conditional_curves=pd.DataFrame(score_curves),training_mechanisms=pd.DataFrame(training_mechanisms),
                      training_outcomes=pd.DataFrame(outcome_rows),training_blocks=pd.DataFrame(block_rows))
        keys=['subject','model','run'];overall=tables['training'].set_index(keys)
        summed=tables['training_outcomes'].groupby(keys).contribution.sum().reindex(overall.index)
        np.testing.assert_allclose(summed,overall.delta_nll,atol=1e-12,rtol=0)
        numerical = compare_score_runs(score_rows,score_seeds)
        if numerical:tables['training_numerical_comparison'] = pd.DataFrame(numerical)
    for name,table in tables.items():table.to_csv(output/(name+'.csv'),index=False)
    write(output/'manifest.json',dict(autonomous=str(autonomous.resolve()),score=[str(p.resolve()) for p in score],
        settings=settings,source_sha256=protocol['source_sha256'],input_sha256=inputs,
        tables={name+'.csv':digest(output/(name+'.csv')) for name in tables},
        evaluation_source_sha256=digest(Path(__file__)),
        scope='fixed-parameter development; complete-seed/path Monte Carlo errors; no external validation'))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--autonomous',type=Path,required=True)
    p.add_argument('--score',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.autonomous,a.score,a.output)


if __name__=='__main__':main()
