"""Conditional parameter profiles and bounded evidence-decay recovery for H.

Profiles score training data with all other coordinates fixed at the frozen fit.
They are sensitivity diagnostics, not a joint posterior or a re-optimization.
Recovery generates synthetic training prefixes with known evidence decay and
fits that coordinate while fixing the remaining parameters to their true values.
All actual held-out choices remain outside these calculations. Every output is
new. The archived cognitive runtime and shared H scoring worker are reused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import REPO,H_PATH,digest,read,write,source_hashes,execute,model_space
from ...optimization.search.adaptive_proposals import candidate
from ...utils.seeding import stable_seed
from ...evaluation.model_0923_h_validation import jackknife_scores,jackknife_se


def initialized(fit: Path, config: Path, output: Path, role: str) -> tuple[dict,dict]:
    if output.exists():raise FileExistsError('Use a new parameter diagnostic destination')
    frozen=read(fit/'nominees.json');settings=yaml.safe_load(config.read_text())
    if frozen['smoke'] or frozen['status']!='frozen_before_test':raise ValueError('Need frozen scientific nominees')
    for name,sha in frozen['source_sha256'].items():
        if digest(REPO/name)!=sha:raise ValueError('Fit source changed')
    output.mkdir(parents=True)
    write(output/'protocol.json',dict(role=role,fit=str(fit.resolve()),nomination_sha256=digest(fit/'nominees.json'),
        settings=settings,source_sha256=digest(Path(__file__)),
        numerical_scoring_sha256=digest(REPO/'src/Bayesian_state/evaluation/model_0923_h_validation.py'),
        all_unvaried_coordinates='frozen_nominee',
        actual_data_boundary='training arrays only; held-out choices excluded'))
    return frozen,settings


def base_job(frozen: dict, settings: dict, row: dict, role: str) -> dict:
    data=Path(frozen['data']);preparation=read(data/'preparation_manifest.json')
    hashes=source_hashes();hashes[str(Path(__file__).resolve().relative_to(REPO))]=digest(Path(__file__))
    return dict(phase='training',folder=str(data/f'subject_{row["subject"]}'),model=row['model'],
        mechanism=frozen['settings']['models'][row['model']],runtime=str((REPO/preparation['settings']['source_fit']/'source').resolve()),
        seeds=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],role=role,subject=row['subject'],repeat=i))
               for i in range(settings['seeds'])],particles=settings['particles'],source_sha256=hashes,smoke=False)


def profile_points(point: dict, space: dict, axes: list[str], pairs: list[list[str]]) -> tuple[dict,list[dict]]:
    """Share duplicate parameter points across one- and two-coordinate profiles."""
    bank={};members=[]
    for axis in axes:
        for value in space[axis]:
            p=deepcopy(point);p[axis]=value;c=candidate(p,'conditional_profile')
            bank[c['id']]=c;members.append(dict(candidate=c['id'],profile=axis,first=value,second=None))
    for first,second in pairs:
        for a in space[first]:
            for b in space[second]:
                p=deepcopy(point);p.update({first:a,second:b});c=candidate(p,'conditional_profile')
                bank[c['id']]=c;members.append(dict(candidate=c['id'],profile=first+' | '+second,first=a,second=b))
    return bank,members


def numerical_scores(file: Path) -> tuple[np.ndarray,np.ndarray]:
    """Retain whole-seed dependence when measuring numerical scoring error."""
    row=read(file)
    if digest(file.with_suffix('.npz'))!=row['predictions_sha256']:raise ValueError('Profile predictions changed')
    with np.load(file.with_suffix('.npz')) as z:
        return z['seeds'],jackknife_scores(z['observed_probability'])


def profiles(fit: Path, config: Path, output: Path) -> None:
    frozen,settings=initialized(fit,config,output,'conditional_training_profiles')
    jobs=[];members=[]
    for row in frozen['nominees']:
        if row['model'] not in settings['profile_models']:continue
        space,_=model_space(frozen['settings'],row['model'])
        axes=[H_PATH+'intervention.decay','engine.modules.memory_mod.kwargs.gamma',
              H_PATH+('intervention.stay_bias' if row['model']=='JS_S' else 'exit_rate'),
              H_PATH+'max_capacity',H_PATH+'search_controller.baseline_probability']
        bank,links=profile_points(row['candidate']['hyperparams'],space,axes,[axes[:2]])
        for item in links:members.append(dict(subject=row['subject'],model=row['model'],**item))
        for c in bank.values():
            job=base_job(frozen,settings,row,'profile')
            job.update(candidate=c,name=f'S{row["subject"]}_{row["model"]}_{c["id"]}',
                target=str((output/f'subject_{row["subject"]}'/row['model']/c['id']).resolve()))
            jobs.append(job)
    write(output/'profile_membership.json',dict(rows=members,unique_points=len(jobs)))
    execute(jobs,output,settings['parallel_budget'])
    lookup={};numerical={};anchors={}
    for row in frozen['nominees']:
        if row['model'] in settings['profile_models']:
            anchors[(row['subject'],row['model'])]=row['candidate']['id']
    for job in jobs:
        file=Path(job['target']).with_suffix('.json');row=read(file)
        key=(row['subject'],row['model'],row['candidate']['id'])
        lookup[key]=row;numerical[key]=numerical_scores(file)
    result=[]
    for link in members:
        row=lookup[(link['subject'],link['model'],link['candidate'])]
        key=(link['subject'],link['model']);anchor_key=(*key,anchors[key])
        seeds,values=numerical[(*key,link['candidate'])];anchor_seeds,anchor_values=numerical[anchor_key]
        np.testing.assert_array_equal(seeds,anchor_seeds)
        result.append(dict(link,mean_nll=row['mean_nll'],trials=row['trials'],particles=row['particles'],
            mean_nll_mc_se=jackknife_se(values),
            delta_anchor_nll=row['mean_nll']-lookup[anchor_key]['mean_nll'],
            delta_anchor_mc_se=jackknife_se(values-anchor_values)))
    frame=pd.DataFrame(result)
    frame['delta_nll']=frame.mean_nll-frame.groupby(['subject','model','profile']).mean_nll.transform('min')
    frame['delta_total_nll']=frame.delta_nll*frame.trials
    frame['delta_anchor_total_nll']=frame.delta_anchor_nll*frame.trials
    frame['delta_anchor_total_mc_se']=frame.delta_anchor_mc_se*frame.trials
    frame.to_csv(output/'profiles.csv',index=False)
    write(output/'completion.json',dict(status='complete',unique_points=len(jobs),
        scientific_status='conditional_profiles_with_fixed_nuisance_parameters; not_full_identifiability'))


def copied_metadata(original: Path, destination: Path) -> dict:
    destination.mkdir(parents=True)
    engine=read(original/'engine.json');write(destination/'engine.json',engine)
    meta=deepcopy(read(original/'prepared.json'))
    meta['engine_sha256']=digest(destination/'engine.json')
    return meta


def save_arrays(folder: Path, name: str, arrays: dict) -> str:
    with (folder/name).open('xb') as f:np.savez_compressed(f,**arrays)
    return digest(folder/name)


def recovery(fit: Path, config: Path, output: Path) -> None:
    frozen,settings=initialized(fit,config,output,'known_nuisance_decay_recovery')
    data=Path(frozen['data']);generation=[];truths=[]
    for subject in settings['recovery_subjects']:
        row=next(r for r in frozen['nominees'] if r['subject']==subject and r['model']==settings['recovery_model'])
        with np.load(data/f'subject_{subject}'/'training.npz') as z:observed={k:z[k] for k in z.files}
        for decay in settings['recovery_decay']:
            label=f'S{subject}_decay{str(decay).replace(".","p")}'
            folder=output/'generation_data'/label
            meta=copied_metadata(data/f'subject_{subject}',folder)
            n=len(observed['choices']);meta.update(training_trials=n,test_trials=0,trials=n,
                array_sha256={'training.npz':save_arrays(folder,'training.npz',observed),
                    'sealed_test.npz':save_arrays(folder,'sealed_test.npz',dict(stimulus=observed['stimulus'][:0],categories=observed['categories'][:0]))},
                diagnostic_boundary='original training prefix only; empty synthetic test suffix')
            write(folder/'prepared.json',meta)
            point=deepcopy(row['candidate']['hyperparams']);point[H_PATH+'intervention.decay']=decay
            truth=candidate(point,'recovery_truth');job=base_job(frozen,settings,row,'recovery_generation')
            job.update(phase='autonomous',name=label,folder=str(folder.resolve()),candidate=truth,
                target=str((output/'generated'/label).resolve()),
                seeds=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],role='recovery_generation',
                    subject=subject,decay=decay,path=i)) for i in range(settings['recovery_repeats'])])
            generation.append(job);truths.append(dict(subject=subject,decay=decay,label=label,candidate=truth))
    write(output/'truths.json',dict(rows=truths))
    execute(generation,output,settings['parallel_budget'])
    jobs=[];membership=[]
    for gen,truth in zip(generation,truths):
        original=Path(gen['folder']);meta=read(original/'prepared.json')
        with np.load(original/'training.npz') as z:physical={k:z[k] for k in ('stimulus','categories')}
        with np.load(Path(gen['target']).with_suffix('.npz')) as z:
            choices,feedback=z['choices'],z['feedback']
        for replicate in range(len(choices)):
            folder=output/'synthetic_data'/(truth['label']+f'_R{replicate:02d}')
            case=copied_metadata(original,folder)
            case.update(array_sha256={'training.npz':save_arrays(folder,'training.npz',
                dict(physical,choices=choices[replicate],feedback=feedback[replicate]))},
                synthetic_truth=truth,synthetic_seed=gen['seeds'][replicate])
            write(folder/'prepared.json',case)
            for value in settings['decay_support']:
                point=deepcopy(truth['candidate']['hyperparams']);point[H_PATH+'intervention.decay']=value
                c=candidate(point,'recovery_candidate');job=deepcopy(gen)
                job.update(phase='training',folder=str(folder.resolve()),candidate=c,
                    name=truth['label']+f'_R{replicate:02d}_{c["id"]}',
                    target=str((output/'recovery_scores'/truth['label']/f'R{replicate:02d}'/c['id']).resolve()),
                    seeds=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],role='recovery_scoring',
                        subject=truth['subject'],true_decay=truth['decay'],replicate=replicate,repeat=i)) for i in range(settings['seeds'])])
                jobs.append(job);membership.append(dict(subject=truth['subject'],true_decay=truth['decay'],
                    replicate=replicate,candidate_decay=value,score_file=job['target']+'.json'))
    execute(jobs,output,settings['parallel_budget'])
    for item in membership:
        row=read(Path(item['score_file']));_,values=numerical_scores(Path(item['score_file']))
        item.update(mean_nll=row['mean_nll'],trials=row['trials'],mean_nll_mc_se=jackknife_se(values))
    scores=pd.DataFrame(membership);scores.to_csv(output/'recovery_scores.csv',index=False)
    estimates=scores.sort_values(['mean_nll','candidate_decay']).groupby(['subject','true_decay','replicate'],sort=True).first().reset_index()
    gaps=[]
    for _,group in scores.groupby(['subject','true_decay','replicate'],sort=True):
        ordered=group.sort_values(['mean_nll','candidate_decay']);best,second=ordered.iloc[:2].to_dict('records')
        seeds,best_values=numerical_scores(Path(best['score_file']))
        second_seeds,second_values=numerical_scores(Path(second['score_file']))
        np.testing.assert_array_equal(seeds,second_seeds)
        gaps.append((second['mean_nll']-best['mean_nll'],jackknife_se(second_values-best_values)))
    estimates['top_gap_nll']=[g[0] for g in gaps];estimates['top_gap_mc_se']=[g[1] for g in gaps]
    estimates.to_csv(output/'recovered_decay.csv',index=False)
    write(output/'completion.json',dict(status='complete',synthetic_sequences=len(estimates),score_jobs=len(jobs),
        scientific_status='evidence_decay_only_recovery_with_all_nuisance_parameters_known; not_joint_parameter_recovery'))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['profiles','recovery'],required=True)
    for name in ('fit','config','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();{'profiles':profiles,'recovery':recovery}[args.phase](args.fit,args.config,args.output)


if __name__=='__main__':main()
