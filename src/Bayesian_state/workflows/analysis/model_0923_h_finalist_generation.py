"""Generate equal path budgets at every frozen training finalist.

This diagnostic measures sensitivity to the retained parameter points. It does
not select a new nominee or interpret the candidate bank as a parameter posterior.
The original generator, archived cognitive runtime and outer worker pool are reused.
The frozen fit's data-use role is preserved in the diagnostic protocol.
Optional seed_role='autonomous', with the primary protocol name and seed,
reuses the primary generator's path seeds. Omitting it preserves earlier seeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import REPO,read,write,digest,source_hashes,score_jobs,execute,combine_evaluation_chunks
from ...utils.seeding import stable_seed
from ...evaluation.model_0923_h_validation import block_paths,distribution_scores


def trajectory_seeds(settings: dict, subject: int) -> list[int]:
    """Keep earlier seed banks, or explicitly share a primary simulation role."""
    role = settings.get('seed_role')
    if role is not None and (not isinstance(role, str) or not role):
        raise ValueError('seed_role must be a nonempty string when specified')
    fields = {'role': role} if role is not None else {}
    return [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                subject=subject, path=i, **fields)) for i in range(settings['paths_per_candidate'])]


def run(fit: Path, config: Path, output: Path) -> None:
    frozen=read(fit/'nominees.json');settings=yaml.safe_load(config.read_text())
    if output.exists():raise FileExistsError('Use a new finalist generation destination')
    if frozen['smoke'] or frozen['status']!='frozen_before_test':raise ValueError('Need frozen scientific nominees')
    for name,sha in frozen['source_sha256'].items():
        if digest(REPO/name)!=sha:raise ValueError('Training implementation changed')
    if settings['paths_per_candidate']<2 or settings['chunk_size']<1:raise ValueError('Invalid path budget')
    output.mkdir(parents=True)
    banks={(r['subject'],r['model']):r['finalists'] for r in frozen['nominees']}
    if any(len(v)!=settings['finalists'] for v in banks.values()):raise ValueError('Unequal finalist bank sizes')
    hashes=source_hashes();hashes[str(Path(__file__).resolve().relative_to(REPO))]=digest(Path(__file__))
    write(output/'protocol.json',dict(fit=str(fit.resolve()),nomination_sha256=digest(fit/'nominees.json'),
        settings=settings,source_sha256=hashes,
        data_use_status=frozen['settings'].get('data_use_status','development'),
        diagnostic_scoring_sha256=digest(REPO/'src/Bayesian_state/evaluation/model_0923_h_validation.py'),
        role='equal_generation_budget_at_all_frozen_training_finalists; nominees_remain_unchanged',
        interpretation='parameter-point sensitivity; candidate bank is not a posterior sample'))
    jobs=score_jobs(Path(frozen['data']),output,frozen['settings'],banks,'evaluate',False)
    groups={};chunks=[]
    for job in jobs:
        subject=read(Path(job['folder'])/'prepared.json')['subject']
        job.update(phase='autonomous',source_sha256=hashes,
            seeds=trajectory_seeds(settings,subject))
        groups[job['name']]=[]
        for first in range(0,len(job['seeds']),settings['chunk_size']):
            chunk=deepcopy(job);target=Path(job['target'])
            chunk.update(seeds=job['seeds'][first:first+settings['chunk_size']],name=job['name']+f'_C{first:03d}',
                target=str(target.parent/'chunks'/(target.name+f'_C{first:03d}')))
            chunks.append(chunk);groups[job['name']].append(chunk)
    execute(chunks,output,min(settings['parallel_budget'],frozen['settings']['parallel_budget']))
    records=[combine_evaluation_chunks(job,groups[job['name']],True) for job in jobs]
    write(output/'generation.json',dict(status='complete',rows=records))
    nominees={(r['subject'],r['model']):r for r in frozen['nominees']};scores=[];blocks=[]
    for job,row in zip(jobs,records):
        folder=Path(job['folder']);meta=read(folder/'prepared.json');parts=[]
        for name in ['training.npz','sealed_test.npz']:
            if digest(folder/name)!=meta['array_sha256'][name]:raise ValueError('Observed evaluation array changed')
            with np.load(folder/name) as z:parts.append((z['choices']==z['categories']).astype(float))
        observed=np.concatenate(parts);cut=meta['training_trials']
        with np.load(Path(job['target']).with_suffix('.npz')) as z:
            realized=(z['choices']==z['categories'][None,:]).astype(float);probability=z['correct_probability']
        original=nominees[(row['subject'],row['model'])]
        tag=dict(subject=row['subject'],model=row['model'],candidate=row['candidate']['id'],
            frozen_nominee=row['candidate']['id']==original['candidate']['id'],
            training_nll=row['candidate']['mean_nll'],
            training_delta_nll=row['candidate']['mean_nll']-original['candidate']['mean_nll'])
        for phase,first,last in [('training',0,cut),('test',cut,len(observed)),('whole',0,len(observed))]:
            y=block_paths(observed[None,:],first,last)[0]
            r=block_paths(realized,first,last);p=block_paths(probability,first,last)
            scores.append(dict(tag,phase=phase,paths=len(realized),trials=last-first,
                probability_mean_mae=float(abs(p.mean(axis=0)-y).mean()),**distribution_scores(r,y)))
            if phase!='whole':
                for index,start in enumerate(range(first,last,32)):
                    blocks.append(dict(tag,phase=phase,first_trial=start+1,last_trial=min(start+32,last),
                        observed=y[index],probability_mean=p[:,index].mean(),realized_mean=r[:,index].mean()))
    pd.DataFrame(scores).to_csv(output/'candidate_scores.csv',index=False)
    pd.DataFrame(blocks).to_csv(output/'candidate_blocks.csv',index=False)
    write(output/'completion.json',dict(status='complete',candidate_points=len(jobs),
        paths=len(jobs)*settings['paths_per_candidate'],nominees_changed=False))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fit','config','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();run(args.fit,args.config,args.output)


if __name__=='__main__':main()
