"""Rescore both generating retention values on the same synthetic sequences.

This numerical audit measures discrimination between two specified values with
all nuisance parameters fixed at their known generating values. It preserves the
original eight-point recovery results and generates no replacement datasets.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import REPO,H_PATH,read,write,digest,execute
from .model_0923_h_training_audit import combine
from ...evaluation.model_0923_h_validation import jackknife_scores,jackknife_se
from ...utils.seeding import stable_seed


def paired_contrast(case: str, low: dict, high: dict, arrays: list[tuple]) -> dict:
    """Compare common-seed likelihoods; an exact tie remains unresolved."""
    for key in ['subject','true_retention','trials']:
        if low[key]!=high[key]:raise ValueError('Retention pair uses different synthetic data')
    np.testing.assert_array_equal(arrays[0][0],arrays[1][0])
    difference=high['mean_nll']-low['mean_nll']
    se=jackknife_se(jackknife_scores(arrays[1][1])-jackknife_scores(arrays[0][1]))
    estimate=low['candidate_retention'] if difference>=0 else high['candidate_retention']
    return dict(case=case,subject=low['subject'],true_retention=low['true_retention'],trials=low['trials'],
        low_retention=low['candidate_retention'],high_retention=high['candidate_retention'],
        nll_low=low['mean_nll'],nll_high=high['mean_nll'],difference_high_minus_low=difference,mc_se=se,
        total_difference=difference*low['trials'],total_mc_se=se*low['trials'],selected_retention=estimate,
        correct=difference!=0. and estimate==low['true_retention'],exact_tie=difference==0.,
        difference_exceeds_two_mc_se=abs(difference)>2*se)


def run(source: Path, config: Path, output: Path, smoke: bool) -> None:
    if output.exists():raise FileExistsError('Use a new retention pair-audit destination')
    if read(source/'completion.json')['status']!='complete':raise ValueError('Recovery is incomplete')
    settings=yaml.safe_load(config.read_text());support=settings['retention_values']
    if len(support)!=2 or support[0]>=support[1]:raise ValueError('Need two ordered retention values')
    original=[]
    for file in sorted((source/'jobs').glob('*.json')):
        job=read(file)
        if job['phase']=='training' and job['candidate']['hyperparams'][H_PATH+'intervention.decay'] in support:
            original.append(job)
    if not original:raise ValueError('No matching synthetic recovery candidates')
    if smoke:original=[j for j in original if j['folder']==original[0]['folder']]
    groups={};jobs=[];chunks=[];output.mkdir(parents=True)
    for before in original:
        for name,sha in before['source_sha256'].items():
            if digest(REPO/name)!=sha:raise ValueError('Recovery implementation changed')
        job=deepcopy(before);case=Path(job['folder']).name
        job.update(name='pair_'+job['name'],particles=2 if smoke else settings['particles'],
            seeds=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],case=case,repeat=i))
                   for i in range(2 if smoke else settings['seeds'])],
            target=str((output/'scores'/case/job['candidate']['id']).resolve()))
        job['source_sha256'][str(Path(__file__).resolve().relative_to(REPO))]=digest(Path(__file__))
        jobs.append(job);groups[job['name']]=[]
        for index,seed in enumerate(job['seeds']):
            chunk=deepcopy(job);target=Path(job['target'])
            chunk.update(name=job['name']+f'_C{index:03d}',seeds=[seed],
                target=str(target.parent/'chunks'/(target.name+f'_C{index:03d}')))
            groups[job['name']].append(chunk);chunks.append(chunk)
    counts={folder:sum(j['folder']==folder for j in jobs) for folder in {j['folder'] for j in jobs}}
    if not counts or any(n!=2 for n in counts.values()):raise ValueError('Each synthetic sequence needs both candidates')
    write(output/'protocol.json',dict(source=str(source.resolve()),source_protocol_sha256=digest(source/'protocol.json'),
        source_completion_sha256=digest(source/'completion.json'),settings=settings,smoke=smoke,
        source_sha256=digest(Path(__file__)),role='two-generating-value discrimination on unchanged synthetic data'))
    lengths={folder:read(Path(folder)/'prepared.json')['training_trials'] for folder in counts}
    chunks.sort(key=lambda j:-lengths[j['folder']])
    execute(chunks,output,1 if smoke else settings['parallel_budget'])
    rows=[]
    for job in jobs:
        candidate=combine(job,groups[job['name']]);meta=read(Path(job['folder'])/'prepared.json')
        rows.append(dict(case=Path(job['folder']).name,subject=meta['subject'],true_retention=meta['synthetic_truth']['decay'],
            candidate_retention=job['candidate']['hyperparams'][H_PATH+'intervention.decay'],
            mean_nll=candidate['mean_nll'],trials=meta['training_trials'],score_file=candidate['score_file']))
    scores=pd.DataFrame(rows);contrasts=[]
    for case,group in scores.groupby('case',sort=True):
        low,high=group.sort_values('candidate_retention').to_dict('records');arrays=[]
        for row in [low,high]:
            with np.load(Path(row['score_file']).with_suffix('.npz')) as z:arrays.append((z['seeds'],z['observed_probability']))
        contrasts.append(paired_contrast(case,low,high,arrays))
    scores.to_csv(output/'scores.csv',index=False);pd.DataFrame(contrasts).to_csv(output/'contrasts.csv',index=False)
    write(output/'completion.json',dict(status='complete',smoke=smoke,sequences=len(contrasts),candidate_evaluations=len(jobs),
        seed_jobs=len(chunks),new_synthetic_sequences=0))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args();run(args.source,args.config,args.output,args.smoke)


if __name__=='__main__':main()
