"""A bounded 2x2 search/retention parameter probe at one frozen NLL estimate.

Uses the existing objective worker and the archived cognitive runtime. Baseline
probabilities come from an independent precision audit; baseline generation
comes from the saved confirmation paths. Seeds are exactly paired. --smoke
uses eight trials and one job per objective before a scientific run is started.
All variants are fixed parameter overrides, with no optimization or test data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import yaml

from . import model_0923_h_objective_comparison as objective


def run(fit: Path, precision: Path, config: Path, output: Path, smoke: bool) -> None:
    v=objective.validation();frozen=v.read(fit/'nominees.json');audit=v.read(precision/'protocol.json')
    settings=yaml.safe_load(config.read_text());subject=settings['subject']
    if output.exists():raise FileExistsError('Use a new fixed-parameter probe directory')
    if frozen['smoke'] or frozen['status']!='frozen_before_test':raise ValueError('Need frozen scientific parameters')
    for name,sha in frozen['source_sha256'].items():
        if v.digest(objective.REPO/name)!=sha:raise ValueError('Fitted source changed')
    baseline=next(r['candidate'] for r in frozen['nominees'] if r['subject']==subject and r['model']==settings['reference_model'])
    old=[j for j in audit['jobs'] if j['kind']=='filter' and j['subject']==subject
         and j['candidate']['id']==baseline['id'] and j['particles']==settings['filter_particles']]
    old.sort(key=lambda j:j['repeat'])
    if len(old)!=settings['filter_repeats']:raise ValueError('Need every declared baseline filter seed')
    references={}
    for job in old:
        file=Path(job['target']).with_suffix('.json');row=v.read(file)
        if v.digest(file.with_suffix('.npz'))!=row['predictions_sha256']:raise ValueError('Baseline filter changed')
        references[str(file)]=v.digest(file);references[str(file.with_suffix('.npz'))]=row['predictions_sha256']
    generation=fit/'confirmation'/f'subject_{subject}'/'GEN'/(baseline['id']+'.npz')
    row=v.read(generation.with_suffix('.json'))
    if v.digest(generation)!=row['predictions_sha256']:raise ValueError('Baseline generation changed')
    references[str(generation)]=row['predictions_sha256']
    with np.load(generation) as saved:seeds=saved['seeds'].astype(int).tolist()
    if len(seeds)!=settings['generation_paths']:raise ValueError('Retain every baseline path')
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
    jobs=[];points={}
    variants=list(settings['variants'].items())[:1] if smoke else list(settings['variants'].items())
    for variant,overrides in variants:
        point=deepcopy(baseline['hyperparams'])
        if set(overrides)-set(point):raise ValueError('Probe only existing fitted coordinates')
        point.update(overrides);point=candidate(point,'fixed_parameter_probe');points[variant]=point
        for arm in ('NLL','GEN'):
            if arm=='NLL':chunks=[j['seeds'] for j in old[:1 if smoke else len(old)]]
            else:
                used=seeds[:4] if smoke else seeds
                chunks=[used[i:i+settings['generation_chunk']] for i in range(0,len(used),settings['generation_chunk'])]
            for index,chunk in enumerate(chunks):
                name=f'{variant}_{arm}_{index:03d}'
                job=dict(old[0],name=name,model=arm,candidate=point,kind='probe',seeds=chunk,
                    case_id=subject,particles=4 if smoke else settings['filter_particles'],smoke=smoke,
                    features=frozen['settings']['features'],objective=frozen['settings']['objectives'][arm],
                    source_sha256=objective.hashes(v),target=str((output/variant/arm/name).resolve()))
                jobs.append(job)
    output.mkdir(parents=True)
    v.write(output/'protocol.json',dict(settings=settings,fit=str(fit.resolve()),precision=str(precision.resolve()),
        baseline=baseline,baseline_filter_jobs=old,baseline_generation=str(generation.resolve()),
        reference_sha256=references,points=points,jobs=jobs,smoke=smoke,
        source_sha256=v.digest(Path(__file__)),test_choices_used=False))
    shutil.copy2(__file__,output/'workflow_source.py');shutil.copy2(config,output/'config.yaml')
    objective.execute(jobs,output,1 if smoke else settings['parallel_budget'])
    records=[v.read(Path(job['target']).with_suffix('.json')) for job in jobs]
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,jobs=len(jobs),
        cpu_seconds=sum(r['cpu_seconds'] for r in records),protocol_sha256=v.digest(output/'protocol.json')))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','precision','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run(args.fit,args.precision,args.config,args.output,args.smoke)


if __name__=='__main__':main()
