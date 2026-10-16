"""Refit three retention-time assumptions with matched numerical budgets.

FREE estimates evidence decay; CUMULATIVE fixes it to one; SHARED reads the
existing memory gamma. All cognitive operations use the unchanged shared H
module and archived runtime. Initial proposals share their 13 common
coordinates. The old NLL/GEN nominees seed development participants only.
Training never opens sealed_test.npz. Confirmation uses independent seeds
before freezing nominees. --smoke uses eight trials and one worker.

Use --phase search --config YAML --data PREPARED --output NEW, followed by
--phase evaluate or autonomous --fit NEW --output ANOTHER_NEW. Evaluation
reuses the established complete-seed/complete-path pooling and data boundary.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

from . import model_0923_h_validation as v
from .model_0923_h_training_audit import combine
from ...optimization.search.adaptive_proposals import initial_points, candidate, propose_round, select_elites
from ...evaluation.model_0923_h_validation import jackknife_scores, jackknife_se
from ...utils.seeding import stable_seed

DECAY=v.H_PATH+'intervention.decay'
MEMORY='engine.modules.memory_mod.kwargs.gamma'


def model_space(settings: dict, model: str) -> tuple[dict,dict]:
    space,anchor=v.model_space(settings,model)
    if model=='CUMULATIVE':
        space[DECAY]=[1.];anchor[DECAY]=1.
    return space,anchor


def project(point: dict, model: str, settings: dict) -> dict:
    result=deepcopy(point)
    if model=='CUMULATIVE':result[DECAY]=1.
    elif model=='SHARED':result.pop(DECAY,None)
    elif DECAY not in result:result[DECAY]=settings['models']['FREE']['decay']
    space,_=model_space(settings,model)
    if set(result)!=set(space) or any(result[k] not in values for k,values in space.items()):
        raise ValueError('Projected point is outside declared support')
    return result


def starting_banks(settings: dict, subjects: list[int], smoke: bool) -> tuple[dict,dict]:
    frozen=v.read(v.REPO/settings['warm_start_fit']/'nominees.json')
    for name,sha in frozen['source_sha256'].items():
        if v.digest(v.REPO/name)!=sha:raise ValueError('Warm-start model source changed')
    count=5 if smoke else settings['initial_candidates'];banks={};warm={}
    base,anchor=model_space(settings,'SHARED')
    for subject in subjects:
        points=initial_points(base,anchor,count+8,
            stable_seed(dict(base=settings['seed'],subject=subject,role='shared_starts')),
            cover_key=v.H_PATH+'max_capacity')
        prior=[r['candidate']['hyperparams'] for r in frozen['nominees']
               if subject in settings['development_subjects'] and r['subject']==subject]
        for model in settings['models']:
            rows={}
            for point in prior:
                row=candidate(project(point,model,settings),'prior_training_nominee');rows[row['id']]=row
            warm[subject,model]=list(rows.values())
            for i,point in enumerate(points):
                p=deepcopy(point)
                if model=='FREE':p[DECAY]=settings['evidence_decay_values'][i%len(settings['evidence_decay_values'])]
                row=candidate(project(p,model,settings),'matched_common_coordinate_start')
                rows.setdefault(row['id'],row)
                if len(rows)==count:break
            if len(rows)!=count:raise ValueError('Initial budget could not be filled')
            banks[subject,model]=list(rows.values())
    return banks,warm


def stage(settings: dict, data: Path, output: Path, banks: dict, name: str, smoke: bool) -> dict:
    """Split full filter repetitions, then pool before the nonlinear score."""
    jobs=v.score_jobs(data,output,settings,banks,name,smoke);chunks=[];groups={}
    for job in jobs:
        subject=v.read(Path(job['folder'])/'prepared.json')['subject']
        if name=='confirmation' and not smoke:
            job['particles']=settings['confirmation_particle_overrides'].get(subject,job['particles'])
        job['source_sha256'][str(Path(__file__).resolve().relative_to(v.REPO))]=v.digest(Path(__file__))
        groups[job['name']]=[]
        # Multiple particles remain inside one process. Each complete filter
        # seed is an independent task, enabling long finalists to share CPUs.
        width=len(job['seeds']) if smoke or name.startswith(('screen','refine')) else settings['seed_chunk']
        for first in range(0,len(job['seeds']),width):
            chunk=deepcopy(job);target=Path(job['target']);suffix=f'_C{first:03d}'
            chunk.update(name=job['name']+suffix,seeds=job['seeds'][first:first+width],
                target=str(target.parent/'chunks'/(target.name+suffix)))
            groups[job['name']].append(chunk);chunks.append(chunk)
    lengths={j['folder']:v.read(Path(j['folder'])/'prepared.json')['training_trials'] for j in jobs}
    chunks.sort(key=lambda j:-lengths[j['folder']]*j['particles']*len(j['seeds']))
    started=time.monotonic();v.execute(chunks,output,1 if smoke else settings['parallel_budget'])
    scored={key:[] for key in banks}
    for job in jobs:
        row=combine(job,groups[job['name']]);subject=v.read(Path(job['folder'])/'prepared.json')['subject']
        scored[subject,job['model']].append(row)
    for rows in scored.values():rows.sort(key=lambda r:(r['mean_nll'],r['id']))
    v.write(output/(name+'_completion.json'),dict(status='complete',stage=name,points=len(jobs),
        seed_jobs=len(chunks),seconds=time.monotonic()-started,
        groups=[dict(subject=k[0],model=k[1],rows=rows) for k,rows in scored.items()]))
    return scored


def shortlist(rows: list[dict], required: list[dict], count: int) -> list[dict]:
    chosen={r['id']:r for r in required}
    for row in rows:
        if len(chosen)>=count:break
        chosen.setdefault(row['id'],row)
    if len(chosen)!=count:raise ValueError('Not enough distinct finalists')
    return list(chosen.values())


def validate_settings(settings: dict) -> None:
    expected={'FREE':.95,'CUMULATIVE':1.,'SHARED':'memory'}
    if set(settings['models'])!=set(expected):raise ValueError('Need three declared retention policies')
    for model,decay in expected.items():
        if settings['models'][model]!=dict(kind='E',decay=decay,capacity='evidence_mean',search='accumulate'):
            raise ValueError('Other H operations must be identical')
    if set(settings['development_subjects']) & set(settings['new_validation_subjects']):
        raise ValueError('Development and validation subjects must be disjoint')
    if settings['selection_candidates']<2 or settings['confirmation_candidates']<2:
        raise ValueError('Retain multiple candidates for numerical comparison')


def search(config: Path, data: Path, output: Path, smoke: bool) -> None:
    settings=yaml.safe_load(config.read_text());validate_settings(settings)
    if output.exists():raise FileExistsError('Use a new retention-refit output')
    prep=v.read(data/'preparation_manifest.json')
    subjects=[s for s in prep['subjects'] if (data/f'subject_{s}'/'prepared.json').exists()]
    if smoke:subjects=subjects[:1]
    assert subjects and set(subjects)<=set(settings['development_subjects']+settings['new_validation_subjects'])
    banks,warm=starting_banks(settings,subjects,smoke);seen={k:{r['id'] for r in rows} for k,rows in banks.items()}
    output.mkdir(parents=True);hashes=v.source_hashes()
    hashes[str(Path(__file__).resolve().relative_to(v.REPO))]=v.digest(Path(__file__))
    v.write(output/'search_protocol.json',dict(settings=settings,smoke=smoke,data=str(data.resolve()),
        subjects=subjects,source_sha256=hashes,data_manifest_sha256=v.digest(data/'preparation_manifest.json'),
        warm_start_nomination_sha256=v.digest(v.REPO/settings['warm_start_fit']/'nominees.json'),
        fitted_coordinates={m:sum(len(x)>1 for x in model_space(settings,m)[0].values()) for m in settings['models']},
        initial_banks=[dict(subject=k[0],model=k[1],points=rows) for k,rows in banks.items()],
        boundary='training only; freeze all nominees before reading held-out choices'))
    shutil.copy2(__file__,output/'workflow_source.py');shutil.copy2(config,output/'config.yaml')
    rows=stage(settings,data,output,banks,'screen',smoke)
    for i in range(1 if smoke else settings['refinement_rounds']):
        name=f'refine{i+1}';settings['budgets'][name]=settings['budgets']['refine']
        for key in banks:
            elites=select_elites(rows[key],1 if smoke else settings['elites'],exclude_key=v.H_PATH+'max_capacity')
            proposals=propose_round(elites,model_space(settings,key[1])[0],seen[key],
                1 if smoke else settings['proposals_per_elite'],
                stable_seed(dict(base=settings['seed'],subject=key[0],role=name)),.5,.25)
            assert len(proposals)==len(elites)*(1 if smoke else settings['proposals_per_elite'])
            banks[key]=proposals+elites;seen[key].update(r['id'] for r in proposals)
        rows=stage(settings,data,output,banks,name,smoke)
    banks={k:shortlist(r,warm[k][:1 if smoke else 2],2 if smoke else settings['selection_candidates']) for k,r in rows.items()}
    rows=stage(settings,data,output,banks,'selection',smoke)
    banks={k:r[:2 if smoke else settings['confirmation_candidates']] for k,r in rows.items()}
    rows=stage(settings,data,output,banks,'confirmation',smoke)
    nominees=[];numerical=[]
    for key,finalists in rows.items():
        p=[];seeds=[]
        for row in finalists:
            with np.load(Path(row['score_file']).with_suffix('.npz')) as a:p.append(a['observed_probability']);seeds.append(a['seeds'])
        for other in seeds[1:]:np.testing.assert_array_equal(seeds[0],other)
        for row,values in zip(finalists,p):
            numerical.append(dict(subject=key[0],model=key[1],candidate=row['id'],
                delta_nll=v.pooled_nll(values)-v.pooled_nll(p[0]),
                paired_mc_se=jackknife_se(jackknife_scores(values)-jackknife_scores(p[0]))))
        nominees.append(dict(subject=key[0],model=key[1],candidate=finalists[0],finalists=finalists,
            unique_candidates=len(seen[key]),scientific_status='bounded_refit_training_confirmation'))
    v.write(output/'nominees.json',dict(status='frozen_before_test',settings=settings,smoke=smoke,
        data=str(data.resolve()),source_sha256=hashes,nominees=nominees,numerical_finalists=numerical))
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,participants=len(subjects),
        models=len(settings['models']),test_choices_used=False,nomination_sha256=v.digest(output/'nominees.json')))
    print('All retention-policy parameters frozen before test evaluation',flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['search','evaluate','autonomous'],required=True)
    for name in ('config','data','fit','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.output is None:p.error('--output is required')
    if a.phase=='search':
        if not a.config or not a.data:p.error('--config and --data are required')
        search(a.config,a.data,a.output,a.smoke)
    else:
        if not a.fit or a.smoke:p.error('Evaluation needs --fit and a scientific frozen run')
        v.evaluate(a.fit,a.output,a.phase=='autonomous')


if __name__=='__main__':main()
