"""Evaluate frozen retention fits in one shared CPU queue.

Conditional filters and task-start autonomous paths are independent after
parameter freezing. They share one 128-worker queue so shorter generation
tasks can occupy slots while long filters finish. Existing scorers and chunk
pooling are reused without changing prediction timing or model behavior.
The training confirmation's subject particle overrides also apply here.
Outputs are compatible with the existing prefix-continuation workflow.
Before evaluation workers start, a supplementary reference chooses the best
confirmation training NLL among each participant's three nominees. This pools
three search budgets and reuses predictions from the chosen original nominee.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from . import model_0923_h_validation as v
from ...utils.seeding import stable_seed


def pooled_training_reference(frozen: dict) -> list[dict]:
    """Fix a supplementary FREE reference using all three training searches."""
    space,_=v.model_space(frozen['settings'],'FREE')
    order={'FREE':0,'CUMULATIVE':1,'SHARED':2};groups={}
    for row in frozen['nominees']:groups.setdefault(row['subject'],[]).append(row)
    selected=[]
    for subject,rows in sorted(groups.items()):
        if {r['model'] for r in rows}!=set(order):raise ValueError('Need all three fitted policies')
        row=min(rows,key=lambda r:(r['candidate']['mean_nll'],order[r['model']]))
        point=deepcopy(row['candidate']['hyperparams'])
        if row['model']=='SHARED':
            point[v.H_PATH+'intervention.decay']=point['engine.modules.memory_mod.kwargs.gamma']
        if set(point)!=set(space) or any(point[k] not in values for k,values in space.items()):
            raise ValueError('Pooled reference is outside FREE support')
        selected.append(dict(subject=subject,source_model=row['model'],candidate=row['candidate']['id'],
            training_nll=row['candidate']['mean_nll'],equivalent_free_hyperparams=point))
    return selected


def plan(frozen: dict, output: Path) -> tuple[dict,list[dict]]:
    settings=frozen['settings'];banks={(r['subject'],r['model']):[r['candidate']] for r in frozen['nominees']}
    # JSON stores the fitted protocol's subject-ID keys as strings.
    overrides={int(subject):particles for subject,particles in settings['confirmation_particle_overrides'].items()}
    groups={};chunks=[]
    for role in ('conditional','autonomous'):
        jobs=v.score_jobs(Path(frozen['data']),output/role,settings,banks,'evaluate',False)
        groups[role]=[]
        for job in jobs:
            meta=v.read(Path(job['folder'])/'prepared.json');subject=meta['subject']
            job['source_sha256'][str(Path(__file__).resolve().relative_to(v.REPO))]=v.digest(Path(__file__))
            job['particles']=overrides.get(subject,job['particles'])
            if role=='autonomous':
                job['phase']='autonomous';job['name']='AUTO_'+job['name']
                job['seeds']=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],
                    role='autonomous',subject=subject,path=i)) for i in range(settings['autonomous_paths'])]
            width=settings['autonomous_chunk'] if role=='autonomous' else settings['evaluate_seed_chunk']
            own=[]
            for start in range(0,len(job['seeds']),width):
                chunk=deepcopy(job);target=Path(job['target']);suffix=f'_C{start:03d}'
                chunk.update(seeds=job['seeds'][start:start+width],name=job['name']+suffix,
                    target=str(target.parent/'chunks'/(target.name+suffix)))
                own.append(chunk);chunks.append(chunk)
            groups[role].append((job,own))
    lengths={j['folder']:sum(v.read(Path(j['folder'])/'prepared.json')[k] for k in ('training_trials','test_trials')) for j in chunks}
    chunks.sort(key=lambda j:-lengths[j['folder']]*len(j['seeds'])*(1 if j['phase']=='autonomous' else j['particles']))
    return groups,chunks


def run(fit: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new combined evaluation directory')
    frozen=v.read(fit/'nominees.json')
    if frozen['smoke'] or frozen['status']!='frozen_before_test':raise ValueError('Need scientific frozen fits')
    for name,sha in frozen['source_sha256'].items():
        if v.digest(v.REPO/name)!=sha:raise ValueError('Frozen fitting implementation changed')
    pooled=pooled_training_reference(frozen)
    groups,chunks=plan(frozen,output);output.mkdir(parents=True)
    v.write(output/'training_pool_reference.json',dict(status='frozen_before_test',
        nomination_sha256=v.digest(fit/'nominees.json'),source_sha256=v.digest(Path(__file__)),
        rows=pooled,budget='All three training searches pooled; predictions reuse the selected original nominee.',
        selection='Minimum confirmation training NLL; exact ties prefer FREE, then CUMULATIVE, then SHARED.'))
    for role in groups:
        folder=output/role;folder.mkdir()
        v.write(folder/'evaluation_protocol.json',dict(nomination_sha256=v.digest(fit/'nominees.json'),
            fit=str(fit.resolve()),source_sha256=v.digest(Path(__file__)),
            test_opening='after_all_parameters_frozen',shared_queue_workers=frozen['settings']['parallel_budget'],
            subject_particle_overrides=frozen['settings']['confirmation_particle_overrides'],
            data_use_status='development and new-validation subjects reported separately'))
    v.write(output/'protocol.json',dict(nomination_sha256=v.digest(fit/'nominees.json'),
        source_sha256=v.digest(Path(__file__)),jobs=chunks))
    v.execute(chunks,output,frozen['settings']['parallel_budget'])
    for role,jobs in groups.items():
        rows=[v.combine_evaluation_chunks(job,parts,role=='autonomous') for job,parts in jobs]
        v.write(output/role/'evaluation.json',dict(status='complete',rows=rows,
            role='autonomous_from_start_prefix_fitted' if role=='autonomous' else 'sequential_heldout_prechoice_predictions'))
    v.write(output/'completion.json',dict(status='complete',jobs=len(chunks),
        nomination_sha256=v.digest(fit/'nominees.json')))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.fit,a.output)


if __name__=='__main__':main()
