"""One bounded precision extension for unresolved particle-count comparisons.

Register after the primary and shape-extension results are complete. Every
finalist whose 512-versus-1024 check remains unresolved receives four more
independent 1024-particle systems. Parameters, 512-particle comparisons, shape
checks and thresholds remain fixed. All work uses the original CPU ledger.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import shutil
import time

from .model_1024_individual_regions import (
    BudgetQueue, classify, completed_arrays, decision, digest, make_jobs,
    read, sources, write,
)
from ...metrics.trajectory_shape import sequence_score


def select_targets(rows: dict) -> list[dict]:
    """Apply the numerical rule regardless of shape or likelihood ranking."""
    return [dict(subject=sid,candidate=r['candidate']['id'])
            for sid,batch in rows.items() for r in batch
            if r.get('particle_check_score') and r.get('particle_check',{}).get('status')!='stable']


def augment_high_checks(row: dict, extra_jobs: list[dict]) -> bool:
    """Pool complete independent likelihood estimates, never averaged NLLs."""
    jobs=row['particle_check_jobs']+extra_jobs
    probability=completed_arrays(jobs,'observed_probability')
    if probability is None:
        return False
    row['primary_particle_check_score']=deepcopy(row['particle_check_score'])
    row['particle_check_jobs']=jobs
    row['particle_check_score']=sequence_score(probability)
    return True


def register(study: Path, ledger: Path, output: Path, job_role: str = 'particle_precision',
             parallel_budget: int | None = None) -> None:
    if output.exists():raise FileExistsError('Use a new extension directory')
    protocol=read(study/'protocol.json');read(study/'completion.json')
    inputs=[study/'protocol.json',study/'completion.json']
    inputs += [study/f'confirmed_S{c["subject"]}.json' for c in protocol['cases']]
    rows={c['subject']:read(study/f'confirmed_S{c["subject"]}.json') for c in protocol['cases']}
    targets=select_targets(rows)
    if not targets:raise ValueError('No unresolved particle-count checks')
    particles=protocol['policy']['search']['particle_check_particles']
    workers=protocol['policy']['parallel_budget'] if parallel_budget is None else parallel_budget
    if not re.fullmatch('[a-z][a-z0-9_]*',job_role) or not 1<=workers<=protocol['policy']['parallel_budget']:
        raise ValueError('Require a safe new job role and a bounded positive worker count')
    caps=dict(calibration=4,search=12,confirmation=64,generation=12,analysis=4,total=96)
    if caps['total']!=protocol['policy']['cpu_hours']['total']:
        raise ValueError('Extension must retain the registered total budget')
    registration=dict(study=str(study.resolve()),ledger=str(ledger.resolve()),targets=targets,
                      extra_repeats=4,particles=particles,cpu_hours=caps,
                      job_role=job_role,parallel_budget=workers,
                      input_sha256={str(f.resolve()):digest(f) for f in inputs},
                      numerical_sources=sources(),extension_source_sha256=digest(Path(__file__)),
                      registered_unix=time.time(),
                      rule='Every completed finalist with a non-stable particle-count check; one batch only; retain all criteria.',
                      reason='Two high-particle systems leave substantial Monte Carlo uncertainty; use unused calibration/search CPU.')
    output.mkdir(parents=True);write(output/'registration.json',registration)
    shutil.copy2(Path(__file__),output/'extension_source.py')
    decision(ledger,'particle_precision_registered',targets=targets,extra_repeats=4,particles=particles,cpu_hours=caps,
             job_role=job_role,parallel_budget=workers)


def run(output: Path) -> None:
    registration=read(output/'registration.json')
    if (output/'completion.json').exists():raise FileExistsError('Extension already complete')
    if digest(Path(__file__))!=registration['extension_source_sha256'] or sources()!=registration['numerical_sources']:
        raise ValueError('Registered code changed')
    for file,sha in registration['input_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Registered input changed: '+file)
    study=Path(registration['study']);ledger=Path(registration['ledger'])
    protocol=read(study/'protocol.json');completion=read(study/'completion.json')
    protocol['policy']['cpu_hours']=registration['cpu_hours']
    protocol['policy']['parallel_budget']=registration.get('parallel_budget',protocol['policy']['parallel_budget'])
    protocol['particle_precision_extension']=registration
    cases={c['subject']:c for c in protocol['cases']}
    rows={sid:read(study/f'confirmed_S{sid}.json') for sid in cases}
    jobs=[]
    for target in registration['targets']:
        row=next(r for r in rows[target['subject']] if r['candidate']['id']==target['candidate'])
        jobs.extend(make_jobs(protocol,cases[target['subject']],[row['candidate']],ledger,
                              registration.get('job_role','particle_precision'),
                              particles=registration['particles'],repeats=registration['extra_repeats']))
    queue=BudgetQueue(ledger,protocol);queue.execute(jobs,'confirmation')
    outcomes=[]
    for target in registration['targets']:
        sid=target['subject'];row=next(r for r in rows[sid] if r['candidate']['id']==target['candidate'])
        current=[j for j in jobs if j['subject']==sid and j['candidate']['id']==target['candidate']]
        integrated=augment_high_checks(row,current)
        outcomes.append(dict(**target,integrated=integrated))
    for sid,batch in rows.items():
        write(output/f'confirmed_S{sid}.json',classify(batch,protocol['policy']))
        (output/f'search_S{sid}.json').symlink_to((study/f'search_S{sid}.json').resolve())
    for name in ['calibration.json','jobs','numerical','decisions.jsonl','registered_config.yaml',
                 'source','reuse_manifest.json','preflight_validation.json']:
        parent=ledger if name in ['jobs','numerical','decisions.jsonl'] else study
        (output/name).symlink_to((parent/name).resolve())
    write(output/'protocol.json',protocol)
    updated=deepcopy(completion)
    updated.update(status='bounded_pilot_with_precision_extensions_complete',
                   cpu_hours={k:v/3600 for k,v in queue.used().items()},particle_precision_outcomes=outcomes,
                   protocol_sha256=digest(output/'protocol.json'),prior_completion_sha256=digest(study/'completion.json'))
    write(output/'completion.json',updated)
    decision(ledger,'particle_precision_finished',outcomes=outcomes,cpu_hours=updated['cpu_hours'])
    write(ledger/'final_extension_handoff.json',dict(extension=str(output.resolve()),completion_sha256=digest(output/'completion.json')))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['register','run'],required=True)
    parser.add_argument('--study',type=Path);parser.add_argument('--ledger',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job-role',default='particle_precision',help='Use a new role only for an audited interrupted restart')
    parser.add_argument('--parallel-budget',type=int,help='Optional lower concurrency cap for the extension')
    args=parser.parse_args()
    if args.phase=='register':register(args.study,args.ledger,args.output,args.job_role,args.parallel_budget)
    else:run(args.output)


if __name__=='__main__':main()
