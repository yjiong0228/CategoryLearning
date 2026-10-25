"""Resume incomplete registered jobs without replacing existing artifacts.

Use --partial INTERRUPTED_FOLLOWUP --output NEW --sealed NEW. Complete jobs
are verified and retained. Incomplete jobs keep their registered parameters,
observations, particle counts and random seeds, with new artifact paths.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

from . import model_1024_state_validation_stage as stage
from . import model_1024_stage_summary as summary

read,write,digest=stage.read,stage.write,stage.digest


def finished(job: dict) -> bool:
    stems=[job['target'],job['state_capture_target'],job.get('marginal_target',job.get('genealogy_target'))]
    return all(Path(stem+'.npz').is_file() and Path(stem+'.json').is_file() for stem in stems)


def resume(partial: Path,output: Path,sealed: Path) -> None:
    if output.exists() or sealed.exists():raise FileExistsError('Use new resume and sealed-manifest directories')
    p=read(partial/'protocol.json');stage.registered(Path(p['prepared']))
    if p['worker_sources']!=stage.worker_tools.sources() or (partial/'completion.json').exists() or not p['previous']:
        raise ValueError('Require an interrupted frozen follow-up')
    expected=len(p['jobs']);complete=[];missing=[]
    for job in p['jobs']:
        if finished(job):summary.capture(job);complete.append(job)
        else:missing.append(job)
    if not missing:raise ValueError('All artifacts exist; controller completion needs a separate audit')
    output.mkdir(parents=True);jobs=[];replacement={}
    for old in missing:
        job=deepcopy(old);job['name']=old['name']+'__resume'
        for key,folder in [('target','scores'),('state_capture_target','states'),('marginal_target','captures'),('genealogy_target','captures')]:
            if key in job:job[key]=str((output/folder/job['name']).resolve())
        jobs.append(job);replacement[old['name']]=job
    write(output/'protocol.json',dict(partial=str(partial.resolve()),partial_protocol_sha256=digest(partial/'protocol.json'),
        prepared=p['prepared'],jobs=jobs,retained_names=[j['name'] for j in complete],
        interrupted_names=[j['name'] for j in missing],worker_sources=p['worker_sources'],
        interruption='SIGTERM observed as exit status 143; complete artifacts retained',
        parameters_seeds_trials_particles_unchanged=True,oral_used=False))
    started=time.monotonic();stage.dispatch(jobs,output,p['policy']['parallel_budget'])
    combined=[replacement.get(job['name'],job) for job in p['jobs']]
    for job in combined:
        if not finished(job):raise ValueError('Resumed registered bank is incomplete')
        summary.capture(job)
    original={(j['recovery_case'],j['particles'],tuple(j['seeds'])) for j in p['jobs']}
    actual={(j['recovery_case'],j['particles'],tuple(j['seeds'])) for j in combined}
    if original!=actual or len(actual)!=expected:raise ValueError('Registered system identities changed')
    write(output/'completion.json',dict(status='complete',filters=len(jobs),retained=len(complete),
        seconds=time.monotonic()-started,protocol_sha256=digest(output/'protocol.json'),source_sha256=digest(Path(__file__))))
    sealed.mkdir(parents=True);q=deepcopy(p);q['jobs']=combined
    q['resumption']=dict(partial=str(partial.resolve()),partial_protocol_sha256=digest(partial/'protocol.json'),
        resume=str(output.resolve()),resume_completion_sha256=digest(output/'completion.json'),
        complete_retained=len(complete),incomplete_retried=len(missing),registered_systems=expected)
    write(sealed/'protocol.json',q)
    write(sealed/'completion.json',dict(status='complete',filters=expected,protocol_sha256=digest(sealed/'protocol.json'),
        resumption=True,oral_used=False,all_registered_seeds_present_once=True))
    print('Interrupted registered batch completed and sealed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('partial','output','sealed'):parser.add_argument('--'+flag,type=Path,required=True)
    resume(**vars(parser.parse_args()))
