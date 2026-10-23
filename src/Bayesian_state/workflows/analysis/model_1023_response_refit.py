"""Training-only precision-grid checks and equal-budget category-response fits.

Use --config YAML --phase precision|joint --output NEW [--smoke|--prepare-only].
Precision uses the complete declared beta-init x learning-rate grid at the
previous estimate. Joint search reuses the bounded population optimizer with
identical starting populations, supports and seed banks for exponents one/two.
All numerical jobs share a 128-process queue; each uses one numeric thread.
--after COMPLETE_DIRECTORY delays preparation until that preceding queue exits.
Joint production additionally requires --decision DYNAMIC_SUMMARY/decision.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import importlib.metadata
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import yaml

from . import model_1023_estimation_reliability as reliability
from . import model_1023_response_dynamics as dynamics
from ...optimization.search.adaptive_proposals import candidate

ROOT,read,write,digest=dynamics.ROOT,dynamics.read,dynamics.write,dynamics.digest
refit,driver,proposals=reliability.refit,reliability.driver,reliability.proposals
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
ORIGINAL_SOURCES=refit.sources


def sources(v) -> dict:
    return dict(ORIGINAL_SOURCES(v),**{k:digest(ROOT/k) for k in
        (ENTRY,reliability.ENTRY,reliability.PROPOSALS,dynamics.ENTRY,dynamics.ADAPTER,
         dynamics.PARENT,dynamics.parent.HELPER,dynamics.parent.OBSERVER)})


def precision_bank(space: dict,point: dict,coordinates: list[str]) -> list[dict]:
    if coordinates!=['engine.modules.beta_mod.kwargs.beta_init','engine.modules.beta_mod.kwargs.learning_rate']:
        raise ValueError('Precision check must vary exactly initial precision and learning rate')
    proposals.validate_point(point,space)
    bank=[]
    for values in product(*(space[k] for k in coordinates)):
        changed=dict(point,**dict(zip(coordinates,values)))
        bank.append(candidate(changed,'joint_precision_grid'))
    if len({r['id'] for r in bank})!=len(bank):raise ValueError('Duplicate precision candidates')
    return bank


def prepare(policy: dict,phase: str,output: Path,smoke: bool) -> list[dict]:
    source=ROOT/policy['study'];finished=read(source/'completion.json')
    if finished['status']!='complete' or finished['phase']!='human' or finished['smoke']:
        raise ValueError('Require completed human prefix estimates')
    original=read(source/'cases.json')['cases']
    if sorted(c['subject'] for c in original)!=sorted(policy['subjects']):
        raise ValueError('Keep all six registered subjects')
    inputs={str(source/k):digest(source/k) for k in ('completion.json','cases.json')}
    cases=[]
    for old in original[:1] if smoke else original:
        frozen=reliability.checked_frozen(old)
        inputs[str(Path(old['output'])/'nominees.json')]=digest(Path(old['output'])/'nominees.json')
        chosen=frozen['nominee']['candidate']
        anchors=[chosen['hyperparams']]+[r['hyperparams'] for r in frozen['nominee']['finalists'] if r['id']!=chosen['id']]
        if smoke:anchors=anchors[:1]
        for exponent in [1.] if phase=='precision' else policy['response_exponents']:
            case_id=f"S{old['subject']}_{phase.upper()}_Q{int(exponent)}"
            settings=reliability.settings_for_case(old['settings'],policy,case_id)
            settings.update(protocol=policy['protocol']+'_'+phase,
                seed=refit.stable_seed(dict(base=policy['seed'],subject=old['subject'],phase=phase)),
                response_exponent=float(exponent))
            if phase=='precision':
                for k in ('selection_candidates','confirmation_candidates','audit_candidates'):
                    settings[k]=policy['precision_check'][k]
                starts=precision_bank(old['space'],chosen['hyperparams'],policy['precision_check']['coordinates'])
                if smoke:
                    starts=[candidate(chosen['hyperparams'],'previous_training_fit')]+[
                        r for r in starts if r['id']!=chosen['id']][:3]
            else:
                starts=proposals.initial_bank(old['space'],anchors,4 if smoke else policy['initial_candidates'],settings['seed'])
            required=next(r for r in starts if r['id']==chosen['id'])
            banks=reliability.independent_seed_banks(settings,old['subject'])
            cases.append(dict(deepcopy(old),case_id=case_id,settings=settings,starts=starts,required=required,
                output=str((output/'human_fit'/case_id).resolve()),source_fit=old['output'],
                estimation_phase=phase,independent_seed_banks=banks,response_exponent=float(exponent),
                optimizer_development_case=True))
    if phase=='joint':
        for sid in sorted({c['subject'] for c in cases}):
            a,b=[c for c in cases if c['subject']==sid]
            if a['starts']!=b['starts'] or a['space']!=b['space'] or a['independent_seed_banks']!=b['independent_seed_banks']:
                raise ValueError('Paired response fits require identical starts, supports and seed banks')
    write(output/'cases.json',dict(cases=cases,inputs_sha256=inputs,oral_used=False,human_later_choices_used=False,
                                 phase=phase,parameter_changes='precision pair only' if phase=='precision' else 'existing joint support'))
    return cases


def configure_executor(pool) -> None:
    driver.base.source_hashes=sources

    def launch(original,folder):
        settings=read(folder/'protocol.json')['settings'];job=deepcopy(original)
        if job['phase']!='training':raise ValueError('Refit accepts training behavior only')
        job['mechanism']['capacity']='evidence'
        job.update(compact_variant='TRANSFER',entry_control='PRECISION',case_id=settings['case_id'],
            response_exponent=settings['response_exponent'],dynamic_role='training',
            bindings={str(ROOT/k):v for k,v in job['source_sha256'].items()})
        directory=folder/'jobs';directory.mkdir(exist_ok=True)
        file=directory/(job['name']+'.json');write(file,job)
        launcher=('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='response_refit_') as cache,file.with_suffix('.log').open('x') as log:
            env.update(NUMBA_CACHE_DIR=cache,MPLCONFIGDIR=cache)
            subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/dynamics.ENTRY),'--worker',str(file.resolve())],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

    def execute(jobs,folder,budget):
        from .model_0923_h_training_audit import combine
        expanded,groups=refit.recovery.split_filter_jobs(jobs)
        for future in as_completed([pool.submit(launch,j,folder) for j in expanded]):future.result()
        for original,children in groups:combine(original,children)
    driver.base.execute=execute


def precision_case(case: dict,smoke: bool) -> dict:
    settings,output=case['settings'],Path(case['output']);output.mkdir(parents=True)
    hashes=sources(driver.io)
    write(output/'protocol.json',dict(settings=settings,case_id=case['case_id'],smoke=smoke,source_sha256=hashes,
        space=case['space'],phase='precision',fixed_nonprecision_parameters=case['required']['hyperparams']))
    key=case['subject'],'COMPACT'
    stage=lambda bank,name:driver.base.stage(settings,ROOT/settings['data'],output,{key:bank},name,smoke)[key]
    rows=stage(case['starts'],'screen')
    for name in ('selection','confirmation'):
        rows=stage(driver.base.shortlist(rows,case['required'],2 if smoke else settings[name+'_candidates']),name)
    nominee=dict(case_id=case['case_id'],subject=case['subject'],model='COMPACT',compact_variant='TRANSFER',
        candidate=rows[0],finalists=rows,unique_candidates=len(case['starts']),reference_id=case['required']['id'])
    write(output/'nominees.json',dict(status='frozen_before_test',settings=settings,smoke=smoke,source_sha256=hashes,
        data=str((ROOT/settings['data']).resolve()),nominee=nominee,nominees=[nominee]))
    write(output/'completion.json',dict(status='complete',smoke=smoke,test_choices_used=False,
        nomination_sha256=digest(output/'nominees.json'),bounded_rounds=0))
    print(case['case_id']+': precision estimate frozen',flush=True)
    return nominee


def run(config: Path,phase: str,output: Path,smoke: bool,prepare_only: bool,
        after: Path|None,decision: Path|None) -> None:
    if output.exists():raise FileExistsError('Use a new response fitting directory')
    if after is not None:
        print('Waiting for preceding numerical queue: '+str(after),flush=True)
        while not (after/'completion.json').exists():time.sleep(15)
        if read(after/'completion.json')['status']!='complete':raise ValueError('Preceding stage incomplete')
    policy=yaml.safe_load(config.read_text())
    if policy['response_exponents']!=[1.,2.] or not 1<=policy['parallel_budget']<=128:
        raise ValueError('Keep the registered response arms and worker budget')
    if phase=='joint' and not (smoke or prepare_only):
        if decision is None or not read(decision)['proceed_to_equal_budget_refit']:
            raise ValueError('Joint production requires the registered dynamic evidence gate')
    output.mkdir(parents=True)
    refit.sources=sources
    driver.base.archive(output,config,sources(driver.io))
    cases=prepare(policy,phase,output,smoke)
    workers=1 if smoke else min(policy['parallel_budget'],len(os.sched_getaffinity(0)))
    write(output/'protocol.json',dict(settings=policy,phase=phase,smoke=smoke,workers=workers,
        source_sha256=sources(driver.io),python=sys.version,
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','PyYAML')},
        objective='Original training choice likelihood; oral and later responses excluded',
        response_mechanism='q_k proportional to p_k ** exponent; raw rule support retained',
        preceding_queue=str(after) if after else None,
        decision_sha256=digest(decision) if decision else None,starting_time=time.time()))
    if prepare_only:
        write(output/'preparation_complete.json',dict(status='complete',cases=len(cases),numeric_jobs=0));return
    with reliability.NumericQueue(workers,output) as pool:
        configure_executor(pool)
        with ThreadPoolExecutor(max_workers=1 if smoke else len(cases)) as controls:
            fn=precision_case if phase=='precision' else reliability.fit_case
            nominees=[f.result() for f in [controls.submit(fn,c,smoke) for c in cases]]
        write(output/'selected_nominees.json',dict(status='frozen_before_external_evaluation',smoke=smoke,nominees=nominees))
        audit=output/'audit_fit';audit.mkdir()
        with ThreadPoolExecutor(max_workers=1 if smoke else len(cases)) as controls:
            audited=[f.result() for f in [controls.submit(reliability.audit_case,c,audit/c['case_id'],smoke) for c in cases]]
        write(audit/'cases.json',dict(cases=[r[0] for r in audited]))
        write(audit/'completion.json',dict(status='complete',smoke=smoke,selected_before_audit=True))
    write(output/'completion.json',dict(status='complete',phase=phase,smoke=smoke,cases=len(cases),
        human_evaluation='pending',oral_used_for_fitting=False,selected_before_audit=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('precision','joint'),required=True)
    parser.add_argument('--after',type=Path);parser.add_argument('--decision',type=Path)
    group=parser.add_mutually_exclusive_group();group.add_argument('--smoke',action='store_true');group.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();run(args.config,args.phase,args.output,args.smoke,args.prepare_only,args.after,args.decision)
