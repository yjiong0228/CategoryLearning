"""Zero-inertia execution marginalization with the shared cognitive runtime.

Use --config YAML --output NEW [--smoke]. Workers use the existing MIX
likelihood as a numerical carrier for STICKY with kappa=0. Latent execution
posteriors are choice-conditioned explicitly. Oral validation is deferred.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
CAPTURE='src/Bayesian_state/workflows/analysis/model_1023_execution_capture.py'
PF='src/Bayesian_state/inference/backends/particle_filter.py'
HELPER='src/Bayesian_state/evaluation/marginal_execution.py'
ANCESTRY='src/Bayesian_state/evaluation/particle_smoothing.py'


def load(name: str,relative: str):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


capture=load('marginal_execution_capture',CAPTURE);study=capture.study
read,write,digest=study.read,study.write,study.digest
ancestry=load('src.Bayesian_state.evaluation.particle_smoothing',ANCESTRY)
helper=load('src.Bayesian_state.evaluation._marginal_execution_current',HELPER)


def sources() -> dict:
    return dict(study.sources(study.driver.io),**capture.sources(),
        **{p:digest(ROOT/p) for p in (ENTRY,HELPER,ANCESTRY)})


def validate_carrier(kwargs: dict) -> None:
    """Restrict the numerical equivalence to the registered shared modules."""
    engine=kwargs['engine_config'];modules=engine['modules']
    h=modules['hypo_transitions_mod']['kwargs'];beta=modules['beta_mod']['kwargs']
    if (h.get('execution_inertia')!=0 or h.get('exit_rate')!=0
            or h.get('mechanism_version')!='0923-full-v3'
            or beta.get('update_scope','active_hypotheses')!='active_hypotheses'
            or 'mapping_mod' in modules
            or kwargs['choice_readout_power']!=1
            or any(kwargs.get(k,0)!=0 for k in ('output_lapse','strategy_confidence_gain','rule_commitment_confidence_gain'))):
        raise ValueError('Require zero inertia, shared belief workspace, all-rule learning and unmodified readout')


def worker(file: Path) -> None:
    job=read(file)
    if (job['marginal_execution_sources']!=sources() or job['candidate']['hyperparams'][study.KAPPA]!=0
            or job.get('capture_oral') or job['execution_mode']!='MIX'):
        raise ValueError('Require unchanged registered zero-inertia numerical carrier')
    if job['kind']=='bare':
        study.worker(file);return
    original_load=capture.study.base.load;audit={}
    def instrument(name,path):
        module=original_load(name,path)
        if path==PF:
            original_filter=module.run_state_model_particle_filter
            def observe(**kwargs):
                validate_carrier(kwargs);existing=kwargs.pop('state_observer')
                observer=helper.MarginalExecutionCollector(np.asarray(kwargs['choices'],int),job['target_rules'])
                def both(t,phase,snapshot):
                    existing(t,phase,snapshot);observer(t,phase,snapshot)
                result=original_filter(**kwargs,state_observer=both)
                arrays=observer.arrays();weights=arrays.pop('filtered_weights')
                predictive=arrays.pop('predictive_target_execution')
                parents=ancestry.resampling_parents(result.resampling_log,len(weights),job['particles'])
                summary=helper.marginal_smoothing(arrays,parents,weights,module._trace_ancestral_indices,
                    min(job['prefix_trials'],len(weights)),job['fixed_lag'])
                target=Path(job['marginal_target']);target.parent.mkdir(parents=True,exist_ok=True)
                with target.with_suffix('.npz').open('xb') as stream:
                    np.savez_compressed(stream,**{'particle_'+k:v for k,v in arrays.items()},
                        parents=parents,filtered_weights=weights,predictive_target_execution=predictive,**summary)
                audit.update(trials=len(weights),particles=job['particles'],array_sha256=digest(target.with_suffix('.npz')),
                    minimum_particle_choice_mass=observer.minimum_choice_mass,
                    resampling_steps=int(np.count_nonzero(result.resampled)),last_trial_resampled=bool(result.resampled[-1]))
                return result
            module.run_state_model_particle_filter=observe
        return module
    capture.study.base.load=instrument
    try:capture.worker(file)
    finally:capture.study.base.load=original_load
    if not audit or job['marginal_execution_sources']!=sources():
        raise ValueError('Incomplete or changing marginalized execution capture')
    with np.load(job['marginal_target']+'.npz') as z,np.load(job['state_capture_target']+'.npz') as c:
        for key,old in [('present','target_present'),('high_belief','target_high_belief'),('target_belief','target_belief')]:
            np.testing.assert_allclose(z['filtered_'+key],c['filtered_'+old],atol=2e-12,rtol=0)
        np.testing.assert_allclose(z['filtered_source'],c['conditional_decision_sources'],atol=2e-12,rtol=0)
        np.testing.assert_allclose(z['filtered_target_execution'],z['filtered_source'][:,2],atol=2e-12,rtol=0)
    write(Path(job['marginal_target']).with_suffix('.json'),dict(status='complete',**audit,
        case_id=job['recovery_case'],subject=job['case_id'],seeds=job['seeds'],sources=sources(),
        scientific_model='STICKY with zero execution inertia',numerical_carrier='MIX shared likelihood',
        terminal_timing='after terminal choice, before terminal feedback and resampling',
        execution_timing='predictive prior or current-choice-conditioned execution posterior',
        state_capture_sha256=digest(Path(job['state_capture_target']).with_suffix('.npz')),
        passive_original_score_replay=bool(job.get('original_score')),oral_used=False))


def run(config: Path,output: Path,smoke: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import os
    import yaml
    from . import model_1023_execution_validation as validation
    if output.exists():raise FileExistsError('Use a new marginal execution output directory')
    p=yaml.safe_load(config.read_text())
    if (p['cases']!=['C005','C006','C007','C008','C010','C013'] or p['resolution_cases']!=['C006','C008','C013']
            or p['particles']!=[512,1024] or p['filter_seeds']!=3 or p['fixed_lag']!=32
            or p['oral_status']!='deferred_pending_user_manual_review'):
        raise ValueError('Require the registered marginal-execution budget')
    old_dir=ROOT/p['previous_run'];old=read(old_dir/'protocol.json');done=read(old_dir/'completion.json')
    if done['status']!='complete' or done['protocol_sha256']!=digest(old_dir/'protocol.json'):
        raise ValueError('Require complete previous smoothing diagnostics')
    output.mkdir(parents=True);jobs=[];cases=[]
    for case in old['cases']:
        cid=case['case_id']
        if cid not in p['cases'] or smoke and cid not in ('C005','C010'):continue
        originals=[j for j in old['jobs'] if j['recovery_case']==cid]
        if any(j['candidate']['hyperparams'][study.KAPPA]!=0 for j in originals):
            raise ValueError('Nonzero-inertia case cannot use the exact local marginalizer')
        cases.append(dict(case,baseline_jobs=originals))
        for level in ([2] if smoke else [512,1024] if cid in p['resolution_cases'] else [512]):
            for index,original in enumerate(originals[:1] if smoke else originals):
                job=deepcopy(original);name=f'{cid}_RB_P{level}_S{index}'
                for key in ('smoothing_sources','genealogy_target','original_state','original_state_sha256','original_score','original_score_sha256'):
                    job.pop(key,None)
                job.update(name=name,kind='filter',model='COMPACT',execution_mode='MIX',comparison_arm='ADAPT',
                    particles=level,seeds=original['seeds'],smoke=smoke,phase='training' if smoke else 'evaluate',
                    marginal_execution_sources=sources(),source_sha256=study.sources(study.driver.io),
                    target=str((output/'scores'/name).resolve()),state_capture_target=str((output/'states'/name).resolve()),
                    marginal_target=str((output/'marginal'/name).resolve()),capture_sources=capture.sources())
                job['mechanism']['capacity']='evidence'
                if smoke:
                    # Actual full fixture remains unchanged; the standard
                    # worker limits training smoke filters to eight trials.
                    job['prefix_trials']=8
                jobs.append(job)
    def dispatch(batch):
        workers=1 if smoke else min(p['parallel_budget'],len(os.sched_getaffinity(0)),len(batch))
        original=validation.ENTRY;validation.ENTRY=ENTRY
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                validation.NumericalQueue(pool).execute(batch,output,workers)
        finally:validation.ENTRY=original
    controls=[]
    control_jobs=jobs if smoke else [j for j in jobs if j['recovery_case'] in ('C005','C013') and j['particles']==512 and j['seeds']==next(x['seeds'] for x in jobs if x['recovery_case']==j['recovery_case'])]
    for job in control_jobs:
        bare=deepcopy(job);bare.update(kind='bare',name=job['name']+'_bare',target=job['target']+'_bare')
        bare.pop('capture_sources');bare.pop('state_capture_target');bare.pop('marginal_target')
        controls.append(bare)
    started=time.monotonic()
    if smoke:
        dispatch(controls)
        for job in jobs:
            bare=next(b for b in controls if b['name']==job['name']+'_bare')
            job.update(original_score=bare['target']+'.npz',original_score_sha256=digest(Path(bare['target']).with_suffix('.npz')))
    write(output/'protocol.json',dict(policy=p,sources=sources(),config_sha256=digest(config),
        previous_protocol_sha256=digest(old_dir/'protocol.json'),jobs=jobs,bare_controls=controls,cases=cases,
        smoke=smoke,oral_used=False,parameter_refitting=False,
        interpretation='same zero-inertia single-rule generative model; execution summed for inference'))
    dispatch(jobs if smoke else controls+jobs)
    passive=[]
    for bare in controls:
        job=next(j for j in jobs if bare['name']==j['name']+'_bare')
        with np.load(bare['target']+'.npz') as a,np.load(job['target']+'.npz') as b:
            if set(a.files)!=set(b.files):raise ValueError('Passive recording changed score arrays')
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        passive.append(dict(case_id=job['recovery_case'],particles=job['particles'],seeds=job['seeds'],
            exact_prediction_replay=True,bare_array_sha256=digest(Path(bare['target']).with_suffix('.npz'))))
    write(output/'completion.json',dict(status='complete',smoke=smoke,filters=len(jobs),bare_filters=len(controls),
        seconds=time.monotonic()-started,protocol_sha256=digest(output/'protocol.json'),
        passive_replay_controls=passive,oral_used=False))
    print('Zero-inertia execution marginalization and recovery capture complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config and args.output:run(args.config,args.output,args.smoke)
    else:parser.error('Provide --worker or --config and --output')
