"""Test dominant-rule execution in the two frozen compact variants.

Use --config YAML --output NEW [--smoke [--smoke-case CASE]]. Full runs require
the completed primary evaluation. All six participants, both fitted variants
and all 128 original path seeds are retained. Only the existing dominant-rule
execution adapter changes; individual parameters remain frozen. The original
mixture paths are reused. This post-hoc development diagnostic measures the
autonomous effect before any decision about behavior refitting or oral checks.

Smoke runs use two original seeds and sixteen original training trials, one
worker, with both disabled and enabled execution. The disabled paths must
reproduce the corresponding original prefixes exactly in every saved array.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
EVALUATE = 'src/Bayesian_state/workflows/analysis/model_1022_compact_evaluate.py'
EXECUTION = 'src/Bayesian_state/model/modules/hypothesis_transition/dominant_rule_execution.py'
PRECISION_DRIVER = 'src/Bayesian_state/workflows/analysis/model_1021_entry_precision_probe.py'


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluation = load_file('compact_execution_evaluation', EVALUATE)
read, write, digest = evaluation.read, evaluation.write, evaluation.digest


def sources() -> dict:
    return {p:digest(ROOT/p) for p in (ENTRY, EVALUATE, EXECUTION, PRECISION_DRIVER)}


def worker(file: Path) -> None:
    job = read(file)
    if (job['phase'] != 'autonomous' or job['model'] != 'COMPACT'
            or job['execution_probe_sources'] != sources()
            or job['evaluation_route_sources'] != evaluation.sources()
            or type(job['execution_enabled']) is not bool):
        raise ValueError('Execution probe lost its frozen mechanism or source identity')
    variant = job['compact_variant']
    if variant == 'TRANSFER':
        if job.get('entry_control') != 'PRECISION':
            raise ValueError('Transfer control requires its original precision adapter')
        inner = load_file('compact_execution_precision', PRECISION_DRIVER)
        driver, model_worker = inner.driver, inner.worker
    elif variant == 'COMPACT':
        driver = load_file('compact_execution_original',
            'src/Bayesian_state/workflows/analysis/model_1021_simplification.py')
        model_worker = driver.worker
    else:
        raise ValueError('Unknown frozen compact variant')
    original_load = driver.base.load
    audits = []

    def load(name, path):
        module = original_load(name, path)
        if path == driver.COMPACT:
            adapter = original_load('src.Bayesian_state.model.modules.hypothesis_transition.dominant_rule_execution', EXECUTION)
            original_context = module.compact_evidence

            @contextmanager
            def combined(cls):
                with original_context(cls) as compact_audit:
                    with adapter.dominant_rule_execution(cls, enabled=job['execution_enabled']) as events:
                        audits.append(events)
                        yield compact_audit

            module.compact_evidence = combined
        return module

    driver.base.load = load
    try:
        model_worker(file)
    finally:
        driver.base.load = original_load
    if len(audits) != 1:
        raise ValueError('Execution adapter was not installed exactly once')
    target = Path(job['target']);record = read(target.with_suffix('.json'))
    expected = record['trials']*len(job['seeds']) if job['execution_enabled'] else 0
    if len(audits[0]) != expected:
        raise ValueError('Execution did not cover every simulated trial')
    with np.load(target.with_suffix('.npz')) as z:
        if job['execution_enabled'] and not np.isin(z['contribution'], [0.,1.]).all():
            raise ValueError('Dominant execution retained mixed response contributions')
    write(target.with_suffix('.execution.json'), dict(status='complete', enabled=job['execution_enabled'],
        compact_variant=variant, events=len(audits[0]), trials=record['trials'], paths=len(job['seeds']),
        source_sha256=sources(), original_parameters_unchanged=True, additional_parameters=0))


def launch(job: dict, folder: Path) -> None:
    folder.mkdir(exist_ok=True)
    file = folder/(job['name']+'.json');write(file,job)
    launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
    with tempfile.TemporaryDirectory(prefix='compact_execution_') as cache:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                   NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
        with file.with_suffix('.log').open('x') as log:
            result = subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/ENTRY),
                '--worker',str(file)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Execution probe failed: '+str(file.with_suffix('.log')))


def run(config: Path, output: Path, smoke: bool, smoke_case: str | None = None) -> None:
    from src.Bayesian_state.evaluation.model_0923_h_validation import checked_arrays, block_paths, distribution_scores, jackknife_se
    from src.Bayesian_state.evaluation.model_0923_h_generation_uncertainty import path_jackknife
    if output.exists():raise FileExistsError('Use a new execution-probe directory')
    if smoke_case and not smoke:raise ValueError('A single case is a software check only')
    policy = yaml.safe_load(config.read_text());study,source = ROOT/policy['study'],ROOT/policy['evaluation']
    if not smoke and read(source/'completion.json')['status']!='complete':
        raise ValueError('Finish the primary evaluation before the full autonomous probe')
    cases = [c for c in read(study/'cases.json')['cases'] if c['kind']=='human']
    if ({c['subject'] for c in cases}!=set(policy['subjects'])
            or {c['variant'] for c in cases}!=set(policy['variants'])
            or len(cases)!=len(policy['subjects'])*len(policy['variants'])):
        raise ValueError('Probe must retain the complete registered human comparison')
    if smoke:
        cases=[c for c in cases if c['subject'] in policy['smoke_subjects']]
        if smoke_case:cases=[c for c in cases if c['case_id']==smoke_case]
    if not cases:raise ValueError('No registered probe cases')
    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    shutil.copyfile(config,output/'config.yaml')
    bindings={};jobs=[];groups=[]
    def bind(file):bindings[str(file.resolve())]=digest(file)
    for file in (config,study/'cases.json',study/'human_nominees.json',source/'protocol.json'):bind(file)
    for case in cases:
        case_id=case['case_id'];original=source/case_id
        if read(original/'completion.json')['status']!='complete':
            raise ValueError('Require complete original paths for '+case_id)
        for file in (original/'completion.json',original/'protocol.json',original/'autonomous/evaluation.json',
                     original/'conditional/evaluation.json'):bind(file)
        baseline,=read(original/'autonomous/evaluation.json')['rows']
        original_arrays=checked_arrays(original/'autonomous',baseline)
        array=original/'autonomous/evaluate'/f'subject_{case["subject"]}'/'COMPACT'/(baseline['candidate']['id']+'.npz')
        bind(array)
        conditional,=read(original/'conditional/evaluation.json')['rows']
        behavior=checked_arrays(original/'conditional',conditional)
        bind(original/'conditional/evaluate'/f'subject_{case["subject"]}'/'COMPACT'/(conditional['candidate']['id']+'.npz'))
        if conditional['candidate']['hyperparams']!=baseline['candidate']['hyperparams']:
            raise ValueError('Observed comparison uses another frozen parameter point')
        np.testing.assert_array_equal(behavior['categories'],original_arrays['categories'])
        for mode in (('MIX','SELECT') if smoke else ('SELECT',)):
            own=output/case_id/mode;own.mkdir(parents=True)
            _,planned=evaluation.make_jobs(case,own,False)
            whole,parts=planned['autonomous'][0]
            if whole['candidate']['hyperparams']!=baseline['candidate']['hyperparams']:
                raise ValueError('Execution changed fitted parameters')
            if smoke:
                data=evaluation.smoke_fixture(Path(whole['folder']).parent,own/'software_fixture',case['subject'])
                parts=parts[:1]
                for job in [whole]+parts:
                    job.update(folder=str((data/f'subject_{case["subject"]}').resolve()),smoke=True,particles=2)
                    job['seeds']=job['seeds'][:2]
            for job in parts:
                job.update(execution_enabled=mode=='SELECT',execution_probe_sources=sources())
                jobs.append((job,own/'jobs'))
            groups.append((case,mode,whole,parts,own,baseline,original_arrays,behavior['choices']))
    workers=1 if smoke else min(policy['workers'],128,len(os.sched_getaffinity(0)),len(jobs))
    write(output/'protocol.json',dict(policy=policy,smoke=smoke,workers=workers,source_sha256=sources(),
        inputs_sha256=bindings,jobs=[j for j,_ in jobs],
        scope='Fixed-parameter post-hoc autonomous diagnostic; no new filters, fitting or oral selection'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(launch,j,d) for j,d in jobs]):future.result()
    scores=[];contrasts=[];checks=[]
    for case,mode,whole,parts,own,baseline,original_arrays,observed_choices in groups:
        row=evaluation.driver.io.combine_evaluation_chunks(whole,parts,True)
        write(own/'autonomous/evaluation.json',dict(status='complete',rows=[row]))
        arrays=checked_arrays(own/'autonomous',row)
        np.testing.assert_array_equal(arrays['categories'],original_arrays['categories'][:len(arrays['categories'])])
        indices=[original_arrays['seeds'].tolist().index(s) for s in arrays['seeds']]
        if smoke and mode=='MIX':
            for key,actual in arrays.items():
                reference=original_arrays[key]
                expected=reference[indices] if key=='seeds' else (reference[indices,:16] if reference.ndim>1 else reference[:16])
                np.testing.assert_array_equal(actual,expected)
            checks.append(dict(case_id=case['case_id'],exact_original_prefix=True,trials=16,paths=2))
        if smoke:continue
        np.testing.assert_array_equal(arrays['seeds'],original_arrays['seeds'])
        cut=row['training_trials'];length=row['trials'];truth=observed_choices==arrays['categories']
        for phase,first,last in [('training',0,cut),('test',cut,length),('whole',0,length)]:
            own_scores={};deleted={}
            for label,bank in [('MIX',original_arrays),('SELECT',arrays)]:
                correct=bank['choices']==bank['categories'][None,:]
                samples=block_paths(correct,first,last,32);observed=block_paths(truth[None,:],first,last,32)[0]
                own_scores[label]=distribution_scores(samples,observed);deleted[label]=path_jackknife(samples,observed)
                late=max(first,last-128);paths=correct[:,late:last].mean(1)
                scores.append(dict(case_id=case['case_id'],subject=case['subject'],variant=case['variant'],
                    cohort=case['cohort'],mode=label,phase=phase,**own_scores[label],late_accuracy=float(paths.mean()),
                    observed_late=float(truth[late:last].mean()),late_q10=float(np.quantile(paths,.1)),
                    late_q90=float(np.quantile(paths,.9)),target_late_present=float(bank['present'][:,late:last].mean()),
                    target_late_contribution=float(bank['contribution'][:,late:last].mean())))
            contrast=dict(case_id=case['case_id'],subject=case['subject'],variant=case['variant'],cohort=case['cohort'],phase=phase)
            for metric in ('energy','crps'):
                contrast['delta_'+metric]=own_scores['SELECT'][metric]-own_scores['MIX'][metric]
                contrast['delta_'+metric+'_mc_se']=jackknife_se(deleted['SELECT'][metric]-deleted['MIX'][metric])
            contrasts.append(contrast)
    if not smoke:
        pd.DataFrame(scores).to_csv(output/'scores.csv',index=False)
        pd.DataFrame(contrasts).to_csv(output/'contrasts.csv',index=False)
    write(output/'completion.json',dict(status='complete',smoke=smoke,cases=len(cases),jobs=len(jobs),workers=workers,
        checks=checks,inputs_sha256=bindings,artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')},
        interpretation='SELECT minus original MIX at frozen parameters; full refitting remains a separate decision'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-case')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config and args.output:run(args.config,args.output,args.smoke,args.smoke_case)
    else:parser.error('Use --config and --output, or internal --worker')
