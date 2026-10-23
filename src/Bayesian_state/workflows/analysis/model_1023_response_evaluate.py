"""Frozen response-mechanism evaluation with an explicit joint oral bridge.

Run --study COMPLETE_RESPONSE_REFIT --output NEW [--smoke]. Selected parameters
remain fixed. The original conditional filter captures report support without
oral inputs; complete autonomous paths use generated task feedback. Original
archived seed banks pair both response mechanisms and the previous references.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
DYNAMICS='src/Bayesian_state/workflows/analysis/model_1023_response_dynamics.py'
spec=importlib.util.spec_from_file_location('response_eval_dynamics',ROOT/DYNAMICS)
dynamics=importlib.util.module_from_spec(spec);sys.modules[spec.name]=dynamics;spec.loader.exec_module(dynamics)
read,write,digest,load=dynamics.read,dynamics.write,dynamics.digest,dynamics.load
EVALUATOR='src/Bayesian_state/workflows/analysis/model_1022_compact_evaluate.py'
BRIDGE='src/Bayesian_state/evaluation/oral/response_basis.py'
COLLECTOR='src/Bayesian_state/evaluation/oral/decision_basis.py'


def sources() -> dict:
    return {k:digest(ROOT/k) for k in (ENTRY,DYNAMICS,EVALUATOR,BRIDGE,COLLECTOR,dynamics.ADAPTER,
        'src/Bayesian_state/workflows/analysis/model_1021_simplification_capture.py',
        'src/Bayesian_state/inference/backends/particle_filter.py')}


def worker(file: Path) -> None:
    job=read(file)
    if job['response_evaluation_sources']!=sources():raise ValueError('Response evaluation sources changed')
    if job['phase']=='autonomous':
        dynamics.worker(file)
    else:
        old=load('response_original_evaluator',EVALUATOR);capture=old.capture
        load('src.Bayesian_state.model.modules.readout.category_power',dynamics.ADAPTER)
        load('src.Bayesian_state.evaluation.oral.decision_basis',COLLECTOR)
        bridge=load('src.Bayesian_state.evaluation.oral.response_basis',BRIDGE)
        original_load,original_worker=capture.driver.load,capture.driver.worker

        def adapted_load(name,path):
            if path==capture.COLLECTOR:
                return SimpleNamespace(DecisionBasisCollector=lambda space,choices:
                    bridge.ResponseBasisCollector(space,choices,exponent=job['response_exponent']))
            return original_load(name,path)

        capture.driver.load=adapted_load;capture.driver.worker=dynamics.worker
        try:capture.worker(file)
        finally:capture.driver.load,capture.driver.worker=original_load,original_worker
    write(Path(job['target']).with_suffix('.response_evaluation.json'),dict(status='complete',
        response_exponent=job['response_exponent'],source_sha256=sources(),
        oral_bridge='q(c) times original within-state relative rule support for c',
        oral_inputs_used=False,additional_filters_for_capture=0))


def run(study: Path,output: Path,smoke: bool) -> None:
    from . import model_1022_compact_evaluate as old
    from .model_1023_frozen_evaluate import paired_seed_jobs
    from .model_1023_estimation_reliability import NumericQueue
    if output.exists():raise FileExistsError('Use a new response evaluation directory')
    done,protocol=read(study/'completion.json'),read(study/'protocol.json')
    if done['status']!='complete' or done['smoke'] is not smoke:
        raise ValueError('Require completed matching response fits')
    cases=read(study/'cases.json')['cases']
    reference=ROOT/'results/model_1023/estimation_concurrent_20261023_v1/human_evaluation'
    prior=read(reference/'cases.json')['cases']
    inputs={str(p):digest(p) for p in [study/'completion.json',study/'protocol.json',study/'cases.json',reference/'cases.json']}
    output.mkdir(parents=True)
    workers=1 if smoke else min(128,len(os.sched_getaffinity(0)))
    planned=[]
    for c in cases:
        original=next(r for r in prior if r['subject']==c['subject'] and 'selected' in r['roles'])
        seed_banks={k:v[:2] if smoke else v for k,v in original['reference_seeds'].items()}
        training_seeds={s for k,v in c['independent_seed_banks'].items() if k!='evaluate' for s in v}
        if any(set(v)&training_seeds for v in seed_banks.values()):raise ValueError('Evaluation and fitting seeds overlap')
        frozen=read(Path(c['output'])/'nominees.json')
        planned.append(dict(case_id=c['case_id'],subject=c['subject'],condition=c['condition'],cohort=c['cohort'],
            response_exponent=c['response_exponent'],reference_seeds=seed_banks,case=c,
            candidate=frozen['nominee']['candidate']['id'],candidate_hyperparams=frozen['nominee']['candidate']['hyperparams'],
            evaluation=str((output/c['case_id']).resolve()),reference_evaluation=original['evaluation']))
    write(output/'cases.json',dict(cases=planned,inputs_sha256=inputs,oral_used_for_selection=False))
    write(output/'protocol.json',dict(study=str(study.resolve()),smoke=smoke,workers=workers,source_sha256=sources(),
        fitting_phase=protocol['phase'],oral_bridge='q(c) * w(h) * p(c|h) / p(c), within each particle',
        report_definition='Original conditional rule/component basis and frozen coding rules',
        data_scope='Repeated development evaluation using original temporal boundaries'))

    def launch(job,own):
        file=own/'jobs'/(job['name']+'.json');write(file,job)
        launcher=('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='response_evaluation_') as cache,file.with_suffix('.log').open('x') as log:
            env.update(NUMBA_CACHE_DIR=cache,MPLCONFIGDIR=cache)
            subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/ENTRY),'--worker',str(file.resolve())],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

    def evaluate(record,pool):
        case=record['case'];own=Path(record['evaluation']);own.mkdir(parents=True);(own/'jobs').mkdir()
        chunks,groups=old.make_jobs(case,own,smoke)
        paired_seed_jobs(groups,record['reference_seeds'])
        for j in chunks:
            j.update(response_exponent=record['response_exponent'],response_evaluation_sources=sources(),
                dynamic_role='task_start' if j['phase']=='autonomous' else 'evaluate',
                bindings={str(ROOT/k):v for k,v in dict(j['source_sha256'],**sources()).items()})
        fit=Path(case['output'])
        write(own/'protocol.json',dict(fit=str(fit),nomination_sha256=digest(fit/'nominees.json'),
            compact_variant='TRANSFER',case_id=case['case_id'],smoke=smoke,cohort=case['cohort'],
            response_exponent=record['response_exponent'],route_source_sha256=sources(),jobs=chunks))
        write(own/'decision_basis_capture_protocol.json',dict(source_sha256=old.capture.sources(),
            response_bridge_sources=sources(),additional_filters=0,timing='after_choice_before_current_feedback'))
        for future in as_completed([pool.submit(launch,j,own) for j in chunks]):future.result()
        for role,items in groups.items():
            rows=[old.driver.io.combine_evaluation_chunks(j,parts,role=='autonomous') for j,parts in items]
            write(own/role/'evaluation.json',dict(status='complete',rows=rows))
        write(own/'completion.json',dict(status='complete',smoke=smoke,jobs=len(chunks),compact_variant='TRANSFER',
            response_exponent=record['response_exponent'],nomination_sha256=digest(fit/'nominees.json')))
        print(case['case_id']+': frozen response evaluation complete',flush=True)

    with NumericQueue(workers,output) as pool:
        with ThreadPoolExecutor(max_workers=1 if smoke else len(planned)) as controls:
            for future in as_completed([controls.submit(evaluate,r,pool) for r in planned]):future.result()
    dynamics.verify(inputs)
    write(output/'completion.json',dict(status='complete',smoke=smoke,candidates=len(planned),
        cases_sha256=digest(output/'cases.json'),additional_filters_for_oral_capture=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('study','output','worker'):parser.add_argument('--'+k,type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.study and args.output:run(args.study,args.output,args.smoke)
    else:parser.error('Supply --study and --output, or --worker')
