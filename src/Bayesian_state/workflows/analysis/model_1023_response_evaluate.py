"""Frozen response-mechanism evaluation with an explicit joint oral bridge.

Run --study COMPLETE_RESPONSE_REFIT --output NEW [--smoke]. Selected parameters
remain fixed. The original conditional filter captures report support without
oral inputs; complete autonomous paths use generated task feedback. Original
archived seed banks pair both response mechanisms and the previous references.
Original-readout points with identical scientific jobs reuse archived results;
the new fit identity and the complete reuse checks remain in the case manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util
import json
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


def scientific_job(job: dict) -> dict:
    """Remove output locations and bookkeeping, retaining every other input."""
    bookkeeping={'name','target','case_id','source_sha256','evaluation_route_sources',
        'decision_basis_target','decision_basis_sources','response_evaluation_sources','bindings','dynamic_role'}
    value={k:v for k,v in job.items() if k not in bookkeeping}
    value['candidate']={k:job['candidate'][k] for k in ('id','hyperparams')}
    value['response_exponent']=float(job.get('response_exponent',1.))
    return value


def reusable_reference(record: dict,planned_jobs: list[dict]) -> dict|None:
    """Check same task, mechanism, parameters, seeds, precision and saved arrays."""
    import numpy as np
    if record['response_exponent']!=1. or any(j['smoke'] for j in planned_jobs):return None
    own=Path(record['reference_evaluation']);protocol=read(own/'protocol.json')
    point=read(Path(protocol['fit'])/'nominees.json')['nominee']['candidate']
    if point['id']!=record['candidate'] or point['hyperparams']!=record['candidate_hyperparams']:return None
    original=[(f,read(f)) for f in sorted((own/'jobs').glob('*.json'))]
    canonical=lambda jobs:sorted(json.dumps(scientific_job(j),sort_keys=True,allow_nan=False) for j in jobs)
    if canonical([j for _,j in original])!=canonical(planned_jobs):return None
    inputs={}

    def bind(file,expected=None):
        file=Path(file).resolve();sha=digest(file)
        if expected is not None and sha!=expected:raise ValueError('Reuse source changed: '+str(file))
        inputs[str(file)]=sha

    done=read(own/'completion.json')
    if done['status']!='complete' or done['smoke'] or protocol['compact_variant']!='TRANSFER':
        raise ValueError('Reuse requires the complete original TRANSFER evaluation')
    bind(own/'completion.json');bind(own/'protocol.json')
    bind(Path(protocol['fit'])/'nominees.json',protocol['nomination_sha256'])
    for file,job in original:
        bind(file)
        for name,sha in job['source_sha256'].items():bind(ROOT/name,sha)
        for key in ('evaluation_route_sources','decision_basis_sources'):
            for name,sha in job.get(key,{}).items():bind(ROOT/name,sha)
    for job in planned_jobs:
        for name,sha in job['source_sha256'].items():bind(ROOT/name,sha)
    folder=Path(planned_jobs[0]['folder']);meta=read(folder/'prepared.json');bind(folder/'prepared.json')
    bind(folder/'engine.json',meta['engine_sha256'])
    for name,sha in meta['array_sha256'].items():bind(folder/name,sha)
    for name,sha in meta['frozen_cognitive_source_sha256'].items():bind(Path(planned_jobs[0]['runtime'])/name,sha)
    for role in ('conditional','autonomous'):
        file=own/role/'evaluation.json';bind(file);saved,=read(file)['rows']
        if saved['candidate']!=point:raise ValueError('Reused prediction identity differs')
        arrays=own/role/'evaluate'/f"subject_{record['subject']}"/'COMPACT'/(point['id']+'.npz')
        bind(arrays,saved['predictions_sha256'])
        with np.load(arrays) as archive:
            np.testing.assert_array_equal(archive['seeds'],record['reference_seeds'][role])
    captures=list((own/'state_capture').glob('*.json'))
    if len(captures)!=8:raise ValueError('Require all archived original-filter captures')
    for file in captures:
        bind(file);saved=read(file);bind(file.with_suffix('.npz'),saved['array_sha256'])
        bind(saved['original_job'],saved['original_job_sha256']);bind(saved['score_file'],saved['score_sha256'])
    return dict(status='verified',response_exponent=1.,identical_scientific_jobs=len(planned_jobs),
        evaluation=str(own),inputs_sha256=inputs,
        condition='Exact parameters and scientific job inputs; exponent one is the unchanged observation channel')


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
        for name in ('nominees.json','completion.json'):
            file=Path(c['output'])/name;inputs[str(file.resolve())]=digest(file)
        record=dict(case_id=c['case_id'],subject=c['subject'],condition=c['condition'],cohort=c['cohort'],
            response_exponent=c['response_exponent'],reference_seeds=seed_banks,case=c,
            candidate=frozen['nominee']['candidate']['id'],candidate_hyperparams=frozen['nominee']['candidate']['hyperparams'],
            evaluation=str((output/c['case_id']).resolve()),reference_evaluation=original['evaluation'],reuse_archive=False)
        if not smoke and c['response_exponent']==1.:
            chunks,groups=old.make_jobs(c,Path(record['evaluation']),False)
            paired_seed_jobs(groups,seed_banks)
            proof=reusable_reference(record,chunks)
            if proof is not None:
                record.update(reuse_archive=True,reuse_proof=proof,evaluation=proof['evaluation'])
                inputs.update(proof['inputs_sha256'])
        planned.append(record)
    write(output/'cases.json',dict(cases=planned,inputs_sha256=inputs,oral_used_for_selection=False))
    write(output/'protocol.json',dict(study=str(study.resolve()),smoke=smoke,workers=workers,source_sha256=sources(),
        fitting_phase=protocol['phase'],oral_bridge='q(c) * w(h) * p(c|h) / p(c), within each particle',
        archived_reuses=sum(r['reuse_archive'] for r in planned),
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
        pending=[r for r in planned if not r['reuse_archive']]
        with ThreadPoolExecutor(max_workers=1 if smoke else max(1,len(pending))) as controls:
            for future in as_completed([controls.submit(evaluate,r,pool) for r in pending]):future.result()
    dynamics.verify(inputs)
    write(output/'completion.json',dict(status='complete',smoke=smoke,candidates=len(planned),
        archived_reuses=len(planned)-len(pending),new_evaluations=len(pending),
        cases_sha256=digest(output/'cases.json'),additional_filters_for_oral_capture=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('study','output','worker'):parser.add_argument('--'+k,type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.study and args.output:run(args.study,args.output,args.smoke)
    else:parser.error('Supply --study and --output, or --worker')
