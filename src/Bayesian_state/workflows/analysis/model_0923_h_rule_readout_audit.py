"""Replay ONLY paths and audit residual perception/precision contributions.

The saved choices must replay exactly. Counterfactual probabilities evaluate
the task-correct rule on the SAME physical or perceived stimuli at fixed beta
values; they never feed back into learning or generate a new trajectory.
Use --source DIAGNOSTIC_RUN --output NEW. Reuses its bounded worker budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_h_rule_diagnosis.py'
BETAS = (5., 10., 25., 100.)


def workflow():
    spec = importlib.util.spec_from_file_location('readout_diagnostic_workflow', REPO / BASE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def worker(file: Path):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_info
    w = workflow(); v = w.validation(); job = v.read(file)
    if v.digest(Path(__file__)) != job['audit_source_sha256']: raise ValueError('Readout audit source changed')
    intervention = v.intervention_workflow(); intervention.install_transition()
    d = intervention.cohort().load_diagnosis()
    helper = w.load('src.Bayesian_state.simulation.h_rule_diagnostic', w.HELPER)
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    engine, meta, arrays = w.inputs(job, v)
    reference = Path(job['only_source'])
    if v.digest(reference) != job['only_sha256']: raise ValueError('ONLY reference changed')
    with np.load(reference) as z:
        expected = {k:z[k] for k in ('choice','prediction','seeds')}
    np.testing.assert_array_equal(expected['seeds'], job['seeds'])
    n = len(arrays['categories']); late = max(meta['training_trials'], n-128); y = arrays['categories'][late:]
    rows=[]; cpu=time.process_time()

    class Recorder:
        def __init__(self):
            self.next_path=0; self.perceived=[]; self.beta=[]; self.model=None
        def before(self, model, choice, feedback):
            prepared=model._pending_trial
            if prepared.trial_index >= late:
                self.perceived.append(prepared.perceived_stimulus.copy())
                self.beta.append(float(prepared.beta[job['oracle']]))
            self.model=model
        def after(self, *args): pass

    for index, seed in enumerate(job['seeds']):
        recorder=Recorder()
        with helper.intervention(cls, arm='ONLY', target=job['oracle'], onset=meta['training_trials']):
            with d.record_lifecycle(recorder):
                result=d.autonomous.run_autonomous_category_learning(engine_config=engine, subject_id=job['subject'],
                    condition=meta['condition'], stimulus=arrays['stimulus'], categories=arrays['categories'],
                    trajectory_seed=seed,processed_data_dir=meta['paths']['processed_dir'],dataset_paths=meta['paths'])
        np.testing.assert_array_equal(result.trajectory.choices, expected['choice'][index])
        np.testing.assert_array_equal(result.trajectory.observed_probabilities, expected['prediction'][index])
        model=recorder.model
        def probabilities(x, beta):
            return model.partition_model.get_category_probabilities(hypo=job['oracle'],
                data=(x,np.ones(len(x),int),np.ones(len(x))),beta=float(beta),distance_mode=model.engine.distance_mode)
        perceived=np.asarray(recorder.perceived).reshape(len(y),-1); physical=arrays['stimulus'][late:]
        learned=np.asarray(recorder.beta)
        for kind,x in [('perceived',perceived),('physical',physical)]:
            for beta in BETAS:
                p=probabilities(x,beta)
                rows.append(dict(subject=job['subject'],model=job['model'],seed=seed,stimulus=kind,beta=str(beta),
                    correct_probability=float(p[y-1,np.arange(len(y))].mean()),mean_learned_beta=float(learned.mean())))
            values=np.array([probabilities(x[t:t+1],b)[y[t]-1,0] for t,b in enumerate(learned)])
            if kind=='perceived': np.testing.assert_allclose(values,expected['prediction'][index,late:][np.arange(len(y)),y-1],atol=1e-12,rtol=0)
            rows.append(dict(subject=job['subject'],model=job['model'],seed=seed,stimulus=kind,beta='learned',
                correct_probability=float(values.mean()),mean_learned_beta=float(learned.mean())))
    target=Path(job['audit_target']); target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target.with_suffix('.csv'),index=False)
    v.write(target.with_suffix('.json'),dict(replay_exact=True,paths=len(job['seeds']),cpu_seconds=time.process_time()-cpu,
        csv_sha256=v.digest(target.with_suffix('.csv')),numeric_pools=threadpool_info()))


def run(source: Path, output: Path):
    import pandas as pd
    import numpy as np
    from src.Bayesian_state.utils.parallel import parallel_job_count
    w=workflow();v=w.validation();complete=v.read(source/'completion.json');protocol=v.read(source/'protocol.json')
    if complete['status']!='complete' or complete['smoke']: raise ValueError('Need complete scientific diagnostics')
    if output.exists(): raise FileExistsError('Use a new readout-audit directory')
    output.mkdir(parents=True); (output/'jobs').mkdir()
    jobs=[]
    for original in v.read(source/'trace_plan.json')['jobs']:
        reference=Path(original['target'])/'ONLY.npz'; record=v.read(reference.parent/'completion.json')
        jobs.append(dict(original,only_source=str(reference),only_sha256=record['artifact_sha256']['ONLY.npz'],
            audit_target=str((output/'chunks'/original['name']).resolve()),audit_source_sha256=v.digest(Path(__file__))))
    v.write(output/'protocol.json',dict(source=str(source.resolve()),source_protocol_sha256=v.digest(source/'protocol.json'),
        jobs=jobs,betas=BETAS,operation='readout only on fixed perceived stimuli and fixed learned states; no counterfactual feedback'))
    def launch(job):
        file=output/'jobs'/(job['name']+'.json');v.write(file,job)
        launcher=('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='h_rule_readout_') as cache:
            env['NUMBA_CACHE_DIR']=cache
            with file.with_suffix('.log').open('x') as stream:
                result=subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(Path(__file__).resolve()),'--worker',str(file)],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Audit worker failed: {file.with_suffix(".log")}')
    workers=parallel_job_count(protocol['settings']['parallel_budget'],len(jobs))
    print(f'{len(jobs)} readout jobs, {workers} workers, numeric threads=1',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(launch,j) for j in jobs]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%32==0 or i==len(jobs): print(f'{i}/{len(jobs)} complete',flush=True)
    records=[];frames=[]
    for job in jobs:
        target=Path(job['audit_target']);record=v.read(target.with_suffix('.json'))
        if v.digest(target.with_suffix('.csv'))!=record['csv_sha256']: raise ValueError('Audit CSV changed')
        records.append(record);frames.append(pd.read_csv(target.with_suffix('.csv')))
    data=pd.concat(frames,ignore_index=True);data.to_csv(output/'paths.csv',index=False)
    rows=[]
    for key,group in data.groupby(['subject','model','stimulus','beta']):
        values=group.correct_probability.to_numpy()
        rows.append(dict(zip(['subject','model','stimulus','beta'],key),mean=float(values.mean()),
            mc_se=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,
            paths=len(values),mean_learned_beta=float(group.mean_learned_beta.mean())))
    pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
    v.write(output/'completion.json',dict(status='complete',replay_paths=sum(r['paths'] for r in records),
        cpu_seconds=sum(r['cpu_seconds'] for r in records),source_sha256=v.digest(Path(__file__)),
        summary_sha256=v.digest(output/'summary.csv'),paths_sha256=v.digest(output/'paths.csv')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.worker:worker(a.worker)
    elif a.source and a.output:run(a.source,a.output)
    else:p.error('Provide --source and --output')


if __name__=='__main__':main()
