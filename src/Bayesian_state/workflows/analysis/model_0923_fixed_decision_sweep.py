"""Check feature effects using one completed particle-basis capture.

Run --source CAPTURE --output NEW_DIRECTORY [--grid-points 11]. All original
trials and particles are retained. --trial-limit N marks a short diagnostic.
Each perceived feature is swept from zero to one with every other coordinate
fixed. Sixteen additional joint probes reflect selected coordinates as 1-x.
The unchanged probe must reconstruct the recorded particle predictions before
any summaries are saved. Both original prior and current-choice-conditioned
particle weights are held fixed across all probes. Oral reports are excluded.

This is an instantaneous model sensitivity diagnostic. It does not simulate
learning under new stimuli, establish human causal effects, or define a verbal
report likelihood. One archived-runtime process uses one numerical thread.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
HELPER='src/Bayesian_state/evaluation/oral/fixed_decision.py'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def verify(inputs: dict) -> None:
    for path,sha in inputs.items():
        if digest(Path(path))!=sha:raise ValueError('Bound input changed: '+path)


def worker(file: Path) -> None:
    import pandas as pd
    from threadpoolctl import threadpool_info, threadpool_limits
    from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
    from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    from src.Bayesian_state.hypothesis_space.geometry.boundary import BoundaryGeometry
    from src.Bayesian_state.utils.numeric import softmax

    plan=read(file);verify(plan['inputs_sha256']);source=read(Path(plan['source_job']))
    target=Path(source['target']);record=read(target.with_suffix('.json'))
    meta=read(Path(source['folder'])/'prepared.json');engine=read(Path(source['folder'])/'engine.json')
    if (engine['provenance']['label_mapping']['mode']!='fixed_task_labels'
            or engine['likelihood']['distance_mode']!='boundary'
            or engine['output_noise']['kwargs']['enabled'] is not False
            or engine['choice_readout']['kwargs']['method']!='expectation'
            or engine['choice_readout']['kwargs'].get('strategy_confidence_gain',0.)!=0.
            or engine['choice_readout']['kwargs'].get('power',1.)!=1.):
        raise ValueError('This diagnostic requires the declared fixed-label boundary readout')
    spec=importlib.util.spec_from_file_location('fixed_decision_diagnostic',ROOT/HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    space=build_continuous_hypothesis_space(4,2) if record['condition']==1 else build_relational_0923_space()
    if json.loads(json.dumps(space.signature))!=record['catalogue_signature']:
        raise ValueError('Source category geometry changed')
    kwargs=engine['partition']['kwargs']
    geometry=BoundaryGeometry(space,method=kwargs.get('boundary_distance_method',BoundaryGeometry.METHOD_DYKSTRA),
        tolerance=kwargs.get('boundary_distance_tolerance',1e-9),
        projection_iterations=kwargs.get('boundary_projection_iterations',100))
    output=Path(plan['output']);grid=np.asarray(plan['grid']);rows=[];curve_rows=[]
    with np.load(target.with_suffix('.npz')) as z: saved={k:z[k] for k in z.files}
    n=plan['trials'];prefix='particle_basis_';features=['neck','tail','head','leg']
    corners=((np.arange(16)[:,None]>>np.arange(4))&1).astype(bool)
    ranges=[];endpoints=[];joint_predictions=[];baseline_error=0.;checked=0
    with threadpool_limits(limits=1):
        for trial in range(n):
            x=saved[prefix+'perceived_stimulus'][trial]
            probes=helper.feature_grid(x,grid)
            reflected=np.where(corners[None,:,:],1.-x[:,None,:],x[:,None,:])
            probes=np.concatenate((probes,reflected),axis=1)
            probabilities=helper.fixed_rule_readout(saved[prefix+'rule_ids'][trial],
                saved[prefix+'choice_weights'][trial],saved[prefix+'active_beta'][trial],probes,
                category_distances=geometry.category_distances,softmax=softmax)
            expected=saved[prefix+'prediction'][trial]
            error=float(np.max(abs(probabilities[:,0]-expected)));baseline_error=max(error,baseline_error)
            np.testing.assert_allclose(probabilities[:,0],expected,atol=1e-12,rtol=0)
            np.testing.assert_allclose(probabilities[:,-16],expected,atol=1e-12,rtol=0)
            choice=int(saved['choices'][trial])-1
            curve=probabilities[:,1:1+4*len(grid),choice].reshape(len(x),4,len(grid))
            joint=probabilities[:,-16:,choice];joint_predictions.append(joint)
            ranges.append(np.ptp(curve,axis=-1));endpoints.append(curve[:,:,-1]-curve[:,:,0])
            prior=saved[prefix+'prior_weights'][trial]
            conditional=saved['conditional_particle_weights'][trial]
            reconstructed=prior*expected[:,choice];reconstructed/=reconstructed.sum()
            np.testing.assert_allclose(reconstructed,conditional,atol=1e-12,rtol=0)
            for label,weights in [('prior',prior),('choice_conditioned',conditional)]:
                summary=helper.summarize_feature_sweep(curve,weights)
                for feature,name in enumerate(features):
                    rows.append(dict(trial=trial+1,feature=name,weighting=label,
                        **{k:float(v[feature]) for k,v in summary.items() if k!='curve'}))
                    for index,value in enumerate(grid):
                        curve_rows.append(dict(trial=trial+1,feature=name,weighting=label,
                                               feature_value=float(value),choice_probability=float(summary['curve'][feature,index])))
            checked+=expected.size+conditional.size
            if (trial+1)%32==0 or trial+1==n:print(f'{trial+1}/{n} fixed decision states checked',flush=True)
        pools=threadpool_info()
        if any(p['num_threads']!=1 for p in pools):raise ValueError('Need one numerical thread')
    pd.DataFrame(rows).to_csv(output/'effects.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(output/'curves.csv',index=False)
    with (output/'particle_effects.npz').open('xb') as stream:
        np.savez_compressed(stream,particle_range=np.asarray(ranges),endpoint_effect=np.asarray(endpoints),
            joint_reflection_probability=np.asarray(joint_predictions),reflection_masks=corners,
            prior_weights=saved[prefix+'prior_weights'][:n],conditional_weights=saved['conditional_particle_weights'][:n],
            choices=saved['choices'][:n],grid=grid,seeds=saved['seeds'])
    verify(plan['inputs_sha256'])
    write(output/'completion.json',dict(status='complete',subject=record['subject'],condition=record['condition'],
        arm=record['arm'],trials=n,training_trials=record['training_trials'],particles=record['particles'],
        seeds=record['seeds'],candidate=record['candidate'],short_diagnostic=n<record['trials'] or bool(source['trial_limit']),
        baseline_probability_checks=checked,maximum_baseline_error=baseline_error,numeric_pools=pools,
        inputs_sha256=plan['inputs_sha256'],protocol_sha256=digest(file),parameters_changed=False,oral_used=False,
        scope='Fixed decision states, not learning trajectories or a report likelihood; one original filter',
        outputs_sha256={name:digest(output/name) for name in ('effects.csv','curves.csv','particle_effects.npz')}))


def run(source: Path, output: Path, grid_points: int=11, trial_limit: int=0) -> None:
    if output.exists():raise FileExistsError('Use a new fixed-decision diagnostic directory')
    if not 3<=grid_points<=21 or trial_limit<0:raise ValueError('Declare a bounded grid and nonnegative trial limit')
    done=read(source/'completion.json')
    if done['status']!='complete' or done['parameters_changed'] or done['oral_used']:
        raise ValueError('Require a completed report-independent particle capture')
    jobs=list((source/'jobs').glob('*.json'))
    if len(jobs)!=1:raise ValueError('One original filter per capture is required')
    job=read(jobs[0]);target=Path(job['target']);record=read(target.with_suffix('.json'))
    proof=read(target.with_suffix('.particle_basis.json'))
    if (proof!=done['worker'] or proof['schema_version']!=1
            or proof['array_sha256']!=digest(target.with_suffix('.npz'))
            or record['array_sha256']!=proof['array_sha256']):
        raise ValueError('Captured decision arrays changed')
    original_file=Path(done['source_job'])
    original=read(original_file)
    extras=('particle_basis_source_job','particle_basis_source_job_sha256','particle_basis_inputs_sha256')
    restored={key:value for key,value in job.items() if key not in extras}
    restored['target'],restored['replay_source_sha256']=original['target'],original['replay_source_sha256']
    if (restored!=original or Path(job['particle_basis_source_job']).resolve()!=original_file.resolve()
            or job['particle_basis_source_job_sha256']!=digest(original_file)):
        raise ValueError('Capture job changed its original scientific identity')
    inputs=dict(done['inputs_sha256'])
    for path in [Path(__file__),ROOT/HELPER,source/'completion.json',jobs[0],
                 *[target.with_suffix(s) for s in ('.json','.npz','.particle_basis.json')],
                 Path(job['folder'])/'engine.json',Path(job['folder'])/'prepared.json']:
        path=path.resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:raise ValueError('Input binding changed')
        inputs[str(path)]=sha
    meta=read(Path(job['folder'])/'prepared.json')
    for name,sha in meta['frozen_cognitive_source_sha256'].items():
        path=str(Path(job['runtime'])/name)
        if path in inputs and inputs[path]!=sha:raise ValueError('Archived source bindings disagree')
        inputs[path]=sha
    verify(inputs);output.mkdir(parents=True)
    plan=dict(source=str(source.resolve()),source_job=str(jobs[0].resolve()),output=str(output.resolve()),
        grid=np.linspace(0,1,grid_points).tolist(),trials=min(record['trials'],trial_limit) if trial_limit else record['trials'],
        inputs_sha256=inputs,workers=1,numeric_threads=1,original_particle_weights_held_fixed=True,
        joint_contrast='all16 subsets of perceived features reflected as1-x; mask0 unchanged')
    file=output/'protocol.json';write(file,plan)
    launcher=('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
              "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
    with tempfile.TemporaryDirectory(prefix='fixed_decision_sweep_') as cache:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1',NUMBA_CACHE_DIR=cache,MPLCONFIGDIR=cache,PYTHONDONTWRITEBYTECODE='1')
        with (output/'worker.log').open('x') as stream:
            subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/ENTRY),'--worker',str(file.resolve())],
                           cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--grid-points',type=int,default=11)
    parser.add_argument('--trial-limit',type=int,default=0)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.source is None or args.output is None:parser.error('--source and --output are required')
    else:run(args.source,args.output,args.grid_points,args.trial_limit)
