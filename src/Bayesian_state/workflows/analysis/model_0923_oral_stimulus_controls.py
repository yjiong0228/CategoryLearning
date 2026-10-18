"""Compare dynamic and prefix-frozen rule mixtures under current stimuli.

Use --alignment COMPLETE_ALIGNMENT --subject ID --output NEW_DIRECTORY.
--draws fixes independent perception samples per trial and integration bank;
--integration-seeds declares two common random banks, shared by every arm.
Both frozen controls use only predictive rule weights available at the prefix
boundary. This supplements the original particle readout, retained alongside
its common-kernel approximation. No parameter selection or state replay occurs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import model_0923_oral_parameter_comparison as comparison
from .model_0923_h_validation import write
from ...evaluation.oral import stimulus_controls as control
from ...evaluation.oral.decision_basis_scoring import DOMAINS,LABELS
from ...evaluation.oral.information_controls import jackknife_error
from ...hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from ...hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from ...model.modules.perception import PerceptionModule

ROOT=Path(__file__).resolve().parents[4]


def archived_geometry(runtime: Path, space, inputs: dict):
    """Use the same archived distance solver that produced the saved readout."""
    digest=comparison.alignment.digest
    def archived(relative,name):
        path=runtime/relative;inputs[str(path.resolve())]=digest(path)
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    prefix='src/Bayesian_state/hypothesis_space/'
    for relative in ('spaces/__init__.py','spaces/continuous.py','spaces/structural_0923.py','geometry/distance_cache.py'):
        current=ROOT/prefix/relative;old=runtime/prefix/relative
        if digest(current)!=digest(old):raise ValueError('Archived geometry dependency differs: '+relative)
        inputs[str(current.resolve())]=digest(current);inputs[str(old.resolve())]=digest(old)
    numeric=Path('src/Bayesian_state/utils/numeric.py')
    if digest(ROOT/numeric)!=digest(runtime/numeric):raise ValueError('Archived numerical helper differs')
    inputs[str((ROOT/numeric).resolve())]=digest(ROOT/numeric)
    inputs[str((runtime/numeric).resolve())]=digest(runtime/numeric)
    acceleration=runtime/prefix/'geometry/dykstra_acceleration.py'
    if acceleration.exists():
        archived(prefix+'geometry/dykstra_acceleration.py','src.Bayesian_state.hypothesis_space.geometry.dykstra_acceleration')
    module=archived(prefix+'geometry/boundary.py','src.Bayesian_state.hypothesis_space.geometry._oral_control_boundary')
    stimulus=archived(prefix+'geometry/stimuli.py','src.Bayesian_state.hypothesis_space.geometry._oral_control_stimuli')
    module.as_stimuli=stimulus.as_stimuli
    return module.BoundaryGeometry(space)


def load_group(directory: Path, subject: int, arm: str) -> tuple:
    """Read mixed BASE/ONLINE experiments using their actual per-job interventions."""
    read,digest=comparison.alignment.read,comparison.alignment.digest
    inputs={}
    def bind(path,expected=None):
        path=Path(path).resolve();sha=digest(path)
        if expected is not None and sha!=expected:raise ValueError('Changed frozen input: '+str(path))
        inputs[str(path)]=sha;return path
    manifest=read(bind(directory/'manifest.json'))
    if (manifest['status']!='complete' or manifest['smoke'] or not manifest['full_replay_complete']
            or subject not in manifest['subjects'] or arm not in manifest['arms']):
        raise ValueError('Require a complete declared development alignment')
    source,audit=Path(manifest['source']),Path(manifest['oral_audit'])
    protocol=read(bind(source/'protocol.json',manifest['source_protocol_sha256']))
    if read(bind(source/'completion.json'))['status']!='complete':raise ValueError('Incomplete state replay')
    coding=read(bind(audit/'manifest.json',manifest['oral_manifest_sha256']))
    if coding['status']!='complete' or coding['config']['sample_role']!='development':
        raise ValueError('Only the declared development corpus is supported')
    for folder,hashes in [(directory,manifest['artifacts']),(audit,coding['artifacts'])]:
        for name,sha in hashes.items():bind(folder/name,sha)
    for name,sha in coding['input_sha256'].items():bind(ROOT/name,sha)
    records=[];identity=None
    for path,sha in manifest['state_arrays'].items():
        file=Path(path).with_suffix('.json');record=read(file)
        if (record['subject'],record['arm'])!=(subject,arm):continue
        bind(file);bind(path,sha);job=read(bind(source/'jobs'/file.name))
        if (record['array_sha256']!=sha or not record['exact_behavior_replay']
                or record['candidate']!=job['candidate'] or record['seeds']!=job['seeds'] or len(record['seeds'])!=1
                or job.get('reported_arm',job['model'])!=arm
                or record.get('online_belief',False) is not job.get('online_belief',False)
                or record.get('structural_intervention')!=('retained_rule_precision' if job.get('retain_precision',False) else None)):
            raise ValueError('State identity differs from its executed mechanism')
        if job.get('online_belief',False) and (record.get('online_evidence_matches_retained_state') is not True
                or record.get('online_reconstructions')!=record['particles']*(record['trials']-1)):
            raise ValueError('Missing actual ONLINE reconstruction checks')
        bind(job['original_job'],job['original_job_sha256']);reference=bind(job['reference'],job['reference_sha256'])
        with np.load(path) as saved,np.load(reference) as original:
            np.testing.assert_array_equal(saved['prediction'],original['probabilities'][0,:record['trials']])
            np.testing.assert_array_equal(saved['choices'],original['choices'][:record['trials']])
        task=Path(job['folder']);meta=read(bind(task/'prepared.json'))
        for name,expected in meta['array_sha256'].items():bind(task/name,expected)
        bind(task/'engine.json',meta['engine_sha256'])
        current=dict(candidate=record['candidate'],scientific={key:record[key] for key in
            ('particles','trials','training_trials','condition','catalogue_signature','component_policy','timing')},
            mechanism=dict(job['mechanism'],retain_precision=job.get('retain_precision',False),online_belief=job.get('online_belief',False)),
            task_sha256={name:digest(task/name) for name in ('training.npz','sealed_test.npz','engine.json')})
        if identity is not None and current!=identity:raise ValueError('Mixed parameter or mechanism identities within a group')
        identity=current;records.append((file,record))
    records.sort(key=lambda item:item[1]['seeds'][0])
    if (len(records)!=protocol['settings']['seeds_per_arm'] or len({r['seeds'][0] for _,r in records})!=len(records)):
        raise ValueError('Incomplete or duplicated filter seeds')
    scored=comparison.alignment.group_scores(source,records,pd.read_csv(audit/'trials.csv'),pd.read_csv(audit/'component_matches.csv'))
    trials=pd.read_csv(directory/'trials.csv');trials=trials[trials.iSub.eq(subject)&trials.arm.eq(arm)].reset_index(drop=True)
    summary=pd.read_csv(directory/'summary.csv');summary=summary[summary.subject.eq(subject)&summary.arm.eq(arm)].reset_index(drop=True)
    pd.testing.assert_frame_equal(scored[0].reset_index(drop=True),trials,check_dtype=False,atol=1e-12,rtol=0)
    pd.testing.assert_frame_equal(pd.DataFrame(scored[1]),summary,check_dtype=False,atol=1e-12,rtol=0)
    return scored,identity,inputs,records


def run(alignment: Path, subject: int, output: Path, draws: int, integration_seeds: list[int]) -> None:
    if output.exists():
        raise FileExistsError('Use a new current-stimulus control directory')
    if draws<1 or len(integration_seeds)!=2 or len(set(integration_seeds))!=2:
        raise ValueError('Declare positive draws and two distinct integration seeds')
    read,digest=comparison.alignment.read,comparison.alignment.digest
    manifest=read(alignment/'manifest.json')
    source=Path(manifest['source']);audit=Path(manifest['oral_audit'])
    arms=manifest['arms']
    groups={};inputs={};identities={};records_by_arm={}
    for arm in arms:
        group,identity,more,records=load_group(alignment,subject,arm)
        groups[arm]=group;identities[arm]=identity;inputs.update(more)
        records_by_arm[arm]=records
    first_path,first=records_by_arm[arms[0]][0]
    job=read(source/'jobs'/first_path.name)
    folder=Path(job['folder']);meta=read(folder/'prepared.json');engine=read(folder/'engine.json')
    perception_file=ROOT/'src/Bayesian_state/model/modules/perception.py'
    archived=Path(job['runtime'])/perception_file.relative_to(ROOT)
    if digest(perception_file)!=digest(archived):
        raise ValueError('Supplementary integration requires the executed perception implementation')
    module=engine['modules']['perception_mod']
    if module['class']!='src.Bayesian_state.model.modules.perception.PerceptionModule':
        raise ValueError('Unsupported perception model')
    kwargs=dict(module.get('kwargs',{}));kwargs.pop('module_seed',None)
    proxy=SimpleNamespace(subject_id=subject,processed_data_dir=meta['paths']['processed_dir'],dataset_paths=meta['paths'])
    stimuli=[];choices=[]
    for name in ('training.npz','sealed_test.npz'):
        path=folder/name
        if digest(path)!=meta['array_sha256'][name]:raise ValueError('Prepared task changed')
        with np.load(path) as saved:stimuli.append(saved['stimulus']);choices.append(saved['choices'])
    stimuli,choices=np.concatenate(stimuli),np.concatenate(choices)
    cut=meta['training_trials']
    space=build_continuous_hypothesis_space(4,2) if meta['condition']==1 else build_relational_0923_space()
    catalogue_signature=json.loads(json.dumps(space.signature))
    if catalogue_signature!=first['catalogue_signature'] or len(choices)!=first['trials']:
        raise ValueError('Current-stimulus control changed catalogue or trial sequence')
    for path in [archived,perception_file,folder/'prepared.json',folder/'engine.json',
                 *[Path(v) for k,v in meta['paths'].items() if k!='processed_dir']]:
        inputs[str(path.resolve())]=digest(path)
    geometry=archived_geometry(Path(job['runtime']),space,inputs);kernels=[];noise=[]
    for seed in integration_seeds:
        perception=PerceptionModule(proxy,module_seed=seed,**kwargs)
        # Use the executed module's sampler and the same [0,1] clipping as process().
        perceived=np.stack([np.clip(perception.sample(np.broadcast_to(x,(draws,len(x)))),0.,1.) for x in stimuli])
        kernels.append(control.component_kernel(space,perceived,choices,geometry))
        noise.append(dict(seed=seed,mode=perception.noise_mode,mean=perception.mean.tolist(),std=perception.std.tolist(),
            uniform_half_range=None if perception.uniform_half_range is None else perception.uniform_half_range.tolist()))
    kernels=np.stack(kernels);kernel=kernels.mean(0)
    tables=[];trial_tables=[];estimates={};saved_arrays=dict(component_kernels=kernels,choices=choices,stimulus=stimuli)
    query_matches=pd.read_csv(audit/'component_matches.csv')
    for arm in arms:
        scored,_,_=groups[arm];prior=[];seeds=[]
        for path,record in records_by_arm[arm]:
            if (record['catalogue_signature']!=catalogue_signature or record['training_trials']!=cut
                    or record['trials']!=len(choices)):
                raise ValueError('Source groups disagree on the task boundary or catalogue')
            own_job=read(source/'jobs'/path.name)
            if any(identities[arm]['task_sha256'][key]!=identities[arms[0]]['task_sha256'][key]
                   for key in ('training.npz','sealed_test.npz','engine.json')):
                raise ValueError('All arms need the identical task and perception configuration')
            if Path(own_job['folder']).resolve()!=folder.resolve():
                raise ValueError('Ambiguous prepared source across arms')
            with np.load(path.with_suffix('.npz')) as z:
                np.testing.assert_array_equal(z['choices'],choices)
                prior.append(z['prior_component_mass']);seeds.append(record['seeds'][0])
        prior=np.stack(prior);seeds=np.array(seeds)
        labels=comparison.alignment.make_labels(scored,query_matches,*prior.shape[2:])
        if arm!=arms[0]:
            pd.testing.assert_frame_equal(scored[[c for c in comparison.IDENTITY if c!='arm']],
                groups[arms[0]][0][[c for c in comparison.IDENTITY if c!='arm']])
            np.testing.assert_array_equal(seeds,saved_arrays[arms[0]+'_seeds'])
        values=control.rule_mixture_controls(prior,kernel,labels,cut)
        split=np.stack([control.rule_mixture_controls(prior,k,labels,cut) for k in kernels])
        for d,domain in enumerate(DOMAINS):
            for k,label in enumerate(LABELS):
                np.testing.assert_allclose(values[:,0,:,d,k].mean(0),scored['prior_'+domain+'_'+label],rtol=0,atol=1e-12)
        saved_arrays.update({arm+'_values':values,arm+'_seeds':seeds,arm+'_rules':prior.sum(-1),arm+'_labels':labels,
                             arm+'_split_values':split})
        complete=scored.scope.eq('complete').to_numpy()
        windows=dict(test=np.arange(len(choices))>=cut,last64=np.arange(len(choices))>=max(cut,len(choices)-64))
        out=scored[comparison.IDENTITY].copy()
        for m,name in enumerate(control.CONTROL_NAMES):
            for d,domain in enumerate(DOMAINS):
                for k,label in enumerate(LABELS):out[name+'_'+domain+'_'+label]=values[:,m,:,d,k].mean(0)
        trial_tables.append(out)
        for window,mask in windows.items():
            mask=mask&complete
            if not mask.any():raise ValueError('No complete reports in the declared later window')
            point=values[:,:,mask].mean((0,2))
            deleted=np.stack([np.delete(values,i,axis=0)[:,:,mask].mean((0,2)) for i in range(len(seeds))])
            error=jackknife_error(deleted)
            integration_difference=split[1][:,:,mask].mean((0,2))-split[0][:,:,mask].mean((0,2))
            estimates[arm,window]=dict(point=point,deleted=deleted,seeds=seeds,mask=mask)
            for m,name in enumerate(control.CONTROL_NAMES):
                for d,domain in enumerate(DOMAINS):
                    row=dict(subject=subject,arm=arm,window=window,metric=name,domain=domain,reports=int(mask.sum()),
                             filter_seeds=len(seeds),draws_per_integration_bank=draws)
                    for k,label in enumerate(LABELS):row.update({label:point[m,d,k],label+'_mcse':error[m,d,k],
                        label+'_integration_bank_difference':integration_difference[m,d,k]})
                    tables.append(row)
    contrasts=[]
    for arm in arms[1:]:
        for window in ('test','last64'):
            a,b=estimates[arms[0],window],estimates[arm,window]
            np.testing.assert_array_equal(a['mask'],b['mask']);np.testing.assert_array_equal(a['seeds'],b['seeds'])
            difference=b['point']-a['point'];error=jackknife_error(b['deleted']-a['deleted'])
            for m,name in enumerate(control.CONTROL_NAMES):
                for d,domain in enumerate(DOMAINS):
                    row=dict(subject=subject,reference_arm=arms[0],arm=arm,window=window,metric=name,domain=domain,
                             reports=int(a['mask'].sum()))
                    for k,label in enumerate(LABELS):row.update({label+'_difference':difference[m,d,k],label+'_difference_mcse':error[m,d,k]})
                    contrasts.append(row)
    for path,h in inputs.items():
        if digest(Path(path))!=h:raise ValueError('Input changed during current-stimulus controls')
    output.mkdir(parents=True)
    scores=pd.DataFrame(tables);scores.to_csv(output/'scores.csv',index=False)
    pd.concat(trial_tables).to_csv(output/'trials.csv',index=False)
    pd.DataFrame(contrasts).to_csv(output/'contrasts.csv',index=False)
    with (output/'controls.npz').open('xb') as stream:np.savez_compressed(stream,**saved_arrays)
    note=('动态权重、前段末端冻结权重和前段平均冻结权重，共用当前刺激及独立知觉采样。'
        '当前选择仅指定待解释类别；规则权重均为本题选择加权前的作答贡献。'
        '末端冻结使用第一道后段题选择前的权重，前段平均按前段试次等权。'
        '原粒子读出同时保留，factorization_difference记录公共知觉核与原读出的差异。'
        '模拟标准误仅反映完整滤波重复；两套知觉采样的差另列。全部比较属于开发诊断。')
    display=scores[scores.domain.eq('structural')&scores.window.eq('test')]
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>保留当前刺激的冻结规则对照</title><style>body{max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>保留当前刺激的冻结规则对照</h1><p>'+note+'</p>'+display.to_html(index=False,float_format=lambda v:f'{v:.6f}')
        +'<p><a href="contrasts.csv">方案差值</a> · <a href="trials.csv">逐题读出</a></p></html>')
    sources=[Path(__file__),Path(control.__file__),Path(comparison.__file__),Path(comparison.alignment.__file__),perception_file]
    (output/'source').mkdir()
    for path in sources:shutil.copy2(path,output/'source'/path.name)
    write(output/'manifest.json',dict(status='complete',alignment=str(alignment.resolve()),subject=subject,arms=arms,
        training_trials=cut,draws=draws,integration_seeds=integration_seeds,noise=noise,identities=identities,
        inputs_sha256=inputs,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()},
        control_scope='Marginal rule-mixture weights; original particle readout retained; common independent perception kernel',
        geometry_runtime=str(Path(job['runtime']).resolve()),geometry_method=geometry.method,
        geometry_backend=geometry.dykstra_backend,
        boundary='Predictive first-later-trial weights; current choice and feedback excluded from the frozen mixture',
        uncertainty='Filter MCSE conditional on the common kernel; independent integration-bank differences reported separately',
        data_use='Frozen behavioral parameters; no oral selection or fitting; development participants'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment',type=Path,required=True);parser.add_argument('--subject',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--draws',type=int,default=256)
    parser.add_argument('--integration-seeds',type=int,nargs=2,default=[640001,640003])
    args=parser.parse_args();run(args.alignment,args.subject,args.output,args.draws,args.integration_seeds)
