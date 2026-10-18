"""Check retained precision on existing SEARCH paths in several task conditions.

Run --config CONFIG --output NEW_DIRECTORY. Reuse the original shared model,
the precision-retention worker and its summary definitions. Each subject keeps
one saved parameter point and the declared first original seeds. All original
SEARCH paths are retained as RESET references, with one exact replay per subject.
--smoke checks one seed per subject and sixteen trials, with one worker.
Sources can be fixed-parameter SEARCH studies or completed SEARCH refit evaluations;
the parameter provenance remains explicit in the subject records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import yaml

from . import model_0923_reentry_precision_probe as probe
from .model_0923_recalled_capacity_memory_probe import reference_jobs

ROOT=probe.ROOT
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))


def source_provenance(source: Path, v) -> tuple[str, dict[str, str]]:
    """Recognize scientific fixed-point studies and frozen refit evaluations."""
    completion=v.read(source/'completion.json');protocol=v.read(source/'protocol.json')
    if completion['status']!='complete':raise ValueError('Incomplete SEARCH source')
    inputs={str((source/name).resolve()):v.digest(source/name) for name in ('completion.json','protocol.json')}
    if 'fit' in protocol:
        frozen=Path(protocol['fit'])/'nominees.json';nominees=v.read(frozen)
        if (nominees.get('smoke') is not False or nominees['status']!='frozen_before_test'
                or v.digest(frozen)!=protocol['nomination_sha256']):
            raise ValueError('Refit evaluation lacks a frozen scientific nomination')
        inputs[str(frozen.resolve())]=v.digest(frozen)
        return 'joint_SEARCH_refit',inputs
    if completion.get('smoke') is not False or protocol.get('smoke') is not False:
        raise ValueError('Require a scientific fixed-parameter SEARCH study')
    return 'fixed_prior_POOL_parameters_applied_to_SEARCH',inputs


def run(config: Path, output: Path, smoke: bool) -> None:
    recalled=probe.helpers();v=recalled.base.validation();settings=yaml.safe_load(config.read_text())
    if output.exists():raise FileExistsError('Use a new development-comparison directory')
    if not settings['subjects'] or type(settings['paths']) is not int or settings['paths']<3:
        raise ValueError('Require subjects and at least three scientific paths')
    jobs=[];hashes={};inputs={str(config.resolve()):v.digest(config)};groups=[]
    for subject,source_name in settings['subjects'].items():
        source=ROOT/source_name;provenance,source_inputs=source_provenance(source,v);inputs.update(source_inputs)
        files=reference_jobs(source,subject);remaining=1 if smoke else settings['paths'];parts=[]
        for file in files:
            if remaining==0:break
            original=v.read(file);ref=Path(original['target']).with_suffix('.npz');record=v.read(ref.with_suffix('.json'))
            if v.digest(ref)!=record['predictions_sha256'] or record['candidate']!=original['candidate']:
                raise ValueError('SEARCH reference identity or predictions changed')
            if original.get('prototype',dict(proposal=True,prior=False))!=dict(proposal=True,prior=False):
                raise ValueError('Reference uses a different recalled-search mechanism')
            for name,expected in original['source_sha256'].items():
                if v.digest(ROOT/name)!=expected:raise ValueError(f'Original executing dependency changed: {name}')
                if name in hashes and hashes[name]!=expected:raise ValueError('Subject sources use different implementations')
                hashes[name]=expected
            for path in (file,ref,ref.with_suffix('.json')):inputs[str(path.resolve())]=v.digest(path)
            seeds=original['seeds'][:remaining];remaining-=len(seeds)
            name=f'RETAIN_S{subject}_{len(parts):03d}'
            job=dict(deepcopy(original),name=name,seeds=seeds,smoke=smoke,retain_precision=True,
                reference=str(ref.resolve()),reference_sha256=v.digest(ref),
                target=str((output/f'subject_{subject}'/'paths'/name).resolve()))
            parts.append(job);jobs.append(job)
            if len(parts)==1:
                control=deepcopy(job);name=f'RESET_CONTROL_S{subject}'
                control.update(name=name,seeds=seeds[:1],retain_precision=False,
                    target=str((output/f'subject_{subject}'/'paths'/name).resolve()))
                jobs.append(control)
        if remaining or not parts:raise ValueError('Insufficient original seeds for a declared subject')
        point=parts[0]['candidate']['id']
        groups.append(dict(subject=subject,point=point,source=str(source.resolve()),
            parameter_provenance=provenance,
            candidate=parts[0]['candidate'],settings=dict(settings,points=[point])))
    dependencies=(ENTRY,probe.ENTRY,probe.MECHANISM,probe.DIAGNOSIS,
        'src/Bayesian_state/workflows/analysis/model_0923_recalled_search_refit.py',
        'src/Bayesian_state/workflows/analysis/model_0923_recalled_capacity_memory_probe.py',
        'src/Bayesian_state/evaluation/model_0923_h_generation_uncertainty.py')
    hashes.update({name:v.digest(ROOT/name) for name in dependencies})
    output.mkdir(parents=True);recalled.base.archive(output,config,hashes)
    for job in jobs:job['source_sha256']=hashes
    v.write(output/'protocol.json',dict(settings=settings,smoke=smoke,jobs=len(jobs),groups=groups,
        source_sha256=hashes,input_sha256=inputs,python=sys.version,
        scope='Three-condition development replication at existing fixed SEARCH points',
        data_use='No fitting; own-choice feedback; no oral data; task target used only in diagnostics'))
    recalled.ENTRY=probe.ENTRY
    recalled.execute(jobs,output,1 if smoke else settings['parallel_budget'])
    tables={key:[] for key in ('scores','contrasts','path_windows','curves')}
    completed=[]
    for group in groups:
        subject=group['subject'];selected=[j for j in jobs if f'/subject_{subject}/' in j['target']]
        destination=output/f'subject_{subject}'
        probe.summarize(destination,selected,group['settings'],smoke)
        completed.append(dict(subject=subject,completion_sha256=v.digest(destination/'completion.json')))
        for key in tables:
            frame=pd.read_csv(destination/f'{key}.csv')
            if set(frame.subject)!={subject}:raise ValueError('Summary mixes subjects')
            tables[key].append(frame.assign(parameter_provenance=group['parameter_provenance']))
    for key,frames in tables.items():pd.concat(frames,ignore_index=True).to_csv(output/f'{key}.csv',index=False)
    for name,expected in dict(inputs,**hashes).items():
        if v.digest(Path(name))!=expected:raise ValueError('Input or executing source changed')
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,groups=completed,
        new_paths=(1 if smoke else settings['paths'])*len(groups),exact_baseline_paths=len(groups),
        artifacts={p.name:v.digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.config,args.output,args.smoke)
