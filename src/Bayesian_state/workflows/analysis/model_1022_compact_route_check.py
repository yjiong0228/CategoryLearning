"""Check passive oral capture for both compact variants in all three conditions.

Use --study FROZEN_REFIT --output NEW after all fits freeze. Six short checks
use two particles, eight valid original trials and one numerical worker. Each
captured filter is compared exactly to an uncaptured filter with the same
parameters and seed. Support-valid test points exercise precision transfer;
they are software fixtures and do not select or evaluate scientific estimates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from . import model_1022_compact_evaluate as evaluation
from .model_1022_compact_refit import candidate
from .model_0923_decision_basis_alignment import validate_basis_record

read,write,digest = evaluation.read,evaluation.write,evaluation.digest


def run(study: Path, output: Path):
    if output.exists():
        raise FileExistsError('Use a new routing-validation directory')
    frozen = read(study/'human_nominees.json')
    if frozen['status']!='frozen_before_test' or frozen['smoke']:
        raise ValueError('Wait until all scientific human and recovery estimates freeze')
    cases = [c for c in read(study/'cases.json')['cases'] if c['kind']=='human' and c['cohort']=='development']
    if {(c['condition'],c['variant']) for c in cases}!={(i,v) for i in (1,2,3) for v in ('COMPACT','TRANSFER')}:
        raise ValueError('Need the original three-condition, two-variant design')
    hashes = read(study/'protocol.json')['source_sha256']
    output.mkdir(parents=True)
    remaining = sum(c['settings']['budgets']['confirmation']['seeds']
                    for c in read(study/'cases.json')['cases'] if c['kind']=='recovery')
    # Frozen fits leave at most the declared truth-scoring filters. This check
    # uses one worker alongside them, then the main evaluation starts.
    maximum=min(128,len(os.sched_getaffinity(0)))
    if remaining>=maximum and not (study/'completion.json').exists():
        raise ValueError('Wait for truth filters to finish before the one-worker check')
    gate = evaluation.ResourceGate(study,maximum,remaining)
    write(output/'protocol.json',dict(smoke=True,scientific_inference=False,numeric_workers=1,
        study=str(study.resolve()),source_sha256=hashes,route_sha256=evaluation.sources(),
        validation_source_sha256=digest(Path(__file__)),human_nomination_sha256=digest(study/'human_nominees.json')))
    records=[]
    for case in cases:
        subject,variant=case['subject'],case['variant']
        point=deepcopy(case['required']['hyperparams'])
        h=evaluation.driver.H
        # Stay within the declared support while exercising entries and learning.
        for key,end in ((h+'search_controller.baseline_probability',-1),
                        (h+'search_controller.error_gain',-1),
                        (h+'max_capacity',0),('engine.modules.beta_mod.kwargs.beta_init',0),
                        ('engine.modules.beta_mod.kwargs.learning_rate',-1)):
            if key not in case['space']:
                raise ValueError('The test point must use declared parameter names: '+key)
            point[key]=case['space'][key][end]
        proposed=candidate(point,'software_route_check')
        own=output/case['case_id'];own.mkdir()
        fixture=evaluation.smoke_fixture(Path(case['settings']['data']),own/'software_fixture',subject)
        job,=evaluation.driver.io.score_jobs(fixture,own/'base',
            evaluation.driver.base.settings_for_scoring(case['settings']),
            {(subject,'COMPACT'):[proposed]},'evaluate',True)
        job.update(arm_settings=case['settings']['arms']['COMPACT'],source_sha256=hashes,
            compact_variant=variant,case_id=case['case_id'],evaluation_route_sources=evaluation.sources(),
            seeds=job['seeds'][:1])
        job['mechanism']['capacity']='evidence'
        if variant=='TRANSFER': job['entry_control']='PRECISION'
        jobs=[]
        for observed in (False,True):
            row=deepcopy(job);tag='captured' if observed else 'uncaptured'
            row.update(name=case['case_id']+'_'+tag,target=str((own/tag/'score').resolve()))
            if observed:
                row.update(decision_basis_sources=evaluation.capture.sources(),
                    decision_basis_target=str((own/'state_capture'/row['name']).resolve()))
            evaluation.launch(row,own,gate);jobs.append(row)
        a,b=[Path(j['target']).with_suffix('.npz') for j in jobs]
        with np.load(a) as x,np.load(b) as y:
            if set(x.files)!=set(y.files): raise ValueError('Capture changed output fields')
            for key in x.files: np.testing.assert_array_equal(x[key],y[key])
        capture_file=Path(jobs[1]['decision_basis_target']).with_suffix('.json')
        record=read(capture_file)
        validate_basis_record(capture_file.with_suffix('.npz'),record)
        transferred=None
        if variant=='TRANSFER':
            audits=[read(Path(j['target']).with_suffix('.entry_precision.json')) for j in jobs]
            if audits[0]!=audits[1] or audits[0]['transferred_entries']<=0:
                raise ValueError('The software fixture did not exercise unchanged precision transfer')
            transferred=audits[0]['transferred_entries']
        records.append(dict(subject=subject,condition=case['condition'],variant=variant,
            exact_arrays=True,trials=record['trials'],particles=record['particles'],
            transferred_entries=transferred,input_job_sha256={str((own/'jobs'/(j['name']+'.json')).resolve()):
                digest(own/'jobs'/(j['name']+'.json')) for j in jobs},
            output_sha256={str(p.resolve()):digest(p) for p in (a,b,capture_file,capture_file.with_suffix('.npz'))}))
        print(case['case_id']+': captured and uncaptured arrays match exactly',flush=True)
    write(output/'completion.json',dict(status='complete',smoke=True,scientific_inference=False,
        numeric_workers=1,cases=records,scope='Eight-trial routing parity across three conditions and two variants'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.study,args.output)
