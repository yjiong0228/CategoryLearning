"""Prespecified numerical follow-up for boundary shape checks in the pilot.

Register every boundary candidate after the 256-path check. Once the main study
finishes, add 512 independent calibration paths per registered point while
keeping its original 128 reference paths fixed. Cognitive code and thresholds
remain unchanged; all CPU is charged to the original 96-hour study budgets.
Primary artifacts remain intact and the extension receives separate reports.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.distance import cdist

from .model_1024_individual_regions import (
    BudgetQueue, ROOT, classify, completed_arrays, decision, digest,
    load_observed, make_jobs, read, sources, write,
)
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...metrics.trajectory_shape import describe_ensemble, wilson_interval


def fixed_reference_check(features, observed, reference_count, alpha):
    """Use the registered reference unchanged as calibration precision grows."""
    x=np.asarray(features,float);y=np.asarray(observed,float)
    if (x.ndim!=2 or y.shape!=x.shape[1:] or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not 2<=reference_count<=len(x)-2):
        raise ValueError('Need finite features with independent reference and calibration sets')
    reference=x[:reference_count];calibration=x[reference_count:]
    neighbors=max(2,int(np.sqrt(reference_count)))
    observed_distance=float(np.sort(np.linalg.norm(reference-y,axis=1))[:neighbors].mean())
    distance=np.sort(cdist(calibration,reference),axis=1)[:,:neighbors].mean(axis=1)
    count=int(np.count_nonzero(distance>=observed_distance));n=len(calibration)
    lo,hi=wilson_interval(count,n)
    return dict(status='compatible' if lo>alpha else 'incompatible' if hi<alpha else 'undetermined',
                rank_tail=(count+1)/(n+1),tail_count=count,neighbors=neighbors,calibration_paths=n,
                reference_paths=reference_count,tail_mc_low=lo,tail_mc_high=hi,
                observed_discrepancy=observed_distance,reference_q95=float(np.quantile(distance,.95)),alpha=alpha)


def register(study: Path, output: Path):
    if output.exists():raise FileExistsError('Use a new extension directory')
    protocol=read(study/'protocol.json');summary=read(study/'generation_confirmation_summary.json')
    targets=[dict(subject=r['subject'],candidate=r['candidate']) for r in summary if r['shape_check']['status']=='undetermined']
    output.mkdir(parents=True)
    policy=dict(study=str(study.resolve()),targets=targets,extra_paths=512,reference_paths=128,alpha=.05,
                source_summary_sha256=digest(study/'generation_confirmation_summary.json'),
                protocol_sha256=digest(study/'protocol.json'),numerical_sources=protocol['sources'],
                extension_source_sha256=digest(Path(__file__)),registered_unix=time.time(),
                rule='All current boundary points; retain fixed reference, thresholds, parameters and original paths. One extra batch only.',
                budget='Original generation/analysis and total study caps; no additional CPU allocation.')
    write(output/'registration.json',policy)
    shutil.copy2(Path(__file__),output/'extension_source.py')
    decision(study,'shape_extension_registered',targets=targets,extra_paths=512,reference_paths=128,
             reason='Reduce counting uncertainty at all current shape boundaries without changing the reference or criteria.')


def run(output: Path):
    registration=read(output/'registration.json');study=Path(registration['study'])
    if (output/'completion.json').exists():raise FileExistsError('Extension already complete')
    if digest(Path(__file__))!=registration['extension_source_sha256']:
        raise ValueError('Registered extension code changed')
    protocol=read(study/'protocol.json');completion=read(study/'completion.json')
    if digest(study/'protocol.json')!=registration['protocol_sha256'] or sources()!=registration['numerical_sources']:
        raise ValueError('Main protocol or numerical sources changed')
    cases={c['subject']:c for c in protocol['cases']}
    rows={sid:read(study/f'confirmed_S{sid}.json') for sid in cases}
    jobs=[]
    for target in registration['targets']:
        case=cases[target['subject']]
        point=next(r['candidate'] for r in rows[target['subject']] if r['candidate']['id']==target['candidate'])
        jobs.extend(make_jobs(protocol,case,[point],study,'shape_extension',paths=registration['extra_paths']))
    queue=BudgetQueue(study,protocol);queue.execute(jobs,'generation')
    outcomes=[]
    for target in registration['targets']:
        sid=target['subject'];case=cases[sid]
        row=next(r for r in rows[sid] if r['candidate']['id']==target['candidate'])
        current=[j for j in jobs if j['subject']==sid and j['candidate']['id']==target['candidate']]
        extra=completed_arrays(current,'choices')
        if extra is None:
            outcomes.append(dict(**target,status='extra_batch_incomplete'));continue
        _,observed=load_observed(Path(case['template']['folder']))
        saved=read(Path(row['shape_file']));records=saved['paths']+describe_ensemble(extra,observed['categories'],protocol['policy']['shape'])
        features=np.array([r['features'] for r in records]);actual=np.array(saved['observed']['features'])
        row['primary_shape_check']=deepcopy(row['shape_check'])
        row['shape_check']=fixed_reference_check(features,actual,registration['reference_paths'],registration['alpha'])
        row['shape_energy']=energy_u(features,actual)
        row['shape_labels']={label:sum(r['label']==label for r in records) for label in sorted({r['label'] for r in records})}
        row['generation_jobs'].extend(current)
        choices=completed_arrays(row['generation_jobs'],'choices');categories=observed['categories']
        classical=behavior_features(choices,categories,n_categories=2 if case['condition']==1 else 4)
        empirical=behavior_features(observed['choices'][None,:],categories,n_categories=2 if case['condition']==1 else 4)[0]
        row['ordered_energy_diagnostic']=energy_u(classical,empirical)
        location=output/'shapes'/f'S{sid}_{target["candidate"]}.json'
        write(location,dict(paths=records,observed=saved['observed'],policy=saved['policy'],fixed_reference_paths=128))
        row['shape_file']=str(location.resolve())
        outcomes.append(dict(**target,status=row['shape_check']['status'],shape_check=row['shape_check']))
    for sid,batch in rows.items():
        write(output/f'confirmed_S{sid}.json',classify(batch,protocol['policy']))
        (output/f'search_S{sid}.json').symlink_to((study/f'search_S{sid}.json').resolve())
    for name in ['calibration.json','jobs','numerical','decisions.jsonl','registered_config.yaml','source',
                 'reuse_manifest.json','preflight_validation.json']:
        (output/name).symlink_to((study/name).resolve())
    extended_protocol=deepcopy(protocol);extended_protocol['shape_extension']=registration
    write(output/'protocol.json',extended_protocol)
    updated=deepcopy(completion)
    updated.update(status='bounded_pilot_with_shape_precision_extension_complete',
                   cpu_hours={k:v/3600 for k,v in queue.used().items()},shape_extension_outcomes=outcomes,
                   protocol_sha256=digest(output/'protocol.json'),primary_completion_sha256=digest(study/'completion.json'))
    write(output/'completion.json',updated)
    decision(study,'shape_extension_finished',outcomes=outcomes,cpu_hours=updated['cpu_hours'])
    write(study/'extension_handoff.json',dict(extension=str(output.resolve()),completion_sha256=digest(output/'completion.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['register','run'],required=True)
    parser.add_argument('--study',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.phase=='register':register(args.study,args.output)
    else:run(args.output)


if __name__=='__main__':main()
