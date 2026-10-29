"""Supplementary belief/precision checks from already computed state arrays.

This adds no particle filtering and does not modify the six registered adaptive
gates. Run --phase cache after the particle queue finishes, then --phase summarize
with the final interpretation analysis. Existing anchor caches remain read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

from . import model_1024_state_precision_distributions as dist
from .model_1024_state_interpretation import read, write, digest, verify
from .model_1024_state_interpretation_summary import candidate_difference, sensitivity_status, calculation_record


def cache(source: Path, output: Path) -> None:
    verify(source)
    if calculation_record(source) is None:
        raise ValueError('Finish the particle queue before supplementary parallel analysis')
    registration=read(source/'supplemental_distribution_registration.json')
    if any(digest(Path(k))!=v for k,v in registration['sources'].items()):
        raise ValueError('Distribution analysis helpers changed')
    jobs=[]
    for file in sorted((source/'jobs').glob('*.json')):
        ledger=source/'ledger'/file.name
        if ledger.exists() and read(ledger)['status']=='complete':
            jobs.append(read(file))
    output.mkdir(parents=True,exist_ok=True)
    plan=output/'registration.json'
    if not plan.exists():
        write(plan,dict(registration=registration,jobs=[j['name'] for j in jobs],
            source_sha256=digest(Path(__file__)),created_unix=time.time()))
    else:
        if read(plan)['jobs']!=[j['name'] for j in jobs]:
            raise ValueError('Registered completed-job scope changed')
    started=time.time()
    workers=min(registration['cache_workers'],len(os.sched_getaffinity(0)),len(jobs))
    with ProcessPoolExecutor(max_workers=max(1,workers)) as pool:
        for i,_ in enumerate(pool.map(dist.cache_one,[(j,output) for j in jobs]),1):
            if i%32==0 or i==len(jobs):
                print(f'Cached {i}/{len(jobs)} supplementary state distributions',flush=True)
    rows=[read(output/'jobs'/(j['name']+'.json')) for j in jobs]
    write(output/'cache_completion.json',dict(jobs=len(jobs),workers=workers,
        cpu_hours=sum(r['cpu_seconds'] for r in rows)/3600,wall_seconds=time.time()-started))


def summarize(source: Path, output: Path, analysis: Path) -> None:
    p=read(source/'protocol.json')['policy']
    if not (output/'summaries'/'final'/'diagnostics.json').exists():
        dist.summarize(source,output,'final',p['pilot_cases'])
    old=Path(p['precision_source'])/'distribution_diagnostics'/'summaries'/'final'/'diagnostics.json'
    new=output/'summaries'/'final'/'diagnostics.json'
    rows=read(old)+read(new)
    own=read(analysis/'selected.json')
    primary=pd.read_csv(analysis/'numerical_checks.csv')
    selected=[]; checks=[]; arrays={}; usable={}
    for case in own:
        if case['parameter_status']!='retained':
            continue
        match=[r for r in rows if r['case']==case['case'] and r['particles']==case['particles']
               and r['systems']==case['systems'] and not r['confirmation']]
        if len(match)!=1:
            raise ValueError('Missing or ambiguous selected distribution bank')
        r=match[0];file=Path(r['pooled_file'])
        if digest(file)!=r['pooled_sha256']:
            raise ValueError('Distribution summary changed')
        with np.load(file) as z:
            a={mode+'_'+field:z[mode+'_'+field] for mode in ('filtered','full') for field in dist.FIELDS}
        arrays[case['case']]=a
        selected.append(dict(case=case['case'],subject=case['subject'],number=case['number'],
            file=str(file),sha256=r['pooled_sha256'],particles=case['particles'],systems=case['systems']))
        for mode in ('filtered','full'):
            base=primary.loc[primary.case.eq(case['case'])&primary['mode'].eq(mode)].iloc[0]
            for field in dist.FIELDS:
                values={f'{kind}_{stat}':float(r.get(f'{kind}_{mode}_{field}_{stat}',np.nan))
                        for kind in ('half','comparison') for stat in ('mean','q95')}
                limits=p['numerical_checks']
                ok=bool(base.complete and base.weights
                        and values['half_mean']<=limits['half_mean_absolute_difference']
                        and values['half_q95']<=limits['half_q95_absolute_difference']
                        and values['comparison_mean']<=limits['resolution_mean_absolute_difference']
                        and values['comparison_q95']<=limits['resolution_q95_absolute_difference'])
                usable[case['case'],mode,field]=ok
                checks.append(dict(case=case['case'],subject=case['subject'],number=case['number'],
                    mode=mode,measure=field,particles=case['particles'],systems=case['systems'],
                    comparison_particles=r.get('comparison_particles'),comparison_systems=r.get('comparison_systems'),
                    complete=bool(base.complete),weights=bool(base.weights),usable=ok,**values))
    sensitivity=[]
    for sid in sorted({s['subject'] for s in selected}):
        cases=[s for s in selected if s['subject']==sid]
        for mode in ('filtered','full'):
            for field in dist.FIELDS:
                values=np.stack([arrays[c['case']][mode+'_'+field] for c in cases])
                d=candidate_difference(values,True)
                mean=float(d.mean()) if d is not None else np.nan
                q95=float(np.quantile(d,.95)) if d is not None else np.nan
                reliable=all(usable[c['case'],mode,field] for c in cases)
                sensitivity.append(dict(subject=sid,mode=mode,measure=field,candidates=len(cases),
                    numerically_usable_candidates=sum(usable[c['case'],mode,field] for c in cases),
                    mean_difference=mean,q95_difference=q95,
                    status=sensitivity_status(len(cases),reliable,mean,q95,p)))
    destination=output/'interpretation'
    destination.mkdir(exist_ok=False)
    write(destination/'selected.json',selected)
    pd.DataFrame(checks).to_csv(destination/'numerical_checks.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(destination/'parameter_sensitivity.csv',index=False)
    write(destination/'manifest.json',dict(source=str(source),analysis=str(analysis),
        sources={str(old):digest(old),str(new):digest(new),str(Path(__file__)):digest(Path(__file__))},
        role='Supplementary checks only; primary six-quantity scope and escalation unchanged',
        artifacts={f.name:digest(f) for f in destination.iterdir() if f.is_file()},created_unix=time.time()))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['cache','summarize'],required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--analysis',type=Path)
    args=parser.parse_args()
    if args.phase=='cache':
        cache(args.source,args.output)
    else:
        if args.analysis is None:
            parser.error('--analysis required for summarize')
        summarize(args.source,args.output,args.analysis)


if __name__=='__main__':
    main()
