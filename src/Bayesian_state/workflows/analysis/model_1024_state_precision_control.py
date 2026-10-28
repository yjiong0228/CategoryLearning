"""Execute the registered adaptive precision study and preserve every decision.

Start after launching the pilot. This supervisor waits for its completion,
then queues independent confirmation and resolution increases together.
A confirmation is always a fresh block of 16 seeds; previously inspected
confirmation blocks are retained but do not decide a later acceptance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import model_1024_state_precision as precision
from . import model_1024_individual_state_summary as summary

read, write, digest, ROOT = precision.read, precision.write, precision.digest, precision.ROOT
MODULE = 'src.Bayesian_state.workflows.analysis.model_1024_state_precision'


def invoke(output: Path, phase: str, label: str, extra: list[str]) -> None:
    log = output/'control_logs'/f'{label}_{phase}.log'; log.parent.mkdir(exist_ok=True)
    with log.open('x') as stream:
        subprocess.run([sys.executable, '-m', MODULE, '--phase', phase,
            '--output', str(output), '--label', label, *extra], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def check_confirmations(output: Path, rows: list[dict], requests: list[dict], label: str) -> None:
    p = read(output/'protocol.json')['policy']
    for request in requests:
        if not request.get('confirmation'):
            continue
        for cid in request['cases']:
            n = request['particles'][0]; r = request['primary_systems']
            identity = f'{cid}_n{n}_r{r}'
            target = output/'confirmations'/(identity+'.json')
            if target.exists():
                continue
            base = next(x for x in rows if x['case']==cid and x['particles']==n
                        and not x['confirmation'] and x['systems']==r)
            start = request['systems']-p['confirmation_systems']
            jobs = []
            for index in range(start, request['systems']):
                file = output/'jobs'/f'confirm_n{n}_{cid}_{index:03d}.json'
                ledger = output/'ledger'/file.name
                if ledger.exists() and read(ledger)['status']=='complete':
                    jobs.append(read(file))
            record = dict(case=cid, particles=n, primary_systems=r, requested_indices=[start,request['systems']],
                          completed_systems=len(jobs), summary_label=label, accepted=False)
            if len(jobs) >= p['minimum_diagnostic_systems']:
                pooled, row = summary.summarize_bank(jobs)
                with np.load(base['pooled_file']) as z:
                    reference = {k:z[k] for k in z.files}
                for mode in ('filtered', 'full'):
                    difference = summary.discrepancy(pooled, reference, mode)
                    record[mode+'_repeat_pass'] = precision.bank_pass(row, p, mode)
                    record[mode+'_difference'] = difference
                    record[mode+'_consistent'] = (precision.bank_pass(row, p, mode)
                        and precision.difference_pass(difference, p))
                record['accepted'] = bool(len(jobs)==p['confirmation_systems']
                    and record['full_consistent'] and base['full_numerically_consistent'])
                file = output/'confirmations'/(identity+'.npz'); file.parent.mkdir(exist_ok=True)
                with file.open('xb') as stream:
                    np.savez_compressed(stream, **pooled)
                record.update(pooled_file=str(file.resolve()), pooled_sha256=digest(file), diagnostics=row)
            write(target, record)


def actions(output: Path, rows: list[dict], scope: list[str], starting_pair: list[int]) -> tuple[list[dict], list[str]]:
    p = read(output/'protocol.json')['policy']; levels=p['initial_particles']+p['escalation_particles']
    confirmations=[read(f) for f in (output/'confirmations').glob('*.json')]
    stable=sorted({c['case'] for c in confirmations if c['accepted']})
    specs=[]
    for cid in scope:
        if cid in stable:
            continue
        own=[r for r in rows if r['case']==cid and not r['confirmation'] and r['particles'] in levels]
        if not own:
            specs.append(dict(cases=[cid], particles=starting_pair, systems=8, confirmation=False,
                              reason='expansion_first_comparison'))
            continue
        base=max(own,key=lambda r:r['particles']); n=base['particles']; r=base['systems']
        if r < p['systems']:
            specs.append(dict(cases=[cid], particles=sorted({x['particles'] for x in own}),
                              systems=p['systems'], confirmation=False, reason='complete_expansion_banks'))
            continue
        attempted=next((c for c in confirmations if c['case']==cid and c['particles']==n
                        and c['primary_systems']==r),None)
        if base.get('full_numerically_consistent') and attempted is None:
            completed=max([x['systems'] for x in rows if x['case']==cid and x['particles']==n
                           and x['confirmation']] or [0])
            specs.append(dict(cases=[cid], particles=[n], systems=completed+p['confirmation_systems'],
                confirmation=True, primary_systems=r, reason='fresh_independent_confirmation'))
        elif n < max(levels):
            specs.append(dict(cases=[cid], particles=[levels[levels.index(n)+1]], systems=p['systems'],
                              confirmation=False, reason='increase_particles'))
        elif r < p['maximum_systems']:
            specs.append(dict(cases=[cid], particles=[n], systems=min(r*2,p['maximum_systems']),
                              confirmation=False, reason='increase_independent_systems_at_particle_cap'))
    return specs, stable


def run(output: Path) -> None:
    protocol=read(output/'protocol.json'); p=protocol['policy']
    while not (output/'completions'/'pilot.json').exists():
        time.sleep(10)
    scope=list(p['pilot_cases']); starting_pair=list(p['initial_particles']); expanded=False
    previous_requests=[]; latest='pilot'; round_index=0
    while True:
        summary_path=output/'summaries'/latest/'diagnostics.json'
        if not summary_path.exists():
            invoke(output,'summarize',latest,['--cases',*scope])
        rows=read(summary_path)
        check_confirmations(output,rows,previous_requests,latest)
        specs, stable=actions(output,rows,scope,starting_pair)
        if not expanded and precision.may_expand(stable,p):
            accepted=[read(f) for f in (output/'confirmations').glob('*.json') if read(f)['accepted']]
            high=max(c['particles'] for c in accepted); levels=p['initial_particles']+p['escalation_particles']
            starting_pair=[max(2048,high//2),high]
            starting_pair=sorted(set(starting_pair))
            if len(starting_pair)<2:
                starting_pair=p['initial_particles']
            scope=[c['id'] for c in protocol['cases']]; expanded=True
            specs, stable=actions(output,rows,scope,starting_pair)
        completion=read(output/'completions'/(latest+'.json'))
        round_index+=1; label=f'adaptive_{round_index:02d}'
        decision=dict(after_batch=latest, summary=str(summary_path), stable_cases=stable,
                      expansion_authorized=expanded, scope=scope, plan=specs, decided_unix=time.time(),
                      stop_reason=completion['stop_reason'])
        write(output/'decisions'/(label+'.json'),decision)
        if completion['stop_reason']!='batch_complete' or not specs:
            write(output/'adaptive_completion.json',dict(latest_summary=str(summary_path),
                scope=scope, stable_cases=stable, expanded=expanded,
                reason=completion['stop_reason'] if specs else 'registered_actions_exhausted',
                finished_unix=time.time()))
            return
        plan_file=output/'plans'/(label+'.json');write(plan_file,specs)
        invoke(output,'run',label,['--plan',str(plan_file)])
        latest=label; previous_requests=specs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
