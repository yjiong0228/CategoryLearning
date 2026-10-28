"""Use a remaining precision-study budget for a completed repetition comparison.

Internal continuation after the registered adaptive controller has stopped.
The separately registered fallback adds one block of independent systems at
the highest completed resolution that fits the original resource budget.
These results diagnose particle-versus-repetition allocation; they do not
create independent-confirmation acceptance records.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from . import model_1024_state_precision as precision

read, write, digest = precision.read, precision.write, precision.digest


def choose_plan(output: Path, registration: dict, rows: list[dict], now: float) -> tuple[list[dict], list[dict]]:
    protocol=read(output/'protocol.json'); policy=protocol['policy']; budget=policy['budget']
    protocol['resource_policy']=read(output/'resource_policy_v2.json')
    ledger=[read(f) for f in (output/'ledger').glob('*.json')]
    remaining_wall=read(output/'run_clock.json')['started_unix']+budget['wall_hours']*3600-now
    remaining_cpu=budget['cpu_hours']*3600-sum(r['cpu_seconds'] for r in ledger)
    case_map={c['id']:c for c in protocol['cases']}
    accepted={read(f)['case'] for f in (output/'confirmations').glob('*.json') if read(f)['accepted']}
    plan=[]; evidence=[]
    for cid in registration['cases']:
        if cid in accepted:
            continue
        candidates=sorted([r for r in rows if r['case']==cid and not r['confirmation']
                           and r['particles'] in registration['particles']
                           and r['systems']>=policy['systems'] and r['systems']<policy['maximum_systems']],
                          key=lambda r:r['particles'],reverse=True)
        for row in candidates:
            n=row['particles']; extra=min(registration['additional_systems'],policy['maximum_systems']-row['systems'])
            job=precision.study.make_job(case_map[cid],policy,output,f'precision_n{n}',n,row['systems'])
            est=precision.estimate(job,protocol,ledger,64)
            fit=est['wall']<remaining_wall and extra*est['cpu']<remaining_cpu
            evidence.append(dict(case=cid,particles=n,existing_systems=row['systems'],additional_systems=extra,
                estimated_job_wall_seconds=est['wall'],estimated_total_cpu_seconds=extra*est['cpu'],
                remaining_wall_seconds=remaining_wall,remaining_cpu_seconds=remaining_cpu,feasible=fit))
            if fit:
                plan.append(dict(cases=[cid],particles=[n],systems=row['systems']+extra,confirmation=False,
                    reason='budget_repetition_diagnostic'))
                remaining_cpu-=extra*est['cpu']
                break
    return plan,evidence


def run(output: Path) -> None:
    registration=read(output/'budget_completion_registration.json')
    if digest(Path(__file__))!=registration['source_sha256']:
        raise ValueError('Registered budget-continuation source changed')
    while not (output/'adaptive_completion.json').exists():
        time.sleep(10)
    previous=read(output/'adaptive_completion.json')
    label='budget_repetition_01'; scope=previous['scope']
    rows=read(Path(previous['latest_summary']))
    if previous['reason']=='remaining_budget_or_memory_insufficient':
        plan,evidence=choose_plan(output,registration,rows,time.time())
    else:
        plan,evidence=[],[]
    write(output/'decisions'/(label+'.json'),dict(scope=scope,plan=plan,evidence=evidence,
        previous_completion_sha256=digest(output/'adaptive_completion.json'),decided_unix=time.time(),
        purpose='One extra repetition comparison within the original 12-hour/1536-CPU-hour budget; no new acceptance gate'))
    if plan:
        write(output/'plans'/(label+'.json'),plan)
        precision.run(output,[],[],0,label,plan=plan)
        precision.summarize(output,label,scope)
        latest=output/'summaries'/label/'diagnostics.json'
        reason=read(output/'completions'/(label+'.json'))['stop_reason']
    else:
        latest=Path(previous['latest_summary']); reason=previous['reason']
    write(output/'precision_completion.json',dict(latest_summary=str(latest),scope=scope,
        stable_cases=previous['stable_cases'],expanded=previous['expanded'],reason=reason,
        additional_repetition_diagnostic=bool(plan),finished_unix=time.time(),
        independent_confirmation_records_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
