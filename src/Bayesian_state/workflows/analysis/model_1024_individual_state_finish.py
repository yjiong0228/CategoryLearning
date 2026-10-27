"""Finish the registered state study after its initial queue completes.

Numerical checks select at most one extension. Each subsequent analysis writes
a new directory and records CPU/wall use and source hashes. The final PNGs and
prose still require visual and scientific review before handoff.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from .model_1024_individual_states import ROOT, read, write, digest


def finish(source: Path, midpoint: Path, relative: Path) -> None:
    base = source.parent; ledger = base/'analysis_ledger'; ledger.mkdir(exist_ok=True)
    while not (source/'initial_completion.json').exists():
        time.sleep(10)
    initial = read(source/'initial_completion.json')
    if initial['failures']:
        raise RuntimeError('Initial workers failed; inspect preserved logs before postprocessing')
    env = dict(os.environ, MPLCONFIGDIR='/tmp/model1024_states_mpl', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')

    def phase(name, module, arguments):
        record = ledger/(name+'.json')
        if record.exists():
            if read(record)['returncode'] != 0:
                raise RuntimeError(f'Previous {name} failed; retain its output and choose a new destination')
            return
        start = time.time(); before = resource.getrusage(resource.RUSAGE_CHILDREN)
        file = ROOT/(module.replace('.', '/')+'.py')
        with (ledger/(name+'.log')).open('x') as stream:
            result = subprocess.run([sys.executable, '-m', module, *map(str, arguments)],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        write(record, dict(phase=name, module=module, arguments=list(map(str, arguments)),
            returncode=result.returncode, started_unix=start, wall_seconds=time.time()-start,
            cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            source_sha256=digest(file), includes_numerical_worker_usage=name=='extension'))
        print(f'{name}: return code {result.returncode}', flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {ledger/name}.log')

    analysis = 'src.Bayesian_state.workflows.analysis.'
    first = base/'summary_initial'
    phase('summary_initial', analysis+'model_1024_individual_state_summary', ['--source', source, '--output', first])
    ids = read(first/'extension_candidates.json')
    protocol = read(source/'protocol.json'); started = read(source/'run_clock.json')['started_unix']
    remaining = started+protocol['policy']['budget']['wall_hours']*3600-time.time()
    if ids and remaining >= 1800:
        phase('extension', analysis+'model_1024_individual_states',
            ['--phase', 'run', '--output', source, '--extension', first/'extension_candidates.json'])
    else:
        write(ledger/'extension_decision.json', dict(cases=ids, remaining_seconds=remaining,
            decision='not_required' if not ids else 'remaining_wall_budget_below_30_minutes'))
    final = base/'summary_final'
    phase('summary_final', analysis+'model_1024_individual_state_summary', ['--source', source, '--output', final])
    oral = base/'oral_alignment'; distributions = base/'distributions'
    phase('oral_alignment', analysis+'model_1024_individual_oral_alignment',
        ['--source', source, '--summary', final, '--midpoint', midpoint, '--relative', relative, '--output', oral])
    phase('distributions', analysis+'model_1024_individual_state_distributions',
        ['--source', source, '--summary', final, '--output', distributions])
    phase('report', 'src.Bayesian_state.workflows.reports.model_1024_individual_states_report',
        ['--source', source, '--summary', final, '--oral', oral, '--distributions', distributions,
         '--output', base/'report_v1'])
    write(base/'pipeline_completion.json', dict(status='computationally_complete_pending_review',
        finished_unix=time.time(), report=str((base/'report_v1/report.md').resolve())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'midpoint', 'relative'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); finish(args.source, args.midpoint, args.relative)
