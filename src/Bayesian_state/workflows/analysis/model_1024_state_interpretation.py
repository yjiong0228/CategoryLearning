"""Bounded follow-up of retained Model 1024 parameter candidates.

Reuse the validated precision scheduler and passive observer. This controller
registers a separate budget, checks each state quantity, and permits one 8192
particle follow-up only for the prespecified S205 focus cases.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

from . import model_1024_state_precision as precision
from .model_1024_individual_state_summary import TARGETS

ROOT, read, write, digest = precision.ROOT, precision.read, precision.write, precision.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
MEASURES = (*TARGETS, 'belief', 'execution')


def measure_checks(row: dict, policy: dict, mode: str, measure: str) -> dict:
    """Keep quantity-specific agreement separate from completion and weights."""
    t = policy['numerical_checks']
    prefix = mode+'_'
    complete = bool(row.get('bank_complete', False) and row['systems'] >= policy['systems'])
    weights = (row.get(prefix+'ess_min', 0) >= t['minimum_system_ess']
               and row.get(prefix+'maximum_weight', 1) <= t['maximum_system_weight'])
    half = (row.get(prefix+'half_'+measure+'_mean', 1) <= t['half_mean_absolute_difference']
            and row.get(prefix+'half_'+measure+'_q95', 1) <= t['half_q95_absolute_difference'])
    resolution = (row.get(prefix+'comparison_'+measure+'_mean', 1) <= t['resolution_mean_absolute_difference']
                  and row.get(prefix+'comparison_'+measure+'_q95', 1) <= t['resolution_q95_absolute_difference'])
    return dict(complete=complete, weights=bool(weights), repeat=bool(half),
                resolution=bool(resolution), usable=bool(complete and weights and half and resolution))


def escalation_cases(rows: list[dict], policy: dict) -> list[str]:
    """Choose the registered focus points by numerical checks, never fit quality."""
    return sorted({r['case'] for r in rows
        if r['subject'] in policy['interpretation']['escalation_subjects']
        and r['case'] in policy['pilot_cases'] and r['particles'] == 4096
        and not r.get('confirmation', False)
        and any(not measure_checks(r, policy, 'full', field)['usable'] for field in MEASURES)})


def prepare(config: Path, output: Path) -> None:
    precision.prepare(config, output)
    p = read(output/'protocol.json')['policy']
    retained = {c['id'] for c in read(output/'protocol.json')['cases'] if c['parameter_status']=='retained'}
    rows = read(ROOT/p['precision_summary']/'diagnostics.json')
    anchors = {}
    for row in rows:
        cid = row['case']
        if cid not in retained or cid in p['pilot_cases'] or row['confirmation']:
            continue
        if cid not in anchors or row['particles'] > anchors[cid]['particles']:
            anchors[cid] = row
    if set(anchors) | set(p['pilot_cases']) != retained or set(anchors) & set(p['pilot_cases']):
        raise ValueError('The retained set must be covered exactly once')
    inputs = {}
    for path in (ROOT/p['precision_summary']/'diagnostics.json', ROOT/p['source_summary']/'selected.json'):
        inputs[str(path)] = digest(path)
    # Freeze all pooled references, including lower resolutions and independent
    # confirmations, to preserve the basis of the inherited diagnostic rows.
    for row in rows:
        file = Path(row['pooled_file'])
        if digest(file) != row['pooled_sha256']:
            raise ValueError('Previous pooled state changed')
        inputs[str(file)] = row['pooled_sha256']
    for row in read(ROOT/p['source_summary']/'selected.json'):
        if row['subject'] in p['interpretation']['include_diagnostic_subjects']:
            inputs[row['file']] = digest(Path(row['file']))
    write(output/'interpretation_registration.json', dict(
        anchors=anchors, inputs=inputs, source={ENTRY: digest(ROOT/ENTRY)},
        registered_unix=time.time(), new_jobs_initial=len(p['pilot_cases'])*len(p['initial_particles'])*p['systems'],
        new_jobs_maximum=(len(p['pilot_cases'])*len(p['initial_particles'])+sum(
            c.startswith('S205_') for c in p['pilot_cases']))*p['systems'],
        source_policy='Reuse archived estimates; no parameter averaging or oral-data changes',
        supplementary_diagnostics='Inherited precision-distribution limitations remain in scope'))
    target = output/'source'/ENTRY; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/ENTRY, target)


def verify(output: Path) -> None:
    protocol = read(output/'protocol.json'); own = read(output/'interpretation_registration.json')
    for name, expected in dict(protocol['inputs'], **own['inputs'], **own['source']).items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Registered input/source changed: {name}')


def run(output: Path) -> None:
    verify(output)
    p = read(output/'protocol.json')['policy']
    precision.run(output, p['pilot_cases'], p['initial_particles'], p['systems'], 'initial')
    if not (output/'summaries'/'initial'/'diagnostics.json').exists():
        precision.summarize(output, 'initial', p['pilot_cases'])
    rows = read(output/'summaries'/'initial'/'diagnostics.json')
    ids = escalation_cases(rows, p)
    initial = read(output/'completions'/'initial.json')
    eligible = initial['stop_reason']=='batch_complete'
    write(output/'interpretation_decision.json', dict(cases=ids, eligible=eligible,
        reason=p['interpretation']['escalation_reason'],
        maximum_rounds=p['interpretation']['escalation_maximum_rounds'],
        initial_stop=initial['stop_reason']))
    label = 'initial'
    if ids and eligible:
        precision.run(output, ids, p['escalation_particles'], p['systems'], 'focus_followup')
        if not (output/'summaries'/'focus_followup'/'diagnostics.json').exists():
            precision.summarize(output, 'focus_followup', p['pilot_cases'])
        label = 'focus_followup'
    verify(output)
    ledger = [read(f) for f in (output/'ledger').glob('*.json')]
    completions = [read(f) for f in (output/'completions').glob('*.json')]
    write(output/'calculation_completion.json', dict(status='bounded_calculation_complete',
        latest_summary=str(output/'summaries'/label), completed_unix=time.time(),
        jobs_complete=sum(r['status']=='complete' for r in ledger),
        jobs_failed=sum(r['status']=='failed' for r in ledger),
        jobs_terminated=sum(r['status']=='budget_terminated' for r in ledger),
        cpu_hours=(sum(r['cpu_seconds'] for r in ledger)
                   +sum(r['supervisor_cpu_seconds'] for r in completions))/3600,
        wall_hours=(time.time()-read(output/'run_clock.json')['started_unix'])/3600,
        peak_workers=max(r['peak_workers'] for r in completions),
        reports_complete=False, note='Numerical and psychological interpretation are reported separately.'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run'], required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/shared/model1024_state_interpretation.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase=='prepare':
        prepare(args.config, args.output)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
