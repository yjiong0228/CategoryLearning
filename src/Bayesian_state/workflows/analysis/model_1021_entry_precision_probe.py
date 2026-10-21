"""Finish the S205 entry diagnostic with current-precision transfer and its combination.

Use --source COMPLETED_CENTERED_PROBE --recovery-rescue RESCUE --output NEW.
Both controls use the same fixed parameters, 128 paths and eight 512-particle
training filters. The original workers supply all cognitive calculations.
Launch after all four rescue screens complete: at most 64 recovery filters,
eight remaining primary filters and 56 diagnostic workers share 128 CPUs.
--smoke checks two short filters and two short paths per control in one worker.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('precision_entry_driver',
    ROOT/'src/Bayesian_state/workflows/analysis/model_1021_simplification.py')
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PRECISION = 'src/Bayesian_state/model/modules/hypothesis_transition/entry_precision.py'
CENTER = 'src/Bayesian_state/model/modules/hypothesis_transition/centered_evidence.py'
read, write, digest = driver.io.read, driver.io.write, driver.io.digest
CONTROLS = ('PRECISION', 'CENTERED_PRECISION')


def worker(file: Path) -> None:
    original_load = driver.base.load
    control = read(file)['entry_control']
    if control not in CONTROLS:
        raise ValueError('Unknown entry control')
    audits = []

    def load(name, path):
        module = original_load(name, path)
        if path == driver.COMPACT:
            precision = original_load('src.Bayesian_state.model.modules.hypothesis_transition.entry_precision', PRECISION)
            center = original_load('src.Bayesian_state.model.modules.hypothesis_transition.centered_evidence', CENTER)
            original_context = module.compact_evidence

            @contextmanager
            def intervention(cls):
                with ExitStack() as stack:
                    base_audit = stack.enter_context(original_context(cls))
                    if control == 'CENTERED_PRECISION':
                        stack.enter_context(center.centered_evidence(cls))
                    audits.append(stack.enter_context(precision.entry_precision(cls)))
                    yield base_audit

            module.compact_evidence = intervention
        return module

    driver.base.load = load
    try:
        driver.worker(file)
    finally:
        driver.base.load = original_load
    if len(audits) != 1:
        raise ValueError('Entry precision adapter did not execute')
    write(Path(read(file)['target']).with_suffix('.entry_precision.json'), dict(control=control, **audits[0]))


def run(source: Path, rescue: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.workflows.analysis.model_0923_h_training_audit import combine
    if output.exists():
        raise FileExistsError('Use a new entry-precision diagnostic directory')
    if read(source/'completion.json')['status'] != 'complete' or read(source/'completion.json')['smoke']:
        raise ValueError('Complete the original scientific centering probe first')
    cases = sorted((rescue/'fit').glob('S*'))
    if len(cases) != 4 or any(read(p/'screen_completion.json')['status'] != 'complete' for p in cases):
        raise ValueError('All four recovery rescue screens must finish first')
    for case in cases:
        settings = read(case/'protocol.json')['settings']
        if (settings['selection_candidates']*settings['budgets']['selection']['seeds'] > 16
                or settings['confirmation_candidates']*settings['budgets']['confirmation']['seeds'] > 16):
            raise ValueError('Remaining recovery work exceeds the shared CPU allowance')
    parent = Path(read(rescue/'protocol.json')['parent'])
    if read(rescue/'protocol.json')['remaining_primary_filters_upper_bound'] != 8:
        raise ValueError('Primary evaluation resource gate differs')
    previous = read(source/'protocol.json')
    fit = Path(previous['fit'])
    if digest(fit/'nominees.json') != previous['nomination_sha256']:
        raise ValueError('Frozen primary nominee changed')
    hashes = dict(read(fit/'nominees.json')['source_sha256'])
    for path, sha in hashes.items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen scientific source changed')
    hashes.update({p: digest(ROOT/p) for p in (ENTRY, PRECISION, CENTER)})
    original = read(source/'jobs_plan.json')['jobs']
    if len(original) != 40:
        raise ValueError('Require eight training filters and 32 generation chunks')
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(fit=str(fit.resolve()), smoke=smoke, workers=1 if smoke else 56,
        source_sha256=hashes, parent_evaluation=str(parent), source=str(source.resolve()),
        source_protocol_sha256=digest(source/'protocol.json'), nomination_sha256=previous['nomination_sha256'],
        scope='Final two fixed-parameter S205 entry controls; no refitting or oral selection'))
    jobs, groups = [], {}
    for control in CONTROLS:
        own = output/control
        own.mkdir()
        write(own/'protocol.json', dict(fit=str(fit.resolve()), comparison_label=control, smoke=smoke,
            nomination_sha256=previous['nomination_sha256'], source_sha256=hashes))
        for phase in ('training', 'autonomous'):
            templates = [j for j in original if j['phase'] == phase]
            selected = templates[:(2 if phase == 'training' else 1)] if smoke else templates
            groups[control, phase] = []
            for i, template in enumerate(selected):
                job = deepcopy(template)
                job.update(name=control+'_'+job['name'], entry_control=control, smoke=smoke,
                    particles=2 if smoke else job['particles'], source_sha256=hashes,
                    target=str((own/phase/'chunks'/f"{job['candidate']['id']}_C{i:03d}").resolve()))
                if smoke and phase == 'autonomous':
                    job['seeds'] = job['seeds'][:2]
                groups[control, phase].append(job)
                jobs.append(job)
    write(output/'jobs_plan.json', dict(jobs=jobs))
    driver.ENTRY = ENTRY
    driver.execute(jobs, output, 1 if smoke else 56)
    records = {c: {} for c in CONTROLS}
    for (control, phase), chunks in groups.items():
        job = dict(chunks[0], seeds=[s for c in chunks for s in c['seeds']],
            target=str((output/control/phase/chunks[0]['candidate']['id']).resolve()))
        records[control][phase] = (combine(job, chunks) if phase == 'training'
                                   else driver.io.combine_evaluation_chunks(job, chunks, True))
    for control in CONTROLS:
        write(output/control/'completion.json', dict(status='complete', smoke=smoke, records=records[control],
            primary_nominee_unchanged=True, additional_parameters=0))
    write(output/'completion.json', dict(status='complete', smoke=smoke, controls=list(CONTROLS),
        primary_nominee_unchanged=True, additional_parameters=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'recovery-rescue', 'output', 'worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.source and args.recovery_rescue and args.output:
        run(args.source, args.recovery_rescue, args.output, args.smoke)
    else:
        parser.error('Supply --source, --recovery-rescue and --output, or --worker')
