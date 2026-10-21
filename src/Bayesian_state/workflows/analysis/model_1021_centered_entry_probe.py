"""One-mechanism S205 diagnostic: center online support before admitting new rules.

Use --plan RETENTION_PLAN --output NEW; --smoke runs two short paths and filters.
All behavioral parameters stay at the original COMPACT nominee. The primary
worker supplies every cognitive operation; an isolated adapter changes the
support reference for entry. Original 128 generation and eight training seeds
are retained. At most 16 processes share the established recovery resource gate.
Fixed-parameter effects diagnose a mechanism; a final comparison needs refitting.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('centered_entry_driver',
    ROOT/'src/Bayesian_state/workflows/analysis/model_1021_simplification.py')
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)

ENTRY = str(Path(__file__).resolve().relative_to(driver.ROOT))
HELPER = 'src/Bayesian_state/model/modules/hypothesis_transition/centered_evidence.py'
read, write, digest = driver.io.read, driver.io.write, driver.io.digest


def worker(file: Path) -> None:
    original_load = driver.base.load
    audits = []

    def load(name, path):
        module = original_load(name, path)
        if path == driver.COMPACT:
            helper = original_load('src.Bayesian_state.model.modules.hypothesis_transition.centered_evidence', HELPER)
            original_context = module.compact_evidence

            @contextmanager
            def intervention(cls):
                with original_context(cls) as base_audit, helper.centered_evidence(cls) as audit:
                    audits.append(audit)
                    yield base_audit

            module.compact_evidence = intervention
        return module

    driver.base.load = load
    try:
        driver.worker(file)
    finally:
        driver.base.load = original_load
    if len(audits) != 1 or audits[0]['centered_feedback_updates'] <= 0:
        raise ValueError('Centering adapter did not execute')
    write(Path(read(file)['target']).with_suffix('.centered.json'), audits[0])


def run(plan_file: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.workflows.analysis.model_1021_simplification_retention_probe import resource_gate
    from src.Bayesian_state.workflows.analysis.model_0923_h_training_audit import combine
    if output.exists():
        raise FileExistsError('Use a new centered-entry diagnostic directory')
    plan = read(plan_file)
    workers = resource_gate(plan)
    fit, generation = Path(plan['fit']), Path(plan['generation_source'])
    if digest(fit/'nominees.json') != plan['nomination_sha256']:
        raise ValueError('Primary nominee changed')
    frozen = read(fit/'nominees.json')
    nominee, = [r for r in frozen['nominees'] if (r['subject'], r['model']) == (205, 'COMPACT')]
    key = nominee['candidate']['id']
    hashes = dict(frozen['source_sha256'])
    for path, sha in hashes.items():
        if digest(driver.ROOT/path) != sha:
            raise ValueError('Frozen scientific source changed: '+path)
    hashes.update({p: digest(driver.ROOT/p) for p in (ENTRY, HELPER)})
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(fit=str(fit.resolve()), smoke=smoke, workers=1 if smoke else workers,
        source_sha256=hashes, nomination_sha256=plan['nomination_sha256'],
        intervention='base-prior-weighted mean exponential active support centered to one before entry',
        fixed_candidate=nominee['candidate'], reason='S205 target removals occur before precision has developed',
        scope='Development fixed-parameter mechanism diagnostic; original primary nominees unchanged',
        oral_used_for_selection=False, plan_sha256=digest(plan_file)))
    jobs, groups, aggregate = [], {}, {}
    templates = sorted((fit/'jobs').glob(f'confirmation_S205_COMPACT_{key}_C*.json'))
    if len(templates) != 8:
        raise ValueError('Need eight original training confirmation jobs')
    groups['training'] = []
    for i, file in enumerate(templates[:2] if smoke else templates):
        job = read(file)
        if job['phase'] != 'training' or job['particles'] != 512:
            raise ValueError('Require original training likelihood and particle budget')
        job.update(source_sha256=hashes, smoke=smoke, particles=2 if smoke else 512,
                   name=f'centered_training_C{i:03d}',
                   target=str((output/'training/chunks'/f'{key}_C{i:03d}').resolve()))
        groups['training'].append(job)
        jobs.append(job)
    aggregate['training'] = dict(groups['training'][0], target=str((output/'training'/key).resolve()),
                                 seeds=[s for j in groups['training'] for s in j['seeds']])
    templates = sorted((generation/'jobs').glob(f'*S205_COMPACT_{key}_C*.json'))
    if len(templates) != 32:
        raise ValueError('Need the original 128-path generation bank')
    groups['autonomous'] = []
    for i, file in enumerate(templates[:1] if smoke else templates):
        job = read(file)
        job.update(source_sha256=hashes, smoke=smoke, name=f'centered_autonomous_C{i:03d}',
                   target=str((output/'autonomous/chunks'/f'{key}_C{i:03d}').resolve()))
        if smoke:
            job['seeds'] = job['seeds'][:2]
        groups['autonomous'].append(job)
        jobs.append(job)
    aggregate['autonomous'] = dict(groups['autonomous'][0], target=str((output/'autonomous'/key).resolve()),
                                  seeds=[s for j in groups['autonomous'] for s in j['seeds']])
    write(output/'jobs_plan.json', dict(jobs=jobs))
    driver.ENTRY = ENTRY
    driver.execute(jobs, output, 1 if smoke else workers)
    records = {}
    for role, chunks in groups.items():
        job = aggregate[role]
        Path(job['target']).parent.mkdir(parents=True, exist_ok=True)
        records[role] = (combine(job, chunks) if role == 'training'
                        else driver.io.combine_evaluation_chunks(job, chunks, True))
    write(output/'completion.json', dict(status='complete', smoke=smoke, records=records,
        primary_nominee_unchanged=True, additional_parameters=0,
        scope='Fixed-parameter development diagnostic; no refit or oral evaluation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.plan and args.output:
        run(args.plan, args.output, args.smoke)
    else:
        parser.error('Supply --plan and --output, or --worker')
