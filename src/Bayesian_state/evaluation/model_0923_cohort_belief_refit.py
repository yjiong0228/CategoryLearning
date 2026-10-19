"""Summarize common-policy cohort belief fits using the existing scores.

Run --fit FIT --evaluation EVALUATION --output NEW_DIRECTORY after the cohort
search and frozen evaluation finish. Actual workers identify ONLINE or SEPARATE;
training and evaluation data retain their original boundaries. The established
NLL, Energy, whole-replicate Monte Carlo errors and parameter tables are reused.
cohort_manifest.json binds the numerical summary to the executed mechanism and
common search policy. Oral validation follows the frozen behavioral nomination.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from . import model_0923_episodic_refit as scores
from . import model_0923_separate_belief_refit as separate
from .model_0923_h_validation import digest, load
from ..workflows.analysis import model_0923_cohort_belief_refit as driver
from ..workflows.analysis.model_0923_h_validation import write
from ..workflows.analysis.model_0923_online_parameter_cross import check_chunk

ROOT = driver.ROOT


def check_execution(job: dict, record: dict, belief: dict, precision: dict, mechanism: str) -> None:
    """Bind the declared cohort mechanism to its actual state updates."""
    if (mechanism not in driver.INTERVENTIONS or job.get('cohort_belief_mechanism') != mechanism
            or job.get('probe_mechanism') != mechanism or job.get('phase') not in ('training', 'evaluate', 'autonomous')):
        raise ValueError('Worker mechanism or role differs from the cohort declaration')
    if mechanism == 'SEPARATE':
        separate.check_execution(job, record, belief, precision)
        return
    fixed = {separate.TRANSFER: .01}
    copies = len(job['seeds']) * (1 if job['phase'] == 'autonomous' else job['particles'])
    if (job.get('belief_intervention') != driver.INTERVENTIONS[mechanism]
            or job.get('online_belief') is not True or job.get('fixed_coordinates') != fixed
            or job['candidate']['hyperparams'].get(separate.TRANSFER) != .01
            or job.get('structural_intervention') != 'retained_rule_precision'
            or job.get('retain_precision') is not True
            or precision.get('structural_intervention') != 'retained_rule_precision'
            or precision.get('retained_precision') is not True
            or precision.get('transitions') != copies * record['trials']
            or belief.get('enabled') is not True
            or belief.get('belief_intervention') != driver.INTERVENTIONS[mechanism]
            or belief.get('fixed_coordinates') != fixed
            or belief.get('exact_transition_timing') is not True
            or belief.get('evidence_matches_retained_state') is not True
            or belief.get('reconstructions') != copies * (record['trials'] - 1)):
        raise ValueError('ONLINE evidence, retained precision or feedback timing differs')


def audit_job(file: Path, mechanism: str, bind, *, smoke: bool = False) -> dict:
    """Check one original worker without changing its parameters or role."""
    job = load(bind(file)); target = Path(job['target'])
    if job.get('smoke') is not smoke:
        raise ValueError('Worker scientific scope changed')
    for name, sha in job['source_sha256'].items():
        if digest(bind(ROOT / name)) != sha:
            raise ValueError('Executing cohort source changed')
    for name, sha in job['retention_reference_inputs'].items():
        if digest(bind(Path(name))) != sha:
            raise ValueError('Common cohort input changed')
    if job['phase'] == 'training':
        record = load(bind(target.with_suffix('.json')))
        path = bind(target.with_suffix('.npz'))
        if digest(path) != record['predictions_sha256']:
            raise ValueError('Training arrays changed')
        with np.load(path) as arrays:
            separate.check_training_arrays(job, record, arrays)
    else:
        record, _ = check_chunk(job, mechanism, SimpleNamespace(read=load, digest=digest), bind)
    precision = load(bind(target.with_suffix('.precision.json')))
    extension = '.online.json' if mechanism == 'ONLINE' else '.belief.json'
    belief = load(bind(target.with_suffix(extension)))
    check_execution(job, record, belief, precision, mechanism)
    experiment = load(bind(target.with_suffix('.experiment.json')))
    metadata = load(bind(Path(job['folder']) / 'prepared.json'))
    expected_trials = ({'training': 8, 'autonomous': 16, 'evaluate': metadata['trials']}[job['phase']]) if smoke else (
        metadata['training_trials'] if job['phase'] == 'training' else metadata['trials'])
    expected_role = 'autonomous_from_start' if job['phase'] == 'autonomous' else job['phase']
    if (experiment.get('proposal') != (job['model'] == 'SEARCH') or experiment.get('prior') is not False
            or record['subject'] != metadata['subject'] or record['condition'] != metadata['condition']
            or record['trials'] != expected_trials or record['role'] != expected_role):
        raise ValueError('Executed proposal, participant, trial boundary or role changed')
    copies = len(job['seeds']) * (1 if job['phase'] == 'autonomous' else job['particles'])
    return dict(target=str(target.resolve()), subject=record['subject'], model=record['model'],
                phase=job['phase'], state_updates=copies * record['trials'], runs=len(job['seeds']))


def audit_jobs(source: Path, mechanism: str, bind) -> dict:
    files = sorted((source / 'jobs').glob('*.json'))
    if not files:
        raise ValueError('No executed cohort jobs')
    rows = [audit_job(file, mechanism, bind) for file in files]
    declared = {r['target'] for r in rows}
    actual = {str(f.with_suffix('').resolve()) for f in source.glob('**/chunks/*.json')
              if not f.name.endswith(separate.SIDECARS)}
    if len(declared) != len(files) or actual != declared:
        raise ValueError('Executed cohort chunks differ from declared workers')
    return dict(status='complete', jobs=len(files), rows=rows)


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new cohort summary directory')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError('Summary input changed')
        inputs[str(path)] = sha
        return path

    frozen, done, protocol, declaration = [load(bind(fit / name)) for name in
        ('nominees.json', 'completion.json', 'protocol.json', 'cohort_belief_protocol.json')]
    settings = frozen['settings']; mechanism = settings['belief_mechanism']
    if (frozen['smoke'] or protocol['smoke'] or done['smoke'] or done['status'] != 'complete'
            or done['test_choices_used'] is not False or frozen['status'] != 'frozen_before_test'
            or done['nomination_sha256'] != digest(fit / 'nominees.json')
            or protocol['settings'] != settings or protocol['source_sha256'] != frozen['source_sha256']
            or declaration['belief_mechanism'] != mechanism):
        raise ValueError('Require completed training-only cohort fits')
    modules = driver.helpers()
    restored, expected_inputs = driver.prepare_settings({k: settings[k] for k in driver.CONFIG_KEYS}, mechanism, modules)
    if restored != settings or any(declaration['input_sha256'].get(k) != v for k, v in expected_inputs.items()):
        raise ValueError('Saved cohort settings differ from the common policy')
    for name, expected in declaration['input_sha256'].items():
        if digest(bind(Path(name))) != expected:
            raise ValueError('Declared cohort source changed')
    driver.configure(modules, settings, declaration['input_sha256'])
    recalled = modules[-1]
    warm = load(bind(ROOT / settings['warm_start_fit'] / 'nominees.json'))
    banks, _ = recalled.starting_banks(settings, warm, False)
    actual_banks = {(r['subject'], r['model']): r['rows'] for r in protocol['initial_banks']}
    if actual_banks != banks or len(actual_banks) != len(protocol['initial_banks']):
        raise ValueError('Cohort initial banks differ from the common policy')
    evaluation_protocol = load(bind(evaluation / 'protocol.json'))
    evaluation_done = load(bind(evaluation / 'completion.json'))
    plans = {j['name']: j for j in evaluation_protocol['jobs']}
    files = sorted((evaluation / 'jobs').glob('*.json'))
    if (evaluation_done['status'] != 'complete' or len(plans) != len(evaluation_protocol['jobs'])
            or {p.stem for p in files} != set(plans) or len(files) != evaluation_done['jobs']
            or evaluation_protocol['nomination_sha256'] != digest(fit / 'nominees.json')):
        raise ValueError('Evaluation is incomplete or names different nominees')
    for file in files:
        job = load(bind(file))
        if any(job.get(k) != v for k, v in plans[file.stem].items()):
            raise ValueError('Executed evaluation job differs from the frozen plan')
    audits = {role: audit_jobs(path, mechanism, bind) for role, path in [('fit', fit), ('evaluation', evaluation)]}
    if (any(r['phase'] != 'training' for r in audits['fit']['rows'])
            or {r['phase'] for r in audits['evaluation']['rows']} != {'evaluate', 'autonomous'}):
        raise ValueError('Training and evaluation roles were mixed')
    spec = importlib.util.spec_from_file_location('src.Bayesian_state.evaluation._cohort_belief_summary', Path(scores.__file__))
    summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)
    summary.training_chunk_budgets = separate.training_chunk_budgets
    if mechanism == 'SEPARATE':
        summary.refit_inactive_channels = separate.refit_inactive_channels
    summary.summarize(fit, evaluation, output, reference_model='POOL')
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError('Cohort input changed during scoring')
    write(output / 'execution_audit.json', audits)
    shutil.copy2(__file__, output / 'cohort_analysis_source.py')
    sources = [Path(__file__), Path(scores.__file__), Path(separate.__file__), Path(driver.__file__)]
    write(output / 'cohort_manifest.json', dict(status='complete', belief_mechanism=mechanism,
        inputs_sha256=inputs, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        generic_manifest_sha256=digest(output / 'manifest.json'),
        execution_audit_sha256=digest(output / 'execution_audit.json'),
        scores='Original NLL/Energy and whole-filter/whole-path Monte Carlo errors',
        interpretation='Development prefix fit; subsequent choices and oral are external to parameter selection'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); summarize(args.fit, args.evaluation, args.output)
