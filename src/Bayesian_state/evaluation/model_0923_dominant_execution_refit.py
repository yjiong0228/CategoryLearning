"""Summarize SELECT refits using the unchanged SEPARATE behavioral scores.

Run --fit FIT --evaluation EVALUATION --output NEW_DIRECTORY. Every actual
worker must execute the declared dominant-rule policy before each choice,
including trial zero. Execution sidecars describe these same workers and are
excluded from computational budgets. Original SEPARATE audits, NLL, Energy
and Monte Carlo definitions are retained. execution_manifest.json binds the
additional policy checks and the inherited summary. Oral validation follows
from SELECT states replayed under this same execution policy.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil

from .model_0923_h_validation import digest, load
from ..workflows.analysis.model_0923_h_validation import write

SOURCE = Path(__file__).with_name('model_0923_separate_belief_refit.py')
POLICY = 'existing_dominant_rule_only'


def separate_summary():
    """Keep adapter bindings private to this analysis process."""
    spec = importlib.util.spec_from_file_location(
        'src.Bayesian_state.evaluation._dominant_execution_summary', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SIDECARS += ('.execution.json',)
    return module


class _TrainingResults:
    def __init__(self, stage: Path):
        self.stage, self.name = stage, stage.name

    def glob(self, pattern: str):
        return (file for file in self.stage.glob(pattern)
                if not file.name.endswith(('.belief.json', '.execution.json')))


def training_chunk_budgets(stage: Path) -> list[dict]:
    return separate_summary().base.training_chunk_budgets(_TrainingResults(stage))


def check_selection(job: dict, result: dict, execution: dict) -> None:
    """Check the response policy and its role-specific per-choice count."""
    if job['phase'] not in ('training', 'evaluate', 'autonomous'):
        raise ValueError('Unknown SELECT worker role')
    copies = len(job['seeds']) * (1 if job['phase'] == 'autonomous' else job['particles'])
    if (job.get('execution_policy') != 'SELECT' or execution.get('enabled') is not True
            or execution.get('policy') != 'SELECT'
            or execution.get('rule_execution_policy') != POLICY
            or execution.get('uses_existing_dominant') is not True
            or execution.get('task_categories_used_for_selection') is not False
            or execution.get('updates') != copies * result['trials']):
        raise ValueError('SELECT execution policy or per-choice count differs')


def audit_jobs(source: Path, *, smoke: bool = False) -> dict:
    """Retain all belief/precision checks and bind every execution sidecar."""
    audit = separate_summary().audit_jobs(source, smoke=smoke)
    for file in sorted((source / 'jobs').glob('*.json')):
        job = load(file)
        target = Path(job['target'])
        result = load(target.with_suffix('.json'))
        sidecar = target.with_suffix('.execution.json').resolve()
        audit['inputs_sha256'][str(sidecar)] = digest(sidecar)
        check_selection(job, result, load(sidecar))
    for path, value in audit['inputs_sha256'].items():
        if digest(Path(path)) != value:
            raise ValueError('SELECT audit input changed')
    audit.update(execution_policy='SELECT', rule_execution_policy=POLICY)
    return audit


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    frozen = load(fit / 'nominees.json')
    protocol = load(fit / 'execution_refit_protocol.json')
    if (frozen['settings'].get('execution_policy') != 'SELECT'
            or protocol.get('execution_policy') != 'SELECT'
            or protocol.get('rule_execution_policy') != POLICY
            or protocol.get('execution_reference_fit') != frozen['settings'].get('execution_reference_fit')):
        raise ValueError('Require frozen SELECT nominees and their reference policy')
    inputs = dict(protocol['inputs_sha256'])
    for path in (fit / 'execution_refit_protocol.json', Path(__file__), SOURCE):
        inputs[str(path.resolve())] = digest(path)
    for path, value in inputs.items():
        if digest(Path(path)) != value:
            raise ValueError('SELECT reference or implementation changed')
    module = separate_summary()
    module.audit_jobs = audit_jobs
    module.training_chunk_budgets = training_chunk_budgets
    module.summarize(fit, evaluation, output)
    shutil.copy2(__file__, output / 'execution_analysis_source.py')
    for path, value in inputs.items():
        if digest(Path(path)) != value:
            raise ValueError('SELECT summary input changed')
    write(output / 'execution_manifest.json', dict(status='complete',
        execution_policy='SELECT', rule_execution_policy=POLICY, inputs_sha256=inputs,
        separate_manifest_sha256=digest(output / 'separate_manifest.json'),
        interpretation='Training choices estimate SELECT parameters; later choices and autonomous behavior are evaluated separately',
        inertia='Controls dominant-rule switching; within-rule response weights stay one-hot'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.fit, args.evaluation, args.output)
