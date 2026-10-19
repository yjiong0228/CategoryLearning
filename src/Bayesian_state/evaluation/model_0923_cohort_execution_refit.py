"""Score cohort SELECT refits with the original behavioral definitions.

Use --fit FIT --evaluation EVALUATION --output NEW after the frozen evaluation.
The private cohort adapter checks SEPARATE evidence and retained precision;
every worker additionally requires SELECT's per-choice execution record.
The original NLL, Energy and Monte Carlo errors remain unchanged. Execution
sidecars describe existing work and are excluded from computational budgets.
execution_manifest.json binds SELECT's identity to the cohort summary.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

from . import model_0923_dominant_execution_refit as execution
from .model_0923_h_validation import digest, load
from ..workflows.analysis import model_0923_cohort_execution_refit as driver
from ..workflows.analysis.model_0923_h_validation import write

SOURCE = Path(__file__).with_name('model_0923_cohort_belief_refit.py')
FROZEN_CONFIG_KEYS = {'execution_reference_fit', 'execution_policy', 'parallel_budget', 'scope'}


def frozen_config(settings: dict) -> dict:
    return dict(reference_fit=settings['execution_reference_fit'], execution_policy=settings['execution_policy'],
                parallel_budget=settings['parallel_budget'], scope=settings['scope'])


def cohort_summary():
    """Adapt an isolated scorer; keep ordinary cohort summaries unchanged."""
    spec = importlib.util.spec_from_file_location('src.Bayesian_state.evaluation._cohort_execution_summary', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_audit = module.audit_job
    module.separate = execution.separate_summary()
    module.separate.training_chunk_budgets = execution.training_chunk_budgets

    def prepare(settings, mechanism, modules):
        if mechanism != 'SEPARATE':
            raise ValueError('Cohort SELECT requires SEPARATE evidence')
        return driver.prepare_settings(frozen_config(settings), modules)

    module.driver = SimpleNamespace(__file__=driver.__file__, CONFIG_KEYS=FROZEN_CONFIG_KEYS,
        helpers=driver.helpers, prepare_settings=prepare, configure=driver.configure,
        INTERVENTIONS={'SEPARATE': 'separate_gamma_belief_evidence'})

    def audit(file, mechanism, bind, *, smoke=False):
        if mechanism != 'SEPARATE':
            raise ValueError('Cohort SELECT requires SEPARATE evidence')
        result = original_audit(file, mechanism, bind, smoke=smoke)
        job = load(bind(file)); target = Path(job['target'])
        execution.check_selection(job, load(bind(target.with_suffix('.json'))),
                                  load(bind(target.with_suffix('.execution.json'))))
        return result

    module.audit_job = audit
    return module


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    frozen = load(fit / 'nominees.json')
    declaration = load(fit / 'execution_refit_protocol.json')
    belief = load(fit / 'cohort_belief_protocol.json')
    settings = frozen['settings']
    if (settings.get('execution_policy') != 'SELECT' or settings.get('scope') != driver.SCOPE
            or declaration.get('execution_policy') != 'SELECT'
            or declaration.get('rule_execution_policy') != execution.POLICY
            or declaration.get('scope') != driver.SCOPE
            or declaration.get('execution_reference_fit') != settings.get('execution_reference_fit')
            or declaration.get('inputs_sha256') != belief.get('input_sha256')):
        raise ValueError('Require coherent cohort and SELECT execution declarations')
    modules = driver.helpers()
    restored, expected = driver.prepare_settings(frozen_config(settings), modules)
    if restored != settings:
        raise ValueError('Cohort SELECT settings differ from their completed reference')
    inputs = driver.checked_config_inputs(expected, declaration['inputs_sha256'],
                                          frozen_config(settings), modules[-1].base.validation())
    for path in (fit / 'execution_refit_protocol.json', Path(__file__), SOURCE,
                 Path(driver.__file__), Path(execution.__file__)):
        inputs[str(path.resolve())] = digest(path)
    cohort_summary().summarize(fit, evaluation, output)
    for path, value in inputs.items():
        if digest(Path(path)) != value:
            raise ValueError('Cohort SELECT inputs changed during scoring')
    shutil.copy2(__file__, output / 'execution_analysis_source.py')
    write(output / 'execution_manifest.json', dict(status='complete', execution_policy='SELECT',
        rule_execution_policy=execution.POLICY, scope=driver.SCOPE, inputs_sha256=inputs,
        cohort_manifest_sha256=digest(output / 'cohort_manifest.json'),
        interpretation='Behavior-only cohort SELECT refit; subsequent choices and oral remain outside parameter selection'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.fit, args.evaluation, args.output)
