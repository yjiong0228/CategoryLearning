"""Refit POOL and SEARCH with retained precision at the original search budget.

Use --phase search --config CONFIG --output NEW_DIRECTORY [--smoke], then
--phase evaluate --fit FIT --output NEW_DIRECTORY. Both proposal mechanisms
retain learned precision; their original RESET fits remain the comparison.
All parameter supports, initial candidates, stage budgets and seed namespaces
must match that completed reference. Training choices select parameters; later
choices remain development diagnostics because these subjects informed design.
The existing search, particle filter and autonomous engine execute unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
RECALLED = 'src/Bayesian_state/workflows/analysis/model_0923_recalled_search_refit.py'
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/reentry_precision.py'
INTERVENTION = 'retained_rule_precision'


def helpers():
    spec = importlib.util.spec_from_file_location('precision_retention_refit_base', ROOT / RECALLED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_settings(settings: dict, original: dict) -> None:
    """Preserve the original numerical experiment while changing rule storage."""
    if settings.get('structural_intervention') != INTERVENTION:
        raise ValueError('Require the declared retained-precision intervention')
    subjects = settings['subjects']
    if (not subjects or len(subjects) != len(set(subjects))
            or not set(subjects) <= set(original['subjects'])):
        raise ValueError('Refit only declared original development subjects')
    changes = {'subjects', 'parallel_budget', 'scope', 'structural_intervention', 'reset_reference_fit'}
    if ({k: v for k, v in settings.items() if k not in changes}
            != {k: v for k, v in original.items() if k not in changes}):
        raise ValueError('Original search supports, budgets, seeds or task settings changed')
    if not 1 <= settings['parallel_budget'] <= 128:
        raise ValueError('Worker budget must lie within the shared CPU allocation')


def reference_inputs(recalled, settings: dict) -> dict:
    """Check the completed RESET fit and the exact initial banks before reuse."""
    v = recalled.base.validation()
    source = ROOT / settings['reset_reference_fit']
    protocol = v.read(source / 'protocol.json')
    frozen = v.read(source / 'nominees.json')
    done = v.read(source / 'completion.json')
    if (done['status'] != 'complete' or done['smoke'] or frozen['smoke']
            or frozen['status'] != 'frozen_before_test'
            or done['nomination_sha256'] != v.digest(source / 'nominees.json')
            or frozen['settings'] != protocol['settings']):
        raise ValueError('Require the complete unchanged RESET fit')
    validate_settings(settings, protocol['settings'])
    if frozen['source_sha256'] != protocol['source_sha256']:
        raise ValueError('RESET nomination and protocol implementations disagree')
    for name, expected in protocol['source_sha256'].items():
        if v.digest(ROOT / name) != expected:
            raise ValueError('Executing RESET implementation changed before the comparison')
    warm = v.read(ROOT / settings['warm_start_fit'] / 'nominees.json')
    banks, _ = recalled.starting_banks(settings, warm, False)
    original = {(r['subject'], r['model']): r['rows'] for r in protocol['initial_banks']}
    if any(rows != original.get(key) for key, rows in banks.items()):
        raise ValueError('Retained-precision initial banks differ from RESET')
    paths = [source / name for name in ('protocol.json', 'nominees.json', 'completion.json')]
    paths.append(ROOT / settings['warm_start_fit'] / 'nominees.json')
    return {str(path.resolve()): v.digest(path) for path in paths}


def worker(file: Path) -> None:
    recalled = helpers()
    v = recalled.base.validation()
    job = v.read(file)
    if (job.get('structural_intervention') != INTERVENTION
            or job.get('retain_precision') is not True):
        raise ValueError('Worker requires an explicit retained-precision job')
    for name, expected in job['retention_reference_inputs'].items():
        if v.digest(Path(name)) != expected:
            raise ValueError('RESET fitting reference changed')
    original_validation = recalled.base.validation
    original_autonomous = recalled.refit.autonomous

    def with_precision(function, own_job, workflow):
        # Called inside the existing worker after admission and proposal contexts.
        module = recalled.base.load(
            'src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision', MECHANISM)
        cls = sys.modules[
            'src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'
        ].MechanisticRuleSearchHypothesisTransitionModule
        with module.reentry_precision(cls, enabled=True) as events:
            function(own_job, workflow)
        v.write(Path(own_job['target']).with_suffix('.precision.json'), dict(
            structural_intervention=INTERVENTION, retained_precision=True,
            transitions=len(events), restores=sum(len(e['precision_restored_rules']) for e in events),
            event_unit='autonomous path transitions' if own_job['phase'] == 'autonomous'
                       else 'computational particle transitions'))

    def validation():
        module = original_validation()
        original_score = module.score
        module.score = lambda own_job, workflow: with_precision(original_score, own_job, workflow)
        return module

    recalled.base.validation = validation
    recalled.refit.autonomous = lambda own_job, workflow: with_precision(
        original_autonomous, own_job, workflow)
    recalled.worker(file)
    for name, expected in job['retention_reference_inputs'].items():
        if v.digest(Path(name)) != expected:
            raise ValueError('RESET fitting reference changed during execution')


def configure(recalled, inputs: dict) -> None:
    recalled.configure()
    original_hashes = recalled.base.source_hashes
    original_execute = recalled.execute
    original_archive = recalled.base.archive

    def hashes(v):
        result = original_hashes(v)
        result.update({name: v.digest(ROOT / name) for name in (ENTRY, RECALLED, MECHANISM)})
        return result

    def execute(jobs, output, budget):
        tagged = [dict(deepcopy(job), structural_intervention=INTERVENTION,
                       retain_precision=True, retention_reference_inputs=inputs) for job in jobs]
        original_execute(tagged, output, budget)

    def archive(output, config, source_hashes):
        original_archive(output, config, source_hashes)
        recalled.base.validation().write(output / 'retention_protocol.json', dict(
            structural_intervention=INTERVENTION, input_sha256=inputs,
            parameter_selection='original training choices only; oral excluded',
            evaluation_use='development; earlier and later behavior informed mechanism design',
            comparison='Both POOL and SEARCH retain precision; compare each with its original RESET fit'))

    recalled.ENTRY = ENTRY
    recalled.base.source_hashes = hashes
    recalled.base.execute = execute
    recalled.base.archive = archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--phase', choices=('search', 'evaluate'))
    for name in ('config', 'fit', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    recalled = helpers()
    if args.phase == 'search' and args.config and args.output:
        settings = yaml.safe_load(args.config.read_text())
        configure(recalled, reference_inputs(recalled, settings))
        recalled.search(args.config, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke:
        settings = recalled.base.validation().read(args.fit / 'nominees.json')['settings']
        configure(recalled, reference_inputs(recalled, settings))
        recalled.base.evaluate(args.fit, args.output)
    else:
        parser.error('Use search with --config or evaluate with --fit, and a new --output')


if __name__ == '__main__':
    main()
