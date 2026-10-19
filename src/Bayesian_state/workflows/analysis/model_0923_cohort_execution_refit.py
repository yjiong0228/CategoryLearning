"""Refit SELECT against a completed SEPARATE cohort reference.

Use --phase search --config YAML --output NEW [--smoke], followed by --phase
evaluate --fit FIT --output NEW. Participants, parameter supports, projected
starts, seed namespaces and stage limits come from the completed cohort fit.
Only the response policy changes. The existing cognitive worker and search
perform all numerical work; original training choices select parameters.
Oral reports remain external to fitting. Evaluation retains the search's
verified source bindings, including its small input YAML.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = 'src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_refit.py'
SELECT = 'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_refit.py'
CONFIG_KEYS = {'reference_fit', 'execution_policy', 'parallel_budget', 'scope'}
SCOPE = 'development_cohort_dominant_execution_refit'


def helper(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    cohort = helper('execution_cohort_driver', COHORT)
    return (cohort, *cohort.helpers())


def expanded_settings(config: dict, reference: dict) -> dict:
    """Preserve the complete cohort policy while changing execution identity."""
    if (set(config) != CONFIG_KEYS or config['scope'] != SCOPE
            or reference.get('belief_mechanism') != 'SEPARATE'):
        raise ValueError('Use the declared cohort SELECT scope and a SEPARATE reference')
    return helper('execution_cohort_select_definition', SELECT).expanded_settings(config, reference)


def checked_config_inputs(expected: dict, declared: dict, config: dict, validation) -> dict:
    """Allow exactly the original YAML in addition to reconstructed inputs."""
    extra = set(declared) - set(expected)
    if (any(declared.get(k) != value for k, value in expected.items()) or len(extra) != 1
            or yaml.safe_load(Path(next(iter(extra))).read_text()) != config):
        raise ValueError('Reference bindings differ from the reconstructed configuration and its YAML')
    for path, value in declared.items():
        if validation.digest(Path(path)) != value:
            raise ValueError('Bound cohort input changed')
    return dict(declared)


def prepare_settings(config: dict, modules: tuple) -> tuple[dict, dict]:
    cohort, separate, online, retained, recalled = modules
    if set(config) != CONFIG_KEYS:
        raise ValueError('Declare only the completed reference, SELECT policy, worker budget and scope')
    v = recalled.base.validation()
    source = ROOT / config['reference_fit']
    protocol, frozen, done, declaration = [v.read(source / name) for name in
        ('protocol.json', 'nominees.json', 'completion.json', 'cohort_belief_protocol.json')]
    original = protocol['settings']
    if (done['status'] != 'complete' or done['smoke'] or protocol['smoke'] or frozen['smoke']
            or done.get('test_choices_used') is not False or frozen['status'] != 'frozen_before_test'
            or frozen['settings'] != original or frozen['source_sha256'] != protocol['source_sha256']
            or done['nomination_sha256'] != v.digest(source / 'nominees.json')
            or declaration['belief_mechanism'] != 'SEPARATE'):
        raise ValueError('Require a completed training-only SEPARATE cohort fit')
    settings = expanded_settings(config, original)
    original_config = {key: original[key] for key in cohort.CONFIG_KEYS}
    restored, expected = cohort.prepare_settings(original_config, 'SEPARATE', modules[1:])
    if restored != original:
        raise ValueError('The cohort reference differs from its inherited search policy')
    inputs = checked_config_inputs(expected, declaration['input_sha256'], original_config, v)
    for path, value in protocol['source_sha256'].items():
        if v.digest(ROOT / path) != value:
            raise ValueError('Completed cohort implementation changed')
        inputs[str((ROOT / path).resolve())] = value
    for name in ('protocol.json', 'nominees.json', 'completion.json', 'cohort_belief_protocol.json', 'parameter_spaces.json'):
        inputs[str((source / name).resolve())] = v.digest(source / name)
    spaces = v.read(source / 'parameter_spaces.json')
    old_space = recalled.base.model_space
    try:
        recalled.base.model_space = lambda own, arm: online.fixed_space(old_space(own, arm), own['fixed_coordinates'])
        if any(recalled.base.model_space(settings, arm) != spaces[arm] for arm in settings['arms']):
            raise ValueError('SELECT supports differ from the actual cohort reference')
        warm = v.read(ROOT / settings['warm_start_fit'] / 'nominees.json')
        banks, _ = recalled.starting_banks(settings, online.projected_warm_starts(warm, settings['fixed_coordinates']), False)
    finally:
        recalled.base.model_space = old_space
    actual = {(row['subject'], row['model']): row['rows'] for row in protocol['initial_banks']}
    if banks != actual or len(actual) != len(protocol['initial_banks']):
        raise ValueError('SELECT starts differ from the actual projected cohort starts')
    return settings, inputs


def worker(file: Path) -> None:
    execution = helper('execution_cohort_worker', SELECT)
    job = execution.helpers()[-1].base.validation().read(file)
    if (job.get('cohort_belief_mechanism') != 'SEPARATE' or job.get('probe_mechanism') != 'SEPARATE'
            or job.get('execution_policy') != 'SELECT'):
        raise ValueError('Cohort SELECT workers must retain the SEPARATE belief identity')
    execution.worker(file)


def configure(modules: tuple, settings: dict, inputs: dict) -> None:
    cohort, _, _, _, recalled = modules
    cohort.configure(modules[1:], settings, inputs)
    execution = helper('execution_cohort_configuration', SELECT)
    old_hashes, old_execute, old_archive = recalled.base.source_hashes, recalled.base.execute, recalled.base.archive

    def hashes(v):
        result = old_hashes(v)
        result.update({name: v.digest(ROOT / name) for name in (ENTRY, SELECT, execution.PROBE, execution.EXECUTION)})
        return result

    def execute(jobs, output, budget):
        old_execute([dict(deepcopy(job), execution_policy='SELECT') for job in jobs], output, budget)

    def archive(output, config, source_hashes):
        old_archive(output, config, source_hashes)
        recalled.base.validation().write(output / 'execution_refit_protocol.json', dict(
            execution_policy='SELECT', rule_execution_policy='existing_dominant_rule_only',
            execution_reference_fit=settings['execution_reference_fit'], inputs_sha256=inputs,
            cohort_belief_mechanism='SEPARATE', scope=SCOPE,
            parameter_selection='Original training choice marginal likelihood; oral excluded',
            comparison='Exact projected starts, supports, stage limits and seed namespaces from the completed cohort reference',
            interpretation='Dominant-rule switching retains inertia; the selected rule supplies each response'))

    recalled.ENTRY = ENTRY
    recalled.base.source_hashes, recalled.base.execute, recalled.base.archive = hashes, execute, archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('search', 'evaluate'))
    for name in ('config', 'fit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    modules = helpers(); recalled = modules[-1]; v = recalled.base.validation()
    if args.phase == 'search' and args.config and args.output:
        config = yaml.safe_load(args.config.read_text())
        settings, inputs = prepare_settings(config, modules)
        inputs[str(args.config.resolve())] = v.digest(args.config)
        configure(modules, settings, inputs)
        with tempfile.TemporaryDirectory(prefix='cohort_select_config_') as temporary:
            expanded = Path(temporary) / 'config.yaml'
            expanded.write_text(yaml.safe_dump(settings, sort_keys=False))
            recalled.search(expanded, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke:
        frozen = v.read(args.fit / 'nominees.json')['settings']
        config = dict(reference_fit=frozen['execution_reference_fit'], execution_policy=frozen['execution_policy'],
                      parallel_budget=frozen['parallel_budget'], scope=frozen['scope'])
        settings, expected = prepare_settings(config, modules)
        if settings != frozen:
            raise ValueError('Frozen cohort SELECT policy changed')
        declaration = v.read(args.fit / 'execution_refit_protocol.json')
        if (declaration['execution_policy'] != 'SELECT' or declaration['scope'] != SCOPE
                or declaration['execution_reference_fit'] != config['reference_fit']):
            raise ValueError('Frozen SELECT reference identity changed')
        inputs = checked_config_inputs(expected, declaration['inputs_sha256'], config, v)
        configure(modules, settings, inputs)
        recalled.base.evaluate(args.fit, args.output)
    else:
        parser.error('Use search with --config or evaluate with --fit, and a new --output')


if __name__ == '__main__':
    main()
