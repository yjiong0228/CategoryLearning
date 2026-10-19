"""Refit SELECT at the completed SEPARATE fit's common search policy.

Use --phase search --config YAML --output NEW [--smoke], then --phase evaluate
--fit FIT --output NEW. The small YAML names a completed SEPARATE reference fit,
SELECT execution and the available worker allocation. All supports, projected
common starts, stage budgets, seed namespaces and training boundaries are copied
from that reference. The existing search and numerical worker perform the fit;
oral and later choices remain outside parameter selection. The additional
execution_refit_protocol.json identifies the changed response policy. Summary
and oral consumers must validate SELECT execution alongside SEPARATE belief.
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
SEPARATE = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_refit.py'
PROBE = 'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_probe.py'
EXECUTION = 'src/Bayesian_state/model/modules/hypothesis_transition/dominant_rule_execution.py'
CONFIG_KEYS = {'reference_fit', 'execution_policy', 'parallel_budget', 'scope'}


def helpers():
    spec = importlib.util.spec_from_file_location('dominant_execution_refit_separate', ROOT / SEPARATE)
    separate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = separate
    spec.loader.exec_module(separate)
    online = separate.load_helper('dominant_execution_refit_online', separate.ONLINE)
    retention = online.helpers()
    recalled = retention.helpers()
    return separate, online, retention, recalled


def expanded_settings(config: dict, reference: dict) -> dict:
    """Copy the policy; only execution identity, run scope and workers vary."""
    if (set(config) != CONFIG_KEYS or config['execution_policy'] != 'SELECT'
            or not isinstance(config['parallel_budget'], int) or isinstance(config['parallel_budget'], bool)
            or not 1 <= config['parallel_budget'] <= 128 or not config['reference_fit']
            or reference.get('belief_intervention') != 'separate_gamma_belief_evidence'
            or reference.get('execution_policy', 'MIX') != 'MIX'):
        raise ValueError('Declare SELECT, its completed SEPARATE/MIX reference and an available worker budget')
    result = deepcopy(reference)
    result.update(execution_policy='SELECT', execution_reference_fit=config['reference_fit'],
                  parallel_budget=config['parallel_budget'], scope=config['scope'])
    return result


def prepare_settings(config: dict, modules: tuple) -> tuple[dict, dict]:
    separate, online, retention, recalled = modules
    v = recalled.base.validation()
    if set(config) != CONFIG_KEYS:
        raise ValueError('The SELECT config can change only reference, execution policy, workers and scope')
    source = ROOT / config['reference_fit']
    protocol, frozen, done = (v.read(source / n) for n in ('protocol.json', 'nominees.json', 'completion.json'))
    if (done['status'] != 'complete' or done['smoke'] or protocol['smoke'] or frozen['smoke']
            or frozen['status'] != 'frozen_before_test' or done.get('test_choices_used') is not False
            or frozen['settings'] != protocol['settings'] or frozen['source_sha256'] != protocol['source_sha256']
            or done['nomination_sha256'] != v.digest(source / 'nominees.json')):
        raise ValueError('Require the complete frozen training-only SEPARATE reference')
    settings = expanded_settings(config, protocol['settings'])
    inputs = separate.reference_inputs(online, retention, recalled, protocol['settings'])
    for name, value in protocol['source_sha256'].items():
        if v.digest(ROOT / name) != value:
            raise ValueError('Completed SEPARATE implementation changed')
        inputs[str((ROOT / name).resolve())] = value
    for name in ('protocol.json', 'nominees.json', 'completion.json', 'separate_refit_protocol.json'):
        inputs[str((source / name).resolve())] = v.digest(source / name)
    # Confirm the actual projected initial bank, not just the configured count.
    warm = v.read(ROOT / settings['warm_start_fit'] / 'nominees.json')
    original_space = recalled.base.model_space
    try:
        recalled.base.model_space = lambda own, arm: online.fixed_space(original_space(own, arm), own['fixed_coordinates'])
        starts, _ = recalled.starting_banks(settings, online.projected_warm_starts(warm, settings['fixed_coordinates']), False)
    finally:
        recalled.base.model_space = original_space
    expected = {(b['subject'], b['model']): b['rows'] for b in protocol['initial_banks']}
    if starts != expected:
        raise ValueError('Projected SELECT starts differ from the actual SEPARATE reference')
    return settings, inputs


def worker(file: Path) -> None:
    separate, _, _, recalled = helpers()
    job = recalled.base.validation().read(file)
    if job.get('execution_policy') != 'SELECT':
        raise ValueError('SELECT refit workers must declare their response policy')
    probe = separate.load_helper('dominant_execution_refit_worker', PROBE)
    probe.worker(file)


def configure(modules: tuple, settings: dict, inputs: dict) -> None:
    separate, online, retention, recalled = modules
    separate.configure(online, retention, recalled, settings, inputs)
    old_hashes, old_execute, old_archive = recalled.base.source_hashes, recalled.base.execute, recalled.base.archive

    def hashes(v):
        result = old_hashes(v)
        result.update({name: v.digest(ROOT / name) for name in (ENTRY, SEPARATE, PROBE, EXECUTION)})
        return result

    def execute(jobs, output, budget):
        old_execute([dict(deepcopy(job), execution_policy='SELECT') for job in jobs], output, budget)

    def archive(output, config, source_hashes):
        old_archive(output, config, source_hashes)
        recalled.base.validation().write(output / 'execution_refit_protocol.json', dict(
            execution_policy='SELECT', rule_execution_policy='existing_dominant_rule_only',
            belief_intervention=settings['belief_intervention'],
            execution_reference_fit=settings['execution_reference_fit'], inputs_sha256=inputs,
            initialization='Exact projected common starts from the completed SEPARATE reference',
            parameter_selection='Original training choice marginal likelihood; oral excluded',
            comparison='Same supports, stage limits and seeds; actual deduplicated work recorded separately',
            interpretation='Inertia changes dominant-rule switching; response uses that rule only'))

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
    modules = helpers()
    recalled = modules[-1]
    if args.phase == 'search' and args.config and args.output:
        settings, inputs = prepare_settings(yaml.safe_load(args.config.read_text()), modules)
        inputs[str(args.config.resolve())] = recalled.base.validation().digest(args.config)
        configure(modules, settings, inputs)
        with tempfile.TemporaryDirectory(prefix='select_refit_config_') as directory:
            expanded = Path(directory) / 'expanded.yaml'
            expanded.write_text(yaml.safe_dump(settings, sort_keys=False))
            recalled.search(expanded, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke:
        frozen = recalled.base.validation().read(args.fit / 'nominees.json')['settings']
        config = dict(reference_fit=frozen['execution_reference_fit'], execution_policy=frozen['execution_policy'],
                      parallel_budget=frozen['parallel_budget'], scope=frozen['scope'])
        settings, inputs = prepare_settings(config, modules)
        if settings != frozen:
            raise ValueError('Frozen SELECT policy changed')
        configure(modules, settings, inputs)
        recalled.base.evaluate(args.fit, args.output)
    else:
        parser.error('Use search with --config or evaluate with --fit, and a new --output')


if __name__ == '__main__':
    main()
