"""Compare ONLINE and SEPARATE on declared development subjects.

Use --phase search --config YAML --mechanism ONLINE|SEPARATE --output NEW
[--smoke], then --phase evaluate --fit FIT --output NEW. A completed ONLINE
reference supplies the full search policy. Both mechanisms use that policy and
the same projected, subject-specific warm starts. Completed fixed-parameter
development runs establish which participants are eligible. Choices in the
original training segment select parameters; oral is reserved for evaluation.

Smoke executes the original small five-stage search for the first configured
participant, using one worker. It verifies integration, not scientific fit.
This adapter reuses the existing search, numerical workers and cognitive core.
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
INTERVENTIONS = {'ONLINE': 'online_retention_evidence', 'SEPARATE': 'separate_gamma_belief_evidence'}
REFERENCE_KEYS = {'reset_reference_fit', 'retained_reference_fit', 'online_reference_fit'}
CONFIG_KEYS = {'policy_reference_fit', 'development_reference', 'subjects', 'parallel_budget'}


def helpers():
    spec = importlib.util.spec_from_file_location('cohort_belief_separate', ROOT / SEPARATE)
    separate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = separate
    spec.loader.exec_module(separate)
    online = separate.load_helper('cohort_belief_online', separate.ONLINE)
    retained = online.helpers()
    return separate, online, retained, retained.helpers()


def expanded_settings(config: dict, reference: dict, eligible: set[int], mechanism: str) -> dict:
    """Inherit every numerical policy coordinate; vary participants and mechanism."""
    if set(config) != CONFIG_KEYS or mechanism not in INTERVENTIONS:
        raise ValueError('Specify the common policy, development source, subjects and worker budget')
    subjects = config['subjects']
    if (not isinstance(subjects, list) or not subjects
            or any(type(s) is not int for s in subjects)
            or len(subjects) != len(set(subjects)) or not set(subjects) <= eligible):
        raise ValueError('Use unique subjects already included in the declared development comparison')
    if type(config['parallel_budget']) is not int or not 1 <= config['parallel_budget'] <= 128:
        raise ValueError('Use a worker budget from 1 to 128')
    if reference.get('belief_intervention') != INTERVENTIONS['ONLINE']:
        raise ValueError('Use the completed ONLINE numerical policy')
    result = {k: deepcopy(v) for k, v in reference.items() if k not in REFERENCE_KEYS}
    result.update(deepcopy(config), belief_mechanism=mechanism, belief_intervention=INTERVENTIONS[mechanism],
                  scope='common_policy_development_belief_refit')
    return result


def prepare_settings(config: dict, mechanism: str, modules: tuple) -> tuple[dict, dict]:
    if set(config) != CONFIG_KEYS or mechanism not in INTERVENTIONS:
        raise ValueError('Specify the common policy, development source, subjects and worker budget')
    separate, online, retained, recalled = modules
    v = recalled.base.validation()
    inputs = {}

    def read(path):
        path = Path(path).resolve(); inputs[str(path)] = v.digest(path)
        return v.read(path)

    source = ROOT / config['policy_reference_fit']
    protocol, nominees, done = [read(source / name) for name in ('protocol.json', 'nominees.json', 'completion.json')]
    if (done['status'] != 'complete' or done['smoke'] or done['test_choices_used'] is not False
            or nominees['smoke'] or protocol['smoke'] or nominees['status'] != 'frozen_before_test'
            or nominees['settings'] != protocol['settings']
            or nominees['source_sha256'] != protocol['source_sha256']
            or done['nomination_sha256'] != v.digest(source / 'nominees.json')):
        raise ValueError('Require a completed training-only reference policy')
    development = ROOT / config['development_reference']
    previous, completed = [read(development / name) for name in ('protocol.json', 'completion.json')]
    if (previous['smoke'] or completed['smoke'] or completed['status'] != 'complete'
            or completed['protocol_sha256'] != v.digest(development / 'protocol.json')):
        raise ValueError('Require the completed development comparison')
    settings = expanded_settings(config, protocol['settings'], {b['subject'] for b in previous['banks']}, mechanism)
    if (settings['structural_intervention'] != retained.INTERVENTION
            or set(settings['fixed_coordinates']) != {online.TRANSFER}
            or list(settings['arms']) != ['POOL', 'SEARCH']):
        raise ValueError('Require retained precision, one inactive coordinate and both proposal arms')
    reference_spaces = read(source / 'parameter_spaces.json')
    support = ROOT / settings['parameter_support']
    inputs[str(support.resolve())] = v.digest(support)
    for arm in settings['arms']:
        current = online.fixed_space(recalled.base.model_space(settings, arm), settings['fixed_coordinates'])
        if current != reference_spaces[arm]:
            raise ValueError('Parameter support values differ from the archived common policy')
    warm_source = ROOT / settings['warm_start_fit']
    warm, warm_done = [read(warm_source / name) for name in ('nominees.json', 'completion.json')]
    if (warm['smoke'] or warm_done['smoke'] or warm_done['status'] != 'complete'
            or warm['status'] != 'frozen_before_test'
            or warm_done['nomination_sha256'] != v.digest(warm_source / 'nominees.json')):
        raise ValueError('Require completed behavior-only warm starts')
    for subject in settings['subjects']:
        if {r['model'] for r in warm['nominees'] if r['subject'] == subject} != {'BASE', 'NEW', 'POOL', 'DRAW'}:
            raise ValueError('Every development participant needs all four common warm starts')
        folder = ROOT / settings['data'] / f'subject_{subject}'
        meta = read(folder / 'prepared.json')
        if meta['subject'] != subject or meta['training_trials'] + meta['test_trials'] != meta['trials']:
            raise ValueError('Prepared participant or trial boundary changed')
        file = folder / 'training.npz'
        inputs[str(file.resolve())] = v.digest(file)
        if inputs[str(file.resolve())] != meta['array_sha256']['training.npz']:
            raise ValueError('Prepared training choices changed')
    for hashes in (protocol['source_sha256'], warm['source_sha256']):
        for name, expected in hashes.items():
            path = ROOT / name
            if v.digest(path) != expected:
                raise ValueError(f'Reference implementation changed: {name}')
            inputs[str(path.resolve())] = expected
    return settings, inputs


def worker(file: Path) -> None:
    separate, online, retained, recalled = helpers()
    job = recalled.base.validation().read(file)
    mechanism = job.get('cohort_belief_mechanism')
    if (mechanism not in INTERVENTIONS or job.get('belief_intervention') != INTERVENTIONS[mechanism]
            or job.get('online_belief') is not (mechanism == 'ONLINE')):
        raise ValueError('Cohort job does not identify its executed belief mechanism')
    (online.worker if mechanism == 'ONLINE' else separate.worker)(file)


def configure(modules: tuple, settings: dict, inputs: dict) -> None:
    separate, online, retained, recalled = modules
    recalled.configure()
    old_hashes, old_execute, old_archive = recalled.base.source_hashes, recalled.execute, recalled.base.archive
    old_space, old_starts = recalled.base.model_space, recalled.starting_banks
    probe = separate.load_helper('cohort_belief_probe', separate.PROBE)

    def hashes(v):
        result = old_hashes(v)
        for name in (ENTRY, SEPARATE, separate.ONLINE, separate.PROBE, separate.BELIEF,
                     probe.CROSS, online.ONLINE, online.RETENTION, retained.MECHANISM):
            result[name] = v.digest(ROOT / name)
        return result

    def execute(jobs, output, budget):
        mechanism = settings['belief_mechanism']
        tagged = [dict(deepcopy(job), structural_intervention=retained.INTERVENTION, retain_precision=True,
            retention_reference_inputs=inputs, belief_intervention=INTERVENTIONS[mechanism],
            online_belief=mechanism == 'ONLINE', probe_mechanism=mechanism,
            cohort_belief_mechanism=mechanism, fixed_coordinates=settings['fixed_coordinates']) for job in jobs]
        old_execute(tagged, output, budget)

    def archive(output, config, source_hashes):
        old_archive(output, config, source_hashes)
        recalled.base.validation().write(output / 'cohort_belief_protocol.json', dict(
            belief_mechanism=settings['belief_mechanism'], input_sha256=inputs,
            policy_reference_fit=settings['policy_reference_fit'],
            development_reference=settings['development_reference'],
            fixed_coordinates=settings['fixed_coordinates'],
            parameter_selection='Original training choices; oral excluded',
            comparison='Common supports, projected subject-specific starts, stage limits and seed namespaces',
            interpretation='Separate mechanism comparisons use equal policy; actual adaptive candidate counts can differ'))

    recalled.ENTRY = ENTRY
    recalled.base.source_hashes, recalled.base.execute, recalled.base.archive = hashes, execute, archive
    recalled.base.model_space = lambda own, arm: online.fixed_space(old_space(own, arm), own['fixed_coordinates'])
    recalled.starting_banks = lambda own, frozen, smoke: old_starts(
        own, online.projected_warm_starts(frozen, own['fixed_coordinates']), smoke)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('search', 'evaluate'))
    parser.add_argument('--mechanism', choices=tuple(INTERVENTIONS))
    for name in ('config', 'fit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker); return
    modules = helpers(); recalled = modules[-1]
    if args.phase == 'search' and args.config and args.output and args.mechanism:
        settings, inputs = prepare_settings(yaml.safe_load(args.config.read_text()), args.mechanism, modules)
        inputs[str(args.config.resolve())] = recalled.base.validation().digest(args.config)
        if args.smoke:
            settings['parallel_budget'] = 1
        configure(modules, settings, inputs)
        with tempfile.TemporaryDirectory(prefix='cohort_belief_config_') as temporary:
            expanded = Path(temporary) / 'config.yaml'
            expanded.write_text(yaml.safe_dump(settings, sort_keys=False))
            recalled.search(expanded, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke:
        frozen = recalled.base.validation().read(args.fit / 'nominees.json')['settings']
        settings, inputs = prepare_settings({k: frozen[k] for k in CONFIG_KEYS}, frozen['belief_mechanism'], modules)
        if settings != frozen:
            raise ValueError('Frozen cohort settings differ from their declared policy')
        inputs.update(recalled.base.validation().read(args.fit / 'cohort_belief_protocol.json')['input_sha256'])
        configure(modules, settings, inputs)
        recalled.base.evaluate(args.fit, args.output)
    else:
        parser.error('Use search with --config --mechanism, or evaluate with --fit, and a new --output')


if __name__ == '__main__':
    main()
