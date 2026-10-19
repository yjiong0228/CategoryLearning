"""Behavior-only SEPARATE refit at the existing ONLINE search policy.

Use --phase search --config YAML --output NEW [--smoke], then --phase evaluate
--fit FIT --output NEW. The completed ONLINE fit supplies the comparison policy:
same supports, projected common starts, stage limits, seed namespaces and
training boundary. The existing SEPARATE numerical worker executes the new
mechanism. Both POOL and SEARCH remain in the common search. Actual unique
candidate counts can differ after adaptive search and union deduplication.
Oral reports enter only subsequent external state validation.
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
ONLINE = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_refit.py'
PROBE = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_probe.py'
BELIEF = 'src/Bayesian_state/model/modules/hypothesis_transition/separate_belief_evidence.py'
INTERVENTION = 'separate_gamma_belief_evidence'


def load_helper(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_settings(settings: dict, reference: dict) -> None:
    """Only the belief mechanism and run metadata differ from ONLINE."""
    if (settings.get('belief_intervention') != INTERVENTION
            or reference.get('belief_intervention') != 'online_retention_evidence'
            or not settings.get('online_reference_fit')
            or not 1 <= settings.get('parallel_budget', 0) <= 128):
        raise ValueError('Declare SEPARATE and its completed ONLINE reference policy')
    allowed = {'belief_intervention', 'scope', 'parallel_budget', 'online_reference_fit'}
    if ({k: v for k, v in settings.items() if k not in allowed}
            != {k: v for k, v in reference.items() if k not in allowed}):
        raise ValueError('ONLINE supports, common starts, seeds, subjects or stage budgets changed')


def reference_inputs(online, retention, recalled, settings: dict) -> dict:
    v = recalled.base.validation(); source = ROOT / settings['online_reference_fit']
    protocol, frozen, done = (v.read(source / name) for name in ('protocol.json', 'nominees.json', 'completion.json'))
    validate_settings(settings, protocol['settings'])
    if (done['status'] != 'complete' or done['smoke'] or done['test_choices_used'] is not False
            or frozen['smoke'] or protocol['smoke'] or frozen['status'] != 'frozen_before_test'
            or done['nomination_sha256'] != v.digest(source / 'nominees.json')
            or frozen['settings'] != protocol['settings'] or frozen['source_sha256'] != protocol['source_sha256']):
        raise ValueError('Require the complete training-only ONLINE fit')
    equivalent = {k: deepcopy(value) for k, value in settings.items() if k != 'online_reference_fit'}
    equivalent['belief_intervention'] = online.INTERVENTION
    inputs = online.reference_inputs(retention, recalled, equivalent)
    for name, sha in frozen['source_sha256'].items():
        if v.digest(ROOT / name) != sha: raise ValueError('Completed ONLINE implementation changed')
        inputs[str((ROOT / name).resolve())] = sha
    for name in ('protocol.json', 'nominees.json', 'completion.json', 'online_refit_protocol.json'):
        inputs[str((source / name).resolve())] = v.digest(source / name)
    return inputs


def worker(file: Path) -> None:
    probe = load_helper('separate_refit_probe', PROBE)
    online = probe.helpers().helpers(); v = online.helpers().helpers().base.validation()
    job = v.read(file)
    if (job.get('belief_intervention') != INTERVENTION or job.get('probe_mechanism') != 'SEPARATE'
            or job.get('online_belief') is not False
            or set(job.get('fixed_coordinates', {})) != {online.TRANSFER}
            or any(job['candidate']['hyperparams'].get(k) != value for k, value in job['fixed_coordinates'].items())):
        raise ValueError('Refit must execute SEPARATE with its declared inactive coordinate')
    probe.worker(file)


def configure(online, retention, recalled, settings: dict, inputs: dict) -> None:
    """Reuse the original search and scoring, changing only the worker adapter."""
    recalled.configure()
    old_hashes, old_execute, old_archive = recalled.base.source_hashes, recalled.execute, recalled.base.archive
    old_space, old_starts = recalled.base.model_space, recalled.starting_banks
    probe = load_helper('separate_refit_probe_policy', PROBE)

    def hashes(v):
        result = old_hashes(v)
        for name in (ENTRY, ONLINE, PROBE, BELIEF, probe.CROSS, online.RETENTION, retention.MECHANISM):
            result[name] = v.digest(ROOT / name)
        return result

    def execute(jobs, output, budget):
        tagged = [dict(deepcopy(job), structural_intervention=retention.INTERVENTION, retain_precision=True,
            retention_reference_inputs=inputs, belief_intervention=INTERVENTION, online_belief=False,
            probe_mechanism='SEPARATE', fixed_coordinates=settings['fixed_coordinates']) for job in jobs]
        old_execute(tagged, output, budget)

    def archive(output, config, source_hashes):
        old_archive(output, config, source_hashes)
        recalled.base.validation().write(output / 'separate_refit_protocol.json', dict(
            belief_intervention=INTERVENTION, structural_intervention=retention.INTERVENTION,
            online_reference_fit=settings['online_reference_fit'], input_sha256=inputs,
            fixed_coordinates=settings['fixed_coordinates'],
            parameter_selection='Original training choices only; oral excluded',
            initialization='Same projected common starts as completed ONLINE',
            comparison='Same supports, stages, seed namespaces and budget limits; actual deduplicated work reported separately',
            interpretation='Gamma controls belief evidence persistence; delta controls candidate management evidence'))

    recalled.ENTRY = ENTRY
    recalled.base.source_hashes, recalled.base.execute, recalled.base.archive = hashes, execute, archive
    recalled.base.model_space = lambda own, arm: online.fixed_space(old_space(own, arm), own['fixed_coordinates'])
    recalled.starting_banks = lambda own, frozen, smoke: old_starts(
        own, online.projected_warm_starts(frozen, own['fixed_coordinates']), smoke)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('search', 'evaluate'))
    for name in ('config', 'fit', 'output', 'worker'): parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker); return
    online = load_helper('separate_refit_online', ONLINE); retention = online.helpers(); recalled = retention.helpers()
    if args.phase == 'search' and args.config and args.output:
        settings = yaml.safe_load(args.config.read_text())
        inputs = reference_inputs(online, retention, recalled, settings)
        configure(online, retention, recalled, settings, inputs)
        recalled.search(args.config, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke:
        settings = recalled.base.validation().read(args.fit / 'nominees.json')['settings']
        inputs = reference_inputs(online, retention, recalled, settings)
        configure(online, retention, recalled, settings, inputs)
        recalled.base.evaluate(args.fit, args.output)
    else: parser.error('Use search with --config or evaluate with --fit, and a new --output')


if __name__ == '__main__': main()
