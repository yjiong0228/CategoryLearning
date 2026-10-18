"""Replay completed online-belief choice screens using the existing state observer.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Each proposal arm gets its
own BASE/ONLINE comparison, with unchanged parameters, particles and seeds.
--smoke replays the first seed of both variants for at most sixteen trials in
one worker. The source may itself be the completed eight-trial scoring fixture.
Every prediction must exactly match the saved filter. Oral content enters only
the subsequent, separate decision-basis alignment command.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
REPLAY = 'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_replay.py'
ONLINE = 'src/Bayesian_state/model/modules/hypothesis_transition/online_evidence_belief.py'


def helpers():
    spec = importlib.util.spec_from_file_location('online_belief_state_replay', ROOT / REPLAY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked_job(planned: dict, actual: dict, record: dict, sidecar: dict) -> None:
    """Require the exact executed variant and parameter identity before replay."""
    if (actual != planned or actual.get('retain_precision') is not True
            or actual.get('phase') != 'evaluate' or actual.get('model') not in ('POOL', 'SEARCH')
            or type(actual.get('online_belief')) is not bool
            or any(record[key] != actual[key] for key in ('candidate', 'model', 'seeds', 'particles'))
            or sidecar.get('enabled') is not actual['online_belief']
            or sidecar.get('evidence_matches_retained_state') is not True
            or sidecar.get('exact_transition_timing') is not True):
        raise ValueError('Saved online-belief filter identity or execution differs')


def worker(file: Path) -> None:
    replay = helpers()
    job = replay.read(file)
    events = []
    original_load, original_write = replay.load, replay.write

    def load(name, relative):
        module = original_load(name, relative)
        if relative == replay.REFIT:
            base_load = module.base.load

            def with_online(child_name, child_relative):
                child = base_load(child_name, child_relative)
                if child_relative == replay.REENTRY:
                    retained = child.reentry_precision
                    proposal_name = 'src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal'
                    if proposal_name not in sys.modules:
                        # ONLINE imports its shared normalization helper in POOL too.
                        base_load(proposal_name, replay.PROPOSAL)
                    online = base_load('src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief', ONLINE)

                    @contextmanager
                    def combined(cls, **kwargs):
                        # Install in the same order as the original choice worker.
                        with retained(cls, **kwargs) as precision_events:
                            with online.online_evidence_belief(cls, enabled=job['online_belief']) as own:
                                yield precision_events
                                events.extend(own)

                    child.reentry_precision = combined
                return child

            module.base.load = with_online
        return module

    def write(path, value):
        if path == Path(job['target']).with_suffix('.json'):
            counts = Counter(event['trial'] for event in events)
            expected = {t:job['particles'] for t in range(1, value['trials'])} if job['online_belief'] else {}
            if counts != expected:
                raise ValueError('State replay online evidence timing differs from choice scoring')
            value = dict(value, arm=job['reported_arm'], proposal_arm=job['model'],
                         online_belief=job['online_belief'], online_reconstructions=len(events),
                         online_evidence_matches_retained_state=True)
        original_write(path, value)

    replay.load, replay.write = load, write
    replay.worker(file)


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, list[dict]]:
    replay = helpers()
    settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError('Use a new online-belief state destination')
    if (settings['proposal_arm'] not in ('POOL', 'SEARCH') or settings['sample_role'] != 'development'
            or not 1 <= settings['workers'] <= 128):
        raise ValueError('Require a declared development proposal arm and bounded workers')
    source = ROOT / settings['source']
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = replay.digest(path)
        return path

    bind(config)
    original = replay.read(bind(source / 'protocol.json'))
    done = replay.read(bind(source / 'completion.json'))
    if (done['status'] != 'complete' or done['nominees_changed']
            or done['protocol_sha256'] != replay.digest(source / 'protocol.json')
            or done['smoke'] != original['smoke'] or (original['smoke'] and not smoke)):
        raise ValueError('Require a completed unchanged choice screen; short fixtures need --smoke')
    for field in ('input_sha256', 'source_sha256'):
        for path, sha in original[field].items():
            if replay.digest(bind(ROOT / path)) != sha:
                raise ValueError('Choice-screen source changed')
    plans = [p for p in original['plans'] if p['model'] == settings['proposal_arm']]
    if len(plans) != 2 or {p['variant'] for p in plans} != {'baseline', 'online'}:
        raise ValueError('Need both executed variants of the specified proposal arm')
    baseline = next(p['job'] for p in plans if p['variant'] == 'baseline')
    for plan in plans:
        job = plan['job']
        for key in ('candidate', 'seeds', 'particles', 'folder', 'runtime', 'model', 'arm_settings'):
            if job[key] != baseline[key]:
                raise ValueError('Variants differ beyond the online-belief mechanism')
        if job['online_belief'] != (plan['variant'] == 'online'):
            raise ValueError('Variant label differs from its executed mechanism')
    hashes = {path:replay.digest(ROOT / path) for path in
              (ENTRY, REPLAY, ONLINE, replay.PF, replay.COLLECTOR, replay.REFIT, replay.PROPOSAL, replay.REENTRY)}
    jobs = []
    for plan in plans:
        if len(plan['chunks']) != len(baseline['seeds']):
            raise ValueError('Incomplete seed plan')
        for planned in plan['chunks'][:1] if smoke else plan['chunks']:
            file = bind(source / 'jobs' / (planned['name'] + '.json'))
            actual = replay.read(file)
            reference = bind(Path(actual['target']).with_suffix('.npz'))
            record = replay.read(bind(reference.with_suffix('.json')))
            sidecar = replay.read(bind(reference.with_suffix('.online.json')))
            precision = replay.read(bind(reference.with_suffix('.precision.json')))
            checked_job(planned, actual, record, sidecar)
            if (record['predictions_sha256'] != replay.digest(reference)
                    or precision.get('retained_precision') is not True
                    or precision.get('structural_intervention') != actual['structural_intervention']):
                raise ValueError('Changed reference array or precision intervention')
            job = dict(deepcopy(actual), reference=str(reference), reference_sha256=replay.digest(reference),
                original_job=str(file), original_job_sha256=replay.digest(file), replay_source_sha256=hashes,
                archived_pf_sha256=replay.digest(Path(actual['runtime']) / replay.PF),
                trial_limit=16 if smoke else 0, reported_arm='ONLINE' if actual['online_belief'] else 'BASE',
                parameter_estimation_scope='existing_behavior_parameters',
                evaluation_use='online_evidence_fixed_parameter_development',
                target=str((output / 'states' / actual['name']).resolve()))
            jobs.append(job)
    for job in jobs:
        job['refit_replay_inputs'] = dict(inputs)
        replay.verify_sources(job)
    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    for relative in hashes:
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        replay.shutil.copy2(ROOT / relative, target)
    replay.shutil.copy2(config, output / 'protocol.yaml')
    workers = min(1 if smoke else settings['workers'], len(replay.os.sched_getaffinity(0)), len(jobs))
    export_settings = dict(settings, subjects=[original['score_settings']['subject']],
                           arms=['BASE', 'ONLINE'], seeds_per_arm=original['filter_seeds'])
    protocol = dict(settings=export_settings, smoke=smoke, workers=workers, jobs=len(jobs),
        source_sha256=hashes, input_sha256=inputs, config_sha256=replay.digest(config),
        original_completion_sha256=replay.digest(source / 'completion.json'),
        parameter_estimation_scope='existing_behavior_parameters', evaluation_use='online_evidence_fixed_parameter_development',
        particles=original['particles'], parameter_candidate_policy='frozen_behavior_nominee_no_new_selection',
        candidate=baseline['candidate'], data_use='Behavior-only exact state replay; oral enters later alignment',
        python=sys.version, versions={name:replay.importlib.metadata.version(name) for name in ('numpy','scipy','pandas','PyYAML')},
        git_head=replay.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    replay.write(output / 'protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool = False) -> None:
    replay = helpers()
    replay.prepare, replay.ENTRY = prepare, ENTRY
    replay.run(config, output, smoke)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Use --config and --output, or an internal --worker')
