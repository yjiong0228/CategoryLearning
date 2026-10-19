"""Replay the three new SEARCH mechanism/parameter cells for oral validation.

Use --config YAML --output NEW_DIRECTORY [--smoke]. The two original nominees
retain their existing state banks; this command replays ONLINE_RETAIN,
RETAIN_ONLINE and ONLINE_NEAR only. It preserves every executed conditional
seed and requires an exact match to saved choice predictions. --smoke uses one
seed per cell, at most sixteen trials and one worker; completed eight-trial
cross fixtures are accepted in this mode. Oral content enters a later alignment.
The adapter reuses the existing state observer and intervention worker.
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
OBSERVER = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_states.py'
CELLS = ('ONLINE_RETAIN', 'RETAIN_ONLINE', 'ONLINE_NEAR')
SCOPE = 'prefix_selected_parameter_cross'
USE = 'fixed_mechanism_parameter_cross_development'


def helpers():
    spec = importlib.util.spec_from_file_location('parameter_cross_state_observer', ROOT / OBSERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_identity(cell: str, planned: dict, actual: dict, record: dict,
                   precision: dict, online: dict | None) -> None:
    """Keep the mechanism, candidate, seeds and trial-level execution intact."""
    if cell not in CELLS:
        raise ValueError('Only the three new cross cells need state replay')
    mechanism, parameter = cell.split('_')
    if (actual != planned or actual.get('cross_mechanism') != mechanism
            or actual.get('cross_parameter') != parameter or actual.get('model') != 'SEARCH'
            or actual.get('phase') != 'evaluate' or actual.get('retain_precision') is not True
            or bool(actual.get('online_belief')) != (mechanism == 'ONLINE')
            or len(actual['seeds']) != 1
            or any(record[k] != actual[k] for k in ('candidate', 'model', 'seeds', 'particles'))
            or precision.get('retained_precision') is not True
            or precision.get('structural_intervention') != actual['structural_intervention']
            or precision.get('transitions') != actual['particles'] * record['trials']):
        raise ValueError('Cross filter identity or retained precision differs')
    if mechanism == 'ONLINE':
        if (online is None or online.get('enabled') is not True
                or online.get('exact_transition_timing') is not True
                or online.get('evidence_matches_retained_state') is not True
                or online.get('reconstructions') != actual['particles'] * (record['trials'] - 1)):
            raise ValueError('Cross ONLINE execution differs')
    elif online is not None:
        raise ValueError('RETAIN cell unexpectedly contains an ONLINE execution')


def worker(file: Path) -> None:
    observer = helpers()
    job = observer.helpers().read(file)
    if (job.get('reported_arm') not in CELLS
            or job['reported_arm'] != job['cross_mechanism'] + '_' + job['cross_parameter']
            or job.get('parameter_estimation_scope') != SCOPE or job.get('evaluation_use') != USE
            or job['online_belief'] != (job['cross_mechanism'] == 'ONLINE')):
        raise ValueError('Replay must retain its explicit cross-cell identity and scope')
    observer.worker(file)


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, list[dict]]:
    observer = helpers(); replay = observer.helpers()
    settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError('Use a new cross-state destination')
    if (settings['sample_role'] != 'development' or settings['cells'] != list(CELLS)
            or not 1 <= settings['workers'] <= 128):
        raise ValueError('Expected the declared development cells and bounded worker budget')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); digest = replay.digest(path)
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError('Cross-state input changed')
        inputs[str(path)] = digest
        return path

    source = ROOT / settings['source']; bind(config)
    original = replay.read(bind(source / 'protocol.json'))
    done = replay.read(bind(source / 'completion.json'))
    if (done['status'] != 'complete' or done['nominees_changed'] or done['cells'] != 5
            or done['protocol_sha256'] != replay.digest(source / 'protocol.json')
            or done['smoke'] != original['smoke'] or (done['smoke'] and not smoke)):
        raise ValueError('Require a complete cross; short fixtures need --smoke')
    for mapping in (done['inputs_sha256'], original['sources']):
        for path, digest in mapping.items():
            if replay.digest(bind(ROOT / path)) != digest:
                raise ValueError('Cross execution input or source changed')
    banks = {b['cell']: b for b in original['banks']}
    if len(banks) != 5 or not set(CELLS) <= set(banks):
        raise ValueError('Missing cross-cell plan')
    hashes = {p: replay.digest(ROOT / p) for p in
              (ENTRY, OBSERVER, observer.REPLAY, observer.ONLINE, replay.PF,
               replay.COLLECTOR, replay.REFIT, replay.PROPOSAL, replay.REENTRY)}
    jobs, selected, seed_banks = [], {}, []
    for cell in CELLS:
        bank = banks[cell]; selected[cell] = bank['candidate']
        plans = bank['roles']['evaluate']; seeds = [item['job']['seeds'][0] for item in plans]
        if len(set(seeds)) != len(seeds) or (not original['smoke'] and len(seeds) != 8):
            raise ValueError('Incomplete independent filter seed bank')
        seed_banks.append(seeds)
        for item in plans[:1] if smoke else plans:
            if item['reused']:
                raise ValueError('Original nominee state banks are reused separately')
            planned = item['job']; file = bind(source / 'jobs' / (planned['name'] + '.json'))
            actual = replay.read(file); target = Path(actual['target'])
            reference = bind(target.with_suffix('.npz'))
            record = replay.read(bind(target.with_suffix('.json')))
            precision = replay.read(bind(target.with_suffix('.precision.json')))
            sidecar = target.with_suffix('.online.json')
            online = replay.read(bind(sidecar)) if sidecar.exists() else None
            check_identity(cell, planned, actual, record, precision, online)
            if actual['candidate'] != bank['candidate'] or record['predictions_sha256'] != replay.digest(reference):
                raise ValueError('Cross parameter point or saved predictions changed')
            job = dict(deepcopy(actual), reference=str(reference), reference_sha256=replay.digest(reference),
                original_job=str(file), original_job_sha256=replay.digest(file), replay_source_sha256=hashes,
                archived_pf_sha256=replay.digest(Path(actual['runtime']) / replay.PF),
                trial_limit=16 if smoke else 0, reported_arm=cell, online_belief=bank['mechanism'] == 'ONLINE',
                parameter_estimation_scope=SCOPE, evaluation_use=USE,
                target=str((output / 'states' / actual['name']).resolve()))
            jobs.append(job)
    if any(seeds != seed_banks[0] for seeds in seed_banks):
        raise ValueError('Cross cells must share the original seed schedule')
    if len({job['particles'] for job in jobs}) != 1:
        raise ValueError('Cross cells must share the particle budget')
    for job in jobs:
        job['refit_replay_inputs'] = dict(inputs)
        replay.verify_sources(job)
    output.mkdir(parents=True); (output / 'jobs').mkdir()
    for relative in hashes:
        target = output / 'source' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        replay.shutil.copy2(ROOT / relative, target)
    replay.shutil.copy2(config, output / 'protocol.yaml')
    workers = min(1 if smoke else settings['workers'], len(replay.os.sched_getaffinity(0)), len(jobs))
    exported = dict(settings, subjects=original['settings']['subjects'], arms=list(CELLS),
                    seeds_per_arm=len(seed_banks[0]))
    protocol = dict(settings=exported, smoke=smoke, workers=workers, jobs=len(jobs),
        source_sha256=hashes, input_sha256=inputs, config_sha256=replay.digest(config),
        original_completion_sha256=replay.digest(source / 'completion.json'),
        parameter_estimation_scope=SCOPE, evaluation_use=USE, selected_parameters=selected,
        parameter_candidate_policy='frozen_behavior_nominees_and_declared_near_tie',
        particles=jobs[0]['particles'], data_use='Behavior-only exact state replay; oral enters later alignment',
        original_nominations_changed=False, python=sys.version,
        versions={name: replay.importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        git_head=replay.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    replay.write(output / 'protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool = False) -> None:
    replay = helpers().helpers()
    replay.prepare, replay.ENTRY = prepare, ENTRY
    replay.run(config, output, smoke)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Use --config and --output, or an internal --worker')
