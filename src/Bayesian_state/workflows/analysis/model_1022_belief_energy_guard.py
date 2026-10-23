"""Independently confirm a proposed training-generation improvement.

Run --config YAML --output NEW [--smoke]. Original and proposed parameters
receive matched fresh training-only paths. A paired whole-path uncertainty
guard accepts a proposal only when its Energy difference plus the declared
MCSE multiplier is negative and the previous likelihood guard passed.
This development diagnostic preserves existing nominations and evaluations.
--smoke checks one participant with four eight-trial paths per parameter.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from . import model_1022_belief_training_paths as generation
from .model_1022_belief_summary import jackknife_se
from ...metrics.autonomous_fit import energy_u
from ...utils.seeding import stable_seed

pilot = generation.pilot
ROOT, read, write, digest = pilot.ROOT, pilot.read, pilot.write, pilot.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def paired_energy_guard(original: np.ndarray, proposed: np.ndarray,
                        observed: np.ndarray, multiplier: float) -> dict:
    """Preserve complete path pairing when estimating uncertainty in the gain."""
    original, proposed = np.asarray(original), np.asarray(proposed)
    if (original.shape != proposed.shape or original.ndim != 2 or len(original) < 4
            or not np.isfinite(multiplier) or multiplier < 0):
        raise ValueError('Require matched paths, at least four repeats and a finite nonnegative multiplier')
    old, new = energy_u(original, observed), energy_u(proposed, observed)
    difference = new['value'] - old['value']
    deleted = np.array([
        energy_u(np.delete(proposed, i, 0), observed)['value']
        - energy_u(np.delete(original, i, 0), observed)['value']
        for i in range(len(original))])
    se = jackknife_se(deleted)
    upper = difference + multiplier * se
    return dict(original_energy=old['value'], proposed_energy=new['value'],
                energy_difference=difference, difference_mcse=se, guard_value=upper,
                energy_accepted=bool(upper < 0), paths=len(original))


def validate_fresh_seeds(seeds: list[int], forbidden: set[int]) -> None:
    if len(seeds) != len(set(seeds)) or set(seeds).intersection(forbidden):
        raise ValueError('Confirmation seeds must be distinct and independent of earlier simulations')


def selected_candidate(decision: dict, energy: dict) -> str:
    """Both declared guards must pass before replacing the NLL nominee."""
    return decision['proposed'] if decision['accepted'] and energy['energy_accepted'] else decision['original']


def feature_paths(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read already validated generation chunks in their declared seed order."""
    parts, seeds, observed = [], [], None
    for target in case['targets']:
        with np.load(Path(target).with_suffix('.npz')) as z:
            parts.append(z['features'].copy())
            seeds.extend(z['seeds'].tolist())
            if observed is None:
                observed = z['observed_features'].copy()
            else:
                np.testing.assert_array_equal(observed, z['observed_features'])
    if seeds != case['seeds']:
        raise ValueError('Confirmation path order changed')
    return np.concatenate(parts), observed


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new independent generation-guard directory')
    policy = yaml.safe_load(config.read_text())
    if (policy['scope'] != 'independent_training_generation_guard'
            or policy['acceptance'] != 'energy_difference_plus_mcse_strictly_below_zero'
            or policy['fallback'] != 'retain_original_nll_nominee'
            or policy['maximum_confirmation_batches'] != 1
            or not pilot.valid_variants(policy['variants'])
            or policy['paths'] < 4 or policy['chunk'] < 4
            or policy['paths'] % policy['chunk'] != 0
            or not 1 <= policy['parallel_budget'] <= 128
            or not np.isfinite(policy['guard_mcse_multiplier']) or policy['guard_mcse_multiplier'] < 0):
        raise ValueError('Require the registered single-batch independent generation guard')
    study = ROOT / policy['source_study']
    inputs = {}

    def bind(file: Path, expected: str | None = None) -> None:
        file = file.resolve()
        sha = digest(file)
        if expected is not None and sha != expected:
            raise ValueError('Frozen selection input changed: ' + str(file))
        inputs[str(file)] = sha

    protocol, done, nomination, decisions, templates = [read(study / name) for name in
        ('protocol.json', 'completion.json', 'nominees.json', 'selection_decisions.json', 'autonomous_jobs.json')]
    if (done['status'] != 'complete' or done['smoke'] or nomination['smoke']
            or nomination['status'] != 'frozen_before_test' or nomination['later_choices_used']
            or nomination['oral_used'] or decisions['later_choices_used'] or decisions['oral_used']
            or protocol['policy']['cohort'] != policy['cohort']
            or protocol['policy']['subjects'] != policy['subjects']
            or protocol['policy']['variants'] != policy['variants']
            or protocol['policy']['features'] != policy['features']):
        raise ValueError('Require completed matching training-only selection records')
    summary = read(study / 'summary/completion.json')
    bind(study / 'nominees.json', summary['nominee_sha256'])
    bind(study / 'summary/summary.csv', done['summary_sha256'])
    for name in ('protocol.json', 'completion.json', 'selection_decisions.json',
                 'autonomous_jobs.json', 'summary/completion.json'):
        bind(study / name)
    initial = ROOT / protocol['policy']['source_study']
    old_nomination = read(initial / 'nominees.json')
    if (old_nomination['smoke'] or old_nomination['status'] != 'frozen_before_test'
            or old_nomination['later_choices_used'] or old_nomination['oral_used']):
        raise ValueError('Original parameters must have been nominated using training choices only')
    bind(initial / 'nominees.json', read(initial / 'summary/completion.json')['nominee_sha256'])
    old_jobs = read(initial / 'autonomous_jobs.json')['jobs']
    bind(initial / 'autonomous_jobs.json')
    ranking_file = Path(protocol['generation']) / 'cases.json'
    ranking_cases = read(ranking_file)['cases']
    bind(ranking_file)
    previous = {(r['subject'], r['arm']): r for r in old_nomination['nominees']}
    nominated = {(r['subject'], r['arm']): r for r in nomination['nominees']}
    expected = {(sid, arm) for sid in policy['subjects'] for arm in policy['variants']}
    if ({(d['subject'], d['arm']) for d in decisions['decisions']} != expected
            or len(decisions['decisions']) != len(expected) or set(nominated) != expected
            or len(nomination['nominees']) != len(expected)):
        raise ValueError('Selection records must cover every participant/model once')
    selected_subjects = policy['subjects'][:1] if smoke else policy['subjects']
    hashes = dict(generation.sources(), **{ENTRY: digest(ROOT / ENTRY)})
    cases, jobs, active_decisions = [], [], []
    for sid in selected_subjects:
        forbidden = {seed for c in ranking_cases if c['subject'] == sid for seed in c['seeds']}
        forbidden.update(seed for j in templates['jobs'] + old_jobs if j['subject'] == sid for seed in j['seeds'])
        seeds = [stable_seed(dict(protocol=policy['protocol'], base=policy['seed'], subject=sid,
                    role='training_autonomous', repeat=i)) for i in range(4 if smoke else policy['paths'])]
        validate_fresh_seeds(seeds, forbidden)
        for decision in decisions['decisions']:
            if decision['subject'] != sid:
                continue
            arm, key = decision['arm'], (sid, decision['arm'])
            row = nominated[key]
            if (decision['original'] != previous[key]['candidate']['id']
                    or row['candidate']['id'] != (decision['proposed'] if decision['accepted'] else decision['original'])):
                raise ValueError('Original or likelihood-selected parameter identity changed')
            bank = {point['id']: point for point in row['finalists']}
            confirmed = {point['id']: point['hyperparams'] for point in previous[key]['finalists']}
            template = next(j for j in templates['jobs'] if j['subject'] == sid and j['model'] == arm)
            active_decisions.append(decision)
            for candidate_id in dict.fromkeys((decision['original'], decision['proposed'])):
                point = bank[candidate_id]
                if point['hyperparams'] != confirmed[candidate_id]:
                    raise ValueError('Proposed parameters differ from the original confirmed candidate bank')
                score_file = Path(point['score_file'])
                score = read(score_file)
                if score['subject'] != sid or score['candidate']['hyperparams'] != point['hyperparams']:
                    raise ValueError('Guarded training parameter changed')
                bind(score_file)
                bind(score_file.with_suffix('.npz'), score['predictions_sha256'])
                with np.load(score_file.with_suffix('.npz')) as z:
                    validate_fresh_seeds(seeds, set(z['seeds'].tolist()))
                targets = []
                for i in range(0, len(seeds), policy['chunk']):
                    name = f'energy_guard_S{sid}_{arm}_{candidate_id}_C{i:03d}'
                    target = str((output / 'generated' / name).resolve())
                    job = deepcopy(template)
                    job.update(name=name, case_id=sid, candidate=point, seeds=seeds[i:i+policy['chunk']], smoke=smoke,
                               features=policy['features'], source_sha256=hashes, target=target)
                    jobs.append(job)
                    targets.append(target)
                cases.append(dict(subject=sid, arm=arm, candidate=point, folder=template['folder'],
                    original_nominee=candidate_id == decision['original'], smoke=smoke, targets=targets, seeds=seeds))
    output.mkdir(parents=True)
    pilot.driver.base.archive(output, config, hashes)
    write(output / 'protocol.json', dict(policy=policy, smoke=smoke, source_sha256=hashes,
        inputs_sha256=inputs, python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        decision='independent paired training Energy guard; existing study predictions and nominations remain unchanged',
        development_review_preceded_guard=True, later_choices_read=False, oral_used=False))
    write(output / 'cases.json', dict(cases=cases))
    generation.execute(jobs, output, 1 if smoke else policy['parallel_budget'])
    scores = [generation.combine(c, policy['features'], inputs) for c in cases]
    index = {(c['subject'], c['arm'], c['candidate']['id']): c for c in cases}
    checked = []
    for decision in active_decisions:
        key = decision['subject'], decision['arm']
        old, new = [index[*key, decision[name]] for name in ('original', 'proposed')]
        if old['seeds'] != new['seeds']:
            raise ValueError('Original and proposed paths must share seed order')
        x, y = feature_paths(old)
        proposed, target = feature_paths(new)
        np.testing.assert_array_equal(y, target)
        energy = paired_energy_guard(x, proposed, y, policy['guard_mcse_multiplier'])
        chosen = selected_candidate(decision, energy)
        checked.append(dict(subject=key[0], arm=key[1], original=decision['original'], proposed=decision['proposed'],
            likelihood_accepted=decision['accepted'], **energy, selected=chosen,
            accepted=bool(decision['accepted'] and energy['energy_accepted'])))
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Frozen input changed during confirmation: ' + file)
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    pd.DataFrame(checked).to_csv(output / 'decisions.csv', index=False)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, groups=len(checked), cases=len(cases),
        paths=sum(len(c['seeds']) for c in cases), nominees_changed=False, later_choices_read=False, oral_used=False,
        inputs_sha256=inputs, source_sha256=hashes, decisions_sha256=digest(output / 'decisions.csv'),
        scope='software check' if smoke else 'development confirmation after outcome diagnosis; extension evaluation pending'))
    print(pd.DataFrame(checked).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)
