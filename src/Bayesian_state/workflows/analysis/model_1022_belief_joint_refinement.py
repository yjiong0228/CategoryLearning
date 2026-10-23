"""Refit a bounded common bank on both training likelihood and generation.

Run --config YAML --output NEW [--smoke]. Preserve every previous confirmed
point and add a declared factorial neighborhood of a training-selected point.
Every arm evaluates the same bank with fresh filters and training paths. The
registered Energy-within-NLL-tolerance rule and independent likelihood guard
then freeze parameters before full-sequence prediction and autonomous testing.
No cognitive module is changed. --smoke uses two points per arm and a short
fixture made from original training trials, with one numerical worker.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from . import model_1022_belief_balanced_selection as balanced
from . import model_1022_belief_comparison as pilot
from . import model_1022_belief_training_paths as generation
from .model_1022_compact_evaluate import smoke_fixture
from ...optimization.search.adaptive_proposals import candidate
from ...utils.seeding import stable_seed

ROOT, read, write, digest = pilot.ROOT, pilot.read, pilot.write, pilot.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PROPOSALS = 'src/Bayesian_state/optimization/search/adaptive_proposals.py'


def sources(v) -> dict:
    return dict(balanced.sources(v), **{name: digest(ROOT / name) for name in (ENTRY, PROPOSALS)})


def expand_bank(previous: list[dict], anchor: dict, axes: dict, space: dict) -> list[dict]:
    """Preserve old points and vary only the declared coordinates of the anchor."""
    if not axes or any(key not in space or not values or len(set(values)) != len(values)
                       or any(v not in space[key] for v in values) for key, values in axes.items()):
        raise ValueError('Factorial axes must be distinct values inside the original support')
    points = [r['hyperparams'] for r in previous] + [anchor]
    if any(set(p) != set(space) or any(p[k] not in space[k] for k in space) for p in points):
        raise ValueError('Every existing parameter point must match the original support')
    bank = {candidate(p, 'previous_training_confirmation')['id']:
            candidate(p, 'previous_training_confirmation') for p in points}
    keys = list(axes)
    for values in product(*(axes[k] for k in keys)):
        point = {**deepcopy(anchor), **dict(zip(keys, values))}
        row = candidate(point, 'registered_factorial_neighborhood')
        bank.setdefault(row['id'], row)
    return list(bank.values())


def frozen_source(folder: Path, inputs: dict) -> tuple[dict, dict]:
    """Bind a completed source's training nominations without reading test scores."""
    protocol, nominees, done = [read(folder / name) for name in
                                 ('protocol.json', 'nominees.json', 'completion.json')]
    if (done['status'] != 'complete' or nominees['smoke']
            or nominees['status'] != 'frozen_before_test' or nominees['later_choices_used']
            or nominees['oral_used']
            or read(folder / 'summary/completion.json')['nominee_sha256'] != digest(folder / 'nominees.json')):
        raise ValueError('Source requires intact frozen training nominations')
    for name in ('protocol.json', 'nominees.json', 'completion.json', 'summary/completion.json',
                 'autonomous_jobs.json'):
        inputs[str((folder / name).resolve())] = digest(folder / name)
    return protocol, nominees


def verify_generation_job(job: dict, saved: Path, inputs: dict) -> None:
    """Check the numerical worker used the requested mechanism and all inputs."""
    expected = deepcopy(job)
    expected.update(comparison_arm=job['model'], model='COMPACT')
    expected['mechanism']['capacity'] = 'evidence'
    if read(saved) != expected:
        raise ValueError('Training generation worker differs from the declared job')
    inputs[str(saved.resolve())] = digest(saved)


def train_generation(confirmed: dict, settings: dict, policy: dict, templates: list[dict],
                     output: Path, inputs: dict, smoke: bool) -> list[dict]:
    """Generate every confirmed point from the task start using training only."""
    cases, jobs = [], []
    for (sid, arm), points in confirmed.items():
        template = next(j for j in templates if j['subject'] == sid and j['model'] == arm)
        folder = (ROOT / settings['data'] / f'subject_{sid}').resolve()
        seeds = [stable_seed(dict(protocol=policy['protocol'], base=policy['seed'], subject=sid,
                                  role='training_generation', repeat=i))
                 for i in range(4 if smoke else policy['training_paths'])]
        width = 4 if smoke else policy['training_chunk']
        for point in points:
            targets = []
            for i in range(0, len(seeds), width):
                name = f'training_S{sid}_{arm}_{point["id"]}_C{i:03d}'
                target = str((output / 'training_generation' / name).resolve())
                job = deepcopy(template)
                # The software fixture already contains only four training
                # trials. Full training mode scores those four consistently.
                job.update(name=name, case_id=sid, candidate=point, folder=str(folder), seeds=seeds[i:i+width],
                           features=policy['features'], smoke=False, source_sha256=sources(pilot.driver.io),
                           target=target)
                jobs.append(job)
                targets.append(target)
            cases.append(dict(subject=sid, arm=arm, candidate=point, folder=str(folder),
                original_nominee=point['id'] == points[0]['id'], smoke=False, targets=targets, seeds=seeds))
    write(output / 'training_generation_cases.json', dict(cases=cases, software_fixture=smoke))
    generation.execute(jobs, output, settings['parallel_budget'])
    for job in jobs:
        verify_generation_job(job, output / 'jobs' / (job['name'] + '.json'), inputs)
    rows = [generation.combine(case, policy['features'], inputs) for case in cases]
    pd.DataFrame(rows).to_csv(output / 'training_generation.csv', index=False)
    return rows


def guarded_nominees(confirmed: dict, checked: dict, proposals: list[dict], policy: dict,
                     inputs: dict) -> tuple[list[dict], list[dict]]:
    """Use fresh paired filters to accept an Energy proposal or retain the NLL point."""
    decisions, nominees = [], []
    for proposal in proposals:
        key = proposal['subject'], proposal['arm']
        rows = {r['id']: r for r in checked[key]}
        old, new = rows[proposal['original']], rows[proposal['proposed']]
        arrays = []
        for point in (old, new):
            file = Path(point['score_file'])
            meta = read(file)
            if (meta['subject'] != key[0] or meta['candidate']['id'] != point['id']
                    or meta['candidate']['hyperparams'] != point['hyperparams']
                    or digest(file.with_suffix('.npz')) != meta['predictions_sha256']):
                raise ValueError('Likelihood guard identity or probability file changed')
            for bound in (file, file.with_suffix('.npz')):
                inputs[str(bound.resolve())] = digest(bound)
            with np.load(file.with_suffix('.npz')) as z:
                arrays.append((z['observed_probability'], z['seeds']))
        np.testing.assert_array_equal(arrays[0][1], arrays[1][1])
        decision = balanced.likelihood_guard(arrays[0][0], arrays[1][0], policy['training_nll_tolerance'],
                                             policy['guard_mcse_multiplier'])
        chosen = new if decision['accepted'] else old
        decisions.append(dict(proposal, **decision, selected=chosen['id']))
        nominees.append(dict(subject=key[0], arm=key[1], candidate=chosen, finalists=confirmed[key]))
    return decisions, nominees


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new joint-refinement directory')
    policy = yaml.safe_load(config.read_text())
    if (policy['scope'] != 'development_joint_training_bank_refinement'
            or policy['variants'] != ['SUPPORT', 'BELIEF', 'SLOT']
            or policy['fallback'] != 'retain_new_bank_nll_nominee'
            or not 1 <= policy['parallel_budget'] <= 128
            or policy['training_chunk'] < 3 or policy['training_paths'] < 3
            or policy['training_paths'] % policy['training_chunk']):
        raise ValueError('Require the registered common-bank training policy')
    inputs = {}
    source, seed_source = ROOT / policy['source_study'], ROOT / policy['seed_study']
    source_protocol, old = frozen_source(source, inputs)
    seed_protocol, seed_nominees = frozen_source(seed_source, inputs)
    settings = deepcopy(source_protocol['settings'])
    if seed_protocol['settings']['data'] != settings['data']:
        raise ValueError('Source training datasets differ')
    banks = {}
    for sid in policy['subjects']:
        rows = {r['arm']: r for r in old['nominees'] if r['subject'] == sid}
        if set(rows) != set(policy['variants']):
            raise ValueError('Source must include all three mechanisms')
        previous = rows['SUPPORT']['finalists']
        expected = {r['id']: r['hyperparams'] for r in previous}
        if any({r['id']: r['hyperparams'] for r in row['finalists']} != expected for row in rows.values()):
            raise ValueError('Previous confirmed parameter banks must match across arms')
        anchor = next(r['candidate']['hyperparams'] for r in seed_nominees['nominees']
                      if r['subject'] == sid and r['arm'] == policy['seed_arm'])
        space_file = ROOT / source_protocol['policy']['warm_fit'] / f'S{sid}_TRANSFER/nominees.json'
        inputs[str(space_file.resolve())] = digest(space_file)
        condition = read(ROOT / settings['data'] / f'subject_{sid}/prepared.json')['condition']
        space = pilot.driver.model_space(read(space_file)['settings'], 'COMPACT', condition)
        bank = expand_bank(previous, anchor, policy['axes'], space)
        if len(bank) != policy['expected_candidates_per_subject']:
            raise ValueError('Registered common candidate count changed')
        for arm in policy['variants']:
            banks[sid, arm] = deepcopy(bank[:2] if smoke else bank)
    settings.update({k: deepcopy(policy[k]) for k in ('protocol', 'seed', 'subjects', 'parallel_budget',
                                                     'budgets', 'autonomous_paths', 'autonomous_chunk')})
    output.mkdir(parents=True)
    if smoke:
        if len(settings['subjects']) != 1:
            raise ValueError('Software validation uses one participant')
        settings['parallel_budget'] = 1
        fixture = smoke_fixture(ROOT / settings['data'], output / 'software_fixture', settings['subjects'][0])
        settings['data'] = str(fixture.resolve())
    pilot.driver.base.source_hashes, pilot.driver.base.execute = sources, pilot.execute
    pilot.driver.base.archive(output, config, sources(pilot.driver.io))
    write(output / 'protocol.json', dict(settings=settings, policy=policy, smoke=smoke,
        source_sha256=sources(pilot.driver.io), inputs_sha256=inputs, python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        decision='Bounded common-bank refits using training NLL and training Energy with a paired NLL guard'))
    write(output / 'starting_banks.json', dict(banks=[dict(subject=sid, arm=arm, candidates=points)
          for (sid, arm), points in banks.items()], inputs_sha256=inputs))
    stage = lambda bank, name: pilot.driver.base.stage(settings, ROOT / settings['data'], output, bank, name, smoke)
    confirmed = stage(banks, 'confirmation')
    templates = read(source / 'autonomous_jobs.json')['jobs']
    scores = train_generation(confirmed, settings, policy, templates, output, inputs, smoke)
    proposals, guard_banks = [], {}
    for key, points in confirmed.items():
        selected = balanced.choose_energy([r for r in scores if (r['subject'], r['arm']) == key],
                                         policy['training_nll_tolerance'])
        proposed = next(r for r in points if r['id'] == selected['candidate'])
        guard_banks[key] = list({r['id']: r for r in (points[0], proposed)}.values())
        proposals.append(dict(subject=key[0], arm=key[1], original=points[0]['id'], proposed=proposed['id'],
                              training_generation_score=selected))
    write(output / 'energy_proposals.json', dict(proposals=proposals, candidate_scores=scores,
                                               later_choices_used=False, oral_used=False))
    checked = stage(guard_banks, 'guard')
    decisions, nominees = guarded_nominees(confirmed, checked, proposals, policy, inputs)
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Training selection input changed before nomination')
    write(output / 'selection_decisions.json', dict(decisions=decisions, inputs_sha256=inputs,
                                                   later_choices_used=False, oral_used=False))
    write(output / 'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
                                        later_choices_used=False, oral_used=False))
    stage({(r['subject'], r['arm']): [r['candidate']] for r in nominees}, 'evaluate')
    jobs = []
    for row in nominees:
        sid, arm = row['subject'], row['arm']
        template = next(j for j in templates if j['subject'] == sid and j['model'] == arm)
        seeds = [stable_seed(dict(base=policy['seed'], subject=sid, stage='autonomous', repeat=i))
                 for i in range(4 if smoke else policy['autonomous_paths'])]
        for i in range(0, len(seeds), policy['autonomous_chunk']):
            name = f'autonomous_S{sid}_{arm}_{i:03d}'
            job = deepcopy(template)
            job.update(name=name, candidate=row['candidate'], folder=str((ROOT / settings['data'] / f'subject_{sid}').resolve()),
                       seeds=seeds[i:i+policy['autonomous_chunk']], smoke=smoke,
                       source_sha256=sources(pilot.driver.io), target=str((output / 'autonomous' / name).resolve()))
            jobs.append(job)
    write(output / 'autonomous_jobs.json', dict(jobs=jobs))
    pilot.execute(jobs, output, settings['parallel_budget'])
    if smoke:
        write(output / 'completion.json', dict(status='complete', smoke=True, cases=len(nominees),
            scope='Four training trials, eight evaluation trials, sixteen autonomous trials; software only'))
    else:
        pilot.summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)
