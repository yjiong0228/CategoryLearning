"""Select frozen belief-model candidates using training generation and an NLL guard.

Use --config YAML --generation COMPLETED_GENERATION --output NEW [--smoke].
Eligible points lie within the declared original training-NLL tolerance.
Independent training Energy selects one point per model. Fresh particle
filters compare it with that model's original nominee; failure of the paired
NLL guard retains the original point. Only then are later predictions scored.
An optional generation_guard mapping adds fresh paired training paths before
freezing. Both guards must pass to replace the original NLL nominee.
This bounded selection keeps the searched candidate bank and cognitive model
unchanged. --smoke exercises three arms on short sequences with one worker.
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

from . import model_1022_belief_comparison as pilot
from . import model_1022_belief_training_paths as generation
from . import model_1022_belief_energy_guard as energy_guard
from .model_1022_belief_summary import nll, jackknife_se
from ...utils.seeding import stable_seed

ROOT, read, write, digest = pilot.ROOT, pilot.read, pilot.write, pilot.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
SMOKE_FIXTURE = 'src/Bayesian_state/workflows/analysis/model_1022_compact_evaluate.py'


def sources(v) -> dict:
    return dict(pilot.sources(v), **generation.sources(),
                **{name: digest(ROOT / name) for name in (ENTRY, SMOKE_FIXTURE, energy_guard.ENTRY)})


def valid_scope(policy: dict) -> bool:
    cohort = policy.get('cohort', 'development')
    return (cohort in ('development', 'extension')
            and policy['scope'] == f'{cohort}_training_likelihood_constrained_generation_selection')


def generation_guard_policy(policy: dict, output: Path) -> dict | None:
    """Inherit cohort and feature definitions without allowing conflicting overrides."""
    if 'generation_guard' not in policy:
        return None
    supplied = policy['generation_guard']
    required = {'protocol', 'seed', 'paths', 'chunk', 'guard_mcse_multiplier',
                'maximum_confirmation_batches', 'acceptance'}
    if not isinstance(supplied, dict) or set(supplied) != required:
        raise ValueError('Generation guard must declare exactly its protocol, seeds, budget and acceptance rule')
    result = dict(deepcopy(supplied), source_study=str(output.resolve()),
                  scope='independent_training_generation_guard', fallback=policy['fallback'])
    result.update({key: deepcopy(policy[key]) for key in
                   ('subjects', 'variants', 'parallel_budget', 'features', 'cohort')})
    energy_guard.validate_policy(result)
    return result


def confirm_generation(guard_policy: dict, study: Path, generated: Path,
                       output: Path, nominees: list[dict], decisions: list[dict],
                       templates: list[dict], inputs: dict, smoke: bool) -> None:
    """Apply the generation decision before any final nomination or evaluation."""
    likelihood_file = output / 'likelihood_decisions.json'
    candidates_file = output / 'likelihood_candidates.json'
    write(likelihood_file, dict(decisions=decisions, later_choices_used=False, oral_used=False))
    write(candidates_file, dict(nominees=nominees, scope='provisional likelihood-guard candidates'))
    guard_config = output / 'generation_guard_policy.yaml'
    with guard_config.open('x') as stream:
        yaml.safe_dump(guard_policy, stream, sort_keys=False)
    confirmation_inputs = dict(inputs)
    for file in (likelihood_file, candidates_file, guard_config, study / 'nominees.json',
                 study / 'autonomous_jobs.json', generated / 'cases.json'):
        confirmation_inputs[str(file.resolve())] = digest(file)
    result = energy_guard.confirm(guard_config, output / 'generation_guard', policy=guard_policy,
        nomination=dict(nominees=nominees), decisions=dict(decisions=decisions),
        old_nomination=read(study / 'nominees.json'), templates=dict(jobs=templates),
        old_jobs=read(study / 'autonomous_jobs.json')['jobs'],
        ranking_cases=read(generated / 'cases.json')['cases'], inputs=confirmation_inputs,
        smoke=smoke, before_evaluation=True, extra_sources=sources(pilot.driver.io))
    by_key = {(r['subject'], r['arm']): r for r in result}
    if len(by_key) != len(nominees) or set(by_key) != {(r['subject'], r['arm']) for r in nominees}:
        raise ValueError('Generation confirmation omitted or duplicated a model case')
    for row, decision in zip(nominees, decisions):
        if (row['subject'], row['arm']) != (decision['subject'], decision['arm']):
            raise ValueError('Likelihood decision order differs from the provisional candidate order')
        checked = by_key[row['subject'], row['arm']]
        if any(checked[key] != decision[key] for key in ('original', 'proposed')):
            raise ValueError('Generation confirmation changed the parameter pair')
        decision.update(likelihood_accepted=decision['accepted'], generation_accepted=checked['energy_accepted'],
                        accepted=checked['accepted'], selected=checked['selected'],
                        generation_guard_value=checked['guard_value'])
        row['candidate'] = next(point for point in row['finalists'] if point['id'] == checked['selected'])
    for name in ('completion.json', 'decisions.csv'):
        file = output / 'generation_guard' / name
        inputs[str(file.resolve())] = digest(file)


def choose_energy(rows: list[dict], tolerance: float) -> dict:
    """Minimize training Energy inside the frozen likelihood tolerance."""
    if not rows or tolerance < 0 or len({r['candidate'] for r in rows}) != len(rows):
        raise ValueError('Require distinct candidates and a nonnegative NLL tolerance')
    values = np.array([[r['confirmation_nll'], r['training_energy']] for r in rows])
    if not np.isfinite(values).all():
        raise ValueError('Candidate training scores must be finite')
    minimum = min(r['confirmation_nll'] for r in rows)
    eligible = [r for r in rows if r['confirmation_nll'] <= minimum + tolerance]
    return min(eligible, key=lambda r: (r['training_energy'], r['confirmation_nll'], r['candidate']))


def likelihood_guard(original: np.ndarray, proposed: np.ndarray, tolerance: float, multiplier: float) -> dict:
    """Use complete paired filter repeats for numerical uncertainty in the guard."""
    if (original.shape != proposed.shape or original.ndim != 2 or len(original) < 2
            or np.any(original > 1) or np.any(proposed > 1) or tolerance < 0 or multiplier < 0):
        raise ValueError('Require matched probability arrays and nonnegative guard settings')
    difference = nll(proposed) - nll(original)
    deleted = np.array([nll(np.delete(proposed, i, 0)) - nll(np.delete(original, i, 0))
                        for i in range(len(original))])
    se = jackknife_se(deleted)
    upper = difference + multiplier * se
    return dict(original_nll=nll(original), proposed_nll=nll(proposed), nll_difference=difference,
                difference_mcse=se, guard_value=upper, accepted=bool(upper <= tolerance))


def inspect_generation(policy: dict, folder: Path, smoke: bool) -> tuple[list[dict], dict, dict]:
    """Rebuild all training scores and verify the prospective generation policy."""
    done, protocol = read(folder / 'completion.json'), read(folder / 'protocol.json')
    if (done['status'] != 'complete' or done['smoke'] is not smoke or done['nominees_changed']
            or done['later_choices_read'] or done['oral_used']):
        raise ValueError('Require completed matching training-only generation')
    gp = protocol['policy']
    if not smoke:
        if (gp.get('selection_policy') != policy or gp['paths'] != policy['generation_confirmation_paths']
                or gp['chunk'] != policy['generation_chunk'] or gp['seed'] != policy['seed']
                or gp['protocol'] != policy['protocol'] + '_generation'):
            raise ValueError('Generation differs from the registered selection policy')
    if gp['source_study'] != policy['source_study'] or gp['features'] != policy['features']:
        raise ValueError('Generation source or feature definition changed')
    study = ROOT / policy['source_study']
    nominees = read(study / 'nominees.json')
    if nominees['smoke'] or nominees['later_choices_used'] or nominees['oral_used']:
        raise ValueError('Original nominees must be scientific training-only estimates')
    original = {(r['subject'], r['arm']): r for r in nominees['nominees']}
    cases = read(folder / 'cases.json')['cases']
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Frozen generation input changed: ' + str(path))
        inputs[str(path)] = sha

    for name in ('completion.json', 'protocol.json', 'cases.json', 'summary.csv'):
        bind(folder / name, done['summary_sha256'] if name == 'summary.csv' else None)
    for name, sha in done['inputs_sha256'].items():
        bind(name, sha)
    for name, sha in protocol['source_sha256'].items():
        bind(folder / 'source' / name, sha)
    for name in ('protocol.json', 'nominees.json', 'autonomous_jobs.json'):
        bind(study / name)
    expected_subjects = {gp['smoke_subject']} if smoke else set(policy['subjects'])
    expected_cases = {(sid, arm) for sid in expected_subjects for arm in policy['variants']}
    if {(c['subject'], c['arm']) for c in cases} != expected_cases:
        raise ValueError('Generation does not cover every declared participant/model')
    scores = []
    for key in sorted(expected_cases):
        selected = [c for c in cases if (c['subject'], c['arm']) == key]
        allowed = [original[key]['candidate']] if smoke else original[key]['finalists']
        bank = {r['id']: r['hyperparams'] for r in allowed}
        if ({c['candidate']['id']: c['candidate']['hyperparams'] for c in selected} != bank
                or len(selected) != len(bank)):
            raise ValueError('Generation candidate bank changed')
        for case in selected:
            confirmed = next(r for r in allowed if r['id'] == case['candidate']['id'])
            if (case['candidate']['mean_nll'] != confirmed['mean_nll']
                    or case['original_nominee'] != (confirmed['id'] == original[key]['candidate']['id'])):
                raise ValueError('Original likelihood score or nomination marker changed')
            if not smoke:
                expected_seeds = [stable_seed(dict(protocol=gp['protocol'], base=gp['seed'], subject=key[0],
                    role='training_autonomous', repeat=i)) for i in range(gp['paths'])]
                if case['seeds'] != expected_seeds:
                    raise ValueError('Generation seed bank changed')
            seen = []
            for target in case['targets']:
                file = folder / 'jobs' / (Path(target).name + '.json')
                job = read(file)
                bind(file)
                if (job['comparison_arm'] != case['arm'] or job['subject'] != case['subject']
                        or job['candidate']['id'] != case['candidate']['id']
                        or job['candidate']['hyperparams'] != case['candidate']['hyperparams']
                        or job['folder'] != case['folder'] or job['source_sha256'] != protocol['source_sha256']
                        or job['phase'] != 'autonomous' or job['features'] != policy['features']):
                    raise ValueError('Generation worker differs from its declared mechanism or inputs')
                seen.extend(job['seeds'])
            if seen != case['seeds']:
                raise ValueError('Generation worker seed order differs')
            scores.append(generation.combine(case, policy['features'], inputs))
    actual = pd.DataFrame(scores).sort_values(['subject', 'arm', 'candidate']).reset_index(drop=True)
    saved = pd.read_csv(folder / 'summary.csv').sort_values(['subject', 'arm', 'candidate']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, saved, check_exact=False, rtol=1e-13, atol=1e-14)
    return scores, original, inputs


def run(config: Path, generated: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new balanced-selection output directory')
    policy = yaml.safe_load(config.read_text())
    if (not pilot.valid_variants(policy['variants'])
            or policy['fallback'] != 'retain_original_nll_nominee'
            or not valid_scope(policy)
            or not 1 <= policy['parallel_budget'] <= 128):
        raise ValueError('Require the registered balanced-selection policy')
    guard_policy = generation_guard_policy(policy, output)
    scores, original, inputs = inspect_generation(policy, generated, smoke)
    study = ROOT / policy['source_study']
    settings = deepcopy(read(study / 'protocol.json')['settings'])
    settings.update({k: deepcopy(policy[k]) for k in ('protocol', 'seed', 'subjects', 'parallel_budget',
                                                     'budgets', 'autonomous_paths', 'autonomous_chunk')})
    keys = sorted({(r['subject'], r['arm']) for r in scores})
    if smoke:
        settings['subjects'] = sorted({sid for sid, _ in keys})
        settings['parallel_budget'] = 1
    proposals, banks = [], {}
    for key in keys:
        selected = choose_energy([r for r in scores if (r['subject'], r['arm']) == key],
                                 policy['training_nll_tolerance'])
        old = original[key]['candidate']
        proposed = next(r for r in original[key]['finalists'] if r['id'] == selected['candidate'])
        banks[key] = list({r['id']: deepcopy(r) for r in (old, proposed)}.values())
        proposals.append(dict(subject=key[0], arm=key[1], original=old['id'], proposed=proposed['id'],
                              training_generation_score=selected))
    output.mkdir(parents=True)
    if smoke:
        from .model_1022_compact_evaluate import smoke_fixture
        fixture = smoke_fixture(ROOT / settings['data'], output / 'software_fixture', settings['subjects'][0])
        settings['data'] = str(fixture.resolve())
    pilot.driver.base.source_hashes, pilot.driver.base.execute = sources, pilot.execute
    pilot.driver.base.archive(output, config, sources(pilot.driver.io))
    write(output / 'protocol.json', dict(settings=settings, policy=policy, smoke=smoke,
        source_sha256=sources(pilot.driver.io), inputs_sha256=inputs, generation=str(generated.resolve()),
        python=sys.version, versions={name: importlib.metadata.version(name)
                                     for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        decision=('training Energy inside original NLL tolerance, followed by independent paired NLL'
                  + (' and generation guards' if guard_policy else ' guard'))))
    write(output / 'energy_proposals.json', dict(proposals=proposals, candidate_scores=scores,
                                               later_choices_used=False, oral_used=False))
    stage = lambda bank, name: pilot.driver.base.stage(settings, ROOT / settings['data'], output,
                                                      bank, name, smoke)
    checked = stage(banks, 'guard')
    decisions, nominees = [], []
    for proposal in proposals:
        key = proposal['subject'], proposal['arm']
        rows = {r['id']: r for r in checked[key]}
        old, new = rows[proposal['original']], rows[proposal['proposed']]
        arrays = []
        for point in (old, new):
            file = Path(point['score_file'])
            meta = read(file)
            if meta['subject'] != key[0] or meta['candidate']['hyperparams'] != point['hyperparams']:
                raise ValueError('Likelihood guard identity changed')
            if digest(file.with_suffix('.npz')) != meta['predictions_sha256']:
                raise ValueError('Likelihood guard predictions changed')
            with np.load(file.with_suffix('.npz')) as z:
                arrays.append((z['observed_probability'], z['seeds']))
        np.testing.assert_array_equal(arrays[0][1], arrays[1][1])
        decision = likelihood_guard(arrays[0][0], arrays[1][0], policy['training_nll_tolerance'],
                                    policy['guard_mcse_multiplier'])
        chosen = new if decision['accepted'] else old
        decisions.append(dict(proposal, **decision, selected=chosen['id']))
        nominees.append(dict(subject=key[0], arm=key[1], candidate=chosen, finalists=checked[key]))
    templates = read(study / 'autonomous_jobs.json')['jobs']
    if guard_policy:
        guard_templates = deepcopy(templates)
        if smoke:
            for job in guard_templates:
                job['folder'] = str(Path(settings['data']) / f"subject_{job['subject']}")
        confirm_generation(guard_policy, study, generated, output, nominees,
                           decisions, guard_templates, inputs, smoke)
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Selection input changed before freezing: ' + file)
    write(output / 'selection_decisions.json', dict(decisions=decisions, later_choices_used=False, oral_used=False))
    write(output / 'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
                                        later_choices_used=False, oral_used=False))
    stage({(r['subject'], r['arm']): [r['candidate']] for r in nominees}, 'evaluate')
    jobs = []
    for row in nominees:
        sid, arm = row['subject'], row['arm']
        template = next(j for j in templates if j['subject'] == sid and j['model'] == arm)
        seeds = [stable_seed(dict(base=policy['seed'], subject=sid, stage='autonomous', repeat=i))
                 for i in range(4 if smoke else policy['autonomous_paths'])]
        width = policy['autonomous_chunk']
        for i in range(0, len(seeds), width):
            job = deepcopy(template)
            name = f'autonomous_S{sid}_{arm}_{i:03d}'
            job.update(name=name, candidate=row['candidate'], seeds=seeds[i:i+width], smoke=smoke,
                       source_sha256=sources(pilot.driver.io), target=str((output / 'autonomous' / name).resolve()))
            if smoke:
                job['folder'] = str((Path(settings['data']) / f'subject_{sid}').resolve())
            jobs.append(job)
    write(output / 'autonomous_jobs.json', dict(jobs=jobs))
    pilot.execute(jobs, output, settings['parallel_budget'])
    if smoke:
        write(output / 'completion.json', dict(status='complete', smoke=True, cases=len(nominees),
            scope='software selection, eight-trial filters and sixteen-trial autonomous paths'))
    else:
        pilot.summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.generation, args.output, args.smoke)
