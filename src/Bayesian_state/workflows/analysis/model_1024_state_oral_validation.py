"""Frozen conditional state recovery and external oral compatibility for Model 1024.

Run prepare, fit, infer and oral into separate new destinations. Shared workers
retain the archived cognitive runtime, chronological split and task feedback.
The fit phase reads cases.json only; generating points live in truths.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from . import model_1024_generation_selection as shared
from . import model_1024_retention_capacity as profile_tools
from . import model_1023_execution_recovery as recovery
from . import model_1023_execution_recovery_score as scores
from . import model_1023_execution_capture as capture
from . import model_1023_execution_validation as validation
from . import model_1023_execution_oral as oral_tools
from . import model_0923_decision_basis_alignment as alignment
from . import model_0923_oral_constraint_audit as audit
from . import model_0923_oral_reference_sensitivity as sensitivity
from ...metrics.autonomous_fit import behavior_features, energy_u

ROOT, read, write, digest, nll = shared.ROOT, shared.read, shared.write, shared.digest, shared.nll
DELTA, CAPACITY = profile_tools.DELTA, profile_tools.CAPACITY
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def sources() -> dict:
    return dict(shared.sources(), **validation.sources(), **{ENTRY: digest(Path(__file__))})


def policy(path: Path) -> dict:
    p = yaml.safe_load(path.read_text())
    if (p['subjects'] != [129, 229, 317] or p['retention_values'] != [0., .3, .6, .8, 1.]
            or p['capacity_values'] != [2, 3, 6] or p['replicate_indices'] != [0, 1]
            or not 1 <= p['parallel_budget'] <= 128):
        raise ValueError('Require the registered subjects, grid and fixed path indices')
    return p


def far_alternative(anchor: dict, points: list[dict], training: list[dict], p: dict) -> dict:
    """Choose a training-qualified point by normalized grid-rank distance."""
    by_id = {r['candidate']: r for r in training}
    a = [p['retention_values'].index(anchor['hyperparams'][DELTA]),
         p['capacity_values'].index(anchor['hyperparams'][CAPACITY])]
    def key(point):
        b = [p['retention_values'].index(point['hyperparams'][DELTA]),
             p['capacity_values'].index(point['hyperparams'][CAPACITY])]
        distance = ((b[0]-a[0])/4)**2 + ((b[1]-a[1])/2)**2
        return -distance, by_id[point['id']]['training_nll'], point['id']
    eligible = [c for c in points if c['id'] != anchor['id'] and by_id[c['id']]['eligible']]
    if not eligible:
        raise ValueError('No non-anchor training-qualified alternative')
    return deepcopy(min(eligible, key=key))


def load_path(case: dict, index: int) -> tuple[dict, dict]:
    offset = 0
    for job in case['autonomous_jobs']:
        if offset <= index < offset + len(job['seeds']):
            target = Path(job['target']); arrays = scores.checked_arrays(target)
            i = index-offset
            np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
            raw = {k: v.copy() if k == 'categories' else v[i].copy() for k, v in arrays.items()}
            return raw, dict(source=str(target.resolve()), source_sha256=digest(target.with_suffix('.npz')),
                             path_index=index, seed=int(raw['seeds']))
        offset += len(job['seeds'])
    raise IndexError(index)


def prepare(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new preparation directory')
    p = policy(config); previous = ROOT/p['source_evaluation']
    if read(previous/'completion.json')['status'] != 'complete':
        raise ValueError('Previous trajectory bank is incomplete')
    previous_protocol = read(previous/'protocol.json')
    training_file = ROOT/p['source_summary']/'training_profile.csv'
    training = pd.read_csv(training_file)
    originals = read(ROOT/p['source_training']/'nominees.json')['nominees']
    output.mkdir(parents=True); definitions = []; truths = []; human = []
    inputs = {str(config.resolve()): digest(config), str(training_file): digest(training_file),
              str(previous/'protocol.json'): digest(previous/'protocol.json')}
    sid_list = p['subjects'][:1] if smoke else p['subjects']
    for sid in sid_list:
        selected = {}
        for mode in ('MIX', 'STICKY'):
            original = next(c for c in originals if (c['subject'], c['mode']) == (sid, mode))
            own_training = training.loc[training.subject.eq(sid) & training['mode'].eq(mode)].to_dict('records')
            anchor = original['candidate']; points = original['candidates']
            alt = far_alternative(anchor, points, own_training, p)
            selected[mode] = (anchor, alt, points)
            for label, point in ((mode, anchor), (mode+'_ALT', alt)):
                bank = next(c for c in previous_protocol['cases'] if (c['subject'], c['mode'], c['candidate']['id']) == (sid, mode, point['id']))
                reference = next(c for c in previous_protocol['cases'] if c['subject'] == sid and c['mode'] == mode and c['is_anchor'])
                human.append(dict(subject=sid, arm=label, candidate=point, bank=bank, reference=reference))
        anchor, alt, points = selected['STICKY']
        geometry = ROOT/p['geometry']/f'subject_{sid}'/'audit.json'
        targets = read(geometry)['training_perfect_rules']
        inputs[str(geometry)] = digest(geometry)
        banks = [next(c for c in previous_protocol['cases'] if (c['subject'], c['mode'], c['candidate']['id']) == (sid, 'STICKY', point['id'])) for point in (anchor, alt)]
        requests = [(bank, index, 'fixed') for bank in banks for index in p['replicate_indices']]
        if smoke:
            requests = requests[:1]
        else:
            challenges = []
            for point_index, bank in enumerate(banks):
                for index in range(2, 128):
                    raw, _ = load_path(bank, index)
                    present = np.isin(raw['rule'], targets).any(axis=1)
                    rare = present & ~np.isin(raw['executed'], targets) & (raw['choices'] != raw['categories'])
                    challenges.append((int(rare.sum()), index, point_index))
            count, index, point_index = min(challenges, key=lambda x: (-x[0], x[1], x[2]))
            if count:
                requests.append((banks[point_index], index, 'enriched_challenge'))
        for bank, index, scope in requests:
            identity = f'C{len(definitions):03d}'
            raw, provenance = load_path(bank, index)
            original_folder = Path(bank['folder'])
            with np.load(original_folder/'training.npz') as a, np.load(original_folder/'sealed_test.npz') as b:
                stimulus = np.concatenate([a['stimulus'], b['stimulus']])
            n = 128 if smoke else len(raw['choices'])
            observations = dict(stimulus=stimulus[:n], **{k: raw[k][:n] for k in ('choices', 'categories', 'feedback')})
            folder = output/'observations'/identity/f'subject_{sid}'
            template = profile_tools.clean_template(bank)
            template.update(candidate=deepcopy(points[0]), seeds=[], folder=str(folder.resolve()))
            meta = recovery.fixture(original_folder, folder, observations, template['runtime'], .7)
            raw_file = output/'truth_states'/identity
            raw_file.parent.mkdir(exist_ok=True)
            with raw_file.with_suffix('.npz').open('xb') as stream:
                np.savez_compressed(stream, **{k: v[:n] if k != 'seeds' else v for k, v in raw.items()})
            definition = dict(case_id=identity, subject=sid, scope=scope, folder=str(folder.resolve()),
                              candidates=points, template=template, target_rules=targets,
                              arrays_sha256=meta['array_sha256'], training_trials=meta['training_trials'])
            definitions.append(definition)
            truths.append(dict(case_id=identity, subject=sid, scope=scope, candidate=bank['candidate'],
                               states=str(raw_file.resolve()), states_sha256=digest(raw_file.with_suffix('.npz')),
                               **provenance))
    human = human[:1] if smoke else human
    write(output/'cases.json', dict(cases=definitions, smoke=smoke, policy=p, truths_opened=False))
    write(output/'truths.json', dict(rows=truths, selection='fixed indices 0/1 plus separately labelled enriched challenges'))
    write(output/'human.json', dict(rows=human, smoke=smoke))
    write(output/'provenance.json', dict(inputs_sha256=inputs, numerical_sources=sources()))
    write(output/'completion.json', dict(status='complete', smoke=smoke, cases=len(definitions),
        cases_sha256=digest(output/'cases.json'), truths_sha256=digest(output/'truths.json'),
        human_sha256=digest(output/'human.json')))
    print(f'Prepared {len(definitions)} opaque cases', flush=True)


def fit(prepared: Path, output: Path) -> None:
    """Estimate the same two coordinates without reading the generating manifest."""
    if output.exists():
        raise FileExistsError('Use a new fitting directory')
    done = read(prepared/'completion.json'); definition = read(prepared/'cases.json')
    if done['cases_sha256'] != digest(prepared/'cases.json') or done['status'] != 'complete':
        raise ValueError('Case definitions changed')
    p, smoke = definition['policy'], definition['smoke']; output.mkdir(parents=True)
    started = time.monotonic(); jobs = []; cases = []
    for original in definition['cases']:
        if original['scope'] != 'fixed':
            continue
        c = deepcopy(original); identity = c['case_id']; folder = Path(c['folder'])
        for name, sha in c['arrays_sha256'].items():
            # The sealed suffix is not opened or hashed during fitting.
            if name == 'training.npz' and digest(folder/name) != sha:
                raise ValueError('Training observations changed')
        with np.load(folder/'training.npz') as z:
            count = 8 if smoke else len(z['choices'])
            empirical = behavior_features(z['choices'][None, :count], z['categories'][:count],
                n_categories=2 if c['subject'] == 129 else 4, **p['features'])[0]
        c.update(empirical=empirical.tolist(), screen_filters={}, screen_paths={})
        if smoke:
            c['candidates'] = c['candidates'][:2]
        fs = shared.seed_bank(p, c['subject'], identity+'_screen_filter', 2 if smoke else p['screen']['seeds'], set())
        gs = shared.seed_bank(p, c['subject'], identity+'_screen_generation', 4 if smoke else p['screen']['paths'], set(fs))
        for point in c['candidates']:
            fj = shared.point_jobs(c['template'], point, output, identity+'_screen_filter', fs, p,
                                  phase='training', particles=2 if smoke else p['screen']['particles'], smoke=smoke)
            aj = shared.point_jobs(c['template'], point, output, identity+'_screen_generation', gs, p,
                                  training_generation=True, smoke=smoke)
            c['screen_filters'][point['id']] = fj; c['screen_paths'][point['id']] = aj
            jobs.extend(fj+aj)
        c['forbidden'] = fs+gs; cases.append(c)
    write(output/'protocol.json', dict(policy=p, cases=cases, smoke=smoke, sources=sources(),
        cases_sha256=digest(prepared/'cases.json'), truths_opened=False, later_choices_used=False, oral_used=False))
    budget = 1 if smoke else p['parallel_budget']; shared.execute(jobs, output, budget)
    rows = []; confirmation = []
    for c in cases:
        own = []
        for point in c['candidates']:
            pred = shared.filter_arrays(c['screen_filters'][point['id']])
            features = shared.training_features(c['screen_paths'][point['id']], np.array(c['empirical']))
            energy = energy_u(features, np.array(c['empirical']))
            own.append(dict(case_id=c['case_id'], subject=c['subject'], candidate=point['id'],
                delta=point['hyperparams'][DELTA], capacity=point['hyperparams'][CAPACITY],
                training_nll=nll(pred['observed_probability']), training_energy=energy['value']))
        reference = min(own, key=lambda x: (x['training_nll'], x['candidate']))
        proposal = profile_tools.choose_profile(own, reference['training_nll'], p['training_nll_tolerance'])
        c['reference'] = next(x for x in c['candidates'] if x['id'] == reference['candidate'])
        c['proposed'] = next(x for x in c['candidates'] if x['id'] == proposal['candidate'])
        c['confirmation_filters'], c['confirmation_paths'] = {}, {}
        for row in own:
            row['eligible'] = row['training_nll'] <= reference['training_nll']+p['training_nll_tolerance']
        rows.extend(own)
        if c['reference']['id'] == c['proposed']['id']:
            continue
        fs = shared.seed_bank(p, c['subject'], c['case_id']+'_guard_filter', 2 if smoke else p['guard']['seeds'], set(c['forbidden']))
        gs = shared.seed_bank(p, c['subject'], c['case_id']+'_guard_generation', 4 if smoke else p['confirmation_paths'], set(c['forbidden']+fs))
        for point in (c['reference'], c['proposed']):
            fj = shared.point_jobs(c['template'], point, output, c['case_id']+'_guard_filter', fs, p,
                                  phase='training', particles=2 if smoke else p['guard']['particles'], smoke=smoke)
            aj = shared.point_jobs(c['template'], point, output, c['case_id']+'_guard_generation', gs, p,
                                  training_generation=True, smoke=smoke)
            c['confirmation_filters'][point['id']] = fj; c['confirmation_paths'][point['id']] = aj
            confirmation.extend(fj+aj)
    pd.DataFrame(rows).to_csv(output/'screen_scores.csv', index=False)
    write(output/'proposals.json', dict(cases=cases, truths_opened=False, later_choices_used=False))
    shared.execute(confirmation, output, budget)
    nominees = []
    for c in cases:
        ref, prop = c['reference'], c['proposed']; likelihood = {}; generation = {}
        if ref['id'] != prop['id']:
            a, b = [shared.filter_arrays(c['confirmation_filters'][x['id']]) for x in (ref, prop)]
            np.testing.assert_array_equal(a['seeds'], b['seeds'])
            likelihood = shared.likelihood_guard(a['observed_probability'], b['observed_probability'],
                                                p['training_nll_tolerance'], p['mcse_multiplier'])
            a, b = [shared.training_features(c['confirmation_paths'][x['id']], np.array(c['empirical'])) for x in (ref, prop)]
            generation = shared.paired_energy_guard(a, b, np.array(c['empirical']), p['mcse_multiplier'])
        decision = shared.final_decision(ref['id'], prop['id'], likelihood, generation)
        nominees.append(dict(case_id=c['case_id'], subject=c['subject'], reference=ref, proposed=prop,
            candidate=next(x for x in c['candidates'] if x['id'] == decision['selected']), decision=decision))
    write(output/'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
        later_choices_used=False, oral_used=False, truths_opened=False, scope=p['scope'], sources=sources()))
    write(output/'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        screen_jobs=len(jobs), confirmation_jobs=len(confirmation), seconds=time.monotonic()-started,
        nominees_sha256=digest(output/'nominees.json'), protocol_sha256=digest(output/'protocol.json')))
    print('All synthetic estimates frozen before opening truths', flush=True)


def execute_numeric(jobs: list[dict], output: Path, budget: int) -> None:
    if not jobs:
        return
    workers = min(budget, len(os.sched_getaffinity(0)), len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        validation.NumericalQueue(pool).execute(jobs, output, workers)


def inference_job(c: dict, point: dict, seed: int, particles: int, output: Path,
                  name: str, smoke: bool, *, oral: bool = False, reference: Path | None = None) -> dict:
    job = deepcopy(c['template'])
    for key in ('training_generation', 'selection_sources', 'freeze_file', 'freeze_sha256'):
        job.pop(key, None)
    job.update(name=name, candidate=point, seeds=[seed], particles=particles,
        phase='training' if smoke else 'evaluate', smoke=smoke, case_id=c['subject'],
        target=str((output/'scores'/name).resolve()), state_capture_target=str((output/'states'/name).resolve()),
        capture_sources=capture.sources(), target_rules=c['target_rules'], capture_oral=oral)
    if reference is not None:
        job.update(original_score=str(reference.resolve()), original_score_sha256=digest(reference))
    return job


def infer(prepared: Path, fitted: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new inference directory')
    d = read(prepared/'cases.json'); ready = read(prepared/'completion.json')
    frozen = read(fitted/'nominees.json'); done = read(fitted/'completion.json')
    if (done['status'] != 'complete' or done['nominees_sha256'] != digest(fitted/'nominees.json')
            or ready['truths_sha256'] != digest(prepared/'truths.json') or frozen['truths_opened']
            or frozen['later_choices_used'] or frozen['status'] != 'frozen_before_test'):
        raise ValueError('Require complete intact training freeze before opening truths')
    truths = {r['case_id']: r for r in read(prepared/'truths.json')['rows']}
    p, smoke = d['policy'], d['smoke']; output.mkdir(parents=True)
    jobs = []; cases = []
    for c in d['cases']:
        identity = c['case_id']; truth = truths[identity]
        if digest(Path(truth['states']).with_suffix('.npz')) != truth['states_sha256']:
            raise ValueError('Generating state changed')
        fitrow = next((r for r in frozen['nominees'] if r['case_id'] == identity), None)
        roles = dict(true=truth['candidate'])
        if fitrow:
            roles.update(adopted=fitrow['candidate'], nll_reference=fitrow['reference'])
        fs = shared.seed_bank(p, c['subject'], identity+'_inference', 2 if smoke else p['inference']['seeds'],
                              {truth['seed']})
        states, point_jobs, resolution = {}, {}, {}
        for point in {r['id']: r for r in roles.values()}.values():
            capture_state = point['id'] in {roles[k]['id'] for k in ('true', 'adopted') if k in roles}
            own = []
            for i, seed in enumerate(fs):
                name = f'{identity}_{point["id"]}_P{2 if smoke else p["inference"]["particles"]}_{i}'
                job = inference_job(c, point, seed, 2 if smoke else p['inference']['particles'], output, name, smoke)
                if not capture_state:
                    job.pop('capture_sources'); job.pop('state_capture_target')
                own.append(job); jobs.append(job)
            point_jobs[point['id']] = own
            if capture_state:
                states[point['id']] = [j['state_capture_target'] for j in own]
            # The first fixed anchor case for each subject uses a paired resolution audit.
            if not smoke and identity == next(x['case_id'] for x in d['cases'] if x['subject'] == c['subject'] and x['scope'] == 'fixed') and capture_state:
                high = []
                for i, seed in enumerate(fs[:p['resolution']['seeds']]):
                    name = f'{identity}_{point["id"]}_P{p["resolution"]["particles"]}_{i}'
                    job = inference_job(c, point, seed, p['resolution']['particles'], output, name, False)
                    high.append(job); jobs.append(job)
                resolution[point['id']] = high
        cases.append(dict(definition=c, truth=truth, roles=roles, point_jobs=point_jobs, states=states,
                          resolution=resolution, fitted=fitrow))
    # Human alternatives receive an observer-free replay before passive capture.
    humans = read(prepared/'human.json')['rows']; preflight = []; human_plan = []
    for h in humans:
        bank, reference = h['bank'], h['reference']; sid = h['subject']; mode = bank['mode']
        c = dict(subject=sid, template=profile_tools.clean_template(bank), target_rules=read(ROOT/p['geometry']/f'subject_{sid}'/'audit.json')['training_perfect_rules'])
        seeds = [j['seeds'][0] for j in reference['filter_jobs'][:1 if smoke else p['human_filters']]]
        for i, seed in enumerate(seeds):
            name = f'H{sid}_{h["arm"]}_{i}'
            original = Path(reference['filter_jobs'][i]['target']).with_suffix('.npz')
            if h['arm'].endswith('_ALT') or smoke:
                bare = inference_job(c, h['candidate'], seed, 2 if smoke else p['inference']['particles'], output, name+'_bare', smoke)
                bare.pop('capture_sources'); bare.pop('state_capture_target')
                preflight.append(bare); original = Path(bare['target']).with_suffix('.npz')
            human_plan.append((c, h, seed, name, original))
    write(output/'preflight.json', dict(jobs=preflight, sources=sources()))
    execute_numeric(preflight, output, 1 if smoke else p['parallel_budget'])
    human_jobs = []
    for c, h, seed, name, original in human_plan:
        job = inference_job(c, h['candidate'], seed, 2 if smoke else p['inference']['particles'], output, name, smoke, oral=True, reference=original)
        job['oral_arm'] = h['arm']; human_jobs.append(job)
    jobs.extend(human_jobs)
    write(output/'protocol.json', dict(policy=p, prepared=str(prepared.resolve()), fitted=str(fitted.resolve()),
        cases=cases, human_jobs=human_jobs, jobs=jobs, smoke=smoke, sources=sources(),
        truths_opened_after_all_fits_freeze=True, fit_nominees_sha256=digest(fitted/'nominees.json'),
        prepared_completion_sha256=digest(prepared/'completion.json')))
    execute_numeric(jobs, output, 1 if smoke else p['parallel_budget'])
    exact = all(read(Path(j['state_capture_target']).with_suffix('.json'))['exact_saved_filter_replay'] for j in human_jobs)
    write(output/'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs), preflight_jobs=len(preflight),
        exact_behavior_replay=exact, protocol_sha256=digest(output/'protocol.json')))
    print('Known/estimated states and exact human decision-basis replays complete', flush=True)


def export_basis(source: Path, output: Path) -> dict:
    """Adapt four labelled human arms to the shared oral alignment interface."""
    done, protocol = read(source/'completion.json'), read(source/'protocol.json')
    if not done['exact_behavior_replay'] or done['protocol_sha256'] != digest(source/'protocol.json'):
        raise ValueError('Human capture replay is incomplete')
    output.mkdir(parents=True); (output/'states').mkdir(); (output/'jobs').mkdir()
    jobs = protocol['human_jobs']; counts = {}; bindings = {}
    for job in jobs:
        target = Path(job['state_capture_target']); record = read(target.with_suffix('.json'))
        if not record['exact_saved_filter_replay'] or record['capture_sources'] != capture.sources():
            raise ValueError('Passive capture changed')
        arrays = scores.checked_arrays(target, 'array_sha256')
        basis = {k.removeprefix('oral_'): v for k, v in arrays.items() if k.startswith('oral_')}
        basis['seeds'] = arrays['seeds']
        reference = Path(job['original_score'])
        if digest(reference) != job['original_score_sha256']:
            raise ValueError('Original prediction changed')
        with np.load(reference) as saved:
            i, = np.flatnonzero(saved['seeds'] == basis['seeds'][0])
            np.testing.assert_array_equal(basis['prediction'][np.arange(len(basis['choices'])), basis['choices']-1], saved['observed_probability'][i])
            if not done['smoke']:
                np.testing.assert_array_equal(basis['prediction'], saved['probabilities'][i])
        destination = output/'states'/target.name
        with destination.with_suffix('.npz').open('xb') as stream:
            np.savez_compressed(stream, **basis)
        converted = dict(record, arm=job['oral_arm'], exact_behavior_replay=True,
            exact_original_filter_capture=False, additional_filters=1,
            timing='after_choice_before_current_feedback', array_sha256=digest(destination.with_suffix('.npz')),
            source_capture=str(target.with_suffix('.json').resolve()), source_capture_sha256=digest(target.with_suffix('.json')))
        write(destination.with_suffix('.json'), converted); write(output/'jobs'/(target.name+'.json'), job)
        counts.setdefault((record['subject'], job['oral_arm']), []).append(int(basis['seeds'][0]))
        bindings[str(target.with_suffix('.npz'))] = record['array_sha256']
    subjects = sorted({k[0] for k in counts}); arms = ['MIX'] if done['smoke'] else ['MIX', 'MIX_ALT', 'STICKY', 'STICKY_ALT']
    expected = 1 if done['smoke'] else protocol['policy']['human_filters']
    if set(counts) != {(sid, mode) for sid in subjects for mode in arms} or any(len(set(v)) != expected for v in counts.values()):
        raise ValueError('Missing or duplicate human arms/seeds')
    for sid in subjects:
        if len({tuple(sorted(v)) for (s, _), v in counts.items() if s == sid}) != 1:
            raise ValueError('Human comparison requires paired filter seeds')
    own = dict(settings=dict(subjects=subjects, arms=arms, seeds_per_arm=expected), smoke=done['smoke'],
        capture=str(source.resolve()), input_sha256=bindings, source_sha256=digest(Path(__file__)),
        parameter_estimation_scope='chronological_prefix_behavior_only',
        evaluation_use='development_state_recovery_oral_compatibility', oral_observations_used_for_inference=False)
    write(output/'protocol.json', own)
    write(output/'completion.json', dict(status='complete', protocol_sha256=digest(output/'protocol.json')))
    return own


def oral(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral validation directory')
    p = read(source/'protocol.json')['policy']; smoke = read(source/'completion.json')['smoke']
    output.mkdir(parents=True); exported = export_basis(source, output/'basis')
    config = yaml.safe_load((ROOT/p['oral_config']).read_text())
    config['subjects'] = exported['settings']['subjects']
    workers = 1 if smoke else min(p['parallel_budget'], len(os.sched_getaffinity(0)))
    config['workers'] = workers; config_path = output/'oral_config.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    audit.run(config_path, output/'coding_midpoint', workers)
    sensitivity.run(output/'coding_midpoint', output/'coding_relative', workers)
    checks = {}
    for reading in p['oral_readings']:
        coding, scored = output/('coding_'+reading), output/('alignment_'+reading)
        alignment.run(output/'basis', coding, scored, reference_model='MIX')
        checks[reading] = oral_tools.audit_denominators(scored, coding)
    write(output/'completion.json', dict(status='complete', smoke=smoke, validation=checks,
        source_sha256=digest(Path(__file__)), inference_protocol_sha256=digest(source/'protocol.json'),
        oral_used_for_parameter_estimation=False, oral_used_for_state_inference=False,
        meaning='external content compatibility', denominator='all complete reports; catalogue coverage separate'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'fit', 'infer', 'oral'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path); parser.add_argument('--prepared', type=Path)
    parser.add_argument('--fit', type=Path); parser.add_argument('--inference', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == 'prepare' and args.config:
            prepare(args.config, args.output, args.smoke)
        elif args.phase == 'fit' and args.prepared:
            fit(args.prepared, args.output)
        elif args.phase == 'infer' and args.prepared and args.fit:
            infer(args.prepared, args.fit, args.output)
        elif args.phase == 'oral' and args.inference:
            oral(args.inference, args.output)
        else:
            parser.error('Provide the input directories required by the selected phase')
