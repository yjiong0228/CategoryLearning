"""Diagnose complete learning distributions with frozen, train-only candidates.

Run from the repository root with --config YAML and a new --output directory:
--phase nominate; --phase generate --nomination DIR [--smoke]; then
--phase analyze --generation DIR. This reuses the execution-study simulator
and saved fits. It changes analysis and candidate sampling, not cognition.
Protocol: docs/model_architecture/model_1024_trajectory_diagnosis.md.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import yaml

from . import model_1023_execution_study as execution
from .model_1022_belief_summary import arrays, digest, jackknife_se, load_case, nll, read
from .model_1023_execution_summary import trajectory_diagnostics
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
write = execution.write


def manifest(config: Path) -> dict:
    """Record analysis identity independently of the shared generative code."""
    return dict(config_sha256=digest(config), source_sha256=digest(Path(__file__)),
        python=sys.version, versions={k: importlib.metadata.version(k)
        for k in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'PyYAML')})


def policy_at(config: Path) -> dict:
    policy = yaml.safe_load(config.read_text())
    if (policy['modes'] != ['MIX', 'MAP', 'STICKY'] or policy['clusters'] != [3, 2, 4]
            or not 1 <= policy['parallel_budget'] <= 128):
        raise ValueError('Require the registered mechanisms, clusters and process budget')
    return policy


def frozen_study(policy: dict) -> tuple[Path, dict, list[dict]]:
    study = ROOT / policy['study']
    complete, frozen = read(study / 'completion.json'), read(study / 'nominees.json')
    if (complete['status'] != 'complete' or complete['smoke'] or frozen['later_choices_used']
            or frozen['oral_used'] or frozen['status'] != 'frozen_before_test'
            or complete['nominees_sha256'] != digest(study / 'nominees.json')):
        raise ValueError('Require completed, training-only frozen estimates')
    return study, frozen, read(study / 'autonomous_jobs.json')['jobs']


def select_candidates(rows: list[dict], space: dict, tolerance: float, alternatives: int) -> list[dict]:
    """Choose diverse training-near points without accessing observed suffixes."""
    if not rows or tolerance < 0 or alternatives < 0:
        raise ValueError('Invalid candidate budget or tolerance')
    ordered = sorted(rows, key=lambda r: (r['mean_nll'], r['id']))
    eligible = [r for r in ordered if r['mean_nll'] <= ordered[0]['mean_nll'] + tolerance]
    keys = sorted(k for k, values in space.items() if len(values) > 1)
    if not keys:
        return eligible[:1]
    coordinates = np.array([[space[k].index(r['hyperparams'][k]) / (len(space[k])-1)
                            for k in keys] for r in eligible])
    distances = cdist(coordinates, coordinates, metric='cityblock') / len(keys)
    selected = [0]
    while len(selected) < min(alternatives + 1, len(eligible)):
        remaining = [i for i in range(len(eligible)) if i not in selected]
        selected.append(min(remaining, key=lambda i: (-distances[i, selected].min(),
                                                    eligible[i]['mean_nll'], eligible[i]['id'])))
    return [eligible[i] for i in selected]


def nominate(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new nomination directory')
    policy = policy_at(config)
    study, frozen, old_jobs = frozen_study(policy)
    original = read(study / 'protocol.json')
    if original['source_sha256'] != execution.sources(execution.driver.io):
        raise ValueError('Shared scientific implementation differs from the frozen fit')
    spaces = {(r['subject'], r['mode']): r['space'] for r in original['parameter_spaces']}
    settings = policy['sensitivity']
    rows, audits, sources = [], [], {}
    for nominee in frozen['nominees']:
        sid, mode = nominee['subject'], nominee['arm']
        if sid not in settings['subjects'] or mode not in settings['modes']:
            continue
        finalists = nominee['finalists']
        selected = select_candidates(finalists, spaces[sid, mode], settings['training_nll_tolerance'],
                                     settings['alternatives'])
        if selected[0]['id'] != nominee['candidate']['id']:
            raise ValueError('Original frozen nominee is not the training leader')
        # Audit every confirmation point before selection is frozen. No test
        # arrays or generated paths are read in this phase.
        probabilities = {}
        for r in finalists:
            file = study / 'confirmation' / f'subject_{sid}' / mode / (r['id'] + '.json')
            meta, saved = read(file), arrays(file)
            if meta['role'] != 'training_only_high_particle_audit' or meta['candidate']['hyperparams'] != r['hyperparams']:
                raise ValueError('Candidate is not the declared training confirmation point')
            p = saved['observed_probability']
            np.testing.assert_allclose(nll(p), r['mean_nll'], rtol=0, atol=1e-12)
            probabilities[r['id']] = (p, saved['seeds'])
            sources[str(file.relative_to(ROOT))] = digest(file)
        p0, seeds0 = probabilities[selected[0]['id']]
        for r in finalists:
            p, seeds = probabilities[r['id']]
            np.testing.assert_array_equal(seeds, seeds0)
            deleted = np.array([nll(np.delete(p, i, 0))-nll(np.delete(p0, i, 0)) for i in range(len(p))])
            audits.append(dict(subject=sid, mode=mode, candidate=r['id'], training_nll=nll(p),
                training_nll_delta=nll(p)-nll(p0), delta_mcse=jackknife_se(deleted),
                eligible=bool(nll(p)-nll(p0) <= settings['training_nll_tolerance']),
                selected=r['id'] in {s['id'] for s in selected}))
        template = next(j for j in old_jobs if j['subject'] == sid and j['model'] == mode)
        for index, r in enumerate(selected):
            rows.append(dict(subject=sid, mode=mode, role='nominee' if index == 0 else f'alternative_{index}',
                             candidate=r, template=template))
    output.mkdir(parents=True)
    pd.DataFrame(audits).to_csv(output / 'training_candidates.csv', index=False)
    write(output / 'nomination.json', dict(status='frozen_train_only', rows=rows, policy=policy,
        suffix_choices_used=False, oral_used=False, new_paths_opened=False,
        old_nominees_sha256=digest(study / 'nominees.json'), confirmation_sources=sources,
        provenance=manifest(config)))
    print(f'Frozen {len(rows)} parameter points; no suffix choices or autonomous outcomes used', flush=True)


def generate(config: Path, nomination: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new generation directory')
    policy = policy_at(config)
    plan = read(nomination / 'nomination.json')
    if (plan['status'] != 'frozen_train_only' or plan['policy'] != policy
            or plan['suffix_choices_used'] or plan['oral_used'] or plan['new_paths_opened']):
        raise ValueError('Generation requires a matching train-only nomination')
    study, _, old_jobs = frozen_study(policy)
    if plan['old_nominees_sha256'] != digest(study / 'nominees.json'):
        raise ValueError('Original nominees changed')
    for filename, sha in plan['confirmation_sources'].items():
        if digest(ROOT / filename) != sha:
            raise ValueError('Training confirmation changed after nomination')
    output.mkdir(parents=True)
    jobs = []
    setting = policy['sensitivity']
    for row in plan['rows'][:1 if smoke else len(plan['rows'])]:
        sid, mode = row['subject'], row['mode']
        seeds = [stable_seed(dict(base=setting['seed'], subject=sid, stage='trajectory_sensitivity', repeat=i))
                 for i in range(1 if smoke else setting['paths'])]
        old_seeds = {s for j in old_jobs if j['subject'] == sid for s in j['seeds']}
        if len(set(seeds)) != len(seeds) or old_seeds.intersection(seeds):
            raise ValueError('New autonomous seeds are not distinct from original seeds')
        for index in range(0, len(seeds), setting['chunk']):
            job = deepcopy(row['template'])
            name = f'S{sid}_{mode}_{row["role"]}_{index:03d}'
            job.update(name=name, candidate=row['candidate'], smoke=smoke,
                seeds=seeds[index:index + setting['chunk']],
                source_sha256=execution.sources(execution.driver.io),
                target=str((output / 'autonomous' / name).resolve()))
            jobs.append(job)
    write(output / 'protocol.json', dict(policy=policy, nomination=str(nomination.resolve()),
        nomination_sha256=digest(nomination / 'nomination.json'), jobs=jobs, smoke=smoke,
        provenance=manifest(config)))
    execution.execute(jobs, output, 1 if smoke else policy['parallel_budget'])
    write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        paths=sum(len(j['seeds']) for j in jobs), protocol_sha256=digest(output / 'protocol.json')))


def probability_record(events: np.ndarray) -> dict:
    """Wilson interval for Monte Carlo event counts, including zero counts."""
    values = np.asarray(events, bool)
    if values.ndim != 1 or not len(values):
        raise ValueError('Need nonempty independent path events')
    count, total = int(values.sum()), len(values)
    p, z = count / total, 1.959963984540054
    center = (p + z*z/(2*total)) / (1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return dict(count=count, paths=total, probability=p, low=max(0., center-half), high=min(1., center+half))


def complete_followup(choices: np.ndarray, categories: np.ndarray, window: int) -> dict:
    """Largest adjacent gain restricted to positions with full maintenance."""
    correct = np.asarray(choices) == np.asarray(categories)[None, :]
    if correct.ndim != 2 or window < 1 or correct.shape[1] < 3*window:
        raise ValueError('Need three complete windows')
    c = np.pad(np.cumsum(correct, axis=1), ((0, 0), (1, 0)))
    centers = np.arange(window, correct.shape[1]-2*window+1)
    gains = (c[:, centers+window]-2*c[:, centers]+c[:, centers-window])/window
    peak = gains.argmax(axis=1)
    center, row = centers[peak], np.arange(len(correct))
    return dict(complete_gain=gains[row, peak], complete_center=center,
        complete_maintenance=(c[row, center+2*window]-c[row, center+window])/window)


def process_features(auto: dict, targets: list[int], mode: str, cut: int) -> dict:
    """Describe saved pre-choice generating states, with mutually exclusive errors."""
    ids = auto['rule']
    active = ids >= 0
    if not targets or ids.ndim != 3 or cut < 1 or cut >= ids.shape[1]:
        raise ValueError('Require targets, sparse states and an internal split')
    for key in ('belief', 'contribution'):
        values = auto[key]
        if values.shape != ids.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid sparse state probabilities')
        np.testing.assert_allclose(values.sum(axis=2), 1., atol=1e-12, rtol=0)
        if np.any(values[~active] != 0):
            raise ValueError('Inactive padded slots have probability mass')
    target = np.isin(ids, targets) & active
    present = target.any(axis=2)
    belief = (auto['belief']*target).sum(axis=2)
    contribution = (auto['contribution']*target).sum(axis=2)
    precision_mass = (auto['precision']*auto['belief']*target).sum(axis=2)
    ever = present.any(axis=1)
    first = present.argmax(axis=1)
    after = np.arange(present.shape[1])[None, :] >= first[:, None]
    after &= ever[:, None]
    errors = auto['feedback'] != 1
    result = dict(ever_target=ever, first_target_trial=np.where(ever, first+1, np.nan),
        absent_after_entry=np.divide(((~present)&after).sum(axis=1), after.sum(axis=1),
            out=np.full(len(ids), np.nan), where=after.sum(axis=1)>0),
        late_target_present=present[:, cut:].mean(axis=1),
        late_target_belief=belief[:, cut:].mean(axis=1),
        late_target_contribution=contribution[:, cut:].mean(axis=1),
        late_target_precision=np.divide(precision_mass[:, cut:].sum(axis=1), belief[:, cut:].sum(axis=1),
            out=np.full(len(ids), np.nan), where=belief[:, cut:].sum(axis=1)>0),
        search_rate=auto['search'].mean(axis=1),
        late_feedback=auto['feedback'][:, cut:].mean(axis=1),
        error_absent=((~present)&errors).sum(axis=1), total_errors=errors.sum(axis=1))
    for label, mask in [('absent', ~present), ('present', present)]:
        result['correct_probability_' + label] = np.divide(
            (auto['correct_probability']*mask).sum(axis=1), mask.sum(axis=1),
            out=np.full(len(ids), np.nan), where=mask.sum(axis=1)>0)
        result['trials_' + label] = mask.sum(axis=1)
    if mode == 'MIX':
        if np.any(auto['executed'] != -1):
            raise ValueError('MIX must not have a discrete executed identity')
        for key in ('error_present_other', 'error_correct_executed', 'switch_rate', 'forced_rate',
                    'correct_probability_correct_executed', 'correct_probability_present_other',
                    'trials_correct_executed', 'trials_present_other'):
            result[key] = np.full(len(ids), np.nan)
    else:
        executed = np.isin(auto['executed'], targets)
        if not np.all((ids == auto['executed'][:, :, None]).any(axis=2)):
            raise ValueError('Execution is outside the active set')
        np.testing.assert_allclose(contribution, executed, atol=1e-12, rtol=0)
        result.update(error_present_other=(present&~executed&errors).sum(axis=1),
            error_correct_executed=(executed&errors).sum(axis=1),
            switch_rate=auto['switch'].mean(axis=1), forced_rate=auto['forced'].mean(axis=1))
        np.testing.assert_array_equal(result['error_absent']+result['error_present_other']+
                                      result['error_correct_executed'], errors.sum(axis=1))
        for label, mask in [('correct_executed', executed), ('present_other', present&~executed)]:
            result['correct_probability_' + label] = np.divide(
                (auto['correct_probability']*mask).sum(axis=1), mask.sum(axis=1),
                out=np.full(len(ids), np.nan), where=mask.sum(axis=1)>0)
            result['trials_' + label] = mask.sum(axis=1)
    return result


def fit_centers(features: np.ndarray, late_accuracy: np.ndarray, clusters: int, seed: int,
                n_init: int) -> np.ndarray:
    if len(features) < clusters or not np.isfinite(features).all():
        raise ValueError('Insufficient finite construction paths')
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=n_init).fit(features)
    sizes = np.bincount(model.labels_, minlength=clusters)
    if np.any(sizes == 0):
        raise ValueError('Empty shape cluster; inspect degeneracy rather than inventing categories')
    order = sorted(range(clusters), key=lambda k: (late_accuracy[model.labels_ == k].mean(), k))
    return model.cluster_centers_[order]


def assignments(features: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = cdist(features, centers)
    labels = distances.argmin(axis=1)
    return labels, distances[np.arange(len(features)), labels]


def feature_matrix(choices: np.ndarray, categories: np.ndarray, ncat: int, policy: dict) -> np.ndarray:
    return behavior_features(choices, categories, n_categories=ncat,
        block_width=policy['block_width'], temporal_weight=policy['temporal_weight'])


def checked_new_paths(jobs: list[dict], row: dict, categories: np.ndarray) -> dict:
    matching = [j for j in jobs if j['subject'] == row['subject'] and j['model'] == row['mode']
                and j['candidate']['id'] == row['candidate']['id']]
    if not matching:
        raise ValueError('Missing new parameter-point paths')
    parts = []
    for j in matching:
        part, meta = arrays(Path(j['target'])), read(Path(j['target']).with_suffix('.json'))
        if meta['candidate']['hyperparams'] != row['candidate']['hyperparams'] or meta['model'] != row['mode']:
            raise ValueError('Generated parameter identity changed')
        np.testing.assert_array_equal(part['categories'], categories)
        np.testing.assert_array_equal(part['seeds'], j['seeds'])
        parts.append(part)
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != 'categories'}
    if len(set(result['seeds'].tolist())) != len(result['seeds']):
        raise ValueError('Repeated trajectory seeds')
    probabilities = result['probabilities']
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError('Invalid generated probabilities')
    np.testing.assert_allclose(probabilities.sum(axis=2), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(result['correct_probability'],
        np.take_along_axis(probabilities, np.broadcast_to(categories[None, :, None]-1,
            probabilities.shape[:2]+(1,)), axis=2)[:, :, 0], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result['feedback'] == 1, result['choices'] == categories[None, :])
    return result


def summarize_paths(auto: dict, obs: dict, cut: int, ncat: int, targets: list[int], mode: str,
                    policy: dict) -> tuple[dict, dict, dict]:
    behavior = trajectory_diagnostics(auto['choices'], obs['categories'], policy['jump_window'],
                                     policy['low_window'], policy['low_stride'], ncat)
    behavior.update(complete_followup(auto['choices'], obs['categories'], policy['jump_window']))
    behavior['late_accuracy'] = (auto['choices'][:, cut:] == obs['categories'][None, cut:]).mean(axis=1)
    truth = trajectory_diagnostics(obs['choices'][None, :], obs['categories'], policy['jump_window'],
                                  policy['low_window'], policy['low_stride'], ncat)
    truth.update(complete_followup(obs['choices'][None, :], obs['categories'], policy['jump_window']))
    truth['late_accuracy'] = np.array([(obs['choices'][cut:] == obs['categories'][cut:]).mean()])
    joint = ((behavior['complete_gain'] >= truth['complete_gain'][0]) &
             (behavior['complete_maintenance'] >= truth['complete_maintenance'][0]))
    events = dict(jump_and_maintenance=joint,
        jump_maintenance_and_late=joint & (behavior['late_accuracy'] >= truth['late_accuracy'][0]),
        late_at_least_observed=behavior['late_accuracy'] >= truth['late_accuracy'][0],
        low_at_most_observed=behavior['minimum_window_accuracy'] <= truth['minimum_window_accuracy'][0],
        low_and_bias_adjusted=(behavior['minimum_window_accuracy'] <= truth['minimum_window_accuracy'][0]) &
          (behavior['minimum_window_accuracy_minus_marginal_chance'] <=
           truth['minimum_window_accuracy_minus_marginal_chance'][0]))
    behavior.update(process_features(auto, targets, mode, cut))
    return behavior, {k: v[0].item() for k, v in truth.items()}, events


def analyze(config: Path, generation: Path, output: Path) -> None:
    generation = generation.resolve()
    if output.exists():
        raise FileExistsError('Use a new analysis directory')
    policy = policy_at(config)
    study, frozen, old_jobs = frozen_study(policy)
    complete, generated = read(generation / 'completion.json'), read(generation / 'protocol.json')
    if (complete['status'] != 'complete' or complete['smoke'] or generated['policy'] != policy
            or complete['protocol_sha256'] != digest(generation / 'protocol.json')):
        raise ValueError('Require completed formal generation matching this analysis')
    nomination = Path(generated['nomination'])
    if generated['nomination_sha256'] != digest(nomination / 'nomination.json'):
        raise ValueError('Parameter nomination changed')
    plan = read(nomination / 'nomination.json')
    output.mkdir(parents=True)
    (output / 'curves').mkdir()
    rows, cluster_rows, event_rows, summary, stability, observed_rows, center_bank = [], [], [], [], [], [], {}
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in
        (study/'nominees.json', study/'autonomous_jobs.json', generation/'protocol.json')}
    train_count = policy['construction_paths_per_mode']
    for sid in policy['subjects']:
        cases, features, accuracies = {}, {}, {}
        geometry_file = ROOT / policy['geometry'] / f'subject_{sid}' / 'audit.json'
        targets = read(geometry_file)['training_perfect_rules']
        input_hashes[str(geometry_file.relative_to(ROOT))] = digest(geometry_file)
        for mode in policy['modes']:
            nominee = next(r for r in frozen['nominees'] if r['subject'] == sid and r['arm'] == mode)
            case = load_case(study, nominee, old_jobs)
            cases[mode] = dict(case, candidate=nominee['candidate'])
            auto, obs, meta = [case[k] for k in ('autonomous', 'observed', 'meta')]
            ncat = 2 if meta['condition'] == 1 else 4
            features[mode] = feature_matrix(auto['choices'], obs['categories'], ncat, policy)
            accuracies[mode] = (auto['choices'][:, meta['training_trials']:] ==
                               obs['categories'][None, meta['training_trials']:]).mean(axis=1)
            if len(features[mode]) != 2*train_count:
                raise ValueError('Original construction/evaluation split is not balanced')
        # Each subject has common frozen shape centers. Human responses do
        # not enter center fitting, scaling or model complexity selection.
        train_x = np.concatenate([features[m][:train_count] for m in policy['modes']])
        check_x = np.concatenate([features[m][train_count:] for m in policy['modes']])
        train_acc = np.concatenate([accuracies[m][:train_count] for m in policy['modes']])
        check_acc = np.concatenate([accuracies[m][train_count:] for m in policy['modes']])
        centers_by_k, reference = {}, {}
        for k in policy['clusters']:
            centers = fit_centers(train_x, train_acc, k, policy['cluster_seed'], policy['cluster_n_init'])
            repeated = fit_centers(check_x, check_acc, k, policy['cluster_seed'], policy['cluster_n_init'])
            all_x = np.concatenate([train_x, check_x])
            labels, distances = assignments(check_x, centers)
            centers_by_k[k] = centers
            reference[k] = (labels, distances)
            ari = adjusted_rand_score(assignments(all_x, centers)[0], assignments(all_x, repeated)[0])
            stability.append(dict(subject=sid, clusters=k, ari=ari, construction_paths=len(train_x),
                                  evaluation_paths=len(check_x)))
        center_bank[str(sid)] = {str(k): c.tolist() for k, c in centers_by_k.items()}
        obs, meta = cases['MIX']['observed'], cases['MIX']['meta']
        cut, ncat = meta['training_trials'], 2 if meta['condition'] == 1 else 4
        empirical = feature_matrix(obs['choices'][None, :], obs['categories'], ncat, policy)[0]
        ensembles = [dict(subject=sid, mode=m, role='nominee', batch='original',
                         candidate=cases[m]['candidate'], autonomous=cases[m]['autonomous']) for m in policy['modes']]
        for r in plan['rows']:
            if r['subject'] == sid:
                auto = checked_new_paths(generated['jobs'], r, obs['categories'])
                if len(auto['seeds']) != policy['sensitivity']['paths']:
                    raise ValueError('Incomplete independent generation')
                ensembles.append(dict(r, batch='independent', autonomous=auto))
        for ensemble in ensembles:
            mode, auto = ensemble['mode'], ensemble['autonomous']
            ident = dict(subject=sid, mode=mode, role=ensemble['role'], batch=ensemble['batch'],
                         candidate=ensemble['candidate']['id'])
            behavior, truth, events = summarize_paths(auto, obs, cut, ncat, targets, mode, policy)
            if mode == 'MIX' and ensemble['batch'] == 'original':
                observed_rows.append(dict(subject=sid, condition=meta['condition'], trials=meta['trials'],
                    training_trials=cut, late_feedback=float(obs['feedback'][cut:].mean()), **truth))
            x = feature_matrix(auto['choices'], obs['categories'], ncat, policy)
            full_score = energy_u(x, empirical)
            late_score = energy_u(feature_matrix(auto['choices'][:, cut:], obs['categories'][cut:], ncat, policy),
                feature_matrix(obs['choices'][None, cut:], obs['categories'][cut:], ncat, policy)[0])
            summary.append(dict(**ident, paths=len(x), full_energy=full_score['value'],
                full_energy_mcse=full_score['mc_se'], late_energy=late_score['value'],
                late_energy_mcse=late_score['mc_se'],
                **{key: float(np.nanmean(value)) if np.isfinite(value).any() else np.nan
                   for key, value in behavior.items()}))
            for key, event in events.items():
                event_rows.append(dict(**ident, event=key, **probability_record(event)))
            labels_main = None
            for k, centers in centers_by_k.items():
                labels, distances = assignments(x, centers)
                human_label, human_distance = assignments(empirical[None, :], centers)
                ref_labels, ref_distances = reference[k]
                if k == 3:
                    labels_main = labels
                split_masks = [('all', np.ones(len(x), bool))]
                if ensemble['batch'] == 'original':
                    split_masks += [('construction', np.arange(len(x)) < train_count),
                                    ('evaluation', np.arange(len(x)) >= train_count)]
                for split, mask in split_masks:
                    for cluster in range(k):
                        members = mask & (labels == cluster)
                        reference_dist = ref_distances[ref_labels == cluster]
                        cluster_rows.append(dict(**ident, clusters=k, cluster=cluster, split=split,
                            **probability_record(labels[mask] == cluster),
                            observed_cluster=int(human_label[0]), observed_distance=float(human_distance[0]),
                            reference_paths=len(reference_dist),
                            reference_radius90=float(np.quantile(reference_dist, .9)) if len(reference_dist) else np.nan,
                            observed_distance_percentile=float((reference_dist <= human_distance[0]).mean())
                                if len(reference_dist) and cluster == human_label[0] else np.nan,
                            mean_late_accuracy=float(behavior['late_accuracy'][members].mean()) if members.any() else np.nan))
            for i, seed in enumerate(auto['seeds']):
                rows.append(dict(**ident, seed=int(seed), path_index=i, cluster=int(labels_main[i]),
                                 **{key: value[i].item() for key, value in behavior.items()},
                                 **{key: bool(value[i]) for key, value in events.items()}))
            # Every path is retained in a compact plot input. Representative
            # paths are medoids within each primary class, never the closest
            # simulated path to the participant.
            representatives = []
            for cluster in range(3):
                indices = np.flatnonzero(labels_main == cluster)
                representatives.append(int(indices[np.argmin(cdist(x[indices], x[indices]).mean(axis=1))])
                                       if len(indices) else -1)
            block = policy['block_width']
            correct = auto['choices'] == obs['categories'][None, :]
            curves = np.stack([correct[:, start:start+block].mean(axis=1)
                               for start in range(0, correct.shape[1], block)], axis=1)
            observed_curve = np.array([(obs['choices'][start:start+block] == obs['categories'][start:start+block]).mean()
                                      for start in range(0, len(obs['choices']), block)])
            identity = f'S{sid}_{mode}_{ensemble["batch"]}_{ensemble["role"]}'
            np.savez_compressed(output/'curves'/f'{identity}.npz', curves=curves, observed=observed_curve,
                seeds=auto['seeds'], labels=labels_main, representatives=representatives,
                end_trial=np.minimum(np.arange(1, curves.shape[1]+1)*block, correct.shape[1]), cut=cut)
        print(f'S{sid}: original distributions and registered sensitivity points analyzed', flush=True)
    for name, data in [('paths', rows), ('clusters', cluster_rows), ('events', event_rows),
                       ('summary', summary), ('stability', stability), ('observed', observed_rows)]:
        pd.DataFrame(data).to_csv(output / f'{name}.csv', index=False)
    write(output / 'centers.json', center_bank)
    write(output / 'completion.json', dict(status='complete', scope=policy['scope'],
        original_paths=sum(r['batch'] == 'original' for r in rows),
        independent_paths=sum(r['batch'] == 'independent' for r in rows),
        candidate_sets_are_posteriors=False, policy=policy, provenance=manifest(config),
        inputs_sha256=input_hashes, generation=str(generation.resolve()),
        artifacts_sha256={str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('nominate', 'generate', 'analyze'), required=True)
    for key in ('config', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('nomination', 'generation'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == 'nominate':
            nominate(args.config, args.output)
        elif args.phase == 'generate' and args.nomination:
            generate(args.config, args.nomination, args.output, args.smoke)
        elif args.phase == 'analyze' and args.generation and not args.smoke:
            analyze(args.config, args.generation, args.output)
        else:
            parser.error('Generate needs --nomination; analyze needs --generation and a formal run')


if __name__ == '__main__':
    main()
