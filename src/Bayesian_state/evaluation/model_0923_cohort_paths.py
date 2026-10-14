"""Pure summaries of complete autonomous paths, with participant-level weighting.

Task-compatible rules are scored on physical stimuli for retrospective diagnosis.
A set residence continues when one compatible rule replaces another. Exit risks
exclude the final trial, whose following transition is unobserved. Path intervals
measure stochastic variation at fixed fitted parameters, not parameter uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def set_residences(active: np.ndarray, horizon: int) -> list[dict]:
    """Contiguous presence of any compatible rule, allowing member replacement."""
    active = np.asarray(active, bool)
    if active.ndim != 2 or horizon < 1:
        raise ValueError('Expected trial-by-path presence and a positive horizon')
    rows = []
    n, paths = active.shape
    for path in range(paths):
        edges = np.diff(np.r_[False, active[:, path], False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            rows.append(dict(path=path, start_trial=int(start+1), last_trial=int(end), duration=int(end-start),
                right_censored=bool(end == n), retention_eligible=bool(start+horizon <= n),
                retained=bool(end-start >= horizon)))
    return rows


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def removal_cause(a: dict, trial: int, path: int, rule: int) -> str:
    """Keep joint set omissions distinct from legacy exit/replacement events."""
    if 'joint_dropped' in a and rule in a['joint_dropped'][trial,path]:
        return 'joint_selection'
    if rule in a['exits'][trial,path]:
        return 'autonomous_exit'
    return 'capacity' if a['capacity_dropped'][trial,path] == rule else 'retained'


def summarize_trace(a: dict, rules: list[int], late_window: int, support: float, horizon: int):
    ids = a['rule']
    n, paths, _ = ids.shape
    mask = np.isin(ids, rules)
    active = mask.any(axis=2)
    mass = {key: np.where(mask, a[key], 0.).sum(axis=2)
            for key in ('prior', 'posterior', 'forgotten', 'contribution')}
    correct = np.take_along_axis(a['prediction'], a['categories'][:, None, None]-1, axis=2)[..., 0]
    late = slice(max(0, n-late_window), n)
    rows = []
    for path in range(paths):
        hits = np.flatnonzero(active[:, path])
        rows.append(dict(path=path, ever_active=bool(len(hits)),
            first_entry=float(hits[0]+1) if len(hits) else np.nan,
            ever_supported=bool(np.any(mass['posterior'][:, path] >= support)),
            ever_contributing=bool(np.any(mass['contribution'][:, path] >= support)),
            proposed=int((a['proposed'][:, path] >= 0).sum()),
            useful_proposed=int(np.isin(a['proposed'][:, path], rules).sum()),
            late_active=float(active[late, path].mean()),
            late_contribution=float(mass['contribution'][late, path].mean()),
            late_correct=float(correct[late, path].mean()),
            late_realized=float((a['choice'][late, path] == a['categories'][late]).mean()),
            overall_correct=float(correct[:, path].mean())))
    exits, exposure = [], []
    # A rule's t+1 exit is associated with its post-feedback state at t.
    for trial, path, slot in zip(*np.where(mask[:-1])):
        rule = int(ids[trial, path, slot])
        cause = removal_cause(a, trial+1, path, rule)
        high = bool(a['posterior'][trial, path, slot] >= support)
        gain = float(a['posterior'][trial, path, slot]-a['forgotten'][trial, path, slot])
        exposure.append(dict(high_support=high, exited=cause == 'autonomous_exit', capacity=cause == 'capacity',
                             joint=cause == 'joint_selection'))
        if cause != 'retained':
            exits.append(dict(path=int(path), rule=rule, last_trial=int(trial+1), cause=cause,
                posterior=float(a['posterior'][trial, path, slot]),
                contribution=float(a['contribution'][trial, path, slot]), feedback_support_gain=gain,
                feedback=float(a['feedback'][trial, path]), high_support=high,
                set_lost=not bool(active[trial+1, path])))
    curve = pd.DataFrame(dict(trial=np.arange(1, n+1), active=active.mean(axis=1),
        contribution=mass['contribution'].mean(axis=1), correct=correct.mean(axis=1),
        correct_q10=np.quantile(correct, .1, axis=1), correct_q90=np.quantile(correct, .9, axis=1)))
    late_mask = mask[late]
    slot_stats = []
    for key in ('prior', 'posterior', 'contribution', 'beta_pre', 'rule_correct', 'pairing_pre_true'):
        values = a[key][late][late_mask]
        values = values[np.isfinite(values)]
        slot_stats.append(dict(measure=key, total=float(values.sum()), count=len(values)))
    return rows, set_residences(active, horizon), exits, exposure, curve, slot_stats, correct


def summarize_cohort(output: Path, records: list[dict], settings: dict) -> None:
    summaries, all_paths, all_episodes, all_exits, all_curves, parameter_rows = [], [], [], [], [], []
    checks = []
    # The main threshold is specified before seeing the cohort; adjacent values
    # expose sensitivity of a retrospective diagnostic set, not model changes.
    thresholds = sorted(set([.7, settings['geometric_accuracy_threshold'], .9]))
    for record in records:
        directory = output / record['id']
        prepared = json.loads((directory / 'prepared.json').read_text())
        metadata = {k: record[k] for k in ('id', 'subject', 'dataset', 'primary')}
        metadata.update(condition=prepared['condition'], trials=prepared['trials'], fit_status=prepared['fit_status'])
        catalog = pd.read_csv(directory / 'rule_catalog.csv')
        observed = pd.read_csv(directory / 'observed_trials.csv')
        for key, value in json.loads((directory / 'parameters.json').read_text()).items():
            parameter_rows.append(dict(metadata, parameter=key.split('.')[-1], value=value))
        traces = []
        for file in sorted((directory / 'traces').glob('C_*.npz')):
            with np.load(file) as z:
                traces.append((file.stem, {k: z[k] for k in z.files}))
            checks.append(dict(id=record['id'], chunk=file.stem, **json.loads(file.with_suffix('.json').read_text())))
        for threshold in thresholds:
            rules = catalog.loc[catalog.physical_argmax_accuracy >= threshold, 'rule'].astype(int).tolist()
            path_rows, episode_rows, exit_rows, risk_rows, slots, correctness = [], [], [], [], [], []
            offset = 0
            for batch, a in traces:
                p, ep, ex, risk, curve, stats, correct = summarize_trace(a, rules, settings['late_window'],
                    settings['support_threshold'], settings['retention_trials'])
                tags = dict(metadata, threshold=threshold)
                for rows in (p, ep, ex):
                    for row in rows:
                        row['path'] += offset
                        row.update(tags, batch=batch)
                path_rows.extend(p); episode_rows.extend(ep); exit_rows.extend(ex); risk_rows.extend(risk)
                slots.extend(stats); correctness.append(correct)
                offset += a['rule'].shape[1]
            paths, episodes, exits, risk = map(pd.DataFrame, (path_rows, episode_rows, exit_rows, risk_rows))
            row = dict(metadata, threshold=threshold, rules=','.join(map(str, rules)), paths=len(paths))
            for key in ('ever_active', 'ever_supported', 'ever_contributing', 'proposed', 'useful_proposed',
                        'late_active', 'late_contribution', 'late_correct', 'late_realized', 'overall_correct'):
                row[key] = float(paths[key].mean())
            row['proposals_per_100_trials'] = row['proposed']/prepared['trials']*100
            row['proposal_hit_fraction'] = ratio(paths.useful_proposed.sum(), paths.proposed.sum())
            row['first_entry_median_given_found'] = float(paths.first_entry.median())
            for q in (.1, .5, .9):
                row[f'late_correct_q{int(q*100)}'] = float(paths.late_correct.quantile(q))
            row['late_observed'] = float((observed.choice == observed.category).iloc[-settings['late_window']:].mean())
            row['late_conditional'] = float(observed.conditional_correct.iloc[-settings['late_window']:].mean()) if 'conditional_correct' in observed else np.nan
            row['episodes'] = len(episodes)
            row['retention_eligible'] = int(episodes.retention_eligible.sum()) if len(episodes) else 0
            row['retention_fraction'] = ratio(int((episodes.retention_eligible & episodes.retained).sum()), row['retention_eligible']) if len(episodes) else np.nan
            row['right_censored'] = int(episodes.right_censored.sum()) if len(episodes) else 0
            row['rule_exits'] = len(exits)
            row['set_losses'] = len(exits.loc[exits.set_lost, ['path', 'last_trial']].drop_duplicates()) if len(exits) else 0
            row['autonomous_exits'] = int((exits.cause == 'autonomous_exit').sum()) if len(exits) else 0
            row['supported_autonomous_exits'] = int(((exits.cause == 'autonomous_exit') & exits.high_support).sum()) if len(exits) else 0
            row['positive_feedback_support_exits'] = int(((exits.cause == 'autonomous_exit') & (exits.feedback_support_gain > 1e-12)).sum()) if len(exits) else 0
            row['capacity_exits'] = int((exits.cause == 'capacity').sum()) if len(exits) else 0
            for high, name in ((True, 'high'), (False, 'low')):
                group = risk.loc[risk.high_support == high] if len(risk) else risk
                row[name+'_support_exposure'] = len(group)
                row[name+'_support_exit_hazard'] = float(group.exited.mean()) if len(group) else np.nan
            for measure, group in pd.DataFrame(slots).groupby('measure'):
                row['late_'+measure+'_given_active'] = ratio(group.total.sum(), group['count'].sum())
            summaries.append(row)
            if threshold == settings['geometric_accuracy_threshold']:
                all_paths.extend(path_rows); all_episodes.extend(episode_rows); all_exits.extend(exit_rows)
                # Quantiles must pool whole paths, never average chunk quantiles.
                correct = np.concatenate(correctness, axis=1)
                curve_rows = []
                for start in range(0, len(correct), 32):
                    end = min(start+32, len(correct))
                    per_path = correct[start:end].mean(axis=0)
                    curve_rows.append(dict(metadata, first_trial=start+1, last_trial=end,
                        observed=float((observed.choice == observed.category).iloc[start:end].mean()),
                        conditional=float(observed.conditional_correct.iloc[start:end].mean()) if 'conditional_correct' in observed else np.nan,
                        autonomous=float(per_path.mean()), q10=float(np.quantile(per_path, .1)),
                        q50=float(np.quantile(per_path, .5)), q90=float(np.quantile(per_path, .9))))
                all_curves.extend(curve_rows)
                # Full paths retained for background trajectories in report figures.
                with (directory / 'correct_paths.npz').open('xb') as stream:
                    np.savez_compressed(stream, correct_probability=correct)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'threshold_sensitivity.csv', index=False)
    main = summary.loc[summary.threshold == settings['geometric_accuracy_threshold']]
    main.to_csv(output / 'subjects.csv', index=False)
    for name, rows in [('paths', all_paths), ('set_residences', all_episodes), ('exits', all_exits),
                        ('curves', all_curves), ('parameters', parameter_rows)]:
        pd.DataFrame(rows).to_csv(output / (name+'.csv'), index=False)
    measures = ['ever_active', 'late_active', 'retention_fraction', 'late_correct', 'late_observed',
                'late_conditional', 'late_rule_correct_given_active', 'proposals_per_100_trials']
    # One fixed representative per participant; initial/Plus duplicates remain
    # visible in subject tables but never inflate these descriptive stratum means.
    main.loc[main.primary].groupby(['dataset', 'condition'])[measures].agg(['count', 'mean', 'median']).to_csv(output / 'strata.csv')
    with (output / 'validation.json').open('x') as stream:
        json.dump(dict(status='passed', checks=checks, primary_participants=int(main.primary.sum()),
                       fit_records=len(main), path_records=int(main.paths.sum())), stream, indent=2)
