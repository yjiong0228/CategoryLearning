"""Source tables for descriptive Fig3/4: no type assignment or causal inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bottleneck_analysis import ROOT, conditional_support, contiguous_runs, sha256
from .nine_subject_states import CONFIG, cohort
from ..fig1.behavior import order_trials, adjacent_gain
from src.Bayesian_state.evaluation.oral.scoring import OralAlignmentScoringMixin
from src.Bayesian_state.hypothesis_space import ContinuousPartition

TASKS = {1: 1, 3: 2, 2: 3}
FIELDS = {'marginal_prior': 'belief', 'marginal_active_probability': 'available',
          'marginal_executed_probability': 'executed',
          'predictive_swap_probability': 'search', 'predictive_search_range': 'global_range',
          'predictive_replacement_fraction': 'replacement',
          'predictive_newcomer_distance': 'newcomer_distance',
          'predictive_execution_switch_event_probability': 'execution_switch',
          'predictive_failure_pressure': 'failure_pressure',
          'predictive_executed_beta': 'executed_beta'}


def first_sustained(values: np.ndarray, threshold: float, duration: int) -> float:
    """One-based start of first strictly-above-threshold run; absent is NaN."""
    runs = contiguous_runs(np.asarray(values) > threshold, duration)
    return float(runs[0][0] + 1) if runs else np.nan


def feedback_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Associate previous feedback with pre-choice diagnostics, within session.

    The difference in marginal belief includes observer filtering and cognitive
    changes. It is not a causal effect of feedback or a within-particle update.
    """
    d = frame.copy()
    d['previous_feedback'] = d.feedback.shift()
    d['previous_belief'] = d.belief.shift()
    d['belief_change'] = d.belief - d.previous_belief
    d = d.loc[d.iSession.eq(d.iSession.shift())]
    out = []
    for feedback, group in d.groupby('previous_feedback'):
        strong = group.loc[group.previous_belief > .5]
        out.append({'feedback': feedback, 'n': len(group), 'search': group.search.mean(),
                    'global_range': group.global_range.mean(),
                    'replacement': group.replacement.mean(),
                    'strong_n': len(strong), 'strong_belief_change': strong.belief_change.mean(),
                    'strong_retention': (strong.belief > .5).mean() if len(strong) else np.nan})
    return pd.DataFrame(out)


def block_table(frame: pd.DataFrame, window: int = 64) -> pd.DataFrame:
    """All nonoverlapping blocks, including a final partial block if present."""
    return frame.assign(block=(frame.trial - 1) // window).groupby('block').agg(
        start=('trial', 'min'), end=('trial', 'max'), n=('trial', 'size'),
        accuracy=('correct', 'mean'), predicted=('predicted', 'mean'),
        available=('available', 'mean'), belief=('belief', 'mean'), executed=('executed', 'mean'),
        search=('search', 'mean'), replacement=('replacement', 'mean'),
        global_range=('global_range', 'mean'), pairing=('pairing', 'mean')).reset_index()


def load_repeats(state_root: Path, sid: int, variant: str, count: int) -> dict:
    runs = [dict(np.load(state_root / f'S{sid}' / f'{variant}_{r:02d}.npz', allow_pickle=False))
            for r in range(count)]
    for run in runs:
        if int(run['subject']) != sid or int(run['particles']) != 128:
            raise ValueError('Wrong subject or particle budget')
        for name in ('observed_choice', 'observed_feedback', 'true_category_index',
                     'valid_trial_mask', 'score_trial_mask', 'point_id'):
            np.testing.assert_array_equal(run[name], runs[0][name])
    return {k: np.stack([r[k] for r in runs]) for k in runs[0]}


def build(state_root: Path, output: Path, config_path: Path = CONFIG) -> None:
    state_root = state_root.resolve()
    config_path=config_path.resolve()
    config = json.loads(config_path.read_text())
    rows = cohort(config)
    state_manifest = json.loads((state_root / 'manifest.json').read_text())
    if state_manifest['config'] != config or state_manifest['smoke']:
        raise ValueError('State export/config mismatch')
    if not (state_root / 'completion.json').exists():
        raise ValueError('State replay incomplete')
    output.mkdir(parents=True, exist_ok=False)
    raw_path = ROOT / 'data/exp123/processed/Task2_processed.csv'
    data = order_trials(pd.read_csv(raw_path))
    encoder = OralAlignmentScoringMixin()
    partitions = {c: ContinuousPartition(n_dims=4, n_cats=2 if c == 1 else 4) for c in (1, 2, 3)}
    summaries, frames, blocks, feedbacks, sensitivity, seed_events, candidates = [], [], [], [], [], [], []
    split_checks = []
    for row in rows:
        result = row['fit']; sid = result['subject']; condition = result['condition']
        d = data.loc[data.iSub.eq(sid)].reset_index(drop=True).copy()
        n, target = len(d), 0 if condition == 1 else 42
        selected = load_repeats(state_root, sid, 'selected', config['selected_repeats'])
        alternative = load_repeats(state_root, sid, 'alternative', config['alternative_repeats'])
        for run in (selected, alternative):
            np.testing.assert_array_equal(run['observed_choice'][0], d.choice.to_numpy())
            np.testing.assert_array_equal(run['observed_feedback'][0], d.feedback.to_numpy())
            np.testing.assert_array_equal(run['true_category_index'][0], d.category.to_numpy() - 1)
        np.testing.assert_array_equal(d.correct, d.choice.eq(d.category))
        labels = partitions[condition].get_category_assignments(
            target, d[[f'feature{i}' for i in range(1, 5)]].to_numpy(), distance_mode='boundary')
        np.testing.assert_array_equal(labels, d.category.to_numpy() - 1)
        for name, label in FIELDS.items():
            if name in selected:
                x = selected[name]
                x = x[:, :, target] if x.ndim == 3 else x
                d[label] = x.mean(axis=0)
                d[label + '_seed_sd'] = x.std(axis=0, ddof=1)
            else:
                d[label] = np.nan
        q_all = selected['marginal_prior'].mean(axis=0)
        active_all = selected['marginal_active_probability'].mean(axis=0)
        np.testing.assert_allclose(active_all.sum(axis=1), result['parameters']['M'], atol=1e-9)
        d['conditional_support'] = conditional_support(d.belief.to_numpy(), d.available.to_numpy())
        d['alternative_belief'] = alternative['marginal_prior'][:, :, target].mean(axis=0)
        d['pairing'] = selected['pairing_prior'][:, :, 0].mean(axis=0) if condition == 3 else np.nan
        if condition == 3:
            np.testing.assert_allclose(selected['pairing_prior'].sum(axis=2), 1, atol=1e-9)
        p = selected['pred_category_probs'].mean(axis=0)
        d['predicted'] = p[np.arange(n), d.category.to_numpy(dtype=int) - 1]
        d['scored'] = selected['valid_trial_mask'][0] & selected['score_trial_mask'][0]
        d['task'] = TASKS[condition]
        for col in ['correct', 'predicted', 'available', 'belief', 'executed', 'search', 'pairing']:
            d[col + '_w32'] = d[col].rolling(config['display_window'], min_periods=config['display_window']).mean()
        oral = encoder.compute_oral_mass_probabilities(
            d, subjects=[sid], oral_center_sigma=config['oral_sigma'],
            partitions_by_subject={sid: partitions[condition]})[sid]
        current = np.asarray(oral['instantaneous_oral_mass'])
        valid = np.asarray(oral['valid_oral_report'], dtype=bool)
        d['oral_valid'] = valid
        d['oral_current_target'] = current[:, target]
        # A current category-specific report can tie between several full rules.
        best = np.full(n, np.nan)
        best[valid] = np.max(current[valid], axis=1)
        d['oral_target_top'] = valid & np.isclose(current[:, target], best, rtol=1e-6, atol=1e-12)
        d['oral_top_ties'] = np.sum(np.isclose(current, best[:, None], rtol=1e-6, atol=1e-12), axis=1)
        d['oral_carried_target'] = np.asarray(oral['oral_mass'])[:, target]
        np.savez_compressed(output / f'S{sid}_rule_distributions.npz',
                            belief=q_all, available=active_all,
                            instantaneous_oral=current, carried_oral=oral['oral_mass'],
                            oral_valid=valid)
        s = {'subject': sid, 'condition': condition, 'task': TASKS[condition], 'n': n,
             'target': target, **result['parameters'], 'status': result['status'],
             'issues': '|'.join(result['issues']), 'selected': result['selected'],
             'alternative': row['alternative']['id'], 'accuracy': d.correct.mean(),
             'end64_accuracy': d.correct.iloc[-64:].mean(), 'valid_reports': int(valid.sum()),
             'belief_mean': d.belief.mean(), 'belief_end64': d.belief.iloc[-64:].mean(),
             'search_mean': d.search.mean(), 'replacement_mean': d.replacement.mean(),
             'search_range_mean': d.global_range.mean()}
        score = d.scored.to_numpy()
        s['replay_nll'] = float(-np.log(p[np.arange(n), d.choice.to_numpy(dtype=int)-1][score]).mean())
        s['fit_audit_nll'] = result['selected_mean_nll']
        s['criterion'] = first_sustained(d.correct.rolling(config['criterion_window']).mean().to_numpy(),
                                         config['criterion_threshold'], 1)
        s['max_gain32'], s['max_gain32_split'] = adjacent_gain(d.correct.to_numpy(), 32)
        for name in ('available', 'belief', 'executed'):
            s[name + '_event'] = first_sustained(d[name].to_numpy(), config['state_threshold'], config['state_duration'])
        for threshold in (.4, .5, .6, .75):
            for duration in (8, 16, 32):
                for name in ('available', 'belief', 'executed'):
                    sensitivity.append({'subject': sid, 'state': name, 'threshold': threshold,
                                        'duration': duration,
                                        'event': first_sustained(d[name].to_numpy(), threshold, duration)})
        for variant, run in [('selected', selected), ('alternative', alternative)]:
            for name, label in [('marginal_active_probability', 'available'), ('marginal_prior', 'belief'),
                                ('marginal_executed_probability', 'executed')]:
                if name not in run:
                    continue
                for rep, x in enumerate(run[name]):
                    seed_events.append({'subject': sid, 'variant': variant, 'repeat': rep,
                                        'state': label, 'event': first_sustained(x[:, target], .5, 16)})
            x = run['marginal_prior'][:, :, target].mean(axis=0)
            candidates.append({'subject': sid, 'variant': variant, 'belief_mean': x.mean(),
                               'belief_event': first_sustained(x, .5, 16),
                               'belief_end64': x[-64:].mean()})
        x = selected['marginal_prior'][:, :, target]
        split_checks.append({'subject': sid, 'split_half_belief_mae': np.abs(x[:4].mean(0)-x[4:].mean(0)).mean(),
                             'alternative_belief_mae': np.abs(d.belief-d.alternative_belief).mean(),
                             'mean_seed_sd': x.std(0, ddof=1).mean()})
        b = block_table(d); b['subject'] = sid; b['task'] = TASKS[condition]; blocks.append(b)
        f = feedback_table(d); f['subject'] = sid; f['task'] = TASKS[condition]; feedbacks.append(f)
        frames.append(d); summaries.append(s)
        print(f'S{sid}: {n} trials; belief event {s["belief_event"]}; behavior {s["criterion"]}', flush=True)
    tables = {'trials': pd.concat(frames, ignore_index=True), 'subjects': pd.DataFrame(summaries),
              'blocks64': pd.concat(blocks, ignore_index=True), 'feedback': pd.concat(feedbacks, ignore_index=True),
              'threshold_sensitivity': pd.DataFrame(sensitivity), 'seed_events': pd.DataFrame(seed_events),
              'candidate_sensitivity': pd.DataFrame(candidates), 'numerical_checks': pd.DataFrame(split_checks)}
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    inputs = {str(raw_path.relative_to(ROOT)): sha256(raw_path), str(config_path.relative_to(ROOT)): sha256(config_path)}
    inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in state_root.rglob('*.npz')})
    manifest = {'config': config, 'input_sha256': inputs, 'participants': len(rows),
                'trial_count': len(tables['trials']), 'scope': 'Descriptive, full-sequence fitted parameters; pre-choice state filtering.',
                'event_definition': 'First 16 consecutive marginal estimates > 0.5; not first latent entry or confirmed mastery.',
                'criterion_definition': 'First trailing 64-trial accuracy >0.9; later declines retained.',
                'oral_definition': 'Current post-choice, pre-feedback report; target among top encoder rules may include ties.',
                'uncertainty': 'PF seed spread is numerical, not between-participant uncertainty; one near-candidate sensitivity only.',
                'exclusions': 'No participants/trials dropped. Lagged feedback comparisons exclude first trial of each session. Structural execution NA retained for mixture models.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    build(args.states, args.output, args.config)


if __name__ == '__main__':
    main()
