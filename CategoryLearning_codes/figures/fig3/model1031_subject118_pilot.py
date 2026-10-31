"""Individual pilot: behavior, representation use and provisional oral correspondence.

The original S118 module name remains a compatible entry point; subject identity,
trial count and final candidate labels are read from the supplied report.

One participant; all available trials; two numerical repeats per final parameter;
64 independent autonomous paths per parameter. The interval in the behavior
panel is a pointwise 10th–90th percentile of paths, not a confidence interval.
Parameters use the complete behavioral record. Oral data selected this development
case but never weighted particles or parameter scores. No hypothesis tests.
PNG-only export follows the repository instructions for research review figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


NEW, OLD, HUMAN = '#0072B2', '#B45B18', '#171717'
GROUP_COLORS = ['#A0A0A0', '#E69F00', '#009E73', '#8C63AB']
WIDTH = 183 / 25.4


def trailing(values: np.ndarray, minimum: int = 32) -> np.ndarray:
    """Average the last 32 trials, preserving gaps with too few scored reports."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        return pd.Series(values).rolling(32, min_periods=minimum).mean().to_numpy()
    return np.stack([trailing(row, minimum) for row in values])


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(title, loc='left', pad=9, fontsize=10)
    ax.text(-.09, 1.06, letter, transform=ax.transAxes, fontweight='bold', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E8E8E8', lw=.5, zorder=0)
    ax.set_axisbelow(True)


def trial_axis(ax, n: int) -> None:
    ax.set_xlim(1, n)
    ax.set_xticks(sorted(set([1, n, *range(64, n + 1, 64)])))
    ax.set_xlabel('Trial')


def run(report: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    source = report / 'plot_data.npz'
    with np.load(source, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    summary = json.loads((report / 'summary.json').read_text())
    points = pd.read_csv(report / 'final_candidates.csv')
    trial = data['trial']; n = len(trial)
    subject = int(summary.get('subject', 118))
    alternative = points.loc[(points.model == 'model_1031') & ~points.selected, 'candidate'].iloc[0]
    for name in ('model_1031', 'model_1024'):
        assert data[name + '_generated'].shape == (64, n)
        np.testing.assert_allclose(data[name + '_feature_sets'].sum(axis=1), 1.)
        np.testing.assert_allclose(data[name + '_group'].sum(axis=1), 1.)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.linewidth': .6, 'lines.linewidth': 1.4,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'legend.frameon': False})
    output.mkdir(parents=True)
    figures = []

    def save(fig, name):
        fig.savefig(output / name, dpi=300)
        plt.close(fig)
        figures.append(name)

    fig, axs = plt.subplots(4, 1, figsize=(WIDTH, 9.5), sharex=True, layout='constrained')
    fig.suptitle(f'S{subject} | Experience-based candidate and rule-based baseline', fontsize=11)
    actual = trailing(data['actual_feedback'])
    for model, color, label in [('model_1031', NEW, 'Model 1031'), ('model_1024', OLD, 'Model 1024')]:
        axs[0].plot(trial, trailing(data[model + '_conditional_correct']), color=color,
                    ls='--', label=label)
        ensemble = trailing(data[model + '_generated'])
        lower, upper = np.full(n, np.nan), np.full(n, np.nan)
        lower[31:], upper[31:] = np.quantile(ensemble[:, 31:], [.1, .9], axis=0)
        axs[1].fill_between(trial, lower, upper, color=color, alpha=.12, linewidth=0)
        axs[1].plot(trial, ensemble.mean(axis=0), color=color, ls='--')
    for ax in axs[:2]:
        ax.plot(trial, actual, color=HUMAN, lw=1.7, label='Observed accuracy')
        ax.set_ylim(.25, 1.02); ax.set_yticks([.25, .5, .75, 1.]); ax.set_ylabel('Accuracy / probability')
    panel(axs[0], 'a', 'Choice prediction given observed history | trailing 32 trials')
    axs[0].legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0)
    # Give the first legend its own space beneath the panel title.
    axs[0].set_title('Choice prediction given observed history | trailing 32 trials', loc='left', pad=29, fontsize=10)
    panel(axs[1], 'b', 'Autonomous learning | 64 paths; shaded 10th–90th percentiles')
    for index, (name, color) in enumerate(zip(['Uncommitted', 'Specific example', 'Local summary', 'Catalogue rule'], GROUP_COLORS)):
        axs[2].plot(trial, trailing(data['model_1031_group'][:, index], 1), color=color, label=name)
    axs[2].set_ylim(-.02, 1.02); axs[2].set_ylabel('Execution probability')
    panel(axs[2], 'c', 'Model 1031: inferred representation | trailing 32 trials')
    axs[2].legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0)
    axs[2].set_title('Model 1031: inferred representation | trailing 32 trials', loc='left', pad=29, fontsize=10)
    axs[3].scatter(trial, data['oral_feature_count'], color=HUMAN, s=6, alpha=.25, linewidths=0)
    axs[3].plot(trial, trailing(data['oral_feature_count'], 8), color=HUMAN, label='Oral: named parts')
    for model, color, label in [('model_1031', NEW, '1031: expected features'), ('model_1024', OLD, '1024: expected features')]:
        axs[3].plot(trial, trailing(data[model + '_feature_count'], 1), color=color, label=label)
    axs[3].set_ylim(-.1, 4.3); axs[3].set_yticks([0, 1, 2, 3, 4]); axs[3].set_ylabel('Number of features')
    panel(axs[3], 'd', 'Feature scope | oral codes and inferred execution content')
    axs[3].legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0)
    axs[3].set_title('Feature scope | oral codes and inferred execution content', loc='left', pad=29, fontsize=10)
    trial_axis(axs[3], n)
    save(fig, 'behavior_and_representation.png')

    fig, axs = plt.subplots(3, 1, figsize=(WIDTH, 7.2), sharex=True, layout='constrained',
                            gridspec_kw={'height_ratios': [1, 1, 1.2]})
    fig.suptitle(f'S{subject} | Which features are used?', fontsize=11)
    oral = data['oral_mask'].astype(float).T
    oral[:, ~data['oral_eligible']] = np.nan
    cmap = ListedColormap(['#EDEDED', '#303030']); cmap.set_bad('white')
    axs[0].imshow(oral, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest',
                  extent=(.5, n+.5, 3.5, -.5))
    panel(axs[0], 'a', 'Original oral codes | dark: mentioned; light: absent; white: unscored')
    image = axs[1].imshow(data['model_1031_feature_probability'].T, cmap='Blues', vmin=0, vmax=1,
        aspect='auto', interpolation='nearest', extent=(.5, n+.5, 3.5, -.5))
    panel(axs[1], 'b', 'Model 1031 | feature-use probability after choice, before feedback')
    for ax in axs[:2]:
        ax.set_yticks(range(4), [name.capitalize() for name in summary['feature_names']]); ax.grid(False)
    fig.colorbar(image, ax=axs[1], location='right', shrink=.8, ticks=[0, .5, 1], pad=.02)
    for model, color, label in [('model_1031', NEW, 'Model 1031'), ('model_1024', OLD, 'Model 1024')]:
        axs[2].plot(trial, trailing(data[model + '_feature_exact'], 8), color=color, label=label)
    panel(axs[2], 'c', 'Exact feature-set correspondence | trailing 32 trials, ≥8 scored reports')
    axs[2].set_ylim(-.02, 1.02); axs[2].set_ylabel('Correspondence probability'); axs[2].legend(loc='upper center', ncol=2)
    trial_axis(axs[2], n)
    save(fig, 'feature_use.png')

    fig, axs = plt.subplots(3, 1, figsize=(WIDTH, 7.8), layout='constrained')
    fig.suptitle(f'S{subject} | Final candidates and numerical sensitivity', fontsize=11)
    positions = np.arange(len(points))
    for x, row in enumerate(points.to_dict('records')):
        color = NEW if row['model'] == 'model_1031' else OLD
        axs[0].plot([x, x], [row['nll_min'], row['nll_max']], color=color, lw=1.)
        axs[0].scatter([x, x], [row['nll_min'], row['nll_max']], color=color, s=20, marker='o')
        axs[0].scatter(x, row['nll'], color=color, s=26, marker='_')
        axs[1].errorbar(x, row['energy'], yerr=row['energy_mc_se'], fmt='o', ms=4, color=color, capsize=3)
    labels = [r['model'].replace('model_', '') + ' / ' + r['candidate'] + (' *' if r['selected'] else '')
              for r in points.to_dict('records')]
    for ax in axs[:2]:
        ax.set_xticks(positions, labels); ax.set_xlim(-.5, len(points)-.5)
    axs[0].set_ylabel('Choice NLL'); panel(axs[0], 'a', 'Two runs at 512 particles | lower is better; * main candidate')
    axs[1].set_ylabel('Energy score'); panel(axs[1], 'b', 'All 64 autonomous paths | lower is better; ± simulation standard error')
    group = data['model_1031_groups_by_repeat'][:, :, 3]
    repeats = trailing(group, 1)
    axs[2].fill_between(trial, repeats.min(axis=0), repeats.max(axis=0), color=NEW, alpha=.2, linewidth=0)
    axs[2].plot(trial, repeats.mean(axis=0), color=NEW, label='1031 / ' + summary['selected']['model_1031'])
    axs[2].plot(trial, trailing(data['model_1031_alternative_group'][:, 3], 1), color=OLD, ls='--', label='1031 / ' + alternative)
    axs[2].set_ylabel('Rule execution probability'); axs[2].set_ylim(-.02, 1.02)
    panel(axs[2], 'c', '1031 rule execution across finalists | band: two numerical repeats')
    axs[2].legend(loc='lower right'); trial_axis(axs[2], n)
    save(fig, 'candidate_sensitivity.png')
    diagnostic_path = report / 'rule_diagnostics.npz'
    if diagnostic_path.exists():
        with np.load(diagnostic_path, allow_pickle=False) as z:
            rules = {key: z[key] for key in z.files}
        rule_meta = json.loads((report / 'rule_diagnostics.json').read_text())
        fig, axs = plt.subplots(3, 1, figsize=(WIDTH, 8.3), sharex=True, layout='constrained')
        fig.suptitle(f'S{subject} | Wrong comparison and correct rule', fontsize=11)
        axs[0].plot(trial, trailing(rules['oral_head_neck_scope'], 8), color=HUMAN,
                    label='Head + neck only')
        axs[0].plot(trial, trailing(rules['oral_neck_scope'], 8), color='#8C63AB',
                    label='Neck only')
        panel(axs[0], 'a', 'Original oral feature scope | trailing 32 trials')
        axs[0].set_ylabel('Fraction of scored reports')
        axs[0].legend(loc='upper center', ncol=2)
        for point in rule_meta['points']:
            color = NEW if point['model'] == 'model_1031' else OLD
            label = point['model'].replace('model_', '') + ' / ' + point['candidate']
            label += ' (main)' if point['selected'] else ' (alternative)'
            for ax, target in zip(axs[1:], ['wrong', 'correct']):
                repeats = trailing(rules[point['key'] + '_' + target], 1)
                if point['selected']:
                    ax.fill_between(trial, repeats.min(axis=0), repeats.max(axis=0),
                                    color=color, alpha=.13, linewidth=0)
                ax.plot(trial, repeats.mean(axis=0), color=color,
                        ls='-' if point['selected'] else '--', label=label)
        panel(axs[1], 'b', 'Wrong rule | head > neck: option 1; head < neck: option 2')
        panel(axs[2], 'c', 'Correct rule | short neck: option 1; long neck: option 2')
        for ax in axs:
            ax.set_ylim(-.02, 1.02)
            ax.axvspan(89, 115, color='#888888', alpha=.08, linewidth=0)
            ax.axvspan(321, 384, color='#888888', alpha=.08, linewidth=0)
        for ax in axs[1:]:
            ax.set_ylabel('Execution probability')
        axs[1].legend(ncol=2, loc='upper center', fontsize=7)
        trial_axis(axs[2], n)
        save(fig, 'wrong_and_correct_rules.png')
    qa = dict(backend='Python / matplotlib', format='PNG only per repository instructions',
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), figures=figures,
        claim='Compare behavior and the inferred shift from specific experience to rules in the selected case.',
        subject=subject, n_participants=1, trials=n, numerical_repeats=2, autonomous_paths_per_parameter=64,
        data_use='complete_behavior_for_parameters; oral_selected_development_case',
        intervals='pathwise pointwise 10–90%; numerical repeat range; Energy simulation standard error',
        exclusions=f"{int((~data['oral_eligible']).sum())} oral records without explicit parts unscored; all behavior retained",
        smoothing='trailing32; minimum32 behavior, minimum1 state, minimum8 scored oral',
        tests='No hypothesis tests or p values', width_mm=183, dpi=300,
        visual_review='pending')
    if diagnostic_path.exists():
        qa['rule_input_sha256'] = hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
        qa['rule_intervals'] = 'main candidate numerical repeat range; alternative candidates shown individually'
    (output / 'figure_provenance.json').write_text(json.dumps(qa, indent=2, ensure_ascii=False))
    print(json.dumps(dict(output=str(output), figures=figures)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.report, args.output)
