"""Plot audited full-path distributions and parameter sensitivity (PNG only).

Run --analysis RESULTS --output NEW. All observations and paths are retained.
These are development diagnostics; intervals describe Monte Carlo variability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import digest, read

COLORS = dict(MIX='#4477AA', MAP='#777777', STICKY='#CC8844')
SHAPES = ['#999999', '#4477AA', '#AA6699']
MODES = ['MIX', 'MAP', 'STICKY']


def style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / (name + '.png'), dpi=300, facecolor='white')
    plt.close(fig)


def overview(paths: pd.DataFrame, observed: pd.DataFrame, output: Path) -> None:
    old = paths[paths.batch == 'original']
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8), sharey=True)
    rng = np.random.default_rng(20261025)
    for ax, (sid, obs) in zip(axes.flat, observed.set_index('subject').sort_index().iterrows()):
        for i, mode in enumerate(MODES):
            values = old[(old.subject == sid) & (old['mode'] == mode)].late_accuracy.to_numpy()
            ax.scatter(i+rng.uniform(-.19, .19, len(values)), values, s=6, alpha=.25,
                       color=COLORS[mode], linewidths=0)
            low, median, high = np.quantile(values, [.05, .5, .95])
            ax.plot([i, i], [low, high], color=COLORS[mode], lw=2)
            ax.scatter([i], [median], color=COLORS[mode], marker='D', s=22)
        ax.axhline(obs.late_accuracy, color='black', lw=1.3)
        ax.set(title=f'S{sid} · observed {obs.late_accuracy:.2f}', xticks=range(3),
               xticklabels=MODES, ylim=(-.03, 1.03), xlim=(-.5, 2.5))
        ax.set_yticks([0, .5, 1])
    for ax in axes[:, 0]:
        ax.set_ylabel('Suffix accuracy')
    fig.suptitle('Observed suffix performance within complete autonomous paths', y=.98, fontsize=11)
    fig.text(.5, .018, '128 full paths per mechanism and person · diamonds: median; bars: 5th–95th percentiles\n'
             'Black line: observed suffix accuracy · all nine people are development cases', ha='center', fontsize=7)
    fig.subplots_adjust(left=.09, right=.98, bottom=.12, top=.90, hspace=.4, wspace=.2)
    save(fig, output, 'nine_subject_path_distribution')


def focus(analysis: Path, clusters: pd.DataFrame, subject: int, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), gridspec_kw={'width_ratios': [1.9, 1]})
    for row, mode in enumerate(['MIX', 'STICKY']):
        with np.load(analysis/'curves'/f'S{subject}_{mode}_independent_nominee.npz') as data:
            curves, obs, labels = data['curves'], data['observed'], data['labels']
            representatives, end, cut = data['representatives'], data['end_trial'], int(data['cut'])
        left, right = axes[row]
        for k, index in enumerate(representatives):
            if index >= 0:
                left.plot(end, curves[index], color=SHAPES[k], lw=1.1, ls=[':', '--', '-.'][k],
                          label=f'Shape {k+1} ({(labels == k).mean():.0%})')
        left.plot(end, obs, color='black', lw=1.6, label='Observed')
        left.axvline(cut, color='#bbbbbb', lw=.8)
        left.set(xlabel='Trial (non-overlapping 32-trial blocks)', ylabel='Accuracy', ylim=(-.03, 1.03))
        left.set_title(f'{mode}: complete representative paths', y=1.23, fontsize=8)
        left.legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 1.01), fontsize=6)
        own = clusters[(clusters.subject == subject) & (clusters['mode'] == mode) &
                       (clusters.role == 'nominee') & (clusters.clusters == 3)]
        for offset, batch, split, marker, label in [(-.1, 'original', 'evaluation', 'o', 'Old check (64)'),
                                                   (.1, 'independent', 'all', 's', 'New paths (128)')]:
            table = own[(own.batch == batch) & (own.split == split)].sort_values('cluster')
            y = table.probability.to_numpy()
            right.errorbar(np.arange(1, 4)+offset, y,
                yerr=np.stack([y-table.low, table.high-y]), marker=marker, ls='none',
                color='#777777' if batch == 'original' else COLORS[mode], capsize=2, label=label, ms=4)
        selected = own[(own.batch == 'independent') & (own.split == 'all') &
                       (own.cluster == own.observed_cluster)].iloc[0]
        right.set(xlim=(.6, 3.4), ylim=(-.03, 1.03), xticks=[1, 2, 3], xlabel='Shape', ylabel='Probability')
        right.legend(loc='upper right', fontsize=6)
        right.set_title('Frozen shape probabilities\n'
                        f'Observed: nearest shape {int(selected.observed_cluster)+1}\n'
                        f'Within-shape distance rank: {selected.observed_distance_percentile:.0%}', fontsize=7)
    fig.suptitle(f'S{subject}: path shapes and their probability mass', y=.98, fontsize=11)
    fig.text(.5, .012, 'Representatives are new-batch cluster medoids; each line is one complete path.\n'
             'Centers were fitted on old simulation only. Bars: 95% Wilson Monte Carlo intervals.\n'
             'Distance percentiles use pooled old check paths; nearest shape is a descriptive assignment.',
             ha='center', fontsize=6.5)
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.17, hspace=.72, wspace=.36)
    save(fig, output, f'subject_{subject}_path_shapes')


def sensitivity(paths: pd.DataFrame, output: Path) -> None:
    data = paths[paths.batch == 'independent']
    metrics = [('late_target_present', 'Correct rule present', '#999999', 'o'),
               ('late_target_contribution', 'Correct rule contribution', '#4477AA', 's'),
               ('late_accuracy', 'Suffix accuracy', '#CC8844', '^')]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.8), sharey=True)
    for row, sid in enumerate([129, 229, 317]):
        for col, mode in enumerate(['MIX', 'STICKY']):
            ax = axes[row, col]
            own = data[(data.subject == sid) & (data['mode'] == mode)]
            roles = ['nominee', 'alternative_1', 'alternative_2']
            for offset, (metric, label, color, marker) in zip([-.15, 0, .15], metrics):
                groups = [own[own.role == role][metric] for role in roles]
                means = np.array([g.mean() for g in groups])
                errors = np.array([1.96*g.std(ddof=1)/np.sqrt(len(g)) for g in groups])
                ax.errorbar(np.arange(3)+offset, means, yerr=errors, marker=marker, color=color,
                            ls='none', capsize=2, ms=4, label=label)
            ax.set(title=f'S{sid} · {mode}', xticks=range(3), xticklabels=['Original', 'Alternative 1', 'Alternative 2'],
                   ylim=(-.03, 1.03), xlim=(-.5, 2.5), yticks=[0, .5, 1])
            if col == 0:
                ax.set_ylabel('Mean over autonomous paths')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False, fontsize=7)
    fig.suptitle('Internal-process sensitivity across training-near parameters', y=.99, fontsize=11)
    fig.text(.5, .018, '128 independent paths per point · bars: ±1.96 Monte Carlo standard errors\n'
             'Alternatives selected from training NLL and parameter distance; points have no posterior weights.',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=.09, right=.98, bottom=.12, top=.85, hspace=.45, wspace=.2)
    save(fig, output, 'parameter_process_sensitivity')


def run(analysis: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure destination')
    complete = read(analysis/'completion.json')
    if complete['status'] != 'complete':
        raise ValueError('Analysis is incomplete')
    for filename, sha in complete['artifacts_sha256'].items():
        if digest(analysis/filename) != sha:
            raise ValueError('Analysis source changed: ' + filename)
    output.mkdir(parents=True)
    style()
    paths, observed, clusters = [pd.read_csv(analysis/f'{k}.csv') for k in ('paths', 'observed', 'clusters')]
    overview(paths, observed, output)
    for sid in (129, 229, 317):
        focus(analysis, clusters, sid, output)
    sensitivity(paths, output)
    (output/'provenance.json').write_text(json.dumps(dict(analysis=str(analysis.resolve()),
        completion_sha256=digest(analysis/'completion.json'), source_sha256=digest(Path(__file__)),
        backend='python', export='PNG only by repository policy', figure_count=5,
        paths=complete['original_paths']+complete['independent_paths'],
        interval_scope='process spread or simulation uncertainty as labelled; development cases'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.analysis, args.output)
