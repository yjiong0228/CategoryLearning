"""Plot known and inferred target-rule states for each synthetic sequence.

Figure contract: compare internal-state recovery at generating and fitted
parameters, alongside their choice NLL. A quantitative grid uses columns for
parameter points and rows for membership, choice contribution and precision.
Every trial and declared parameter group is retained. Shading is the empirical
5--95% latent-state interval, conditional on one parameter point; it excludes
parameter uncertainty. Precision conditions on rule membership. The figures
support descriptive development checks, not population calibration.

Use --source COMPLETED_SUMMARY --output NEW_DIRECTORY. Python/matplotlib, white
background, 183 × 165 mm, PNG only at 300 dpi; exact source CSVs accompany plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_target_state_recovery import digest, read, verify, write


LABELS = {'truth': 'Generating parameters', 'fit': '40-start estimate', 'expanded_fit': '80-start estimate'}


def render(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new target-state figure directory')
    done = read(source/'completion.json')
    if done['status'] != 'complete' or done['smoke']:
        raise ValueError('Require full target-state summaries')
    verify(done['inputs_sha256'])
    for name, sha in done['artifacts_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError('Target-state summary changed')
    trials, scores = [pd.read_csv(source/name) for name in ('trials.csv', 'scores.csv')]
    output.mkdir(parents=True)
    for name in ('trials.csv', 'scores.csv', 'filter_scores.csv'):
        shutil.copyfile(source/name, output/name)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':7, 'axes.titlesize':8, 'axes.labelsize':7, 'legend.fontsize':7,
        'svg.fonttype':'none', 'pdf.fonttype':42,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65,
        'xtick.labelsize':6.5, 'ytick.labelsize':6.5, 'figure.facecolor':'white'})
    color = '#397A9D'
    files = []
    for replicate, group in trials[trials.phase.eq('filtered')].groupby('replicate', sort=True):
        labels = list(dict.fromkeys(group.label))
        fig, axes = plt.subplots(3, len(labels), figsize=(183/25.4, 165/25.4), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.09, right=.98, bottom=.14, top=.82, hspace=.20, wspace=.18)
        fig.suptitle(f'Synthetic sequence {replicate+1}: target-rule state recovery', y=.98, fontsize=10)
        fig.legend(handles=[Line2D([0], [0], color='black', lw=1.2, ls='--', label='Known state'),
            Line2D([0], [0], color=color, lw=1.2, label='Posterior mean'),
            Patch(facecolor=color, alpha=.18, label='90% latent-state interval')],
            loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
        for column, label in enumerate(labels):
            data = group[group.label.eq(label)].sort_values('trial')
            score = scores[scores.label.eq(label) & scores.phase.eq('filtered')].iloc[0]
            if len(data) != score.trials or not np.array_equal(data.trial, np.arange(1, len(data)+1)):
                raise ValueError('Missing or reordered plotted trials')
            x = data.trial.to_numpy()
            ax = axes[0, column]
            ax.set_title(f"{LABELS[label.split('_', 1)[1]]}\nChoice NLL = {score.choice_nll:.4f}", pad=10)
            ax.step(x, data.truth_present, where='mid', color='black', lw=.9, ls='--', zorder=3)
            ax.plot(x, data.presence_probability, color=color, lw=1.1)
            for row, prefix, truth_key in ((1, 'contribution', 'truth_contribution'),
                                            (2, 'conditional_beta', 'truth_beta')):
                ax = axes[row, column]
                known = data[truth_key].to_numpy().copy()
                if row == 2:
                    known[~data.truth_present.astype(bool)] = np.nan
                ax.fill_between(x, data[prefix+'_q05'].to_numpy(), data[prefix+'_q95'].to_numpy(),
                    color=color, alpha=.18, lw=0)
                ax.plot(x, data[prefix+'_mean'], color=color, lw=1.1)
                ax.plot(x, known, color='black', lw=1, ls='--', zorder=3)
            for row in (0, 1):
                axes[row, column].set_ylim(-.04, 1.04)
                axes[row, column].set_yticks([0, .5, 1])
            axes[2, column].set_ylim(0, 25.5)
            axes[2, column].set_yticks([0, 10, 20, 25])
            axes[2, column].set_xlabel('Trial')
            for row in range(3):
                axes[row, column].set_xlim(1, len(data))
                axes[row, column].set_xticks([1, 32, 64, 96, 128])
                if column:
                    axes[row, column].tick_params(labelleft=False)
        for ax, label in zip(axes[:, 0], ['Target membership', 'Choice contribution', 'Precision given presence']):
            ax.set_ylabel(label)
        fig.text(.5, .07, 'Estimates use all 128 trials; state after current choice and before feedback.',
            ha='center', fontsize=6.6)
        fig.text(.5, .045, 'Each point: 8 filters × 512 particles. Intervals omit parameter uncertainty.',
            ha='center', fontsize=6.6)
        name = f'sequence_{replicate+1}.png'
        fig.savefig(output/name, dpi=300, facecolor='white')
        plt.close(fig)
        files.append(name)
    write(output/'completion.json', dict(status='complete', backend='python', source=str(source.resolve()),
        summary_sha256=digest(source/'completion.json'), script_sha256=digest(Path(__file__)),
        figures={name:digest(output/name) for name in files},
        source_data={name:digest(output/name) for name in ('trials.csv','scores.csv','filter_scores.csv')},
        plotted_phase='filtered', trial_selection='all trials and all groups; predictive summaries retained in source CSV',
        dimensions_mm=[183,165], dpi=300, format='PNG'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
