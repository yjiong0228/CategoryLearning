"""Render all paired selection outcomes and fixed focus cases (PNG only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import digest, read

COLORS = dict(MIX='#4477AA', MAP='#777777', STICKY='#CC8844')


def run(summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure destination')
    complete = read(summary/'completion.json')
    if complete['status'] != 'complete':
        raise ValueError('Require completed audited analysis')
    for name, sha in complete['artifacts_sha256'].items():
        if digest(summary/name) != sha:
            raise ValueError('Figure input changed: '+name)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    output.mkdir(parents=True)
    changes = pd.read_csv(summary/'comparisons.csv')
    subjects = sorted(changes.subject.unique())
    modes = ['MIX', 'MAP', 'STICKY']
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), sharey=True)
    quantities = [('test_nll', 'Suffix NLL (lower is better)'), ('late_energy', 'Suffix Energy (lower is better)'),
                  ('full_energy', 'Full-path Energy (lower is better)'), ('late_accuracy', 'Suffix accuracy (higher is better)')]
    for panel, (ax, (metric, title)) in enumerate(zip(axes.flat, quantities)):
        for shift, mode in zip([-.17, 0, .17], modes):
            own = changes[changes['mode'] == mode].set_index('subject').loc[subjects]
            values, errors = own[metric+'_delta'].to_numpy(), own[metric+'_delta_mcse'].to_numpy()*2
            ax.errorbar(values, np.arange(len(subjects))+shift, xerr=errors, color=COLORS[mode],
                        marker=['o', 's', '^'][modes.index(mode)], ls='none', ms=4, capsize=2, label=mode)
        ax.axvline(0, color='#aaaaaa', lw=.8)
        ax.set_title(title, fontsize=8)
        ax.set(yticks=range(len(subjects)), yticklabels=[f'S{s}' for s in subjects], xlabel='New estimate − original NLL estimate')
        ax.text(-.16, 1.03, chr(97+panel), fontweight='bold', transform=ax.transAxes)
        ax.invert_yaxis() if panel == 0 else None
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .956), ncol=3, fontsize=8)
    fig.suptitle('Frozen evaluation after training-distribution selection', y=.99, fontsize=11)
    fig.text(.5, .018, 'All 27 development comparisons · paired seeds, equal budgets\n'
             'Bars: ±2 Monte Carlo standard errors · unchanged parameter estimates remain at zero', ha='center', fontsize=7)
    fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.88, hspace=.35, wspace=.30)
    fig.savefig(output/'selection_outcomes.png', dpi=300, facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8), sharey=True)
    for row, sid in enumerate([129, 229, 317]):
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            for role, color, ls in [('NLL', '#4477AA', '--'), ('JOINT', '#CC8844', '-')]:
                with np.load(summary/'curves'/f'S{sid}_{mode}_{role}.npz') as z:
                    curves, observed, end, cut = z['curves'], z['observed'], z['end_trial'], int(z['cut'])
                if role == 'JOINT':
                    low, high = np.quantile(curves, [.05, .95], axis=0)
                    ax.fill_between(end, low, high, color=color, alpha=.15, lw=0)
                ax.plot(end, curves.mean(axis=0), color=color, ls=ls, lw=1.2, label=role)
            ax.plot(end, observed, color='black', lw=1.3, label='Observed')
            ax.axvline(cut, color='#aaaaaa', lw=.7)
            ax.set(title=f'S{sid} · {mode}', ylim=(-.03, 1.03), yticks=[0, .5, 1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            if col == 0:
                ax.set_ylabel('Block accuracy')
            if row == 2:
                ax.set_xlabel('Trial')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .952), ncol=3, fontsize=8)
    fig.suptitle('Whole-task generation at frozen parameter estimates', y=.99, fontsize=11)
    fig.text(.5, .018, '128 complete paths per estimate · model lines: ensemble means\n'
             'Shading: JOINT pointwise 5th–95th percentiles · blocks: 32 trials · vertical line: training end\n'
             'Identical parameter estimates have overlapping mean curves',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=.09, right=.98, bottom=.12, top=.87, hspace=.35, wspace=.2)
    fig.savefig(output/'selection_focus_paths.png', dpi=300, facecolor='white')
    plt.close(fig)
    (output/'provenance.json').write_text(json.dumps(dict(summary=str(summary.resolve()),
        summary_sha256=digest(summary/'completion.json'), source_sha256=digest(Path(__file__)),
        backend='python', export='PNG only', panels='27 paired comparisons; fixed three-person focus',
        interval_scope='Monte Carlo standard errors or pointwise process percentiles as labelled'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.summary, args.output)
