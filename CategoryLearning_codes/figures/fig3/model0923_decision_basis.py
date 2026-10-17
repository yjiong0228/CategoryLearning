"""Plot oral content coverage, rule use and behavior of frozen 0923 fits.

Run from the repository root with --alignment ALIGNED --behavior SUMMARY
--output NEW_DIRECTORY. Python/Matplotlib exports PNG discussion figures only.
Every declared participant and arm is retained; unparsed reports are shown as
coverage loss, and unresolved complete content remains in the scored denominator.
Error bars are two Monte Carlo standard errors, not participant confidence limits.
The established episodic-refit palette is reused as a style reference only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ORDER = ['BASE', 'NEW', 'POOL', 'DRAW']
COLORS = {'BASE': '#68778B', 'NEW': '#78A7AD', 'POOL': '#277F8E', 'DRAW': '#A16F99'}
MARKERS = {'BASE': 'o', 'NEW': 's', 'POOL': 'D', 'DRAW': '^'}
WIDTH_MM = 183


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finish(ax, label: str, title: str, *, legend_above: bool = False) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.12, 1.22 if legend_above else 1.08, label, transform=ax.transAxes, weight='bold', fontsize=8)
    ax.set_title(title, loc='left', pad=28 if legend_above else 9)
    ax.tick_params(length=2.5, width=.6)


def plot(alignment: Path, behavior: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure destination')
    manifest = json.loads((alignment / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['smoke']:
        raise ValueError('Need completed scientific alignment')
    for name in ('summary.csv', 'contrasts.csv'):
        if digest(alignment / name) != manifest['artifacts'][name]:
            raise ValueError('Alignment source changed')
    subjects = sorted(manifest['subjects'])
    summary = pd.read_csv(alignment / 'summary.csv')
    full = summary.loc[summary.window.eq('test') & summary.domain.eq('structural') & summary.subset.eq('complete')].copy()
    represented = summary.loc[summary.window.eq('test') & summary.domain.eq('structural') & summary.subset.eq('catalogue_supported')]
    contrasts = pd.read_csv(alignment / 'contrasts.csv')
    contrasts = contrasts.loc[contrasts.window.eq('test') & contrasts.domain.eq('structural') & contrasts.subset.eq('complete')]
    scores = pd.read_csv(behavior / 'scores.csv')
    if len(full) != len(subjects) * 4 or set(full.arm) != set(ORDER):
        raise ValueError('Need all four fitted arms per declared participant')
    if full.duplicated(['subject', 'arm']).any():
        raise ValueError('Ambiguous alignment summary')
    base = full.loc[full.arm.eq('BASE')].set_index('subject').loc[subjects]
    available = represented.loc[represented.arm.eq('BASE')].set_index('subject').loc[subjects]
    data = []
    for subject in subjects:
        for arm in ORDER:
            row = full.loc[full.subject.eq(subject) & full.arm.eq(arm)].iloc[0].to_dict()
            for phase, metric in [('test', 'choice_nll'), ('whole', 'energy')]:
                values = scores.loc[scores.subject.eq(subject) & scores.phase.eq(phase)].set_index('model')[metric]
                row[metric] = values.loc[arm]
                row[f'{metric}_difference'] = values.loc[arm] - values.loc['BASE']
            row['support_difference'] = row['support'] - base.loc[subject, 'support']
            row['support_difference_mcse'] = (np.nan if arm == 'BASE' else
                contrasts.loc[contrasts.subject.eq(subject) & contrasts.arm.eq(arm), 'support_difference_mcse'].item())
            row['catalogue_supported_reports'] = int(available.loc[subject, 'reports'])
            data.append(row)
    data = pd.DataFrame(data)
    output.mkdir(parents=True)
    data.to_csv(output / 'plot_source.csv', index=False)
    contract = dict(
        claim='Content coverage, behavior-conditioned rule use and autonomous learning provide distinct evidence about a fitted mechanism.',
        archetype='cohort diagnosis plus paired mechanism tradeoffs', backend='python',
        panels={'overview_a': 'complete encoding and catalogue-supported report counts',
                'overview_b': 'BASE support/conflict/undetermined mass over all complete reports',
                'overview_c': 'paired arm-minus-BASE whole-report support and filter MCSE',
                'overview_d': 'availability bounds and actual decision-basis support, BASE and NEW',
                'tradeoffs_a': 'later-trial choice NLL difference versus support difference',
                'tradeoffs_b': 'autonomous whole-curve Energy difference versus support difference'},
        subjects=subjects, arms=ORDER, sample_role='development', width_mm=WIDTH_MM,
        overview_height_mm=160, tradeoffs_height_mm=83, dpi=300, formats=['PNG'],
        statistics={'n': 'participants shown separately; eight independent filters per arm',
                    'center': 'mean over each participant\'s complete later-trial reports',
                    'error_bars': 'plus/minus two delete-one-filter jackknife Monte Carlo SE',
                    'tests': 'none', 'multiple_comparison_correction': 'not applicable',
                    'parameter_and_participant_uncertainty': 'not included'},
        data_boundary='parameters estimated from training choices; later reports used for development evaluation',
        limitations=['automatic complete parsing requires semantic review',
                     'unresolved body references remain undetermined',
                     'content support is not a probability of producing an utterance',
                     'availability uses posterior states conditional on actual choices',
                     'Energy uses fixed observed stimulus schedules',
                     'subject labels in scatter panels emphasize the predeclared 207/208/308 process probes'],
        style_reference='CategoryLearning_codes/figures/fig3/model0923_episodic_refit.py; palette only',
        source_preflight_exceptions={'EXPORT-VECTOR': 'Repository standing instructions require PNG only.',
                                     'EXPORT-RASTER': 'TIFF omitted under the same PNG-only instruction.',
                                     'RASTER-DPI': '300 dpi discussion figures; submission export is outside this task.'},
        source_sha256={str(p.resolve()): digest(p) for p in
                       [alignment / 'manifest.json', alignment / 'summary.csv', alignment / 'contrasts.csv', behavior / 'scores.csv']})
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7, 'axes.titlesize': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'legend.fontsize': 6, 'axes.linewidth': .6,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, 160 / 25.4))
    fig.subplots_adjust(left=.10, right=.97, bottom=.08, top=.86, wspace=.42, hspace=.65)
    y = np.arange(len(subjects))
    ax = axes[0, 0]
    ax.barh(y, base.reports / base.window_trials, color='#DBDFE4', height=.62)
    ax.barh(y, available.reports / base.window_trials, color='#68778B', height=.62)
    for i, subject in enumerate(subjects):
        ax.text(1.03, i, f'{int(available.loc[subject, "reports"])}/{int(base.loc[subject, "reports"])}', va='center', fontsize=6)
    ax.set(yticks=y, yticklabels=subjects, xlim=(0, 1.24), xticks=[0, .5, 1], xlabel='Fraction of later trials')
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color='#DBDFE4', label='Complete encoding'), Patch(color='#68778B', label='Catalogue supports report')],
              frameon=False, loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0)
    finish(ax, 'a', 'Report coverage; supported / complete counts', legend_above=True)
    ax = axes[0, 1]
    left = np.zeros(len(subjects))
    for label, color in [('support', '#277F8E'), ('conflict', '#BE7867'), ('undetermined', '#DBDFE4')]:
        value = base[label].to_numpy()
        ax.barh(y, value, left=left, color=color, height=.62, label=label.capitalize())
        left += value
    ax.set(yticks=y, yticklabels=subjects, xlim=(0, 1), xlabel='Mean content mass; BASE')
    ax.invert_yaxis(); ax.legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.01), columnspacing=.8, borderaxespad=0)
    finish(ax, 'b', 'Complete reports retain unresolved content', legend_above=True)
    ax = axes[1, 0]
    for offset, arm in zip([-.20, 0, .20], ORDER[1:]):
        rows = data.loc[data.arm.eq(arm)].set_index('subject').loc[subjects]
        ax.errorbar(rows.support_difference, y + offset, xerr=2 * rows.support_difference_mcse,
                    fmt=MARKERS[arm], color=COLORS[arm], markersize=3.2, elinewidth=.7, capsize=1.5, label=arm)
    ax.axvline(0, color='#9A9A9A', linewidth=.6)
    ax.set(yticks=y, yticklabels=subjects, xlabel='Support difference from BASE')
    ax.invert_yaxis(); ax.legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0)
    finish(ax, 'c', 'Mechanism differences; bars = 2 MCSE', legend_above=True)
    ax = axes[1, 1]
    ax.plot([0, 1], [0, 1], color='#B5B5B5', linewidth=.7, linestyle=':')
    for subject in subjects:
        pair = data.loc[data.subject.eq(subject) & data.arm.isin(['BASE', 'NEW'])].set_index('arm').loc[['BASE', 'NEW']]
        ax.plot(pair.availability_any_component, pair.support, color='#CFD2D6', linewidth=.7, zorder=1)
    for arm in ['BASE', 'NEW']:
        rows = data.loc[data.arm.eq(arm)]
        lower, upper = rows.availability_all_components, rows.availability_any_component
        ax.hlines(rows.support, lower, upper, color=COLORS[arm], linewidth=.8)
        ax.scatter(upper, rows.support, s=18, marker=MARKERS[arm], color=COLORS[arm], label=arm)
        for row in rows.itertuples():
            if arm == 'NEW' and row.subject in (207, 208, 308):
                ax.annotate(str(row.subject), (row.availability_any_component, row.support),
                            xytext=(4, -8), textcoords='offset points', fontsize=6, color=COLORS[arm])
    ax.set(xlim=(-.02, 1.05), ylim=(-.02, 1.05), xlabel='Supporting-rule availability; all–any bound',
           ylabel='Decision-basis support')
    ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0)
    finish(ax, 'd', 'Available rules and their contribution', legend_above=True)
    fig.savefig(output / 'oral_decision_basis.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, 83 / 25.4))
    fig.subplots_adjust(left=.10, right=.97, bottom=.22, top=.82, wspace=.34)
    for ax, metric, title, letter in zip(axes, ['choice_nll', 'energy'],
                                       ['Later-trial prediction', 'Autonomous learning curves'], ['a', 'b']):
        ax.axhline(0, color='#AAAAAA', linewidth=.6); ax.axvline(0, color='#AAAAAA', linewidth=.6)
        for arm in ORDER[1:]:
            rows = data.loc[data.arm.eq(arm)]
            ax.scatter(rows[f'{metric}_difference'], rows.support_difference, marker=MARKERS[arm],
                       color=COLORS[arm], s=21, linewidths=.4, edgecolors='white', label=arm)
        ax.margins(x=.12, y=.18)
        for subject in (207, 208, 308):
            rows = data.loc[data.subject.eq(subject) & data.arm.ne('BASE')]
            if len(rows):
                ax.annotate(str(subject), (rows[f'{metric}_difference'].mean(), rows.support_difference.mean()),
                            xytext=(4, 7 if subject == 308 else -12), textcoords='offset points',
                            fontsize=6, color='#444444')
        ax.set(xlabel=f'{"Choice NLL" if metric == "choice_nll" else "Curve Energy"} difference from BASE (lower is better)',
               ylabel='Oral support difference (higher is better)')
        finish(ax, letter, title)
    fig.legend(handles=[Line2D([], [], color=COLORS[a], marker=MARKERS[a], linestyle='none', label=a) for a in ORDER[1:]],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, .99))
    fig.text(.5, .04, 'Development participants; paired fitted-model differences. Points do not include parameter uncertainty.',
             ha='center', fontsize=6)
    fig.savefig(output / 'oral_behavior_tradeoffs.png', dpi=300)
    plt.close(fig)
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', subjects=subjects,
        plotting_source_sha256=digest(Path(__file__)),
        artifacts={p.name: digest(p) for p in output.iterdir()}), indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--behavior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.alignment, args.behavior, args.output)


if __name__ == '__main__':
    main()
