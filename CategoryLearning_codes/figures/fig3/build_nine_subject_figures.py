"""Render nine-participant Fig3 and parameter-sensitivity companion (PNG only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .bottleneck_analysis import ROOT, sha256

TASK_COLORS = {1: '#487DA8', 2: '#A46A8A', 3: '#648B76'}
COLORS = {'available': '#92B8C6', 'belief': '#167D8D', 'executed': '#CC873C',
          'observed': '#333B43', 'predicted': '#617EA0', 'alternative': '#9B7199'}
SUBJECTS = [[102, 118, 122], [307, 314, 315], [206, 221, 222]]
ORDER = sum(SUBJECTS, [])


def style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': .6,
        'axes.spines.right': False, 'axes.spines.top': False,
        'legend.fontsize': 6, 'legend.frameon': False,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def read_tables(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((source / 'manifest.json').read_text())
    d = pd.read_csv(source / 'trials.csv')
    s = pd.read_csv(source / 'subjects.csv').set_index('subject').loc[ORDER]
    assert len(s) == 9 and len(d) == manifest['trial_count']
    return d, s, manifest


def title(fig, headline: str, subtitle: str) -> None:
    fig.text(.075, .977, headline, fontsize=10, weight='bold', va='top')
    fig.text(.075, .951, subtitle, fontsize=6.6, color='.4', va='top')


def label(ax, letter: str, heading: str) -> None:
    ax.set_title(heading, loc='left', pad=9, fontsize=7.5)
    ax.text(-.14, 1.07, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)


def lines_legend(fig, entries: list[tuple[str, str, str]], y: float, ncol: int) -> None:
    handles = [Line2D([], [], color=color, lw=1.2, ls=ls, label=name) for name, color, ls in entries]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.535, y),
               ncol=ncol, columnspacing=1.1, handlelength=1.7)


def mark_events(axes, row: pd.Series) -> None:
    for ax in axes:
        if np.isfinite(row.criterion):
            ax.axvline(row.criterion, color='.55', lw=.6, ls=':', zorder=0)


def finish(fig, output: Path, filename: str) -> None:
    fig.savefig(output / filename, dpi=450)
    plt.close(fig)


def source_receipt(source: Path, output: Path, figure_name: str, source_files: list[str]) -> None:
    (output / 'sources').mkdir()
    for name in source_files:
        shutil.copy2(source / name, output / 'sources' / name)
    code = Path(__file__).resolve()
    manifest = {'figure': figure_name, 'source_analysis': str(source.resolve().relative_to(ROOT)),
                'format': 'PNG', 'dpi': 450, 'width_mm': 183,
                'source_sha256': {name: sha256(source/name) for name in source_files},
                'plotting_source': str(code.relative_to(ROOT)), 'plotting_source_sha256': sha256(code)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def fig3(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    d, s, _ = read_tables(source)
    blocks = pd.read_csv(source / 'blocks64.csv')
    alternatives = pd.read_csv(source / 'candidate_sensitivity.csv').query("variant == 'alternative'").set_index('subject')
    style()
    fig = plt.figure(figsize=(183/25.4, 235/25.4))
    title(fig, 'Fig. 3 | Beliefs and performance follow different timelines',
          'Nine fitted participants  |  all recorded trials  |  target rule defined by the task')
    lines_legend(fig, [('Observed accuracy', COLORS['observed'], '-'),
                       ('Model P(correct)', COLORS['predicted'], '--'),
                       ('Considered: A', COLORS['available'], '-'),
                       ('Belief mass: Q', COLORS['belief'], '-'),
                       ('Executed: E', COLORS['executed'], '-')], .933, 5)
    outer = fig.add_gridspec(3, 3, left=.075, right=.975, bottom=.335, top=.873,
                            hspace=.50, wspace=.28)
    for col, subjects in enumerate(SUBJECTS):
        x = .075 + col * .323
        fig.text(x, .893, f'a{col+1}   Task {col+1}', color=TASK_COLORS[col+1], weight='bold', fontsize=8)
        for row, sid in enumerate(subjects):
            g = d.loc[d.iSub.eq(sid)]; info = s.loc[sid]
            sub = outer[row, col].subgridspec(2, 1, hspace=.07, height_ratios=[1, 1.15])
            top, bottom = [fig.add_subplot(sub[i]) for i in (0, 1)]
            mode = 'persistent' if info.chi else 'mixture'
            top.set_title(f'S{sid}  ·  {mode}', loc='left', pad=3)
            top.plot(g.trial, g.correct_w32, color=COLORS['observed'], lw=.8)
            top.plot(g.trial, g.predicted_w32, color=COLORS['predicted'], ls='--', lw=.8)
            top.axhline(.5 if info.task == 1 else .25, color='.85', lw=.6, zorder=0)
            for state in ('available', 'belief', 'executed'):
                if g[state].notna().any():
                    bottom.plot(g.trial, g[state], color=COLORS[state], lw=.75, zorder=3)
            # Current reports, never carried-forward reports. Gray = valid;
            # dark = target among the encoder's tied best full-space rules.
            valid = g.oral_valid.astype(bool)
            compatible = g.oral_target_top.astype(bool)
            bottom.vlines(g.trial[valid], -.105, -.075, color='.8', lw=.35)
            bottom.vlines(g.trial[compatible], -.11, -.06, color='.25', lw=.45)
            for ax in (top, bottom):
                ax.set(xlim=(1, len(g)), yticks=[0, 1])
                ax.tick_params(axis='both', pad=2)
            top.set(ylim=(-.03, 1.06), xticks=[])
            bottom.set(ylim=(-.13, 1.05), xticks=[1, int(len(g)/2), len(g)])
            bottom.set_xlabel('Trial', labelpad=1)
            if col == 0:
                top.set_ylabel('Accuracy', labelpad=3)
                bottom.set_ylabel('Rule state', labelpad=3)
            mark_events([top, bottom], info)
    fig.text(.075, .303, 'Accuracy: trailing 32 trials. Rule states: unsmoothed pre-choice marginals. Dotted line: first behavioral criterion.',
             fontsize=6, color='.35')
    fig.text(.075, .290, 'Report ticks: current report available (gray); target among tied best encoded rules (dark). Ties do not identify a unique rule.',
             fontsize=6, color='.35')
    ax = fig.add_axes([.105, .09, .43, .155])
    label(ax, 'b', 'Temporal landmarks for each learner')
    for i, sid in enumerate(ORDER):
        info = s.loc[sid]
        ax.plot([0, info.n], [i, i], color='.9', lw=2.2, zorder=0)
        alt_event = alternatives.loc[sid, 'belief_event']
        if np.isfinite(alt_event):
            ax.plot([info.belief_event, alt_event], [i,i], color=COLORS['alternative'], lw=.7, zorder=1)
            ax.scatter(alt_event, i, s=25, facecolors='white', edgecolors=COLORS['alternative'],
                       linewidths=.8, zorder=2)
        for state, marker, color in [('available_event', '^', COLORS['available']),
                                      ('belief_event', 'o', COLORS['belief']),
                                      ('criterion', 'D', COLORS['observed'])]:
            if np.isfinite(info[state]):
                ax.scatter(info[state], i, s=15, marker=marker, color=color, zorder=3, linewidths=.3)
    ax.set(yticks=range(9), yticklabels=[f'S{x}' for x in ORDER], ylim=(8.6, -.6),
           xlim=(-20, 1450), xticks=[0, 400, 800, 1200], xlabel='Recorded trial')
    for tick, sid in zip(ax.get_yticklabels(), ORDER):
        tick.set_color(TASK_COLORS[int(s.loc[sid].task)])
    ax.legend(handles=[Line2D([], [], ls='', marker=m, markersize=4, color=c, label=n)
                       for n, m, c in [('A > .5', '^', COLORS['available']),
                                        ('Q > .5', 'o', COLORS['belief']),
                                        ('Accuracy > .9', 'D', COLORS['observed'])]] +
              [Line2D([],[],ls='',marker='o',markersize=4,markerfacecolor='white',
                      color=COLORS['alternative'],label='Q: near candidate')],
              ncol=4, loc='upper left', bbox_to_anchor=(-.09, -.27), columnspacing=.6, handletextpad=.3,
              fontsize=5.5)
    ax = fig.add_axes([.64, .09, .325, .155])
    label(ax, 'c', 'Similar accuracy, different target belief')
    ax.axvspan(.6, .8, color='.95', zorder=0)
    for task, g in blocks.groupby('task'):
        ax.scatter(g.accuracy, g.belief, s=14, facecolors='none', edgecolors=TASK_COLORS[int(task)],
                   linewidths=.75, alpha=.8, label=f'Task {task}')
    ax.set(xlim=(.15, 1.025), ylim=(-.025, 1.025), yticks=[0, .5, 1], xticks=[.25, .5, .75, 1],
           xlabel='Observed accuracy / 64-trial block', ylabel='Mean target belief Q')
    ax.legend(loc='upper left', fontsize=5.5, handletextpad=.2, labelspacing=.2)
    fig.text(.075, .020, 'b: A/Q sustained for 16 trials; behavior uses trailing 64 trials. c: all 98 blocks; blocks are not independent participants.',
             fontsize=6, color='.35')
    finish(fig, output, 'Figure3_nine_subjects.png')
    sensitivity_figure(d, s, output)
    source_receipt(source, output, 'Fig3', ['trials.csv', 'subjects.csv', 'blocks64.csv',
                   'threshold_sensitivity.csv', 'seed_events.csv', 'candidate_sensitivity.csv',
                   'numerical_checks.csv', 'manifest.json'])


def sensitivity_figure(d: pd.DataFrame, s: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(183/25.4, 155/25.4))
    fig.subplots_adjust(left=.08, right=.98, bottom=.10, top=.85, hspace=.60, wspace=.25)
    title(fig, 'Companion | How dependent are the belief paths on the fitted candidate?',
          'Original nominee: 8 PF repeats; one documented near candidate: 4 PF repeats (128 particles each)')
    lines_legend(fig, [('Nominee', COLORS['belief'], '-'), ('Near candidate', COLORS['alternative'], '--')], .906, 2)
    for col, subjects in enumerate(SUBJECTS):
        for row, sid in enumerate(subjects):
            g = d.loc[d.iSub.eq(sid)]; ax = axes[row, col]
            ax.fill_between(g.trial, np.maximum(g.belief-g.belief_seed_sd, 0),
                            np.minimum(g.belief+g.belief_seed_sd, 1), color=COLORS['belief'], alpha=.14, lw=0)
            ax.plot(g.trial, g.belief, color=COLORS['belief'], lw=.85)
            ax.plot(g.trial, g.alternative_belief, color=COLORS['alternative'], ls='--', lw=.85)
            ax.set(title=f'S{sid}  |  Task {col+1}', xlim=(1,len(g)), ylim=(-.02,1.02),
                   yticks=[0,.5,1], xticks=[1,len(g)//2,len(g)], xlabel='Trial')
            if col == 0:
                ax.set_ylabel('Target belief Q')
    fig.text(.08,.018,'Band: ±1 PF-seed SD, not a confidence interval. A single near candidate does not cover parameter uncertainty.',fontsize=6,color='.35')
    finish(fig, output, 'belief_candidate_sensitivity.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig3(args.source, args.output)


if __name__ == '__main__':
    main()
