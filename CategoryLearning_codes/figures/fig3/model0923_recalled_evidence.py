"""PNG diagnostics of the fixed-parameter recalled-evidence factorial study.

--summary names a validated evaluation directory; --output must be new.
Bands show the central 80% of autonomous paths, not uncertainty of the mean.
The process comparison retains all paths, including undiscovered target rules.
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

ORDER = ['POOL', 'PRIOR', 'SEARCH', 'BOTH']
COLORS = dict(POOL='#68778B', PRIOR='#A16F99', SEARCH='#277F8E', BOTH='#C7894B')
MARKERS = dict(POOL='o', PRIOR='s', SEARCH='^', BOTH='D')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    manifest = json.loads((summary / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['smoke']:
        raise ValueError('Require a completed scientific comparison')
    tables = {}
    for name in ('curves', 'scores', 'contrasts'):
        file = summary / f'{name}.csv'
        if digest(file) != manifest['artifacts'][file.name]:
            raise ValueError('Scored input changed')
        tables[name] = pd.read_csv(file)
    curves = tables['curves']; scores = tables['scores'].query("phase == 'whole'")
    subjects = sorted(scores.subject.unique())
    if scores.duplicated(['subject', 'arm']).any() or set(zip(scores.subject, scores.arm)) != {
            (s, a) for s in subjects for a in ORDER}:
        raise ValueError('Incomplete factorial cells')
    if curves[['mean', 'lower', 'upper', 'observed']].isna().any().any():
        raise ValueError('Missing curves')
    output.mkdir(parents=True)
    curves.to_csv(output / 'curve_source.csv', index=False)
    scores.to_csv(output / 'process_source.csv', index=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7, 'axes.titlesize': 8,
        'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.linewidth': .6, 'lines.linewidth': 1.0, 'savefig.dpi': 300})
    fig, axes = plt.subplots(4, len(subjects), figsize=(183/25.4, 190/25.4), squeeze=False)
    for column, subject in enumerate(subjects):
        for row, arm in enumerate(ORDER):
            ax = axes[row, column]
            frame = curves.loc[curves.subject.eq(subject) & curves.arm.eq(arm)].sort_values('trial')
            ax.fill_between(frame.trial, frame.lower, frame.upper, color=COLORS[arm], alpha=.2, linewidth=0)
            ax.plot(frame.trial, frame['mean'], color=COLORS[arm])
            ax.plot(frame.trial, frame.observed, color='#222222', linewidth=.85)
            ax.set_ylim(0, 1.02); ax.set_yticks([0, .5, 1]); ax.set_xlim(0, frame.trial.max())
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2.5, width=.6)
            ax.text(.03, .95, arm, color=COLORS[arm], weight='bold', va='top', transform=ax.transAxes)
            if row == 0:
                ax.set_title(f'Participant {subject}', pad=8)
            if column == 0:
                ax.set_ylabel('Accuracy / 32 trials')
            if row == 3:
                ax.set_xlabel('Trial')
    fig.suptitle('Autonomous learning at the same fitted POOL parameters', fontsize=9, y=.985)
    sizes = ', '.join(str(int(n)) for n in sorted(scores.paths.unique()))
    fig.text(.5, .955, f'{sizes} autonomous paths per cell; parameters held fixed', ha='center', fontsize=7)
    handles = [Line2D([], [], color='#222222', label='Observed'),
               Line2D([], [], color='#777777', label='Simulated mean'),
               Patch(facecolor='#999999', alpha=.3, label='10–90% of paths')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.075, right=.985, bottom=.085, top=.91, hspace=.34, wspace=.27)
    fig.savefig(output / 'autonomous_curves.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 76/25.4))
    metrics = [('target_ever', 'Correct rule discovered', 'Fraction of all paths'),
               ('target_late_contribution', 'Correct rule used: last 128 trials', 'Mean contribution to choice'),
               ('energy', 'Whole-curve distribution score', 'Energy (lower is better)')]
    centers = np.arange(len(subjects))
    for ax, (metric, title, label) in zip(axes, metrics):
        for i, arm in enumerate(ORDER):
            frame = scores.loc[scores.arm.eq(arm)].set_index('subject').loc[subjects]
            x = centers + (i - 1.5) * .16
            ax.plot(x, frame[metric], ls='', marker=MARKERS[arm], color=COLORS[arm], ms=4, label=arm)
            if metric == 'energy':
                ax.errorbar(x, frame[metric], yerr=2*frame.energy_mcse, fmt='none', color=COLORS[arm], linewidth=.7)
        ax.set_xticks(centers, subjects); ax.set_xlabel('Participant'); ax.set_ylabel(label)
        ax.set_title(title, fontsize=7, pad=10)
        ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, width=.6)
        ax.set_xlim(-.5, len(subjects)-.5)
        if metric != 'energy':
            ax.set_ylim(0, 1.05)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False)
    fig.subplots_adjust(left=.075, right=.985, bottom=.25, top=.84, wspace=.48)
    fig.savefig(output / 'discovery_and_use.png', dpi=300)
    plt.close(fig)
    contract = dict(backend='python', formats=['PNG'], dpi=300, width_mm=183,
        claim='History-guided proposals improve discovery in 207 and 308 at fixed parameters; belief reconstruction adds no consistent benefit.',
        source=str(summary.resolve()), source_manifest_sha256=digest(summary / 'manifest.json'),
        subjects=[int(s) for s in subjects], arms=ORDER, paths_per_cell=sorted(int(n) for n in scores.paths.unique()),
        sample_role='development; frozen POOL parameters; no refitted-model ranking',
        panels={'autonomous_curves': 'all subject/arm means with central 80% path bands and observed curves',
                'discovery_and_use': 'discovery, late decision contribution, whole-curve Energy'},
        uncertainty={'bands': '10th and 90th path quantiles, descriptive predictive spread',
                     'energy_bars': 'two whole-path jackknife Monte Carlo SE',
                     'excluded': 'parameter and participant sampling uncertainty'},
        limitations=['fixed observed stimulus schedule', 'three selected development participants',
                     'target identity used for recording only', 'no oral input or parameter selection'],
        export_exceptions=['PNG only by standing repository instruction; discussion figures use 300 dpi'])
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)),
        artifacts={p.name: digest(p) for p in output.iterdir()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); plot(args.summary, args.output)
