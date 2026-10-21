"""Plot the completed S205 entry-support × entry-precision diagnostic.

Use --centered, --precision and --combined for the three completed summary
directories, and --output for a new figure directory. Each control uses the
original COMPACT parameters, 128 paired autonomous paths and eight 512-particle
training filters. No refitting or participant-level inference is performed.

Figure contract: precision transfer improves autonomous behavior at this point;
support centering alone does not. A quantitative grid separates training cost,
later autonomous Energy, rule availability/use and mean learning curves. Python
exports a 183 × 170 mm research-review PNG and all source tables. Error bars are
two paired Monte Carlo standard errors, conditional on the fixed parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load


def run(centered: Path, precision: Path, combined: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new entry-diagnostic figure directory')
    arms = ['COMPACT', 'CENTERED', 'PRECISION', 'CENTERED_PRECISION']
    labels = ['COMPACT', 'Support\ncenter', 'Precision\ntransfer', 'Both']
    colors = ['#737C86', '#B78262', '#477697', '#8C7B9E']
    styles = ['-', ':', '--', '-.']
    tables, bindings = {}, {}
    for arm, folder in zip(arms[1:], (centered, precision, combined)):
        complete = load(folder/'completion.json')
        if complete['status'] != 'complete':
            raise ValueError('Require complete scientific summaries')
        bindings[str((folder/'completion.json').resolve())] = digest(folder/'completion.json')
        group = {}
        for name, sha in complete['artifacts_sha256'].items():
            file = folder/name
            if digest(file) != sha:
                raise ValueError('Diagnostic summary changed')
            group[file.stem] = pd.read_csv(file)
            bindings[str(file.resolve())] = sha
        if set(group['scores'].arm) != {'COMPACT', arm}:
            raise ValueError('Wrong control in summary')
        tables[arm] = group
    for name in ('scores', 'curves', 'paths'):
        base = tables[arms[1]][name].query('arm == "COMPACT"').reset_index(drop=True)
        for arm in arms[2:]:
            pd.testing.assert_frame_equal(base, tables[arm][name].query('arm == "COMPACT"').reset_index(drop=True))
    scores = pd.concat([tables[arms[1]]['scores'].query('arm == "COMPACT"')] +
                       [tables[a]['scores'].query('arm == @a') for a in arms[1:]])
    scores = scores.query('phase == "test"').set_index('arm').loc[arms]
    if not np.isfinite(scores.select_dtypes(include=np.number)).all().all():
        raise ValueError('Nonfinite diagnostic score')
    curves = pd.concat([tables[arms[1]]['curves'].query('arm == "COMPACT"')] +
                       [tables[a]['curves'].query('arm == @a') for a in arms[1:]])
    base_curve = curves.query('arm == "COMPACT"')
    base_seeds = tables[arms[1]]['paths'].query('arm == "COMPACT"').seed.to_numpy()
    for arm in arms[1:]:
        paths = tables[arm]['paths'].query('arm == @arm')
        if len(paths) != 128 or paths.seed.nunique() != 128:
            raise ValueError('Require all 128 autonomous paths')
        np.testing.assert_array_equal(base_seeds, paths.seed.to_numpy())
        np.testing.assert_array_equal(base_curve[['trial', 'observed', 'training_trials']].to_numpy(),
                                     curves.query('arm == @arm')[['trial', 'observed', 'training_trials']].to_numpy())
    output.mkdir(parents=True)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False, 'axes.linewidth': .65}):
        fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 170/25.4))
        fig.subplots_adjust(left=.18, right=.98, top=.90, bottom=.22, wspace=.85, hspace=.60)
        for ax, metric, title in zip(axes[0], ('training_nll', 'energy'),
                                    ('a  Training choice NLL / trial', 'b  Later autonomous Energy')):
            ax.axvline(0, color='#B5BAC1', lw=.8)
            for i, arm in enumerate(arms[1:]):
                row, = tables[arm]['contrasts'].query('phase == "test"').to_dict('records')
                ax.errorbar(row['delta_'+metric], i, xerr=2*row['delta_'+metric+'_mc_se'],
                            fmt='o', color=colors[i+1], capsize=2, ms=4)
            ax.set_yticks(range(3), labels[1:], fontsize=7)
            ax.set_ylim(2.5, -.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Difference from COMPACT\nLower is better', fontsize=7)
        ax = axes[1, 0]
        for i, arm in enumerate(arms):
            present, contribution = scores.loc[arm, ['target_late_present', 'target_late_contribution']]
            ax.plot([contribution, present], [i, i], color=colors[i], lw=1)
            ax.scatter(present, i, marker='s', s=20, color=colors[i])
            ax.scatter(contribution, i, marker='o', s=22, color=colors[i], facecolor='white', zorder=3)
        ax.set_yticks(range(4), labels, fontsize=7)
        ax.set(xlim=(0, 1.02), ylim=(4.6, -.5), xlabel='Mean over last 128 trials')
        ax.set_title('c  Target rule: availability and use', fontsize=8)
        ax.legend(handles=[Line2D([], [], marker='s', color='#777777', ls='', label='Present'),
                           Line2D([], [], marker='o', color='#777777', markerfacecolor='white', ls='',
                                  label='Contribution to choice')], fontsize=6.5, loc='lower left')
        ax = axes[1, 1]
        for arm, color, style in zip(arms, colors, styles):
            curve = curves.query('arm == @arm')
            ax.plot(curve.trial, curve.autonomous, color=color, ls=style, lw=1.2)
        ax.plot(base_curve.trial, base_curve.observed, color='#222222', lw=1, marker='o', ms=2)
        ax.axvspan(base_curve.training_trials.iloc[0], base_curve.trial.max(), color='#F1F2F4', zorder=0)
        ax.set(xlim=(0, base_curve.trial.max()), ylim=(0, 1), xlabel='Trial (32-trial blocks)', ylabel='Accuracy')
        ax.set_title('d  Autonomous mean learning curves', fontsize=8)
        fig.legend(handles=[Line2D([], [], color=c, ls=s, label=l.replace('\n', ' '))
                            for c, s, l in zip(colors, styles, labels)] +
                           [Line2D([], [], color='#222222', marker='o', ms=2, label='Participant')],
                   loc='lower center', bbox_to_anchor=(.5, .06), ncol=3, fontsize=7)
        fig.suptitle('S205 · initial support × initial precision of new rules', fontsize=9, y=.975)
        fig.text(.5, .025, 'Fixed parameters; 128 paths / arm; 8 filters × 512 particles. Bars: ±2 paired MCSE.',
                 ha='center', fontsize=7)
        file = output/'entry_support_and_precision.png'
        fig.savefig(file, dpi=300, facecolor='white')
        plt.close(fig)
    (output/'source_data').mkdir()
    for arm, group in tables.items():
        for name, table in group.items():
            table.to_csv(output/'source_data'/f'{arm}_{name}.csv', index=False)
    (output/'manifest.json').write_text(json.dumps(dict(status='complete', inputs_sha256=bindings,
        figure_sha256=digest(file), source_sha256=digest(Path(__file__)), backend='python',
        export='PNG at 300 dpi, 183 × 170 mm', scope='One development participant, fixed parameters, no refit',
        uncertainty='Two paired MCSE, conditional on parameters; no population test',
        data_use='All four mechanisms; all 128 paths; training filters use 480 trials, later Energy uses 224 trials',
        adaptation='Style-only inheritance; direct diagnostic-table mapping; no log or scaling transform'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('centered', 'precision', 'combined', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.centered, args.precision, args.combined, args.output)
