"""Show the S205 two-coordinate diagnostic at the original frozen parameter point.

Use --source COMPLETE_PROBE --output NEW. Python/matplotlib exports one 183-mm
PNG and source tables. Panels separate training likelihood, later autonomous
Energy, learning curves and target-rule availability. Paired Monte Carlo errors
condition on parameters. This development diagnostic leaves primary nominees
unchanged; it provides no independent participant-level validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new diagnostic figure directory')
    complete = load(source/'completion.json')
    if complete.get('smoke') or complete['status'] != 'complete' or not complete['primary_nominee_unchanged']:
        raise ValueError('Require the completed scientific diagnostic')
    tables, bindings = {}, {}
    for name, sha in complete['artifacts_sha256'].items():
        file = source/name
        if digest(file) != sha:
            raise ValueError('Diagnostic source changed')
        tables[file.stem] = pd.read_csv(file)
        bindings[str(file.resolve())] = sha
    scores = tables['scores'].query('phase == "test"').sort_values(['baseline_probability', 'evidence_decay'])
    if len(scores) != 4 or scores.candidate.nunique() != 4:
        raise ValueError('Four coordinate combinations are required')
    contrast = tables['contrasts'].query('phase == "test"').set_index('candidate')
    colors = {0.02: '#527697', 0.05: '#AD785B'}
    styles = {0.95: '--', 0.99: '-'}
    labels = [f'p₀={r.baseline_probability:.2f}, δ={r.evidence_decay:.2f}' +
              (' *' if r.role == 'frozen_nominee' else '') for r in scores.itertuples()]
    output.mkdir(parents=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False, 'axes.linewidth': .65}):
        fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 160/25.4))
        fig.subplots_adjust(left=.21, right=.98, top=.90, bottom=.17, wspace=.95, hspace=.55)
        for ax, metric, title in zip(axes[0], ['training_nll', 'energy'],
                                    ['a  Training choice NLL / trial', 'b  Later autonomous Energy']):
            ax.axvline(0, color='#80868E', lw=.8)
            for i, point in enumerate(scores.itertuples()):
                r = contrast.loc[point.candidate]
                ax.errorbar(r['delta_'+metric], i, xerr=2*r['delta_'+metric+'_mc_se'], fmt='o',
                    color=colors[point.baseline_probability], capsize=2, ms=4,
                    markerfacecolor=colors[point.baseline_probability] if point.evidence_decay == .99 else 'white')
            ax.set_yticks(range(4), labels, fontsize=7)
            ax.set_ylim(3.5, -.5)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Difference from frozen nominee\nLower is better', fontsize=7)
        ax = axes[1, 0]
        for point in scores.itertuples():
            curve = tables['curves'][tables['curves'].candidate == point.candidate]
            ax.plot(curve.trial, curve.autonomous, color=colors[point.baseline_probability],
                    ls=styles[point.evidence_decay], lw=1.2)
        ax.plot(curve.trial, curve.observed, color='#222222', lw=1, marker='o', ms=2)
        ax.axvspan(curve.training_trials.iloc[0], curve.trial.max(), color='#F1F2F4', zorder=0)
        ax.set(xlim=(0, curve.trial.max()), ylim=(0, 1), xlabel='Trial (32-trial blocks)', ylabel='Accuracy')
        ax.set_title('c  Autonomous mean learning curves', fontsize=8)
        ax = axes[1, 1]
        for i, point in enumerate(scores.itertuples()):
            paths = tables['paths'][tables['paths'].candidate == point.candidate]
            if len(paths) != 128 or paths.seed.nunique() != 128:
                raise ValueError('Incomplete autonomous path bank')
            ax.plot([paths.discovered.mean(), paths.late_present.mean()], [i, i], color='#B9BDC3', lw=1)
            ax.scatter(paths.discovered.mean(), i, s=20, marker='s', color=colors[point.baseline_probability])
            ax.scatter(paths.late_present.mean(), i, s=20, marker='o', color=colors[point.baseline_probability],
                       facecolor='white', zorder=3)
        ax.set_yticks(range(4), labels, fontsize=7)
        ax.set(xlim=(0, 1.03), ylim=(4.4, -.5), xlabel='Proportion')
        ax.set_title('d  Target-rule availability', fontsize=8)
        ax.legend(handles=[Line2D([], [], marker='s', color='#777777', ls='', label='Ever found'),
                           Line2D([], [], marker='o', color='#777777', markerfacecolor='white', ls='',
                                  label='Present in last 128 trials')], fontsize=6.5,
                  loc='lower left')
        fig.suptitle('S205 · search frequency × active evidence retention', fontsize=9, y=.975)
        fig.text(.5, .075, 'p₀: baseline search probability; δ: evidence retention. * Original frozen nominee.', ha='center', fontsize=7)
        fig.text(.5, .04, 'Bars: ±2 paired MCSE. Curves: 128 paths / point; black: participant. Development diagnostic.', ha='center', fontsize=7)
        file = output/'search_and_retention.png'
        fig.savefig(file, dpi=300, facecolor='white')
        plt.close(fig)
    (output/'source_data').mkdir()
    for name, table in tables.items():
        table.to_csv(output/'source_data'/(name+'.csv'), index=False)
    (output/'manifest.json').write_text(json.dumps(dict(status='complete', source=str(source.resolve()),
        inputs_sha256=bindings, figure_sha256=digest(file), source_sha256=digest(Path(__file__)),
        export='PNG at 300 dpi, 183 mm width', primary_nominee_unchanged=True,
        uncertainty='Two paired MCSE at fixed parameters; no population inference'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
