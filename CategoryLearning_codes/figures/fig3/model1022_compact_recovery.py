"""Show what fitted parameters recover about a known synthetic rule trajectory.

The quantitative grid compares generating and estimated parameter points for
membership, choice contribution and precision conditional on membership. Every
trial in each completed recovery case is shown. The existing recovery figure
supplies the layout; case_id replaces replicate as the grouping key and role
replaces its older parameter labels. Statistics come from the shared summary.
Shading is the 5--95% weighted latent-state interval at fixed parameters.

Use --source COMPLETED_RECOVERY_STATES --output NEW. Python/matplotlib exports
183 x 165 mm PNG review figures at 300 dpi, with source CSVs. Predictive summaries
remain in the CSVs; panels show states after choice and before current feedback.
These development cases assess inference on a few generating points, and the
intervals omit parameter uncertainty.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1021_simplification_recovery import read, write, digest

LABELS = {'truth': 'Generating parameters', 'estimate': 'Estimated parameters'}


def render(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery figure directory')
    done = read(source/'completion.json')
    if done['status'] != 'complete' or done['smoke']:
        raise ValueError('Require full recovery-state summaries')
    bindings = dict(done['inputs_sha256'])
    bindings[str((source/'completion.json').resolve())] = digest(source/'completion.json')
    for name, sha in done['artifacts_sha256'].items():
        bindings[str((source/name).resolve())] = sha
    for name, sha in bindings.items():
        if digest(Path(name)) != sha:
            raise ValueError('Recovery figure input changed: '+name)
    trials = pd.read_csv(source/'trials.csv')
    scores = pd.read_csv(source/'scores.csv')
    if (trials.case_id.nunique() != done['cases']
            or set(trials.case_id) != set(scores.case_id)
            or set(trials.phase) != {'predictive', 'filtered'}):
        raise ValueError('Incomplete recovery cases or inference timings')
    output.mkdir(parents=True)
    for name in ('trials.csv', 'scores.csv', 'filter_scores.csv'):
        shutil.copyfile(source/name, output/name)
    files = []
    color = '#527697'
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':7,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.linewidth':.65, 'legend.frameon':False}):
        for case_id, group in trials.query('phase == "filtered"').groupby('case_id', sort=True):
            own = scores.query('case_id == @case_id and phase == "filtered"').set_index('role')
            counts = own[['trials', 'filters', 'particles']].drop_duplicates().to_numpy()
            if (set(own.index) != set(LABELS) or own.index.duplicated().any()
                    or set(group.role) != set(LABELS) or counts.shape != (1, 3)
                    or not np.isfinite(counts).all() or (counts <= 0).any()
                    or (counts != np.floor(counts)).any()):
                raise ValueError('Require both parameter points with common numerical budgets')
            n_trials, filters, particles = map(int, counts[0])
            panels = {role:group[group.role.eq(role)].sort_values('trial') for role in LABELS}
            for panel in panels.values():
                np.testing.assert_array_equal(panel.trial, np.arange(1, n_trials+1))
            truth_cols = ['truth_present', 'truth_contribution', 'truth_beta']
            np.testing.assert_array_equal(panels['truth'][truth_cols], panels['estimate'][truth_cols])
            values = group[['conditional_beta_mean', 'conditional_beta_q95', 'truth_beta']].to_numpy()
            finite = values[np.isfinite(values)]
            upper_beta = max(1., float(finite.max())*1.05) if len(finite) else 1.
            fig, axes = plt.subplots(3, 2, figsize=(183/25.4, 165/25.4), sharex=True, sharey='row')
            fig.subplots_adjust(left=.12, right=.98, bottom=.16, top=.81, hspace=.25, wspace=.18)
            fig.suptitle(case_id+' · known and inferred target-rule states', y=.985, fontsize=10)
            fig.legend(handles=[Line2D([], [], color='#222222', ls='--', lw=1, label='Known state'),
                Line2D([], [], color=color, lw=1.2, label='Inferred mean'),
                Patch(facecolor=color, alpha=.18, label='90% latent-state interval')],
                loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, fontsize=7)
            for col, role in enumerate(LABELS):
                panel, score = panels[role], own.loc[role]
                x = panel.trial.to_numpy()
                axes[0, col].set_title(f'{LABELS[role]}\nChoice NLL / trial = {score.choice_nll:.4f}', fontsize=8)
                axes[0, col].step(x, panel.truth_present, where='mid', color='#222222', lw=.9, ls='--')
                axes[0, col].plot(x, panel.presence_probability, color=color, lw=1)
                for row, prefix, key in ((1, 'contribution', 'truth_contribution'),
                                         (2, 'conditional_beta', 'truth_beta')):
                    known = panel[key].to_numpy().copy()
                    if row == 2:
                        known[~panel.truth_present.astype(bool)] = np.nan
                    ax = axes[row, col]
                    ax.fill_between(x, panel[prefix+'_q05'].to_numpy(), panel[prefix+'_q95'].to_numpy(),
                                    color=color, alpha=.18, lw=0)
                    ax.plot(x, panel[prefix+'_mean'], color=color, lw=1)
                    ax.plot(x, known, color='#222222', ls='--', lw=.9)
                for row in (0, 1):
                    axes[row, col].set(ylim=(-.04, 1.04), yticks=[0, .5, 1])
                axes[2, col].set(ylim=(0, upper_beta), xlabel='Trial')
                if not panel.truth_present.any():
                    axes[2, col].set_title('Known trajectory: target absent', fontsize=6.5, pad=3)
                for ax in axes[:, col]:
                    ax.set_xlim(1, n_trials)
                    ax.set_xticks(np.unique(np.r_[1, np.rint(np.linspace(0, n_trials, 5)[1:]).astype(int)]))
            for ax, label in zip(axes[:, 0], ['Target membership', 'Choice contribution', 'Precision given presence']):
                ax.set_ylabel(label)
            fig.text(.5, .09, f'All {n_trials} trials; state after choice and before feedback. '
                     f'{filters} filters × {particles} particles per point.', ha='center', fontsize=6.6)
            fig.text(.5, .055, 'Intervals condition on fixed parameters. Absent-rule precision is undefined.',
                     ha='center', fontsize=6.6)
            fig.text(.5, .02, 'Both columns share scales within a case; precision scales vary between cases.',
                     ha='center', fontsize=6.6)
            name = case_id+'.png'
            fig.savefig(output/name, dpi=300, facecolor='white')
            plt.close(fig)
            files.append(name)
    page = '<!doctype html><meta charset="utf-8"><title>Compact state recovery</title>'
    page += '<style>body{max-width:1100px;margin:auto;font:16px sans-serif}img{width:100%}</style>'
    page += '<h1>Compact state recovery</h1><p>Few generating points and synthetic paths. '
    page += 'State intervals condition on fixed parameters; they omit parameter uncertainty.</p>'
    page += ''.join('<h2>'+html.escape(name.removesuffix('.png'))+'</h2><img src="'+name+'">' for name in files)
    page += '<h2>Descriptive errors</h2>'+scores.query('phase == "filtered"').to_html(index=False)
    (output/'index.html').write_text(page, encoding='utf-8')
    shutil.copyfile(Path(__file__), output/'source_snapshot.py')
    write(output/'completion.json', dict(status='complete', backend='python', inputs_sha256=bindings,
        script_sha256=digest(Path(__file__)), source=str(source.resolve()),
        figures={name:digest(output/name) for name in files},
        source_data={name:digest(output/name) for name in ('trials.csv', 'scores.csv', 'filter_scores.csv')},
        plotted_phase='filtered', plotted_cases=sorted(trials.case_id.unique().tolist()),
        trial_selection='Every recorded trial; predictive rows retained in source tables',
        dimensions_mm=[183, 165], dpi=300, format='PNG', publication_readiness='Review figures; visual QA pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
