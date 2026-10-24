"""Plot bounded model, parameter and state recovery from completed results.

Figure contract: show all declared recovery cases, individual kappa estimates,
and paired generating-parameter versus fitted-parameter state scores. These
are synthetic checks on three fixed task schedules, not population estimates.
Four quantitative panels, Python/matplotlib, 180 x 155 mm, 600 dpi PNG only.
Use --summary COMPLETE_RECOVERY_SUMMARY --output NEW. Source tables and input
hashes accompany the figure; no paths are selected by recovery success.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1023_execution_study import read, write, digest, KAPPA


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery-figure destination')
    done = read(source / 'completion.json')
    if done['status'] != 'complete' or done['smoke']:
        raise ValueError('Require completed scientific recovery scores')
    for name, sha in done['artifacts_sha256'].items():
        if digest(source / name) != sha:
            raise ValueError('Recovery summary changed')
    selected, parameters, states = [pd.read_csv(source / name) for name in
        ('model_selections.csv', 'parameter_recovery.csv', 'state_recovery.csv')]
    if selected.case_id.nunique() != done['cases']:
        raise ValueError('Recovery cases missing from model selections')
    selected['generator'] = [f'STICKY κ={k:g}' if mode == 'STICKY' else mode
        for mode, k in zip(selected.generating_model, selected.generating_kappa)]
    generators = ['MIX', 'MAP'] + [f'STICKY κ={k:g}' for k in
        sorted(selected.loc[selected.generating_model.eq('STICKY'), 'generating_kappa'].unique())]
    modes = ['MIX', 'MAP', 'STICKY']
    counts = pd.crosstab(selected.generator, selected.fitted_model).reindex(
        index=generators, columns=modes, fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    kappa = parameters.loc[parameters.parameter.eq(KAPPA)].copy()
    if len(kappa) != selected.generating_model.eq('STICKY').sum():
        raise ValueError('Missing fitted execution-maintenance parameters')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    figure, axes = plt.subplots(2, 2, figsize=(7.0866, 6.1024))
    matrix, parameter_axis, lag_axis, error_axis = axes.flat
    matrix.imshow(proportions.to_numpy(), cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i, generator in enumerate(generators):
        total = int(counts.loc[generator].sum())
        for j, mode in enumerate(modes):
            fraction = proportions.loc[generator, mode]
            matrix.text(j, i, f'{counts.loc[generator, mode]}/{total}', ha='center', va='center',
                        color='white' if fraction > .55 else '#252525')
    matrix.set(xticks=np.arange(3), xticklabels=modes, yticks=np.arange(len(generators)),
               yticklabels=generators, xlabel='Selected by later-choice likelihood', ylabel='Generating model')
    matrix.set_title('Model recovery: all declared cases', loc='left')
    colors = {108: '#437C8F', 205: '#B48943', 306: '#85659F'}
    markers = {108: 'o', 205: 's', 306: '^'}
    for sid, group in kappa.groupby('subject'):
        offsets = group.groupby('truth').cumcount().to_numpy()
        sizes = group.groupby('truth')['case_id'].transform('size').to_numpy()
        group = group.copy()
        group['display_true_kappa'] = group.truth + .12 * (offsets - (sizes-1)/2)
        parameter_axis.scatter(group.display_true_kappa, group.estimate, color=colors[sid],
            marker=markers[sid], s=18, linewidths=.5, edgecolors='white', label=f'S{sid}')
        kappa.loc[group.index, 'display_true_kappa'] = group.display_true_kappa
    parameter_axis.plot([0, 8], [0, 8], color='#777777', linestyle=':', linewidth=.8)
    parameter_axis.set(xlim=(-.4, 8.3), ylim=(-.4, 8.4), xticks=[0, 2, 6], yticks=[0, 2, 4, 6, 8],
                       xlabel='Generating κ (small display offsets)', ylabel='Re-estimated κ')
    parameter_axis.set_title('Parameter recovery: individual paths', loc='left')
    parameter_axis.legend(loc='lower right', fontsize=6)
    source_pairs = []
    for axis, field, title in [(lag_axis, 'high_belief_other_execution', 'Belief ≥ 0.5, other execution'),
                                (error_axis, 'error_source', 'Three sources of observed errors')]:
        rows = states.loc[states.generating_model.eq('STICKY') & states.timing.eq('filtered') &
            states.state.eq(field) & states.inference.isin(['oracle', 'fitted_STICKY'])]
        pairs = rows.pivot(index='case_id', columns='inference', values='brier')
        expected = set(selected.loc[selected.generating_model.eq('STICKY'), 'case_id'])
        # A path without observed errors has no error-source score. Retain its
        # identity explicitly; the joint-state panel always has every path.
        missing = sorted(expected - set(pairs.index))
        if field != 'error_source' and missing:
            raise ValueError('Missing joint-state recovery scores')
        if pairs.isna().any().any():
            raise ValueError('Incomplete generating/fitted parameter pair')
        subjects = rows.groupby('case_id').subject.first()
        for case_id, pair in pairs.iterrows():
            sid = int(subjects.loc[case_id]); values = pair[['oracle', 'fitted_STICKY']].to_numpy()
            axis.plot([0, 1], values, color=colors[sid], alpha=.5, linewidth=.6)
            axis.scatter([0, 1], values, color=colors[sid], marker=markers[sid], s=12, alpha=.85)
            source_pairs.append(dict(case_id=case_id, subject=sid, state=field,
                generating_parameter_brier=values[0], fitted_parameter_brier=values[1]))
        axis.set(xticks=[0, 1], xticklabels=['Generating parameters', 'Re-estimated parameters'],
                 xlim=(-.2, 1.2), ylabel='Brier error (smaller is better)')
        axis.set_ylim(bottom=0)
        axis.set_title(title + f'\n{len(pairs)} paired paths; {len(missing)} without a score', loc='left')
        source_pairs.extend(dict(case_id=case_id, state=field, reason='no observed errors') for case_id in missing)
    for index, axis in enumerate(axes.flat):
        axis.text(-.16, 1.12, chr(ord('a')+index), transform=axis.transAxes, fontweight='bold')
    figure.subplots_adjust(left=.16, right=.98, bottom=.1, top=.92, wspace=.55, hspace=.65)
    output.mkdir(parents=True)
    figure.savefig(output / 'execution_recovery.png', dpi=600)
    plt.close(figure)
    counts.to_csv(output / 'model_selection_counts.csv')
    kappa.to_csv(output / 'kappa_source_data.csv', index=False)
    pd.DataFrame(source_pairs).to_csv(output / 'paired_state_source_data.csv', index=False)
    write(output / 'manifest.json', dict(status='complete', cases=done['cases'],
        input_completion_sha256=digest(source / 'completion.json'), source_sha256=digest(Path(__file__)),
        input_artifacts_sha256=done['artifacts_sha256'], raster_dpi=600,
        caption='Fixed task schedules and generating points; two paths per point. MIX and STICKY κ=0 '
                'share the same marginal behavior. All declared cases enter model recovery. '
                'Panels c–d use STICKY-generated paths, including κ=0; state inference conditions '
                'on current/past choices before current feedback. Generating/fitted scores both '
                'include finite-particle error. These are development recovery checks.',
        model_exclusions=0, parameter_exclusions=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('summary', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    run(args.summary, args.output)
