"""Plot choice-conditioned availability, rule belief and discrete execution.

Figure contract: separate the three model quantities on the same trial axis,
without treating inferred states as empirical measurements. One quantitative
four-panel figure per participant shows all observed blocks and all saved
state trials. MIX displays its belief readout; MAP/STICKY also display execution
and the joint event of target belief >= 0.5 with another rule executed. All
state curves condition on current/past choices and fixed prefix-fit parameters.
Python/matplotlib; 180 x 155 mm; 600 dpi PNG only. Source curves and provenance
are saved alongside the figures. Use --capture COMPLETE_STATES --output NEW.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1023_execution_recovery_score import checked_arrays, pool_states
from src.Bayesian_state.workflows.analysis.model_1023_execution_study import read, write, digest, KAPPA


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new state-figure destination')
    completed = read(source / 'completion.json')
    if completed['status'] != 'complete' or completed['smoke'] or not completed['exact_behavior_replay']:
        raise ValueError('State figures require completed scientific replay')
    if digest(source / 'protocol.json') != completed['protocol_sha256']:
        raise ValueError('State capture protocol changed')
    groups, inputs = {}, {}
    for file in sorted((source / 'states').glob('*.json')):
        meta = read(file)
        arrays = checked_arrays(file, 'array_sha256')
        arrays = {k: v for k, v in arrays.items() if not k.startswith('oral_')}
        if digest(Path(meta['original_job'])) != meta['original_job_sha256']:
            raise ValueError('Captured trial source changed')
        groups.setdefault((meta['subject'], meta['model']), []).append((meta, arrays))
        inputs[str(file.resolve())] = digest(file)
        inputs[str(file.with_suffix('.npz').resolve())] = meta['array_sha256']
    subjects = sorted({sid for sid, mode in groups})
    modes = ['MIX', 'MAP', 'STICKY']
    if set(groups) != {(sid, mode) for sid in subjects for mode in modes}:
        raise ValueError('Missing execution model in state figure')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    properties = [('target_present', 'Correct rule available', '#437C8F', '-'),
        ('target_belief', 'Belief in correct rule', '#B48943', '-'),
        ('target_execution', 'Correct rule executed', '#85659F', '-'),
        ('high_belief_other_execution', 'Belief ≥ 0.5, other execution', '#AE5F61', '--')]
    output.mkdir(parents=True)
    source_rows, rendered = [], []
    for sid in subjects:
        first = groups[sid, 'MIX'][0][0]
        reference = groups[sid, 'MIX'][0][1]
        n = len(reference['choices']); cut = first['training_trials']
        x = np.arange(1, n+1)
        correct = reference['choices'] == reference['categories']
        centers = [(start + min(n, start+32) + 1)/2 for start in range(0, n, 32)]
        accuracies = [correct[start:start+32].mean() for start in range(0, n, 32)]
        figure, axes = plt.subplots(4, 1, figsize=(7.0866, 6.1024), sharex=True,
                                    gridspec_kw={'height_ratios': [1., 1., 1., 1.]})
        axes[0].plot(centers, accuracies, 'o-', color='#252525', markersize=2.2, linewidth=1.)
        axes[0].axhline(.5 if first['condition'] == 1 else .25, color='#777777', linewidth=.7, linestyle=':')
        axes[0].set_ylabel('Observed\naccuracy / 32 trials')
        axes[0].set_title(f'S{sid} · condition {first["condition"]} · parameters fitted to the prefix', loc='left')
        handles, labels = {}, {}
        for row_index, mode in enumerate(modes, 1):
            records = groups[sid, mode]
            state = pool_states([a for _, a in records])
            np.testing.assert_array_equal(state['choices'], reference['choices'])
            np.testing.assert_array_equal(state['categories'], reference['categories'])
            np.testing.assert_array_equal(state['target_rules'], reference['target_rules'])
            axis = axes[row_index]
            kappa = records[0][0]['candidate']['hyperparams'][KAPPA]
            axis.set_ylabel(mode + (f'\nκ = {kappa:g}' if mode == 'STICKY' else '') + '\nProbability / belief')
            for name, label, color, style in properties[:2 if mode == 'MIX' else 4]:
                values = state['filtered_' + name]
                line, = axis.plot(x, values, color=color, linestyle=style, linewidth=.75)
                handles[name], labels[name] = line, label
            for t in range(n):
                source_rows.append(dict(subject=sid, condition=first['condition'], model=mode, trial=t+1,
                    choice=int(state['choices'][t]), category=int(state['categories'][t]),
                    fitting_prefix=t<cut, filter_seeds=len(records),
                    **{name: float(state['filtered_'+name][t]) for name, *_ in properties}))
        for index, axis in enumerate(axes):
            axis.axvspan(cut+.5, n+.5, color='#EEEEEE', zorder=0)
            axis.axvline(cut+.5, color='#777777', linestyle='--', linewidth=.6)
            axis.set(ylim=(-.03, 1.03), xlim=(.5, n+.5), yticks=[0, .5, 1])
            axis.text(-.12, 1.02, chr(ord('a')+index), transform=axis.transAxes, fontweight='bold')
        axes[-1].set_xlabel('Valid trial')
        figure.legend(list(handles.values()), list(labels.values()), loc='lower center',
                      ncol=2, bbox_to_anchor=(.54, .005))
        figure.subplots_adjust(left=.14, right=.98, bottom=.15, top=.94, hspace=.25)
        file = output / f'S{sid}_execution_states.png'
        figure.savefig(file, dpi=600)
        plt.close(figure)
        rendered.append(file.name)
    pd.DataFrame(source_rows).to_csv(output / 'state_curve_source_data.csv', index=False)
    write(output / 'manifest.json', dict(status='complete', subjects=subjects, figures=rendered,
        source_sha256=digest(Path(__file__)), input_sha256=inputs, filtered_state_rows=len(source_rows),
        temporal_scope='current/past choices; before current feedback; fixed prefix-fit parameters',
        belief_definition='mean within-person belief over inferred paths',
        execution_definition='probability of an executed target rule across discrete latent paths',
        lag_definition='joint event: target present, target belief >= 0.5, other rule executed',
        mixture_display='MIX contribution equals its belief; no discrete execution curve',
        uncertainty='latent-path uncertainty conditional on fixed parameters; finite-particle approximation',
        exclusions=0, raster_dpi=600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.capture, args.output)
