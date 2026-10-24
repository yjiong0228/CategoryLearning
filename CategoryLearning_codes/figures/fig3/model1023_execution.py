"""Observed learning and frozen execution-model predictions (PNG drafts).

Figure contract: describe the nine selected development cases and assess
whether each model reproduces their trajectories. Every 32-trial block and
every generated path contributes. A quantitative grid shares accuracy axes;
vertical lines mark the fitted prefix, dotted lines uniform guessing. Model
bands are 10th--90th percentiles across autonomous paths, not confidence
intervals. The plotted curves alone do not identify cognitive mechanisms.

Use --data PREPARED --output NEW [--study COMPLETE_FIT]. Python/matplotlib,
180 x 170 mm, 600 dpi PNG only. Trial/block source tables and a manifest are
retained. No manuscript figures are replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import read, digest, load_case
from src.Bayesian_state.workflows.analysis.model_1023_execution_study import write

COLORS = {'MIX': '#4A719A', 'MAP': '#B6853F', 'STICKY': '#7A6095'}
SUBJECTS = [129, 229, 317, 108, 205, 306, 110, 209, 309]


def blocks(choices, categories, width=32):
    return np.array([(choices[:, start:start+width] == categories[None, start:start+width]).mean(axis=1)
                     for start in range(0, len(categories), width)]).T


def run(data: Path, output: Path, study: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure destination')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'legend.frameon': False, 'lines.linewidth': 1.2})
    figure, axes = plt.subplots(3, 3, figsize=(7.0866, 6.6929), sharey=True)
    if study:
        if read(study / 'completion.json')['smoke']:
            raise ValueError('Figures require full scientific generation')
        nominees = read(study / 'nominees.json')['nominees']
        jobs = read(study / 'autonomous_jobs.json')['jobs']
    records, trials, hashes = [], [], {}
    for index, (sid, axis) in enumerate(zip(SUBJECTS, axes.flat)):
        folder = data / f'subject_{sid}'
        meta = read(folder / 'prepared.json')
        parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            if digest(folder / name) != meta['array_sha256'][name]:
                raise ValueError('Prepared data changed')
            with np.load(folder / name) as saved:
                parts.append({k: saved[k] for k in ('choices', 'categories')})
            hashes[str(folder / name)] = digest(folder / name)
        observed = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
        n, cut = meta['trials'], meta['training_trials']
        x = np.array([(start + min(start+32, n) + 1) / 2 for start in range(0, n, 32)])
        accuracy = blocks(observed['choices'][None], observed['categories'])[0]
        axis.axvspan(cut+.5, n+.5, color='#ECECEC', zorder=0)
        axis.axvline(cut+.5, color='#777777', linewidth=.7, linestyle='--')
        axis.axhline(.5 if meta['condition'] == 1 else .25, color='#777777', linewidth=.7, linestyle=':')
        if study:
            for mode in COLORS:
                nominee, = [r for r in nominees if r['subject'] == sid and r['arm'] == mode]
                case = load_case(study, nominee, jobs)
                values = blocks(case['autonomous']['choices'], observed['categories'])
                lo, hi = np.quantile(values, [.1, .9], axis=0)
                mean = values.mean(axis=0)
                axis.fill_between(x, lo, hi, color=COLORS[mode], alpha=.09, linewidth=0)
                axis.plot(x, mean, color=COLORS[mode], label=mode)
                for block, midpoint in enumerate(x):
                    records.append(dict(subject=sid, condition=meta['condition'], source=mode,
                        trial_midpoint=midpoint, mean=mean[block], p10=lo[block], p90=hi[block], paths=len(values)))
        axis.plot(x, accuracy, color='#242424', marker='o', markersize=2.2, linewidth=1.05,
                  label='Observed', zorder=5)
        for block, midpoint in enumerate(x):
            records.append(dict(subject=sid, condition=meta['condition'], source='Observed',
                trial_midpoint=midpoint, mean=accuracy[block], p10=np.nan, p90=np.nan, paths=1))
        trials.extend(dict(subject=sid, trial=t+1, choice=int(choice), category=int(category),
            fitting_prefix=t<cut) for t, (choice, category) in enumerate(zip(observed['choices'], observed['categories'])))
        axis.set(title=f'S{sid} · condition {meta["condition"]}', ylim=(-.03, 1.04), xlim=(.5, n+.5))
        axis.set_yticks([0, .25, .5, .75, 1])
        if index % 3 == 0:
            axis.set_ylabel('Accuracy per 32 trials')
        if index >= 6:
            axis.set_xlabel('Valid trial')
        axis.text(-.18, 1.08, chr(ord('a') + index), transform=axis.transAxes, fontweight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.53, 1.))
    figure.subplots_adjust(left=.10, right=.98, bottom=.09, top=.92, wspace=.19, hspace=.45)
    output.mkdir(parents=True)
    target = output / ('execution_model_trajectories.png' if study else 'observed_development_cases.png')
    figure.savefig(target, dpi=600)
    plt.close(figure)
    pd.DataFrame(records).to_csv(output / 'block_source_data.csv', index=False)
    pd.DataFrame(trials).to_csv(output / 'trial_source_data.csv', index=False)
    write(output / 'manifest.json', dict(status='complete', figure=target.name, subjects=SUBJECTS,
        source_sha256=digest(Path(__file__)), input_sha256=hashes,
        study=None if study is None else str(study.resolve()),
        nominees_sha256=None if study is None else digest(study / 'nominees.json'),
        counts=dict(subjects=9, trials=len(trials)), excluded_trials=0, block_width=32,
        shaded_region='chronological suffix outside parameter fitting',
        bands='10th--90th percentile of all autonomous paths; no bands for empirical accuracy',
        status_interpretation='development cases; behavior descriptions do not establish latent mechanisms'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--study', type=Path)
    args = parser.parse_args()
    run(args.data, args.output, args.study)
