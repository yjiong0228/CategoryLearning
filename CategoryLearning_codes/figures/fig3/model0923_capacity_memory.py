"""PNG of the crossed capacity and belief-retention parameter intervention.

Run with --source completed_probe --output NEW_DIRECTORY. Cells retain every
simulated path; the original SEARCH parameter point is outlined. Entries show
estimates and Monte Carlo standard errors. The probe curve table retains path spread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_table(source: Path) -> pd.DataFrame:
    completion = json.loads((source / 'completion.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    if completion['status'] != 'complete' or completion['smoke'] or summary['smoke']:
        raise ValueError('Require a completed scientific parameter intervention')
    if digest(source / 'summary.json') != completion['summary_sha256']:
        raise ValueError('Probe summary changed')
    for file in ('scores.csv', 'path_windows.csv'):
        if digest(source / file) != summary['artifacts'][file]:
            raise ValueError(f'Probe table changed: {file}')
    scores = pd.read_csv(source / 'scores.csv')
    paths = pd.read_csv(source / 'path_windows.csv')
    late = paths.loc[paths.stop.eq(paths.groupby('subject').stop.transform('max'))].copy()
    if not late.stop.sub(late.start).add(1).eq(128).all():
        raise ValueError('Require a complete final 128-trial window for each participant')
    rows = []
    for (subject, point), group in late.groupby(['subject', 'point']):
        if group.seed.duplicated().any() or len(group) != group.paths.iloc[0]:
            raise ValueError('Incomplete final-window path bank')
        row = dict(subject=subject, point=point, observed_accuracy=group.observed_accuracy.iloc[0])
        for metric in ('accuracy', 'target_contribution'):
            row[metric] = group[metric].mean()
            row[metric + '_mcse'] = group[metric].std(ddof=1) / np.sqrt(len(group))
        rows.append(row)
    table = scores.merge(pd.DataFrame(rows), on=['subject', 'point'], validate='one_to_one', how='outer')
    if table.isna().any().any():
        raise ValueError('Missing parameter-cell statistics')
    for _, group in table.groupby('subject'):
        if (group.duplicated(['memory','capacity']).any()
                or len(group) != group.memory.nunique() * group.capacity.nunique()
                or group.anchor.sum() != 1):
            raise ValueError('Require a full crossed grid with one original point')
    return table


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    table = load_table(source); subjects = sorted(table.subject.unique())
    metrics = [('energy', 'Whole-curve Energy (lower is better)', 'viridis_r'),
               ('accuracy', 'Accuracy: last 128 trials', 'viridis'),
               ('target_contribution', 'Correct-rule contribution: last 128', 'viridis')]
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':7, 'axes.titlesize':7, 'axes.labelsize':7, 'xtick.labelsize':6,
        'ytick.labelsize':6, 'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.dpi':300})
    single = len(subjects) == 1
    height_mm = 90 if single else 125
    fig, axes = plt.subplots(len(subjects), 3, figsize=(183/25.4, height_mm/25.4), squeeze=False)
    fig.subplots_adjust(left=.09, right=.96, bottom=.20 if single else .15,
                        top=.74 if single else .84, hspace=.65, wspace=.7)
    for column, (metric, title, cmap) in enumerate(metrics):
        norm = Normalize(table[metric].min(), table[metric].max())
        for row, subject in enumerate(subjects):
            ax = axes[row,column]; group = table.loc[table.subject.eq(subject)]
            memory = sorted(group.memory.unique()); capacity = sorted(group.capacity.unique())
            values = group.pivot(index='memory', columns='capacity', values=metric).loc[memory,capacity]
            shown = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')
            for y,gamma in enumerate(memory):
                for x,cap in enumerate(capacity):
                    cell = group.loc[group.memory.eq(gamma) & group.capacity.eq(cap)].iloc[0]
                    rgb = shown.cmap(norm(cell[metric]))[:3]
                    text_color = 'black' if np.dot(rgb,[.2126,.7152,.0722]) > .52 else 'white'
                    ax.text(x,y,f'{cell[metric]:.3f}\n±{cell[metric+"_mcse"]:.3f}',
                            ha='center',va='center',fontsize=6,color=text_color)
                    if cell.anchor:
                        ax.add_patch(Rectangle((x-.47,y-.46),.94,.92,fill=False,color='white',linewidth=2.5))
                        ax.add_patch(Rectangle((x-.47,y-.46),.94,.92,fill=False,color='black',linewidth=.8))
            ax.set_xticks(range(len(capacity)),capacity); ax.set_yticks(range(len(memory)),[f'{g:g}' for g in memory])
            ax.set_xlabel('Active rule capacity'); ax.set_ylabel('Belief retention')
            label = f'Participant {subject}'
            if metric == 'accuracy':
                label += f'\nObserved {group.observed_accuracy.iloc[0]:.3f}'
            ax.set_title(label,pad=7)
            if row == 0:
                ax.text(.5,1.25 if single else 1.34,title,ha='center',transform=ax.transAxes,fontsize=7)
        fig.colorbar(shown,ax=axes[:,column],fraction=.035,pad=.04).ax.tick_params(labelsize=6)
    fig.suptitle('Capacity and belief retention at fixed SEARCH parameters',fontsize=9,y=.98)
    counts=', '.join(str(int(n)) for n in sorted(table.paths.unique()))
    fig.text(.5,.045,f'{counts} paths per cell; outline: original point; entries: estimate ± Monte Carlo SE',
             ha='center',fontsize=7)
    output.mkdir(parents=True); table.to_csv(output/'plot_source.csv',index=False)
    fig.savefig(output/'capacity_memory.png',dpi=300);plt.close(fig)
    contract = dict(backend='python', formats=['PNG'], dpi=300, width_mm=183, height_mm=height_mm,
        claim='Separate effects of belief retention and active capacity on autonomous learning and rule use.',
        source=str(source.resolve()), summary_sha256=digest(source/'summary.json'),
        subjects=[int(s) for s in subjects], paths_per_cell=[int(n) for n in sorted(table.paths.unique())],
        sample_role='development; exploratory parameter intervention; all other SEARCH parameters fixed',
        uncertainty='Energy: whole-path jackknife MCSE; late means: path SD/sqrt(n); no parameter or participant sampling CI',
        limitations=['whole observed stimulus schedule','not a refit or an independent validation','no oral input'],
        export_exceptions=['PNG only by repository instruction; 300dpi discussion figure'])
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)),
        artifacts={p.name:digest(p) for p in output.iterdir()}),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.source,args.output)
