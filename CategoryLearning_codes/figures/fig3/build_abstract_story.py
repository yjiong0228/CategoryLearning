"""Fig3: explain behavior-selected learning trajectories with reconstructed beliefs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .bottleneck_analysis import ROOT, sha256
from .build_nine_subject_figures import COLORS, TASK_COLORS, style

MARKERS = {1: 'o', 2: '^', 3: 's'}
SEARCH = '#887299'
REALLOCATION = '#AB6C79'


def read_story(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((source/'manifest.json').read_text())
    subjects = pd.read_csv(source/'subjects.csv').set_index('subject')
    trials = pd.read_csv(source/'trials.csv')
    assert len(subjects) == manifest['participants']
    assert len(trials) == manifest['trials']
    assert subjects.index.is_unique and not trials.duplicated(['iSub','trial']).any()
    assert set(subjects.index) == set(trials.iSub)
    return trials, subjects, manifest


def task_legend(fig, y: float) -> None:
    fig.legend(handles=[Line2D([], [], marker=MARKERS[t], color=TASK_COLORS[t],
        lw=0, markersize=4, label=f'Task {t}') for t in (1, 2, 3)],
        loc='upper right', bbox_to_anchor=(.97,y), ncol=3,
        columnspacing=1.1, handletextpad=.4)


def export(fig, source: Path, output: Path, name: str, code: Path) -> None:
    """Keep PNG-only export and a complete, hash-addressed analysis snapshot."""
    fig.savefig(output/name, dpi=450, facecolor='white')
    physical_size = [float(x*25.4) for x in fig.get_size_inches()]
    plt.close(fig)
    shutil.copytree(source, output/'source_data')
    (output/'code').mkdir()
    dependencies = [code, Path(__file__), Path(__file__).with_name('abstract_story_analysis.py'),
                    Path(__file__).with_name('abstract_story_design.md'),
                    Path(__file__).with_name('build_nine_subject_figures.py')]
    for path in dict.fromkeys(dependencies):
        shutil.copy2(path, output/'code'/path.name)
    receipt = {'figure':name, 'source':str(source.resolve().relative_to(ROOT)),
        'dimensions_mm':physical_size, 'dpi':450, 'format':'PNG',
        'versions':{'python':platform.python_version(), 'numpy':np.__version__,
                    'pandas':pd.__version__, 'matplotlib':matplotlib.__version__},
        'source_sha256':{p.name:sha256(p) for p in source.iterdir() if p.is_file()},
        'code_sha256':{str(p.resolve().relative_to(ROOT)):sha256(p) for p in dict.fromkeys(dependencies)}}
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')


def build(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    d,s,m = read_story(source)
    examples = m['examples']
    reversals = pd.read_csv(source/'support_reversals.csv')
    jumps = pd.read_csv(source/'jump_windows.csv')
    style()
    fig = plt.figure(figsize=(183/25.4, 235/25.4))
    fig.text(.085,.978,'Fig. 3 | Different learning curves, different belief dynamics',
             fontsize=10, weight='bold', va='top')
    fig.text(.085,.954,f'{len(s)} learners · {len(d):,} trials | Start with behavior; then ask what changed inside the learner.',
             fontsize=7, color='.35', va='top')
    fig.text(.065,.918,'a',fontsize=9,weight='bold')
    fig.text(.095,.918,'When learning improved, and how',fontsize=8,weight='bold')
    task_legend(fig,.939)
    ax = fig.add_axes([.13,.777,.81,.115])
    ax.axhspan(-6,6,color='#F3F3F3',lw=0,zorder=0)
    ax.axhline(0,color='.65',lw=.6,zorder=0)
    highlighted = set(examples.values())
    offsets = {102:(5,3),118:(5,-10),122:(-9,9),206:(-7,-13),
               221:(-16,7),222:(7,-6),307:(-31,-7),314:(5,4),315:(-20,6),
               104:(5,-12),215:(8,3),328:(-30,7)}
    for sid,row in s.iterrows():
        task = int(row.task)
        chosen = sid in highlighted
        ax.scatter(row.criterion,row.delta_bic,marker=MARKERS[task],
                   s=35 if chosen else 18, c=TASK_COLORS[task],
                   edgecolor='white',linewidth=.5,zorder=4)
        ax.annotate(f'S{sid}',(row.criterion,row.delta_bic),xytext=offsets.get(sid,(5,5)),
                    textcoords='offset points', fontsize=6.3,
                    weight='bold' if chosen else 'normal', color=TASK_COLORS[task])
    ax.set(xlim=(0,max(1460,s.criterion.max()*1.06)),
           ylim=(min(-60,s.delta_bic.min()-8),max(32,s.delta_bic.max()+8)),
           xticks=[0,400,800,1200],yticks=[-50,0,20],
           xlabel='Trial of first behavioral criterion',ylabel='Step vs. gradual\nfit (ΔBIC)')
    ax.text(.99,.90,'Step-like',ha='right',transform=ax.transAxes,fontsize=6,color='.4')
    ax.text(.99,.06,'Gradual',ha='right',transform=ax.transAxes,fontsize=6,color='.4')
    fig.text(.085,.726,'Three illustrations selected from behavior alone; bold labels above identify the cases below.',
             fontsize=6.4,color='.35')
    handles = [Line2D([],[],color=COLORS['observed'],lw=1,label='Observed accuracy'),
               Line2D([],[],color='.55',lw=1,ls='--',label='Behavioral shape fit'),
               Line2D([],[],color=COLORS['available'],lw=1,label='Rule considered'),
               Line2D([],[],color=COLORS['belief'],lw=1,label='Belief in rule'),
               Line2D([],[],color=COLORS['executed'],lw=1,label='Rule executed')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.535,.711),ncol=5,
               columnspacing=1,handlelength=1.7,fontsize=6)
    xs = [.10,.414,.728]
    headings = ['Earlier learning','Slow, gradual improvement','Stagnation then improvement']
    ypos = [.526,.355,.207,.067]
    heights = [.113,.135,.098,.096]
    ylabels = ['Accuracy','Target-rule state','Global-search\ntendency','Belief\nreallocation']
    for col,(key,sid) in enumerate(examples.items()):
        g=d.loc[d.iSub.eq(sid)];row=s.loc[sid];x=xs[col];task=int(row.task)
        fig.text(x-.033,.665,chr(ord('b')+col),fontsize=9,weight='bold')
        fig.text(x,.665,headings[col],fontsize=7.4,weight='bold')
        fig.text(x,.650,f'S{sid} · Task {task} · {len(g)} trials',fontsize=6.2,color=TASK_COLORS[task])
        axes=[fig.add_axes([x,y,.235,h]) for y,h in zip(ypos,heights)]
        axes[0].plot(g.trial,g.correct_w32,color=COLORS['observed'],lw=.95)
        axes[0].plot(g.trial,g['behavior_'+row.preferred],color='.55',lw=1,ls='--')
        axes[0].axhline(.5 if task==1 else .25,color='.85',lw=.5,zorder=0)
        for measure in ('available','belief','executed'):
            if g[measure].notna().any():
                axes[1].plot(g.trial,g[measure],color=COLORS[measure],
                             lw=.65 if measure=='available' else .85)
        axes[1].axhline(.5,color='.82',lw=.5,ls=':',zorder=0)
        axes[2].plot(g.trial,g.global_range_w32,color=SEARCH,lw=.85)
        axes[2].fill_between(g.trial,g.global_range_w32,0,color=SEARCH,alpha=.10,lw=0)
        axes[3].plot(g.trial,g.reallocation_w32,color=REALLOCATION,lw=.85)
        axes[3].fill_between(g.trial,g.reallocation_w32,0,color=REALLOCATION,alpha=.10,lw=0)
        for track,ax in enumerate(axes):
            ax.set(xlim=(1,len(g)),ylim=(-.03,1.04) if track<3 else (-.005,.33),
                   yticks=[0,.5,1] if track<3 else [0,.15,.30])
            if track<3:ax.tick_params(axis='x',bottom=False,labelbottom=False)
            else:ax.set(xticks=[1,len(g)//2,len(g)],xlabel='Trial')
            if col==0:ax.set_ylabel(ylabels[track],labelpad=4)
            if key=='abrupt':
                ax.axvspan(row.split-32,row.split+32,color='#ECE8DE',alpha=.65,lw=0,zorder=0)
                ax.axvline(row.split+.5,color='#9B8C6F',ls=':',lw=.7,zorder=1)
        if key=='gradual':
            for _,ev in reversals.loc[reversals.subject.eq(sid)].iterrows():
                axes[1].axvspan(ev.support_start,ev.loss_start,color=COLORS['belief'],alpha=.08,lw=0,zorder=0)
                axes[1].annotate('',xy=(ev.loss_start,.18),xytext=(ev.loss_start,.72),
                                 arrowprops={'arrowstyle':'->','lw':.7,'color':COLORS['belief']})
        if key=='rapid':
            if np.isfinite(row.belief_event):
                event=int(row.belief_event)
                axes[1].annotate('Sustained support',xy=(event,g.iloc[event-1].belief),
                                 xytext=(len(g)*.45,.20),fontsize=6,color=COLORS['belief'],
                                 arrowprops={'arrowstyle':'-','lw':.5,'color':COLORS['belief']})
            description=f'First sustained support: t{int(row.belief_event)}' if np.isfinite(row.belief_event) else 'No sustained high target support'
        elif key=='gradual':
            description=f'{int(row.support_reversals)} sustained losses of target support'
        else:
            window=jumps.query('subject==@sid and window==32 and variant=="selected" and measure=="belief"')
            description=f'Belief around change: {window.before.mean():.2f} → {window.after.mean():.2f}'
        if key=='abrupt':
            axes[0].text(.03,.06,f'Change after t{int(row.split)}',transform=axes[0].transAxes,
                         fontsize=6,color='#75664B')
        fig.text(x,.323,description,fontsize=6.1,color=COLORS['belief'])
    fig.text(.085,.024,'Accuracy, search and reallocation: trailing 32 trials. Belief states: pre-choice estimates. Same scales within each row.',
             fontsize=6,color='.35')
    fig.text(.085,.011,'These are illustrative trajectories, not three established learner types. Shading in d marks the ±32-trial comparison window.',
             fontsize=6,color='.35')
    export(fig,source,output,'Figure3_learning_dynamics.png',Path(__file__))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.output)


if __name__=='__main__':main()
