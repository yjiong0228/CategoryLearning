"""Fig4: observed-history module diagnostics; no invented intervention results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..fig3.bottleneck_analysis import ROOT, sha256
from ..fig3.build_nine_subject_figures import (
    ORDER, COLORS, TASK_COLORS, style, read_tables, title, label, finish, mark_events, source_receipt,
)


def module_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    renewal, readout = [], []
    for sid in ORDER:
        g = d.loc[d.iSub.eq(sid)]
        for strong in (False, True):
            group = g.loc[(g.belief > .5) == strong]
            renewal.append({'subject': sid, 'strong': strong, 'n': len(group),
                            'replacement': group.replacement.mean(), 'search': group.search.mean()})
        group = g.loc[g.belief > .75]
        readout.append({'subject': sid, 'n': len(group), 'belief': group.belief.mean(),
                        'predicted': group.predicted.mean(), 'observed': group.correct.mean(),
                        'executed': group.executed.mean()})
    return pd.DataFrame(renewal), pd.DataFrame(readout)


def build(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    d, s, _ = read_tables(source)
    feedback = pd.read_csv(source / 'feedback.csv')
    renewal, readout = module_tables(d)
    renewal.to_csv(output / 'renewal_source.csv', index=False)
    readout.to_csv(output / 'readout_source.csv', index=False)
    style()
    fig = plt.figure(figsize=(183/25.4, 205/25.4))
    title(fig, 'Fig. 4 | From searching for rules to making reliable choices',
          'Process diagnostics from fitted trajectories  |  nine participants  |  no causal intervention contrast')
    axes = [fig.add_axes([x, .615, .24, .26]) for x in (.095, .405, .715)]
    label(axes[0], 'a', 'When is search more likely?')
    label(axes[1], 'b', 'Does renewal fall as belief grows?')
    label(axes[2], 'c', 'Strong belief and choice accuracy')
    for i, sid in enumerate(ORDER):
        g = feedback.loc[feedback.subject.eq(sid)]
        axes[0].plot(g.search, np.full(len(g), i), color='.8', lw=.8, zorder=1)
        for _, row in g.iterrows():
            marker, color = {1.: ('o', COLORS['belief']), 0.: ('x', '#A2674D'), .5: ('^', '#A46A8A')}[row.feedback]
            axes[0].scatter(row.search, i, marker=marker, color=color, s=17, lw=.8, zorder=2)
        g = renewal.loc[renewal.subject.eq(sid)].sort_values('strong')
        axes[1].plot(g.replacement, np.full(len(g), i), color='.8', lw=.8, zorder=1)
        for _, row in g.iterrows():
            axes[1].scatter(row.replacement, i, marker='o' if row.strong else 's',
                            color=COLORS['belief'] if row.strong else '.6', s=16, lw=.5, zorder=2)
        row = readout.loc[readout.subject.eq(sid)].iloc[0]
        if row.n:
            axes[2].plot([row.predicted,row.belief], [i,i], color='.8', lw=1)
            axes[2].scatter(row.belief, i, color=COLORS['belief'], s=17, zorder=2)
            axes[2].scatter(row.predicted, i, color=COLORS['predicted'], marker='D', s=16,zorder=2)
            axes[2].scatter(row.observed, i, color=COLORS['observed'], marker='|', s=36, lw=1,zorder=3)
            axes[2].text(1.055, i, str(int(row.n)), fontsize=5.8, va='center', color='.4')
    for i, ax in enumerate(axes):
        ax.set(ylim=(8.7,-.6), yticks=range(9),
               yticklabels=[f'S{x}' for x in ORDER] if i == 0 else [],
               xlabel=['Search probability', 'Replaced fraction of candidate set', 'Mean probability (Q > .75)'][i])
        ax.axhline(2.5,color='.9',lw=.5)
        ax.axhline(5.5,color='.9',lw=.5)
    for tick, sid in zip(axes[0].get_yticklabels(), ORDER):
        tick.set_color(TASK_COLORS[int(s.loc[sid].task)])
    axes[0].set_xlim(-.02,max(.35,feedback.search.max()*1.12))
    axes[1].set_xlim(-.005,renewal.replacement.max()*1.15)
    axes[1].ticklabel_format(axis='x',style='plain')
    axes[2].set_xlim(.2,1.02)
    axes[2].set_xticks([.25,.5,.75,1])
    axes[2].text(1.055,-.6,'n',fontsize=5.8,color='.4',ha='left')
    legends = [
        [('After full success','o',COLORS['belief']),('After failure','x','#A2674D'),('After partial success','^','#A46A8A')],
        [('Q ≤ .5','s','.6'),('Q > .5','o',COLORS['belief'])],
        [('Target belief Q','o',COLORS['belief']),('Model P(correct)','D',COLORS['predicted']),
         ('Observed accuracy','|',COLORS['observed'])],
    ]
    for ax, entries in zip(axes, legends):
        ax.legend(handles=[Line2D([],[],ls='',marker=m,color=c,label=n,markersize=4) for n,m,c in entries],
                  loc='upper left',bbox_to_anchor=(-.09,-.17),fontsize=5.8,handletextpad=.4,labelspacing=.25)
    fig.text(.075,.49,'When a rule is already supported, inspect execution and rule precision separately',fontsize=8,weight='bold')
    outer = fig.add_gridspec(1,3,left=.09,right=.97,bottom=.095,top=.445,wspace=.28)
    persistent = [sid for sid in ORDER if s.loc[sid,'chi'] == 1]
    assert len(persistent)==3
    for col, sid in enumerate(persistent):
        g=d.loc[d.iSub.eq(sid)]; info=s.loc[sid]
        inner=outer[0,col].subgridspec(3,1,hspace=.15,height_ratios=[1,1,1])
        aa=[fig.add_subplot(inner[i]) for i in range(3)]
        label(aa[0], chr(ord('d')+col), f'S{sid}  |  Task {int(info.task)}')
        for name in ('belief','executed'):
            aa[0].plot(g.trial,g[name],color=COLORS[name],lw=.8,label={'belief':'Belief Q','executed':'Execution E'}[name])
        aa[1].plot(g.trial,g.correct_w32,color=COLORS['observed'],lw=.8,label='Observed')
        aa[1].plot(g.trial,g.predicted_w32,color=COLORS['predicted'],lw=.9,ls='--',label='Model')
        aa[2].plot(g.trial,g.executed_beta,color='#795E78',lw=.8)
        aa[2].axhline(info.beta_0,color='.65',lw=.6,ls='--')
        for i, ax in enumerate(aa):
            ax.set(xlim=(1,len(g)),xticks=[] if i<2 else [1,len(g)//2,len(g)])
            if i<2:
                ax.set(ylim=(-.04,1.05),yticks=[0,.5,1])
            else:
                ax.set(ylim=(0,25.5),yticks=[0,10,20],xlabel='Trial')
            if np.isfinite(info.belief_event):
                ax.axvline(info.belief_event,color=COLORS['belief'],lw=.5,ls=':',alpha=.7)
        mark_events(aa,info)
        if col==0:
            aa[0].set_ylabel('Target rule')
            aa[1].set_ylabel('Accuracy')
            aa[2].set_ylabel('Executed-rule β')
            aa[0].legend(loc='upper left',fontsize=5.5,handlelength=1.2)
            aa[1].legend(loc='upper left',fontsize=5.5,handlelength=1.2)
    fig.text(.075,.045,'d–f: all three persistent-execution fits. β controls rule precision; dashed horizontal line marks its initial value.',fontsize=6,color='.35')
    fig.text(.075,.030,'Vertical dots: sustained target belief (teal), first behavioral criterion (gray). Belief and performance criteria differ.',fontsize=6,color='.35')
    fig.text(.075,.015,'Associations locate mechanisms to test; memory compensation and transfer benefits still require controlled comparisons.',fontsize=6,color='.35')
    finish(fig,output,'Figure4_process_diagnostics.png')
    pairing_figure(d,output)
    source_receipt(source,output,'Fig4',['trials.csv','subjects.csv','feedback.csv','manifest.json'])
    receipt=json.loads((output/'manifest.json').read_text())
    receipt.update(plotting_source=str(Path(__file__).resolve().relative_to(ROOT)),
                   plotting_source_sha256=sha256(Path(__file__)))
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')


def pairing_figure(d: pd.DataFrame, output: Path) -> None:
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,110/25.4))
    fig.subplots_adjust(left=.09,right=.98,top=.80,bottom=.14,hspace=.30,wspace=.28)
    title(fig,'Companion | Task 2 includes a second inference problem',
          'Target-rule belief and belief in the correct response pairing are different states')
    fig.texts[1].set_y(.935)
    for col,sid in enumerate([307,314,315]):
        g=d.loc[d.iSub.eq(sid)]
        axes[0,col].plot(g.trial,g.belief,color=COLORS['belief'],lw=.9,label='Target rule Q')
        axes[0,col].plot(g.trial,g.pairing,color='#A46A8A',lw=.9,label='Correct pairing')
        axes[0,col].set_title(f'S{sid}',loc='left')
        axes[1,col].plot(g.trial,g.correct_w32,color=COLORS['observed'],lw=.8)
        axes[1,col].plot(g.trial,g.predicted_w32,color=COLORS['predicted'],lw=.8,ls='--')
        for row in (0,1):
            axes[row,col].set(xlim=(1,len(g)),ylim=(-.02,1.02),yticks=[0,.5,1],
                              xticks=[] if row==0 else [1,len(g)//2,len(g)])
        axes[1,col].set_xlabel('Trial')
    axes[0,0].set_ylabel('Belief / probability')
    axes[1,0].set_ylabel('Accuracy')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.89),ncol=2,fontsize=6)
    fig.text(.09,.025,'Correct pairing = 12|34 in recorded choice coordinates; its initial probability is 1/3, not task knowledge given to the model.',fontsize=6,color='.35')
    finish(fig,output,'task2_pairing_diagnostic.png')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.output)


if __name__=='__main__':
    main()
