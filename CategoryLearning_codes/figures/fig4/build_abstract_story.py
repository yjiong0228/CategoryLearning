"""Fig4: direct tests of the fast, gradual and abrupt narratives in the abstract."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..fig3.build_nine_subject_figures import COLORS, TASK_COLORS, style
from ..fig3.build_abstract_story import MARKERS, export, read_story, task_legend


def subject_points(ax, subjects: pd.DataFrame, x: str, y: str,
                   offsets: dict | None = None) -> None:
    for sid,r in subjects.iterrows():
        if not np.isfinite([r[x],r[y]]).all():continue
        task=int(r.task)
        ax.scatter(r[x],r[y],s=23,color=TASK_COLORS[task],marker=MARKERS[task],
                   edgecolors='white',linewidths=.4,zorder=3)
        dx,dy=(offsets or {}).get(sid,(4,4))
        ax.annotate(str(sid),(r[x],r[y]),xytext=(dx,dy),textcoords='offset points',
                    fontsize=6,color=TASK_COLORS[task])


def panel(fig, x: float, y: float, letter: str, heading: str):
    ax=fig.add_axes([x,y,.225,.218])
    ax.text(-.20,1.18,letter,transform=ax.transAxes,fontsize=9,weight='bold')
    ax.set_title(heading,loc='left',fontsize=7.1,pad=18)
    return ax


def build(source: Path, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False)
    _,s,_=read_story(source)
    jumps=pd.read_csv(source/'jump_windows.csv')
    inventory=pd.read_csv(source/'rule_inventory.csv')
    for sid,row in s.iterrows():
        arity=2 if row.condition==1 else 4
        target_dim=inventory.loc[inventory.n_categories.eq(arity)&inventory.hypothesis_index.eq(row.target),'active_dimensions'].item()
        s.loc[sid,'excess_dimensions']=row.early_rule_dimensions-target_dim
    s['gradual_preference']=-s.delta_bic
    step_cases=s.loc[(s.delta_bic>=6)&s.preferred.eq('step')]
    style()
    fig=plt.figure(figsize=(183/25.4,198/25.4))
    fig.text(.085,.976,'Fig. 4 | Which processes explain the learning trajectories?',
             fontsize=10,weight='bold',va='top')
    fig.text(.085,.947,'Test the abstract\u2019s mechanisms: early focus, repeated rebuilding, and the source of a breakthrough.',
             fontsize=6.8,color='.35',va='top')
    task_legend(fig,.916)
    xs=[.105,.428,.75]
    for x,heading in zip(xs,['Early focus & stability','Search & rebuilding','Belief before breakthrough']):
        fig.text(x-.023,.863,heading,fontsize=8,weight='bold')
    ax=panel(fig,xs[0],.554,'a','Do faster learners\nfavor simpler rules?')
    subject_points(ax,s,'criterion','excess_dimensions',
                   {122:(5,4),102:(-21,5),118:(5,5),307:(-13,-12),314:(-25,5),
                    315:(-24,-12),221:(-24,7),222:(5,4),206:(-20,5)})
    ax.axhline(0,color='.82',lw=.6,zorder=0)
    ax.set(xlim=(0,1460),ylim=(-.12,1.25),xticks=[0,600,1200],yticks=[0,.5,1],
           xlabel='Trial of behavioral criterion',ylabel='Rule dimensions above target\n(first 128 trials)')
    ax=panel(fig,xs[1],.554,'b','Is broader search linked\nto gradual change?')
    subject_points(ax,s,'early_global_range','gradual_preference',
                   {102:(-8,-12),118:(5,2),122:(5,3),206:(5,3),307:(4,-12),
                    314:(-5,5),315:(-24,4),221:(-23,-12),222:(3,4)})
    ax.axhline(0,color='.8',lw=.6,zorder=0)
    ax.set(xlim=(-.04,.89),ylim=(-22,64),xticks=[0,.4,.8],yticks=[-20,0,20,40,60],
           xlabel='Global-search tendency\n(first 128 trials)',ylabel='Preference for gradual over step\n(−ΔBIC)')
    ax=panel(fig,xs[2],.554,'c','Was useful belief\nalready strong?')
    measures=['available','belief']
    for pos,(sid,row) in enumerate(step_cases.iterrows()):
        for i,measure in enumerate(measures):
            xx=pos+(-.15 if i==0 else .15)
            main=jumps.query('subject==@sid and window==32 and variant=="selected" and measure==@measure').before.mean()
            alt=jumps.query('subject==@sid and window==32 and variant=="alternative" and measure==@measure').before.mean()
            ax.plot([xx,xx],[main,alt],color=COLORS[measure],lw=.8,zorder=2)
            ax.scatter(xx,alt,s=26,facecolors='white',edgecolors=COLORS[measure],linewidths=1,zorder=3)
            ax.scatter(xx,main,s=25,color=COLORS[measure],zorder=4)
            if measure=='belief':ax.annotate(f'{main:.2f}',(xx,main),xytext=(5,-4),textcoords='offset points',fontsize=6.2,color=COLORS['belief'])
    ax.axhline(.5,color='.6',lw=.6,ls=':',zorder=0)
    ax.set(xlim=(-.5,len(step_cases)-.3),ylim=(-.02,1.02),yticks=[0,.5,1],
           xticks=range(len(step_cases)),xticklabels=[f'S{sid}\nTask {int(row.task)}' for sid,row in step_cases.iterrows()],
           ylabel='Mean state before behavioral change\n(previous 32 trials)')
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS[k],markersize=4,label=l)
                       for k,l in [('available','Considered'),('belief','Belief')]],
              loc='upper left',fontsize=6,handletextpad=.4)
    fig.text(xs[2]-.015,.490,'Filled: fitted nominee; open: near candidate.',fontsize=6,color='.4')
    ax=panel(fig,xs[0],.182,'d','Does useful belief\nsurvive an error?')
    subject_points(ax,s,'criterion','error_retention',
                   {102:(-20,7),118:(-15,-12),122:(4,4),206:(-20,5),221:(3,5),
                    222:(7,5),307:(-3,10),314:(-24,-12),315:(-24,-12)})
    ax.set(xlim=(0,1460),ylim=(.76,1.09),xticks=[0,600,1200],yticks=[.8,.9,1.],
           xlabel='Trial of behavioral criterion',ylabel='Fraction retaining high support\nafter negative feedback')
    ax=panel(fig,xs[1],.182,'e','Is weaker retention linked\nto rebuilding?')
    subject_points(ax,s,'gamma','early_reallocation',
                   {102:(-24,-13),118:(4,4),122:(-20,5),206:(4,-10),221:(-19,6),
                    222:(-20,5),307:(5,-10),314:(-22,-10),315:(-24,-12)})
    ax.set(xlim=(.30,1.035),ylim=(.045,.26),xticks=[.4,.7,1],yticks=[.05,.15,.25],
           xlabel='Evidence-retention parameter (γ)',ylabel='Mean belief reallocation\n(first 128 trials)')
    ax=panel(fig,xs[2],.182,'f','What changes\nat the breakthrough?')
    names=['available','belief','executed','predicted']
    positions=np.arange(len(names))
    for index,(sid,row) in enumerate(step_cases.iterrows()):
        task=int(row.task);off=-.10 if index==0 else .10
        for i,name in enumerate(names):
            sub=jumps.query('subject==@sid and window==32 and variant=="selected" and measure==@name')
            if sub.empty:
                ax.text(i+off,.01,'n/a',rotation=90,fontsize=5.5,color=TASK_COLORS[task],ha='center',va='bottom')
                continue
            change=(sub.after-sub.before).mean()
            ax.plot([i+off,i+off],[0,change],color=TASK_COLORS[task],lw=1)
            ax.scatter(i+off,change,marker=MARKERS[task],s=23,color=TASK_COLORS[task],zorder=3)
    ax.axhline(0,color='.7',lw=.6)
    ax.set(xlim=(-.5,3.55),ylim=(-.035,.69),yticks=[0,.3,.6],
           xticks=positions,xticklabels=['Considered','Belief','Executed','P(correct)'],
           ylabel='After − before behavioral change\n(32 trials on each side)')
    ax.tick_params(axis='x',labelrotation=45)
    ax.legend(handles=[Line2D([],[],marker=MARKERS[int(row.task)],ls='',color=TASK_COLORS[int(row.task)],
        label=f'S{sid}',markersize=4) for sid,row in step_cases.iterrows()],loc='upper right',fontsize=6)
    fig.text(.085,.069,'a, b, d, e: all 9 learners; labels identify individuals. Compare within tasks. c, f: all 2 clearly step-preferring cases.',
             fontsize=6.2,color='.35')
    fig.text(.085,.049,'Reallocation measures movement of inferred belief, not observed resets. Retention is a fitted parameter, not measured memory capacity.',
             fontsize=6,color='.35')
    fig.text(.085,.029,'The present data distinguish candidate explanations; causal claims about compensation require controlled model comparisons.',
             fontsize=6.2,color='.35')
    export(fig,source,output,'Figure4_learning_mechanisms.png',Path(__file__))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.output)


if __name__=='__main__':main()
