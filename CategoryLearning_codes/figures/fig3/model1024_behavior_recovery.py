"""Draw audited known-state references and targeted particle-resolution checks.

The adjacent contract specifies data scope, conditioning and export policy.
All 15 reference cases and 7 fixed numerical windows remain in the PNGs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1024_behavior_recovery import read,write,digest

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'legend.fontsize':6,'legend.frameon':False,'axes.spines.right':False,'axes.spines.top':False,
    'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
BLUE,ORANGE,GREY='#39769C','#B77E40','#8D949C'
LEVELS=(512,1024,2048)
COLORS={512:GREY,1024:BLUE,2048:ORANGE}


def save(fig,output:Path,name:str):
    fig.savefig(output/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def run(source:Path,output:Path)->None:
    if output.exists():raise FileExistsError('Use a new figure directory')
    done=read(source/'completion.json')
    if done['status']!='complete' or done['smoke']:raise ValueError('Require complete formal numerical summary')
    for name,sha in done['artifacts_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('Audited table changed')
    output.mkdir(parents=True)
    reference=pd.read_csv(source/'reference_metrics.csv')
    metrics=pd.read_csv(source/'numerical_metrics.csv');drift=pd.read_csv(source/'numerical_drift.csv')
    curves=pd.read_csv(source/'numerical_curves.csv');episodes=pd.read_csv(source/'state_episodes.csv')
    chosen=reference.loc[reference.window.eq('whole')&reference.state.eq('error_available_other')]
    cases=chosen[['case_id','subject','scope']].drop_duplicates()
    cases['order']=cases.scope.ne('fixed').astype(int)
    cases=cases.sort_values(['order','subject','case_id']);ids=cases.case_id.tolist();yy=np.arange(len(ids))
    fig,axes=plt.subplots(1,2,figsize=(7.2,5.8),sharey=True)
    fig.subplots_adjust(left=.25,right=.97,top=.91,bottom=.19,wspace=.22)
    for i,(field,title) in enumerate([('true_event_probability','True-class probability on key errors'),('hard_recall','Key-error classification recall')]):
        ax=axes[i]
        groups={role:chosen.loc[chosen.role.eq(role)].set_index('case_id').reindex(ids)
                for role in ['behavior_filter','known_state_reference']}
        for t in yy:
            a,b=[g[field].iloc[t] for g in groups.values()]
            if np.isfinite([a,b]).all():ax.plot([a,b],[t,t],color='.80',lw=.7,zorder=1)
            else:ax.text(.025,t,'NA',va='center',fontsize=6,color=GREY)
        for role,color,marker in [('behavior_filter',BLUE,'o'),('known_state_reference',ORANGE,'s')]:
            ax.plot(groups[role][field],yy,marker,ls='',ms=3.7,color=color)
        ax.axhline(11.5,color='.55',ls='--',lw=.7)
        ax.set_xlim(0,1.03);ax.set_ylim(len(ids)-.5,-.5)
        ax.set_title(f'{chr(97+i)}  {title}',loc='left',fontweight='bold')
        ax.grid(axis='x',color='.93',lw=.5)
    counts=chosen.loc[chosen.role.eq('behavior_filter')].set_index('case_id').reindex(ids).positives.to_numpy(int)
    axes[0].set_yticks(yy,[f'{r.case_id} / {r.subject}   n={n}' for r,n in zip(cases.itertuples(),counts)])
    fig.legend(handles=[Line2D([],[],ls='',marker='o',color=BLUE,label='Behavior-only filter; known parameters'),
                        Line2D([],[],ls='',marker='s',color=ORANGE,label='Known pre-execution state + current choice')],
               loc='lower center',bbox_to_anchor=(.55,.105),ncol=2)
    fig.text(.25,.025,'12 fixed paths above dashed line; 3 truth-enriched paths below. n = actual key errors.\nKnown-state reference has extra information; it is a conditional diagnostic.\nClassification uses largest source probability; zero-event conditional metrics are NA.',fontsize=7)
    save(fig,output,'known_state_reference')

    ids=metrics.case_id.drop_duplicates().sort_values().tolist();yy=np.arange(len(ids))
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.9),sharey=True)
    fig.subplots_adjust(left=.14,right=.98,top=.89,bottom=.25,wspace=.28)
    panels=[('present','brier','Rule presence: Brier'),('error_available_other','true_event_probability','True-class probability')]
    for i,(state,field,title) in enumerate(panels):
        ax=axes[i]
        for level,shift in [(512,-.16),(1024,0.),(2048,.16)]:
            g=metrics.loc[metrics.particles.eq(level)&metrics.state.eq(state)&metrics.timing.eq('filtered')].set_index('case_id').reindex(ids)
            ax.errorbar(g[field],yy+shift,xerr=2*g[field+'_mcse'],fmt='o',color=COLORS[level],ms=3.5,capsize=1.5,lw=.6)
        ax.set_title(f'{chr(97+i)}  {title}',loc='left',fontweight='bold');ax.set_xlim(left=0)
        ax.grid(axis='x',color='.93',lw=.5)
        if i==1:ax.set_xlim(0,1.03)
    for lo,hi,shift,color in [(512,1024,-.09,BLUE),(1024,2048,.09,ORANGE)]:
        g=drift.loc[drift.particles_from.eq(lo)&drift.particles_to.eq(hi)&drift.field.eq('filtered_target_present')].set_index('case_id').reindex(ids)
        axes[2].plot(g.mean_absolute_change,yy+shift,'o',color=color,ms=3.5,label=f'{lo} to {hi}')
    axes[2].set_xlim(left=0);axes[2].set_title('c  Presence-probability drift',loc='left',fontweight='bold')
    axes[2].legend(loc='upper right',fontsize=6);axes[2].grid(axis='x',color='.93',lw=.5)
    axes[0].set_yticks(yy,ids);axes[0].set_ylim(len(ids)-.5,-.5)
    fig.legend(handles=[Line2D([],[],color=COLORS[n],marker='o',ls='',label=f'{n} particles') for n in LEVELS],
               loc='lower center',bbox_to_anchor=(.54,.105),ncol=3)
    fig.text(.14,.025,'7 prespecified event windows; 3 paired filter seeds per level; generating parameters known.\nBars = +/-2 numerical MCSE; full history precedes each scored window.',fontsize=7)
    save(fig,output,'particle_resolution')

    fig,axes=plt.subplots(len(ids),1,figsize=(7.2,7.4),sharex=True)
    fig.subplots_adjust(left=.11,right=.98,top=.95,bottom=.15,hspace=.70)
    for i,cid in enumerate(ids):
        ax=axes[i];own=curves.loc[curves.case_id.eq(cid)&curves.in_window]
        ep=episodes.loc[episodes.case_id.eq(cid)&episodes.episode.eq(1)&episodes.role.eq('numerical_prefix')].iloc[0]
        entry=int(ep.entry_trial)
        first=own.loc[own.particles.eq(512)].sort_values('trial')
        ax.step(first.trial-entry,first.true_present.astype(int),where='post',color='#202020',lw=.9)
        for level in LEVELS:
            g=own.loc[own.particles.eq(level)].sort_values('trial')
            ax.plot(g.trial-entry,g.presence,color=COLORS[level],lw=.9)
        ax.set_ylim(-.03,1.06);ax.set_yticks([0,1]);ax.axvline(0,color='.65',ls=':',lw=.6)
        ax.set_title(f'{chr(97+i)}  {cid}, subject {int(first.subject.iloc[0])}; first true episode: {int(ep.true_length)} trials',loc='left',fontweight='bold')
    axes[-1].set_xlabel('Trials relative to first entry')
    fig.legend(handles=[Line2D([],[],color='#202020',lw=.9,label='True presence')]+
               [Line2D([],[],color=COLORS[n],lw=.9,label=f'{n} particles') for n in LEVELS],
               loc='lower center',bbox_to_anchor=(.55,.065),ncol=4)
    fig.text(.11,.02,'Lines: presence probability after current choice, before feedback.\nAll window trials shown; filters start at trial 1. Oral validation is deferred.',fontsize=7)
    save(fig,output,'entry_windows')
    write(output/'manifest.json',dict(status='rendered_pending_visual_review',backend='python',dpi=300,
        source_sha256=digest(Path(__file__)),summary_completion_sha256=digest(source/'completion.json'),
        files=[p.name for p in sorted(output.glob('*.png'))],export_policy='PNG only',oral_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
