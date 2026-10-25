"""Audited zero-inertia execution marginalization figures; PNG only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1024_marginal_execution import read,write,digest

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'legend.fontsize':6,'legend.frameon':False,'axes.spines.right':False,'axes.spines.top':False,
    'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
COLORS={'sampled':'#818A91','filtered':'#39769C','lag':'#B77E40','full':'#8D6C9F'}
LABELS={'sampled':'Sampled baseline','filtered':'Marginal: filter','lag':'Marginal: lag 32','full':'Marginal: full'}


def save(fig,path:Path,name:str):
    fig.savefig(path/(name+'.png'),dpi=300,facecolor='white');plt.close(fig)


def run(source:Path,output:Path)->None:
    if output.exists():raise FileExistsError('Use a new figure directory')
    done=read(source/'completion.json')
    if done['status']!='complete' or done['smoke']:raise ValueError('Require complete formal recovery tables')
    for file,sha in done['artifacts_sha256'].items():
        if digest(source/file)!=sha:raise ValueError('Audited source CSV changed')
    output.mkdir(parents=True)
    metrics=pd.read_csv(source/'metrics.csv');curves=pd.read_csv(source/'curves.csv');episodes=pd.read_csv(source/'episodes.csv')
    ids=metrics.case_id.drop_duplicates().tolist();yy=np.arange(len(ids))
    fig,axes=plt.subplots(1,3,figsize=(7.2,4.5),sharey=True)
    fig.subplots_adjust(left=.17,right=.98,top=.90,bottom=.25,wspace=.38)
    offsets={'sampled':-.21,'filtered':-.07,'lag':.07,'full':.21}
    settings=[('sampled','sampled','filtered'),('filtered','marginal','filtered'),('lag','marginal','lag'),('full','marginal','full')]
    for name,method,mode in settings:
        select=metrics.loc[metrics.method.eq(method)&metrics.particles.eq(512)&metrics['mode'].eq(mode)&metrics.window.eq('entry_window')]
        presence=select.loc[select.state.eq('present')].set_index('case_id').reindex(ids)
        error=select.loc[select.state.eq('error_available_other')].set_index('case_id').reindex(ids)
        axes[0].plot(presence.brier,yy+offsets[name],'o',color=COLORS[name],ms=3.5)
        axes[2].plot(error.true_event_probability,yy+offsets[name],'o',color=COLORS[name],ms=3.5)
    for name,method,mode in [('sampled','sampled','full'),('lag','marginal','lag'),('full','marginal','full')]:
        select=curves.loc[curves.method.eq(method)&curves.particles.eq(512)&curves['mode'].eq(mode)&curves.in_entry_window]
        ess=select.groupby('case_id').mean_ancestral_ess.median().reindex(ids)
        if not (ess>0).all():raise ValueError('Ancestor ESS must be positive before log scaling')
        axes[1].plot(ess,yy+offsets[name],'o',color=COLORS[name],ms=3.5)
    axes[0].set_title('a  Presence recovery',loc='left',fontweight='bold');axes[0].set_xlabel('Brier score');axes[0].set_xlim(left=0)
    axes[1].set_title('b  Window genealogy',loc='left',fontweight='bold');axes[1].set_xlabel('Median ancestor ESS')
    axes[1].set_xscale('log');axes[1].set_xlim(.8,600);axes[1].set_xticks([1,10,100],['1','10','100'])
    axes[2].set_title('c  Key-error probability',loc='left',fontweight='bold');axes[2].set_xlabel('True-class probability');axes[2].set_xlim(0,1)
    for ax in axes:ax.grid(axis='x',color='.92',lw=.6)
    count=metrics.loc[metrics.method.eq('sampled')&metrics['mode'].eq('filtered')&metrics.window.eq('entry_window')&metrics.state.eq('error_available_other')].set_index('case_id').reindex(ids).positives.to_numpy(int)
    axes[0].set_yticks(yy,[f'{cid}   n={n}' for cid,n in zip(ids,count)]);axes[0].set_ylim(len(ids)-.5,-.5)
    fig.legend(handles=[Line2D([],[],ls='',marker='o',color=color,label=LABELS[name]) for name,color in COLORS.items()],
        loc='lower center',bbox_to_anchor=(.54,.13),ncol=4)
    fig.text(.17,.025,'512 particles; 3 seeds averaged equally; n = true key errors in each window.\na,c: sampled baseline = filtering; b: sampled baseline = full smoothing.\nAll six zero-inertia cases retained; parameters known; oral validation deferred.',fontsize=7)
    save(fig,output,'marginal_summary')

    fig,axes=plt.subplots(len(ids),1,figsize=(7.2,7.3),sharex=True)
    fig.subplots_adjust(left=.10,right=.98,top=.95,bottom=.18,hspace=.77)
    for i,cid in enumerate(ids):
        ax=axes[i];own=curves.loc[curves.case_id.eq(cid)&curves.in_entry_window]
        base=own.loc[own.method.eq('sampled')&own['mode'].eq('filtered')].sort_values('trial')
        ep=episodes.loc[episodes.case_id.eq(cid)&episodes.method.eq('sampled')&episodes['mode'].eq('filtered')&episodes.episode.eq(1)].iloc[0]
        entry=int(ep.entry_trial);ax.step(base.trial-entry,base.true_present.astype(int),where='post',color='#202020',lw=1.0)
        for name,method,mode in settings:
            levels=[512] if method=='sampled' else sorted(own.loc[own.method.eq(method)].particles.unique())
            for level in levels:
                if level==1024 and mode=='filtered':continue
                group=own.loc[own.method.eq(method)&own['mode'].eq(mode)&own.particles.eq(level)].sort_values('trial')
                ax.plot(group.trial-entry,group.presence,color=COLORS[name],lw=.9,ls='-' if level==512 else '--')
        ax.set_ylim(-.04,1.06);ax.set_yticks([0,1]);ax.axvline(0,color='.7',ls=':',lw=.6)
        ax.set_title(f'{chr(97+i)}  {cid}, subject {int(base.subject.iloc[0])}; first true episode: {int(ep.true_length)} trials',loc='left',fontweight='bold')
    axes[-1].set_xlabel('Trials relative to first true entry')
    handles=[Line2D([],[],color='#202020',label='True presence')]+[Line2D([],[],color=color,label=LABELS[name]) for name,color in COLORS.items()]
    handles.append(Line2D([],[],color='#202020',ls='--',label='1024 particles (dashed)'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.075),ncol=3)
    fig.text(.10,.018,'Solid lines: 512 particles; dashed marginal smoothers: C006, C008 and C013 only.\nStates before current feedback; smoothers use future choices; 3 seeds per setting.',fontsize=7)
    save(fig,output,'entry_windows')
    write(output/'manifest.json',dict(status='rendered_pending_visual_review',backend='python',dpi=300,
        summary_completion_sha256=digest(source/'completion.json'),source_sha256=digest(Path(__file__)),
        files=[p.name for p in sorted(output.glob('*.png'))],export_policy='PNG only',oral_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.source,args.output)
