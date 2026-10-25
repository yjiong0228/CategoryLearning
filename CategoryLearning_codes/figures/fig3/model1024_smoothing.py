"""Plot audited exact-reference errors, recovery and ancestry diversity.

The adjacent contract defines all panels and their observation conditioning.
Exports are PNG only under the repository's standing user instructions.
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

from src.Bayesian_state.workflows.analysis.model_1024_smoothing import read,write,digest

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'legend.fontsize':6,'legend.frameon':False,'axes.spines.right':False,'axes.spines.top':False,
    'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
COLORS={'filtered':'#39769C','lag':'#818A91','prefix':'#B77E40','full':'#8D6C9F'}
LABELS={'filtered':'Online filter','lag':'Lag 32','prefix':'Prefix endpoint','full':'Full endpoint'}


def save(fig,output:Path,name:str):
    fig.savefig(output/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def run(source:Path,output:Path)->None:
    if output.exists():raise FileExistsError('Use a new figure output directory')
    done=read(source/'completion.json')
    if done['status']!='complete' or done['smoke']:raise ValueError('Require a complete formal smoothing summary')
    for file,sha in done['artifacts_sha256'].items():
        if digest(source/file)!=sha:raise ValueError('Audited source table changed')
    output.mkdir(parents=True)
    benchmark=pd.read_csv(source/'benchmark_metrics.csv');bc=pd.read_csv(source/'benchmark_curves.csv')
    curves=pd.read_csv(source/'curves.csv');metrics=pd.read_csv(source/'metrics.csv')
    functionals=pd.read_csv(source/'functionals.csv');episodes=pd.read_csv(source/'episodes.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1))
    fig.subplots_adjust(left=.10,right=.98,top=.93,bottom=.24,hspace=.70,wspace=.35)
    chosen=bc.loc[bc.length.eq(128)]
    for i,mode in enumerate(('filtered','smoothed')):
        ax=axes[i,0];first=chosen.loc[chosen.particles.eq(4096)&chosen.seed.eq(chosen.seed.min())].sort_values('trial')
        ax.plot(first.trial,first['exact_'+mode],color='#202020',lw=1.1,label='Exact')
        for n,color in [(64,'#818A91'),(4096,'#39769C')]:
            group=chosen.loc[chosen.particles.eq(n)].groupby('trial')[mode]
            mean=group.mean();spread=group.std()/np.sqrt(group.count())
            ax.fill_between(mean.index,np.maximum(0,mean-2*spread),np.minimum(1,mean+2*spread),color=color,alpha=.15,lw=0)
            ax.plot(mean.index,mean,color=color,lw=.8,label=f'{n} particles')
        ax.set_ylim(-.03,1.03);ax.set_ylabel('Rule 3 probability');ax.set_xlabel('Trial')
        ax.set_title(f'{chr(97+2*i)}  '+('Filtering, 128 trials' if mode=='filtered' else 'Smoothing, 128 trials'),loc='left',fontweight='bold')
    for i,length in enumerate((8,128)):
        ax=axes[i,1]
        for mode,color in [('filtered',COLORS['filtered']),('smoothed',COLORS['full'])]:
            group=benchmark.loc[benchmark.length.eq(length)&benchmark['mode'].eq(mode)].groupby('particles').mean_absolute_error
            mean=group.mean();se=group.std()/np.sqrt(group.count())
            ax.errorbar(mean.index,mean,yerr=2*se,fmt='o-',color=color,ms=3,lw=.9,capsize=2,label=mode.capitalize())
        ax.set_xscale('log',base=2);ax.set_xticks([64,512,4096],['64','512','4096'])
        ax.set_ylim(bottom=0);ax.set_xlabel('Particles');ax.set_ylabel('Mean absolute error')
        ax.set_title(f'{chr(98+2*i)}  Probability error, {length} trials',loc='left',fontweight='bold')
        if i==0:ax.legend(loc='upper right',fontsize=6)
    fig.legend(handles=[Line2D([],[],color=color,label=label) for color,label in
        [('#202020','Exact'),('#818A91','64 particles'),('#39769C','4096 particles')]],
        loc='lower center',bbox_to_anchor=(.40,.14),ncol=3)
    fig.text(.10,.035,'Finite-state execution diagnostic: fixed active rules, beliefs and choice likelihoods.\n4 seeds per setting; bands and bars = +/-2 numerical MCSE; all trials retained.\nRules 1 and 2 have identical emissions. This benchmark does not include rule discovery.',fontsize=7)
    save(fig,output,'exact_benchmark')

    ids=curves.case_id.drop_duplicates().tolist()
    fig,axes=plt.subplots(len(ids),1,figsize=(7.2,7.6),sharex=True)
    fig.subplots_adjust(left=.10,right=.98,top=.95,bottom=.16,hspace=.78)
    for i,cid in enumerate(ids):
        ax=axes[i];own=curves.loc[curves.case_id.eq(cid)&curves.in_entry_window]
        first=own.loc[own['mode'].eq('filtered')].sort_values('trial')
        ep=episodes.loc[episodes.case_id.eq(cid)&episodes['mode'].eq('filtered')&episodes.episode.eq(1)].iloc[0]
        entry=int(ep.entry_trial)
        ax.step(first.trial-entry,first.true_present.astype(int),where='post',color='#202020',lw=1.0)
        for mode in COLORS:
            group=own.loc[own['mode'].eq(mode)].sort_values('trial')
            ax.plot(group.trial-entry,group.presence,color=COLORS[mode],lw=.9)
        ax.axvline(0,color='.7',ls=':',lw=.6);ax.set_ylim(-.04,1.06);ax.set_yticks([0,1])
        scope='; enriched' if str(first.scope.iloc[0])!='fixed' else ''
        ax.set_title(f'{chr(97+i)}  {cid}, subject {int(first.subject.iloc[0])}; first true episode: {int(ep.true_length)} trials{scope}',
            loc='left',fontweight='bold')
    axes[-1].set_xlabel('Trials relative to first true entry')
    fig.legend(handles=[Line2D([],[],color='#202020',label='True presence')]+
        [Line2D([],[],color=color,label=LABELS[mode]) for mode,color in COLORS.items()],
        loc='lower center',bbox_to_anchor=(.54,.075),ncol=5)
    fig.text(.10,.025,'512 particles; 3 seeds averaged equally; all 7 prespecified entry windows.\nStates refer to before current feedback. Smoothers use future choices; oral validation is deferred.',fontsize=7)
    save(fig,output,'entry_windows')

    fig,axes=plt.subplots(1,3,figsize=(7.2,4.5),sharey=True)
    fig.subplots_adjust(left=.11,right=.98,top=.89,bottom=.24,wspace=.40)
    yy=np.arange(len(ids));offsets={'filtered':-.21,'lag':-.07,'prefix':.07,'full':.21}
    for mode,color in COLORS.items():
        group=metrics.loc[metrics['mode'].eq(mode)&metrics.window.eq('entry_window')&metrics.state.eq('present')].set_index('case_id').reindex(ids)
        axes[0].plot(group.brier,yy+offsets[mode],'o',color=color,ms=3,label=LABELS[mode])
        if mode!='filtered':
            own=curves.loc[curves['mode'].eq(mode)&curves.in_entry_window].groupby('case_id').mean_ancestral_ess.median().reindex(ids)
            axes[1].plot(own,yy+offsets[mode],'o',color=color,ms=3)
    axes[0].set_xlabel('Brier score');axes[0].set_xlim(left=0)
    axes[0].set_title('a  Presence recovery',loc='left',fontweight='bold')
    axes[1].set_xlabel('Median ancestor ESS');axes[1].set_xscale('log');axes[1].set_xlim(1,512)
    axes[1].set_xticks([1,10,100],['1','10','100']);axes[1].axvline(1,color='.7',ls=':',lw=.7)
    axes[1].set_title('b  Window genealogy',loc='left',fontweight='bold')
    own=functionals.loc[functionals['mode'].eq('full')&functionals.function.eq('presence_trials')]
    for i,cid in enumerate(ids):
        group=own.loc[own.case_id.eq(cid)].sort_values('seed')
        for shift,row in zip([-.14,0,.14],group.itertuples()):
            axes[2].plot([row.q05,row.q95],[i+shift,i+shift],color=COLORS['full'],lw=.8,alpha=.65)
            axes[2].plot(row.posterior_mean,i+shift,'o',color=COLORS['full'],ms=2.5)
        axes[2].plot(group.true_value.iloc[0],i,'|',color='#202020',ms=12,mew=1.2)
    axes[2].set_xlabel('Trials with target rule');axes[2].set_xlim(left=0)
    axes[2].set_title('c  Full-path occupancy',loc='left',fontweight='bold')
    for ax in axes:ax.grid(axis='x',color='.92',lw=.6)
    axes[0].set_yticks(yy,ids);axes[0].set_ylim(len(ids)-.5,-.5)
    fig.legend(handles=[Line2D([],[],ls='',marker='o',color=color,label=LABELS[mode]) for mode,color in COLORS.items()],
        loc='lower center',bbox_to_anchor=(.54,.115),ncol=4)
    fig.text(.11,.025,'a,b: common entry windows; b: median of the mean ESS across 3 seeds.\nc: each seed shown; terminal-path mean and 5-95% interval; black mark = truth.\nIntervals are finite-particle approximations; assess them with ancestry and seed stability.',fontsize=7)
    save(fig,output,'ancestry_and_scores')
    write(output/'manifest.json',dict(status='rendered_pending_visual_review',backend='python',dpi=300,
        source_sha256=digest(Path(__file__)),summary_completion_sha256=digest(source/'completion.json'),
        files=[p.name for p in sorted(output.glob('*.png'))],export_policy='PNG only',oral_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.source,args.output)
