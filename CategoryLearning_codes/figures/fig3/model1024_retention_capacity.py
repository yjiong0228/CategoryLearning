"""Draw complete two-parameter profiles, planned effects and frozen path means.

Use --source COMPLETE_SUMMARY --output NEW. The adjacent contract defines each
panel, the complete-data rule and numerical versus predictive uncertainty. All
figures are development PNGs under the repository's export policy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1024_retention_capacity import read,write,digest

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'legend.frameon':False,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,
    'svg.fonttype':'none','pdf.fonttype':42})
SUBJECTS=(129,229,317)
MODES=('MIX','STICKY')
DELTA=(0.,.3,.6,.8,1.)
CAPACITY=(2,3,6)
BLUE='#39769C';ORANGE='#B77E40';GREY='#8D949C'


def save(fig,output: Path,name: str):
    fig.savefig(output/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def run(source: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new profile figure directory')
    done=read(source/'completion.json')
    if done['status']!='complete':raise ValueError('Require completed audited summary')
    for name,sha in done['artifacts_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('Audited table changed')
    output.mkdir(parents=True)
    scores=pd.read_csv(source/'summary.csv');curves=pd.read_csv(source/'curves.csv');effects=pd.read_csv(source/'factorial_contrasts.csv')
    eligibility=pd.read_csv(source/'training_profile.csv')[['subject','mode','candidate','eligible']]
    all_profile=scores[scores.in_profile].merge(eligibility,on=['subject','mode','candidate'],validate='one_to_one')
    vmin,vmax=all_profile.full_energy.min(),all_profile.full_energy.max()
    fig,axes=plt.subplots(3,2,figsize=(7.2,6.2))
    fig.subplots_adjust(left=.10,right=.89,top=.94,bottom=.15,hspace=.43,wspace=.26)
    for r,sid in enumerate(SUBJECTS):
        for c,mode in enumerate(MODES):
            ax=axes[r,c];own=all_profile[(all_profile.subject==sid)&(all_profile['mode']==mode)]
            matrix=own.pivot(index='delta',columns='capacity',values='full_energy').reindex(index=DELTA,columns=CAPACITY).to_numpy()
            im=ax.imshow(matrix,cmap='cividis',vmin=vmin,vmax=vmax,aspect='auto')
            ax.set_xticks(range(3),CAPACITY);ax.set_yticks(range(5),[f'{d:g}' for d in DELTA]);ax.tick_params(length=0)
            ax.set_ylabel('Retention');ax.set_xlabel('Capacity')
            ax.set_title(f'{chr(97+r*2+c)}  Subject {sid}, {mode}',loc='left',fontweight='bold')
            for y in range(5):
                for x in range(3):
                    ax.text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center',fontsize=6,
                            color='white' if (matrix[y,x]-vmin)/(vmax-vmin)<.48 else '#202020')
            for point in own[own.eligible].itertuples():
                ax.plot(CAPACITY.index(int(point.capacity))+.30,DELTA.index(float(point.delta))-.28,
                        'o',ms=2.2,mfc='white',mec='#222222',mew=.4)
            for flag,color,lw,offset in [('is_anchor','#161616',1.3,.03),('is_selected',ORANGE,1.8,.11)]:
                point=own[own[flag]].iloc[0];x=CAPACITY.index(int(point.capacity));y=DELTA.index(float(point.delta))
                ax.add_patch(Rectangle((x-.5+offset,y-.5+offset),1-2*offset,1-2*offset,
                    facecolor='none',edgecolor=color,lw=lw))
    colorax=fig.add_axes([.92,.25,.015,.56]);fig.colorbar(im,cax=colorax,label='Full-task Energy (lower is better)')
    fig.legend(handles=[Line2D([],[],color='#161616',lw=1.3,label='Current anchor'),
                        Line2D([],[],color=ORANGE,lw=1.8,label='Training-selected point'),
                        Line2D([],[],ls='',marker='o',ms=3,mfc='white',mec='#222222',label='Screen NLL eligible')],
               loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,fontsize=6)
    save(fig,output,'retention_capacity_profiles')

    fig,axes=plt.subplots(2,2,figsize=(7.2,3.9))
    fig.subplots_adjust(left=.25,right=.97,top=.91,bottom=.18,wspace=.24,hspace=.55)
    names=['retention_only','capacity_only','joint','interaction']
    labels=['Retention: 0 to 0.3','Capacity: 2 to 3','Joint change','Interaction']
    for r,mode in enumerate(MODES):
        own=effects[(effects.subject==317)&(effects['mode']==mode)&(effects.delta_from==0)&
            (effects.delta_to==.3)&(effects.capacity_from==2)&(effects.capacity_to==3)].set_index('contrast').loc[names]
        for c,(key,scale,xlabel) in enumerate([('full_energy',1,'Full-task Energy change'),('late_accuracy',100,'Suffix accuracy change (pp)')]):
            ax=axes[r,c];x=scale*own[key+'_delta'].to_numpy();se=scale*own[key+'_delta_mcse'].to_numpy()
            ax.errorbar(x,np.arange(4),xerr=2*se,fmt='o',color=BLUE,ms=4,capsize=2,lw=.8)
            ax.axvline(0,color='.55',ls='--',lw=.7);ax.set_yticks(range(4),labels if c==0 else [])
            ax.set_ylim(3.5,-.5);ax.set_xlabel(xlabel);ax.grid(axis='x',color='.93',lw=.5)
            ax.set_title(f'{chr(97+2*r+c)}  Subject 317, {mode}',loc='left',fontweight='bold')
    for c in range(2):
        limits=[ax.get_xlim() for ax in axes[:,c]]
        for ax in axes[:,c]:
            ax.set_xlim(min(x[0] for x in limits),max(x[1] for x in limits))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.text(.25,.035,'128 complete paired paths per point; error bars = +/-2 numerical MCSE.\nInteraction is the difference between the joint and summed single changes.',fontsize=7)
    save(fig,output,'retention_capacity_effects')

    fig,axes=plt.subplots(3,2,figsize=(7.2,5.8))
    fig.subplots_adjust(left=.08,right=.98,top=.94,bottom=.13,wspace=.23,hspace=.43)
    for r,sid in enumerate(SUBJECTS):
        for c,mode in enumerate(MODES):
            own=curves[(curves.subject==sid)&(curves['mode']==mode)]
            anchor=own[own.is_anchor];selected=own[own.is_selected];reference=own[own.is_nll_reference]
            ax=axes[r,c];ax.fill_between(anchor.trial,anchor.q10,anchor.q90,color=BLUE,alpha=.12,lw=0)
            ax.plot(anchor.trial,anchor.observed,color='black',lw=1.1)
            # All parameter records stay in the source; identical identities
            # are drawn once and named explicitly in the panel title.
            same=anchor.candidate.iloc[0]==selected.candidate.iloc[0]
            if reference.candidate.iloc[0] not in (anchor.candidate.iloc[0],selected.candidate.iloc[0]):
                ax.plot(reference.trial,reference['mean'],color=GREY,lw=.8,ls=':')
            ax.plot(anchor.trial,anchor['mean'],color=BLUE,lw=1.)
            if not same:ax.plot(selected.trial,selected['mean'],color=ORANGE,lw=1.1)
            ax.axvline(anchor.cut.iloc[0],color='.55',ls='--',lw=.7)
            ax.set_ylim(-.02,1.02);ax.set_yticks([0,.5,1]);ax.grid(axis='y',color='.93',lw=.5)
            ax.set_title(f'{chr(97+2*r+c)}  Subject {sid}, {mode}'+(' (unchanged)' if same else ''),loc='left',fontweight='bold')
            if c==0:ax.set_ylabel('Block accuracy')
            if r==2:ax.set_xlabel('Trial')
    fig.legend(handles=[Line2D([],[],color=c,lw=1.,ls=ls,label=label) for c,ls,label in
        [('black','-','Observed'),(BLUE,'-','Current anchor'),(ORANGE,'-','Training-selected point'),(GREY,':','Original NLL reference')]],
        loc='lower center',bbox_to_anchor=(.53,.012),ncol=2,fontsize=7)
    save(fig,output,'retention_capacity_selected_paths')
    write(output/'figure_QA.json',dict(status='rendered_pending_visual_review',backend='python',width_inches=7.2,dpi=300,
        source=str(source.resolve()),source_completion_sha256=digest(source/'completion.json'),source_sha256=digest(Path(__file__)),
        profile_coverage='All 90 profile points, three participants, two modes',
        intervals='Effect bars +/-2 complete-path MCSE; anchor band central 80% of complete-path block accuracy',
        scope='Fixed-nuisance two-coordinate development profile; no recovery or population claim',
        artifacts_sha256={p.name:digest(p) for p in output.glob('*.png')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
