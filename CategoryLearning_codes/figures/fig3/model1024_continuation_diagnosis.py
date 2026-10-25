"""Draw all frozen continuation cases and planned oracle contrasts as PNGs.

See the adjacent figure contract. Use --source COMPLETE_SUMMARY --output NEW.
Colors map to the same intervention in every panel; all modes and candidates
enter aggregate panels. Focus panels are the three predeclared selected STICKY
points. Uncertainty is numerical error or process variation as labeled.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1024_continuation_diagnosis import read,write,digest

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'axes.titlesize':8,'axes.spines.right':False,'axes.spines.top':False,
    'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
COLORS={'BASE':'#626971','AVAILABLE':'#39769C','AVAILABLE_PRECISION':'#288477',
        'AVAILABLE_EXECUTE_PRECISION':'#88629A'}
LABELS={'BASE':'Original continuation','AVAILABLE':'Available target',
        'AVAILABLE_PRECISION':'Available + precision','AVAILABLE_EXECUTE_PRECISION':'Available + execute + precision'}
ARM_SHORT=['Protect','Enter','Available','Precision','Avail.\n+ precision','Avail.\n+ execute','Avail.\n+ exec. + prec.']
ARMS=['RETAIN','DISCOVER','AVAILABLE','PRECISION','AVAILABLE_PRECISION','AVAILABLE_EXECUTE','AVAILABLE_EXECUTE_PRECISION']
CUTS=['half_training_block','training_end']


def order(frame: pd.DataFrame) -> list[tuple]:
    x=frame[['subject','mode','candidate','is_original']].drop_duplicates()
    return list(x.sort_values(['subject','mode','is_original'],ascending=[True,True,False]).itertuples(index=False,name=None))


def label(key: tuple) -> str:
    sid,mode,_,original=key
    return f'{sid} {mode}'+((' original' if original else ' joint') if sid==317 else '')


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def draw(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new figure directory')
    done=read(source/'completion.json')
    if done['status']!='complete': raise ValueError('Require complete audited summary')
    for name,sha in done['artifacts_sha256'].items():
        if digest(source/name)!=sha: raise ValueError('Source table changed')
    output.mkdir(parents=True)
    scores=pd.read_csv(source/'scores.csv'); end=pd.read_csv(source/'endpoints.csv')
    start=pd.read_csv(source/'task_start.csv'); curves=pd.read_csv(source/'curves.csv')
    cases=order(scores); n=len(cases)
    def own(frame,key):
        return frame[(frame.subject==key[0]) & (frame['mode']==key[1]) & (frame.candidate==key[2])]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),gridspec_kw={'width_ratios':[1,1.4]})
    fig.subplots_adjust(left=.23,right=.98,bottom=.23,top=.88,wspace=.25)
    for i,key in enumerate(cases):
        e=own(end,key); s=own(scores,key); t=own(start,key)
        for cut,color,offset in zip(CUTS,['#39769C','#88629A'],[-.12,.12]):
            values=e[e.cut_label==cut].present
            axes[0].plot(values.mean(),i+offset,'o',ms=4,color=color)
            row=s[(s.cut_label==cut)&(s.arm=='BASE')].iloc[0]
            axes[1].errorbar(row.late_accuracy,i+offset,xerr=2*row.late_accuracy_mc_se,fmt='o',ms=4,
                             color=color,lw=.8,capsize=1.5)
        row=t.iloc[0]
        axes[1].errorbar(row.late_accuracy,i-.32,xerr=2*row.late_accuracy_mc_se,fmt='s',ms=3,
                         color='#9CA2A8',lw=.7,capsize=1.5)
        axes[1].plot(row.observed_late,i,'|',color='black',ms=11,mew=1.1)
    for ax in axes:
        ax.set_ylim(n-.5,-.5); ax.set_yticks(range(n)); ax.grid(axis='x',color='.91',lw=.5)
        ax.set_xlim(-.02,1.02); ax.set_xticks([0,.25,.5,.75,1])
    axes[0].set_yticklabels([label(k) for k in cases]); axes[1].set_yticklabels([])
    axes[0].set_xlabel('Target present in prefix cloud'); axes[1].set_xlabel('Final 64-trial accuracy')
    axes[0].set_title('a  Starting state',loc='left',fontweight='bold')
    axes[1].set_title('b  Autonomous performance',loc='left',fontweight='bold')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker=m,color=c,ls='',label=l,ms=4) for m,c,l in
        [('s','#9CA2A8','Task start'),('o','#39769C','Mid-training prefix'),('o','#88629A','Training-end prefix'),
         ('|','black','Observed')]],loc='lower center',ncol=2,bbox_to_anchor=(.55,.025),fontsize=7)
    save(fig,output,'continuation_baselines')

    matrices=[]
    for cut in CUTS:
        matrix=[]
        for key in cases:
            s=own(scores,key); s=s[s.cut_label==cut].set_index('arm')
            matrix.append([100*(s.loc[a,'late_accuracy']-s.loc['BASE','late_accuracy']) for a in ARMS])
        matrices.append(np.asarray(matrix))
    limit=max(1.,max(np.abs(m).max() for m in matrices))
    fig,axes=plt.subplots(2,1,figsize=(7.2,6.2))
    fig.subplots_adjust(left=.26,right=.91,bottom=.16,top=.93,hspace=.38)
    for ax,matrix,title,panel in zip(axes,matrices,['Mid-training prefix','Training-end prefix'],['a','b']):
        im=ax.imshow(matrix,vmin=-limit,vmax=limit,cmap='BrBG',aspect='auto')
        ax.set_xticks(range(len(ARMS)),ARM_SHORT); ax.set_yticks(range(n),[label(k) for k in cases])
        ax.set_title(f'{panel}  {title}',loc='left',fontweight='bold',pad=7)
        ax.tick_params(length=0)
        for y in range(n):
            for x in range(len(ARMS)):
                value=matrix[y,x]
                ax.text(x,y,f'{value:+.1f}',ha='center',va='center',fontsize=6,
                        color='white' if abs(value)>.65*limit else '#252525')
    color_ax=fig.add_axes([.94,.26,.014,.49])
    fig.colorbar(im,cax=color_ax,label='Accuracy change (percentage points)')
    fig.text(.26,.035,'All changes relative to the same starting cloud and BASE path bank.\n96 paths per cell; six independent prefix filters. Oracle probes; fixed parameters.',fontsize=7)
    save(fig,output,'continuation_interventions')

    fig,axes=plt.subplots(3,2,figsize=(7.2,6.1))
    fig.subplots_adjust(left=.09,right=.98,bottom=.15,top=.94,wspace=.22,hspace=.46)
    for r,sid in enumerate((129,229,317)):
        rows=curves[(curves.subject==sid)&(curves['mode']=='STICKY')&curves.is_selected &
                    (curves.cut_label=='half_training_block')]
        base=rows[rows.arm=='BASE']; x=base.trial.to_numpy()
        axes[r,0].fill_between(x,base.q10,base.q90,color=COLORS['BASE'],alpha=.12,lw=0)
        axes[r,0].plot(x,base.observed,color='black',lw=1.2,label='Observed')
        for arm,color in COLORS.items():
            q=rows[rows.arm==arm]
            axes[r,0].plot(q.trial,q['mean'],color=color,lw=1.,label=LABELS[arm])
            axes[r,1].plot(q.trial,q.contribution,color=color,lw=1.)
        for c in range(2):
            ax=axes[r,c]; ax.set_ylim(-.03,1.03); ax.set_yticks([0,.5,1]); ax.grid(axis='y',color='.92',lw=.5)
            ax.set_title(f'{chr(97+2*r+c)}  Subject {sid}: '+('block accuracy' if c==0 else 'target contribution'),
                         loc='left',fontweight='bold')
            if r==2: ax.set_xlabel('Trial')
            ax.set_ylabel('Accuracy' if c==0 else 'Choice weight')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.55,.005),ncol=2,fontsize=7)
    save(fig,output,'continuation_focus')
    files=sorted(output.glob('*.png'))
    write(output/'figure_QA.json',dict(status='rendered_pending_visual_review',backend='python',width_inches=7.2,dpi=300,
        source=str(source.resolve()),source_completion_sha256=digest(source/'completion.json'),source_sha256=digest(Path(__file__)),
        participants=3,parameter_mode_points=len(cases),cuts=2,all_arms_in_effect_panels=True,
        focus='Prespecified selected STICKY point for all three participants; all paths included',
        intervals='BASE band central 80% of path block accuracies; error bars +/-2 numerical MCSE',
        inference='Development oracle diagnosis; no empirical causal or population claim',
        artifacts_sha256={p.name:digest(p) for p in files}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); draw(a.source,a.output)
