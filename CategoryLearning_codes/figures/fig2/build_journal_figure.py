"""Restore the original Fig2 architecture using the latest cohort's real results."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from .framework_wide import draw_framework, illustrative_episode
from .build_belief_validation import ORDER, compatible
from ..fig3.bottleneck_analysis import ROOT, sha256

TASK_COLORS={1:'#487DA8',2:'#A46A8A',3:'#648B76'}
QCOLOR='#167D8D'
DEFAULT_STATES='results/model_0826/fig34_twelve_subjects_20260921_v1'
DEFAULT_VALIDATION='CategoryLearning_codes/figures/outputs/fig2/belief_validation_20260921_v2'
EXAMPLES=[102,307,221]


def style() -> None:
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':6.5,'axes.titlesize':7,'axes.labelsize':6.5,'axes.linewidth':.6,
        'xtick.labelsize':6,'ytick.labelsize':6,'xtick.major.size':2,'ytick.major.size':2,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'legend.fontsize':5.8,'svg.fonttype':'none','pdf.fonttype':42})


def panel(fig, letter: str, title: str, x: float, y: float, color: str='.15') -> None:
    fig.text(x-.028,y,letter,weight='bold',fontsize=9,va='bottom')
    fig.text(x,y,title,fontsize=7,weight='bold',va='bottom',color=color)


def rolling(values: np.ndarray | pd.Series, minimum: int=32) -> np.ndarray:
    return pd.Series(np.asarray(values,float)).rolling(32,min_periods=minimum).mean().to_numpy()


def build(states: Path, validation: Path, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False)
    people=pd.read_csv(validation/'subjects.csv').set_index('subject').loc[ORDER]
    trials=pd.read_csv(validation/'trials.csv')
    cal=pd.read_csv(validation/'calibration.csv')
    inputs=[validation/x for x in ['subjects.csv','trials.csv','calibration.csv','manifest.json']]
    previous=json.loads((validation/'manifest.json').read_text())
    if previous['n_participants']!=12 or previous['oral_sigma']!=.05:raise ValueError('Validation source mismatch')
    if len(trials)!=7936 or people.n_reports.sum()!=7600:raise ValueError('Unexpected cohort')
    for name in ['subjects','trials','calibration']:
        shutil.copy2(validation/f'{name}.csv',output/f'{name}.csv')
    style();fig=plt.figure(figsize=(183/25.4,230/25.4))
    # Reuse the original native model diagram, including its explicit toy-example label.
    draw_framework(fig,[.07,.654,.90,.327])
    blue_map=LinearSegmentedColormap.from_list('belief',['#FFFFFF','#9DC6CE','#167D8D','#11444E'])
    oral_map=LinearSegmentedColormap.from_list('report',['#FFFFFF','#C4C4C4','#252525'])
    oral_map.set_bad('#E5E8EB')
    source_rows=[]
    for j,sid in enumerate(EXAMPLES):
        g=trials[trials.iSub.eq(sid)].reset_index(drop=True);n=len(g);task=int(g.task.iloc[0])
        path=states/f'base_analysis/S{sid}_rule_distributions.npz';inputs.append(path)
        with np.load(path,allow_pickle=False) as z:
            q=z['belief'];o=z['instantaneous_oral'];valid=z['oral_valid'].astype(bool)
        np.testing.assert_allclose(q.sum(1),1,atol=1e-7)
        np.testing.assert_array_equal(valid,g.oral_valid)
        report=np.full_like(o,np.nan);report[valid]=o[valid]/o[valid].max(1,keepdims=True)
        scoremask=valid&g.scored.to_numpy(bool)
        np.testing.assert_allclose(compatible(q[scoremask],o[scoremask]),g.compatibility[scoremask])
        nrules=q.shape[1];x=.085+j*.312;width=.246
        panel(fig,chr(ord('b')+j),f'Task {task} · S{sid}',x,.615,TASK_COLORS[task])
        a=fig.add_axes([x,.536,width,.060])
        a.plot(g.trial,rolling(g.label_correct),color='#333B43',lw=.9)
        a.plot(g.trial,rolling(g.model_correct),color='#617EA0',lw=.9,ls='--')
        a.set(xlim=(1,n),ylim=(0,1.04),yticks=[0,.5,1]);a.tick_params(axis='x',bottom=False,labelbottom=False)
        if j==0:
            a.set_ylabel('Accuracy')
            a.legend(handles=[Line2D([],[],color='#333B43',lw=.9,label='Observed'),
                              Line2D([],[],color='#617EA0',lw=.9,ls='--',label='Model')],
                     loc='lower right',fontsize=5.5,ncol=2,handlelength=1.3,columnspacing=.7,borderpad=.1)
        matrices=[(.432,q.T,'Inferred beliefs',blue_map),(.325,report.T,'Report support',oral_map)]
        for row,(bottom,matrix,title,cmap) in enumerate(matrices):
            a=fig.add_axes([x,bottom,width,.073])
            im=a.imshow(matrix,aspect='auto',origin='upper',interpolation='nearest',cmap=cmap,vmin=0,vmax=1,
                        extent=(.5,n+.5,nrules-.5,-.5))
            a.set_title(title,loc='left',fontsize=6.5,pad=4)
            target=0 if nrules==29 else 42
            ticks=[0,14,28] if nrules==29 else [0,42,115]
            a.set(yticks=ticks,yticklabels=[f'H{k}' for k in ticks],xticks=[])
            a.tick_params(axis='y',length=2,pad=2,labelsize=5.6)
            # Target marker sits outside the heatmap; no added stripe resembles data.
            a.plot(-.015,target,marker='>',ms=2.5,color=QCOLOR,transform=a.get_yaxis_transform(),clip_on=False)
            if j==2:
                cbax=fig.add_axes([.965,bottom,.006,.073])
                cb=fig.colorbar(im,cax=cbax,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=5,length=1,pad=2)
                cbax.set_title(r'$Q$' if row==0 else r'$L/L_{\max}$',fontsize=5.4,pad=4)
        a=fig.add_axes([x,.238,width,.051])
        a.plot(g.trial,rolling(g.compatibility,16),color=QCOLOR,lw=.9)
        a.plot(g.trial,rolling(g.uniform_compatibility,16),color='.5',lw=.7,ls=':')
        a.set(xlim=(1,n),ylim=(-.015,1.02),yticks=[0,.5,1],xticks=[1,n//2,n],xlabel='Trial')
        if j==0:a.set_ylabel('Report\ncompatibility',labelpad=3)
        else:a.set_ylabel('')
        for h in range(nrules):
            source_rows.append(pd.DataFrame({'subject':sid,'task':task,'trial':g.trial,'hypothesis':h,
                                            'belief':q[:,h],'report_relative_support':report[:,h]}))
    # Cohort summaries mirror the original e–g row, now populated with real data.
    panel(fig,'e','Choice calibration',.085,.182)
    panel(fig,'f','Rule-content correspondence',.397,.182)
    panel(fig,'g','Report-change correspondence',.709,.182)
    axes=[fig.add_axes([.085+j*.312,.051,.246,.109]) for j in range(3)]
    a=axes[0];a.plot([0,1],[0,1],ls=':',color='.65',lw=.7)
    for task in [1,2,3]:
        s=cal[cal.task.eq(task)]
        a.scatter(s.predicted,s.observed,s=9,edgecolors=TASK_COLORS[task],facecolors='none',lw=.6,alpha=.55)
    means=[]
    for _,s in cal.groupby('bin'):
        means.append([np.average(s.predicted,weights=s.n),np.average(s.observed,weights=s.n)])
    means=np.array(means);a.plot(means[:,0],means[:,1],'o-',color='.2',ms=2.5,lw=.9)
    a.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted accuracy',ylabel='Observed accuracy')
    for sid,s in people.iterrows():
        task=int(s.task);c=TASK_COLORS[task]
        a=axes[1];a.plot([0,1,2],[s.uniform_compatibility,s.static_compatibility,s.compatibility],
                       color=c,lw=.55,alpha=.7,marker='o',ms=2.7,mew=0)
        a=axes[2];a.plot([0,1],[s.stable_change,s.report_change],color=c,lw=.6,alpha=.7,marker='o',ms=2.7,mew=0)
    axes[1].set(xlim=(-.23,2.23),ylim=(-.01,.48),yticks=[0,.2,.4],xticks=[0,1,2],
                xticklabels=['Uniform','Static','Dynamic'],ylabel='Report compatibility')
    change_max=float(people[['stable_change','report_change']].max().max())
    change_limit=float(np.ceil(change_max*1.12/.05)*.05)
    axes[2].set(xlim=(-.2,1.2),ylim=(-.002,change_limit),yticks=[0,.1,.2],xticks=[0,1],
                xticklabels=['Stable','Changed'],ylabel='Belief change / trial')
    fig.savefig(output/'Figure2_journal.png',dpi=450,facecolor='white');plt.close(fig)
    pd.concat(source_rows,ignore_index=True).to_csv(output/'case_rule_distributions.csv',index=False)
    (output/'schematic_episode.json').write_text(json.dumps(illustrative_episode(),indent=2)+'\n')
    code=Path(__file__).resolve()
    source_files=[code,code.with_name('framework_wide.py'),code.with_name('build_belief_validation.py'),code.with_name('JOURNAL_FIGURE.md')]
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for p in source_files:shutil.copy2(p,snapshot/p.name)
    inputs+=source_files
    manifest={'examples':EXAMPLES,'n_participants':12,'n_trials':7936,'n_reports':7600,
              'width_mm':183,'height_mm':230,'dpi':450,'python':platform.python_version(),
              'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,
              'state_timing':'pre-choice','oral':'current report only; likelihood relative to trial maximum',
              'fit_scope':'full-sequence choice fitting; not held-out prediction',
              'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in inputs}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output.relative_to(ROOT)),'examples':EXAMPLES,'n_reports':7600}))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states',type=Path,default=Path(DEFAULT_STATES))
    parser.add_argument('--validation',type=Path,default=Path(DEFAULT_VALIDATION))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.states.resolve(),args.validation.resolve(),args.output.resolve())


if __name__=='__main__':main()
