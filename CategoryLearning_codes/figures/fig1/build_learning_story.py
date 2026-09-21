"""Behavior-only Fig1 linking the 96-participant overview to the Fig3 story.

Run from the repository root with a NEW --output directory. PNG only.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .schematics import stimulus_panel, task_panel
from ..fig3.abstract_story_analysis import behavior_shapes
from ..fig3.bottleneck_analysis import ROOT, sha256
from ..fig3.build_nine_subject_figures import style, TASK_COLORS

DEFAULT_SOURCE = 'CategoryLearning_codes/figures/outputs/fig1/fig1_v10'


def heading(fig, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x-.027, y, letter, fontsize=9, weight='bold', va='bottom')
    fig.text(x, y, title, fontsize=7.3, weight='bold', va='bottom')


def behavior_table(trials: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Keep all people; unavailable shape estimates remain missing, never zero."""
    rows = []
    for sid, group in trials.groupby('iSub', sort=True):
        y = group.correct.to_numpy(float)
        if not np.array_equal(group.trial, np.arange(1, len(group)+1)):
            raise ValueError('Trial sequence is not contiguous')
        if len(y) >= 128:
            shape, _ = behavior_shapes(y)
        else:
            shape = {'preferred': 'unavailable', 'delta_bic': np.nan}
        rows.append({'iSub': sid, **shape})
    result = summary.merge(pd.DataFrame(rows), on='iSub', validate='one_to_one')
    result['task'] = result.condition.map({1: 1, 3: 2, 2: 3})
    result['criterion_reached'] = result.first_crossing64.notna()
    result['end_below_criterion'] = result.last64_accuracy.lt(.9)
    return result


def render(trials: pd.DataFrame, people: pd.DataFrame, tasks: list[dict], output: Path) -> None:
    style()
    fig = plt.figure(figsize=(183/25.4, 225/25.4))
    fig.text(.075,.978,'Fig. 1 | Learning differs in timing, shape and later performance',
             fontsize=10, weight='bold', va='top')
    fig.text(.075,.952,'96 participants  ·  32 per task  ·  62,720 recorded trials  ·  behavior only',
             fontsize=6.8, color='.4')
    heading(fig,.075,.915,'a','A shared stimulus, three learning tasks')
    stimulus_panel(fig.add_axes([.075,.787,.19,.12]))
    for k, task in enumerate(tasks):
        task_panel(fig.add_axes([.30+k*.23,.787,.19,.12]), task)
    fig.text(.075,.762,'Features were counterbalanced; category boundaries at 0.5. Partial feedback in Task 2 indicates a correct category pair.',
             fontsize=5.9, color='.4')
    heading(fig,.075,.722,'b','Report the current rule before seeing feedback')
    ax=fig.add_axes([.075,.665,.90,.047]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for j,(title,sub) in enumerate([('Stimulus','varying feature lengths'),('Choice','select a category'),
                                  ('Verbal report','describe the chosen category'),('Feedback','evaluate the choice')]):
        x=.01+j*.253
        ax.text(x,.74,title,weight='bold',fontsize=7)
        ax.text(x,.16,sub,fontsize=5.8,color='.4')
        if j<3:ax.annotate('',xy=(x+.233,.6),xytext=(x+.186,.6),arrowprops={'arrowstyle':'->','lw':.8,'color':'.4'})
    heading(fig,.075,.637,'c','All individual learning records')
    cmap=plt.get_cmap('cividis').copy();cmap.set_bad('#EEEEEE')
    xmax=int(people.n_trials.max())
    for j,task in enumerate(tasks):
        x=.075+j*.312
        ax=fig.add_axes([x,.477,.276,.126])
        sub=people[people.task.eq(task['task'])].sort_values(['n_trials','iSub'])
        matrix=np.full((len(sub),xmax),np.nan)
        for row,sid in enumerate(sub.iSub):
            vals=trials.loc[trials.iSub.eq(sid),'rolling_accuracy'].to_numpy()
            matrix[row,:len(vals)]=vals
        im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest',
                     extent=(.5,xmax+.5,32.5,.5))
        ax.set_title(f"Task {task['task']}  ·  n = 32",color=task['color'],pad=4)
        ax.set(yticks=[1,16,32],xticks=[])
        if j==0:ax.set_ylabel('Participant\n(by record length)')
        else:ax.set_yticklabels([])
        cx=fig.add_axes([x,.447,.276,.019])
        counts=(np.arange(1,xmax+1)[None,:]<=sub.n_trials.to_numpy()[:,None]).sum(0)
        cx.fill_between(np.arange(1,xmax+1),counts,color=task['color'],alpha=.3,lw=0)
        cx.set(xlim=(1,xmax),ylim=(0,32),yticks=[0,32],xticks=[1,512,1024,xmax])
        cx.tick_params(labelsize=5.5,pad=1)
        if j==0:cx.set_ylabel('n',rotation=0,labelpad=4)
    cax=fig.add_axes([.805,.633,.16,.006])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.5,1]);cb.ax.tick_params(labelsize=5.5,length=1,pad=1)
    cax.set_title('Accuracy (trailing 32)',fontsize=5.8,pad=3)
    fig.text(.52,.421,'Recorded trial  ·  gray: no estimate / no recorded trial',ha='center',fontsize=5.8,color='.4')
    heading(fig,.075,.388,'d','Three examples carried forward to the model analysis')
    examples=[(122,'Early improvement'),(206,'Gradual rise'),(215,'Late rise')]
    for j,(sid,name) in enumerate(examples):
        ax=fig.add_axes([.075+j*.312,.273,.276,.082])
        g=trials[trials.iSub.eq(sid)];s=people.set_index('iSub').loc[sid]
        color=TASK_COLORS[int(s.task)]
        ax.plot(g.trial,g.rolling_accuracy,color=color,lw=1)
        ax.axhline(.5 if s.task==1 else .25,color='.78',lw=.5,ls='--')
        if pd.notna(s.first_crossing64):ax.axvline(s.first_crossing64,color='.45',lw=.7,ls=':')
        ax.set(xlim=(1,len(g)),ylim=(0,1.04),yticks=[0,.5,1],xticks=[1,len(g)//2,len(g)],xlabel='Trial')
        ax.set_title(f'{name} · S{sid} (Task {int(s.task)})',fontsize=6.5,loc='left',pad=5)
        if j==0:ax.set_ylabel('Accuracy')
    fig.text(.075,.226,'Illustrations, not learner classes. Dotted line: first 64-trial window with ≥90% correct.',fontsize=5.9,color='.4')
    names=[('e','Time to improvement'),('f','Shape of improvement'),('g','Later performance')]
    axes=[]
    for j,(letter,name) in enumerate(names):
        x=.075+j*.322;heading(fig,x,.190,letter,name)
        axes.append(fig.add_axes([x,.064,.255,.103]))
    for task in (1,2,3):
        sub=people[people.task.eq(task)].sort_values('iSub');c=TASK_COLORS[task]
        # Deterministic offsets, no resampling or hidden point removal.
        xx=task+np.linspace(-.23,.23,len(sub))
        hit=sub.criterion_reached.to_numpy()
        axes[0].scatter(xx[hit],sub.first_crossing64[hit],s=8,color=c,alpha=.75,lw=0)
        axes[0].scatter(xx[~hit],sub.n_trials[~hit],s=14,facecolors='none',edgecolors=c,marker='^',lw=.7)
        median=sub.first_crossing64.median();axes[0].plot([task-.27,task+.27],[median]*2,color='.2',lw=1)
        valid=sub.delta_bic.notna().to_numpy();constant=sub.preferred.eq('constant').to_numpy()
        axes[1].scatter(xx[valid&~constant],sub.delta_bic[valid&~constant],s=9,color=c,lw=0,alpha=.8)
        axes[1].scatter(xx[valid&constant],sub.delta_bic[valid&constant],s=13,edgecolors=c,facecolors='none',lw=.7)
        axes[2].scatter(xx[hit],sub.last64_accuracy[hit],s=9,color=c,lw=0,alpha=.8)
        axes[2].scatter(xx[~hit],sub.last64_accuracy[~hit],s=14,facecolors='none',edgecolors=c,marker='^',lw=.7)
    for ax in axes:ax.set(xticks=[1,2,3],xticklabels=['Task 1','Task 2','Task 3'],xlim=(.5,3.5))
    axes[0].set(ylim=(0,xmax+60),yticks=[0,800,1600])
    axes[0].set_ylabel('First criterion (trial)',fontsize=6,labelpad=3)
    axes[0].text(.02,.96,'△ Unreached: record end',transform=axes[0].transAxes,fontsize=5.5,va='top')
    axes[1].axhline(0,color='.6',lw=.6);axes[1].set_ylabel('BIC(trend) − BIC(step)',labelpad=2,fontsize=6)
    axes[1].text(.02,.97,'Positive: step favored',transform=axes[1].transAxes,fontsize=5.4,va='top')
    axes[1].text(.02,.03,'○ Constant fits best',transform=axes[1].transAxes,fontsize=5.4,va='bottom')
    axes[2].axhline(.9,color='.5',lw=.6,ls=':');axes[2].set(ylim=(.35,1.03),yticks=[.5,.75,1])
    axes[2].set_ylabel('Final 64-trial accuracy',fontsize=6,labelpad=2)
    fallen=int((people.criterion_reached&people.end_below_criterion).sum())
    axes[2].text(.02,.04,f'{fallen} fell below 90% after\nan earlier crossing',transform=axes[2].transAxes,fontsize=5.5,va='bottom')
    fig.text(.075,.017,'Each point: one participant. Shape comparison: n = 95 (one 64-trial record too short). First crossing does not imply lasting mastery.',
             fontsize=5.6,color='.4')
    fig.savefig(output/'Figure1_learning_story.png',dpi=450,facecolor='white');plt.close(fig)


def build(source: Path, output: Path) -> None:
    trials=pd.read_csv(source/'trial_source.csv');summary=pd.read_csv(source/'subject_summary.csv')
    config=json.loads(Path(__file__).with_name('config.json').read_text())
    raw_path=ROOT/'data/exp123/processed/Task2_processed.csv';raw=pd.read_csv(raw_path)
    keys=['condition','iSub','iSession','iBlock','iTrial']
    observed=raw.sort_values(keys,kind='stable').reset_index(drop=True)
    for key in keys+['choice','feedback']:
        np.testing.assert_array_equal(trials[key],observed[key])
    if len(trials)!=62720 or len(summary)!=96:raise ValueError('Unexpected cohort size')
    people=behavior_table(trials,summary)
    output.mkdir(parents=True,exist_ok=False)
    people.to_csv(output/'subjects.csv',index=False)
    trials.to_csv(output/'trials.csv',index=False)
    render(trials,people,config['tasks'],output)
    files=[source/'trial_source.csv',source/'subject_summary.csv',raw_path,Path(__file__).resolve(),
           Path(__file__).with_name('LEARNING_STORY_DESIGN.md').resolve()]
    manifest={'n_participants':96,'n_trials':len(trials),'shape_n':int(people.delta_bic.notna().sum()),
              'dpi':450,'width_mm':183,'height_mm':225,'python':platform.python_version(),
              'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,
              'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in files}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(people.groupby('task').agg(n=('iSub','size'),reached=('criterion_reached','sum'),
                                   median_crossing=('first_crossing64','median')).to_string())


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(DEFAULT_SOURCE))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source.resolve(),args.output.resolve())


if __name__=='__main__':main()
