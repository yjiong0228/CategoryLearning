"""Python-only PNG rendering of task design, learning trajectories and oral reports."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from .schematics import stimulus_panel, task_panel, procedure_panel

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .6, 'lines.linewidth': .85,
                     'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    # PNG only is the repository's explicit export policy.
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-.025 / ax.get_position().width, 1.07, letter, transform=ax.transAxes,
            weight='bold', fontsize=9, va='bottom')
    ax.set_title(title, loc='left', pad=8)


def curve(ax: plt.Axes, group: pd.DataFrame, task: dict, xmax: int | None = None) -> None:
    ax.plot(group.trial,group.rolling_accuracy,color=task['color'],lw=.9)
    ax.axhline(task['chance'],ls='--',lw=.5,color='.55')
    for t in group.loc[group.iSession.diff().fillna(0).ne(0),'trial']:
        ax.axvline(t-.5,color='.65',ls=':',lw=.5)
    ax.set(ylim=(-.03,1.04),xlim=(.5,(xmax or int(group.trial.max()))+.5),yticks=[0,.5,1])
    ax.tick_params(length=2,pad=1.5)


def report_strips(ax: plt.Axes, group: pd.DataFrame, color: str) -> None:
    """Aligned F1–F4 literal mentions; task-colored short marks retain actual trial positions."""
    for i in range(4):
        values = group[f'feature{i+1}_use'].to_numpy(dtype=float, na_value=np.nan)
        # Missing reports are distinct from available text with no explicit mention.
        for mask, mark_color in [(~group.report_present.to_numpy(), '#C8C8C8'), (values == 1, color)]:
            ax.vlines(group.loc[mask, 'trial'], i-.25, i+.25, colors=mark_color, linewidth=.55)
    for boundary in [.5,1.5,2.5]:
        ax.axhline(boundary,color='#E6E6E6',lw=.25)
    ax.set(yticks=[0,1,2,3],yticklabels=['F1','F2','F3','F4'],
           xlim=(.5,len(group)+.5),ylim=(3.5,-.5))
    ax.tick_params(axis='y',length=0,pad=2,labelsize=5)
    ax.tick_params(axis='x',length=2,pad=1,labelsize=5.5)
    for spine in ax.spines.values():spine.set_visible(False)


def oral_summary(ax: plt.Axes, summary: pd.DataFrame, config: dict, metric: str) -> None:
    settings={'feature_count':('Explicit feature count','Features / report',(.85,4.7),[1,2,3,4]),
              'path_coverage':('Required-feature coverage','Fraction named',(-.04,1.24),[0,.5,1]),
              'irrelevant_fraction':('Task-irrelevant features','Fraction named',(-.04,1.24),[0,.5,1])}
    title,ylabel,limits,ticks=settings[metric]
    for i,task in enumerate(config['tasks']):
        sub=summary[summary.condition==task['condition']]
        paired=sub[sub[f'first_{metric}'].notna() & sub[f'last_{metric}'].notna()]
        xx=[i*1.7,i*1.7+.6]
        for _,item in paired.iterrows():
            ax.plot(xx,[item[f'first_{metric}'],item[f'last_{metric}']],color=task['color'],alpha=.35,lw=.5,marker='.',ms=2.2)
        label_y=4.27 if metric=='feature_count' else 1.08
        ax.text(np.mean(xx),label_y,f"Task {task['task']}\nn = {len(paired)}",ha='center',fontsize=5.5,color=task['color'])
    ax.set(title=title,xticks=[0,.6,1.7,2.3,3.4,4.0],xticklabels=['First','Last']*3,
           ylabel=ylabel,ylim=limits,yticks=ticks,xlim=(-.28,4.3))
    ax.tick_params(axis='x',labelsize=5.3)
    ax.set_xlabel('64-trial periods',fontsize=5.7,labelpad=2)


def main_figure(trials: pd.DataFrame, summary: pd.DataFrame, selected: pd.DataFrame,
                config: dict, output: Path, nonoral_summary: pd.DataFrame) -> None:
    fig=plt.figure(figsize=(config['width_mm']/25.4,config['height_mm']/25.4))
    gs=fig.add_gridspec(5,1,height_ratios=[1.3,1.03,1.37,2.5,1.17],
                        left=.075,right=.975,bottom=.06,top=.965,hspace=.45)
    a=gs[0].subgridspec(1,4,width_ratios=[1.15,1,1,1],wspace=.12)
    ax=fig.add_subplot(a[0]);stimulus_panel(ax);panel(ax,'a','Stimulus and tasks')
    for cell,task in zip(list(a)[1:],config['tasks']):task_panel(fig.add_subplot(cell),task)
    fig.text(.50,ax.get_position().y0-.012,'Four continuously varying lengths; features aligned across participants; boundaries at 0.5.',
             fontsize=5.5,color='.4',ha='center')
    ax=fig.add_subplot(gs[1]);procedure_panel(ax);panel(ax,'b','Trial procedure')
    c=gs[2].subgridspec(1,3,wspace=.21)
    cmap=plt.get_cmap('cividis').copy();cmap.set_bad('#eeeeee')
    xmax=int(summary.n_trials.max());image=None
    for cell,task in zip(c,config['tasks']):
        inner=cell.subgridspec(2,1,height_ratios=[4,1],hspace=.05)
        ax=fig.add_subplot(inner[0]);sub=summary[summary.condition==task['condition']].sort_values(['n_trials','iSub'])
        data=np.full((len(sub),xmax),np.nan)
        for i,s in enumerate(sub.iSub):
            values=trials.loc[trials.iSub==s,'rolling_accuracy'].to_numpy();data[i,:len(values)]=values
        image=ax.imshow(data,aspect='auto',origin='upper',extent=(.5,xmax+.5,len(sub)+.5,.5),
                        cmap=cmap,norm=Normalize(0,1),interpolation='nearest')
        ax.set_title(f"Task {task['task']} · n = {len(sub)}",color=task['color'],pad=5)
        ax.set(yticks=[1,16,32]);ax.tick_params(axis='x',labelbottom=False,bottom=False)
        if task['task']==1:
            ax.set_ylabel('Participants\n(sorted by duration)')
            ax.text(-.025/ax.get_position().width,1.2,'c',weight='bold',fontsize=9,transform=ax.transAxes)
        else:ax.set_yticklabels([])
        count_ax=fig.add_subplot(inner[1],sharex=ax)
        count=(np.arange(1,xmax+1)[None,:]<=sub.n_trials.to_numpy()[:,None]).sum(axis=0)
        count_ax.fill_between(np.arange(1,xmax+1),count,color=task['color'],alpha=.25,lw=0)
        count_ax.plot(np.arange(1,xmax+1),count,color=task['color'],lw=.6)
        count_ax.set(ylim=(0,34),yticks=[0,32],xticks=[1,512,1024,1792],xlabel='Recorded trial')
        if task['task']==1:count_ax.set_ylabel('n',rotation=0,labelpad=5)
    cb_ax=fig.add_axes([.80,ax.get_position().y1+.041,.15,.006])
    cb=fig.colorbar(image,cax=cb_ax,orientation='horizontal',ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=5.3,length=1,pad=1);cb_ax.set_title('Accuracy (32 trials)',fontsize=5.5,pad=2)
    reps=gs[3].subgridspec(3,3,hspace=.55,wspace=.21)
    top=gs[3].get_position(fig).y1
    fig.text(.075,top+.015,'Reported features: F1–F4   ·   Task-colored marks: named   ·   White: not named   ·   Gray: missing text',
             fontsize=5.7,color='.3')
    for col,task in enumerate(config['tasks']):
        chosen=selected[selected.condition==task['condition']]
        for row,(_,item) in enumerate(chosen.iterrows()):
            inner=reps[row,col].subgridspec(2,1,height_ratios=[1.35,1.0],hspace=.07)
            ax=fig.add_subplot(inner[0]);group=trials[trials.iSub==item.iSub]
            curve(ax,group,task)
            ax.text(.01,1.025,f'S{item.iSub} · {int(item.n_trials)} trials',transform=ax.transAxes,fontsize=5.7,va='bottom')
            ax.set_yticks([0,1]);ax.tick_params(axis='x',bottom=False,labelbottom=False)
            if col==0 and row==1:ax.set_ylabel('Accuracy')
            strips=fig.add_subplot(inner[1],sharex=ax);report_strips(strips,group,task['color'])
            if row==2:strips.set_xlabel('Recorded trial',fontsize=5.7,labelpad=2)
    from .nonoral import plot_rt_pairs
    bottom=gs[4].subgridspec(1,2,wspace=.36)
    ax=fig.add_subplot(bottom[0]);plot_rt_pairs(ax,nonoral_summary,config)
    ax.text(-.025/ax.get_position().width,1.1,'d',weight='bold',fontsize=9,transform=ax.transAxes)
    ax=fig.add_subplot(bottom[1]);oral_summary(ax,summary,config,'feature_count')
    fig.text(.075,.018,'All 96 participants in c · paired summaries use disjoint periods · no model-derived state or learner classification',fontsize=5.6,color='.4')
    save(fig,output/'fig1_behavior_draft.png',config['dpi'])


def atlases(trials: pd.DataFrame, summary: pd.DataFrame, config: dict, output: Path) -> None:
    for task in config['tasks']:
        subjects=summary[summary.condition==task['condition']].sort_values(['n_trials','iSub'])
        fig,axes=plt.subplots(8,4,figsize=(183/25.4,240/25.4),sharey=True)
        fig.subplots_adjust(left=.075,right=.98,bottom=.055,top=.94,hspace=.78,wspace=.25)
        for ax,item in zip(axes.flat,subjects.itertuples()):
            group=trials[trials.iSub==item.iSub];curve(ax,group,task)
            ax.plot(group.trial,group.rolling_clear_accuracy,color='.5',lw=.55,alpha=.65,zorder=0)
            ax.set_title(f'S{item.iSub} · {item.n_trials} trials',loc='left',fontsize=6,pad=3)
            ax.tick_params(labelsize=5.5)
        for ax in axes[-1]:ax.set_xlabel('Recorded trial')
        for ax in axes[:,0]:ax.set_ylabel('Accuracy')
        fig.suptitle(f"Task {task['task']} (condition {task['condition']}) · all 32 participants",fontsize=10,y=.985)
        fig.text(.075,.955,'Colored: all trials, trailing 32 · gray: non-ambiguous trials within the same window · dotted: session boundary',fontsize=6)
        fig.text(.075,.015,"Records are sorted by duration. Each x-axis ends at that participant's last recorded trial. No participant is excluded.",fontsize=6)
        save(fig,output/f"figS1_task{task['task']}_all_subjects.png",config['dpi'])


def oral_atlases(trials: pd.DataFrame, summary: pd.DataFrame, config: dict, output: Path) -> None:
    for task in config['tasks']:
        subjects=summary[summary.condition==task['condition']].sort_values(['n_trials','iSub'])
        fig=plt.figure(figsize=(183/25.4,260/25.4))
        outer=fig.add_gridspec(8,4,left=.065,right=.985,bottom=.055,top=.935,hspace=.65,wspace=.26)
        for cell,item in zip(outer,subjects.itertuples()):
            inner=cell.subgridspec(2,1,height_ratios=[1.4,1],hspace=.08)
            group=trials[trials.iSub==item.iSub]
            ax=fig.add_subplot(inner[0]);curve(ax,group,task)
            ax.set_yticks([0,1]);ax.tick_params(axis='x',bottom=False,labelbottom=False)
            ax.set_title(f'S{item.iSub} · {item.n_trials} trials',fontsize=6,loc='left',pad=3)
            strip=fig.add_subplot(inner[1],sharex=ax);report_strips(strip,group,task['color'])
        fig.suptitle(f"Task {task['task']} · behavior and reported features for all participants",fontsize=9,y=.984)
        fig.text(.065,.952,'F1–F4: aligned features   ·   Task-colored marks: named   ·   White: not named   ·   Gray: missing text',fontsize=6)
        fig.text(.065,.018,'Top: trailing accuracy. Bottom: literal mentions aligned using each participant’s feature assignment.',fontsize=6)
        save(fig,output/f"figS3_task{task['task']}_behavior_oral.png",config['dpi'])


def sensitivity(trials: pd.DataFrame, summary: pd.DataFrame, config: dict, output: Path) -> None:
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,130/25.4))
    fig.subplots_adjust(left=.09,right=.98,bottom=.1,top=.89,hspace=.6,wspace=.38)
    for col,task in enumerate(config['tasks']):
        sub=summary[summary.condition==task['condition']];ax=axes[0,col]
        for row in sub.itertuples():
            ax.plot([16,32,64],[getattr(row,f'max_gain_w{w}') for w in [16,32,64]],color=task['color'],alpha=.3,lw=.5,marker='.',ms=2)
        ax.set(title=f"Task {task['task']}",xticks=[16,32,64],xlabel='Adjacent window length',ylim=(-.05,1.05))
        if col==0:ax.set_ylabel('Maximum local gain')
        ax=axes[1,col]
        paired=sub[sub.first_feature_count.notna() & sub.last_feature_count.notna()]
        for row in paired.itertuples():
            ax.plot([0,1],[row.first_feature_count,row.last_feature_count],color=task['color'],alpha=.38,lw=.6,marker='.',ms=3)
        ax.set(xticks=[0,1],xticklabels=['First 64 trials','Last 64 trials'],ylim=(.9,4.1),yticks=[1,2,3,4])
        ax.set_title(f'Disjoint report periods · n = {len(paired)}',fontsize=6.5)
        if col==0:ax.set_ylabel('Mean explicit features / report')
    fig.suptitle('Exploratory checks: window sensitivity and report descriptions',fontsize=9,y=.98)
    fig.text(.09,.925,'Every line is a participant; incomplete windows are unavailable. Gain is not a learner class or a changepoint test.',fontsize=6)
    fig.text(.09,.018,'Report pairs require ≥128 recorded trials and ≥1 recognized report in each period; shorter records remain in Fig. 1.',fontsize=6)
    save(fig,output/'figS2_window_and_oral_sensitivity.png',config['dpi'])
