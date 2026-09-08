"""Exploratory behavior beyond reports: speed, feedback history and stimulus margin.

No trimming, fitted model, significance-based selection or cross-session lagging.
"""
import numpy as np
import pandas as pd


def analyze(trials: pd.DataFrame, window: int = 64):
    frame = trials.copy()
    frame['valid_rt'] = np.isfinite(frame.choRT) & frame.choRT.gt(0)
    frame['log_rt'] = np.log(frame.choRT.where(frame.valid_rt))
    keys = ['condition', 'iSub', 'iSession', 'iBlock']
    grouped = frame.groupby(keys, sort=False)
    contiguous = grouped.iTrial.diff().eq(1)
    frame['previous_feedback'] = grouped.feedback.shift().where(contiguous)
    # Block centering reduces slow learning-related RT drift; it does not remove
    # confounding by stimulus order/difficulty and is not a causal feedback effect.
    frame['centered_log_rt'] = frame.log_rt - grouped.log_rt.transform('median')
    root = (frame.feature1 - .5).abs()
    branch = np.where(frame.feature1 <= .5, frame.feature2, frame.feature3)
    frame['boundary_distance'] = np.where(frame.condition.eq(1), root,
                                         np.minimum(root, np.abs(branch-.5)))
    edges = [0., .05, .10, .20, .50]
    frame['distance_bin'] = pd.cut(frame.boundary_distance, edges, include_lowest=True, labels=False)
    summaries, periods = [], []
    for (condition, subject), group in frame.groupby(['condition', 'iSub']):
        row = {'condition': condition, 'iSub': subject, 'n_trials':len(group),
               'invalid_rt_n':int((~group.valid_rt).sum())}
        for phase, part in [('first',group.iloc[:window]), ('last',group.iloc[-window:])]:
            valid = part[part.valid_rt]
            eligible = len(group) >= 2*window
            row[f'{phase}_median_rt'] = valid.choRT.median() if eligible else np.nan
            row[f'{phase}_correct_median_rt'] = valid.loc[valid.correct.eq(1),'choRT'].median() if eligible else np.nan
            row[f'{phase}_rt_n'] = len(valid) if eligible else 0
            row[f'{phase}_correct_rt_n'] = int(valid.correct.sum()) if eligible else 0
            periods.append({'condition':condition,'iSub':subject,'phase':phase,
                            'eligible':eligible,'n':len(valid),'median_rt':row[f'{phase}_median_rt'],
                            'accuracy':part.correct.mean() if eligible else np.nan})
        summaries.append(row)
    feedback = frame.groupby(['condition','iSub','previous_feedback']).agg(
        median_centered_log_rt=('centered_log_rt','median'), n=('centered_log_rt','count')).reset_index()
    feedback['relative_rt'] = np.exp(feedback.median_centered_log_rt)
    boundary = frame.groupby(['condition','iSub','distance_bin']).agg(
        accuracy=('correct','mean'), n=('correct','size')).reset_index()
    boundary['distance_midpoint'] = boundary.distance_bin.map(dict(enumerate([.025,.075,.15,.35])))
    return frame, pd.DataFrame(summaries), pd.DataFrame(periods), feedback, boundary


def plot_rt_pairs(ax, summary, config):
    for i, task in enumerate(config['tasks']):
        sub = summary[summary.condition.eq(task['condition'])].dropna(subset=['first_median_rt','last_median_rt'])
        xx = [i*1.7, i*1.7+.6]
        for row in sub.itertuples():
            ax.plot(xx, [row.first_median_rt,row.last_median_rt], color=task['color'],
                    lw=.5,alpha=.4,marker='.',ms=2.2)
        ax.text(np.mean(xx), 37, f"Task {task['task']}\nn = {len(sub)}",ha='center',fontsize=5.5,color=task['color'])
    ax.set(yscale='log',ylim=(.45,75),yticks=[.5,1,2,5,10,20],
           yticklabels=['0.5','1','2','5','10','20'],
           xticks=[0,.6,1.7,2.3,3.4,4.0],xticklabels=['First','Last']*3,
           ylabel='Median choice RT (s)',xlabel='64-trial periods',title='Response speed')
    ax.minorticks_off()


def plot_candidates(summary, feedback, boundary, config, output):
    from .render import plt, save
    fig, axes = plt.subplots(3,3,figsize=(183/25.4,190/25.4))
    fig.subplots_adjust(left=.10,right=.975,bottom=.12,top=.91,hspace=.65,wspace=.37)
    for col, task in enumerate(config['tasks']):
        color=task['color'];condition=task['condition']
        sub=summary[summary.condition.eq(condition)].dropna(subset=['first_median_rt','last_median_rt'])
        ax=axes[0,col]
        for row in sub.itertuples():
            ax.plot([0,1],[row.first_median_rt,row.last_median_rt],color=color,lw=.5,alpha=.4)
            ax.plot([0,1],[row.first_correct_median_rt,row.last_correct_median_rt],color='.5',ls=':',lw=.4,alpha=.3)
        ax.set(title=f"Task {task['task']} · n = {len(sub)}",xticks=[0,1],xticklabels=['First 64','Last 64'],
               yscale='log',ylim=(.45,35),yticks=[.5,1,2,5,10,20],yticklabels=['0.5','1','2','5','10','20'])
        ax.minorticks_off()
        if col==0:ax.set_ylabel('Median choice RT (s)')
        ax=axes[1,col]
        sub=feedback[feedback.condition.eq(condition)]
        levels=[0.,.5,1.] if condition==3 else [0.,1.]
        # Complete within-subject sets avoid changing cohorts across feedback levels.
        wide=sub.pivot(index='iSub',columns='previous_feedback',values='relative_rt').reindex(columns=levels).dropna()
        for _,row in wide.iterrows():
            ax.plot(range(len(levels)),row.to_numpy(),color=color,lw=.5,alpha=.35,marker='.',ms=2)
        ax.axhline(1,color='.5',lw=.6,ls='--')
        ax.set(xticks=range(len(levels)),xticklabels=[str(v).rstrip('0').rstrip('.') for v in levels],
               xlabel='Previous feedback',yscale='log',ylim=(.5,10),yticks=[.5,1,2,5,10],
               yticklabels=['0.5','1','2','5','10'],title=f'Within-block RT · n = {len(wide)}')
        ax.minorticks_off()
        if col==0:ax.set_ylabel('Relative RT')
        ax=axes[2,col]
        sub=boundary[boundary.condition.eq(condition)]
        for _,group in sub.groupby('iSub'):
            ax.plot(group.distance_midpoint,group.accuracy,color=color,alpha=.15,lw=.5)
        means=sub.groupby('distance_midpoint').accuracy.mean()
        ax.plot(means.index,means.values,color=color,lw=1.4,marker='o',ms=3)
        ax.axhline(task['chance'],color='.5',lw=.6,ls='--')
        ax.set(ylim=(0,1.03),xticks=[.025,.075,.15,.35],xticklabels=['0–.05','.05–.1','.1–.2','.2–.5'],
               xlabel='Distance to nearest boundary',title='Stimulus difficulty')
        ax.tick_params(axis='x',labelsize=5)
        if col==0:ax.set_ylabel('Accuracy')
    fig.suptitle('Non-oral behavioral candidates · descriptive, participant-level comparisons',fontsize=9,y=.975)
    fig.text(.10,.938,'Top: all choices (color), correct choices (gray dotted); paired non-overlapping periods.',fontsize=6)
    fig.text(.10,.043,'Middle: exp[median(block-centered log RT)] by previous feedback; consecutive trials within a block only.',fontsize=5.8)
    fig.text(.10,.022,'Bottom: thin lines = participants; thick = equally weighted mean. Stimulus difficulty and learning stage can confound comparisons.',fontsize=5.7)
    save(fig,output/'figS4_nonoral_candidates.png',config['dpi'])
