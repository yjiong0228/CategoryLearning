"""Compare frozen response fits using learning trajectories and noisy reports.

Figure contract: quantitative grids distinguish improved late behavior from
improved learning shape. All six participants, both declared oral readings,
complete-report dots and every generated path are retained. Models occupy
columns for oral support/conflict/undetermined curves. Behavioral lines compare
the original estimate, the new original-readout fit and the squared-readout fit
when present. State panels distinguish rule availability, its influence on
choice, and boundary precision in autonomous and conditional trajectories.
These are repeated development evaluations, not a new test cohort.
Python/matplotlib; width 183 mm, 300 dpi PNG only, with source tables and HTML.

Run --source COMPLETE_EXTERNAL_SUMMARY --output NEW. Oral blocks reuse the
established 32-trial denominator and preserve missing-report gaps. No fitting,
oral recoding, new uncertainty calculation or significance tests occur here.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1023_response_dynamics import read,write,digest,verify
from . import model1022_oral_trajectories as oral_time
from .model1023_response_dynamics import STYLE,save

FIG_WIDTH_MM=183
COLORS={'previous':'#8C9299','Q1':'#527697','Q2':'#C47742'}


def learning(tables,arms,names,output):
    fig,axes=plt.subplots(2,3,figsize=(FIG_WIDTH_MM/25.4,138/25.4),sharey=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.83,wspace=.24,hspace=.40)
    for ax,subject in zip(axes.flat,(108,205,306,110,209,309)):
        for arm in arms:
            part=tables['curves'].query('subject==@subject and arm==@arm').sort_values('trial')
            paths=tables['path_blocks'].query('subject==@subject and arm==@arm')
            values=paths.pivot(index='seed',columns='trial',values='accuracy')
            if len(values)!=128:raise ValueError('Missing autonomous paths')
            np.testing.assert_array_equal(values.columns,part.trial)
            np.testing.assert_allclose(values.mean(0),part.autonomous,atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.quantile(values,[.1,.9],axis=0),part[['lower','upper']].to_numpy().T,atol=1e-12,rtol=0)
            ax.plot(part.trial,values.to_numpy().T,color=COLORS[arm],alpha=.025,lw=.3)
            ax.fill_between(part.trial,part.lower,part.upper,color=COLORS[arm],alpha=.10,lw=0)
            ax.plot(part.trial,part.autonomous,color=COLORS[arm],lw=1.5)
        ax.plot(part.trial,part.observed,color='#222222',lw=1.1,marker='o',ms=2)
        ax.axvline(part.training_trials.iloc[0],color='#888888',ls=':',lw=.7)
        ax.set(title=f'Subject {subject}',xlim=(0,part.trial.max()),ylim=(-.02,1.02),yticks=[0,.5,1],xlabel='Trial (32-trial blocks)')
    for ax in axes[:,0]:ax.set_ylabel('Autonomous accuracy')
    handles=[Line2D([],[],color=COLORS[a],label=names[a],lw=1.5) for a in arms]
    handles.append(Line2D([],[],color='#222222',marker='o',ms=3,label='Participant'))
    fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,.95))
    fig.suptitle('Behavior after prefix-only parameter estimation',fontsize=10,y=.995)
    fig.text(.09,.045,'All 128 paths per model; bands are central 80% predictive ranges. Dotted line: parameter-estimation boundary.\nLater responses were excluded from estimation; these participants have already informed model development.',fontsize=7)
    return save(fig,output,'learning_after_refit.png')


def behavior_scores(tables,arms,names,output):
    target='Q2' if 'Q2' in arms else 'Q1';reference='Q1' if 'Q2' in arms else 'previous'
    part=tables['contrasts'].query('arm==@target and reference_arm==@reference and phase=="test"').set_index('subject')
    subjects=[108,205,306,110,209,309];part=part.loc[subjects]
    fig,axes=plt.subplots(1,2,figsize=(FIG_WIDTH_MM/25.4,105/25.4))
    fig.subplots_adjust(left=.11,right=.98,bottom=.25,top=.82,wspace=.40)
    for ax,metric,label in zip(axes,('choice_nll','energy'),('Later choice NLL / trial','Later autonomous Energy score')):
        ax.errorbar(part['delta_'+metric],range(6),xerr=part['delta_'+metric+'_mc_se'],fmt='o',color=COLORS[target],ms=4,capsize=2)
        ax.axvline(0,color='#777777',lw=.8,ls=':');ax.set_yticks(range(6),subjects);ax.invert_yaxis()
        ax.set_xlabel(label+' change');ax.set_ylabel('Subject')
        ax.axhline(2.5,color='#BBBBBB',lw=.5)
    fig.suptitle(names[target]+' minus '+names[reference],fontsize=10,y=.96)
    fig.text(.11,.065,'Negative values favor the target model. Bars: one paired Monte Carlo standard error.\nTop three: development participants. Bottom three: extension participants with uncertain prior influence.\nThese errors describe simulation resolution, not population-level confidence.',fontsize=7)
    return save(fig,output,'later_behavior_scores.png')


def conditional_state_blocks(trials):
    """Pool precision over rule presence within each chronological block."""
    rows=[]
    for (subject,arm),group in trials.groupby(['subject','arm']):
        group=group.sort_values('trial')
        np.testing.assert_array_equal(group.trial,np.arange(1,len(group)+1))
        cut=int(group.training_trials.iloc[0])
        if group.training_trials.nunique()!=1 or cut%32:raise ValueError('State trial boundaries changed')
        for first in range(0,len(group),32):
            b=group.iloc[first:first+32];presence=b.target_present.to_numpy()
            precision=b.target_precision_given_presence.to_numpy()
            if not np.isfinite(presence).all() or np.any((presence<0)|(presence>1+1e-12)):
                raise ValueError('Invalid target-rule presence probability')
            if np.any(~np.isfinite(precision[presence>0])):raise ValueError('Missing precision for a present rule')
            if np.any((precision[presence>0]<0)|(precision[presence>0]>25+1e-12)):
                raise ValueError('Precision exceeds the declared model range')
            beta=np.dot(presence,np.nan_to_num(precision,nan=0.))/presence.sum() if presence.sum()>0 else np.nan
            rows.append(dict(subject=subject,arm=arm,trial=int(b.trial.iloc[-1]),training_trials=cut,
                target_present=presence.mean(),target_contribution=b.target_contribution.mean(),
                target_precision_given_presence=beta))
    return pd.DataFrame(rows)


def state_grid(tables,conditional,arms,names,subject,output):
    fig,axes=plt.subplots(2,3,figsize=(FIG_WIDTH_MM/25.4,140/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.81,hspace=.46,wspace=.35)
    keys=('target_present','target_contribution','target_precision_given_presence')
    for row,source in enumerate((tables['autonomous_states'],conditional)):
        for col,key in enumerate(keys):
            ax=axes[row,col]
            for arm in arms:
                own=source.query('subject==@subject and arm==@arm').sort_values('trial')
                if own.empty:raise ValueError('Missing internal state trajectory')
                ax.plot(own.trial,own[key],color=COLORS[arm],lw=1.3)
            ax.axvline(own.training_trials.iloc[0],color='#888888',ls=':',lw=.7)
            ax.set(xlim=(0,own.trial.max()),xlabel='Trial (32-trial blocks)')
            ax.set_ylim((-.02,1.02) if col<2 else (0,25.5))
            ax.set_title(('Target-rule availability','Cognitive mixture weight' if row==0 else 'Choice-conditioned rule basis',
                          'Precision given rule presence')[col],fontsize=7)
        axes[row,0].set_ylabel('Autonomous paths' if row==0 else 'Conditional states')
    fig.suptitle(f'Subject {subject}: rule states during learning',fontsize=10,y=.99)
    fig.legend(handles=[Line2D([],[],color=COLORS[a],label=names[a],lw=1.5) for a in arms],
               ncol=2,loc='upper center',bbox_to_anchor=(.5,.95))
    fig.text(.10,.035,'Top: means over 128 autonomous paths. Bottom: states inferred from observed choices, before current feedback.\nPrecision conditions on rule presence; gaps mean the rule is absent. The declared precision ceiling is 25.\nRead precision alongside availability: rare-state precision has little posterior support.\nAutonomous contribution uses the raw mixture; conditional contribution is the basis for the observed choice.\nState curves show means, without parameter-uncertainty intervals. Dotted line: estimation boundary.',fontsize=7)
    return save(fig,output,f'S{subject}_rule_states.png')


def oral_grid(trials,blocks,meta,arms,names,cohort,reading,domain,output):
    subjects=sorted(meta.query('cohort==@cohort').subject.unique())
    if len(subjects)!=3:raise ValueError('Need all three cohort participants')
    fig,axes=plt.subplots(3,len(arms),figsize=(FIG_WIDTH_MM/25.4,194/25.4),sharey=True,squeeze=False)
    fig.subplots_adjust(left=.09,right=.98,top=.83,bottom=.17,hspace=.68,wspace=.22)
    for row,subject in enumerate(subjects):
        for col,arm in enumerate(arms):
            ax=axes[row,col];raw=trials.query('iSub==@subject and arm==@arm').sort_values('ordinal_trial')
            b=blocks.query('subject==@subject and arm==@arm').sort_values('trial')
            complete=raw.scope.eq('complete');cut=int(raw.training_trials.iloc[0]);length=len(raw)
            ax.axvspan(cut,length,color='#F3F4F6',zorder=0)
            ax.scatter(raw.loc[complete,'ordinal_trial'],raw.loc[complete,domain+'_support'],color=oral_time.COLORS['support'],s=3,alpha=.18,edgecolors='none')
            for key in oral_time.LABELS:
                ax.plot(b.trial,b[domain+'_'+key],color=oral_time.COLORS[key],ls=oral_time.STYLES[key],
                        lw=1.2 if key=='support' else .85,marker='o' if key=='support' else None,ms=2)
            ax.plot(b.trial,b[domain+'_coverage'],color='#527697',ls='-.',lw=.8)
            ax.plot(raw.loc[~complete,'ordinal_trial'],np.full((~complete).sum(),-.21),'|',color='#666666',
                    ms=2,mew=.5,transform=ax.get_xaxis_transform(),clip_on=False)
            ax.axvline(cut,color='#888888',lw=.7,ls=':')
            ax.set(xlim=(0,length),ylim=(-.025,1.025),yticks=[0,.5,1])
            ax.set_title(f'S{subject} · {names[arm]}\n{complete.sum()} complete reports / {length} trials',fontsize=7,pad=5)
            if col==0:ax.set_ylabel('Compatibility mass')
            if row==2:ax.set_xlabel('Trial')
    fig.suptitle(f'{cohort.capitalize()} · {domain} oral support · {reading} reading',fontsize=10,y=.98)
    handles=[Line2D([],[],color=oral_time.COLORS[k],ls=oral_time.STYLES[k],label=k.capitalize()) for k in oral_time.LABELS]
    handles.append(Line2D([],[],color='#527697',ls='-.',label='Catalogue coverage'))
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,.94))
    fig.text(.09,.045,'Dots: individual complete reports. Lines: equally weighted 32-trial report means. Empty blocks remain undefined.\nMarks below each axis: missing or incomplete reports. Grey background: later evaluation trials.\nReports are noisy external evidence; support, conflict and undetermined mass are shown together.',fontsize=7)
    return save(fig,output,f'oral_{cohort}_{reading}_{domain}.png')


def run(source: Path,output: Path):
    if output.exists():raise FileExistsError('Use a new response figure directory')
    done=read(source/'completion.json')
    if done['status']!='complete' or done['participants']!=6:raise ValueError('Require complete six-person evaluations')
    verify(done['inputs_sha256']);verify({str(source/k):v for k,v in done['artifacts_sha256'].items()})
    tables={k:pd.read_csv(source/(k+'.csv')) for k in ('scores','curves','path_blocks','contrasts','autonomous_states','conditional_state_trials')}
    meta=tables['scores'].query('phase=="whole"');arms=['previous','Q1']+(['Q2'] if done['fitting_phase']=='joint' else [])
    if set(meta.arm)!=set(arms) or meta.duplicated(['subject','arm']).any():raise ValueError('Missing or repeated arm')
    names={'previous':'Previous estimate','Q1':'Original readout refit' if 'Q2' in arms else 'Precision pair refit','Q2':'Squared readout refit'}
    output.mkdir(parents=True);(output/'source_data').mkdir()
    for name in done['artifacts_sha256']:
        dest=output/'source_data'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
    files=[];checks=0
    with plt.rc_context(STYLE):
        files.extend([learning(tables,arms,names,output),behavior_scores(tables,arms,names,output)])
        conditional=conditional_state_blocks(tables['conditional_state_trials'])
        conditional.to_csv(output/'source_data'/'conditional_state_blocks.csv',index=False)
        files.extend(state_grid(tables,conditional,arms,names,s,output) for s in (108,205,306,110,209,309))
        for reading in ('midpoint','relative'):
            trials=pd.read_csv(source/f'oral_{reading}/trials.csv');summary=pd.read_csv(source/f'oral_{reading}/summary.csv')
            if set(trials.arm)!=set(arms):raise ValueError('Oral model identities changed')
            blocks=oral_time.time_blocks(trials);checks+=oral_time.check_summary(trials,blocks,summary)
            blocks.to_csv(output/'source_data'/f'oral_{reading}/time_blocks.csv',index=False)
            for cohort in ('development','extension'):
                for domain in ('functional','structural'):
                    files.append(oral_grid(trials,blocks,meta,arms,names,cohort,reading,domain,output))
    body='<h1>重拟合后的行为与口述检验</h1><p>参数仅使用原训练段的选择数据估计。行为图保留全部自主路径；口述图同时显示支持、冲突、未确定部分和目录覆盖。原被试属于开发评价，后段边界沿用此前划分。</p>'
    body+=''.join(f'<h2>{html.escape(f)}</h2><img src="{f}" style="width:100%;max-width:1100px">' for f in files)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>作答机制重拟合评价</title><body>'+body+'</body></html>')
    write(output/'completion.json',dict(status='complete',figures=files,oral_aggregation_checks=checks,
        source_completion_sha256=digest(source/'completion.json'),plot_source_sha256=digest(Path(__file__)),
        export='183 mm, 300 dpi, PNG only',exclusions=0,
        source_data_sha256={str(p.relative_to(output)):digest(p) for p in (output/'source_data').rglob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
