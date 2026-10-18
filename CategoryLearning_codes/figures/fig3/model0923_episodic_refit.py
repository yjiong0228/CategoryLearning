"""Joint rule-learning refits: every declared participant, arm and saved path.

Python discussion figures at 183 mm / 300 dpi, PNG only. The evidence chain
compares training selection, frozen later-trial probabilities and autonomous
learning distributions. Error bars are Monte Carlo errors, not population CIs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_h_validation import read, write, digest

ORDER = ['BASE','NEW','POOL','DRAW']
COLORS = dict(zip(ORDER,['#68778B','#78A7AD','#277F8E','#A16F99']))
MARKERS = dict(zip(ORDER,['o','s','D','^']))


def run(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new episodic-refit figure directory')
    manifest = read(source/'manifest.json')
    order = manifest['arms']; reference = manifest.get('reference_model', 'BASE')
    if manifest['status'] != 'complete' or reference not in order or len(order) < 2:
        raise ValueError('Incomplete comparison or missing reference')
    order = [reference] + [arm for arm in order if arm != reference]
    colors = dict(COLORS, SEARCH='#277F8E'); markers = dict(MARKERS, SEARCH='^')
    if order == ['POOL', 'SEARCH']:
        colors['POOL'] = '#68778B'; markers['POOL'] = 'o'
    if not set(order) <= set(colors): raise ValueError('Declare colors for every fitted mechanism')
    for name,sha in manifest['summary_sha256'].items():
        if digest(source/name) != sha: raise ValueError('Source table changed')
    tables = {Path(name).stem:pd.read_csv(source/name) for name in manifest['summary_sha256']}
    subjects = sorted(tables['scores'].subject.unique())
    if not subjects: raise ValueError('Need at least one participant')
    grid_columns = min(3, len(subjects)); grid_rows = int(np.ceil(len(subjects) / grid_columns))
    alternatives = order[1:]
    cohort = tables['cohort_scores']
    best_choice = cohort[cohort.phase=='test'].sort_values(['choice_nll','model']).iloc[0].model
    best_generation = cohort[cohort.phase=='whole'].sort_values(['energy','model']).iloc[0].model
    n_filters = int(tables['scores'].filter_seeds.iloc[0])
    n_paths = int(tables['scores'].paths.iloc[0])
    output.mkdir(parents=True)
    write(output/'figure_contract.json',dict(
        conclusion=f'Across {len(subjects)} development participants, {best_choice} has the lowest mean later-trial NLL and {best_generation} the lowest mean autonomous Energy among the {len(order)} joint-refit point estimates.',
        evidence=['independent training repeat, later-trial choice NLL, whole-curve Energy',
                  'all autonomous paths and late-accuracy distributions','target discovery, retention, contribution',
                  'confirmation of selected versus alternative parameters',
                  'correct-answer and error-answer contributions to likelihood changes'],
        archetype='quantitative grid',backend='python',width_mm=183,dpi=300,formats=['PNG'],
        inclusion=f'all {len(subjects)} development participants, {len(order)} arms, all saved paths and filter seeds; no exclusions',
        reference_model=reference,
        statistics='empirical Energy V statistic; error bars +/- 2 whole-filter/path Monte Carlo SE',
        adaptation='style-only inheritance; parameter-refitted results with explicit training/later-trial split',
        limits='bounded parameter search and development validation; fixed task stimulus schedule; no population tests',
        versions=dict(python=sys.version,matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__)))
    for name in manifest['summary_sha256']: shutil.copy2(source/name,output/name)
    shutil.copy2(source/'manifest.json',output/'source_manifest.json')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    images = []
    def save(fig,name):
        fig.savefig(output/name,dpi=300,facecolor='white'); plt.close(fig); images.append(name)
    handles = [Line2D([],[],color=colors[m],marker=markers[m],lw=.8,label=m) for m in order]
    contrasts = tables['contrasts'].set_index(['subject','model','phase'])
    fig,axes = plt.subplots(1,3,figsize=(183/25.4,(80+len(subjects)*20/3)/25.4))
    for ax,phase,metric,title in zip(axes,['training','test','whole'],['choice_nll','choice_nll','energy'],
            ['a  Training repeat','b  Later-trial prediction','c  Autonomous distribution']):
        ax.axvline(0,color='#9AA1AA',lw=.7)
        for i,arm in enumerate(alternatives):
            data = contrasts.loc[[(s,arm,phase) for s in subjects]]
            ax.errorbar(data['delta_'+metric],np.arange(len(subjects))+(i-(len(alternatives)-1)/2)*.18,
                xerr=2*data['delta_'+metric+'_mc_se'],fmt=markers[arm],color=colors[arm],ms=3,capsize=1.5,lw=.7)
        ax.set(yticks=range(len(subjects)),yticklabels=[f'S{s}' for s in subjects],title=title,
            xlabel=('NLL / trial' if metric=='choice_nll' else 'Energy score')+f' change from {reference}\nNegative favors alternative',
            ylim=(len(subjects)-.4,-.6))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.legend(handles=handles[1:],loc='upper center',ncol=3,bbox_to_anchor=(.5,.91))
    fig.suptitle(f'Joint refits: frozen prediction and autonomous learning\n{len(subjects)} development participants; {n_filters} filters / {n_paths} paths per cell; bars: ± 2 Monte Carlo SE',y=.99,fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.82,wspace=.48)
    save(fig,'comparison.png')

    finalist = tables['training_finalists']
    fig,axes = plt.subplots(grid_rows,grid_columns,figsize=(183/25.4,(50+45*grid_rows)/25.4),squeeze=False)
    for unused in axes.flat[len(subjects):]: unused.set_visible(False)
    for ax,subject in zip(axes.flat,subjects):
        for i,arm in enumerate(order):
            rows = finalist[(finalist.subject==subject)&(finalist.model==arm)].sort_values('nll')
            for j,row in enumerate(rows.itertuples()):
                x = i+(j-(len(rows)-1)/2)*.18
                ax.errorbar(x,row.nll,yerr=2*row.mc_se,fmt=markers[arm],color=colors[arm],
                    mfc=colors[arm] if row.selected else 'white',ms=4,capsize=2,lw=.7)
                if row.reference: ax.plot(x,row.nll,marker='o',mfc='none',mec='#30343A',ms=7,lw=0)
        ax.set(title=f'S{subject}',xticks=range(len(order)),xticklabels=order,ylabel='Confirmation NLL / trial')
        ax.tick_params(axis='x',rotation=30)
    fig.suptitle(f'Why the selected parameters win within the finalist set ({n_filters} filters per point)\nFilled: selected; open: alternative; black ring: old reference; bars: ± 2 Monte Carlo SE',fontsize=8,y=.99)
    fig.subplots_adjust(left=.1,right=.98,bottom=.17,top=.81,wspace=.5,hspace=.6)
    save(fig,'parameter_finalists.png')

    fig,axes = plt.subplots(grid_rows,grid_columns,figsize=(183/25.4,(50+45*grid_rows)/25.4),squeeze=False)
    for unused in axes.flat[len(subjects):]: unused.set_visible(False)
    for ax,subject in zip(axes.flat,subjects):
        for i,arm in enumerate(alternatives):
            row = contrasts.loc[subject,arm,'training']
            ax.bar(i-.15,row.delta_correct_loss,width=.28,color='#78A7AD')
            ax.bar(i+.15,row.delta_error_loss,width=.28,color='#B49A7B')
            ax.plot(i,row.delta_choice_nll,'D',color='#222222',ms=3)
        ax.axhline(0,color='#9AA1AA',lw=.6)
        ax.set(title=f'S{subject}',xticks=range(len(alternatives)),xticklabels=alternatives,ylabel=f'NLL / trial change from {reference}')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.legend(handles=[Line2D([],[],color='#78A7AD',lw=5,label='Correct-answer contribution'),
        Line2D([],[],color='#B49A7B',lw=5,label='Error-answer contribution'),
        Line2D([],[],color='#222222',ls='',marker='D',ms=3,label='Total change')],
        loc='upper center',ncol=3,bbox_to_anchor=(.5,.91))
    fig.suptitle(f'Which observed answers explain the likelihood change?\nIndependent repeat ({n_filters} filters); both contributions divided by all training trials',fontsize=8,y=.99)
    fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.79,wspace=.58,hspace=.5)
    save(fig,'correct_error_losses.png')

    scores = tables['scores']; whole = scores[scores.phase=='whole'].set_index(['subject','model'])
    fig,axes = plt.subplots(1,3,figsize=(183/25.4,(75+len(subjects)*20/3)/25.4))
    for ax,field,title in zip(axes,['target_ever','target_late_present','target_late_contribution'],
            ['a  Target ever active','b  Target present late','c  Target choice contribution late']):
        for i,arm in enumerate(order):
            rows = whole.loc[[(s,arm) for s in subjects]]
            ax.plot(rows[field],np.arange(len(subjects))+(i-(len(order)-1)/2)*.13,markers[arm],color=colors[arm],ms=3,label=arm)
        ax.set(xlim=(-.03,1.03),ylim=(len(subjects)-.4,-.6),yticks=range(len(subjects)),
               yticklabels=[f'S{s}' for s in subjects],xlabel='Fraction',title=title)
    fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.5,.91))
    fig.suptitle(f'Discovery, retention and use after joint refitting ({n_paths} paths per cell)\nLate = final 128 trials; target rule is used only for retrospective diagnosis',fontsize=8,y=.99)
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.82,wspace=.45)
    save(fig,'rule_process.png')

    for subject in subjects:
        fig,axes = plt.subplots(len(order),2,figsize=(183/25.4,(65+35*len(order))/25.4),gridspec_kw={'width_ratios':[3,1]})
        for row,arm in enumerate(order):
            ax,dist = axes[row]
            c = tables['curves']; c = c[(c.subject==subject)&(c.model==arm)].sort_values('trial')
            blocks = tables['path_blocks']; blocks = blocks[(blocks.subject==subject)&(blocks.model==arm)]
            paths = tables['paths']; paths = paths[(paths.subject==subject)&(paths.model==arm)]
            count = 0
            for _,group in blocks.groupby('seed'):
                group = group.sort_values('trial'); ax.plot(group.trial,group.accuracy,color=colors[arm],alpha=.07,lw=.45)
                count += 1
            if count != len(paths): raise ValueError('Some autonomous paths are missing from the figure')
            ax.fill_between(c.trial,c.lower,c.upper,color=colors[arm],alpha=.12)
            ax.plot(c.trial,c.autonomous,color=colors[arm],lw=1.4,label='Autonomous mean')
            ax.plot(c.trial,c.observed,color='#222222',lw=1.1,label='Observed')
            ax.plot(c.trial,c.conditional,color=colors[arm],ls='--',lw=1,label='Conditional probability')
            ax.axvline(c.training_trials.iloc[0],color='#777777',ls=':',lw=.8)
            ax.set(ylim=(-.03,1.03),ylabel=f'{arm}\nAccuracy',title=f'{arm}: all {count} autonomous paths',xlim=(0,c.trial.max()))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
            ordered = np.sort(paths.late_accuracy)
            dist.step(np.r_[0.,ordered,1.],np.r_[0.,np.arange(1,len(ordered)+1)/len(ordered),1.],
                where='post',color=colors[arm],lw=1.3)
            dist.axvline(whole.loc[subject,arm].observed_late,color='#222222',lw=1,ls='--')
            reached = int(round(whole.loc[subject,arm].late_reaches_observed*len(paths)))
            dist.set(xlim=(0,1),ylim=(0,1.03),
                title=f'Final 128 trials\n≥ observed: {reached}/{len(paths)}',ylabel='Cumulative fraction')
            dist.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[-1,0].set_xlabel('Trial'); axes[-1,1].set_xlabel('Autonomous accuracy')
        handles_curve = [Line2D([],[],color='#222222',lw=1,label='Observed'),
                         Line2D([],[],color='#68778B',lw=1.4,label='Autonomous mean'),
                         Line2D([],[],color='#68778B',ls='--',lw=1,label='Conditional probability')]
        fig.legend(handles=handles_curve,loc='upper center',ncol=3,bbox_to_anchor=(.5,.947))
        fig.suptitle(f'S{subject}: learning distribution at frozen fitted parameters\n32-trial blocks; band: 10th–90th path percentiles; vertical dotted line: training end',fontsize=8,y=.985)
        fig.subplots_adjust(left=.1,right=.98,bottom=.09,top=.83,wspace=.3,hspace=.6)
        save(fig,f'subject_{subject}.png')
    shutil.copy2(__file__,output/'figure_source.py')
    write(output/'qa_render.json',dict(images={name:digest(output/name) for name in images},
        participants=len(subjects),arms=order,paths=len(tables['paths']),path_blocks=len(tables['path_blocks']),
        exclusions=[],formats=['PNG'],dpi=300,width_mm=183,visual_review='pending',
        export_exceptions='Repository discussion figures require PNG only; vector/TIFF and 600 dpi submission rules do not apply.'))
    body = ''.join(f'<section><h2>{name}</h2><a href="{name}"><img src="{name}" style="width:100%;max-width:1000px"></a></section>' for name in images)
    records = list(manifest['summary_sha256'])+['figure_contract.json','source_manifest.json','qa_render.json','figure_source.py']
    links = '<h2>源表与运行记录</h2><ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in records)+'</ul>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Admission joint refits</title>'
        f'<h1>H规则学习机制联合重拟合</h1><p>{len(subjects)}位开发被试；{len(order)}种机制按同一预算估计参数。全部自主路径及源表保留。'
        '误差线表示有限滤波／模拟次数的计算不确定性。后段数据用于冻结参数后的开发验证。</p>'+links+body,encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); run(args.source,args.output)
