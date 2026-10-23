"""Show whether prefix-state continuation resolves autonomous learning gaps.

Figure contract: quantitative grids compare two starting-state distributions
at the same fixed parameters, stimuli and future trials. Behavior is the main
panel; membership, choice contribution and precision locate possible sources
of disagreement. All three candidates and all 128 paths per start are shown.
Separate pairing panels describe condition 3, where archived task-start pairing
traces are unavailable. Style-only inheritance from earlier Python figures.

Use --source COMPLETE_SUMMARY --output NEW. Export 183-mm 300-dpi PNG review
figures, source CSVs and an HTML guide. Path bands are central 80% predictive
ranges, not confidence intervals; MC errors use eight complete filter groups.
No significance tests, fitting or participant exclusions occur here.
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

from src.Bayesian_state.workflows.analysis.model_1023_prefix_forecast import read, write, digest, verify

STYLE = {'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'], 'font.size':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'}
STARTS = ('task_start','filtered_prefix')
COLORS = {'task_start':'#527697','filtered_prefix':'#C47742'}
LABELS = {'task_start':'From task start','filtered_prefix':'From inferred prefix state'}


def candidates(scores, subject):
    own=scores[(scores.subject==subject)&(scores.start=='task_start')]
    own=own.sort_values(['selected','reference','candidate'],ascending=[False,False,True])
    if len(own)!=3 or own.selected.sum()!=1 or own.reference.sum()!=1:
        raise ValueError('Need all three fixed candidates and unique roles')
    return [(r.candidate,'Selected' if r.selected else 'Reference' if r.reference else 'Alternative')
            for r in own.itertuples()]


def load_data(source):
    done=read(source/'completion.json')
    if done['status']!='complete' or done['candidates']!=9 or done['forecast_paths']!=1152:
        raise ValueError('Need the complete nine-point forecast study')
    verify(done['inputs_sha256'])
    verify({str(source/k):v for k,v in done['artifacts_sha256'].items()})
    tables={name:pd.read_csv(source/(name+'.csv')) for name in
            ('scores','curves','path_blocks','states','endpoints','path_metrics','start_contrasts')}
    for (subject,candidate,start),part in tables['curves'].groupby(['subject','candidate','start']):
        part=part.sort_values('trial')
        paths=tables['path_blocks'].query('subject==@subject and candidate==@candidate and start==@start')
        matrix=paths.pivot(index='seed',columns='trial',values='accuracy')
        if len(matrix)!=128:
            raise ValueError('Missing paths')
        np.testing.assert_array_equal(matrix.columns,part.trial)
        np.testing.assert_allclose(matrix.mean(0),part['mean'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.quantile(matrix,[.1,.9],axis=0),part[['q10','q90']].to_numpy().T,atol=1e-12,rtol=0)
    return tables,done


def save(fig, output, name):
    fig.savefig(output/name,dpi=300,facecolor='white')
    plt.close(fig)
    return name


def trajectories(tables, subject, output):
    fig,axes=plt.subplots(4,3,figsize=(183/25.4,215/25.4),sharex=True,sharey='row')
    fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.85,hspace=.25,wspace=.20)
    for col,(candidate,label) in enumerate(candidates(tables['scores'],subject)):
        for start in STARTS:
            curve=tables['curves'].query('subject==@subject and candidate==@candidate and start==@start').sort_values('trial')
            state=tables['states'].query('subject==@subject and candidate==@candidate and start==@start').sort_values('trial')
            paths=tables['path_blocks'].query('subject==@subject and candidate==@candidate and start==@start')
            matrix=paths.pivot(index='seed',columns='trial',values='accuracy')
            color=COLORS[start]
            axes[0,col].plot(curve.trial,matrix.to_numpy().T,color=color,lw=.3,alpha=.045)
            axes[0,col].fill_between(curve.trial,curve.q10,curve.q90,color=color,alpha=.12,lw=0)
            axes[0,col].plot(curve.trial,curve['mean'],color=color,lw=1.5)
            for row,key in enumerate(('present','contribution','precision_given_presence'),1):
                axes[row,col].plot(state.trial,state[key],color=color,lw=1.3)
        axes[0,col].plot(curve.trial,curve.observed,color='#222222',lw=1.1,marker='o',ms=2)
        axes[0,col].plot(curve.trial,curve.conditional,color='#777777',lw=.8,ls='--')
        axes[0,col].set_title(label,fontsize=8)
        axes[3,col].set_xlabel('Trial (32-trial blocks)')
        for row in range(4):
            axes[row,col].set_xlim(curve.training_trials.iloc[0],curve.trial.max())
            if row<3:
                axes[row,col].set(ylim=(-.02,1.02),yticks=[0,.5,1])
            else:
                axes[row,col].set(ylim=(0,25),yticks=[0,10,20])
    for ax,label in zip(axes[:,0],('Accuracy','Target membership','Choice contribution','Precision if present')):
        ax.set_ylabel(label)
    fig.suptitle(f'Model 1022 · S{subject} · autonomous continuation',y=.985,fontsize=10)
    handles=[Line2D([],[],color=COLORS[s],lw=1.5,label=LABELS[s]) for s in STARTS]
    handles += [Line2D([],[],color='#222222',marker='o',ms=2,lw=1,label='Participant'),
                Line2D([],[],color='#777777',ls='--',lw=1,label='Conditional (observed history)')]
    fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,.95),fontsize=7)
    fig.text(.5,.065,'Behavior: all 128 paths per start; shading: central 80%; solid lines: means.',ha='center',fontsize=6.5)
    fig.text(.5,.038,'Continuation: 8 independent filters × 16 branches. Parameters stay fixed.',ha='center',fontsize=6.5)
    fig.text(.5,.012,'State traces precede current feedback. Precision averages only target-present states.',ha='center',fontsize=6.5)
    return save(fig,output,f'S{subject}_continuation.png')


def late_distribution(tables,output):
    fig,axes=plt.subplots(3,3,figsize=(183/25.4,175/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.85,hspace=.42,wspace=.2)
    for row,subject in enumerate((205,306,309)):
        for col,(candidate,label) in enumerate(candidates(tables['scores'],subject)):
            ax=axes[row,col]
            for start in STARTS:
                samples=tables['path_metrics'].query('subject==@subject and candidate==@candidate and start==@start').late_accuracy.to_numpy()
                if len(samples)!=128:
                    raise ValueError('Missing final-accuracy paths')
                ax.step(np.sort(samples),np.arange(1,len(samples)+1)/len(samples),where='post',color=COLORS[start],lw=1.4)
            observed=tables['scores'].query('subject==@subject and candidate==@candidate').late_observed.iloc[0]
            ax.axvline(observed,color='#222222',lw=1,ls=':')
            ax.set(title=f'S{subject} · {label}',xlim=(0,1),ylim=(0,1.02),xticks=[0,.5,1],yticks=[0,.5,1])
            if col==0:ax.set_ylabel('Cumulative path fraction')
            if row==2:ax.set_xlabel('Final 128-trial accuracy')
    fig.suptitle('Final performance distributions · every fixed candidate',y=.985,fontsize=10)
    fig.legend(handles=[Line2D([],[],color=COLORS[s],lw=1.4,label=LABELS[s]) for s in STARTS]+
        [Line2D([],[],color='#222222',ls=':',label='Observed accuracy')],loc='upper center',bbox_to_anchor=(.5,.95),fontsize=7)
    fig.text(.5,.065,'The dotted line locates the participant within each generated performance distribution.',ha='center',fontsize=6.5)
    fig.text(.5,.03,'All paths retained; an observed value beyond the simulated maximum does not imply zero probability.',ha='center',fontsize=6.3)
    return save(fig,output,'late_distributions.png')


def pairing(tables,subject,output):
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,140/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.79,hspace=.35,wspace=.2)
    colors=['#527697','#9B775D','#3B8C85']
    for col,(candidate,label) in enumerate(candidates(tables['scores'],subject)):
        state=tables['states'].query('subject==@subject and candidate==@candidate and start=="filtered_prefix"').sort_values('trial')
        axes[0,col].plot(state.trial,state.target_correct_given_presence,color='#C47742',lw=1.4)
        axes[0,col].plot(state.trial,state.target_correct_mass+state.other_correct_mass,color='#222222',lw=1.2,ls='--')
        for k,color in enumerate(colors):
            axes[1,col].plot(state.trial,state[f'target_pairing_{k}'],color=color,lw=1.2)
        axes[0,col].set_title(label,fontsize=8)
        axes[1,col].set_xlabel('Trial (32-trial blocks)')
        for ax in axes[:,col]:ax.set(ylim=(-.02,1.02),yticks=[0,.5,1])
    axes[0,0].set_ylabel('Correct-choice probability')
    axes[1,0].set_ylabel('Pairing probability\ngiven target presence')
    fig.suptitle(f'S{subject} · judgment and pairing during continuation',y=.985,fontsize=10)
    fig.legend(handles=[Line2D([],[],color='#C47742',lw=1.4,label='Target prediction if present'),
        Line2D([],[],color='#222222',ls='--',label='Full choice mixture')]+
        [Line2D([],[],color=c,label=f'Pairing {k}') for k,c in enumerate(colors)],
        loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,fontsize=7)
    fig.text(.5,.085,'Pairings 0/1/2 follow model order; concentration alone does not establish correct pairing.',ha='center',fontsize=6.5)
    fig.text(.5,.045,'Target prediction conditions on presence; full-mixture prediction averages every path and trial.',ha='center',fontsize=6.3)
    fig.text(.5,.015,'Pairing governs feedback learning. Archived task-start pairing traces were not collected.',ha='center',fontsize=6.5)
    return save(fig,output,f'S{subject}_judgment_pairing.png')


def build(source,output):
    if output.exists():raise FileExistsError('Use a new figure directory')
    tables,done=load_data(source)
    output.mkdir(parents=True);(output/'source_data').mkdir()
    plt.rcParams.update(STYLE)
    figures=[trajectories(tables,s,output) for s in (205,306,309)]
    figures.append(late_distribution(tables,output))
    figures.extend(pairing(tables,s,output) for s in (306,309))
    for file in source.glob('*.csv'):shutil.copy2(file,output/'source_data'/file.name)
    score=tables['scores'][['subject','roles','start','energy','late_observed','late_accuracy','late_q10','late_q90','paths_reaching_observed']]
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 1022 状态续跑</title>
<style>body{max-width:1100px;margin:35px auto;font-family:sans-serif;line-height:1.65}img{max-width:100%;border:1px solid #ddd}table{border-collapse:collapse;font-size:14px}td,th{padding:6px;border:1px solid #ddd}</style>
<h1>Model 1022：前段状态自主续跑</h1>
<p>三个被试各保留三个已冻结参数候选。蓝色从任务起点自主学习；橙色先根据前段真实行为推断状态，再自主完成后段。
后段刺激相同，续跑使用自身选择产生的反馈。所有路径均保留。</p>
<p>每种起点128条路径。续跑的16条分支共享一个滤波近似，共8个独立滤波组。阴影是路径的中央80%范围；表内MCSE只描述固定参数下的计算误差。
末段正确率使用最后128题，曲线按32题分块。该诊断使用开发数据；309既往参与机制设计的影响仍未确定。</p>
<p>状态图依次展示目标规则在场比例、作答权重、在场时精度。类别配对图仅使用本轮续跑记录。
原始起点对照未保存配对过程，因此该项没有蓝色对照。配对编号遵循模型顺序，集中于某项本身不能证明它正确。</p>'''
    body+=score.to_html(index=False,float_format=lambda x:f'{x:.3f}')
    for name in figures:
        body+=f'<p><a href="{html.escape(name)}">{html.escape(name)}</a></p><img src="{html.escape(name)}">'
    (output/'index.html').write_text(body+'</html>')
    write(output/'completion.json',dict(status='complete',source=str(source.resolve()),figures=figures,
        source_completion_sha256=digest(source/'completion.json'),plot_source_sha256=digest(Path(__file__)),
        outputs_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()},
        contract=dict(archetype='quantitative grid',backend='python',width_mm=183,dpi=300,formats=['PNG'],
            paths_per_start=128,exclusions='None within the predefined three-subject nine-candidate study',
            intervals='Central 80% generated path ranges',precision='Conditional on target presence',
            inference='Development comparison; no participant-level significance tests')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.output)
