"""Review conditional H parameter profiles and known-nuisance recovery.

Figure contract: parameter sensitivity and compensation, not biological certainty.
Python quantitative grids, 183-mm PNG at 300 dpi for development review. All
profile points and recovery replicates are retained; no population tests. Profile
losses fix nuisance parameters and use training data only. Recovery fixes nuisance
parameters to their generating values. Rendered inspection is still required.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest,load


def label(parameter: str) -> str:
    return {'decay':'Evidence retention','gamma':'Memory retention','exit_rate':'Exit rate',
        'stay_bias':'Incumbent retention bias','max_capacity':'Capacity','baseline_probability':'Baseline search probability'}[parameter.split('.')[-1]]


def build(profiles: Path, recovery: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new parameter figure destination')
    for source in (profiles,recovery):
        if load(source/'completion.json')['status']!='complete':raise ValueError('Incomplete parameter diagnostic')
    output.mkdir(parents=True)
    protocol=load(profiles/'protocol.json');frozen=load(Path(protocol['fit'])/'nominees.json')
    data=pd.read_csv(profiles/'profiles.csv');estimates=pd.read_csv(recovery/'recovered_decay.csv')
    models=protocol['settings']['profile_models'];subjects=sorted(data.subject.unique())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.spines.right':False,'axes.spines.top':False,
        'axes.linewidth':.6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    image_files=[]
    for subject in subjects:
        fig,axes=plt.subplots(1,len(models),figsize=(183/25.4,80/25.4),layout='constrained',squeeze=False)
        for ax,model in zip(axes.flat,models):
            rows=data[(data.subject==subject)&(data.model==model)&data.second.notna()]
            if rows.profile.nunique()!=1:raise ValueError('Expected one joint profile per model')
            matrix=rows.pivot(index='second',columns='first',values='delta_total_nll').sort_index().sort_index(axis=1)
            if not np.isfinite(matrix.to_numpy()).all() or (matrix.to_numpy()<0).any():
                raise ValueError('Profile increments must be finite and nonnegative before log coloring')
            # The explicit logarithmic color scale retains every grid point
            # while showing weak differences near the conditional optimum.
            im=ax.imshow(np.log10(1+matrix),origin='lower',aspect='auto',cmap='viridis',vmin=0)
            ax.set_xticks(range(len(matrix.columns)),[f'{x:g}' for x in matrix.columns],rotation=45)
            ax.set_yticks(range(len(matrix.index)),[f'{x:g}' for x in matrix.index])
            ax.set(xlabel='Evidence retention',ylabel='Memory retention',title=model)
            point=next(r for r in frozen['nominees'] if r['subject']==subject and r['model']==model)['candidate']['hyperparams']
            first,second=rows.profile.iloc[0].split(' | ')
            ax.plot(list(matrix.columns).index(point[first]),list(matrix.index).index(point[second]),
                    marker='o',ms=5,mfc='none',mec='white',mew=1,label='Frozen parameter estimate')
            fig.colorbar(im,ax=ax,label='log10(1 + total negative log-score increase)')
        fig.suptitle(f'Participant {subject}: two-parameter training sensitivity; other parameters fixed',fontsize=9)
        file=output/f'subject_{subject}_joint.png';fig.savefig(file,dpi=300);plt.close(fig);image_files.append(file)
        for model in models:
            rows=data[(data.subject==subject)&(data.model==model)&data.second.isna()]
            fig,axes=plt.subplots(2,3,figsize=(183/25.4,110/25.4),layout='constrained')
            for ax,(parameter,group) in zip(axes.flat,rows.groupby('profile',sort=False)):
                group=group.sort_values('first')
                ax.errorbar(group['first'],group.delta_anchor_total_nll,
                    yerr=2*group.delta_anchor_total_mc_se,color='#326C98',marker='o',ms=2,lw=1,
                    elinewidth=.6,capsize=1.5)
                ax.axhline(0,color='#777777',lw=.6,ls=':')
                ax.set(xlabel=label(parameter),ylabel='Change in total negative log score')
            for ax in axes.flat[rows.profile.nunique():]:ax.set_visible(False)
            fig.suptitle(f'Participant {subject}; {model}: one-parameter training profiles',fontsize=9)
            file=output/f'subject_{subject}_{model}_axes.png';fig.savefig(file,dpi=300);plt.close(fig);image_files.append(file)
    groups=list(estimates.groupby(['subject','true_decay'],sort=True))
    fig,ax=plt.subplots(figsize=(183/25.4,90/25.4),layout='constrained')
    colors=['#326C98','#397E79','#B67947'];ids=sorted(estimates.subject.unique())
    for index,((subject,truth),group) in enumerate(groups):
        group=group.sort_values('replicate');x=index+np.linspace(-.15,.15,len(group))
        ax.scatter(x,group.candidate_decay,s=25,color=colors[ids.index(subject)%3],zorder=3)
        ax.hlines(truth,index-.35,index+.35,color='#222222',lw=1.2)
    ax.set_xticks(range(len(groups)),[f'{subject}\ntrue {truth:g}' for (subject,truth),_ in groups])
    ax.set(xlabel='Synthetic task schedule and generating retention',ylabel='Recovered evidence retention',ylim=(-.03,1.03),
        title='Synthetic recovery: all other parameters fixed at their known generating values')
    file=output/'decay_recovery.png';fig.savefig(file,dpi=300);plt.close(fig)
    rel=lambda path:escape(os.path.relpath(path,output))
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>H模块参数解释检查</title>',
        '<style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#222}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 8px;border-bottom:1px solid #ddd}details{margin:20px 0}</style>',
        '<h1>H模块参数解释检查</h1>',
        '<p>所有剖面只使用新增被试的训练数据。每次改变一个参数或指定的一对参数，其余参数固定在已冻结的估计点。剖面显示局部敏感性与参数补偿；它们尚未整合其余参数的不确定性。</p>',
        '<p>二维图中白圈是冻结参数点，颜色是各图最低总负对数评分之上的增量，采用log10(1＋增量)颜色刻度。全部网格值保留。单参数曲线相对同批种子重新评分的冻结参数点计算，误差线为差值的正负2倍模拟标准误，负值表示该参数点得到更低评分。模拟误差只衡量滤波的随机波动；参数和被试总体的不确定性须另外检验。完整CSV也提供二维网格与冻结点差值的模拟标准误。</p>',
        '<h2>保持率恢复</h2>',
        '<p>每个条件使用预先指定的一名被试的训练刺激顺序，其余参数固定为该人的ECM_S训练估计值。改变真实保持率并生成模型自己的选择及反馈，然后估计保持率。图中每个点是一条合成序列，横向位置轻微错开便于查看，黑线表示真值。此处只检验保持率，完整参数共同估计的恢复仍待检验。</p>',
        '<img src="decay_recovery.png" alt="全部合成序列的保持率恢复">',
        '<p>恢复结果表同时报告最佳两点的分数差和该差值的模拟标准误；差异接近数值波动时，最优保持率的排序仍不稳定。</p>',
        estimates[['subject','true_decay','replicate','candidate_decay','mean_nll','trials','top_gap_nll','top_gap_mc_se']].to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>全部参数剖面</h2>']
    for subject in subjects:
        files=[f for f in image_files if f.name.startswith(f'subject_{subject}_')]
        html.append(f'<details><summary>被试{subject}</summary>'+''.join(f'<img src="{rel(f)}" alt="{escape(f.stem)}">' for f in files)+'</details>')
    html.append('<ul>'+''.join(f'<li><a href="{rel(p)}">{escape(p.name)}</a></li>' for p in [profiles/'profiles.csv',profiles/'protocol.json',recovery/'recovery_scores.csv',recovery/'truths.json'])+'</ul></html>')
    with (output/'review.html').open('x') as f:f.write('\n'.join(html))
    with (output/'qa.json').open('x') as f:json.dump(dict(backend='python',archetype='quantitative_grid',format='PNG',
        dpi=300,width_mm=183,source_sha256=digest(Path(__file__)),
        inputs_sha256={str(p):digest(p) for p in [profiles/'profiles.csv',recovery/'recovered_decay.csv']},
        recovery_sequences=len(estimates),profile_rows=len(data),exclusions=[],visual_qa='pending'),f,indent=2)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('profiles','recovery','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();build(args.profiles,args.recovery,args.output)


if __name__=='__main__':main()
