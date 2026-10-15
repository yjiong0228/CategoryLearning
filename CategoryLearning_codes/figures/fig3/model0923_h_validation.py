"""Python review figures for the frozen six-participant H validation.

Figure contract: compare sequential held-out choice scores with whole autonomous
path distributions. Quantitative grid; all participants/models/paths retained.
183-mm PNG, 300 dpi, development report. Model mean, realized predictive interval,
observed curve and sequential conditional mean have distinct encodings. Filter
seed uncertainty is numerical; path variation is fixed-parameter predictive
variation. Participant is the behavioral unit. No population significance claims.
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
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import block_paths,digest,load

COLORS={'B':'#777777','R1':'#704C86','E':'#326C98','ECM':'#397E79','ECM_S':'#668548','JS_S':'#B67947',
        'ECM_S_R0':'#397E79','JF_sum':'#B67947'}


def build(source: Path, output: Path) -> None:
    if output.exists() or (source/'review.html').exists():raise FileExistsError('Use new destinations')
    output.mkdir(parents=True)
    manifest=load(source/'summary_manifest.json');fit=Path(manifest['fit'])
    frozen=load(fit/'nominees.json');autonomous=Path(manifest['autonomous'])
    development = frozen.get('data_use_status') == 'development' or frozen.get('status') == 'frozen_for_development_evaluation'
    models=manifest['models'];subjects=manifest['subjects']
    scores=pd.read_csv(source/'scores.csv');contrasts=pd.read_csv(source/'test_contrasts.csv')
    aggregate=pd.read_csv(source/'aggregate_choice_scores.csv')
    inactive=pd.read_csv(source/'inactive_parameter_channels.csv')
    blocks=pd.read_csv(source/'blocks.csv');parameters=pd.read_csv(source/'parameters.csv')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.spines.right':False,'axes.spines.top':False,
        'axes.linewidth':.6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    palette=LinearSegmentedColormap.from_list('score_change',['#397E79','#FFFFFF','#B67947'])
    fig,axes=plt.subplots(3,1,figsize=(183/25.4,165/25.4),layout='constrained')
    for ax,metric,title in zip(axes,['choice_nll','energy','crps'],
        ['Sequential choice negative log score','Autonomous path energy score','Autonomous block CRPS']):
        data=contrasts.pivot(index='model',columns='subject',values='delta_'+metric).reindex(index=[m for m in models if m!='B'],columns=subjects)
        limit=max(1e-4,float(abs(data).to_numpy().max()))
        im=ax.imshow(data,aspect='auto',cmap=palette,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
        for i in range(len(data)):
            for j in range(len(subjects)):
                value=data.iloc[i,j];ax.text(j,i,f'{value:+.3f}',ha='center',va='center',fontsize=7,
                    color='white' if abs(value)>.75*limit else '#222222')
        ax.set_xticks(range(len(subjects)),subjects);ax.set_yticks(range(len(data)),data.index)
        ax.set_title(title+' — test difference from B; lower is better')
        fig.colorbar(im,ax=ax,label='Score difference')
    axes[-1].set_xlabel('Participant')
    fig.savefig(output/'test_score_comparison.png',dpi=300);plt.close(fig)
    path_counts={}
    for subject in subjects:
        panel_rows=(len(models)+1)//2
        fig,axes=plt.subplots(panel_rows,2,figsize=(183/25.4,60*panel_rows/25.4),
                             layout='constrained',sharex=True,sharey=True,squeeze=False)
        for ax,model in zip(axes.flat,models):
            nominee=next(r for r in frozen['nominees'] if r['subject']==subject and r['model']==model)
            file=autonomous/'evaluate'/f'subject_{subject}'/model/(nominee['candidate']['id']+'.npz')
            with np.load(file) as z:p=z['correct_probability']
            b=blocks[(blocks.subject==subject)&(blocks.model==model)].sort_values('first_trial')
            cut=int(b.loc[b.phase=='training','last_trial'].max());n=p.shape[1]
            generated=np.concatenate([block_paths(p,0,cut),block_paths(p,cut,n)],axis=1)
            x=(b.first_trial.to_numpy()+b.last_trial.to_numpy())/2
            if generated.shape[1]!=len(b):raise ValueError('Figure blocks differ from scored blocks')
            shade=COLORS.get(model,'#326C98')
            ax.fill_between(x,b.realized_q10,b.realized_q90,color=shade,alpha=.12,label='80% sampled-choice interval')
            ax.plot(x,generated.T,color=shade,alpha=.08,lw=.4)
            ax.plot(x,b.autonomous_probability_mean,color=shade,lw=1.7,label='Autonomous probability mean')
            ax.plot(x,b.conditional_probability_mean,color='#666666',ls='--',lw=.8,label='Sequential conditional mean')
            ax.plot(x,b.observed,color='#222222',marker='o',ms=1.4,lw=.8,label='Observed')
            ax.axvline(cut+.5,color='#555555',ls=':',lw=.8)
            ax.set(title=model,xlim=(1,n),ylim=(0,1.03),yticks=[0,.25,.5,.75,1])
            path_counts[f'{subject}/{model}']=len(p)
        for ax in axes.flat[len(models):]:ax.set_visible(False)
        for ax in axes[:,0]:ax.set_ylabel('Correct probability / rate')
        for ax in axes[-1,:]:ax.set_xlabel('Valid trial (32-trial blocks)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=6)
        fig.suptitle(f'Participant {subject}: training | test; all {len(p)} autonomous paths per model',fontsize=9)
        fig.savefig(output/f'subject_{subject}.png',dpi=300);plt.close(fig)
    rel=lambda path:escape(os.path.relpath(path,source))
    test=scores[scores.phase=='test']
    means=test.groupby('model',sort=False)[['choice_nll','energy','crps','interval_score','interval_coverage','interval_width']].mean().reindex(models)
    report_title = 'H模块：开发被试参数复核后的比较' if development else 'H模块：新增被试的冻结方案比较'
    boundary = (f'{len(subjects)}名开发被试；参数更新只用训练前缀，并经过独立随机种子的数值确认。后续序列已在此前机制开发中查看，本报告记录开发复评。'
                if development else f'{len(subjects)}名预先按编号选定的被试；每人{len(models)}个结构、相同搜索预算。全部参数只用训练前缀估计；所有候选冻结后才计算测试段评分。')
    html=[f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{report_title}</title>',
        '<style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#222}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 8px;border-bottom:1px solid #ddd}.scroll{overflow:auto}details{margin:20px 0}</style>',
        f'<h1>{report_title}</h1>',
        f'<p>{boundary}</p>',
        '<p>逐题条件预测先预测当前选择，再吸收真实选择更新状态。自主生成从任务起点开始，使用训练所得参数和模型自己的选择及反馈；任务刺激顺序与序列长度沿用实际实验。</p>',
        '<p>预测区间由模拟的实际作答正确率计算，包含路径和作答变异。浅色细线是全部自主路径的正确概率，粗线是等权均值，黑线是被试，灰色虚线是条件预测。竖虚线标记训练／测试边界。参数保持为估计点，区间未整合参数估计的不确定性。</p>',
        f'<img src="{rel(output/"test_score_comparison.png")}" alt="全部留出评分对照">',
        '<h2>测试段：逐人评分与等权平均</h2>',
        '<p>负对数评分检验每题的实际选项；energy score检验整条块曲线分布，CRPS检验各块分布，interval score同时计入区间宽度及未覆盖观测的距离。这些评分均越低越好。区间覆盖率和宽度合并解释。算法定义见<a href="https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf">Gneiting与Raftery，2007</a>。同一被试的块并不独立，这里报告描述性结果。</p>',
        '<div class="scroll">'+means.to_html(float_format=lambda x:f'{x:.4f}')+'</div>',
        '<p>以下另列按试次数加权的选择评分与总负对数评分。每名被试等权的结果和全部试次的总评分共同保留；较长序列在总评分中占更多权重。</p>',
        '<div class="scroll">'+aggregate.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>',
        '<div class="scroll">'+test[['subject','model','trials','choice_nll','choice_nll_mc_se','energy','crps','interval_score','interval_coverage','interval_width']].to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>',
        '<p>choice_nll_mc_se由完整滤波种子的删一法计算，只表示数值误差。参数解释还需检查不同参数能否产生相近行为，以及模拟数据能否恢复原设定。相关方法见<a href="https://elifesciences.org/articles/49547">Wilson与Collins，2019</a>。</p>',
        '<h2>全部路径</h2>']
    for subject in subjects:html.append(f'<details open><summary>被试{subject}</summary><img src="{rel(output/f"subject_{subject}.png")}" alt="被试{subject}全部路径"></details>')
    html.extend(['<h2>参数与可复核数据</h2><p>以下为有限搜索预算下选出的参数点。不同结构的参数对应各自的更新机制，数值须连同机制解释。</p>',
        '<div class="scroll">'+parameters.to_html(index=False)+'</div>',
        '<p>下表标记当前参数点上明确关闭的H作用通道。对应参数的数值须结合这一条件解释。未列出的参数仍需要敏感性与恢复检验。</p>',
        '<div class="scroll">'+inactive.to_html(index=False)+'</div>',
        '<ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['scores.csv','blocks.csv','parameters.csv','test_contrasts.csv','aggregate_choice_scores.csv','inactive_parameter_channels.csv','summary_manifest.json'])+'</ul></html>'])
    with (source/'review.html').open('x') as f:f.write('\n'.join(html))
    qa=dict(backend='python',archetype='quantitative_grid',format='PNG',dpi=300,width_mm=183,
        purpose='development_report',paths=sum(path_counts.values()),path_counts=path_counts,exclusions=[],
        source_sha256=digest(Path(__file__)),summary_sha256=digest(source/'summary_manifest.json'),
        visual_qa='pending rendered inspection')
    with (output/'qa.json').open('x') as f:json.dump(qa,f,indent=2)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();build(args.source.resolve(),args.output.resolve())


if __name__=='__main__':main()
