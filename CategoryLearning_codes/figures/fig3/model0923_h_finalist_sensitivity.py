"""Render all frozen training-finalist generation means for H comparison.

Python quantitative grids, all candidates retained, PNG only at 183 mm / 300 dpi.
Candidate means use equal simulation budgets. Their spread is a finite-bank
sensitivity diagnostic, with both parameter-point and simulation variation.
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
import pandas as pd

from .model0923_h_validation import COLORS
from src.Bayesian_state.evaluation.model_0923_h_validation import digest,load


def build(source: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new finalist sensitivity figure destination')
    completion=load(source/'completion.json');protocol=load(source/'protocol.json')
    if completion['status']!='complete' or completion['nominees_changed']:raise ValueError('Need frozen completed finalist generation')
    output.mkdir(parents=True)
    scores=pd.read_csv(source/'candidate_scores.csv');blocks=pd.read_csv(source/'candidate_blocks.csv')
    models=scores.model.drop_duplicates().tolist();subjects=sorted(scores.subject.unique())
    count=protocol['settings']['finalists'];paths=protocol['settings']['paths_per_candidate']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.spines.right':False,'axes.spines.top':False,
        'axes.linewidth':.6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    for subject in subjects:
        panel_rows=(len(models)+1)//2
        fig,axes=plt.subplots(panel_rows,2,figsize=(183/25.4,60*panel_rows/25.4),
            layout='constrained',sharex=True,sharey=True,squeeze=False)
        for index,(ax,model) in enumerate(zip(axes.flat,models)):
            rows=blocks[(blocks.subject==subject)&(blocks.model==model)]
            if rows.candidate.nunique()!=count:raise ValueError('Candidate count differs from protocol')
            nominees=rows[rows.frozen_nominee]
            if nominees.candidate.nunique()!=1:raise ValueError('Need exactly one frozen nominee')
            for _,group in rows.groupby('candidate',sort=False):
                group=group.sort_values('first_trial');x=(group.first_trial+group.last_trial)/2
                ax.plot(x,group.probability_mean,color='#AAAAAA',alpha=.8,lw=.7)
            anchor=nominees.sort_values('first_trial');x=(anchor.first_trial+anchor.last_trial)/2
            ax.plot(x,anchor.probability_mean,color=COLORS.get(model,'#326C98'),lw=1.8,label='Frozen nominee mean')
            ax.plot(x,anchor.observed,color='#222222',lw=1,marker='o',ms=2,label='Observed')
            ax.axvline(rows[rows.phase=='training'].last_trial.max()+.5,color='#888888',ls=':',lw=.7)
            ax.set(title=model,ylim=(0,1));ax.set_yticks([0,.25,.5,.75,1])
            if index%2==0:ax.set_ylabel('Correct probability / rate')
            if index+2>=len(models):ax.tick_params(labelbottom=True);ax.set_xlabel('Valid trial (32-trial blocks)')
            if index==0:ax.legend(fontsize=6,loc='best')
        for ax in axes.flat[len(models):]:ax.set_visible(False)
        fig.suptitle(f'Participant {subject}: {count} training finalists; {paths} paths per parameter point',fontsize=9)
        fig.savefig(output/f'subject_{subject}.png',dpi=300);plt.close(fig)
    test=scores[scores.phase=='test']
    ranges=test.groupby(['subject','model'],sort=False).agg(energy_min=('energy','min'),energy_median=('energy','median'),
        energy_max=('energy','max'),mae_min=('probability_mean_mae','min'),mae_max=('probability_mean_mae','max'),
        maximum_training_delta_nll=('training_delta_nll','max')).reset_index()
    rel=lambda p:escape(os.path.relpath(p,source))
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>H模块：训练候选的生成敏感性</title>',
        '<style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#222}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 8px;border-bottom:1px solid #ddd}</style>',
        '<h1>H模块：训练候选的生成敏感性</h1>',
        f'<p>每组保留全部{count}个训练候选，各生成{paths}条完整自主路径。主候选始终采用已冻结的训练选择。当前检验用于判断生成结论对候选参数点的敏感程度。</p>',
        '<p>浅灰线各代表一个候选参数点下的模拟均值，彩色粗线标出主候选，黑线表示观察数据，竖虚线标记训练结束。所有模型均使用相同的候选数和每点路径数；实际被试选择参与事后评分。有限候选点的曲线范围同时包含参数点差异和有限模拟波动，参数后验与置信区间须另行估计。</p>',
        '<h2>全部候选的测试段生成评分范围</h2><p>energy score检验完整块曲线的生成分布；表中最小值、中位数和最大值仅汇总候选范围，主候选保持原训练选择。</p>',
        '<div class="scroll">'+ranges.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>']
    for subject in subjects:html.append(f'<details open><summary>被试{subject}</summary><img src="{rel(output/f"subject_{subject}.png")}" alt="被试{subject}全部候选均值"></details>')
    html.append('<ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['candidate_scores.csv','candidate_blocks.csv','generation.json','protocol.json','completion.json'])+'</ul></html>')
    with (source/'review.html').open('x') as f:f.write('\n'.join(html))
    with (output/'qa.json').open('x') as f:json.dump(dict(backend='python',format='PNG',dpi=300,width_mm=183,
        purpose='development_sensitivity_review',source_sha256=digest(Path(__file__)),
        inputs_sha256={str(p):digest(p) for p in [source/'candidate_scores.csv',source/'candidate_blocks.csv']},
        participant_count=len(subjects),model_count=len(models),candidate_count=count,paths_per_candidate=paths,
        all_paths=completion['paths'],exclusions=[],visual_qa='pending'),f,indent=2)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();build(args.source,args.output)


if __name__=='__main__':main()
