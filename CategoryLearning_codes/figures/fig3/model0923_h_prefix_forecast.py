"""All-path figures for task-start and training-state H forecasts.

Python quantitative grid, 183-mm PNG at 300 dpi. Each participant's paired
columns show every generated path on the same future trial range. Correct
probability paths, their mean, sampled-choice intervals, observed choices and
sequential conditional predictions have distinct encodings. Reuse of the H
validation figure is limited to typography, palette and export conventions.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import block_paths, digest, load
from src.Bayesian_state.evaluation.model_0923_h_prefix_forecast import resolve_summary

COLORS = {'task_start': '#326C98', 'filtered_prefix': '#B67947'}


def build(source: Path, output: Path) -> None:
    if output.exists() or (source/'review.html').exists():raise FileExistsError('Use new figure/report destinations')
    manifest = load(source/'comparison_manifest.json'); summary = Path(manifest['summary']); forecast = Path(manifest['forecast'])
    for file, expected in manifest['inputs_sha256'].items():
        if digest(Path(file)) != expected:raise ValueError('Comparison input changed')
    for name, expected in manifest.get('outputs_sha256', {}).items():
        if digest(source / name) != expected:raise ValueError('Comparison score table changed')
    data_role = manifest.get('data_use_status', 'development')
    participant_role = '预留验证被试' if data_role == 'reserved_validation' else '开发被试'
    original, _ = resolve_summary(summary); autonomous = Path(original['autonomous'])
    reference = manifest.get('reference_model', original.get('reference_model', 'B'))
    a_rows = {(r['subject'],r['model']):r for r in load(autonomous/'evaluation.json')['rows']}
    f_rows = {(r['subject'],r['model']):r for r in load(forecast/'forecast.json')['rows']}
    scores = pd.read_csv(source/'scores.csv'); blocks = pd.read_csv(source/'blocks.csv')
    contrasts = pd.read_csv(source/'start_contrasts.csv'); model_contrasts = pd.read_csv(source/'model_contrasts.csv')
    subjects = sorted(scores.subject.unique().tolist()); models = list(dict.fromkeys(scores.model))
    output.mkdir(parents=True)
    preflight = dict(backend='python',archetype='quantitative_grid',purpose=data_role+'_process_diagnosis',
        data_use_status=data_role,
        question='Does the inferred training state sustain learning without future observed-choice conditioning?',
        comparison='same parameters and future stimulus schedule; different state initialization',
        unit='participant; simulation intervals at fixed parameters',
        uncertainty='filter-seed clusters for continuation, complete independent paths for task-start generation',
        reuse='style only: fig3/model0923_h_validation.py',format='PNG',dpi=300,width_mm=183,
        export_exception='repository instructions specify PNG only; development-report raster export',
        source_sha256={name:digest(source/name) for name in ['comparison_manifest.json','scores.csv','blocks.csv','start_contrasts.csv']})
    with (output/'preflight.json').open('x') as handle:json.dump(preflight,handle,indent=2)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'svg.fonttype':'none','pdf.fonttype':42,
        'font.size':7,'axes.titlesize':8,'axes.spines.right':False,'axes.spines.top':False,
        'axes.linewidth':.6,'legend.frameon':False})
    palette = LinearSegmentedColormap.from_list('forecast_change',['#397E79','#FFFFFF','#B67947'])
    fig,axes = plt.subplots(2,1,figsize=(183/25.4,112/25.4),layout='constrained')
    tables = [contrasts,model_contrasts[model_contrasts.start=='filtered_prefix']]
    for ax,table,title in zip(axes,tables,['Training-state forecast minus task-start generation',f'Training-state model minus training-state {reference}']):
        matrix = table.pivot(index='model',columns='subject',values='delta_energy').reindex(index=models,columns=subjects)
        limit = max(1e-4,float(np.abs(matrix.to_numpy()).max()))
        im = ax.imshow(matrix,aspect='auto',cmap=palette,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
        for i in range(len(models)):
            for j in range(len(subjects)):
                value = matrix.iloc[i,j]
                ax.text(j,i,f'{value:+.3f}',ha='center',va='center',fontsize=7,
                    color='white' if abs(value)>.75*limit else '#222222')
        ax.set_xticks(range(len(subjects)),subjects);ax.set_yticks(range(len(models)),models)
        ax.set_title(title);fig.colorbar(im,ax=ax,label='Energy score difference; lower is better')
    axes[-1].set_xlabel('Participant')
    fig.savefig(output/'energy_comparison.png',dpi=300);plt.close(fig)
    counts = {}; hashes = {}
    for subject in subjects:
        fig,axes = plt.subplots(len(models),2,figsize=(183/25.4,(55*len(models)+20)/25.4),
            layout='constrained',sharex=True,sharey=True,squeeze=False)
        for row,model in enumerate(models):
            key = subject,model; a_record,f_record = a_rows[key],f_rows[key]
            a_file = autonomous/'evaluate'/f'subject_{subject}'/model/(a_record['candidate']['id']+'.npz')
            f_file = Path(f_record['target']).with_suffix('.npz')
            for file,record in [(a_file,a_record),(f_file,f_record)]:
                if digest(file)!=record['predictions_sha256']:raise ValueError('Plotted generated paths changed')
                hashes[str(file)] = record['predictions_sha256']
            cut = f_record['training_trials'];length = f_record['forecast_trials']
            with np.load(a_file) as saved:initial = block_paths(saved['correct_probability'],cut,cut+length)
            with np.load(f_file) as saved:
                correct = np.take_along_axis(saved['probabilities'],(saved['categories']-1)[None,:,None],axis=2)[...,0]
                continued = block_paths(correct,0,length)
            for column,(start,paths) in enumerate([('task_start',initial),('filtered_prefix',continued)]):
                ax = axes[row,column]; shade = COLORS[start]
                b = blocks[(blocks.subject==subject)&(blocks.model==model)&(blocks.start==start)].sort_values('first_trial')
                x = (b.first_trial.to_numpy()+b.last_trial.to_numpy())/2-cut
                if paths.shape[1]!=len(b):raise ValueError('Plotted and scored blocks differ')
                np.testing.assert_allclose(paths.mean(axis=0),b.probability_mean,atol=1e-12,rtol=0)
                ax.fill_between(x,b.realized_q10,b.realized_q90,color=shade,alpha=.14)
                one_block = len(x)==1
                ax.plot(x,paths.T,color=shade,alpha=.3 if one_block else .10,lw=.4,
                    marker='.' if one_block else None,ms=2)
                ax.plot(x,b.probability_mean,color=shade,lw=1.6,marker='s' if one_block else None,ms=2.5)
                ax.plot(x,b.conditional_mean,color='#666666',ls='--',lw=.8,marker='D' if one_block else None,ms=2)
                ax.plot(x,b.observed,color='#222222',marker='o',ms=1.8,lw=.8)
                label = 'From task start' if start=='task_start' else 'From training state'
                ax.set(title=f'{model} | {label}',xlim=(1,length),ylim=(0,1.03),yticks=[0,.25,.5,.75,1])
                counts[f'{subject}/{model}/{start}'] = len(paths)
        for ax in axes[:,0]:ax.set_ylabel('Correct probability / rate')
        for ax in axes[-1,:]:ax.set_xlabel('Trials after training (32-trial blocks)')
        handles = [Line2D([],[],color=COLORS['task_start'],lw=1.6,marker='s' if one_block else None,ms=2.5,label='Task-start mean'),
            Line2D([],[],color=COLORS['filtered_prefix'],lw=1.6,marker='s' if one_block else None,ms=2.5,label='Training-state mean'),
            Line2D([],[],color='#666666',ls='--',lw=.8,marker='D' if one_block else None,ms=2,label='Sequential conditional mean'),
            Line2D([],[],color='#222222',marker='o',ms=2,lw=.8,label='Observed'),
            Patch(facecolor='#999999',alpha=.2,label='80% sampled-choice interval')]
        fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=6)
        fig.suptitle(f'Participant {subject}: {cut} training trials; every generated future path shown',fontsize=9)
        fig.savefig(output/f'subject_{subject}.png',dpi=300);plt.close(fig)
    rel = lambda file:escape(os.path.relpath(file,source))
    means = scores.groupby(['model','start'],sort=False)[['energy','crps','interval_coverage','interval_width','probability_mean_mae']].mean()
    html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>H模块训练后自主续跑</title>',
        '<style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#222}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 8px;border-bottom:1px solid #ddd}.scroll{overflow:auto}</style>',
        '<h1>H模块：从任务起点生成与从训练状态续跑</h1>',
        f'<p>{len(subjects)}名{participant_role}，{len(models)}个版本。两种生成使用相同拟合参数和后续刺激日程。训练后续跑先用真实训练选择形成状态分布，随后依靠模型自身的选择和反馈继续学习。逐题条件预测在后续试次中继续吸收此前真实选择。</p>',
        '<p>两种生成使用的起点信息不同，评分差值用于定位发现过程和后续维持的问题。此处固定后续刺激日程；自适应刺激及达标停止的检验另行报告。</p>',
        '<p>每个面板的浅色细线包含该组全部正确概率路径，粗线是等权均值，阴影是实际模拟作答正确率的中央80%区间，黑线是被试，灰色虚线是条件预测。参数估计的不确定性尚未整合到区间中。</p>',
        f'<img src="{rel(output/"energy_comparison.png")}" alt="两种起点与模型间的路径评分差异">',
        '<h2>被试等权评分</h2><div class="scroll">'+means.to_html(float_format=lambda x:f'{x:.4f}')+'</div>',
        '<h2>全部自主路径</h2>']
    for subject in subjects:html.append(f'<h3>被试{subject}</h3><img src="{rel(output/f"subject_{subject}.png")}" alt="被试{subject}全部路径">')
    html.extend(['<h2>逐人结果与数值误差</h2>',
        '<p>续跑误差按完整滤波种子分组删除，每组包含它产生的全部分支；起点生成按完整独立路径删除。标准误表示固定参数下的有限模拟波动，未计入参数或被试总体的不确定性。模型间差值按共同种子配对。</p>',
        '<div class="scroll">'+scores.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>',
        '<div class="scroll">'+model_contrasts.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>',
        '<ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['scores.csv','blocks.csv','model_contrasts.csv','start_contrasts.csv','aggregate_model_contrasts.csv','comparison_manifest.json'])+'</ul></html>'])
    with (source/'review.html').open('x') as handle:handle.write('\n'.join(html))
    with (output/'qa.json').open('x') as handle:
        json.dump(dict(**preflight,paths=sum(counts.values()),path_counts=counts,exclusions=[],
            plotted_array_sha256=hashes,plotting_source_sha256=digest(Path(__file__)),
            visual_qa='pending rendered inspection'),handle,indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();build(args.source.resolve(),args.output.resolve())


if __name__ == '__main__':main()
