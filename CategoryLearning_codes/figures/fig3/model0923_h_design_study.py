"""Review H structure screens with all participants, variants and complete paths.

Contract: mechanisms differ in discovery, retention and whole learning curves;
higher terminal performance alone does not establish a mechanism. Quantitative
grid, Python/matplotlib, 183-mm PNG at 300 dpi for a development report. Heatmaps
show all configured variants; paginated trajectory panels retain every path.
Participant is the behavioral unit, whole path is the Monte Carlo unit. No
population tests or confidence claims; fitted full-sequence parameters are used.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


def color(variant: str) -> str:
    if variant=='B':return '#777777'
    if variant=='R1':return '#704C86'
    if variant.startswith('J'):return '#B67947'
    if variant.endswith('_S'):return '#668548'
    return '#397E79' if variant.startswith('EC') else '#326C98'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, output: Path, report_name: str = 'review.html') -> None:
    if Path(report_name).name!=report_name or not report_name.endswith('.html'):raise ValueError('Need a local HTML filename')
    if output.exists() or (source/report_name).exists():raise FileExistsError('Use new figure and report destinations')
    output.mkdir(parents=True)
    settings=json.loads((source/'simulation_manifest.json').read_text())
    summary=pd.read_csv(source/'subjects.csv')
    variants=list(settings['settings']['variants'])
    subjects=[r['subject'] for r in settings['records']]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.spines.right':False,'axes.spines.top':False,
        'axes.linewidth':.6,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    palette=LinearSegmentedColormap.from_list('contrast',['#B67947','#FFFFFF','#326C98'])
    matrices=[]
    for metric,title in [('late_correct','Late correct probability'),('block_mean_absolute_error','Whole-curve mean absolute error'),
                          ('late_active','Late compatible-rule presence')]:
        matrix=summary.pivot(index='variant',columns='subject',values=metric).reindex(index=variants,columns=subjects)
        delta=matrix.subtract(matrix.loc['B'],axis=1)*100
        fig,ax=plt.subplots(figsize=(183/25.4,max(80,5*len(variants)+25)/25.4),layout='constrained')
        limit=max(1.,float(np.nanmax(abs(delta.to_numpy()))))
        im=ax.imshow(delta.to_numpy(),aspect='auto',cmap=palette,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
        for i,variant in enumerate(variants):
            for j,subject in enumerate(subjects):
                value=delta.iloc[i,j]
                ax.text(j,i,f'{value:+.0f}',ha='center',va='center',fontsize=6,
                        color='white' if abs(value)>.7*limit else '#222222')
                matrices.append(dict(metric=metric,subject=subject,variant=variant,value=matrix.iloc[i,j],difference_pp=value))
        ax.set_xticks(range(len(subjects)),list(map(str,subjects)))
        ax.set_yticks(range(len(variants)),variants)
        ax.set_xlabel('Participant');ax.set_title(title+' — change from B')
        fig.colorbar(im,ax=ax,label='Percentage points; '+('lower is better' if metric=='block_mean_absolute_error' else 'descriptive difference'))
        fig.savefig(output/(metric+'.png'),dpi=300);plt.close(fig)
    pd.DataFrame(matrices).to_csv(output/'heatmap_source.csv',index=False)
    path_images={};path_counts={};source_hashes={}
    for record in settings['records']:
        subject=record['subject'];observed=pd.read_csv(source/record['id']/'observed_trials.csv')
        actual=(observed.choice==observed.category).to_numpy()
        path_images[subject]=[]
        for page,start in enumerate(range(0,len(variants),4),1):
            group=variants[start:start+4]
            rows=2 if len(group)>2 else 1;columns=2 if len(group)>1 else 1
            fig,axes=plt.subplots(rows,columns,figsize=(183/25.4,(125 if rows==2 else 70)/25.4),
                                 layout='constrained',sharex=True,sharey=True,squeeze=False)
            for ax,variant in zip(axes.flat,group):
                file=source/record['id']/variant/'correct_paths.npz'
                source_hashes[str(file.relative_to(source))]=digest(file)
                with np.load(file) as z:paths=z['correct_probability']
                if paths.shape[0]!=len(actual):raise ValueError('Observed and generated trial orders differ')
                path_counts[f'{subject}/{variant}']=paths.shape[1]
                intervals=[(t,min(t+32,len(actual))) for t in range(0,len(actual),32)]
                x=[(a+b+1)/2 for a,b in intervals]
                block=np.array([paths[a:b].mean(axis=0) for a,b in intervals])
                behavior=[actual[a:b].mean() for a,b in intervals]
                ax.plot(x,block,color=color(variant),alpha=.10,lw=.4)
                ax.plot(x,block.mean(axis=1),color=color(variant),lw=1.8,label='Equal-weight model mean')
                ax.plot(x,behavior,color='#222222',lw=.8,marker='o',ms=1.5,label='Observed')
                ax.set(title=variant,xlim=(1,len(actual)),ylim=(0,1.03),yticks=[0,.25,.5,.75,1])
            for ax in axes.flat[len(group):]:ax.set_visible(False)
            for ax in axes[:,0]:ax.set_ylabel('Correct probability / rate')
            for ax in axes[-1,:]:ax.set_xlabel('Valid trial (32-trial blocks)')
            axes[0,0].legend(loc='lower right',fontsize=6)
            fig.suptitle(f'Participant {subject}; all {paths.shape[1]} paths per variant; page {page}',fontsize=9)
            name=f'subject_{subject}_page_{page}.png'
            fig.savefig(output/name,dpi=300);plt.close(fig)
            path_images[subject].append(output/name)
    rel=lambda p:escape(os.path.relpath(p,source))
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>H模块结构研究</title>',
        '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#222}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 8px;border-bottom:1px solid #ddd}details{margin:18px 0}.scroll{overflow:auto}</style>',
        '<h1>Model 0923：H模块结构筛查</h1>',
        f'<p>{len(subjects)}名开发被试，{len(variants)}个版本，共{sum(path_counts.values())}条完整自主路径。沿用原全序列拟合参数，按相同种子比较。此轮属于固定参数筛查。</p>',
        '<p>热图保留所有版本和被试。每张路径图的背景包含该组全部路径，粗线为等权均值，黑线为实际正确率。连续32题取块均值，最后一块使用剩余题数。路径差异表示固定参数下的模拟变异。</p>']
    for metric in ['late_correct','block_mean_absolute_error','late_active']:
        html.append(f'<img src="{rel(output/(metric+".png"))}" alt="{metric}">')
    html.extend(['<h2>机制配置</h2><pre>'+escape(json.dumps(settings['settings']['variants'],ensure_ascii=False,indent=2))+'</pre>',
        '<h2>全部路径</h2>'])
    for subject,files in path_images.items():
        html.append(f'<details><summary>被试{subject}：全部版本、全部路径</summary>'+''.join(f'<img src="{rel(p)}" alt="被试{subject}路径">' for p in files)+'</details>')
    html.extend(['<h2>指标与数据</h2>',
        '<p>任务相容规则按物理刺激几何准确率≥0.8事后界定。联合集合移除、原自主退出和容量替换分别记录。所有事件比例须连同分母解释；试次与同一路径内事件相互依赖。未进行人群显著性检验。</p>',
        '<ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['subjects.csv','paths.csv','curves.csv','comparisons.csv','batch_comparisons.csv','incompatible_residences.csv','clearance_sensitivity.csv','simulation_manifest.json','analysis_validation.json'])+'</ul>',
        '<div class="scroll">'+summary.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)+'</div></html>'])
    with (source/report_name).open('x') as f:f.write('\n'.join(html))
    qa=dict(backend='python',archetype='quantitative_grid',format='PNG',dpi=300,width_mm=183,
        purpose='development_report',paths=sum(path_counts.values()),path_counts=path_counts,exclusions=[],
        uncertainty='whole-path variation at fixed fitted parameters; no population significance tests',
        data_boundary='development participants with whole-sequence fitted parameters',
        source_sha256={**source_hashes,**{name:digest(source/name) for name in ['subjects.csv','simulation_manifest.json']}},
        plotting_source_sha256=digest(Path(__file__)),
        visual_qa='pending rendered inspection')
    with (output/'qa.json').open('x') as f:json.dump(qa,f,indent=2)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report-name',default='review.html')
    args=p.parse_args();build(args.source.resolve(),args.output.resolve(),args.report_name)


if __name__=='__main__':main()
