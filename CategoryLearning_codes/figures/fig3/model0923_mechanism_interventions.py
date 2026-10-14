"""PNG review of all fixed-parameter intervention paths, with source tables.

Figure contract: compare performance, discovery and persistence under the
configured interventions; retain participant and path-level variation. The quantitative
grid pairs changes in performance with discovery and persistence. Per-subject
figures show every configured path in 32-trial blocks. Each participant remains
separate; uncertainty is stochastic path variation at fixed fitted parameters.
Python/matplotlib only; 183-mm-wide PNGs, 300 dpi, no publication claim or p tests.
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

ORDER = ['B','S_control','S1','R_control','R1','C1','R0']
COLORS = {'B':'#777777','S_control':'#7D9DAE','S1':'#276788',
          'R_control':'#B395AF','R1':'#704C86','C1':'#C4834C','R0':'#397E79'}
SUBJECTS = [104,215,206,307,222,334,335]


def build(source: Path, output: Path) -> None:
    if output.exists() or (source/'review.html').exists():
        raise FileExistsError('Choose unused figure and report destinations')
    output.mkdir(parents=True)
    summary = pd.read_csv(source/'subjects.csv')
    comparisons = pd.read_csv(source/'comparisons.csv')
    manifest = json.loads((source/'simulation_manifest.json').read_text())
    validation = json.loads((source/'analysis_validation.json').read_text())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.6,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    included = [s for s in SUBJECTS if s in summary.subject.unique()]
    order = [v for v in ORDER if v in summary.variant.unique()]
    paths_per_variant = int(summary.paths.iloc[0])
    no_exit = 'R0' in order
    rows = []
    for metric in ('late_correct','ever_active','late_active'):
        table = comparisons[(comparisons.reference=='B')&(comparisons.metric==metric)]
        for row in table.itertuples():
            rows.append(dict(subject=row.subject,variant=row.target,metric=metric,delta_pp=100*row.delta,
                             paired_mc_se_pp=100*row.paired_mc_se))
    effects = pd.DataFrame(rows)
    effects.to_csv(output/'effects_source.csv',index=False)
    fig,axes = plt.subplots(1,3,figsize=(183/25.4,95/25.4),layout='constrained',sharey=True)
    cmap = LinearSegmentedColormap.from_list('change',['#B56732','#FFFFFF','#326C98'])
    for i,(ax,metric,title,limit) in enumerate(zip(axes,['late_correct','ever_active','late_active'],
            ['Late correct probability','Ever found compatible rule','Late compatible-rule presence'],[50,100,100])):
        matrix = effects[effects.metric==metric].pivot(index='subject',columns='variant',values='delta_pp').reindex(index=included,columns=order[1:])
        im = ax.imshow(matrix.to_numpy(),cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
        for y in range(len(included)):
            for x in range(len(order)-1):
                val=matrix.iloc[y,x]
                ax.text(x,y,f'{val:+.0f}',ha='center',va='center',fontsize=6,color='white' if abs(val)>.65*limit else '#222222')
        ax.set_xticks(range(len(order)-1),[v.replace('_control',' ctrl') for v in order[1:]],rotation=45,ha='right')
        ax.set_yticks(range(len(included)),[str(s) for s in included])
        ax.set_title(title)
        ax.text(-.08,1.04,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.7,label='Change from B (percentage points)',orientation='horizontal',pad=.15)
    axes[0].set_ylabel('Participant')
    fig.savefig(output/'mechanism_effects.png',dpi=300)
    plt.close(fig)
    if no_exit:
        paired = comparisons[(comparisons.target=='R0')&(comparisons.reference=='R1')&
            comparisons.metric.isin(['late_correct','late_incompatible_contribution'])]
        paired.to_csv(output/'retention_comparison_source.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(183/25.4,90/25.4),layout='constrained',sharey=True)
        for ax,metric,title in zip(axes,['late_correct','late_incompatible_contribution'],
                ['Late correct probability','Late incompatible-rule contribution']):
            values=paired[paired.metric==metric].set_index('subject').reindex(included)
            ax.axvline(0,color='#999999',lw=.7)
            ax.errorbar(values.delta*100,np.arange(len(included)),xerr=values.paired_mc_se*100,
                        fmt='o',color=COLORS['R0'],ms=4,capsize=3,lw=1)
            ax.set_yticks(range(len(included)),[str(s) for s in included])
            ax.set_title(title);ax.set_xlabel('R0 minus R1 (percentage points)')
        axes[0].invert_yaxis();axes[0].set_ylabel('Participant')
        fig.suptitle('Complete-path mean difference; whiskers: one Monte Carlo SE',fontsize=8)
        fig.savefig(output/'retention_comparison.png',dpi=300)
        plt.close(fig)
    path_rows=[]
    for sid in included:
        record = next(r for r in manifest['records'] if r['subject']==sid)
        observed = pd.read_csv(source/record['id']/'observed_trials.csv')
        cols = 2 if len(order)<=4 else 3
        nrows = int(np.ceil(len(order)/cols))
        fig,axes=plt.subplots(nrows,cols,figsize=(183/25.4,130/25.4),layout='constrained',sharex=True,sharey=True,squeeze=False)
        for ax,variant in zip(axes.flat,order):
            with np.load(source/record['id']/variant/'correct_paths.npz') as z:
                correct=z['correct_probability']; seeds=z['seeds']
            blocks=np.array([correct[t:t+32].mean(axis=0) for t in range(0,len(correct),32)])
            xs=np.array([(t+1+min(t+32,len(correct)))/2 for t in range(0,len(correct),32)])
            actual=np.array([(observed.choice==observed.category).iloc[t:t+32].mean() for t in range(0,len(correct),32)])
            for path in range(blocks.shape[1]):
                ax.plot(xs,blocks[:,path],color=COLORS[variant],lw=.35,alpha=.13)
                for block,value in enumerate(blocks[:,path]):
                    path_rows.append(dict(subject=sid,variant=variant,path=path,seed=int(seeds[path]),
                        first_trial=block*32+1,last_trial=min((block+1)*32,len(correct)),correct_probability=value))
            ax.plot(xs,blocks.mean(axis=1),color=COLORS[variant],lw=1.7,label='Model mean')
            ax.plot(xs,actual,color='#111111',lw=.9,marker='o',ms=1.5,label='Observed')
            ax.set(title=variant,ylim=(0,1.03),xlim=(1,len(correct)))
            ax.set_yticks([0,.25,.5,.75,1])
        for ax in axes.flat[len(order):]: ax.set_visible(False)
        axes[0,0].legend(loc='lower right',fontsize=6)
        for ax in axes[:,0]: ax.set_ylabel('Correct response probability / rate')
        for ax in axes[-1,:]: ax.set_xlabel('Valid trial (32-trial blocks)')
        fig.suptitle(f'Participant {sid}: {blocks.shape[1]} autonomous paths per variant',fontsize=9)
        fig.savefig(output/f'subject_{sid}_paths.png',dpi=300)
        plt.close(fig)
    pd.DataFrame(path_rows).to_csv(output/'paths_source.csv',index=False)
    qa=dict(backend='python',archetype='quantitative_grid',participants=included,
        variants=order,paths_per_variant=paths_per_variant,exclusions=[],format='PNG',dpi=300,
        width_mm=183,center='equal-weight path mean',background='all complete paths, 32-trial nonoverlapping block means',
        comparison='within participant, fixed fitted parameters, shared seeds',
        uncertainty='path variation; no parameter or population uncertainty',
        tests='no significance tests; paired Monte Carlo SE in source tables',
        boundaries='full-sequence fitted development participants; no held-out validation',
        source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('subjects.csv','comparisons.csv','paths.csv')},
        source_rows=len(pd.DataFrame(path_rows)),visual_qa='rendered output inspection required')
    (output/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    rel=lambda p:escape(os.path.relpath(p,source))
    performance=summary.pivot(index='subject',columns='variant',values='late_correct').reindex(index=included,columns=order)
    performance['被试实际']=summary.groupby('subject').late_observed.first().reindex(included)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>0923 机制定向对照</title>',
        '<style>body{max-width:1100px;margin:35px auto;font:16px/1.7 sans-serif;color:#222;padding:0 20px}table{border-collapse:collapse;font-size:14px}th,td{padding:6px 10px;border-bottom:1px solid #ddd;text-align:right}img{width:100%;height:auto}a{color:#276788}details{margin:24px 0}</style>',
        '<h1>Model 0923：'+('关闭自主退出的结构对照' if no_exit else '机制定向对照')+'</h1>',
        f'<p>沿用原拟合参数和真实刺激顺序，仅施加配置中列出的机制干预，每人每版本{paths_per_variant}条自主路径。B为原full-v3；R_control将退出率乘0.25；R1按最近至多4题反馈证据控制退出。'+('R0将退出率设为0，保留搜索和容量替换。' if no_exit else 'S_control将搜索赔率乘4；S1累积失败并在成功后衰减；C1让新旧规则共同竞争容量。')+'</p>',
        '<p>同时查看行为曲线、相容规则发现和保留、非相容规则的驻留与作答贡献。各指标按被试分别呈现。</p>',
        f'<p>合计{manifest["paths"]}条路径；新增模拟{manifest.get("new_paths",manifest["paths"])}条，经校验复用{manifest.get("reused_paths",0)}条。新增运行使用{manifest["actual_workers"]}个工作进程，数值线程为1，用时{manifest["elapsed_seconds"]:.1f}秒。保留{validation["baseline_exact_paths"]}条路径与独立基线的逐项一致校验记录（含复用记录和零效干预）。</p>',
        '<h2>后期表现</h2><p>各值为最后64题的平均正确概率；被试列为实际正确率。</p>',performance.to_html(float_format=lambda x:f'{x:.3f}',border=0),
        f'<img src="{rel(output/"mechanism_effects.png")}" alt="三个指标相对基线的逐人变化">',
        '<h2>完整路径分布</h2><p>每条浅色线对应一条完整自主路径；粗线是等权均值，黑线是被试实际正确率。图中按连续32题取块均值。所有路径均保留，没有按贴合程度筛选。背景展示固定参数下的过程变异。</p>']
    for sid in included:
        body.append(f'<details><summary>被试{sid}：{len(order)}个版本、每组全部{paths_per_variant}条路径</summary><img src="{rel(output/f"subject_{sid}_paths.png")}" alt="被试{sid}的全部路径"></details>')
    if no_exit:
        body.append(f'<img src="{rel(output/"retention_comparison.png")}" alt="R0与R1的完整路径配对差值及模拟标准误">')
        fields=['subject','variant','mean_active_candidates','late_incompatible_candidates',
                'late_incompatible_contribution','total_autonomous_exits','total_capacity_replacements',
                'wrong_clearance_fraction','wrong_clearance_episodes']
        body.extend(['<h2>容量占用、作答贡献与清除</h2>',
            '<p>相容规则的几何准确率阈值为0.8，低于阈值的规则在此记为非相容。容量占用与对回答的贡献分别统计。total开头的事件列为每条路径的平均次数；wrong_clearance_fraction为4题证据窗口下8题内清除的比例。</p>',
            summary[fields].to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0),
            '<p><a href="clearance_sensitivity.csv">反证窗口1、4、8题 × 清除期限8、32、64题</a> · '
            '<a href="incompatible_residences.csv">全部非相容规则驻留</a>。窗口仅用于事后评分，R0的认知机制没有证据窗口。</p>'])
    body.extend(['<h2>过程指标与解释边界</h2>',
        '<p>任务相容规则的物理刺激分类准确率阈值预设为0.8，仅用于事后评分。保留比例表示有足够观察长度的驻留中持续至少8题的比例；错误规则清除比例表示首次近期净反证后8题内退出的比例，末尾观察不足另标删失。两项分母随机制变化，须同时阅读事件数。</p>',
        '<p>共同种子保留配对关系，机制分叉后随机抽样顺序可能变化。comparisons.csv给出完整路径差值的Monte Carlo标准误，batch_comparisons.csv保留4组各16条路径的方向检查。这些量描述固定参数模拟波动。当前被试及其全序列拟合已参与模型开发，等预算重拟合、真正留出预测与参数恢复仍待完成。</p>',
        '<p>每32题实际正确率的80%模拟区间覆盖及宽度在subjects.csv中报告；它们是相关时间块上的描述性诊断。曲线均值误差、覆盖和过程指标需要共同判断。</p>',
        '<ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['subjects.csv','paths.csv','comparisons.csv','batch_comparisons.csv','set_residences.csv','exits.csv','wrong_rule_clearance.csv','curves.csv','analysis_validation.json','simulation_manifest.json'])+'</ul>',
        f'<details><summary>{len(summary)}组结果完整表</summary>'+summary.to_html(index=False,float_format=lambda x:f'{x:.4g}',border=0)+'</details></html>'])
    (source/'review.html').write_text('\n'.join(body),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.output)
