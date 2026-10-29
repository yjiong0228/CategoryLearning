"""Render conditional state explanations and finite-parameter comparisons.

Python quantitative grids, PNG only. Candidate identity has a fixed colour;
state probabilities use all trials. Behaviour uses a trailing 32-trial average,
and block displays aggregate every consecutive 64 trials, retaining the tail.
No curve is an autonomous simulation or a parameter-averaged posterior.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/model1024_interpretation_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from ..analysis.model_1024_state_interpretation import MEASURES, read, write, digest
from ..analysis.model_1024_state_interpretation_summary import load_arrays, behavior, calculation_record

COLORS = ['#277da8', '#d07b32', '#7b66a5', '#408c78']
STYLES = ['-', '--', '-.', ':']
NAMES = dict(target_present='正确规则在场概率', target_belief='正确规则平均信念',
    target_high_belief='在场且信念≥0.5的概率', target_execution='正确规则执行概率',
    belief='全部规则信念分布', execution='全部规则执行分布')
SHORT = ['在场', '信念', '高信念', '执行', '全部信念', '全部执行']
MODES = dict(filtered='过滤：使用截至本题的选择', full='完整历史：进一步使用后续选择')
STATUS = dict(close_within_checked_set='已检查候选中接近', parameter_sensitive='随参数改变',
    numerically_limited='计算差异仍影响判断', single_candidate='仅一组，无法比较',
    diagnostic_parameters_only='参数尚未确立')


def configure() -> None:
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.8,'legend.frameon':False,'axes.unicode_minus':False,
        'svg.fonttype':'none','pdf.fonttype':42})


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output/name,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def overview(sid: int, own: list[dict], cases: dict, arrays: dict, checks: pd.DataFrame, output: Path) -> None:
    case = cases[own[0]['case']]; obs = behavior(case); n = case['trials']; x = np.arange(1,n+1)
    fig=plt.figure(figsize=(12,11),layout='constrained')
    grid=fig.add_gridspec(5,2,height_ratios=[.75,1,1,1,1])
    ax=fig.add_subplot(grid[0,:])
    ax.plot(x,pd.Series(obs['correct']).rolling(32,min_periods=1).mean(),color='#202020',lw=1.4)
    ax.set(ylim=(-.02,1.02),xlim=(1,n),ylabel='真实正确率',title=f'S{sid} · 同一行为记录下，各组参数的状态解释')
    ax.text(.01,.06,'黑线为最近32题正确率；下方彩线为状态概率或平均信念',transform=ax.transAxes,fontsize=8)
    if sid==129:
        ax.text(.99,.08,'诊断参数：个体参数集合尚未确立',ha='right',transform=ax.transAxes,color='#955327')
    for row,field in enumerate(MEASURES[:4],1):
        for col,mode in enumerate(MODES):
            ax=fig.add_subplot(grid[row,col])
            usable=[]
            for item in own:
                i=item['number']-1; cid=item['case']
                flag=bool(checks.loc[checks.case.eq(cid)&checks['mode'].eq(mode)&checks.measure.eq(field),'usable'].iloc[0])
                if flag:
                    usable.append(f'P{i+1}')
                ax.plot(x,arrays[cid][mode+'_'+field],color=COLORS[i],ls=STYLES[i],lw=.95,
                    label=f'P{i+1} · {item["candidate"][:6]}')
            ax.set(ylim=(-.02,1.02),xlim=(1,n),ylabel=NAMES[field] if col==0 else '',
                   xlabel='有效试次' if row==4 else '')
            ax.text(.99,.04,'数值检查通过：'+('、'.join(usable) if usable else '无'),ha='right',transform=ax.transAxes,fontsize=7.5,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
            if row==1:
                ax.set_title(MODES[mode]);ax.legend(loc='lower right',bbox_to_anchor=(1,.12),fontsize=7)
    fig.supxlabel('同色同线型始终对应同一组参数；各面板右下列出该量通过检查的候选。状态时点：本题选择后、反馈前。',fontsize=8)
    save(fig,output,f'S{sid}_overview.png')


def scope_figure(selected: list[dict], checks: pd.DataFrame, output: Path) -> None:
    own=[x for x in selected if x['parameter_status']=='retained']
    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained',sharey=True)
    for col,mode in enumerate(MODES):
        ax=axes[col]; matrix=np.zeros((len(own),len(MEASURES))); labels=[]
        for i,item in enumerate(own):
            labels.append(f'S{item["subject"]} P{item["number"]}')
            for j,field in enumerate(MEASURES):
                r=checks.loc[checks.case.eq(item['case'])&checks['mode'].eq(mode)&checks.measure.eq(field)].iloc[0]
                matrix[i,j]=int(r.usable)
                val=max(r.repeat_q95,r.resolution_q95)*100
                ax.text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=8)
        ax.imshow(matrix,cmap=ListedColormap(['#eeeeee','#c0dce8']),vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(len(MEASURES)),SHORT,rotation=30,ha='right')
        ax.set_yticks(range(len(own)),labels);ax.set_title('过滤' if mode=='filtered' else '完整历史')
        ax.set_xticks(np.arange(-.5,len(MEASURES),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(own),1),minor=True)
        ax.grid(which='minor',color='white',lw=1.4);ax.tick_params(which='minor',length=0)
    fig.suptitle('哪些状态量的重复计算和粒子数检查达到工作容忍？',fontsize=12)
    fig.supxlabel('蓝底：完成次数、权重、逐量平均差与95%分位差均通过；灰底：至少一项受限。\n数字：重复与粒子数比较中较大的95%分位差（个百分点）；全部规则两列使用总变差距离。',fontsize=8)
    save(fig,output,'state_use_scope.png')


def sensitivity_figure(frame: pd.DataFrame, output: Path) -> None:
    subjects=sorted(frame.subject.unique()); codes={'close_within_checked_set':0,'parameter_sensitive':1,
        'numerically_limited':2,'single_candidate':3,'diagnostic_parameters_only':3}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    for col,mode in enumerate(MODES):
        ax=axes[col]; matrix=np.empty((len(subjects),len(MEASURES)))
        for i,sid in enumerate(subjects):
            for j,field in enumerate(MEASURES):
                r=frame.loc[frame.subject.eq(sid)&frame['mode'].eq(mode)&frame.measure.eq(field)].iloc[0]
                matrix[i,j]=codes[r.status]
                label='—' if np.isnan(r.q95_difference) else f'{r.q95_difference*100:.1f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=9)
        ax.imshow(matrix,cmap=ListedColormap(['#c0dce8','#eed2b6','#e6e6e6','#f8f8f8']),vmin=0,vmax=3,aspect='auto')
        ax.set_xticks(range(len(MEASURES)),SHORT,rotation=30,ha='right')
        ax.set_yticks(range(len(subjects)),[f'S{s}' for s in subjects]);ax.set_title('过滤' if mode=='filtered' else '完整历史')
    fig.suptitle('换用同一被试的其他保留参数后，状态解释有多大差异？',fontsize=12)
    fig.supxlabel('蓝：已检查候选中接近；橙：随参数改变；灰：计算差异仍影响判断；浅灰：参数未确立或仅一组。\n数字：参数间逐题最大差异的95%分位（个百分点）。候选间范围没有置信区间含义。',fontsize=8)
    save(fig,output,'parameter_sensitivity.png')


def focus_blocks(frame: pd.DataFrame, output: Path) -> None:
    focus=frame.loc[frame.subject.eq(205)&frame['mode'].eq('full')]
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained',sharex=True)
    fields=['target_present','target_belief','target_execution']
    for ax,field in zip(axes,fields):
        for number,rows in focus.loc[focus.measure.eq(field)].groupby('number',sort=True):
            ax.plot(rows.block,rows.state_mean,color=COLORS[number-1],ls=STYLES[number-1],marker='o',ms=3,lw=1.3,label=f'P{number}')
        ax.set(ylim=(-.02,1.02),ylabel=NAMES[field]);ax.legend(ncol=4,loc='lower right')
    ticks=focus.loc[focus.number.eq(1)&focus.measure.eq(fields[0])]
    axes[-1].set_xticks(ticks.block,[f'{r.first_trial}–{r.last_trial}' for r in ticks.itertuples()],rotation=35,ha='right')
    axes[-1].set_xlabel('有效试次范围（每64题）')
    fig.suptitle('S205 · 哪些阶段的内部解释相近，哪些阶段随参数改变？',fontsize=12)
    fig.supxlabel('每点为该组参数下、该64题内的状态概率或信念均值；使用完整历史。数值使用范围见逐量检查表。',fontsize=8)
    save(fig,output,'S205_blocks.png')


def focus_behavior_predictions(frame: pd.DataFrame, output: Path) -> None:
    own=frame.loc[frame.subject.eq(205)]
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    real=own.loc[own.number.eq(1)].sort_values('trial')
    ax.plot(real.trial,real.observed_correct.rolling(32,min_periods=1).mean(),
            color='#202020',lw=1.6,label='实际正确率')
    for number,rows in own.groupby('number',sort=True):
        rows=rows.sort_values('trial')
        ax.plot(rows.trial,rows.probability_correct.rolling(32,min_periods=1).mean(),
                color=COLORS[number-1],ls=STYLES[number-1],lw=1.1,label=f'P{number} · 本题选对概率')
    ax.set(ylim=(0,1),xlim=(1,real.trial.max()),ylabel='正确率 / 选对概率',xlabel='有效试次',
           title='S205 · 这些内部状态对应怎样的行为概率？')
    ax.legend(ncol=3,loc='lower right',fontsize=8)
    fig.supxlabel('全部曲线均取最近32题均值。模型本题预测只使用此前行为；参数已用完整记录估计，属于拟合内展示。',fontsize=8)
    save(fig,output,'S205_behavior_predictions.png')


def focus_rules(frame: pd.DataFrame, own: list[dict], checks: pd.DataFrame, output: Path) -> dict:
    # Selection changes only the display: correct rules plus seven alternatives
    # with highest mean mass across candidates, modes and both distributions.
    target=sorted(frame.loc[frame.target,'rule'].unique().tolist())
    mass=frame.groupby('rule').probability.mean().sort_values(ascending=False)
    displayed=target+[int(i) for i in mass.index if i not in target][:7]
    all_rules=sorted(int(r) for r in frame.rule.unique()); other=[r for r in all_rules if r not in displayed]
    for item in own:
        cid=item['case'];fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for col,mode in enumerate(MODES):
            for row,field in enumerate(('belief','execution')):
                ax=axes[row,col]
                table=frame.loc[frame.case.eq(cid)&frame['mode'].eq(mode)&frame.measure.eq(field)].pivot(index='rule',columns='block',values='probability')
                values=np.vstack([table.loc[displayed].to_numpy(),table.loc[other].sum(axis=0).to_numpy()])
                np.testing.assert_allclose(values.sum(axis=0),1.,atol=2e-10,rtol=0)
                im=ax.imshow(values,aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='Blues')
                labels=[f'R{r}'+('（正确）' if r in target else '') for r in displayed]+[f'其余{len(other)}条']
                ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(len(table.columns)),table.columns)
                ax.set_xlabel('64题块编号');ax.set_ylabel('规则信念' if field=='belief' else '执行概率')
                flag=checks.loc[checks.case.eq(cid)&checks['mode'].eq(mode)&checks.measure.eq(field),'usable'].iloc[0]
                ax.set_title(('过滤' if mode=='filtered' else '完整历史')+' · '+('数值检查通过' if flag else '数值受限'))
                fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
        fig.suptitle(f'S205 · P{item["number"]} · 正确规则与其他规则怎样分配信念和执行概率',fontsize=12)
        fig.supxlabel('各列在全部规则上的概率和为1。其余规则合并展示，全部规则的原始数值保留在表格中。',fontsize=8)
        save(fig,output,f'S205_P{item["number"]}_rules.png')
    return dict(displayed_rules=displayed,other_rules=other,selection='targets + seven highest mean alternative masses; identical row order across candidates')


def table(headers: list[str], rows: list[list]) -> str:
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def rule_dictionary(case: dict, rule_ids: list[int], source: Path, output: Path) -> dict:
    """Render model rule definitions using behavioural feature-order metadata.

    The existing formatter is a pure geometry renderer. No oral text or oral
    coding columns are loaded, scored, or modified by this report.
    """
    from ..analysis.model_1024_individual_oral_alignment import constraint_text, subject_features
    from ...hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    p=read(source/'protocol.json')['policy']
    original=Path(p['source']);job=read(original/'jobs'/f'primary_{case["id"]}_000.json')
    captured=read(Path(job['individual_target']).with_suffix('.json'))
    space=build_relational_0923_space()
    if json.loads(json.dumps(space.signature))!=captured['catalogue_signature'] or space.version!=captured['catalogue_version']:
        raise ValueError('Rule dictionary catalogue differs from captured states')
    metadata=read(Path(case['template']['folder'])/'prepared.json')
    file=Path(metadata['paths']['feature_order_data'])
    if digest(file)!=metadata['learning_data_sha256']:
        raise ValueError('Behavioral feature-order source changed')
    columns=['iSub']+[f'feature{i}_name' for i in range(1,5)]
    frame=pd.read_csv(file,usecols=columns);features=subject_features(frame.loc[frame.iSub.eq(case['subject'])])
    rows=[]
    for rid in rule_ids:
        for category,region in enumerate(space[rid].categories,1):
            rows.append([f'R{rid}'+('（正确规则）' if rid in case['target_rules'] else ''),category,
                ' 或 '.join('（'+constraint_text(c.A,c.b,features)+'）' for c in region.components)])
    text=['# S205图中规则的具体含义','',
        '下表给出模型目录的类别划分。尺寸使用0—1归一化数值，0.5表示该尺寸范围的中点。'
        '这些定义来自模型与行为任务的特征顺序；口述含义仍待人工复核。','',
        'S205的正确规则R42先按腿的长度分支：腿≤0.5时按头的长度区分类别1和2，腿≥0.5时按脖子的长度区分类别3和4。'
        '边界上的几何关系沿用模型实现。','',table(['规则','类别','该类别的区域约束'],rows),'']
    (output/'S205_rule_dictionary.md').write_text('\n'.join(text))
    return dict(behavior_file=str(file),behavior_sha256=digest(file),loaded_columns=columns,
        feature_order=list(features),catalogue_signature=space.signature,catalogue_version=space.version,
        rules=rule_ids,oral_columns_loaded=False)


def supplemental_report(distributions: Path, selected: list[dict], cases: dict, arrays: dict, output: Path) -> dict:
    """Show the separately registered histogram diagnostics and conditional beta."""
    manifest=read(distributions/'manifest.json')
    if any(digest(distributions/k)!=v for k,v in manifest['artifacts'].items()):
        raise ValueError('Supplementary distribution summary changed')
    checks=pd.read_csv(distributions/'numerical_checks.csv')
    sensitivity=pd.read_csv(distributions/'parameter_sensitivity.csv')
    fields=['belief_histogram','precision_with_absence','executed_rule_precision']
    labels=['正确规则信念分布','正确规则精度分布（含缺席）','执行身份与精度联合分布']
    own=[s for s in selected if s['subject']==205]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained',sharey=True)
    masked={}
    for ax,mode in zip(axes,MODES):
        passed=[]
        for item in own:
            cid=item['case'];i=item['number']-1;a=arrays[cid];targets=cases[cid]['target_rules']
            present=a[mode+'_target_present'];moment=a[mode+'_active_precision'][:,targets].sum(axis=1)
            value=np.divide(moment,present,out=np.full_like(present,np.nan),where=present>=.05)
            masked[mode+'_'+cid]=int(np.isnan(value).sum())
            if np.any(value[np.isfinite(value)]>25+1e-8):
                raise ValueError('Conditional target precision outside model support')
            flag=checks.loc[checks.case.eq(cid)&checks['mode'].eq(mode)&checks.measure.eq('precision_with_absence'),'usable'].iloc[0]
            if flag:passed.append(f'P{i+1}')
            ax.plot(np.arange(1,len(value)+1),value,color=COLORS[i],ls=STYLES[i],lw=1.1,label=f'P{i+1}')
        ax.set(title=MODES[mode],xlabel='有效试次',ylim=(0,25))
        ax.legend(ncol=4,loc='upper left')
        ax.text(.99,.04,'精度分布检查通过：'+('、'.join(passed) if passed else '无'),
                ha='right',transform=ax.transAxes,fontsize=8)
    axes[0].set_ylabel('正确规则在场时的平均判断精度')
    fig.suptitle('S205 · 在正确规则可用的条件下，判断精度怎样变化？',fontsize=12)
    fig.supxlabel('在场概率低于0.05的试次留空；曲线是条件均值。逐组分布检查与参数间差异另列。',fontsize=8)
    save(fig,output,'S205_precision.png')
    text=['# 信念与判断精度的补充分布检查','',
        '本节读取已保存的状态数组，没有新增粒子过滤。采用上一轮固定的20个信念区间、25个精度区间。'
        '正确规则不在场的概率单独保留；规则在场时的精度按0—25划分。联合分布同时保留执行规则身份和其判断精度。','',
        '判断精度是模型中控制分类倾向的状态值。其他条件相同时，该值较高会使规则的分类倾向更集中；'
        '它与有限粒子造成的计算误差分别定义。下面的图展示在正确规则在场这一条件下的均值。','',
        '![S205判断精度](S205_precision.png)','',
        '状态均值接近时，整个分布仍可能不同。下表按完整分布逐项比较重复计算及不同粒子数；'
        '数值工作容忍、权重和完成次数要求沿用主分析。分布通过检查仍保留模拟状态恢复与模型假设的限制。','']
    for mode in MODES:
        rows=[]
        for item in selected:
            if item['parameter_status']!='retained':continue
            row=[f'S{item["subject"]} P{item["number"]}']
            for field in fields:
                r=checks.loc[checks.case.eq(item['case'])&checks['mode'].eq(mode)&checks.measure.eq(field)].iloc[0]
                row.append(('通过' if r.usable else '受限')+f'；{max(r.half_q95,r.comparison_q95)*100:.1f}')
            rows.append(row)
        text += [f'## {MODES[mode]}','',table(['候选']+labels,rows),'',
            '数字为独立重复与粒子数比较中较大的95%分位总变差（以百分点表示）。','']
    text+=['## 参数之间的解释范围','']
    rows=[]
    for sid in sorted(sensitivity.subject.unique()):
        row=[f'S{sid}']
        for field in fields:
            r=sensitivity.loc[sensitivity.subject.eq(sid)&sensitivity['mode'].eq('full')&sensitivity.measure.eq(field)].iloc[0]
            row.append(STATUS[r.status])
        rows.append(row)
    text += [table(['被试：完整历史']+labels,rows),'',
        'S129参数尚未确立，本轮没有为其增加这组分布检查。各候选的完整分布分别保留，没有计算参数平均。','',
        f'[逐分布数值检查]({os.path.relpath(distributions/"numerical_checks.csv",output)})；'
        f'[逐分布参数间差异]({os.path.relpath(distributions/"parameter_sensitivity.csv",output)})。','']
    (output/'precision_distributions.md').write_text('\n'.join(text))
    return dict(manifest_file=str(distributions/'manifest.json'),manifest_sha256=digest(distributions/'manifest.json'),
        precision_display_presence_threshold=.05,masked_trial_counts=masked)


def report_text(source: Path, analysis: Path, output: Path, selected: list[dict], checks: pd.DataFrame,
                sensitivity: pd.DataFrame, whole: pd.DataFrame, blocks: pd.DataFrame, preview: bool) -> tuple[str,str]:
    completion=calculation_record(source)
    text=['# 个体内部状态：概率解释与保留参数间比较','',
        '本报告逐组参数解释行为记录，重点检查S205，并展示其他人的可用范围。状态均为本题选择后、反馈前。',
        '口述仍待人工复核，本轮使用行为资料。所有被试均为开发案例。','']
    if preview:
        text+=['**计算进行中的预览：部分候选仍显示旧结果。本页用于检查报告结构，最终结论等待正式队列结束。**','']
    if completion:
        text+=[f'本轮新增{completion["jobs_complete"]}项成功计算，正式队列{completion["cpu_hours"]:.2f} CPU小时、'
               f'累计执行{completion["wall_hours"]:.2f}小时，峰值{completion["peak_workers"]}个单线程进程。计算完成与各状态量通过检查分别记录。','']
        if 'calendar_hours' in completion:
            text += [f'从首次启动到补齐结束的日历时间为{completion["calendar_hours"]:.2f}小时，其中包含原队列结束后等待继续处理的间隔。'
                '原队列完成287/288项后按调度限制停止；最后一项沿用已登记种子和参数，在单独登记的2.5小时补齐窗口内执行。'
                '原停止记录保留，可选8192粒子批次没有重新开放。','']
    text+=['## 先看哪些结论','',
        '数值检查表示相同设定下的计算是否接近；参数间比较表示换用另一组保留参数后，解释是否改变。'
        '这两项结果共同限定当前使用范围，模拟真值恢复的既有限制继续保留。','']
    rows=[]
    for sid in sorted(sensitivity.subject.unique()):
        own=sensitivity.loc[sensitivity.subject.eq(sid)&sensitivity['mode'].eq('full')]
        rows.append([f'S{sid}']+[STATUS[own.loc[own.measure.eq(f),'status'].iloc[0]] for f in ('target_present','target_belief','target_execution')])
    text+=[table(['被试','正确规则在场','正确规则平均信念','正确规则执行'],rows),'',
        '表中“接近”限定于已检查的有限候选。S229仅有一组，参数间稳健性无法评价；S129的参数集合尚未确立。','',
        '![逐量数值检查](state_use_scope.png)','',
        '这张图每行是一组参数，每列是一项状态量。蓝底表示该量通过完成次数、权重、重复与粒子数检查；'
        '灰底表示至少一项受限。数字只展示两种比较中较大的95%分位差，具体失败原因见逐量数据表。','',
        '![参数间差异](parameter_sensitivity.png)','',
        '这张图比较同一人的不同候选。数值越大，候选之间差异越大；颜色还考虑固定参数下的计算是否稳定。'
        '没有候选概率权重，本轮不计算参数平均后验。','',
        '## 曲线怎样阅读','',
        '黑线是实际选择的最近32题正确率。彩线分别对应各组保留参数；同色同线型在同一被试的全部图中含义相同。'
        '左列过滤使用截至本题的选择，右列完整历史进一步使用后续选择。参数由完整记录估计，所以两列均属于拟合后的回顾性解释。','',
        '在场概率回答“正确规则有多大可能处于当前可用集合”；平均信念回答“学习者分给该规则的权重平均有多少”；'
        '高信念概率回答“正确规则在场且权重至少一半的概率”；执行概率回答“已知这次选择，本题执行正确规则的概率”。'
        '这些量没有被当作被试内部状态的直接观测。','',
        '各面板右下列出该量通过数值检查的候选，未列出的候选保留数值限制。过滤与完整历史的差别还包含条件信息的差别。'
        '在场却执行其他规则、精确进入时点及判断精度的完整心理解释，继续保留既有恢复或数值限制。','']
    for sid in (205,110,306,229,317,129):
        own=[r for r in selected if r['subject']==sid]
        text += [f'## S{sid}','',f'![S{sid}状态](S{sid}_overview.png)','']
        if sid==129:
            text+=['以下四组为既有诊断参数，继续展示其差异；本轮没有重新建立S129的参数集合。','']
        rows=[]
        for r in own:
            scope=checks.loc[checks.case.eq(r['case'])&checks['mode'].eq('full')]
            passed='、'.join(NAMES[x] for x in scope.loc[scope.usable,'measure']) or '暂无量满足全部工作检查'
            rows.append([f'P{r["number"]}',r['candidate'],f'{r["particles"]} × {r["systems"]}',passed])
        text += [table(['图中编号','完整候选编号','粒子数 × 独立计算次数','完整历史的逐量使用范围'],rows),'']
        rows=[]
        for r in own:
            item=[f'P{r["number"]}']
            for field in ('target_present','target_belief','target_execution'):
                v=whole.loc[whole.case.eq(r['case'])&whole['mode'].eq('full')&whole.measure.eq(field)].iloc[0]
                item.append(f'{v.first_block*100:.1f}% → {v.last_block*100:.1f}%')
            rows.append(item)
        text += ['下表比较首64题与末64题的状态均值，用于概括两端；中间过程见完整曲线和每64题数据表。',
                 '',table(['候选','在场：首段 → 末段','平均信念：首段 → 末段','执行：首段 → 末段'],rows),'']
        if sid==205:
            text+=['![S205行为概率](S205_behavior_predictions.png)','',
                '这张图把实际正确率与模型在各题给出的选对概率并列。模型读取此前选择和反馈以及本题刺激，'
                '计算本题的选择分布；本题实际选择随后才用于更新状态。各曲线均取最近32题均值，便于比较行为背景。'
                '参数使用了完整记录，本图属于拟合内条件预测。','']
            text+=['![S205每64题解释](S205_blocks.png)','',
                '每个点汇总连续64题，便于定位参数间差别集中的阶段。所有计算仍使用完整序列；这些块没有用于分段拟合。','']
            for r in own:
                text += [f'![S205 P{r["number"]}规则分布](S205_P{r["number"]}_rules.png)','']
            text+=['规则图各列对应一个64题块，颜色越深表示该规则获得的信念或执行概率越高。R42是本任务的正确规则；'
                '其余行展示主要竞争规则，最后一行合并全部剩余规则。各列总和为1。图题明确该组参数、该种重建的数值检查范围。','']
            text+=['[各规则的具体划分含义](S205_rule_dictionary.md)列出腿、头、脖子和尾巴的区域约束。'
                '正确规则的概率逐渐增加，表示越来越多可能状态支持它；单条内部路径怎样变化仍需路径层面的证据。','']
        if sid==229:
            text+=['本轮直接复用已有高精度结果。正确规则摘要与整个规则目录的数值表现分开解释；'
                '既有自主生成不足及仅一组保留参数的限制继续保留。','']
        if sid==317:
            text+=['执行正确规则概率接近时，其他规则身份、信念和判断精度仍可能有分歧。'
                '本轮按各项检查记录具体范围，既有联合精度分布的限制继续保留。','']
        if sid in (306,317):
            text+=['该被试属于条件3，还涉及类别与反应的配对学习。图中的正确规则指正确的分类结构；'
                '实际作答还受到配对、知觉和判断精度状态的影响。','']
    text+=['## 标准、数据与复现','',
        '数值检查逐量要求独立半批差异、粒子数改变后的差异，平均均不超过2.5个百分点、95%分位均不超过10个百分点；'
        '独立计算有效数量至少4、最大权重不超过0.5，并完成预定次数。参数间接近使用相同数值容忍，前提是所有候选该量均通过数值检查。'
        '两套工作容忍均没有统计置信覆盖含义。','',
        '95%分位描述大多数试次的差异：把差异从小到大排列，约95%的试次不超过该数。剩余少量试次可以更大，'
        '因此精确转折时点继续保留验证限制。权重检查用于发现16次计算的结果是否主要由其中一两次决定。','',
        '全部规则分布使用总变差距离：各规则概率差的绝对值相加后除以2。逐题在场等概率使用直接绝对差。'
        '已有S110、S205独立确认的记录保留，本轮其余候选采用已登记的重复和粒子数工作检查。','',
        f'- [逐状态量数值检查]({os.path.relpath(analysis/"numerical_checks.csv",output)})',
        f'- [每64题状态与行为]({os.path.relpath(analysis/"blocks.csv",output)})',
        f'- [参数间差异]({os.path.relpath(analysis/"parameter_sensitivity.csv",output)})',
        f'- [参数两两比较]({os.path.relpath(analysis/"parameter_pairs.csv",output)})',
        f'- [完整规则块均值]({os.path.relpath(analysis/"S205_rule_blocks.csv",output)})','',
        '原始状态数组及其哈希见分析目录的selected.json和manifest.json。每个候选的全部试次与完整规则目录继续保留。']
    findings=['# 本轮主要发现','',
        '本轮把状态解释分别落实到固定参数下的计算稳定性和保留参数之间的一致性。','',
        table(['被试','正确规则在场','正确规则平均信念','正确规则执行'],
              [[f'S{sid}']+[STATUS[sensitivity.loc[sensitivity.subject.eq(sid)&sensitivity['mode'].eq('full')&sensitivity.measure.eq(f),'status'].iloc[0]]
                for f in ('target_present','target_belief','target_execution')] for sid in (205,110,306,229,317,129)]),'',
        'S205是详细案例，完整曲线、各参数首末阶段对照及竞争规则分布见[完整报告](report.md)。'
        '这里的结论是模型条件下的概率解释；本轮没有增加模拟真值恢复或口述验证的证据。']
    if preview:
        findings.insert(2,'**仅为进行中的报告结构预览，待计算完成后在新目录生成正式报告。**')
    return '\n'.join(text)+'\n','\n'.join(findings)+'\n'


def run(source: Path, analysis: Path, output: Path, distributions: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(output)
    manifest=read(analysis/'manifest.json')
    if any(digest(analysis/k)!=v for k,v in manifest['artifacts'].items()):
        raise ValueError('Analysis artifacts changed')
    output.mkdir(parents=True);started=time.time();cpu=time.process_time();configure()
    selected=read(analysis/'selected.json');protocol=read(source/'protocol.json')
    cases={c['id']:c for c in protocol['cases']};arrays={}
    for item in selected:
        if digest(Path(item['file']))!=item['sha256']:
            raise ValueError('Selected arrays changed')
        arrays[item['case']]=load_arrays(Path(item['file']))
    checks=pd.read_csv(analysis/'numerical_checks.csv');sensitivity=pd.read_csv(analysis/'parameter_sensitivity.csv')
    whole=pd.read_csv(analysis/'whole_record.csv');blocks=pd.read_csv(analysis/'blocks.csv')
    scope_figure(selected,checks,output);sensitivity_figure(sensitivity,output)
    for sid in sorted({r['subject'] for r in selected}):
        overview(sid,[r for r in selected if r['subject']==sid],cases,arrays,checks,output)
    focus_blocks(blocks,output)
    focus_behavior_predictions(pd.read_csv(analysis/'conditional_choice_predictions.csv'),output)
    display=focus_rules(pd.read_csv(analysis/'S205_rule_blocks.csv'),[r for r in selected if r['subject']==205],checks,output)
    focus_case=next(c for c in cases.values() if c['subject']==205)
    dictionary=rule_dictionary(focus_case,display['displayed_rules'],source,output)
    report,findings=report_text(source,analysis,output,selected,checks,sensitivity,whole,blocks,manifest['preview'])
    supplementary=None
    if distributions is not None:
        supplementary=supplemental_report(distributions,selected,cases,arrays,output)
        report+='\n[信念与判断精度的补充分布检查](precision_distributions.md)提供S205精度曲线及各候选的完整分布检查。\n'
    interpretation=analysis/'interpretation.md'
    manual=None
    if interpretation.exists():
        (output/'interpretation.md').write_text(interpretation.read_text())
        manual=dict(file=str(interpretation),sha256=digest(interpretation))
        report=report.replace('## 先看哪些结论','先读[逐人结果的详细解释](interpretation.md)，再按下文核对图表和数值。\n\n## 先看哪些结论',1)
        findings+='\n[详细解释与下一步判断](interpretation.md)。\n'
    (output/'report.md').write_text(report);(output/'findings.md').write_text(findings)
    write(output/'manifest.json',dict(source=str(source),analysis=str(analysis),preview=manifest['preview'],
        analysis_manifest_sha256=digest(analysis/'manifest.json'),report_source_sha256=digest(Path(__file__)),
        figure_contract=read(source/'figure_contract.json'),rule_display=display,rule_dictionary=dictionary,
        supplementary=supplementary,manual_interpretation=manual,
        matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__,
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.time()-started,
        artifacts={f.name:digest(f) for f in output.iterdir() if f.is_file()}))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--distributions',type=Path,help='Optional supplementary interpretation directory')
    args=parser.parse_args();run(args.source,args.analysis,args.output,args.distributions)


if __name__=='__main__':
    main()
