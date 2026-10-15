"""Build a local review report from completed, frozen objective comparisons.

The report links source tables and all participant figures. --findings accepts
a JSON list of interpretation paragraphs written after inspecting the results;
the builder computes all quantitative tables directly from saved analyses.
Every output goes to a new directory.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import shutil
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd

from ...evaluation.model_0923_h_validation import load,digest


def build(args: argparse.Namespace) -> None:
    output=args.output
    if output.exists():raise FileExistsError('Use a new report destination')
    meta=load(args.summary/'summary_manifest.json');frozen=load(Path(meta['fit'])/'nominees.json')
    for folder,name in [(args.features,'completion.json'),(args.prefix_scores,'comparison_manifest.json'),
        (args.uncertainty,'completion.json'),(args.stopping,'completion.json'),(args.primary_figures,'completion.json')]:
        record=load(folder/name)
        assert record['status']=='complete' and record['nomination_sha256']==meta['nomination_sha256']
    for folder in (args.recovery,args.truth_scores,args.recovery_figures,args.task_figures):
        assert load(folder/'completion.json')['status']=='complete'
    recovery_meta=load(args.recovery/'completion.json');truth_meta=load(args.truth_scores/'protocol.json')
    assert recovery_meta['nomination_sha256']==truth_meta['nomination_sha256']
    assert recovery_meta['source_truths_sha256']==truth_meta['truths_sha256']
    assert load(args.recovery_figures/'completion.json')['source_parameters_sha256']==digest(args.recovery/'parameters.csv')
    assert load(args.task_figures/'completion.json')['source_paths_sha256']==digest(args.task/'paths.csv')
    output.mkdir(parents=True)
    inputs={};body=[]
    def read(folder,name):
        file=folder/(name+'.csv');inputs[str(file.resolve())]=digest(file);return pd.read_csv(file)
    def link(path,label):return '<a href="'+html.escape(os.path.relpath(path.resolve(),output.resolve()),quote=True)+'">'+html.escape(label)+'</a>'
    def paragraph(text):body.append('<p>'+html.escape(text)+'</p>')
    def table(frame,folder=None,name=None):
        body.append('<div class="table">'+frame.to_html(index=False,border=0,float_format=lambda n:f'{n:.5g}',na_rep='—',escape=True)+'</div>')
        if folder is not None:body.append('<p>'+link(folder/(name+'.csv'),'下载完整 CSV')+'</p>')
    def picture(folder,name,label):
        path=folder/(name+'.png');inputs[str(path.resolve())]=digest(path)
        body.append('<figure><a href="'+html.escape(os.path.relpath(path.resolve(),output.resolve()),quote=True)+'"><img src="'+
            html.escape(os.path.relpath(path.resolve(),output.resolve()),quote=True)+'" alt="'+html.escape(label,quote=True)+'"></a><figcaption>'+html.escape(label)+'</figcaption></figure>')
    body.append('<h1>参数拟合目标对照：NLL 与自主行为分布</h1>')
    paragraph('本轮固定 ECM_S 的 H 结构，在六名预留被试上，以相同模型状态更新预算比较两个参数拟合目标。每人只用前约70%的行为选择参数，后续行为用于共同评价。NLL 使用每题作答前的预测概率；GEN 使用全部自主路径的行为特征分布。')
    paragraphs=load(args.findings)
    if not isinstance(paragraphs,list) or any(not isinstance(p,str) for p in paragraphs):raise ValueError('Findings must be interpretation paragraphs')
    for text in paragraphs:paragraph(text)
    scores=read(args.summary,'scores');held=scores.query("phase == 'test'")
    choices=read(args.summary,'aggregate_choice_scores');feature=read(args.features,'feature_aggregate')
    generation=read(args.uncertainty,'aggregate');prefix=read(args.prefix_scores,'aggregate_model_contrasts')
    stopping=read(args.stopping,'aggregate');rec=read(args.recovery,'aggregate')
    measurements=[]
    def delta(frame,value,error,metric):
        row=frame[frame.model=='GEN'].iloc[0]
        measurements.append(dict(指标=metric,GEN减NLL=row[value],Monte_Carlo_SE=row[error]))
    delta(choices,'delta_participant_mean_nll','delta_participant_mc_se','留出选择 NLL：被试等权')
    delta(feature,'delta_from_nll','delta_mc_se','留出自主特征 energy U：被试等权')
    delta(generation,'delta_energy','delta_energy_mc_se','留出自主曲线 energy V：被试等权')
    delta(prefix.query("start == 'filtered_prefix'"),'delta_energy','delta_energy_mc_se','训练状态续跑曲线 energy V：被试等权')
    delta(stopping,'delta_crps_blocks','delta_crps_mc_se','自适应任务首次达标 CRPS：被试等权，块')
    body.append('<h2>共同留出评价</h2>');table(pd.DataFrame(measurements))
    paragraph('上表的差值小于0表示 GEN 参数得分更低。Monte Carlo SE 衡量固定参数下有限滤波或模拟产生的数值误差；它不包含参数搜索不确定性或被试总体差异。各指标的量纲不同，分别阅读。')
    picture(args.primary_figures,'common_scores','六名被试的全部共同评分。误差线为 Monte Carlo 标准误。')
    table(choices,args.summary,'aggregate_choice_scores')
    paragraph('选择 NLL 同时保留被试等权均值和逐题总分。留出条件预测每题先预测，再吸收该题真实选择；自主曲线从任务起点运行，续跑曲线仅在训练前缀结束时吸收历史。两类自主预测均用自身选择与反馈推进未来状态。')
    paragraph('从任务起点生成的行为同时反映参数、初始内部状态和随机搜索。训练状态续跑先据真实历史更新内部状态分布，再检验后续学习动态。两种起点对应不同的预测问题，因此分别报告。')
    body.append('<h2>全部生成路径</h2>')
    paragraph('每人上排展示从任务起点生成的全部128条路径，下排展示训练前缀滤波后生成的128条未来路径（8次独立滤波，每次16个分支）。细线为每32题的实际模拟正确率，阴影为逐块10%至90%分位数，粗线为模拟均值。紫色虚线为条件预测，黑线为真实行为。区间衡量预测分散程度，宽度需要结合覆盖率与分布评分解释。')
    paragraph('背景路径来自独立自主模拟，按等权汇总；紫色虚线另用真实历史运行粒子滤波估计。两者共享参数，采用不同的信息条件。')
    for subject in meta['subjects']:
        body.append(f'<details><summary>被试 {subject}</summary>')
        picture(args.primary_figures,f'subject_{subject}',f'被试 {subject}：NLL 与 GEN 的完整路径分布')
        table(held[held.subject==subject][['model','trials','choice_nll','energy','crps','interval_coverage','interval_width']])
        body.append('</details>')
    body.append('<h2>相同候选上的训练目标与参数</h2>')
    paragraph('双方各自搜索后，合并优胜候选，再用两种目标评价同一个候选集合。下表交叉评价两组最终所选参数，便于区分目标取舍与候选覆盖差异。确认阶段使用独立种子；这些分数仍属于训练资料。')
    table(read(args.features,'training_confirmations'),args.features,'training_confirmations')
    table(read(args.features,'training_contrasts'),args.features,'training_contrasts')
    body.append('<details><summary>确认阶段的全部候选及配对数值误差</summary>')
    paragraph('每个候选与本目标所选点使用相同确认种子比较。different_coordinates 给出两点不同的参数个数。这张表描述有限候选集中的评分差异；配对数值标准误只反映滤波或模拟误差。')
    table(read(args.features,'confirmation_candidates').sort_values(['subject','model','objective_score']),args.features,'confirmation_candidates');body.append('</details>')
    parameters=read(args.features,'parameters');parameters['coordinate']=parameters.parameter.str.rsplit('.',n=1).str[-1]
    parameters.loc[parameters.coordinate=='decay','coordinate']='evidence_decay'
    body.append('<details><summary>全部个体参数</summary>')
    table(parameters[['subject','model','coordinate','value']],args.features,'parameters');body.append('</details>')
    inactive=read(args.summary,'inactive_parameter_channels')
    paragraph('当一个参数点关闭某条计算通道时，相关坐标不能在该点被直接赋予独立心理意义。通道关闭核对结果如下；空表表示未命中当前已定义的精确关闭条件。')
    table(inactive,args.summary,'inactive_parameter_channels')
    body.append('<h2>完整任务与首次达标</h2>')
    taskmeta=load(args.task/'protocol.json')
    paragraph('固定刺激重放使用被试实际看到的外部刺激顺序。真实实验会按表现调整后续刺激配额，因此另行模拟响应驱动的完整任务，并计算首次达标时间。20块之后仍未达标的路径统一保留在第21类。达标时间CRPS与覆盖率的主表使用自适应配额环境；平衡配额和会话重置随机数的版本用于任务环境敏感性检验。')
    paragraph('任务核对纳入被试：'+', '.join(map(str,taskmeta['included_subjects']))+'。任务范围排除：'+json.dumps(taskmeta['excluded_subjects'],ensure_ascii=False)+'。')
    picture(args.task_figures,'stopping_cdf','完整累计达标时间分布，包含截止时仍未达标的全部概率质量。')
    table(stopping,args.stopping,'aggregate');table(read(args.stopping,'observed'),args.stopping,'observed')
    paragraph('六人的记录都在首次达标时结束。末段高正确率包含停止规则的筛选效应，因此需结合完整首次达标分布解释曲线末段的预测偏差。当前对照固定了H结构，具体环节的贡献仍需定向机制干预区分。')
    body.append('<details><summary>全部四种任务环境的模拟结果</summary>')
    paragraph('固定刺激重放在各被试记录长度处结束，其余三种任务环境采用共同的20块截止。CDF图展示截止长度相同的三种环境；下表保留四种环境的全部模拟结果。不同截止长度下的达标比例分别解释。')
    table(read(args.task,'summary'),args.task,'summary');body.append('</details>')
    body.append('<h2>全部14个坐标的联合恢复</h2>')
    paragraph('恢复数据由此前六个冻结参数点各生成两条独立序列，共12条。两种目标均重新估计全部14个坐标。下图各点为一份合成序列的估计结果；重合点按原值绘制。误差按该坐标完整搜索范围标准化，14个坐标的均值只用于描述，不构成单一的心理特质可靠性指标。')
    picture(args.recovery_figures,'parameter_recovery','联合参数恢复试验：横轴为生成真值，纵轴为拟合值。')
    paragraph('distinct_truth_values 列记录每个坐标实际覆盖的真值个数；真值范围较窄的坐标，其跨被试差异恢复能力仍需扩大设计后检验。')
    paragraph('spearman_mean_estimate 比较六个生成点的真值排序与各点两次估计均值的排序；估计恒定时该量未定义，以“—”表示。normalized_replicate_gap 衡量同一真值下两份独立序列的估计差，按参数搜索范围标准化，包含行为随机性与搜索随机性的共同影响。这些量描述本次小规模恢复，不作总体可靠性推断。')
    display=rec.assign(coordinate=rec.parameter.str.rsplit('.',n=1).str[-1])
    display.loc[display.coordinate=='decay','coordinate']='evidence_decay'
    table(display[['coordinate','model','distinct_truth_values','normalized_mae','grid_mae',
        'spearman_mean_estimate','normalized_replicate_gap','exact_fraction','boundary_fraction']],args.recovery,'aggregate')
    paragraph('以下检查在恢复参数冻结后才读取生成真值，并用对应确认种子重新评分。estimate_minus_truth 为估计点分数减真值分数；明显为正提示有限搜索损失。负值也可能来自单条有限序列的随机差异，需要结合跨序列重复和坐标误差解释。')
    table(read(args.truth_scores,'truth_scores'),args.truth_scores,'truth_scores')
    body.append('<h2>计算预算、数据边界与复现</h2>')
    cost=read(args.features,'fit_costs')
    aggregate=cost.groupby(['stage','model'],sort=False).agg(jobs=('candidate','size'),state_updates=('state_updates','sum'),cpu_seconds=('cpu_seconds','sum')).reset_index()
    aggregate['cpu_hours']=aggregate.pop('cpu_seconds')/3600
    table(aggregate,args.features,'fit_costs')
    paragraph('每阶段、每被试的两组状态更新次数已经逐项核对。CPU耗时另行记录；一个工作进程使用一个数值线程，计算队列最多128个工作进程。恢复拟合使用同样的完整搜索流程。')
    paragraph('GEN 的特征为每32题正确率和真实类别条件下的作答频率，两组各占距离平方尺度的一半。该分数评价这些特征的联合分布，其识别能力取决于特征保留的信息。两组拟合均使用训练刺激重放，口述数据未参与拟合或目标定义。')
    paragraph('本轮按完整记录长度的约70%划分训练前缀，属于回顾性留出检验。训练工作进程只读取前缀选择；切分长度本身由完整记录长度确定，达标时间没有单独进入训练目标。固定训练长度的前瞻评价需要另行冻结边界。')
    paragraph('本次结果支持对拟合目标的比较。个体参数的心理解释还需要更广范围真值、更多独立合成序列和参数不确定性分析。当前 H 结构固定，研究结果不构成它相对于所有其他机制的唯一性证明。')
    body.append('<p>'+link(args.features/'completion.json','评分与预算核对')+' · '+link(args.recovery/'completion.json','恢复范围与限制')+' · '+link(Path(meta['fit'])/'nominees.json','冻结参数')+' · '+link(args.primary_figures/'figure_contract.json','图表定义与导出约定')+'</p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NLL 与自主分布拟合对照</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#25313a}h1,h2{line-height:1.35}h2{margin-top:2.2em}a{color:#006e9d}.table{overflow:auto;margin:16px 0}table{border-collapse:collapse;font-size:13px}th,td{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{background:#f3f6f8}img{max-width:100%;height:auto}figure{margin:22px 0}figcaption{font-size:14px;color:#58636b}summary{cursor:pointer;font-weight:600;padding:10px}details{border:1px solid #ddd;border-radius:5px;margin:12px 0;padding:5px 12px}</style><body>'+''.join(body)+'</body></html>'
    (output/'index.html').write_text(page)
    shutil.copy2(__file__,output/'report_source.py');shutil.copy2(args.findings,output/'findings.json')
    (output/'runtime_versions.json').write_text(json.dumps(dict(python=sys.version,
        packages={name:version(name) for name in ('numpy','scipy','pandas','PyYAML','numba','threadpoolctl','matplotlib','pytest')},
        observed_at='report_generation',numeric_threads_per_worker=1,
        process_budget=frozen['settings']['parallel_budget']),indent=2))
    (output/'completion.json').write_text(json.dumps(dict(status='complete',nomination_sha256=meta['nomination_sha256'],
        source_sha256=digest(Path(__file__)),inputs_sha256=inputs,subjects=meta['subjects']),indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('summary','features','prefix-scores','uncertainty','task','stopping','recovery','truth-scores',
                 'primary-figures','recovery-figures','task-figures','findings','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    build(parser.parse_args())


if __name__=='__main__':main()
