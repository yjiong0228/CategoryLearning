"""Assemble evidence-linked usage tables for the bounded validation stage.

Use --prepared REGISTERED --panel SUMMARY --history SUMMARY --reference RUN
--sensitivity SUMMARY --output NEW. Generated files preserve all failed gates.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import model_1024_state_validation_stage as stage

read,write,digest=stage.read,stage.write,stage.digest
ERRORS={'error_absent':'source_0','error_available_other':'source_1','error_executed':'source_2'}


def load(folder: Path,name: str) -> pd.DataFrame:
    d=read(folder/'completion.json');file=folder/(name+'.csv')
    if d['status']!='complete' or d['files'][file.name]!=digest(file):raise ValueError('Incomplete or changed summary')
    return pd.read_csv(file)


def usage(panel: Path,a: dict) -> list[dict]:
    bins=load(panel,'calibration');coverage=load(panel,'event_coverage');proper=load(panel,'proper_scores')
    numerical=load(panel,'numerical_stability');labels=load(panel,'clear_labels');rows=[]
    for c in coverage.itertuples():
        base=dict(mode=c.mode,scope=c.scope,state=c.state)
        b=bins[(bins['mode']==c.mode)&(bins.scope==c.scope)&(bins.state==c.state)&(bins.trials>0)]
        n=numerical[(numerical['mode']==c.mode)&(numerical.state==ERRORS.get(c.state,c.state))]
        if c.scope.startswith('context_'):n=n[n.context_id==c.scope.removeprefix('context_')]
        elif c.scope!='all':n=n[n.condition==int(c.scope.split('_')[-1])]
        numeric=bool(len(n) and (n['mad']<=a['numerical_probability_mad']).all() and (n.q95<=a['numerical_probability_q95']).all())
        enough=bool(c.rare_event_coverage and c.negative_paths>=a['minimum_negative_paths']
            if c.state in ERRORS else c.state_coverage)
        sufficient=b[b.sufficient];calibrated=bool(len(sufficient) and sufficient.practical_equivalence.all())
        all_bins=bool(len(b) and b.sufficient.all());explicit_failure=bool(((sufficient.lower>.1)|(sufficient.upper<-.1)).any())
        s=proper[(proper['mode']==c.mode)&(proper.scope==c.scope)&(proper.state==c.state)]
        informative=bool(len(s)==2 and (s.reference_difference_upper<0).all())
        if not numeric:status='未通过';reason='独立种子或粒子数差异超过登记容忍范围'
        elif explicit_failure:status='未通过';reason='有充分资料的概率区间出现明确校准偏差'
        elif not enough or not calibrated:status='证据不足';reason='事件覆盖或校准区间精度尚未满足要求'
        elif not all_bins:status='有限可用';reason='已覆盖概率区间通过；其他出现的区间资料不足'
        elif not informative:status='有限可用';reason='校准检查通过；相对本题选择参照的信息增益未确认'
        else:status='可用';reason='通过本轮登记范围内的数值、覆盖、校准及信息增益检查'
        own=labels[(labels['mode']==c.mode)&(labels.scope==c.scope)&(labels.state==c.state)]
        label_gates={}
        for label in ('clear_all','clear_positive','clear_negative'):
            one=own[own.label==label]
            if len(one):
                item=one.iloc[0];label_gates[label]=bool(item.precision_value>=.8 and item.precision_lower>=.7
                    and item.precision_bootstrap_valid_fraction>=.95 and (label!='clear_all' or item.coverage_value>=.2))
        rare_recall=np.nan
        one=own[own.label=='clear_positive']
        if len(one):rare_recall=float(one.iloc[0].recall_value)
        label_pass=bool(status in ('可用','有限可用') and enough and all(label_gates.values()) and len(label_gates)==3
            and (c.state!='error_available_other' or rare_recall>=.5))
        rows.append(dict(**base,status=status,reason=reason,numerical_pass=numeric,event_coverage_pass=enough,
            sufficient_bins=len(sufficient),nonempty_bins=len(b),calibrated_sufficient_bins=int(sufficient.practical_equivalence.sum()),
            validated_probability_bins='; '.join(f'[{row.bin_lower:g},{row.bin_upper:g}'+(']' if row.bin_upper==1 else ')')
                for row in sufficient[sufficient.practical_equivalence].itertuples()),
            all_bins_sufficient=all_bins,information_gain_confirmed=informative,
            max_numerical_mad=float(n['mad'].max()),max_numerical_q95=float(n.q95.max()),
            positive_events=c.positive_events,positive_paths=c.positive_paths,negative_paths=c.negative_paths,
            clear_label_pass=label_pass,positive_recall=rare_recall,**label_gates,
            scope_limit='six registered generating points; known parameters; simulator-only development validation'))
    return rows


def md_table(frame: pd.DataFrame,columns: list[str]) -> str:
    out=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False,name=None):
        out.append('| '+' | '.join(f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(out)


def run(prepared: Path,panel: Path,history: Path,reference: Path,sensitivity: Path,output: Path,figures: Path|None=None) -> None:
    if output.exists():raise FileExistsError('Use a new delivery directory')
    if figures:
        qa=read(figures/'QA.json');review=read(figures/'visual_review.json')
        if review['status']!='passed':raise ValueError('Require final visual review')
        for name,sha in qa['files'].items():
            if digest(figures/name)!=sha:raise ValueError('Reviewed PNG changed')
    r=stage.registered(prepared);a=r['policy']['acceptance'];output.mkdir(parents=True)
    states=pd.DataFrame(usage(panel,a));states.to_csv(output/'state_usage.csv',index=False)
    finite=load(reference,'finite_metrics');shared=load(reference,'shared_conditional');functional=load(panel,'functionals')
    numeric=load(panel,'numerical_audit');long_numeric=load(history,'numerical_audit');points=load(sensitivity,'point_sensitivity')
    diagnostics=read(history/'bank_diagnostics.json')['cases']
    comparisons=[]
    for case in sorted(functional.case_id.unique()):
        f=functional[functional.case_id==case]
        if 1024 not in f.particles.values:continue
        for mode in ('prefix','full'):
            for name,resolution in [('presence_trials',.05),('first_entry_trial',32.)]:
                low=f[(f.particles==512)&(f['mode']==mode)&(f.function==name)].iloc[0]
                high=f[(f.particles==1024)&(f['mode']==mode)&(f.function==name)].iloc[0]
                difference=abs(low['mean']-high['mean'])/high.endpoint if name=='presence_trials' else abs(low.q50-high.q50)
                comparisons.append(dict(case_id=case,mode=mode,function=name,difference=float(difference),resolution=resolution,
                    numerical_pass=bool(difference<=resolution),interval_coverage_1024=high.truth_in_90_interval))
    pd.DataFrame(comparisons).to_csv(output/'functional_resolution.csv',index=False)
    resolution_table=pd.DataFrame(comparisons);functional_usage=[]
    for mode in ('prefix','full'):
        for name in sorted(functional.function.unique()):
            own=functional[(functional.particles==512)&(functional['mode']==mode)&(functional.function==name)]
            eligible=own[own.truth_in_90_interval.notna()]
            hit=eligible.truth_in_90_interval.map({True:1,False:0,'True':1,'False':0})
            check=resolution_table[(resolution_table['mode']==mode)&(resolution_table.function==name)]
            if name=='presence_trials':
                status='有限可用' if len(check) and check.numerical_pass.any() else '未通过'
                reason='整段在场分布可作开发描述；保留未通过分辨率的案例，逐例复核'
            elif name=='first_entry_trial':
                status='未通过' if len(check) and not check.numerical_pass.all() else '有限可用'
                reason='首次进入时点按32题分辨率检查；未进入质量及截断代码共同保留'
            elif name=='target_execution_trials':
                status='证据不足';reason='解析求和案例保留均值与总方差；其执行题数分位数未完成'
            else:
                status='证据不足';reason='已有条件区间恢复诊断；该功能量的独立数值验收尚待补足'
            functional_usage.append(dict(mode=mode,function=name,status=status,reason=reason,paths=len(own),
                paths_with_interval=len(eligible),covered_90_interval=int(hit.sum()),
                empirical_coverage=float(hit.mean()) if len(hit) else np.nan,median_interval_width=float(eligible.interval_width.median()),
                resolution_cases=len(check),resolution_passed=int(check.numerical_pass.sum()),
                scope='known generating parameters; 48 unselected paths; resolution subset fixed before simulation'))
    functional_usage=pd.DataFrame(functional_usage);functional_usage.to_csv(output/'functional_usage.csv',index=False)
    filtered=states[states.scope.str.startswith('condition_')&(states['mode']=='filtered')]
    checkpoint=states.groupby(['mode','status']).size().unstack(fill_value=0)
    checkpoint=checkpoint.reindex(columns=['可用','有限可用','未通过','证据不足'],fill_value=0).reset_index()
    state_names={'present':'正确规则在场','high_belief':'正确规则高信念','target_execution':'执行正确规则',
        'error_absent':'错误：正确规则缺席','error_available_other':'错误：在场但执行其他规则','error_executed':'错误：执行正确规则后答错'}
    display=filtered.copy();display['scope']=display.scope.str.replace('condition_','条件',regex=False)
    display['state']=display.state.map(state_names)
    display=display.rename(columns={'scope':'范围','state':'目标','status':'概率结论','max_numerical_mad':'最大数值差',
        'sufficient_bins':'覆盖区间数','calibrated_sufficient_bins':'校准通过区间数','clear_label_pass':'明确标签通过'})
    display=display.rename(columns={'event_coverage_pass':'事件覆盖通过'})
    checkpoint['mode']=checkpoint['mode'].map({'filtered':'在线过滤','lag':'32题滞后','prefix':'前缀终点','full':'完整终点'})
    checkpoint=checkpoint.rename(columns={'mode':'观察范围'})
    historical=pd.DataFrame(diagnostics).rename(columns={'case_id':'案例','systems':'独立系统数','outer_ess':'系统有效样本量',
        'maximum_weight':'最大系统权重','half_bank_window_mad':'半批概率平均差','half_bank_window_q95':'半批差95分位',
        'half_bank_whole_q95':'全段半批差95分位','numerical_gate_pass':'数值检查通过'})
    passed_histories='、'.join(d['case_id'] for d in diagnostics if d['numerical_gate_pass']) or '无'
    failed_histories='、'.join(d['case_id'] for d in diagnostics if not d['numerical_gate_pass']) or '无'
    functional_display=functional_usage[functional_usage.function.isin(['presence_trials','first_entry_trial'])].copy()
    functional_display['mode']=functional_display['mode'].map({'prefix':'前缀终点','full':'完整终点'})
    functional_display['function']=functional_display.function.map({'presence_trials':'整段在场题数','first_entry_trial':'首次进入时点'})
    functional_display=functional_display.rename(columns={'mode':'观察范围','function':'功能量','status':'结论',
        'covered_90_interval':'90%区间覆盖路径数','paths':'路径数','resolution_passed':'分辨率通过数','resolution_cases':'复核路径数'})
    limited=states[states.status=='有限可用'].copy()
    limited['scope']=limited.scope.str.replace('condition_','条件',regex=False).str.replace('context_','背景',regex=False)
    limited['mode']=limited['mode'].map({'filtered':'在线过滤','lag':'32题滞后','prefix':'前缀终点','full':'完整终点'})
    limited['state']=limited.state.map(state_names)
    limited=limited.rename(columns={'mode':'观察范围','scope':'资料范围','state':'目标','validated_probability_bins':'已支持概率区间'})
    event_coverage=load(panel,'event_coverage');run_protocol=read(Path(read(panel/'completion.json')['run'])/'protocol.json');contexts=[]
    for context in r['contexts']:
        cid=context['context_id'];hp=context['candidate']['hyperparams']
        case=next(c for c in run_protocol['cases'] if c['context_id']==cid)
        cov=event_coverage[(event_coverage['mode']=='full')&(event_coverage.scope=='context_'+cid)&(event_coverage.state=='present')].iloc[0]
        contexts.append(dict(背景=cid,条件=case['condition'],信念保持率=hp['engine.modules.hypo_transitions_mod.kwargs.intervention.decay'],
            容量=hp['engine.modules.hypo_transitions_mod.kwargs.max_capacity'],执行维持倾向=hp[stage.worker_tools.study.KAPPA],
            出现正确规则的路径=f'{int(cov.positive_paths)}/8'))
    proper=load(panel,'proper_scores');presence=proper[(proper.scope=='all')&(proper.state=='present')&(proper.score=='brier')]
    filter_brier=float(presence[presence['mode']=='filtered'].iloc[0].value)
    full_brier=float(presence[presence['mode']=='full'].iloc[0].value)
    reference_brier=filter_brier-float(presence[presence['mode']=='filtered'].iloc[0].reference_difference_value)
    def link(folder,name):return os.path.relpath(folder/name,output)
    figure_text=''
    if figures:
        figure_text=f'''## 图形与误差范围

![状态概率校准]({link(figures,'calibration.png')})

校准图保留48条路径，展示在线过滤、32题滞后和完整平滑。点为各固定概率区间的平均
预测及实际频率，误差条为按完整路径分层自举2000次得到的逐点95%区间。填实点满足
至少8条路径和100题的覆盖，空心点保留资料不足区间。各条件及前缀平滑保留在源表。

![完整历史与整段在场量]({link(figures,'history.png')})

完整历史图以灰色标记模拟真值，比较系统等权与证据加权。右列显示整段在场题数的条件
均值和90%连贯路径区间，竖虚线为真值。系统权重和半批差异分别检查数值稳定性。
两例均为已知参数的开发诊断。48条未经筛选路径评价的是两个过滤结果的等权汇总，
证据加权新方法的代表性恢复仍待独立检验。

'''
    total_filters=len(numeric)+len(long_numeric);cpu=(numeric.cpu_seconds.sum()+long_numeric.cpu_seconds.sum())/3600
    historical_run=Path(read(history/'completion.json')['run']);history_protocol=read(historical_run/'protocol.json')
    resumed=history_protocol.get('resumption',{});retry_count=resumed.get('incomplete_retried',0);lost_cpu_upper=0.
    if retry_count:
        source_time=(Path(resumed['partial'])/'protocol.json').stat().st_mtime
        resume_time=(Path(resumed['resume'])/'protocol.json').stat().st_mtime
        lost_cpu_upper=retry_count*(resume_time-source_time)/3600
    scope=dict(shared_filters=total_filters,shared_filter_cpu_hours=float(cpu),maximum_shared_filters=r['policy']['stage_budget']['max_new_filters'],
        cpu_budget_hours=r['policy']['stage_budget']['max_cpu_hours'],stage_complete=True,oral_validation='deferred',
        full_parameter_uncertainty='not_integrated',known_parameter_paths=48,
        exact_benchmark_pass=bool((finite.mae<=a['exact_benchmark_probability_mae']).all()),
        shared_conditional_max_error=float(shared.kernel_max_error.max()),long_history=diagnostics,
        state_status_counts={str(k):int(v) for k,v in states.status.value_counts().items()},
        confident_state_labels_passed=int(states.clear_label_pass.sum()))
    scope.update(incomplete_attempts=retry_count,completed_retries=retry_count,
        interrupted_cpu_hours_upper_bound=float(lost_cpu_upper),scientific_validation='partial; target-specific usage tables govern use')
    write(output/'stage_decision.json',scope)
    report=f'''# Model 1024：状态推断方法验证的阶段交付

本阶段已执行登记的检验，并交付使用范围。状态结论按数值稳定性、概率校准、事件覆盖和信息增益分别验收。
口述继续待最终人工检查。本轮固定生成参数；参数点敏感性与完整参数不确定性积分分别记录。

本轮完整通过的概率描述范围为{int((states.status=='可用').sum())}项，另有{int((states.status=='有限可用').sum())}项限定用法获得有限支持。
能够同时满足各项验收条件的逐题标签使用范围为{int(states.clear_label_pass.sum())}项。
行为历史提供状态信息；逐题标签的数值与校准验收继续保留其失败和证据边界。
整段在场量比具体进入时点获得更多支持，两类功能量分别列出。

## 交付范围与证据

48条新增独立轨迹来自三个反馈条件、六个登记参数背景，每背景8条，全部进入评价。
每条轨迹进行两次512粒子过滤；每背景预定第0条轨迹另做两次1024粒子过滤，共108次。
逐题状态时点为选择后、反馈前。过滤、32题滞后、前缀终点及完整终点分别评价。
条件1为二分类二值反馈，条件2为四分类二值反馈，条件3为四分类分层反馈。
恢复主汇总为两个独立过滤的等权均值；整条历史的证据加权另在两个预定长案例检验。
真实状态仅用于评分。新增轨迹按固定种子生成，未依据状态或学习表现筛选。

{md_table(pd.DataFrame(contexts),['背景','条件','信念保持率','容量','执行维持倾向','出现正确规则的路径'])}

其余参数按登记点固定。上述背景来自开发资料；恢复轨迹使用新随机种子。
条件3仅2/16条路径出现正确规则，其状态覆盖保留证据不足结论。

## 逐项使用范围

可用仅指本轮登记参数、任务和观测时点通过相应检查；数值比较覆盖512与1024粒子的规定子集。
有限可用保留部分概率范围或低信息结论。未通过表示明确的验收失败；证据不足保留尚未辨明的项目。
明确标签还需达到预定精确率、置信下界、覆盖及关键错误检出率，概率评分与标签可用性分别记录。
标签正确率为所报告标签中与真值一致的比例，正负标签另查各自的精确率。
完整规则信念向量、连续精度及全部工作区内容继续单独验收。

以下是分条件在线过滤的完整验收条目；其他观测终点及总体汇总见[全部使用范围表](state_usage.csv)。
“最大数值差”先对每条路径计算概率的平均绝对差，再取各路径的最大值。验收同时检查差值
95分位、概率校准和资料覆盖；各项判据的结果保留在源表。

{md_table(display,['范围','目标','概率结论','最大数值差','事件覆盖通过','覆盖区间数','校准通过区间数','明确标签通过'])}

{md_table(checkpoint,list(checkpoint.columns))}

每个观察范围包含60个验收条目：总体6项、三个条件各6项、六个参数背景各6项。
这些汇总共享模拟路径。总体及分条件检验对登记背景的固定组合评价；具体背景的结果另列。
概率区间的95%自举区间为逐点区间；分层自举保留每条完整路径的试次依赖。

有限可用项目的概率范围如下；区间之外继续保留原验收状态。

{md_table(limited,['观察范围','资料范围','目标','已支持概率区间'])}

Brier分数为预测概率与0/1真值差的平方均值，较小表示恢复更准确。
全序列试次加权的在场Brier，在线过滤为{filter_brier:.4f}，利用未来信息的完整回顾重建为
{full_brier:.4f}，本题选择参照为{reference_brier:.4f}。行为历史提供状态信息；逐题数值稳定性、
概率校准及明确标签继续按各自标准验收。

## 精确参照与历史更新

有限状态基准包含8题、128题，64与256粒子，4个独立批次，每批512个新系统。
8题枚举6561条路径；共享模型的条件执行参照给定整段真实工作区、规则信念、精度和知觉路径。
后者的转移概率与生成记录最大差为{scope['shared_conditional_max_error']:.3g}。
有限状态基准所有方法/批次的最大平均绝对误差为{finite.mae.max():.4f}，登记门槛为0.03。

独立完整粒子系统按各自整段行为证据加权。系统内保留连贯祖先路径和终点权重，过滤与固定
滞后分别使用各行的观察终点。证据权重的扩展空间依据及有限系统近似性质见
[阶段协议]({link(stage.ROOT/'src/Bayesian_state/docs/model_architecture','model_1024_state_validation_stage.md')})。
精确参照校准运算，长模型的适用范围由其自身恢复和数值检查决定。

系统有效样本量为归一化系统权重平方和的倒数，衡量外层权重集中程度。
半批差异比较两批独立系统在原进入窗口的概率，完整试次范围的差异另行保存。

{md_table(historical,['案例','独立系统数','系统有效样本量','最大系统权重','半批概率平均差','半批差95分位','全段半批差95分位','数值检查通过'])}

本轮完整历史数值检查通过案例为{passed_histories}，未通过案例为{failed_histories}。
该检查的范围为固定参数的既有开发路径；新方法在完整共享模型中的代表性状态恢复继续单独验收。

## 整段功能量、参数与经验边界

在场题数、首次进入、最长在场片段及进入次数均由同一固定终点的连贯路径计算。
整段在场题数先在每条路径内加总，再按该终点的后验路径权重汇总。
逐题过滤的各行依次条件于不同观察终点，其累计值属于另一项在线统计。
首次进入保留未进入概率及终点后截断代码。90%路径区间描述当前条件后验近似，覆盖率与宽度
一起保留。零维持参数下，每条状态路径的期望执行次数可以为小数。总方差同时包括路径间
差异和给定路径后执行身份的抽样随机性。该项当前提供均值与方差，离散执行题数的分位数待算。

粒子数比较中的功能量分辨率见[分辨率检查](functional_resolution.csv)，全部路径的真实值、
均值、区间及覆盖见[功能量源表]({link(panel,'functionals.csv')})。
下表的分母为48条路径及预定6条粒子数复核路径；各功能量的适用结论见[功能量使用范围](functional_usage.csv)。

{md_table(functional_display,['观察范围','功能量','结论','90%区间覆盖路径数','路径数','分辨率通过数','复核路径数'])}

长案例使用了既有开发路径，两个案例的区间结果支持定向诊断，整体校准需要独立路径资料。

既有12条固定路径均有生成点与采用点状态比较；似然参照点有9条保存了状态，3条缺少对应
采集，已逐项标记。它们是两个参数条件恢复中的固定点比较，完整个体参数后验继续待验收。
固定点敏感性沿用此前四次512粒子抽样执行过滤。点间差异同时包含有限粒子误差；
本轮解析求和的参数积分继续单独验收。独立种子和粒子数稳定性由数值检查表提供。
正维持参数的抽样过滤沿用原有数值概率下限；本阶段两条证据加权长案例均为零维持参数，
逐粒子选择概率下限检查由采集记录验证。真人口述外部效度及人群推广继续待独立检验。

{figure_text}## 计算与复现

正式共享模型过滤完成{total_filters}次，已完成过滤记录CPU合计{cpu:.2f}小时；上限为236次、96 CPU小时。
本批曾有{retry_count}项尝试受到终止信号，已用原参数及原种子补齐，完整结果继续保留。
中断尝试未保存最终CPU计时；按单线程进程的启动至恢复登记墙时，其额外CPU上界为{lost_cpu_upper:.2f}小时。
有限状态系统另按登记数量完成。阶段结束保留通过、失败与证据不足项目，计算完成和科学通过分别记录。

- [冻结协议]({link(prepared,'protocol.json')})、[阶段决定](stage_decision.json)。
- [概率校准及分母]({link(panel,'calibration.csv')})、[相对本题选择参照的评分]({link(panel,'proper_scores.csv')})。
- [明确状态标签]({link(panel,'clear_labels.csv')})、[数值稳定性]({link(panel,'numerical_stability.csv')})。
- [完整历史概率]({link(history,'curves.csv')})、[完整历史功能量]({link(history,'functionals.csv')})。
- [完整历史补算与封存记录]({link(historical_run,'protocol.json')})。
- [精确基准]({link(reference,'finite_metrics.csv')})、[共享条件执行]({link(reference,'shared_conditional.csv')})。
- [固定参数点敏感性]({link(sensitivity,'point_sensitivity.csv')})。

入口为 `model_1024_state_validation_stage`、`model_1024_stage_reference` 和 `model_1024_stage_summary`，
位于共享模型的 `workflows.analysis`；本报告入口位于 `workflows.reports.model_1024_stage_delivery`。
所有命令要求新输出目录。冻结参数、随机种子、源文件校验值及各次工作记录保留在输入协议中。
'''
    (output/'report.md').write_text(report)
    write(output/'completion.json',dict(status='complete',stage_complete=True,
        inputs={str(folder/'completion.json'):digest(folder/'completion.json') for folder in (panel,history,reference,sensitivity)},
        figures=None if figures is None else dict(folder=str(figures.resolve()),qa_sha256=digest(figures/'QA.json'),review_sha256=digest(figures/'visual_review.json')),
        source_sha256=digest(Path(__file__)),files={f.name:digest(f) for f in output.iterdir() if f.is_file()},oral_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared','panel','history','reference','sensitivity','output'):parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--figures',type=Path)
    run(**vars(parser.parse_args()))
