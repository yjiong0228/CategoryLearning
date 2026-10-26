"""Render the registered individual-parameter pilot without rerunning the model.

All candidate rows and all generated paths remain in exported data. Figures are
development diagnostics; intervals describe Monte Carlo variation. Run with
--study EXISTING_STUDY --output NEW_REPORT_DIRECTORY after numerical completion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.model_1024_individual_regions import (
    KEYS, completed_arrays, load_observed, read, write,
)
from ...metrics.trajectory_shape import wilson_interval

LABELS = ['no_clear_increase','stable_high','abrupt_maintained','intermediate_maintained',
          'gradual_maintained','increase_then_relapse','decreasing','complex_fluctuation','transition_censored']
SHORT = ['No clear increase','Stable high','Abrupt + maintained','Intermediate + maintained',
         'Gradual + maintained','Increase then relapse','Decreasing','Complex fluctuation','Censored transition']
CHINESE = dict(zip(LABELS,['无清楚改善','持续较高','短时改善并保持','中间宽度并保持','渐进改善并保持',
                          '改善后回落','下降','明显波动','变化截断']))
STATUS = {'retained':'保留','numerically_undetermined':'数值未确定','likelihood_boundary':'似然边界',
          'likelihood_outside':'似然超出','compatible':'形态相容','incompatible':'形态不相容',
          'undetermined':'形态未确定','incomplete':'未完成'}
COLORS = {'retained':'#29728e','numerically_undetermined':'#b76b2c','likelihood_boundary':'#b76b2c',
          'likelihood_outside':'#8a8a8a','incompatible':'#8a8a8a','undetermined':'#b76b2c','incomplete':'#8a8a8a'}
PARAMETER_NAMES = ['p0','error_gain','failure_decay','global_search','capacity','retention',
                   'beta_init','learning_rate','inertia','gamma']


def configure():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                         'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.8,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig, output, name):
    fig.savefig(output/name,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def flatten(row, phase):
    score=row.get('choice_score',{});check=row.get('shape_check',{});diff=row.get('likelihood_difference',{})
    result=dict(subject=row['subject'],condition=row['condition'],role=row['role'],phase=phase,
                candidate=row['candidate']['id'],proposal_source=row['candidate'].get('source',''),
                **{name:row['candidate']['hyperparams'][key] for name,key in KEYS.items()},
                nll=score.get('nll'),nll_mcse=score.get('mcse'),repeats=score.get('repeats'),
                maximum_system_weight=score.get('maximum_system_weight'),effective_systems=score.get('effective_systems'),
                pooled_trial_nll=score.get('pooled_trial_nll'),delta_nll=diff.get('difference'),delta_mcse=diff.get('mcse'),
                shape_energy=row.get('shape_energy',{}).get('value'),shape_energy_mcse=row.get('shape_energy',{}).get('mc_se'),
                shape_status=check.get('status'),shape_tail=check.get('rank_tail'),shape_tail_low=check.get('tail_mc_low'),
                shape_tail_high=check.get('tail_mc_high'),shape_neighbors=check.get('neighbors'),
                joint_status=row.get('joint_status'),numerically_resolved=row.get('numerically_resolved'),
                failure_decay_inactive=row['candidate']['hyperparams'][KEYS['error_gain']]==0,
                pairing_retention_fixed=row['condition']!=3,
                particle_check_status=row.get('particle_check',{}).get('status'),
                particle_check_gap=row.get('particle_check',{}).get('absolute_difference'),
                particle_check_nll=row.get('particle_check_score',{}).get('nll'),
                observed_shape=row.get('observed_shape',{}).get('label'))
    result.update({f'membership_{k}':v for k,v in row.get('membership',{}).items()})
    return result


def overview(cases, confirmed, output):
    fig,axs=plt.subplots(3,2,figsize=(10,10),layout='constrained')
    for ax,case in zip(axs.flat,cases):
        rows=confirmed[case['subject']]
        for row in rows:
            if not row.get('likelihood_difference') or not row.get('shape_check'):continue
            d=row['likelihood_difference'];s=row['shape_check'];color=COLORS.get(row['joint_status'],'#888888')
            # Plot k/n so the Wilson interval contains its displayed estimate.
            y=s['tail_count']/s['calibration_paths']
            ax.errorbar(d['difference'],y,xerr=2*d['mcse'],
                        yerr=np.array([[y-s['tail_mc_low']],[s['tail_mc_high']-y]]),
                        fmt='o' if row.get('numerically_resolved') else 's',color=color,ms=5,capsize=2)
            ax.annotate(row['candidate']['id'][:6],(d['difference'],y),xytext=(5,5),textcoords='offset points',fontsize=7)
        ax.axvline(.01,color='#777777',ls='--',lw=.8);ax.axhline(.05,color='#777777',ls=':',lw=.8)
        ax.set(title=f"S{case['subject']} | condition {case['condition']} | {case['role']}",
               xlabel='NLL difference per trial (± 2 paired MCSE)',ylabel='Shape calibration tail (Wilson 95%)',ylim=(-.03,1.08))
        if not rows:ax.text(.5,.5,'No completed confirmation',transform=ax.transAxes,ha='center')
    fig.suptitle('Joint candidate checks: blue = retained; amber = unresolved; gray = outside',fontsize=12)
    save(fig,output,'joint_checks.png')


def parameter_lines(cases, confirmed, policy, output):
    names=PARAMETER_NAMES;bounds={k:(min(policy['parameter_values'][k]),max(policy['parameter_values'][k])) for k in names}
    bounds.update(capacity=(2,10),inertia=(0,12),error_gain=(0,12))
    fig,axs=plt.subplots(3,2,figsize=(11,9),layout='constrained')
    for ax,case in zip(axs.flat,cases):
        for row in confirmed[case['subject']]:
            values=[(row['candidate']['hyperparams'][KEYS[k]]-bounds[k][0])/(bounds[k][1]-bounds[k][0]) for k in names]
            ax.plot(range(len(names)),values,'o-',ms=3,lw=1,color=COLORS.get(row.get('joint_status'),'#888888'),
                    alpha=.8,label=row['candidate']['id'][:6])
        ax.set_xticks(range(len(names)),['p0','ae','λ','g0','K','δ','β0','η','κ','γ'])
        ax.set(title=f"S{case['subject']}: complete combinations",ylabel='Scaled to registered outer bounds',ylim=(-.05,1.08))
        ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Each line is one evaluated combination; marginal ranges cannot be freely combined',fontsize=11)
    save(fig,output,'joint_parameters.png')


def subject_paths(case, rows, output):
    valid=[r for r in rows if r.get('shape_file')]
    if not valid:return []
    _,actual=load_observed(Path(case['template']['folder']))
    fig,axs=plt.subplots(len(valid),3,figsize=(15,3.4*len(valid)),squeeze=False,layout='constrained')
    summaries=[]
    for axes,row in zip(axs,valid):
        saved=read(Path(row['shape_file']));records=saved['paths'];observed=saved['observed'];m=len(records)
        choices=completed_arrays(row['generation_jobs'],'choices');feedback=completed_arrays(row['generation_jobs'],'feedback')
        n=choices.shape[1];correct=choices==actual['categories'][:n];real=actual['choices'][:n]==actual['categories'][:n]
        starts=np.arange(0,n,32);centers=np.minimum(starts+16,n)
        blocks=np.array([correct[:,a:a+32].mean(axis=1) for a in starts]).T
        obs=np.array([real[a:a+32].mean() for a in starts])
        ax=axes[0]
        marker='o' if len(centers)==1 else None
        for path in blocks:ax.plot(centers,path,color='#687982',alpha=.035,lw=.6,marker=marker)
        ax.plot(centers,blocks.mean(axis=0),color='#29728e',lw=1.5,marker=marker,label=f'Simulation mean ({m} paths)')
        ax.plot(centers,obs,color='#bd602c',lw=1.5,marker=marker,label='Observed')
        ax.set(xlabel='Trial (non-overlapping 32-trial blocks)',ylabel='Completely correct',ylim=(-.03,1.03),
               title=f"{row['candidate']['id'][:6]} | {row['joint_status']}")
        ax.legend(fontsize=7)
        counts=Counter(r['label'] for r in records);proportions=np.array([counts[l]/m for l in LABELS])
        intervals=np.array([wilson_interval(counts[l],m) for l in LABELS]);ax=axes[1]
        colors=['#bd602c' if l==observed['label'] else '#6c9bae' for l in LABELS]
        ax.barh(range(len(LABELS)),proportions,color=colors,height=.65)
        ax.errorbar(proportions,range(len(LABELS)),xerr=np.maximum(0,np.array([proportions-intervals[:,0],intervals[:,1]-proportions])),
                    fmt='none',ecolor='#555555',lw=.7,capsize=1)
        ax.set_yticks(range(len(LABELS)),SHORT,fontsize=7);ax.invert_yaxis()
        ax.set(xlabel=f'Fraction of all {m} paths (Wilson 95%)',xlim=(0,1.03),title='Orange: observed descriptive label')
        onset=[r['onset'] for r in records if r['onset'] is not None]
        censored=sum(r['censored'] for r in records);ax=axes[2]
        if onset:ax.hist(onset,bins=np.linspace(-n/4,1.25*n,17),color='#6c9bae',edgecolor='white')
        if observed['onset'] is not None:ax.axvline(observed['onset'],color='#bd602c',lw=1.3,label='Observed')
        ax.set(xlabel='Fitted transition center (timing diagnostic)',ylabel='Path count',
               title=f'Transitions {len(onset)}/{m}; censored {censored}/{m}')
        summaries.append(dict(subject=case['subject'],candidate=row['candidate']['id'],paths=m,
             observed_label=observed['label'],transition_paths=len(onset),censored_paths=censored,
             optimizer_unsuccessful=sum(not r['optimizer_success'] for r in records),
             generated_mean_accuracy=float(correct.mean()),observed_mean_accuracy=float(real.mean()),
             generated_final64_mean=float(correct[:,-min(64,n):].mean()),observed_final64=float(real[-min(64,n):].mean()),
             generated_mean_feedback=float(feedback.mean()),observed_mean_feedback=float(actual['feedback'][:n].mean()),
             labels={label:dict(count=counts[label],fraction=counts[label]/m,mc_interval=list(wilson_interval(counts[label],m))) for label in LABELS}))
    fig.suptitle(f"S{case['subject']}: every simulated path retained; timing is a separate diagnostic",fontsize=12)
    save(fig,output,f"paths_S{case['subject']}.png")
    return summaries


def render(study: Path, output: Path):
    if output.exists():raise FileExistsError('Use a new report directory')
    protocol=read(study/'protocol.json');completion=read(study/'completion.json')
    output.mkdir(parents=True);configure()
    cases=sorted(protocol['cases'],key=lambda c:(c['condition'],c['role']=='challenge'))
    searches={c['subject']:read(study/f"search_S{c['subject']}.json") for c in cases}
    confirmed={c['subject']:read(study/f"confirmed_S{c['subject']}.json") for c in cases}
    rows=[flatten(r,phase) for phase,data in [('screen',searches),('confirmation',confirmed)] for batch in data.values() for r in batch]
    pd.DataFrame(rows).to_csv(output/'candidate_scores.csv',index=False)
    baselines=[]
    for case in cases:
        _,observed=load_observed(Path(case['template']['folder']))
        choices=observed['choices'][:case['trials']].astype(int);categories=2 if case['condition']==1 else 4
        frequency=np.bincount(choices,minlength=categories+1)[1:]/len(choices)
        baselines.append(dict(subject=case['subject'],uniform_nll=float(np.log(categories)),
                              constant_response_nll=float(-np.log(frequency[choices-1]).mean()),
                              trials=len(choices),constant_probabilities=frequency.tolist()))
    write(output/'choice_baselines.json',baselines)
    overview(cases,confirmed,output);parameter_lines(cases,confirmed,protocol['policy'],output)
    shape_summaries=[];path_rows=[]
    for case in cases:
        shape_summaries.extend(subject_paths(case,confirmed[case['subject']],output))
        for phase,data in [('screen',searches),('confirmation',confirmed)]:
            for row in data[case['subject']]:
                if not row.get('shape_file'):continue
                records=read(Path(row['shape_file']))['paths']
                for index,item in enumerate(records):
                    path_rows.append(dict(subject=case['subject'],candidate=row['candidate']['id'],phase=phase,path_index=index,
                                          **{k:v for k,v in item.items() if k not in ['features','roughness']},
                                          **{f'feature_{i}':v for i,v in enumerate(item['features'])}))
    pd.DataFrame(path_rows).to_csv(output/'path_shapes.csv',index=False)
    write(output/'shape_summaries.json',shape_summaries)
    all_final=[r for batch in confirmed.values() for r in batch]
    retained=[r for r in all_final if r.get('joint_status')=='retained']
    subject_retained={r['subject'] for r in retained}
    lines=['# Model 1024：六名被试的完整记录联合参数候选研究','',
           f"计算阶段状态：{completion['status']}。本报告为{'短序列流程验证' if protocol['smoke'] else '开发资料先导研究'}。",
           f"{len(cases)}名被试中，{len(subject_retained)}名在主工作容忍0.01下获得保留组合，共{len(retained)}个。",
           '保留表示同时通过本轮选择似然、计算精度与形态相容性工作检查；参数区间置信度和心理特质解释仍待验证。','',
           '## 资料、方法与预算','',
           '每人完整有效选择按原时序计分。所有候选的模型状态逐题学习。实际选择与自主形态均参与参数选择，全部真人案例属于开发资料。',
           '生成沿实际刺激日程和固定长度使用自己的选择与反馈；实验自适应日程及停止规则尚未完整模拟。口述资料未参与。',
           '完整序列似然按整个独立滤波合并；误差单位为完整系统。主要形态距离省略绝对改善时点，并保留水平、变化宽度、保持、波动和截断。',
           'NLL容忍按每题计算：0.01在256题对应总对数似然差2.56，在1088题对应10.88。因此各人的工作集合没有统一置信覆盖含义。',
           '生成相容性使用独立参考/校准模拟和多个近邻距离，全部路径继续参与Energy评分。模拟区间只表示有限模拟误差。','',
           '真人记录已经参与选参；整个选择流程的误判率及参数集合覆盖仍需重复恢复校准。','',
           '“改善并保持”描述变化幅度和持续过程；是否达到高正确率同时依据水平指标判断。','',
           '| 阶段 | 实际CPU小时 | 上限 |','|---|---:|---:|']
    for name,used in completion['cpu_hours'].items():lines.append(f"| {name} | {used:.3f} | {protocol['policy']['cpu_hours'][name]} |")
    if protocol.get('shape_extension'):
        extension=protocol['shape_extension']
        targets='、'.join(f"{r['subject']}/{r['candidate'][:6]}" for r in extension['targets'])
        lines.extend(['',f"形态精度追加：{targets}各增加{extension['extra_paths']}条独立校准路径，原来的{extension['reference_paths']}条参考路径和判断门槛保持固定。",
                      '原256条路径的主结果保留在主任务目录；本报告另外整合已登记的追加批次。追加用于缩小模拟误差，全部新增路径进入分布汇总。'])
    lines.extend(['','## 个体结果','',
                  '| 被试 | 类型 | 有效题数 | 已评分/提出 | 确认记录 | 保留(.005/.01/.02) | 数值未确定 |',
                  '|---|---|---:|---:|---:|---|---:|'])
    for case in cases:
        sid=case['subject'];batch=confirmed[sid]
        counts=[sum(r.get('membership',{}).get(str(e))=='retained' for r in batch) for e in [.005,.01,.02]]
        lines.append(f"| {sid} | {'常规' if case['role']=='regular' else '困难'} | {case['trials']} | "
                     f"{sum(bool(r.get('choice_score')) for r in searches[sid])}/{len(searches[sid])} | {len(batch)} | "
                     f"{' / '.join(map(str,counts))} | {sum(r.get('joint_status')=='numerically_undetermined' for r in batch)} |")
    lines.extend(['','选择评分的描述性参照（每题NLL越小越好）：','',
                  '| 被试 | 均匀作答 | 全段固定反应比例 | 最低确认NLL |','|---|---:|---:|---:|'])
    for baseline in baselines:
        sid=baseline['subject'];values=[r['choice_score']['nll'] for r in confirmed[sid] if r.get('choice_score')]
        best=f'{min(values):.5f}' if values else '未完成'
        lines.append(f"| {sid} | {baseline['uniform_nll']:.5f} | {baseline['constant_response_nll']:.5f} | {best} |")
    lines.extend(['','固定反应比例按该被试整段选择频率估计，全程保持相同概率。这是同资料拟合参照，未作为新的选参门槛，也未构成独立预测检验。'])
    lines.extend(['','![联合检查](joint_checks.png)','',
                  '图中每点是一个独立确认组合。横向误差为两倍配对MCSE，纵向为固定参考模拟下的Wilson计数区间。方块表示数值未确定。',
                  '图中纵坐标为校准计数比例k/n，表中尾部秩为加1校正的(k+1)/(n+1)。',
                  '竖线为每题NLL容忍0.01，横线为形态尾部阈值0.05；最终判断同时考虑误差范围和系统权重。',
                  '','![联合组合](joint_parameters.png)','',
                  '每条线是一组完整参数，按登记外界缩放显示。各参数的边际范围不能任意交叉组合。原始值见下表及CSV。'])
    for case in cases:
        sid=case['subject'];batch=confirmed[sid]
        lines.extend(['',f'## 被试{sid}','',
                      '| 组合ID | NLL ± 2MCSE | 差值 ± 2MCSE | 系统最大权重 | 形态尾部及95%模拟区间 | 主判定 |',
                      '|---|---|---|---:|---|---|'])
        for row in batch:
            c=row['candidate']['id'];score=row.get('choice_score');d=row.get('likelihood_difference');s=row.get('shape_check')
            ns=f"{score['nll']:.5f} ± {2*score['mcse']:.5f}" if score else '未完成'
            ds=f"{d['difference']:.5f} ± {2*d['mcse']:.5f}" if d else '未确定'
            tail=f"{s['rank_tail']:.3f} [{s['tail_mc_low']:.3f}, {s['tail_mc_high']:.3f}]" if s else '未完成'
            weight=f"{score['maximum_system_weight']:.3f}" if score else '—'
            lines.append(f"| {c[:6]} | {ns} | {ds} | {weight} | {tail} | {STATUS.get(row.get('joint_status'),row.get('joint_status'))} |")
        lines.extend(['','| 组合ID | p0 | ae | λ | g0 | K | δ | β0 | η | κ | γ |','|---|'+'---:|'*10])
        for row in batch:
            lines.append('| '+row['candidate']['id'][:6]+' | '+' | '.join(f"{row['candidate']['hyperparams'][KEYS[k]]:.4g}" for k in PARAMETER_NAMES)+' |')
        inactive=[r['candidate']['id'][:6] for r in batch if r['candidate']['hyperparams'][KEYS['error_gain']]==0]
        if inactive:
            lines.extend(['','以下组合的ae=0，λ对行为失去作用，表中统一记0用于去重：'+ '、'.join(inactive)+'。这些组合允许任何合法λ值产生相同的行为概率。'])
        if case['condition']!=3:
            lines.extend(['','本条件的γ固定为1；仅条件3估计该参数。'])
        lines.extend(['','同一参数从初筛到确认的评分变化（两次都使用完整真人记录）：','',
                      '| 组合ID | 初筛NLL | 确认NLL | 初筛形态Energy | 确认形态Energy |',
                      '|---|---:|---:|---:|---:|'])
        initial={r['candidate']['id']:r for r in searches[sid]}
        for row in batch:
            before=initial[row['candidate']['id']]
            if all(x.get('choice_score') and x.get('shape_energy') for x in [before,row]):
                lines.append(f"| {row['candidate']['id'][:6]} | {before['choice_score']['nll']:.5f} | {row['choice_score']['nll']:.5f} | "
                             f"{before['shape_energy']['value']:.4f} | {row['shape_energy']['value']:.4f} |")
        lines.append('参数保持相同；这里的变化来自粒子数、模拟条数和随机种子。Energy越小表示所选形态特征的分布评分越好。')
        summaries=[r for r in shape_summaries if r['subject']==sid]
        if summaries:
            label=CHINESE[summaries[0]['observed_label']]
            lines.extend(['',f'被试的预定描述标签为“{label}”。该标签描述观察行为形态，尚不代表心理学习机制。',
                          '',f'![被试{sid}全部路径及形态](paths_S{sid}.png)','',
                          '左列显示所有模拟路径的32题块正确率、模拟均值和真实路径；均值仅作描述。中列以全部路径为分母，橙色对应被试的描述标签。',
                          '右列只对检测到变化的路径画变化中心，标题列明相应分母与截断数。绝对时点未用于主要形态距离。'])
            for item in summaries:
                lines.append(f"- {item['candidate'][:6]}：完全正确率均值{item['generated_mean_accuracy']:.1%}（真人{item['observed_mean_accuracy']:.1%}），"
                             f"末64题均值{item['generated_final64_mean']:.1%}（真人{item['observed_final64']:.1%}）；"
                             f"清楚变化{item['transition_paths']}/{item['paths']}，截断{item['censored_paths']}/{item['paths']}，描述曲线未收敛{item['optimizer_unsuccessful']}/{item['paths']}。")
        statuses=Counter(r.get('joint_status','incomplete') for r in batch)
        lines.extend(['','本例判定组成：'+'；'.join(f"{STATUS.get(k,k)} {v}个" for k,v in statuses.items())+'。'])
        lines.extend(['','| 组合ID | 增加粒子后的NLL | 两粒子数差值绝对值 | 合并两倍MCSE | 粒子数检查 |',
                      '|---|---:|---:|---:|---|'])
        for row in batch:
            high=row.get('particle_check_score');check=row.get('particle_check',{})
            if high and 'absolute_difference' in check:
                lines.append(f"| {row['candidate']['id'][:6]} | {high['nll']:.5f} | {check['absolute_difference']:.5f} | {2*check['combined_mcse']:.5f} | {check['status']} |")
    calibration=read(study/'calibration.json')
    lines.extend(['','## 计算误差与停止记录','',
                  '| 被试/锚点 | 128粒子NLL | 512粒子NLL | 高粒子两倍MCSE | 高粒子系统最大权重 |',
                  '|---|---:|---:|---:|---:|'])
    for row in calibration:
        low=row.get('calibration_low',{});high=row.get('calibration_high',{})
        if not high:continue
        lines.append(f"| {row['subject']}/{row['candidate'][:6]} | {low.get('nll',float('nan')):.5f} | {high['nll']:.5f} | {2*high['mcse']:.5f} | {high['maximum_system_weight']:.3f} |")
    events=[json.loads(line) for line in (study/'decisions.jsonl').read_text().splitlines()]
    stops=[e for e in events if e['event']=='budget_stop']
    receipts=[read(f) for f in (study/'jobs').glob('*.usage.json')]
    pd.DataFrame([{k:r.get(k) for k in ['job','stage','status','cpu_seconds']} for r in receipts]).to_csv(output/'job_status.csv',index=False)
    failures=[r for r in receipts if r['status']!='complete']
    lines.extend(['',f'预算停止事件{len(stops)}次，工作失败记录{len(failures)}条；完整判断和未提交工作见上级目录 decisions.jsonl。',
                  '初筛和确认使用不同粒子数及种子；初筛排序变化、确认MCSE和系统权重需要联合检查。有限重复稳定不保证粒子近似无偏。',
                  '','## 证据范围与后续判断','',
                  '本轮检验有限候选集合和完整流程的可用性。每人最多24点、最多4点独立确认，仍可能遗漏其他支持区域。',
                  '空集合需要区分数值未确定、选择似然差距、生成形态低概率和未搜索部分。当前结果不能仅凭空集合排除整个认知模型。',
                  '同一组合下的不同形态是过程变异；不同保留组合之间的差异反映参数约束不足。跨组合稳定的行为量才适合进一步解释。',
                  '工作容忍和形态阈值未做参数集合覆盖率校准。完整重复参数恢复、状态恢复和独立真人检验需要后续预算。',
                  '持续执行仍为研究候选，口述待人工复核。既有逐题状态恢复不足的限制继续保留。',
                  '','## 数据与复现','',
                  '- [全部候选评分与参数](candidate_scores.csv)。',
                  '- [均匀作答与固定反应比例参照](choice_baselines.json)。',
                  '- [每条路径的形态记录](path_shapes.csv)。',
                  '- [确认路径概率与汇总](shape_summaries.json)。',
                  '- [全部数值工作的状态与CPU记录](job_status.csv)。',
                  '- 上级目录 protocol.json、registered_config.yaml、source/、jobs/、numerical/ 和 decisions.jsonl 保存输入、配置、代码、种子、原路径和决策。',
                  '- 本报告仅生成PNG；图表使用所有符合其明示分母的路径。不存在挑选最接近真人路径后替代分布评分。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    inputs=[study/'protocol.json',study/'completion.json',study/'calibration.json',study/'decisions.jsonl']
    inputs += [study/f'{prefix}_S{c["subject"]}.json' for c in cases for prefix in ['search','confirmed']]
    write(output/'provenance.json',dict(created_unix=time.time(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          input_sha256={str(f.relative_to(study)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
          candidate_rows=len(rows),path_rows=len(path_rows),retained_main=len(retained),subjects_retained=len(subject_retained),
          statistical_scope='development; Monte Carlo intervals only'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=resource.getrusage(resource.RUSAGE_SELF)
    render(args.study,args.output)
    end=resource.getrusage(resource.RUSAGE_SELF)
    write(args.output/'report_cpu.json',dict(cpu_seconds=end.ru_utime+end.ru_stime-start.ru_utime-start.ru_stime))
    seconds=end.ru_utime+end.ru_stime-start.ru_utime-start.ru_stime
    completion=read(args.study/'completion.json')
    with (args.output/'report.md').open('a') as stream:
        stream.write(f'\n本次报告生成另用{seconds:.2f} CPU秒；加上数值阶段，共{sum(completion["cpu_hours"].values())+seconds/3600:.3f} CPU小时。\n')


if __name__=='__main__':main()
