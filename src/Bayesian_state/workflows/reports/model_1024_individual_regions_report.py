"""Render the registered individual-parameter pilot without rerunning the model.

All candidate rows and all generated paths remain in exported data. Figures are
development diagnostics; intervals describe Monte Carlo variation. Run with
--study EXISTING_STUDY --output NEW_REPORT_DIRECTORY after numerical completion.
Exports include complete retained combinations, per-path descriptors and source
snapshots. Single-feature discrepancies and interrupted CPU accounting remain
visible alongside the registered joint working checks.

Individual panels retain every autonomous path and its ensemble mean. The
observed path is solid black; the mean is dashed blue; one complete generated
path with minimum Euclidean distance in the registered weighted shape features
is solid orange. Orange bars identify the observed label; an open diamond
identifies the selected generated path's label, which can differ. This selection
uses the complete observed record for illustration only. Its identity, seed
and distance are exported, and it does not enter parameter acceptance checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
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
NUMERICAL_STATUS = {'stable':'稳定','sensitive':'粒子数敏感','boundary':'数值边界',
                    'system_weight_unresolved':'系统权重过度集中','incomplete':'未完成'}
PARAMETER_NAMES = ['p0','error_gain','failure_decay','global_search','capacity','retention',
                   'beta_init','learning_rate','inertia','gamma']
PARAMETER_SYMBOLS = dict(zip(PARAMETER_NAMES,['p0','ae','λ','g0','K','δ','β0','η','κ','γ']))


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
                ordered_energy_diagnostic=row.get('ordered_energy_diagnostic',{}).get('value'),
                ordered_energy_mcse=row.get('ordered_energy_diagnostic',{}).get('mc_se'),
                shape_status=check.get('status'),shape_tail=check.get('rank_tail'),shape_tail_low=check.get('tail_mc_low'),
                shape_tail_high=check.get('tail_mc_high'),shape_neighbors=check.get('neighbors'),
                joint_status=row.get('joint_status'),numerically_resolved=row.get('numerically_resolved'),
                failure_decay_inactive=row['candidate']['hyperparams'][KEYS['error_gain']]==0,
                pairing_retention_fixed=row['condition']!=3,
                particle_check_status=row.get('particle_check',{}).get('status'),
                particle_check_gap=row.get('particle_check',{}).get('absolute_difference'),
                particle_check_nll=row.get('particle_check_score',{}).get('nll'),
                particle_check_repeats=row.get('particle_check_score',{}).get('repeats'),
                primary_particle_check_nll=row.get('primary_particle_check_score',{}).get('nll'),
                observed_shape=row.get('observed_shape',{}).get('label'))
    result.update({f'membership_{k}':v for k,v in row.get('membership',{}).items()})
    return result


def overview(cases, confirmed, output):
    fig,axs=plt.subplots(3,2,figsize=(10,10),layout='constrained')
    labels=[]
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
            labels.append((ax,row['candidate']['id'][:6],d['difference'],y,color))
        ax.axvline(.01,color='#777777',ls='--',lw=.8);ax.axhline(.05,color='#777777',ls=':',lw=.8)
        ax.set(title=f"S{case['subject']} | condition {case['condition']} | {case['role']}",
               xlabel='NLL difference per trial (± 2 paired MCSE)',ylabel='Shape calibration tail (Wilson 95%)',ylim=(-.03,1.08))
        if not rows:ax.text(.5,.5,'No completed confirmation',transform=ax.transAxes,ha='center')
    fig.suptitle('Joint candidate checks: blue = retained; amber = unresolved; gray = outside',fontsize=12)
    fig.canvas.draw()
    occupied={}
    for ax,label,x,y,color in labels:
        px,py=ax.transData.transform((x,y));box=ax.get_window_extent()
        width,height=42.,13.;left=np.clip(px+5,box.x0+3,box.x1-width-3)
        positions=occupied.setdefault(ax,[])
        for offset in [7,23,-23,39,-39,55,-55]:
            bottom=np.clip(py+offset,box.y0+3,box.y1-height-3)
            if all(left+width<a or a+width<left or bottom+height<b or b+height<bottom for a,b in positions):break
        positions.append((left,bottom))
        text_position=ax.transAxes.inverted().transform((left,bottom))
        ax.annotate(label,(x,y),xytext=text_position,textcoords='axes fraction',fontsize=7,ha='left',va='bottom',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.3),
                    arrowprops=dict(arrowstyle='-',color=color,lw=.5))
    save(fig,output,'joint_checks.png')


def parameter_lines(cases, confirmed, policy, output):
    names=PARAMETER_NAMES;bounds={k:(min(policy['parameter_values'][k]),max(policy['parameter_values'][k])) for k in names}
    bounds.update(capacity=(2,10),inertia=(0,12),error_gain=(0,12))
    fig,axs=plt.subplots(3,2,figsize=(11,9),layout='constrained')
    for ax,case in zip(axs.flat,cases):
        for index,row in enumerate(confirmed[case['subject']]):
            values=[(row['candidate']['hyperparams'][KEYS[k]]-bounds[k][0])/(bounds[k][1]-bounds[k][0]) for k in names]
            ax.plot(range(len(names)),values,marker=['o','s','^','D'][index%4],ls=['-','--',':','-.'][index%4],
                    ms=3.5,lw=1.2,color=COLORS.get(row.get('joint_status'),'#888888'),alpha=.85,label=row['candidate']['id'][:6])
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
        seeds=completed_arrays(row['generation_jobs'],'seeds')
        features=np.array([record['features'] for record in records],dtype=float)
        target_features=np.asarray(observed['features'],dtype=float)
        if (len(choices)!=m or len(seeds)!=m or features.ndim!=2 or
                target_features.shape!=features.shape[1:] or not np.isfinite(features).all() or
                not np.isfinite(target_features).all()):
            raise ValueError('Path, seed and finite descriptor records must align')
        distances=np.linalg.norm(features-target_features,axis=1)
        # Features already contain the registered scaling and group weights.
        # A tie keeps the first saved path; retain its original trial order.
        nearest=int(np.argmin(distances));nearest_record=records[nearest]
        nearest_summary=dict(path_index=nearest,path_number=nearest+1,seed=int(seeds[nearest]),
                             shape_distance=float(distances[nearest]),paths_considered=m,
                             label=nearest_record['label'],onset=nearest_record['onset'],
                             choice_sha256=hashlib.sha256(np.asarray(choices[nearest],dtype='<i8').tobytes()).hexdigest(),
                             metric='euclidean_registered_weighted_shape_features',
                             tie_break='first_saved_path',use='posthoc_illustration_only')
        n=choices.shape[1];correct=choices==actual['categories'][:n];real=actual['choices'][:n]==actual['categories'][:n]
        starts=np.arange(0,n,32);centers=np.minimum(starts+16,n)
        blocks=np.array([correct[:,a:a+32].mean(axis=1) for a in starts]).T
        obs=np.array([real[a:a+32].mean() for a in starts])
        ax=axes[0]
        marker='o' if len(centers)==1 else None
        for path in blocks:ax.plot(centers,path,color='#687982',alpha=min(.035,9./m),lw=.6,marker=marker)
        ax.plot(centers,blocks.mean(axis=0),color='#29728e',lw=1.5,ls='--',marker=marker,label=f'Simulation mean ({m} paths)')
        ax.plot(centers,blocks[nearest],color='#bd602c',lw=1.5,marker=marker,
                label=f'Closest shape (path {nearest+1}/{m})',zorder=3)
        ax.plot(centers,obs,color='#171717',lw=1.5,marker=marker,label='Observed',zorder=4)
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
        nearest_label_index=LABELS.index(nearest_record['label'])
        ax.scatter([proportions[nearest_label_index]],[nearest_label_index],marker='D',s=24,
                   facecolors='white',edgecolors='#bd602c',linewidths=1.1,zorder=4)
        ax.set(xlabel=f'Fraction of all {m} paths (Wilson 95%)',xlim=(0,1.03),
               title='Orange bar: observed label; diamond: closest-path label')
        onset=[r['onset'] for r in records if r['onset'] is not None]
        censored=sum(r['censored'] for r in records);ax=axes[2]
        if onset:ax.hist(onset,bins=np.linspace(-n/4,1.25*n,17),color='#6c9bae',edgecolor='white')
        if nearest_record['onset'] is not None:
            ax.axvline(nearest_record['onset'],color='#bd602c',lw=1.3,label='Closest shape')
        if observed['onset'] is not None:
            ax.axvline(observed['onset'],color='#171717',lw=1.3,label='Observed')
        if observed['onset'] is not None or nearest_record['onset'] is not None:ax.legend(fontsize=7)
        ax.set(xlabel='Fitted transition center (timing diagnostic)',ylabel='Path count',
               title=f'Transitions {len(onset)}/{m}; censored {censored}/{m}')
        summaries.append(dict(subject=case['subject'],candidate=row['candidate']['id'],paths=m,
             nearest_path=nearest_summary,
             observed_label=observed['label'],transition_paths=len(onset),censored_paths=censored,
             optimizer_unsuccessful=sum(not r['optimizer_success'] for r in records),
             observed_label_count=counts[observed['label']],observed_dispersion=observed['dispersion'],
             generated_dispersion_median=float(np.median([r['dispersion'] for r in records])),
             generated_dispersion_q95=float(np.quantile([r['dispersion'] for r in records],.95)),
             generated_mean_accuracy=float(correct.mean()),observed_mean_accuracy=float(real.mean()),
             generated_final64_mean=float(correct[:,-min(64,n):].mean()),observed_final64=float(real[-min(64,n):].mean()),
             generated_mean_feedback=float(feedback.mean()),observed_mean_feedback=float(actual['feedback'][:n].mean()),
             labels={label:dict(count=counts[label],fraction=counts[label]/m,mc_interval=list(wilson_interval(counts[label],m))) for label in LABELS}))
    fig.suptitle(f"S{case['subject']}: all simulated paths retained; orange = closest shape (post-hoc illustration)",fontsize=12)
    save(fig,output,f"paths_S{case['subject']}.png")
    return summaries


def render(study: Path, output: Path, *, link_base: Path | None = None):
    """Write a fresh report, optionally resolving links for its final location."""
    if output.exists():raise FileExistsError('Use a new report directory')
    link_base=output if link_base is None else link_base
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
    write(output/'nearest_paths.json',dict(
        metric='euclidean_registered_weighted_shape_features',
        selection='minimum over all saved confirmation paths; first saved path breaks a tie',
        temporal_alignment='original full path; no time shift or splicing',
        scope='selected using the complete human record for illustration; no change to scientific scores or membership',
        paths=[dict(subject=r['subject'],candidate=r['candidate'],**r['nearest_path']) for r in shape_summaries]))
    all_final=[r for batch in confirmed.values() for r in batch]
    retained=[r for r in all_final if r.get('joint_status')=='retained']
    subject_retained={r['subject'] for r in retained}
    summary_index={(r['subject'],r['candidate']):r for r in shape_summaries}
    regions=[]
    for case in cases:
        sets={}
        for tolerance in protocol['policy']['search']['score_tolerances']:
            sets[str(tolerance)]=[
                dict(candidate=r['candidate'],choice_score=r['choice_score'],
                     likelihood_difference=r['likelihood_difference'],shape_check=r['shape_check'],
                     path_summary=summary_index.get((case['subject'],r['candidate']['id'])),
                     inactive_parameters=['failure_decay'] if r['candidate']['hyperparams'][KEYS['error_gain']]==0 else [],
                     fixed_parameters=['gamma'] if case['condition']!=3 else [])
                for r in confirmed[case['subject']] if r.get('membership',{}).get(str(tolerance))=='retained']
        regions.append(dict(subject=case['subject'],condition=case['condition'],trials=case['trials'],sets=sets))
    write(output/'retained_regions.json',dict(scope='finite development working sets; no calibrated confidence or posterior weights',
                                             freely_combinable_marginal_ranges=False,cases=regions))
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
           '表中NLL为每题平均负对数似然，越小表示模型给实际选择的概率越高。MCSE表示更换计算随机种子造成的数值误差。',
           '一次完整粒子滤波称为一个独立系统。系统权重表示各次滤波对合并序列似然的贡献，规则信念在模型内部另有定义。',
           '形态尾部比例表示：有多少校准路径比真人离模拟常见形态更远。比例很小，说明真人形态位于模型较少生成的位置。',
           'Energy综合真人与模拟路径的距离及模拟路径之间的距离，越小表示分布评分越好。所有这些判断均限定于预定行为指标。','',
           '波动指标概括32题块正确率偏离描述曲线的程度，较大数值表示更强的反复起伏。下文同时报告这个单项指标及与真人相同描述标签的生成次数。','',
           '真人记录已经参与选参；整个选择流程的误判率及参数集合覆盖仍需重复恢复校准。','',
           '“改善并保持”描述变化幅度和持续过程；是否达到高正确率同时依据水平指标判断。','',
           '九个标签是本轮制定的互斥描述规则，尚未验证为心理学习类型。短时、中间宽度、渐进分别按变化宽度≤32题、32至64题之间、≥64题划分；绝对宽度门槛的解释会受到任务长度影响。',
           '“持续较高”表示整段正确率≥80%且变化未通过识别门槛；“变化截断”表示变化越过记录边界或结束后不足64题可观察，它反映观察限制。明显波动优先分类，其他细节见登记协议。','',
           '分类修订讨论：改善形态与保持证据应分别描述。渐进改善可以同时具有“已观察到保持”或“保持尚无法判断”的状态；观察边界标记保留为独立信息。当前数值表及柱状图仍使用本轮登记的原分类，新的分类方案尚未应用。','',
           '| 阶段 | 计入预算的CPU小时 | 上限 |','|---|---:|---:|']
    for name,used in completion['cpu_hours'].items():lines.append(f"| {name} | {used:.3f} | {protocol['policy']['cpu_hours'][name]} |")
    component_cautions=[]
    for row in all_final:
        summary=summary_index.get((row['subject'],row['candidate']['id']))
        if summary and summary['observed_label_count']==0 and row.get('shape_check',{}).get('status')=='compatible':
            component_cautions.append((row,summary))
    if component_cautions:
        lines.extend(['','综合形态评分之外仍有单项偏差：',''])
        for row,item in component_cautions:
            lines.append(f"- {row['subject']}/{row['candidate']['id'][:6]}通过综合形态检查，但真人的“{CHINESE[item['observed_label']]}”标签出现次数为0/{item['paths']}。该组合对这一行为特征的生成仍不足。")
    if protocol.get('shape_extension'):
        extension=protocol['shape_extension']
        targets='、'.join(f"{r['subject']}/{r['candidate'][:6]}" for r in extension['targets'])
        lines.extend(['',f"形态精度追加：{targets}各增加{extension['extra_paths']}条独立校准路径，原来的{extension['reference_paths']}条参考路径和判断门槛保持固定。",
                      '原256条路径的主结果保留在主任务目录；本报告另外整合已登记的追加批次。追加用于缩小模拟误差，全部新增路径进入分布汇总。'])
    if protocol.get('particle_precision_extension'):
        extension=protocol['particle_precision_extension']
        lines.extend(['',f"粒子数对照追加：全部{len(extension['targets'])}个尚未稳定的对照点，各增加{extension['extra_repeats']}次{extension['particles']}粒子过滤。",
                      '独立系统仍按完整序列似然合并。基础粒子数的结果、参数、轨迹检查和判断门槛固定。',
                      '此次仅将未使用的校准与搜索预算转入数值确认，总上限保持96 CPU小时。一次追加后停止，仍未确定的结果如实保留。'])
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
                  '横轴越靠左表示与该人的最佳确认点选择评分越接近。纵轴很低提示真人形态在该参数的模拟中较少见；纵值不是“复现真人的概率”，也不是用于最大化的拟合优度。',
                  '蓝色保留、橙色未确定、灰色未通过。误差线描述计算或模拟计数的不确定性；落在左上方的点仍须通过粒子数稳定性等检查。',
                  '','![联合组合](joint_parameters.png)','',
                  '每条线是一组完整参数，横轴列出不同参数，连线用于追踪同一组合。纵轴按各参数登记边界缩放到0—1；它不表示概率、正确率或参数的可信度。',
                  '例如容量K按2—10缩放，K=3显示为0.125。颜色沿用联合检查图。各参数的边际范围不能任意交叉组合，原始值见下表及CSV。'])
    for case in cases:
        sid=case['subject'];batch=confirmed[sid]
        reference=next((r.get('reference_id') for r in batch if r.get('reference_id')),None)
        lines.extend(['',f'## 被试{sid}',''])
        if reference:
            lines.extend([f'相对似然以确认NLL最低的组合{reference[:6]}为基准。每个比较要求该组合与基准均通过数值检查；基准未通过时，相关比较记为数值未确定。',''])
        lines.extend([
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
        edges=[]
        for row in batch:
            if row.get('joint_status')!='retained':continue
            values=row['candidate']['hyperparams'];contacts=[]
            for name in PARAMETER_NAMES:
                if name=='gamma' and case['condition']!=3:continue
                if name=='failure_decay' and values[KEYS['error_gain']]==0:continue
                lower=min(protocol['policy']['parameter_values'][name]);upper=max(protocol['policy']['parameter_values'][name])
                value=values[KEYS[name]]
                if value<=lower or value>=upper:
                    contacts.append(f'{PARAMETER_SYMBOLS[name]}={value:.4g}')
            if contacts:edges.append(row['candidate']['id'][:6]+'（'+ '、'.join(contacts)+'）')
        if edges:
            lines.extend(['','保留组合中触及或超过初始粗候选端点的取值：'+ '；'.join(edges)+'。',
                          '端点既包括0、1等机制边界，也包括本轮人为限定的搜索范围。已检查的组合与外扩点见全部候选CSV；当前集合的外缘尚未完整确定。'])
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
            observed=next(r['observed_shape'] for r in batch if r.get('observed_shape'))
            lines.extend(['',f'被试的预定描述标签为“{label}”。该标签描述观察行为形态，尚不代表心理学习机制。'])
            if observed['supported_transition']:
                lines.append(f"原始逐题记录拟合的变化中心为{observed['onset']:.1f}（从0计数），10%—90%变化宽度为{observed['transition_width']:.1f}题；"
                             f"变化范围{observed['transition_start']:.1f}—{observed['transition_end']:.1f}，可观察随访{observed['followup_trials']}题。标签由变化、波动和观察边界规则共同确定，心理学习类型仍需进一步证据。")
                if observed['censored'] and observed['gain']>0:
                    lines.append('该轨迹已经显示改善。原分类优先报告观察边界限制，因此改善宽度与保持证据需要结合上面的连续数值分别阅读；随访不足不能作为“没有改善”或“已经保持”的证据。')
            else:
                lines.append(f"变化描述的BIC改善为{observed['bic_gain']:.2f}，本轮门槛为{protocol['policy']['shape']['bic_gain']:g}；"
                             f"整段正确率为{observed['mean_accuracy']:.1%}。未通过变化识别门槛时仍保留原始曲线及波动特征。")
            lines.extend([
                          '',f'![被试{sid}全部路径及形态](paths_S{sid}.png)','',
                          '左列浅灰线为全部自主路径，蓝色虚线为模拟均值，黑色实线为真人，橙色实线为本组中形态最接近真人的一条完整模拟路径；均以互不重叠的32题块显示。',
                          '橙线按登记的加权形态特征欧氏距离最小值选取，保留原始时间位置和题序；并列时选保存顺序最早者。该距离允许改善时点不同，选中路径用于事后展示，参数判定继续使用原来的全部路径评分。',
                          '中列以全部路径为分母，橙色柱对应真人的描述标签，橙色空心菱形标记左侧橙线所属的标签。两者可能不同；同标签比例概括描述类别，不能直接解释为与真人在所有形态指标上都接近的比例。',
                          '右列统计检测到变化的路径；存在有效变化中心时，黑色实线对应真人，橙色实线对应选中的模拟路径。',
                          '右列标题分别列变化路径数和受观察边界限制的路径数，两者有重叠。绝对时点未用于主要形态距离。',''])
            for item in summaries:
                lines.append(f"- {item['candidate'][:6]}：完全正确率均值{item['generated_mean_accuracy']:.1%}（真人{item['observed_mean_accuracy']:.1%}），"
                             f"末64题均值{item['generated_final64_mean']:.1%}（真人{item['observed_final64']:.1%}）；"
                             f"清楚变化{item['transition_paths']}/{item['paths']}，截断{item['censored_paths']}/{item['paths']}，描述曲线未收敛{item['optimizer_unsuccessful']}/{item['paths']}。")
                lines.append(f"  与真人相同描述标签{item['observed_label_count']}/{item['paths']}；真人波动指标{item['observed_dispersion']:.3f}，"
                             f"模拟中位数{item['generated_dispersion_median']:.3f}、95%分位{item['generated_dispersion_q95']:.3f}。")
                closest=item['nearest_path']
                lines.append(f"  橙色路径为第{closest['path_number']}/{item['paths']}条，种子{closest['seed']}，加权形态距离{closest['shape_distance']:.5f}，标签为“{CHINESE[closest['label']]}”。")
        statuses=Counter(r.get('joint_status','incomplete') for r in batch)
        lines.extend(['','本例判定组成：'+'；'.join(f"{STATUS.get(k,k)} {v}个" for k,v in statuses.items())+'。'])
        lines.extend(['','| 组合ID | 对照重复数 | 对照NLL（追加前→当前） | 两粒子数差值绝对值 | 合并两倍MCSE | 粒子数检查 |',
                      '|---|---:|---:|---:|---:|---|'])
        for row in batch:
            high=row.get('particle_check_score');check=row.get('particle_check',{})
            if high and 'absolute_difference' in check:
                prior=row.get('primary_particle_check_score')
                values=(f"{prior['nll']:.5f} → " if prior else '')+f"{high['nll']:.5f}"
                lines.append(f"| {row['candidate']['id'][:6]} | {high['repeats']} | {values} | {check['absolute_difference']:.5f} | {2*check['combined_mcse']:.5f} | {NUMERICAL_STATUS.get(check['status'],check['status'])} |")
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
    pd.DataFrame([{k:r.get(k) for k in ['job','stage','status','cpu_seconds','cpu_accounting','reserved_cpu_seconds']}
                  for r in receipts]).to_csv(output/'job_status.csv',index=False)
    failures=[r for r in receipts if r['status']!='complete']
    lines.extend(['',f'预算停止事件{len(stops)}次，工作失败记录{len(failures)}条；完整判断和未提交工作见上级目录 decisions.jsonl。'])
    for phase,name in [('screen','初筛'),('confirmation','确认')]:
        subset=[r for r in path_rows if r['phase']==phase]
        lines.append(f"{name}路径的描述曲线有{sum(not r['optimizer_success'] for r in subset)}/{len(subset)}条未报告收敛；全部有限拟合结果按登记规则保留。")
    reuse=study/'reuse_manifest.json'
    if reuse.exists():
        counts=read(reuse).get('status_counts',{})
        lines.extend([f"其中复用的早期账本包含{counts.get('complete',0)}次成功和{counts.get('failed',0)}次失败。早期失败来自λ=1超出实现范围；修订3已改用0.995并重新计算相应组合。所有耗时均计入预算。"])
    if protocol.get('particle_precision_extension'):
        ledger=Path(protocol['particle_precision_extension']['ledger'])
        for file in sorted(ledger.glob('particle_precision_v*/interrupted.json')):
            audit=read(file);link=Path(os.path.relpath(file,link_base)).as_posix()
            lines.extend([f"精度补充发生一次退出码{audit['exit_code']}的进程中断，{len(audit['jobs'])}项工作未留下可复用结果。日志未确定终止原因。",
                          f"丢失CPU记录的单线程工作按完整墙钟时间保守计入{audit['worker_cpu_hours_charged']:.3f} CPU小时，协调器间隙另计{audit['coordinator_gap_cpu_hours_charged']:.3f}小时。实际CPU耗时无法精确还原。",
                          f"恢复采用新任务名和20并发，全部中断费用继续计入96小时预算。详见[中断与预算记录]({link})。"])
        file=ledger/'descriptor_convergence_diagnostic.json'
        if file.exists():
            diagnostic=read(file);link=Path(os.path.relpath(file,link_base)).as_posix()
            lines.append(f"初筛未收敛曲线的独立诊断：迭代上限增加至600后收敛，形态特征距离变化为{diagnostic['feature_distance']:.3g}。原评分保持不变；详见[描述曲线检查]({link})。")
    lines.extend([
                  '初筛和确认使用不同粒子数及种子；初筛排序变化、确认MCSE和系统权重需要联合检查。有限重复稳定不保证粒子近似无偏。',
                  '','## 证据范围与后续判断','',
                  '本轮检验有限候选集合和完整流程的可用性。每人最多24点、最多4点独立确认，仍可能遗漏其他支持区域。',
                  '空集合需要区分数值未确定、选择似然差距、生成形态低概率和未搜索部分。当前结果不能仅凭空集合排除整个认知模型。',
                  '同一组合下的不同形态是过程变异；不同保留组合之间的差异反映参数约束不足。跨组合稳定的行为量才适合进一步解释。',
                  '工作容忍和形态阈值未做参数集合覆盖率校准。完整重复参数恢复、状态恢复和独立真人检验需要后续预算。',
                  '持续执行仍为研究候选，口述待人工复核。既有逐题状态恢复不足的限制继续保留。',
                  '','## 数据与复现','',
                  '- [全部候选评分与参数](candidate_scores.csv)。',
                  '- [三档容忍下保留的完整组合及其轨迹分布](retained_regions.json)。',
                  '- [均匀作答与固定反应比例参照](choice_baselines.json)。',
                  '- [每条路径的形态记录](path_shapes.csv)。',
                  '- [确认路径概率与汇总](shape_summaries.json)。',
                  '- [每组最近形态路径的编号、种子、距离与选择规则](nearest_paths.json)。',
                  '- [全部数值工作的状态与CPU记录](job_status.csv)。',
                  '- 上级目录 protocol.json、registered_config.yaml、source/、jobs/、numerical/ 和 decisions.jsonl 保存输入、配置、代码、种子、原路径和决策。',
                  '- 本报告仅生成PNG；图表保留全部路径，最近形态路径另行突出显示。该路径在看到完整真人记录后选择，全部路径的分布评分及参数判定保持原定义。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    inputs=[study/'protocol.json',study/'completion.json',study/'calibration.json',study/'decisions.jsonl']
    inputs += [study/f'{prefix}_S{c["subject"]}.json' for c in cases for prefix in ['search','confirmed']]
    write(output/'provenance.json',dict(created_unix=time.time(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          input_sha256={str(f.relative_to(study)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
          candidate_rows=len(rows),path_rows=len(path_rows),retained_main=len(retained),subjects_retained=len(subject_retained),
          statistical_scope='development; Monte Carlo intervals only'))
    (output/'report_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'decisions_at_report.jsonl').write_bytes((study/'decisions.jsonl').read_bytes())


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
        stream.write(f'\n本次报告生成另用{seconds:.2f} CPU秒；加上数值阶段，共计入预算{sum(completion["cpu_hours"].values())+seconds/3600:.3f} CPU小时。\n')


if __name__=='__main__':main()
