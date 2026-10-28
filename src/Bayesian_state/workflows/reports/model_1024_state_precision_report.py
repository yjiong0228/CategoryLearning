"""Render numerical precision evidence without reinterpreting oral reports.

Use --source PRECISION_RUN --output NEW_REPORT from the repository root.
Figures retain every checked resolution and distinguish numerical differences
from posterior uncertainty. PNG is the repository's requested export format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis import model_1024_state_precision as precision
from ..analysis import model_1024_individual_state_summary as summary

read, write, digest = precision.read, precision.write, precision.digest
COLORS = {512:'#a8acb0',1024:'#5b738c',2048:'#3a8b82',4096:'#427bb3',8192:'#d28a35',16384:'#8b5e9c'}
TARGET_LABELS = {'target_present':'Correct rule present', 'target_belief':'Belief on correct rule',
                 'target_high_belief':'Correct rule present with belief >= 0.5',
                 'target_execution':'Correct rule executed'}


def load_rows(source: Path) -> list[dict]:
    unique={}
    for folder in sorted((source/'summaries').iterdir(),key=lambda f:f.stat().st_mtime):
        file=folder/'diagnostics.json'
        if not file.exists():
            continue
        for row in read(file):
            key=(row['case'],row['particles'],row['systems'],row['confirmation'])
            unique[key]=row
    return sorted(unique.values(),key=lambda r:(r['case'],r['particles'],r['systems'],r['confirmation']))


def arrays(row: dict) -> dict:
    file=Path(row['pooled_file'])
    if digest(file)!=row['pooled_sha256']:
        raise ValueError('Summary changed')
    with np.load(file) as z:
        return {k:z[k] for k in z.files}


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output/'figures'/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def cost_records(source: Path) -> list[dict]:
    rows=[]
    for file in sorted((source/'ledger').glob('*.json')):
        r=read(file);job=read(source/'jobs'/file.name)
        rows.append(dict(case=r['case'],particles=r['particles'],stage=job['state_stage'],
            name=r['name'],status=r['status'],cpu_hours=r['cpu_seconds']/3600,
            wall_hours=r['wall_seconds']/3600,peak_memory_gib=r['max_rss_kib']/1024**2))
    return rows


def same_work(source: Path, rows: list[dict], output: Path) -> list[dict]:
    result=[]
    for cid in sorted({r['case'] for r in rows}):
        for low,high in [((2048,16),(4096,8)),((2048,32),(4096,16)),((4096,32),(8192,16))]:
            banks=[]
            for n,r in (low,high):
                jobs=[]
                for i in range(r):
                    file=source/'jobs'/f'precision_n{n}_{cid}_{i:03d}.json'
                    ledger=source/'ledger'/file.name
                    if ledger.exists() and read(ledger)['status']=='complete':
                        jobs.append(read(file))
                if len(jobs)!=r:
                    break
                pooled,diag=summary.summarize_bank(jobs)
                cpu=sum(read(source/'ledger'/(j['name']+'.json'))['cpu_seconds'] for j in jobs)
                banks.append((pooled,diag,cpu))
            if len(banks)==2:
                comparison=summary.discrepancy(banks[0][0],banks[1][0],'full')
                result.append(dict(case=cid,low_particles=low[0],low_systems=low[1],high_particles=high[0],high_systems=high[1],
                    low_cpu_hours=banks[0][2]/3600,high_cpu_hours=banks[1][2]/3600,
                    low_full_half_mean=banks[0][1]['full_half_mean'],high_full_half_mean=banks[1][1]['full_half_mean'],
                    low_full_half_q95=banks[0][1]['full_half_q95'],high_full_half_q95=banks[1][1]['full_half_q95'],
                    low_full_system_ess=banks[0][1]['full_ess_min'],high_full_system_ess=banks[1][1]['full_ess_min'],
                    cross_mean=comparison['mean'],cross_q95=comparison['q95']))
    write(output/'same_nominal_work.json',result)
    return result


def report(source: Path, output: Path) -> None:
    started=time.time()
    source_hash=digest(Path(__file__))
    if output.exists():
        raise FileExistsError(output)
    (output/'figures').mkdir(parents=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False,'savefig.bbox':'tight'})
    protocol=read(source/'protocol.json'); rows=load_rows(source)
    distribution_folders=[d for d in (source/'distribution_diagnostics'/'summaries').glob('*')
                          if (d/'completion.json').exists()]
    distribution_files=[d/'diagnostics.json' for d in sorted(distribution_folders,key=lambda d:d.stat().st_mtime)]
    distribution_file=distribution_files[-1] if distribution_files else None
    distribution_unique={}
    for file in distribution_files:
        for row in read(file):
            distribution_unique[(row['case'],row['particles'],row['systems'],row['confirmation'])]=row
    distribution_rows=list(distribution_unique.values())
    if distribution_rows:
        pd.DataFrame(distribution_rows).to_csv(output/'distribution_diagnostics.csv',index=False)
    costs=cost_records(source); confirmations=[read(f) for f in (source/'confirmations').glob('*.json')]
    pd.DataFrame([{k:v for k,v in r.items() if k!='job_names'} for r in rows]).to_csv(output/'numerical_diagnostics.csv',index=False)
    pd.DataFrame(costs).to_csv(output/'compute_costs.csv',index=False)
    write(output/'confirmations.json',confirmations)
    equal=same_work(source,rows,output)
    primary=[r for r in rows if not r['confirmation']]
    cases=[c for c in protocol['cases'] if c['id'] in {r['case'] for r in primary}]
    case_rows={c['id']:[r for r in primary if r['case']==c['id']] for c in cases}
    figures=[]
    # Comparison panels show the registered difference gates, including failures.
    for c in cases:
        own=case_rows[c['id']]; labels=[f'{r["particles"]}\n×{r["systems"]}' for r in own]; x=np.arange(len(own))
        fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
        for col,mode in enumerate(('filtered','full')):
            for index,metric in enumerate(('mean','q95')):
                ax=axes[index,col]
                ax.plot(x,[r.get(mode+'_half_'+metric,np.nan)*100 for r in own],'-o',color='#427bb3',label='Independent halves')
                ax.plot(x,[r.get(mode+'_comparison_'+metric,np.nan)*100 for r in own],'--s',color='#d28a35',label='Adjacent particle counts')
                ax.axhline(2.5 if metric=='mean' else 10,color='#444444',ls=':',lw=1,label='Working threshold')
                ax.set(xticks=x,xticklabels=labels,ylabel='Absolute difference (percentage points)',
                       title=f'{"Filtering" if mode=="filtered" else "Full history"}: {"mean" if metric=="mean" else "95th percentile"}',
                       xlabel='Particles × independent systems',ylim=(0,None))
                if index==0 and col==0:
                    ax.legend(fontsize=8)
        fig.suptitle(f'S{c["subject"]} | {c["candidate"]["id"]} | numerical agreement')
        name=c['id']+'_agreement';save(fig,output,name);figures.append(name)
        # Complete sequence probability curves; the final two halves form a numerical range.
        loaded=[arrays(r) for r in own]
        fig,axes=plt.subplots(4,2,figsize=(13,11),layout='constrained',sharex=True,sharey=True)
        for col,mode in enumerate(('filtered','full')):
            for row_index,field in enumerate(summary.TARGETS):
                ax=axes[row_index,col]
                for row,a in zip(own,loaded):
                    ax.plot(np.arange(len(a[mode+'_'+field]))+1,a[mode+'_'+field],lw=1.,alpha=.82,
                            color=COLORS[row['particles']],ls={8:'-.',16:'-',32:'--',64:':'}.get(row['systems'],'--'),
                            label=f'N={row["particles"]}, R={row["systems"]}')
                latest=loaded[-1];t=np.arange(len(latest[mode+'_'+field]))+1
                if mode+'_half_min_'+field in latest:
                    ax.fill_between(t,latest[mode+'_half_min_'+field],latest[mode+'_half_max_'+field],
                                    color=COLORS[own[-1]['particles']],alpha=.12,lw=0)
                ax.set(ylabel=TARGET_LABELS[field],ylim=(-.02,1.02))
                if row_index==0:
                    ax.set_title('Filtering' if mode=='filtered' else 'Full history')
                if row_index==3:
                    ax.set_xlabel('Trial')
        axes[0,0].legend(fontsize=7,ncols=2)
        fig.suptitle(f'S{c["subject"]} | fixed parameter candidate {c["candidate"]["id"]}\nShading: two independent numerical halves of the largest bank; not a credible interval',fontsize=11)
        name=c['id']+'_states';save(fig,output,name);figures.append(name)
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for row,a in zip(own,loaded):
            anc=a['full_unique_ancestors_per_system'];t=np.arange(anc.shape[1])+1
            axes[0].plot(t,np.median(anc,axis=0),color=COLORS[row['particles']],
                         ls={8:'-.',16:'-',32:'--',64:':'}.get(row['systems'],'--'),label=f'N={row["particles"]}, R={row["systems"]}')
        axes[0].set(xlabel='Trial',ylabel='Distinct ancestors (median across systems)',yscale='log')
        axes[0].legend(fontsize=8,ncols=2)
        axes[1].plot(x,[r['full_ess_min'] for r in own],'-o',color='#427bb3',label='Effective system count')
        axes[1].axhline(4,color='#444444',ls=':',lw=1)
        axes[1].set(xticks=x,xticklabels=labels,xlabel='Particles × independent systems',ylabel='Effective system count')
        ax2=axes[1].twinx();ax2.plot(x,[r['full_maximum_weight'] for r in own],'--s',color='#d28a35')
        ax2.set(ylabel='Largest system weight',ylim=(0,1.02));ax2.axhline(.5,color='#d28a35',ls=':',lw=1)
        fig.suptitle(f'S{c["subject"]} | retained early histories and system weights')
        name=c['id']+'_ancestors';save(fig,output,name);figures.append(name)
    accepted={r['case']:r for r in confirmations if r['accepted']}
    registered_jobs=list((source/'jobs').glob('*.json'))
    without_ledger=sum(not (source/'ledger'/f.name).exists() for f in registered_jobs)
    completed=[r for r in costs if r['status']=='complete']
    failures=[r for r in costs if r['status']=='failed']; stopped=[r for r in costs if r['status']=='budget_terminated']
    supervisor=sum(read(f).get('supervisor_cpu_seconds',0) for f in (source/'completions').glob('*.json'))/3600
    cpuh=sum(r['cpu_hours'] for r in costs)+supervisor
    finish_file=source/'precision_completion.json'
    if not finish_file.exists() and not (source/'budget_completion_registration.json').exists():
        finish_file=source/'adaptive_completion.json'
    finish=read(finish_file) if finish_file.exists() else None
    outstanding_label='尚未启动' if finish else '尚无结束记录'
    clock=read(source/'run_clock.json')
    wall=((finish['finished_unix'] if finish else time.time())-clock['started_unix'])/3600
    peak=max([read(f)['peak_workers'] for f in (source/'completions').glob('*.json')] or [0])
    scheduled=[read(f) for f in registered_jobs]
    budget_lines=['| 粒子数 | 新增登记任务 | 完成 | '+outstanding_label+' | 实际CPU小时 |',
                  '|---:|---:|---:|---:|---:|']
    for n in sorted({j['particles'] for j in scheduled}):
        own_jobs=[j for j in scheduled if j['particles']==n]
        own_costs=[r for r in costs if r['particles']==n]
        budget_lines.append(f'| {n} | {len(own_jobs)} | {sum(r["status"]=="complete" for r in own_costs)} | '
                            f'{len(own_jobs)-len(own_costs)} | {sum(r["cpu_hours"] for r in own_costs):.2f} |')
    lines=['# Model 1024 内部状态计算精度报告','',
        f'当前检查{len(cases)}组固定参数，{len(accepted)}组完成原定六项概率摘要的独立数值确认。',
        f'共登记{len(registered_jobs)}项，新增计算完成{len(completed)}项；运行错误{len(failures)}项，预算终止{len(stopped)}项，{outstanding_label}{without_ledger}项。队列计入{cpuh:.2f} CPU小时，墙钟{wall:.2f}小时，峰值{peak}个单线程工作进程。','',
        '本报告检查同一套参数的状态概率是否能重复算出。过滤使用截至本题的行为，完整历史推断还使用后面的行为；',
        '两者的状态时点均为本题选择后、反馈前。参数候选、模型机制、实际行为和口述编码沿用原设置。','',
        *budget_lines,'',
        '表中包括原批、独立确认和预算补充计算；旧512、1024粒子结果只作参照，不计入新增任务。',
        '更高粒子档若无法在剩余时限内完成，会保留登记并标明尚未启动。计算上限为12小时、1536 CPU小时。','',
        '## 如何读结果','',
        '每个粒子保存模型的一条可能学习历史，并按被试的实际选择更新权重。N是一次计算同时追踪的粒子数，',
        'R是使用不同随机种子的独立计算次数。各次使用同一整段行为；图中的两批指随机种子分组。',
        '蓝线比较同一配置下两批独立计算，橙线比较相邻粒子数。',
        '比较的是全部试次的四个正确规则概率，以及整个规则目录的信念和执行分布。平均差的工作门槛为2.5个百分点，',
        '95%分位差表示把比较差值从小到大排列后，95%的值不超过这个数值。它衡量多次计算之间的差异。',
        '95%分位差的门槛为10个百分点；还要求独立计算的有效数量至少4、单次权重不超过0.5。',
        '通过这些检查说明所列概率摘要在本轮检查中接近。逐题最大差、判断精度的完整分布和模拟真值恢复另有检验范围，',
        '因此数值通过本身不等于已经验证真实心理状态。完成次数与数值结果分别列出。','',
        '## 每例进展','',
        '| 案例 | 最新粒子数×次数（实际/计划） | 过滤重复/粒子检查 | 完整历史重复/粒子检查 | 概率摘要的独立确认 | 首题祖先中位数 |',
        '|---|---:|---|---|---|---:|']
    findings=['# 本轮主要发现','',f'本轮新增{len(completed)}项完成计算，使用{cpuh:.2f} CPU小时；{len(accepted)}/{len(cases)}组完成原定六项概率摘要的独立数值确认。','',
        '这里的通过覆盖正确规则的四个概率摘要，以及整个规则目录的信念和执行分布。判断精度及其联合分布另列检查，',
        '它们仍可能有明显分歧。计算重复性与模拟真值恢复也分别评价。','']
    for c in cases:
        own=case_rows[c['id']];r=own[-1]
        flag=lambda key:'接近' if r.get(key,False) else '未通过'
        lines.append(f'| S{c["subject"]} / {c["candidate"]["id"]} | {r["particles"]}×{r["systems"]}/{r["expected_systems"]} | {flag("filtered_numerically_consistent")} | {flag("full_numerically_consistent")} | {"通过" if c["id"] in accepted else "未通过或未完成"} | {r["ancestor_first_median"]:g} |')
        old=next((x for x in own if x['particles']==1024 and x['systems']==16),None)
        if old:
            findings.append(f'- S{c["subject"]}：完整历史的独立重复95%分位差，从1024×16时的{old["full_half_q95"]*100:.1f}个百分点，变为{r["particles"]}×{r["systems"]}时的{r.get("full_half_q95",np.nan)*100:.1f}个百分点；首题不同祖先的中位数从{old["ancestor_first_median"]:g}变为{r["ancestor_first_median"]:g}。')
        confirmation=accepted.get(c['id'])
        if confirmation and confirmation.get('diagnostics',{}).get('full_half_q95',0)>=.09:
            findings.append(f'  S{c["subject"]}的独立确认批重复95%分位差为{confirmation["diagnostics"]["full_half_q95"]*100:.3f}个百分点，接近10个百分点门槛；通过记录保留这一边界。')
    lines+=['','实际数值门槛基于每组可用结果。独立确认使用此前未检查的16个种子；确认批完成且内部稳定、与原批接近，才计入通过。',
        'S129仍属于参数诊断候选；即使其状态数值通过，也保留此前个体参数尚未确立的限制。','',
        '## 相同名义粒子投入','',
        '每行两种配置的粒子数乘独立次数相同。例如2048粒子×16次与4096粒子×8次。实际耗时和重复差异均保留。',
        '各组内部按独立次数分成两半，例如16次组比较8次对8次；差异反映两种完整计算方案。',
        '预算补充计算若完成4096×32，还会与8192×16比较。','',
        '| 案例 | 较少粒子×较多次数 | CPU小时 | 较多粒子×较少次数 | CPU小时 | 两方案完整历史95%分位差 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in equal:
        lines.append(f'| {row["case"]} | {row["low_particles"]}×{row["low_systems"]} | {row["low_cpu_hours"]:.2f} | '
                     f'{row["high_particles"]}×{row["high_systems"]} | {row["high_cpu_hours"]:.2f} | {row["cross_q95"]*100:.1f}个百分点 |')
    if confirmations:
        lines+=['','## 独立确认的实际结果','',
            '每行使用单独的一批16个新种子。原批、确认批各自的重复检查和两批之间的比较共同决定通过与否。',
            '16384粒子档把确认批和原批排在同一队列，两个种子集合独立，判定仍需等两批完成。','',
            '| 案例 | 粒子数 | 原批次数 | 确认完成次数 | 确认批完整历史重复95%分位差 | 确认与原批95%分位差 | 原定摘要确认 |',
            '|---|---:|---:|---:|---:|---:|---|']
        for c in sorted(confirmations,key=lambda r:(r['case'],r['particles'],r['primary_systems'])):
            half=c.get('diagnostics',{}).get('full_half_q95',np.nan)*100
            difference=c.get('full_difference',{}).get('q95',np.nan)*100
            lines.append(f'| {c["case"]} | {c["particles"]} | {c["primary_systems"]} | {c["completed_systems"]}/16 | '
                         f'{half:.2f} | {difference:.2f} | {"通过" if c["accepted"] else "未通过"} |')
        lines+=['','两列差值单位为百分点。确认批合并结果的有效数量与最大单次权重保存在独立确认记录中。']
    quantity_names={'target_present':'正确规则在场', 'target_belief':'正确规则的信念权重',
        'target_high_belief':'正确规则在场且信念至少0.5', 'target_execution':'执行正确规则',
        'belief':'整个规则目录的信念分布', 'execution':'整个规则目录的执行分布'}
    for c in cases:
        latest=case_rows[c['id']][-1]
        lines+=['',f'## S{c["subject"]} / {c["candidate"]["id"]}','',
            '下表列出最新配置的完整历史结果。四个正确规则摘要可能已接近，同时对其他规则仍有分歧；',
            '整个规则目录的信念和执行分布也参与原定检查。表中数值均以百分点表示。','',
            '| 比较的量 | 独立重复平均差 | 独立重复95%分位差 | 相邻粒子数95%分位差 |',
            '|---|---:|---:|---:|']
        for key,title in quantity_names.items():
            lines.append(f'| {title} | {latest.get("full_half_"+key+"_mean",np.nan)*100:.2f} | '
                         f'{latest.get("full_half_"+key+"_q95",np.nan)*100:.2f} | '
                         f'{latest.get("full_comparison_"+key+"_q95",np.nan)*100:.2f} |')
        lines+=['',f'前四分之一试次中，相邻粒子数的完整历史95%分位差为{latest.get("full_early_comparison_q95",np.nan)*100:.2f}个百分点；'
            f'全记录内六个检查量出现的最大差为{latest.get("full_comparison_maximum",np.nan)*100:.2f}个百分点。',
            '所有计算均使用整段行为，这里单独查看前四分之一的推断差异。整段汇总通过时仍保留早期和个别试次的差异。','',
            f'![计算之间的差异](figures/{c["id"]}_agreement.png)','',
            '上排是平均差，下排是95%分位差；虚线门槛沿用原工作标准。每个横坐标都写出粒子数与独立次数。','',
            f'![状态概率随试次变化](figures/{c["id"]}_states.png)','',
            '每条线是所标计算配置合并后的状态概率。阴影表示最大计算组的两批独立结果之间的范围，用来显示计算差异，',
            '不代表心理状态的可信区间。真人试次完整保留；图中的独立批次按随机种子分组。','',
            f'![早期历史保留与计算权重](figures/{c["id"]}_ancestors.png)','',
            '左图表示最终留下的粒子在早期来自多少条不同路径。首题数量很少时，早期状态可能受少数路径影响。',
            '右图蓝线表示多少次独立计算实际贡献了结果，橙线表示权重最大的那一次贡献多少。祖先数量和权重一起帮助定位数值问题。']
    if distribution_rows:
        lines+=['','## 信念和判断精度的分布是否接近','',
            '分布本身的宽度描述状态不确定性；多次计算得到的分布之间的差异描述数值误差。',
            '判断精度是模型中控制规则作答有多确定的量。下面复用原有20个信念区间和25个精度区间。',
            '正确规则的精度分布单独保留“不在场”的概率；',
            '执行规则与精度的联合分布，同时记录执行哪条规则，以及该规则的精度落在哪个区间。',
            '这些是补充分布检查，原定六项摘要的通过记录保持原样。图中的10个百分点线用于读图参照。',
            '每行纵轴按该项数值设置，并在过滤与完整历史两列共用范围。','']
        for c in cases:
            own=[r for r in distribution_rows if r['case']==c['id'] and not r['confirmation']]
            own.sort(key=lambda r:(r['particles'],r['systems']))
            if not own:
                lines+=[f'S{c["subject"]} / {c["candidate"]["id"]}：补充分布尚未汇总。','']
                continue
            fields=[('belief_histogram','Target belief distribution','正确规则的信念'),
                    ('precision_with_absence','Target precision, including absence','正确规则的精度及不在场概率'),
                    ('executed_rule_precision','Executed rule identity + precision','执行规则身份与精度')]
            fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
            x=np.arange(len(own));labels=[f'{r["particles"]}\n×{r["systems"]}' for r in own]
            for i,(field,title,_) in enumerate(fields):
                observed=[r.get(kind+'_'+mode+'_'+field+'_q95',np.nan)*100 for r in own
                          for kind in ('half','comparison') for mode in ('filtered','full')]
                ymax=min(105.,max(12.,1.12*max(v for v in observed if np.isfinite(v))))
                for col,mode in enumerate(('filtered','full')):
                    ax=axes[i,col]
                    ax.plot(x,[r.get('half_'+mode+'_'+field+'_q95',np.nan)*100 for r in own],'-o',color='#427bb3',label='Independent halves')
                    ax.plot(x,[r.get('comparison_'+mode+'_'+field+'_q95',np.nan)*100 for r in own],'--s',color='#d28a35',label='Adjacent particle counts')
                    ax.axhline(10,color='#444444',ls=':',lw=1,label='10 pp reference')
                    ax.set(xticks=x,xticklabels=labels,ylabel='95th percentile TV distance (pp)',ylim=(0,ymax),
                           title=('Filtering' if mode=='filtered' else 'Full history')+' | '+title)
                    if i==2:
                        ax.set_xlabel('Particles × independent systems')
            axes[0,0].legend(fontsize=8)
            fig.suptitle(f'S{c["subject"]} | {c["candidate"]["id"]} | additional state-distribution checks')
            name=c['id']+'_distributions';save(fig,output,name);figures.append(name)
            lines+=[f'### S{c["subject"]} / {c["candidate"]["id"]}','',
                f'![分布之间的差异](figures/{name}.png)','',
                '| 最新配置的分布 | 过滤重复95%分位差 | 完整历史重复95%分位差 | 完整历史相邻粒子数95%分位差 |',
                '|---|---:|---:|---:|']
            latest=own[-1]
            for field,_,title in fields:
                lines.append(f'| {title} | {latest.get("half_filtered_"+field+"_q95",np.nan)*100:.2f} | '
                    f'{latest.get("half_full_"+field+"_q95",np.nan)*100:.2f} | '
                    f'{latest.get("comparison_full_"+field+"_q95",np.nan)*100:.2f} |')
            lines+=['','表中单位均为百分点，差异为总变差距离。它把所有类别或区间的概率差的绝对值加起来，再除以二。','']
            findings.append(f'- S{c["subject"]}的补充分布检查：最新{latest["particles"]}×{latest["systems"]}配置下，'
                f'“执行规则身份与精度”的独立重复95%分位差，过滤为{latest.get("half_filtered_executed_rule_precision_q95",np.nan)*100:.1f}个百分点，'
                f'完整历史为{latest.get("half_full_executed_rule_precision_q95",np.nan)*100:.1f}个百分点。')
        lines+=['[补充分布的全部数值](distribution_diagnostics.csv)','']
    lines+=['','## 可复核文件','',
        '[逐项数值检查](numerical_diagnostics.csv)、[逐任务计算成本](compute_costs.csv)、[独立确认记录](confirmations.json)、',
        '[相同名义投入比较](same_nominal_work.json)。原始粒子、过滤结果和完整历史结果位于本轮产物的states目录；旧研究产物保留。','',
        '口述尚待人工检查。本轮未更改口述编码，也未用口述选择粒子数、参数或内部状态。']
    replay=source/'storage_validation/full_replays_v1/completion.json'
    if replay.exists():
        validation=read(replay)
        cpu=sum(r['cpu_seconds'] for r in validation['cases'])/3600
        adapted=sum('state_worker_entry' in read(source/'jobs'/(r['name']+'.json')) for r in completed)
        lines+=['','## 计算与实现复核','',
            '机器提供64个物理核心、128个逻辑CPU。工作进程各使用一个数值线程；并发数由有分析用途的任务数量和内存共同决定。',
            f'四个案例的保存方式复核共用{cpu:.2f} CPU小时，单列于上述正式队列耗时之外。每例完整记录的42个状态与选择数组逐项相等，',
            '状态压缩文件的SHA256也与原结果一致。节省内存的实现预先分配历史数组，并按16题分块汇总；模型机制和随机抽样顺序保持原样。',
            f'正式队列中使用该实现并完成的任务为{adapted}项。验证完成与正式启用的数量分别记录。',
            '[完整重算复核](../storage_validation/full_replays_v1/completion.json)、[队列调用复核](../storage_validation/integrated_queue_v1/validation.json)、',
            '[执行策略](../execution_policy_v1.json)及[源码和调度修订](../amendments/005_preallocated_storage_and_confirmation.json)保留了启用依据。','']
    if finish:
        original=read(source/'adaptive_completion.json')
        reasons={'remaining_budget_or_memory_insufficient':'下一项任务的预计耗时或内存超出剩余预算',
                 'batch_complete':'本批登记任务完成','registered_actions_exhausted':'登记的计算步骤已全部处理',
                 'wall_budget':'达到墙钟时限','cpu_budget':'达到CPU预算','repeated_worker_failure':'运行错误达到停止数量'}
        lines+=['',f'原精度升级队列停止原因：{reasons.get(original["reason"],original["reason"])}。',
                f'最后一批的状态：{reasons.get(finish["reason"],finish["reason"])}。是否扩展其余候选：{finish["expanded"]}。']
        if finish.get('additional_repetition_diagnostic'):
            lines+=['剩余预算用于增加已完成粒子档的独立重复。这一批评价重复次数的收益，原有独立确认记录保持不变。',
                    '[补充安排登记](../budget_completion_registration.json)、[粒子档选择依据](../decisions/budget_repetition_01.json)。']
    findings+=['','更高粒子数的收益需结合重复差异、相邻粒子数差异和独立确认一起看。完整历史仍未通过的案例，',
              '继续保留逐题解释限制；过滤的稳定范围单独列在完整报告中。口述编码保持原样，等待人工复核。','',
              '[完整报告](report.md)']
    if digest(Path(__file__))!=source_hash:
        raise ValueError('Report source changed during rendering')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'findings.md').write_text('\n'.join(findings)+'\n')
    usage=resource.getrusage(resource.RUSAGE_SELF)
    write(output/'manifest.json',dict(source=str(source.resolve()),protocol_sha256=digest(source/'protocol.json'),
        report_source_sha256=source_hash,completed_unix=time.time(),wall_seconds=time.time()-started,
        report_process_cpu_seconds=usage.ru_utime+usage.ru_stime,figures=figures,
        retained_all_resolutions=True,export='PNG only',
        distribution_summary=str(distribution_file) if distribution_file else None,
        distribution_summary_sha256=digest(distribution_file) if distribution_file else None,
        distribution_summaries={str(p):digest(p) for p in distribution_files},
        figure_sha256={p.name:digest(p) for p in (output/'figures').glob('*.png')},
        sources={str(p):digest(p) for p in (source/'summaries').glob('*/diagnostics.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.source,args.output)
