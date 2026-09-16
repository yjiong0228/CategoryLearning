"""Six Fig. 4 continuations: evidence, tests and explicit design schematics.

Python-only preview renderer; 183 x 205 mm, PNG. Quantitative panels reuse
saved observations and simulations. No fabricated held-out scores or effects.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3 import story_preview_common as s


def branch_phases(ax,data):
    ax.axis("off")
    rows=[]
    for j,(cond,task) in enumerate([(3,"Task 2"),(2,"Task 3")]):
        inner=ax.inset_axes([j*.54,.18,.40,.72])
        for sid in data.summary.loc[(data.summary.condition==cond)&(data.summary.n_trials>=128),"iSub"]:
            f=data.trials[data.trials.iSub==sid]
            for branch,categories in enumerate([{1,2},{3,4}]):
                values=[]
                for phase,g in [("first",f.iloc[:64]),("last",f.iloc[-64:])]:
                    selected=g[g.category.isin(categories)]
                    value=selected.correct.mean()
                    values.append(value)
                    rows.append(dict(condition=cond,iSub=sid,branch=branch+1,phase=phase,accuracy=value,n_trials=len(selected)))
                inner.plot([branch-.12,branch+.12],values,color=[s.COLORS[1],s.GLOBAL][branch],alpha=.22,lw=.6)
        part=pd.DataFrame(rows)
        part=part[part.condition==cond]
        for branch in [0,1]:
            means=part[part.branch==branch+1].groupby("phase").accuracy.mean().reindex(["first","last"])
            inner.plot([branch-.12,branch+.12],means,"o-",color=[s.COLORS[1],s.GLOBAL][branch],ms=3,lw=1.5)
        inner.set(ylim=(0,1.02),xticks=[-.12,.12,.88,1.12],xticklabels=["首","末","首","末"],title=task)
        inner.title.set_fontsize(7)
        inner.set_xlabel("分支 1       分支 2",fontsize=5.8)
        inner.tick_params(labelsize=5.5)
        if j==0:inner.set_ylabel("完整正确率",fontsize=6)
    data.derived["branch_phase_accuracy"]=pd.DataFrame(rows)


def matched_performance(ax,data):
    ax.axis("off")
    for j,(cond,task) in enumerate(s.TASKS):
        inner=ax.inset_axes([j*.35,.19,.28,.71])
        f=data.summary[data.summary.condition==cond]
        valid=f[["last64_accuracy","last_feature_count"]].notna().all(axis=1)
        g=f.loc[valid]
        inner.scatter(g.last64_accuracy,g.last_feature_count,s=13,c=s.COLORS[cond],alpha=.65,edgecolors="white",lw=.3)
        inner.set(title=f"{task} · n={len(g)}",xlabel="末64 正确率",xlim=(0,1.03),ylim=(.7,4.1))
        inner.title.set_fontsize(7);inner.xaxis.label.set_fontsize(6);inner.tick_params(labelsize=5.5)
        if j==0:inner.set_ylabel("末64 报告特征数",fontsize=6)
    ax.text(.01,-.075,"相近正确率下的报告差异只是观察起点，尚未识别个体的潜在困难机制。",fontsize=5.8,color=s.MUTED,transform=ax.transAxes)


def build_fig4(data: s.PreviewData,output: Path) -> list[Path]:
    files=[]
    fig=s.figure(0,4,"不同理解会留下哪些行为差异","先看已有错误结构，再用诊断性新刺激区分竞争规则")
    s.phase_confusions(s.panel(fig,data,s.HERO,"a","错误混淆怎样随学习变化？","真实数据 · 被试内归一化后等权汇总 · 95 人","fig1_v10 phase-specific confusion matrices"),data)
    s.geometry(s.panel(fig,data,s.LEFT,"b","哪些新刺激最能区分规则？","诊断性探针示意 · 未测迁移结果","analytical F1/F2 disagreement regions"),"probes")
    ax=s.panel(fig,data,s.RIGHT,"c","怎样把内容差异接到泛化？","检验设计 · 不包含虚构预测增益","held-out probe comparison design")
    ax.axis("off")
    s.box(ax,.12,.75,.76,.17,"冻结两种竞争规则",s.COLORS[1],7)
    s.box(ax,.12,.42,.76,.17,"呈现预测分歧最大的刺激",s.GLOBAL,6.5)
    s.box(ax,.12,.09,.76,.17,"比较真实选择与各自预测",s.LOCAL,6.5)
    s.arrow(ax,(.5,.72),(.5,.62));s.arrow(ax,(.5,.39),(.5,.29))
    s.footer(fig,"a 展示错误分布，尚不能直接归因于某条规则。b/c 为未来可实施的区分方案；没有用模拟迁移数据补出理想效果。")
    files.append(s.save(fig,output,"story_01_Fig4_representation"))

    fig=s.figure(1,4,"改变与保持何时可能有价值","先区分模型控制关系、局部行为描述和真正的机制对照")
    ax=s.panel(fig,data,s.HERO,"a","模型中的目标规则与搜索怎样相伴？","已有拟合 · S129 · 内部量描述，不是外部验证","fig3_v4 metrics and fig2_v15 target_alignment")
    c=data.cases[129]
    ax.plot(c["target"].trial,c["target"].model_target_mass,color=s.COLORS[1],label="目标规则信念")
    ax.plot(c["metrics"].trial,c["metrics"].predictive_strategy_global_explore,color=s.GLOBAL,label="全局搜索概率")
    ax.set(xlabel="试次",ylabel="质量 / 概率",ylim=(0,1.03),xlim=(1,256));ax.legend(ncol=2,fontsize=6)
    ax=s.panel(fig,data,s.LEFT,"b","两个报告片段的后续行为","真实行为 · 无匹配对照 · 两例示意范围","fig2_v15 observed rolling32 around report excerpts")
    for sid,center,color in [(129,138,s.COLORS[1]),(229,992,s.GLOBAL)]:
        f=data.cases[sid]["case"]
        g=f[f.trial.between(center-16,center+16)]
        ax.plot(g.trial-center,g.observed_accuracy_rolling32,color=color,label=f"S{sid}")
    ax.axvline(0,color=s.MUTED,ls="--",lw=.7)
    ax.set(xlabel="相对首次新报告的试次",ylabel="32 试次正确率",ylim=(0,1.02));ax.legend(fontsize=6)
    s.mechanism_design(s.panel(fig,data,s.RIGHT,"c","适时调整是否比固定搜索更好？","机制对照设计 · 尚无公平比较结果","matched average search-rate control design"))
    s.footer(fig,"a 的关系部分来自模型定义；b 不能证明修订带来改善。c 才是需要补齐的辨别检验，也需独立口述事件支持。")
    files.append(s.save(fig,output,"story_02_Fig4_timing"))

    fig=s.figure(2,4,"把成绩差异拆成可检验的过程问题","用现有群体描述定位差异，再设计发现、保持与退出的区分")
    matched_performance(s.panel(fig,data,s.HERO,"a","相近成绩下，报告内容是否仍不同？","真实数据 · 95 人；每点一人，各任务分开","fig1_v10 last64_accuracy and last_feature_count"),data)
    s.subject_scatter(s.panel(fig,data,s.LEFT,"b","更常换特征就一定学得更好吗？","真实数据 · Task 1 全部 32 人","fig1_v10 report_change_rate and last64_accuracy"),data,
                      "report_change_rate","last64_accuracy","可比报告中的特征变化比例","末 64 试次正确率",condition=1)
    s.learner_paths(s.panel(fig,data,s.RIGHT,"c","下一步区分哪种困难？","候选路径示意 · 未做人群分类","discovery, stabilization and return hypotheses"))
    s.footer(fig,"特征数量和集合变化均不等于完整策略。群体模型拟合、口述核对和可靠性分析完成后，才能把这些差异解释为具体学习困难。")
    files.append(s.save(fig,output,"story_03_Fig4_individual"))

    fig=s.figure(3,4,"任务差异集中在哪些学习环节","从相关特征覆盖和分支表现寻找具体问题，保留跨任务比较的限制")
    s.paired_phases(s.panel(fig,data,s.HERO,"a","报告是否覆盖任务所需的特征？","真实数据 · 首末不重叠阶段 · 95 人","fig1_v10 oral_phase_summary path_coverage"),data,
                    metric="path_coverage",ylabel="任务路径所需特征的覆盖比例")
    ax=s.panel(fig,data,s.LEFT,"b","各任务的记录长度与表现","真实数据 · 全部 96 人","fig1_v10 subject_summary")
    s.subject_scatter(ax,data,"n_trials","last64_accuracy","记录试次数","末 64 试次正确率")
    branch_phases(s.panel(fig,data,s.RIGHT,"c","两个分支是否同步改善？","真实数据 · Task 2/3，各 32 人","fig1_v10 first/last64 branch correctness"),data)
    s.footer(fig,"特征覆盖不是规则正确性；分支按真实类别 1/2 与 3/4 定义。Task 2 的部分反馈不计为完整正确；任务操纵组合不能分离单因素因果效应。")
    files.append(s.save(fig,output,"story_04_Fig4_task"))

    fig=s.figure(4,4,"预测价值需要哪些新增证据","把当前可用的状态证据与尚待执行的后缀、迁移评价放在一起")
    ax=s.panel(fig,data,s.HERO,"a","已有的状态证据能否成为预测起点？","已有拟合 + 口述 · 两例均不是留出结果","fig2_v15 target_alignment for S129 and S229")
    ax.axis("off")
    for j,sid in enumerate([129,229]):
        inner=ax.inset_axes([j*.54,.21,.43,.70])
        s.case_curve(inner,data,sid,target=True)
        inner.set_title(f"S{sid}",fontsize=7)
        inner.set_ylabel("目标规则质量" if j==0 else "",fontsize=6)
        inner.legend(fontsize=5.5,ncol=1,loc="upper left")
        inner.tick_params(labelsize=5.5)
    ax=s.panel(fig,data,s.LEFT,"b","后缀评价要比较什么？","评价设计 · 评分尚未生成","held-out score and calibration plan")
    ax.axis("off")
    entries=[("相同测试试次","行为基线 vs 基线 + 状态"),("选择概率评分","哪种预测给真实选择更高概率？"),("错误类型与校准","改善是否超出总体正确率？")]
    for i,(heading,text) in enumerate(entries):
        s.box(ax,.03,.73-i*.31,.94,.22,heading+"\n"+text,[s.COLORS[1],s.LOCAL,s.GLOBAL][i],6)
    s.geometry(s.panel(fig,data,s.RIGHT,"c","泛化用什么刺激来检验？","探针设计示意 · 未假定已有迁移结果","analytical disagreement probe map"),"probes")
    s.footer(fig,"真正预测需只用前段拟合并冻结流程。此版故意不填虚构的后缀分数、提升幅度或迁移散点；其视觉位置与比较逻辑已明确。")
    files.append(s.save(fig,output,"story_05_Fig4_prediction"))

    fig=s.figure(5,4,"模型能否自主产生真实学习的变化","直接查看已有 500 条模拟，包括与真实个案不一致的部分")
    s.autonomous_curves(s.panel(fig,data,s.HERO,"a","固定参数下的学习分布覆盖真实个案吗？","历史自主模拟 · 全部 500 条 + 真实 S129","saved autonomous_trajectory_arrays for S129 PMH"),data)
    s.autonomous_onsets(s.panel(fig,data,s.LEFT,"b","持续达到标准的时间怎样分布？","历史定义 · 达标 491；未达标 9","saved autonomous_trajectory_summary and manifest"),data)
    ax=s.panel(fig,data,s.RIGHT,"c","末期表现是否相符？","历史自主模拟 · 全部 500 条 + 一名被试","saved final_block_accuracy and observed feedback")
    ax.hist(data.auto_summary.final_block_accuracy,bins=np.linspace(0,1,17),color=s.LOCAL,edgecolor="white",lw=.4,alpha=.8)
    observed=float(data.auto["observed_feedback"][-64:].mean())
    ax.axvline(observed,color=s.INK,ls="--",lw=1.3,label=f"真实 S129: {observed:.2f}")
    ax.set(xlabel="末 64 试次正确率",ylabel="模拟次数",xlim=(0,1));ax.legend(fontsize=5.8)
    s.footer(fig,"阴影为逐试次分位范围，不是参数置信区间。达标沿用历史 16 试次窗口、0.8 阈值、连续 8 窗口定义；仅一名被试的固定参数检查，尚缺公平机制对照。")
    files.append(s.save(fig,output,"story_06_Fig4_generation"))
    return files
