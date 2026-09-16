"""Build six alternative Fig. 3/4 pairs from saved evidence and design schematics.

Run from the repository root. New version directories are mandatory. Python,
183 x 205 mm, 300 dpi PNG. No fitting, recovery, or new autonomous simulation.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3 import story_preview_common as s


def build_fig3(data: s.PreviewData, output: Path) -> list[Path]:
    files=[]
    fig=s.figure(0,3,"规则内容怎样重新组织","先看真实口述，再看群体特征范围，最后明确分类功能的含义")
    s.report_cards(s.panel(fig,data,s.HERO,"a","两段真实报告改变了什么？","真实口述 · S129 / S229 · 各一个例子","fig3_v4 same-choice reports"),data)
    s.paired_phases(s.panel(fig,data,s.LEFT,"b","报告涉及的维度是否变化？","真实数据 · 首末不重叠阶段 · 95 人","fig1_v10 oral_phase_summary"),data)
    s.geometry(s.panel(fig,data,s.RIGHT,"c","换规则会怎样改变分类？","规则几何示意 · 非拟合结果","analytical F1/F2 threshold rules"))
    s.footer(fig,"b 是特征提及的描述，不是完整规则语义或注意权重。c 展示两条规则的定义，不声称这些边界已从数据恢复。")
    files.append(s.save(fig,output,"story_01_Fig3_representation"))

    fig=s.figure(1,3,"反馈经历与规则修订","从真实片段出发，查看可用的群体描述与外部行为线索")
    s.excerpts(s.panel(fig,data,s.HERO,"a","报告改变前后，搜索怎样变化？","真实反馈 + 已有拟合 · 两例，16 次 PF 平均","fig3_v4 excerpts"),data)
    s.candidate_feedback(s.panel(fig,data,s.LEFT,"b","哪些反馈后更常换特征？","真实数据 · 特征集合变化，非语义事件","fig1_v10 trial_source and previous_feedback"),data)
    s.rt_feedback(s.panel(fig,data,s.RIGHT,"c","反馈后作答速度怎样变化？","真实数据 · 被试内标准化后汇总","fig1_v10 nonoral_feedback_rt"),data)
    s.footer(fig,"b/c 均为未匹配的描述性关系。完整语义事件、匹配对照和修订后行为检验仍缺；不能把反馈—搜索的内置关系当成机制验证。")
    files.append(s.save(fig,output,"story_02_Fig3_timing"))

    fig=s.figure(2,3,"相近成绩可以怎样到达","先保留完整轨迹，再提出可由口述和模型区分的路径解释")
    s.original_examples(s.panel(fig,data,s.HERO,"a","同一任务中的学习路径有多不同？","真实数据 · Task 1 全部 32 人；沿用已选三例","fig1_v10 trial_source and representatives"),data)
    s.learner_paths(s.panel(fig,data,s.LEFT,"b","哪些内部过程值得区分？","路径示意 · 不对应真实被试或时间估计","qualitative candidate paths"))
    ax=s.panel(fig,data,s.RIGHT,"c","首末表现如何对应？","真实数据 · 全部 96 人","fig1_v10 subject_summary")
    s.subject_scatter(ax,data,"first64_accuracy","last64_accuracy","首 64 试次正确率","末 64 试次正确率")
    ax.plot([0,1],[0,1],ls=":",color=s.MUTED,lw=.7);ax.set(xlim=(0,1.02),ylim=(0,1.02))
    s.footer(fig,"灰线保留 Task 1 全部被试；高亮沿用 Fig. 1 的固定示例，不是学习者分类。路径示意只表达待检验解释。")
    files.append(s.save(fig,output,"story_03_Fig3_individual"))

    fig=s.figure(3,3,"任务中的规则内容与学习顺序","在共同特征坐标中比较描述，并区分任务结构与行为表现")
    s.task_feature_heat(s.panel(fig,data,s.HERO,"a","不同任务更常提及哪些特征？","真实数据 · 首末 64 试次；被试等权","fig1_v10 aligned feature mentions"),data)
    ax=s.panel(fig,data,s.LEFT,"b","四分类任务的客观结构","任务结构示意 · 非被试已知规则","fixed F1 branch with F2/F3 subdivision")
    ax.axis("off")
    s.box(ax,.29,.73,.42,.17,"F1 分支",s.INK,7)
    s.box(ax,.02,.38,.40,.17,"F2 区分",s.COLORS[1],7)
    s.box(ax,.58,.38,.40,.17,"F3 区分",s.LOCAL,7)
    s.arrow(ax,(.4,.70),(.22,.57));s.arrow(ax,(.6,.70),(.78,.57))
    ax.text(.22,.20,"类别 1 / 2",ha="center",fontsize=6.5,transform=ax.transAxes)
    ax.text(.78,.20,"类别 3 / 4",ha="center",fontsize=6.5,transform=ax.transAxes)
    ax.text(.5,.03,"层级提示与部分反馈另行区分",ha="center",fontsize=5.5,color=s.MUTED,transform=ax.transAxes)
    s.subject_scatter(s.panel(fig,data,s.RIGHT,"c","训练记录与最终表现","真实数据 · 全部 96 人","fig1_v10 subject_summary"),data,
                      "n_trials","last64_accuracy","记录试次数","末 64 试次正确率")
    s.footer(fig,"F1–F4 按被试特征分配对齐。首末提及比较保留有文本的报告；Task 2 对应 condition 3，部分正确不计为完整正确。")
    files.append(s.save(fig,output,"story_04_Fig3_task"))

    fig=s.figure(4,3,"把可测量状态接到未来预测","已有曲线展示信息来源；时间切分图说明下一步怎样检验")
    s.case_curve(s.panel(fig,data,s.HERO,"a","选择轨迹能否被当前模型描述？","已有拟合 · S129 · 全序列参数","fig2_v15 S129 trial_source"),data)
    s.case_curve(s.panel(fig,data,s.LEFT,"b","规则状态还有口述线索吗？","已有拟合 + 口述 · 描述性一致性","fig2_v15 S129 target_alignment"),data,target=True)
    s.design_split(s.panel(fig,data,s.RIGHT,"c","怎样检验真正的未来预测？","评价设计 · 尚未执行前段重新拟合","temporal prediction design"))
    s.footer(fig,"a/b 不是留出预测。参数仅在前段选择中估计、流程冻结后，才能在后段比较行为基线与过程模型。")
    files.append(s.save(fig,output,"story_05_Fig3_prediction"))

    fig=s.figure(5,3,"模型需要产生哪些真实现象","先完整呈现行为差异，再明确需要被生成机制解释的对象")
    ax=s.panel(fig,data,s.HERO,"a","三个任务的完整学习记录","真实数据 · 96 人；每个子图使用真实试次轴","fig1_v10 trial_source")
    ax.axis("off")
    for j,(cond,task) in enumerate(s.TASKS):
        inner=ax.inset_axes([j*.35,.18,.28,.72])
        s.heatmap_subjects(inner,data,cond)
        inner.set_title(task,fontsize=7)
        inner.set_xlabel("真实试次",fontsize=6)
        inner.set_ylabel("被试" if j==0 else "",fontsize=6)
        inner.tick_params(labelsize=5)
    ax.text(.02,-.065,"每图均为 32 人；颜色表示 32 试次正确率（0–1）；灰色不是低正确率。",fontsize=5.8,color=s.MUTED,transform=ax.transAxes)
    ax=s.panel(fig,data,s.LEFT,"b","轨迹变化对观察窗口敏感吗？","真实数据 · 全部 96 人 · 已有描述指标","fig1_v10 max_gain_w16/w32/w64")
    for cond,task in s.TASKS:
        f=data.summary[data.summary.condition==cond]
        vals=f[["max_gain_w16","max_gain_w32","max_gain_w64"]].to_numpy()
        ax.plot([0,1,2],vals.T,color=s.COLORS[cond],alpha=.13,lw=.5)
        ax.plot([0,1,2],np.nanmedian(vals,axis=0),"o-",color=s.COLORS[cond],ms=3,label=task)
    ax.set(xticks=[0,1,2],xticklabels=[16,32,64],xlabel="前后相邻窗口长度",ylabel="最大正确率增益")
    ax.legend(fontsize=5.5)
    s.learner_paths(s.panel(fig,data,s.RIGHT,"c","哪些路径特征要进入生成检验？","路径示意 · 不是已识别的心理类型","qualitative phenotypes"))
    s.footer(fig,"最大增益是已有描述量，受窗口和记录长度影响，不等于可靠的认知突变。自主模拟应比较完整分布，而不只挑选相似个案。")
    files.append(s.save(fig,output,"story_06_Fig3_generation"))
    return files


def gallery(fig3_files,fig4_files,output3: Path,output4: Path) -> None:
    """A local browsing page; every embedded visual remains a Python PNG."""
    sections=[]
    overview, axes=plt.subplots(6,2,figsize=(11,34),facecolor="white")
    for i,(left,right) in enumerate(zip(fig3_files,fig4_files)):
        pair, pair_axes=plt.subplots(1,2,figsize=(14.4,8.2),facecolor="white")
        for column,path in enumerate([left,right]):
            img=plt.imread(path)
            pair_axes[column].imshow(img);pair_axes[column].axis("off")
            axes[i,column].imshow(img);axes[i,column].axis("off")
        pair.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=.015)
        pair_name=f"story_{i+1:02d}_pair.png"
        pair.savefig(output3/pair_name,dpi=160,facecolor="white")
        plt.close(pair)
        right_rel=Path("../../fig4")/output4.name/right.name
        sections.append(f'<section id="story-{i+1}"><h2>{i+1}. {html.escape(s.STORIES[i][1])}</h2><p>{html.escape(s.STORIES[i][2])}</p><div class="pair"><a href="{left.name}"><img src="{left.name}" alt="方案{i+1} Fig.3"></a><a href="{right_rel.as_posix()}"><img src="{right_rel.as_posix()}" alt="方案{i+1} Fig.4"></a></div><p><a href="{pair_name}">打开并排 PNG</a></p></section>')
    overview.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=.01,hspace=.025)
    overview.savefig(output3/"all_six_pairs.png",dpi=145,facecolor="white")
    plt.close(overview)
    nav=" · ".join(f'<a href="#story-{i+1}">{i+1}. {html.escape(item[1])}</a>' for i,item in enumerate(s.STORIES))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>六种文章主线：Fig. 3 / 4</title><style>body{margin:0;background:#edf1f4;color:#253541;font:16px/1.7 system-ui,sans-serif}main{max-width:1450px;margin:30px auto;padding:0 24px}h1{font-size:30px}nav{line-height:2.2}a{color:#326d87}section{background:white;margin:30px 0;padding:24px;border-radius:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair img{width:100%;display:block}p{color:#61707b}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><main><h1>六种文章主线 · Fig. 3 / Fig. 4</h1><p>每组是一种候选叙事。点击图片可看原尺寸 PNG。面板状态区分真实数据、已有拟合、历史自主模拟与设计示意；示意图不包含虚构的实证效应。</p><nav>'+nav+'</nav>'+''.join(sections)+'<section><h2>怎样比较</h2><p>先看哪组科学问题最清楚，再看两张图能否连续回答这个问题。当前图展示可用证据与后续设计，不代表六套正式分析已经完成。</p><p>完整说明见 <a href="README.md">README</a>，逐面板来源和计算范围见 <a href="manifest.json">manifest</a>。</p></section></main></html>'
    (output3/"index.html").write_text(page)


def build(version: str) -> tuple[Path,Path]:
    if not version or Path(version).name != version or version in {".",".."}:
        raise ValueError("version must be a single new directory name")
    output3=Path("CategoryLearning_codes/figures/outputs/fig3")/version
    output4=Path("CategoryLearning_codes/figures/outputs/fig4")/version
    if output3.exists() or output4.exists():
        raise FileExistsError("Existing figure/research outputs are never overwritten; choose a new --version.")
    s.configure()
    data=s.PreviewData()
    output3.mkdir(parents=True)
    output4.mkdir(parents=True)
    from CategoryLearning_codes.figures.fig4.build_story_previews import build_fig4
    files3=build_fig3(data,output3)
    files4=build_fig4(data,output4)
    gallery(files3,files4,output3,output4)
    data.export(output3)
    (output4/"manifest.json").write_text(json.dumps({"figures":[p.name for p in files4],"shared_manifest":f"../../fig3/{version}/manifest.json","status":"review_previews"},ensure_ascii=False,indent=2)+"\n")
    source=output3/"source_code"
    source.mkdir()
    for path in [Path(__file__),Path(s.__file__),Path("CategoryLearning_codes/figures/fig4/build_story_previews.py")]:
        shutil.copy2(path,source/f"{path.parent.name}_{path.name}")
    (output3/"README.md").write_text(readme(version))
    (output4/"README.md").write_text(f"# 六条主线的 Fig. 4 预览\n\n完整图集与证据说明见 [配对图集](../../fig3/{version}/index.html)。本目录保存六张 Fig. 4 PNG，不代表分析全部完成。\n")
    print(f"Built 12 figure PNGs, 6 pair PNGs and one overview. Gallery: {output3/'index.html'}")
    return output3,output4


def readme(version: str) -> str:
    return f'''# 六种文章主线的 Fig. 3 / 4 预览

[打开图集](index.html) · [全部配对总览](all_six_pairs.png)

## 交付范围

6 组、12 张单图，另附 6 张并排图。Python / matplotlib，单图 183 × 205 mm、300 dpi、PNG。图集用于比较叙事与构图，不是投稿终稿。

## 状态与证据

- 真实数据：复用 Fig1 v10 的 96 人、62,720 试次及保存的口述/反应时汇总，不删除模糊刺激。首末配对要求至少 128 试次，共 95 人；S105 仍保留在行为图中。
- 已有拟合：复用 S129/S229 的全序列参数、16 次在线 PF 平均，不作为留出预测。所有图仍使用原口述编码尺度 0.05。
- 特征集合变化：沿用同类别、同 session、有有效报告且报告间隔符合原配置的候选指标，不能等同于经过语义核验的规则修订。
- 历史自主模拟：复用 S129 PMH 的全部 500 条已保存模拟，无新增拟合或模拟。阴影是逐试次分位区间，不是参数置信区间，也不是整条轨迹同时覆盖区间。
- 设计/几何/路径示意：仅用规则定义和定性流程，不虚构群体散点、显著性或预测增益。

## 计算与展示

- 特征首末配对沿用保存的被试期内均值，粗线是被试等权均值。
- 混淆矩阵先在每位被试的首/末 64 试次内按真实类别归一化，再等权汇总；缺少某真实类别时，该行不参与相应均值。每一类别的有效人数在导出表中记录。
- 特征提及热图分母为有文本报告；未识别特征的有文本报告仍记为未提及，不充当完整规则语义。
- 候选变化与前序反馈、反馈与反应时均是描述性汇总，未匹配刺激、阶段和完整反馈历史，无因果结论。
- 自主达标定义沿用历史 manifest：16 试次窗口、正确率达到其 0.8 阈值并连续满足 8 个移动窗口；491 条达到，9 条未达到，全部保留。此定义与 Fig1 的 64 试次行为标准不同，不混用。
- 方案 5 的真正后缀评价、迁移与方案 2/6 的公平机制比较尚待开展；图中仅展示设计。

## 可追溯性

逐面板角色、来源哈希、版本、分母与排除规则见 `manifest.json`；新增汇总见 `source_data/`，绘图脚本快照见 `source_code/`。没有更新论文确认图或替换旧 Fig3 v4。

复现（使用另一个新版本名）：

```bash
MPLCONFIGDIR=/tmp/categorylearning_story_mpl python -m CategoryLearning_codes.figures.fig3.build_story_previews --version story_previews_YYYYMMDD_v2
```
'''


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version",required=True,help="New shared version directory name under outputs/fig3 and outputs/fig4")
    args=parser.parse_args()
    build(args.version)
