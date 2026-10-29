"""Write the first oral-content interpretation from completed saved-state scores.

Use --source NEW_ANALYSIS_DIRECTORY. Interpretation examples are development
case studies; all reports and parameter candidates remain in the summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.model_1024_oral_preliminary import METRICS, digest, table


def percent(value: float) -> str:
    return '未评分' if pd.isna(value) else f'{value:.1%}'


def span(values: pd.Series) -> str:
    lo, hi = values.min(), values.max()
    return percent(lo) if pd.isna(lo) or round(lo*100, 1) == round(hi*100, 1) else f'{lo:.1%}–{hi:.1%}'


def run(source: Path) -> None:
    target = source/'report.md'
    if target.exists():
        raise FileExistsError('The interpretation already exists; preserve the completed report')
    manifest = json.loads((source/'manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        if digest(source/name) != expected:
            raise ValueError('Analysis artifact changed: '+name)
    review = pd.read_csv(source/'trial_review.csv')
    summary = pd.read_csv(source/'summary.csv')
    precision = pd.read_csv(source/'particle_sensitivity.csv')
    scores = pd.read_csv(source/'candidate_trial_scores.csv.gz')
    if review.duplicated(['subject', 'iSession', 'iTrial']).any():
        raise ValueError('Repeated original identity')
    if scores.duplicated(['case', 'mode', 'trial']).any():
        raise ValueError('Repeated state score')
    for _, group in scores.groupby(['case', 'mode']):
        np.testing.assert_array_equal(group.trial, np.arange(1, len(group)+1))
        eligible = group.predicate_primary_eligible
        np.testing.assert_allclose(group.loc[eligible,
            ['predicate_supported', 'predicate_conflicting', 'predicate_unexplained']].sum(axis=1), 1., atol=2e-10)
    for metric in METRICS:
        finite = scores[metric].dropna()
        if not np.isfinite(finite).all() or not finite.between(-1e-10, 1+1e-10).all():
            raise ValueError('Invalid content score')
    if scores.loc[~scores.feature_eligible, list(METRICS[:6])].notna().any().any():
        raise ValueError('Missing reason features must remain unscored')
    filtered = summary[summary['mode'].eq('filtered')]
    expected_lengths = {110:448,129:256,205:704,229:1088,306:768,317:1024}
    if review.groupby('subject').size().to_dict()!=expected_lengths or manifest['candidates']!=17:
        raise ValueError('This first interpretation describes the registered six-person cohort')
    def rows_for(sid: int, metric: str, window: str = 'whole') -> pd.DataFrame:
        return filtered[filtered.subject.eq(sid)&filtered.metric.eq(metric)&filtered.window.eq(window)]
    counts, contents, late, differences, numeric = [], [], [], [], []
    scope_names = {'complete': '完整编码', 'parsed_fragment': '解析片段', 'needs_review': '待复核',
                   'inconsistent_constraints': '编码矛盾', 'missing': '缺失'}
    for sid, group in review.groupby('subject'):
        counts.append([f'S{sid}', len(group), int(group.feature_eligible.sum()),
                       int(group.predicate_primary_eligible.sum()),
                       '；'.join(f'{scope_names[k]}{v}' for k, v in group.scope.value_counts().items())])
        value = lambda metric: span(rows_for(sid, metric)['mean'])
        contents.append([f'S{sid}'+('（诊断）' if sid==129 else ''),
            value('reason_feature_coverage'), value('basis_feature_coverage'), value('predicate_supported'),
            value('predicate_conflicting'), value('predicate_unexplained')])
        row = rows_for(sid, 'predicate_supported', 'last64')
        late.append([f'S{sid}', int(row.reports.iloc[0]), span(row['mean']),
                     percent(row.stimulus_choice_mean.iloc[0]),
                     span(rows_for(sid, 'predicate_supported', 'first64')['mean'])])
        feat = rows_for(sid, 'reason_feature_coverage'); pred = rows_for(sid, 'predicate_supported')
        differences.append([f'S{sid}', percent(feat.stimulus_choice_mean.iloc[0]),
            span(feat.gain_over_stimulus_choice), percent(pred.stimulus_choice_mean.iloc[0]),
            span(pred.gain_over_stimulus_choice), int(pred.reference_reports.iloc[0])])
        check = precision[precision.subject.eq(sid)&precision['mode'].eq('filtered')&precision.metric.eq('predicate_supported')]
        numeric.append([f'S{sid}', percent(check.mean_difference.max()), percent(check.q95_difference.max()),
            '尚无本轮比较' if check.empty else ('各候选均通过' if check.whole_execution_distribution_checked.all() else '部分或全部候选受限')])
    count_table = table(['被试','全部原文','可比较部位','主语义分母','旧编码范围'], counts)
    content_table = table(['被试','理由部位被覆盖','规则部位被说到','判断被支持','判断冲突','判断未确定'], contents)
    late_table = table(['被试','最后64题的语义分母','最后64题判断支持','同期刺激参照','最初64题判断支持'], late)
    reference_table = table(['被试','刺激参照：部位覆盖','模型减参照：部位','刺激参照：判断支持',
        '模型减参照：判断','判断共同试次数'], differences)
    precision_table = table(['被试','两档差值逐题平均的最大值','差值95%分位的最大值','原整执行分布检查'], numeric)
    example_rows = []
    for sid, trial in [(110,448), (205,1), (205,656), (205,668), (317,1000)]:
        row = review[review.subject.eq(sid)&review.trial.eq(trial)].iloc[0]
        example_rows.append([f'S{sid} / {trial}', row.text, row.feature_range, row.basis_range,
            row.predicate_range, row.representative_basis])
    examples = table(['被试 / 有效试次','原话','理由部位覆盖','规则部位覆盖','判断支持','P1概率最高的三项局部内容'], example_rows)
    source_link = '../../../src/Bayesian_state/docs/model_architecture/model_1024_oral_preliminary.md'
    text = f'''# 本题选择理由与内部状态：第一版口述对照

本版完成六人{len(review)}条记录、17组既有参数状态的对照：五人13组保留候选，S129四组诊断候选。
指导语已由用户确认：“汇报本题选择这个选项的理由”。原文和既有编码仍待人工复核。
本次后处理耗时{manifest['wall_seconds']:.1f}秒、{manifest['cpu_seconds']:.1f} CPU秒，新增粒子滤波0次。

先看[可筛选的逐条对照表](review.html)：默认显示S205，可按被试、试次及文字筛选。
列定义和下面的例子对应；[方法协议]({source_link})记录具体计算范围。

## 目前能看到什么

S205后段口述与模型局部规则有明确的内容对应，前段的最长、最短理由支持较低。
S306整段的判断支持为73.0%–73.8%，是本组六人中较清楚的正面内容证据。
S110常把多个部位一起说出，模型只解释其中一部分；S317后段的3/4躯干参照还没有完整数值编码。
S129仍使用诊断参数，S229只有一组保留参数。这些结果帮助定位可解释内容与待解决问题。

## 怎样把口述和内部状态接上

实验流程是作答、口述、反馈。因此主对照使用本题选择后、反馈前的状态。
它由真实行为推断，给各执行规则及所选类别局部分量分配概率。
这里的“局部分量”可理解为规则对当前所选类别规定的条件，例如“腿短且头长”。
本版逐个检查这些条件，然后按行为推断的原概率平均。口述没有参与选择参数或改变规则概率。
完整历史重建的结果另存于表格；它还使用后续选择。主对照的参数也已使用完整行为记录估计。
全部案例属于开发资料的回顾性分析，当前口述保留作为外部内容观察。

对每句口述，从两个方向检查部位。假设口述是“腿短、头长、尾巴短”，规则为“腿短且头长”：

| 检查问题 | 本例结果 | 含义 |
|---|---|---|
| 口述的理由部位，有多少被规则涉及？ | 2/3 | 腿、头得到覆盖，尾巴仍待解释 |
| 规则需要的部位，有多少在口述中出现？ | 2/2 | 这条局部规则涉及的部位全部被说到了 |
| 具体判断是否一致？ | 两条支持，一条未确定 | 规则支持腿短和头长，对尾巴长短没有限制 |

如果规则说“头短”，口述说“头长”，部位虽相同，具体判断会记为冲突。
“支持”要求该局部规则区域都满足口述判断；“冲突”表示二者没有正体积的共同区域；
其余情况记为未确定。仅接触同一边界按冲突处理。分数按规则概率加权，再在每份报告内平均判断，
最后在有效报告间平均。因此，表中的判断支持百分比表示内容覆盖程度，不能读作“这么多试次的内部状态已经证实”。

指导语使我们将肯定陈述保留为候选理由。口述可能省略内容，也可能包含事后的解释，人工复核仍有作用。
“最长”同时比较其他部位；“腿最长”会涉及四个部位，具体大小关系还要单独检查。
一条最长语句可能形成多个比较，重复比较会合并，相等形成带容差的单个判断。
本版沿用旧编码的判断单位，单位划分影响权重；逐句等权可在人工确认分句后另行定义。

## 数据范围和需要复核的地方

{count_table}

主语义分母要求整句完整、约束一致、判断可数值表达，并且理由角色未被标记。
部位统计可保留尚未数值解释的文字，但只说明提到了哪些对象；片段的判断分数只解释已解析部分。
S317有11条包含“不是最长／不是最短”的口述，被初版保守词句筛查标记，等待确认其否定判断的表达方式。
表内完整编码总数3230，满足本轮主语义条件的是3031；全部4288条原文均可在逐条表中查看。

“长／短”沿用归一化长度0.5作为分界；相等沿用旧容差；躯干参照保留原符号编码。
这些都是暂定的语义约定。特别是“3/4躯干”需要核对实际刺激长度单位与参照含义，才可接到规则阈值上。
方向、参照或文字尚未明确的项目保留待复核标记，缺失没有填为零分。

## 整段内容结果

{content_table}

区间端点来自该被试已检查参数候选的最小值和最大值；它没有置信区间或参数后验含义。
支持、冲突、未确定使用相同的主语义试次，三项对每个候选相加为100%。各列的区间端点可能来自不同候选。

S110的“规则部位被说到”约97%，但“理由部位被覆盖”约41%–42%。这说明模型所需内容通常在口述里，
口述却还包含许多模型局部规则没有解释的内容。其291条主语义报告的整组判断共同支持仍为0，
部分覆盖结果保留了这些缺口。

S205整段判断支持只有16.3%–16.6%，不能据后段吻合概括为整段都解释得好。
S306的支持较高、冲突较低，同时仍有约25%–26%的判断未确定。
S317整体语义样本仅覆盖381条；特别是后段尚缺可用数值编码，整体平均不能代表那一段。

## 后段为何值得单独看

下面仅将最初和最后各64题作描述性摘要。参数使用完整行为记录估计，64题窗口只用于展示，
没有新增前段拟合或后段预测任务。每个窗口的口述内容和有效分母不同，变化同时反映模型状态与口述内容的变化。

{late_table}

S205最后64题中55条可做主语义比较，判断支持约79.3%，其中仍保留未被局部规则解释的理由。
S229和S306最后64题约99%的判断支持提供了清楚的局部内容对应。
S129后段只有18条进入语义分母，且其参数尚未确立。S317最后64题语义分母为0，
部位一致也不能补上这一层缺失。

## 用原话检查含义

以下例子为结果审查示例，按首题、末题和后段明确两部位表达选取，保留吻合与不完整解释两种情况。
P1为固定展示候选，前三项按原状态概率排序。R编号只在对应分类目录内解释。

{examples}

S110末题说“头短，腿长”，P1给“头短”约99.8%的局部执行质量。因此两个部位只覆盖一个，
口述却完整说到了该局部规则所需的头；“腿长”继续作为待解释理由保留。
S205第656题“腿短，头短”和第668题“腿短，头长”分别与R42的当前类别条件接近。
第1题“脖子最长，尾巴最短”保留为关系内容支持较低的例子。
S317第1000题的3/4躯干原话仍清楚展示，判断栏为未评分。

## 与简单参照比较

简单参照每题只看物理刺激和所选类别，把能够容纳这对刺激与类别的目录规则等权混合。
它不使用学习历史，也忽略知觉噪声。以下模型与参照的差值均使用共同有效试次；
差值以百分比形式显示，数值应读作百分点，例如10.0%表示高10.0个百分点。

{reference_table}

S110与S205的部位覆盖低于这个参照，具体判断支持却较高。二者衡量不同问题：
规则涉及更多部位容易提高部位覆盖，只有方向或关系符合才会提高判断支持。
S306具体判断支持比该参照高约45个百分点；S317只有约2–3个百分点。
这些差值说明相对于此种简单目录匹配的表现，学习过程的独立贡献还需匹配知觉和读出过程的参照检验。
S205第50题、S229第87题、S317第664题没有兼容的物理刺激参照，差值分母排除这三题，模型自身分数保留。
另一个全目录等权参照及整条规则的部位覆盖已存入原始表格。

## 数值稳定性与证据范围

新指标另投影到已保存的较低粒子数结果，下面列各被试候选中差异最大的数值。
逐题平均绝对差与95%分位均使用各自主语义试次；这些是计算差异，单位同样为百分点。

{precision_table}

保留参数的新判断支持指标，两档粒子数的逐题平均差最大约0.69个百分点，95%分位最大约2.47个百分点。
本轮没有对新指标重做独立重复分组检查；整执行分布仍有受限候选。
数值差较小支持当前摘要的初步展示，对完整内部状态的可信度仍沿用已有恢复和参数不确定性边界。

## 下一层怎样检验“理由”

本版检验了理由内容与规则内容是否对应。更直接的理由检验需要固定本题推断状态，
按口述所指判断改变输入，再看当前所选选项的概率如何变化。例如“腿短”作为理由时，
在合法刺激范围内把腿从短改到长，并保持其他条件；同时比较其他选项及联合理由。
变化方式需要事先确定，某项单独改变作用小也可能来自多项理由相互替代。

这一步需要粒子内的知觉输入、规则精度、其他类别的比较和适用条件下的配对状态。
现有高精度汇总档案没有完整保存所需信息。本轮交付到内容对照；后续可复用既有固定状态读出工具，
先使未改变输入的概率精确复现，再进行有明确含义的理由改变。
人工复核优先核对3/4躯干、裸长短的参照、比较链的省略主语及否定判断。

## 可复现文件

- [全部逐条审查](trial_review.csv)：原话、旧编码、未解析内容、候选范围和模型局部内容。
- [所有候选逐题分数](candidate_trial_scores.csv.gz)：过滤与完整历史结果分别保留。
- [分段摘要](summary.csv)与[粒子数敏感性](particle_sensitivity.csv)：有效分母、参照和计算差异。
- [输入与执行记录](manifest.json)、[状态来源](state_sources.json)及[产物核验](report_audit.json)。

本版没有修改原文、旧编码、参数或模型机制。当前所有数值为待人工复核口述上的开发结果。
软件版本：Python {manifest['versions']['python']}，NumPy {manifest['versions']['numpy']}，
pandas {manifest['versions']['pandas']}，SciPy {manifest['versions']['scipy']}。
'''
    target.write_text(text)
    audit = dict(status='passed', trials=len(review), scored_candidate_trial_rows=len(scores),
        original_identity_unique=True, candidate_trial_order_complete=True,
        score_bounds_valid=True, predicate_partition_normalized=True, missing_features_not_zero=True,
        analysis_artifacts_verified=True, independent_new_metric_half_bank_check=False,
        source_sha256=digest(Path(__file__)), manifest_sha256=digest(source/'manifest.json'),
        report_sha256=digest(target), example_policy='Development case review, not independent evaluation')
    (source/'report_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    run(parser.parse_args().source)
