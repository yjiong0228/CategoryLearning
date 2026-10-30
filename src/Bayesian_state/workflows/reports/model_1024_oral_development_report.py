"""Build a reviewable Chinese report from the bounded oral development study.

Use --source STUDY --figures FIGURES --output NEW_REPORT_DIRECTORY. Tables
pair comparisons on identical trial identities; raw transcripts are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/model1024_oral_development_mpl')
import numpy as np
import pandas as pd

from ...evaluation.oral.partial_reason_alignment import component_geometry, predicate_coverage
from ..analysis.model_0923_oral_constraint_audit import catalogue


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers: list[str], rows: list[list]) -> str:
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|'] +
        ['| '+' | '.join(str(x).replace('|', '/') for x in row)+' |' for row in rows])


def relation_audit(source: Path, atoms: pd.DataFrame) -> pd.DataFrame:
    """All S306 head/neck directional atoms, across every saved candidate."""
    def relation(token):
        if not token.startswith('facet:'):
            return False
        r = np.array(json.loads(token[6:]))
        return bool(len(r) == 1 and set(np.flatnonzero(abs(r[0, :4]) > 1e-9)) == {1, 2}
                    and abs(r[0, :4].sum()) < 1e-9 and r[0, -1] == 0)
    frame = atoms[atoms.subject.eq(306) & atoms.atom.map(relation)].copy()
    with np.load(source/'diagnosis_v2/S306_diagnostic_arrays.npz') as z:
        x, ref, choices = z['stimulus'], z['stimulus_reference_mass'], z['choice']
    frame['leg_short'] = [x[t-1, 0] < .5 for t in frame.trial]
    metadata = json.loads((source/'diagnosis_v2/S306_metadata.json').read_text())
    space, cache, values = catalogue(4), {}, []
    for row in frame.itertuples():
        key = (int(choices[row.trial-1]), row.atom)
        if key not in cache:
            score = np.zeros(ref.shape[1:])
            for h, m, A, b in component_geometry(space, key[0])[0]:
                score[h, m] = predicate_coverage((row.atom,), A, b)[0]
            cache[key] = score
        values.append(cache[key])
    values = np.asarray(values)
    frame['stimulus_reference'] = [np.sum(ref[t-1]*v) for t,v in zip(frame.trial, values)]
    for state in metadata['selected']:
        with np.load(state['file']) as z:
            mass = z['filtered_component']
        frame[f'P{state["number"]}'] = [np.sum(mass[t-1]*v) for t,v in zip(frame.trial, values)]
    return frame


def numerical_audit(source: Path) -> pd.DataFrame:
    """Evaluate the registered finite-bank checks; preserve every block."""
    p = json.loads((source/'anchor_v2/registration.json').read_text())['policy']['numerical']
    numeric = pd.read_csv(source/'anchor_v2/numerical_checks.csv')
    score = pd.read_csv(source/'anchor_v2/summary.csv')
    score = score[score.split.eq('assessment') & score.window.str.startswith('block64_')]
    rows = []
    for sid in sorted(numeric.subject.unique()):
        for block in sorted(numeric[numeric.subject.eq(sid)].block64.unique()):
            selected = numeric[numeric.subject.eq(sid) & numeric.block64.eq(block) & numeric.bank.eq('all')].set_index('variant')
            before, after = selected.loc['behavior'], selected.loc['oral_temperature_2']
            half = score[score.subject.eq(sid) & score.window.eq(f'block64_{block}')].pivot(
                index='bank', columns='variant', values='subset_log_probability')
            gains = half.oral_temperature_2-half.behavior
            difference = abs(gains.first_half-gains.second_half)
            ratio = after.terminal_ess/before.terminal_ess
            passed = (min(before.system_ess, after.system_ess) >= p['minimum_system_ess'] and
                max(before.maximum_system_weight, after.maximum_system_weight) <= p['maximum_system_weight'] and
                min(before.anchor_ancestral_ess, after.anchor_ancestral_ess) >= p['minimum_anchor_ancestral_ess'] and
                ratio >= p['minimum_importance_ess_ratio'] and difference <= p['half_log_gain_difference'])
            rows.append(dict(subject=int(sid), block64=int(block), system_ess=float(after.system_ess),
                maximum_system_weight=float(after.maximum_system_weight), anchor_ancestral_ess=float(after.anchor_ancestral_ess),
                terminal_ess_ratio=float(ratio), half_log_gain_difference=float(difference),
                assessment_gain=float(gains['all']), passed_registered_working_checks=bool(passed)))
    return pd.DataFrame(rows)


def review_html(frame: pd.DataFrame, output: Path) -> None:
    columns = [('subject','被试'), ('trial','试次'), ('split','本轮用途'), ('text','本题选择理由原文'),
        ('choice','选择'), ('feedback','反馈'), ('scope','旧编码范围'), ('stimulus_description','刺激数值'),
        ('decoded_constraints','旧编码的数值解释'), ('P1_rule_content','行为推断的主要局部规则'),
        ('P1_support','原语义支持'), ('atomic_ceiling','目录表达上限'),
        ('alternative_specific_support','其他规则特有内容支持'), ('readout_gain','相对表达参照的得分增益'),
        ('physical_all_true','按当前编码是否全部符合刺激'), ('unparsed','未解析文字')]
    rows = []
    for row in frame.to_dict('records'):
        cells = []
        for key,_label in columns:
            value = row.get(key, '')
            if isinstance(value, float):
                value = '' if not np.isfinite(value) else f'{value:.4f}'
            cells.append('<td>'+html.escape(str(value) if value is not None else '')+'</td>')
        rows.append(f'<tr data-subject="{row["subject"]}" data-trial="{row["trial"]}">'+''.join(cells)+'</tr>')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述理由开发研究：逐题复核</title>
<style>body{font:14px/1.5 sans-serif;margin:20px;color:#26313a}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:7px;vertical-align:top;min-width:65px}th{position:sticky;top:0;background:#e6edf3}td:nth-child(4),td:nth-child(9),td:nth-child(10){min-width:230px}tr:nth-child(even){background:#f7f8fa}input,select{font:inherit;padding:5px}</style>
<h1>全部4288条口述：原文、编码和模型内容</h1><p>原文待人工复核。P1为固定展示候选，状态由完整行为拟合参数并逐题过滤得到。
“其他规则特有内容支持”表示由非正确任务规则支持、正确任务规则的该选项分区未要求的内容；它不等同于确认真实错误信念。
相对表达参照的增益是本轮读出分数，口述辅助结果另见报告。空白表示未评分。所有原始试次保留。</p>
<p><a href="report.md">完整报告</a> · <a href="findings.md">通俗解释</a> · <a href="review.csv">下载逐题表</a></p>
<label>被试 <select id="subject"><option value="">全部</option>'''
    page += ''.join(f'<option value="{s}"'+(' selected' if s == 205 else '')+f'>S{s}</option>' for s in sorted(frame.subject.unique()))
    page += '''</select></label> <label>从 <input type="number" id="lo" value="1" style="width:70px"></label>
<label>到 <input type="number" id="hi" value="2000" style="width:70px"></label>
<label>搜索原文或规则 <input id="query"></label> <span id="count"></span><table><thead><tr>'''
    page += ''.join('<th>'+label+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join(rows)+'''</tbody></table>
<script>const rows=[...document.querySelectorAll('tbody tr')];function filter(){const s=document.getElementById('subject').value,l=+document.getElementById('lo').value,h=+document.getElementById('hi').value,q=document.getElementById('query').value.toLowerCase();let n=0;for(const r of rows){const ok=(!s||r.dataset.subject===s)&&+r.dataset.trial>=l&&+r.dataset.trial<=h&&r.textContent.toLowerCase().includes(q);r.hidden=!ok;n+=ok;}document.getElementById('count').textContent='显示 '+n+' 条';}for(const k of ['subject','lo','hi','query'])document.getElementById(k).oninput=filter;filter();</script></html>'''
    (output/'review.html').write_text(page)


def run(source: Path, figures: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new report directory')
    output.mkdir(parents=True)
    inputs = {}

    def read(relative):
        path = source/relative; inputs[relative] = digest(path); return pd.read_csv(path)

    trials = read('diagnosis_v2/trials.csv')
    diag = read('diagnosis_v2/summary.csv')
    scores = read('readout_v1/trial_scores.csv.gz')
    fixed = read('content_v1/fixed_rule_contrasts.csv')
    content = read('content_v1/trials.csv')
    atoms = read('content_v1/atoms.csv.gz')
    ranks = read('content_v1/rank_ceilings.csv')
    rank_choice = read('content_v1/rank_choice.csv')
    anchor = read('anchor_v2/summary.csv')
    checks = numerical_audit(source)
    checks.to_csv(output/'anchor_numerical_audit.csv', index=False)
    relation = relation_audit(source, atoms)
    relation.to_csv(output/'S306_relation_audit.csv', index=False)
    comparisons = []
    for sid in sorted(trials.subject.unique()):
        own = scores[scores.subject.eq(sid) & scores.selection_window.eq('early_middle') &
            scores.early_middle & scores.split.eq('assessment') & scores.eligible]
        pivot = own.pivot(index='trial', columns='branch', values='atom_log_score')
        correct = fixed[fixed.subject.eq(sid) & fixed.comparator.eq('correct_rule')].set_index('trial').loc[pivot.index]
        mean = pivot.mean()
        comparisons.append(dict(subject=int(sid), reports=len(pivot),
            P1_vs_background=mean.semantic_P1-mean.expression_background,
            selected_vs_background=mean.semantic_selected-mean.expression_background,
            P1_vs_stimulus=mean.semantic_P1-mean.semantic_stimulus_choice,
            P1_vs_correct=float((pivot.semantic_P1-correct.atom_log_score).mean()),
            P1_vs_shuffled=mean.semantic_P1-mean.semantic_P1_shuffled))
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output/'matched_early_assessment.csv', index=False)
    primary = scores[scores.selection_window.eq('early_middle')].pivot(index=['subject','trial'], columns='branch', values='atom_log_score')
    gain = (primary.semantic_P1-primary.expression_background).rename('readout_gain').reset_index()
    review = trials.merge(content[['subject','trial','alternative_specific_support']], on=['subject','trial'], how='left', validate='one_to_one')
    review = review.merge(gain, on=['subject','trial'], how='left', validate='one_to_one')
    review.to_csv(output/'review.csv', index=False, encoding='utf-8-sig')
    review_html(review, output)
    img = lambda name: os.path.relpath(figures/(name+'.png'), output)
    first = diag[diag.window.eq('early_middle') & diag.split.eq('all')]
    rows = [[f'S{r.subject}',int(r.reports),f'{r.atomic_ceiling:.1%}',f'{r.P1_support:.1%}',
        f'{r.stimulus_reference:.1%}',f'{r.physical_all_true:.1%}'] for r in first.itertuples()]
    score_rows = [[f'S{r.subject}',r.reports,f'{r.P1_vs_background:+.4f}',f'{r.selected_vs_background:+.4f}',
        f'{r.P1_vs_correct:+.4f}',f'{r.P1_vs_shuffled:+.4f}'] for r in comparisons.itertuples()]
    rank_means = ranks[ranks.early_middle].groupby('subject')[['original_ceiling','combined_ceiling']].mean()
    rank_choice['gain'] = rank_choice.log_choice_probability-rank_choice.baseline_log_choice_probability
    choice_means = rank_choice.groupby(['subject','strategy']).gain.mean()
    rank_table = [[f'S{s}',f'{r.original_ceiling:.1%}',f'{r.combined_ceiling:.1%}',
        f'{choice_means[s,"longest"]:+.4f}',f'{choice_means[s,"shortest"]:+.4f}'] for s,r in rank_means.iterrows()]
    anchor_table = []
    for sid in [205,306]:
        own = anchor[anchor.subject.eq(sid) & anchor.bank.eq('all') & anchor.window.eq('early_middle') &
            anchor.split.eq('assessment')].set_index('variant')
        before, after = own.loc['behavior'], own.loc['oral_temperature_2']
        anchor_table.append([f'S{sid}',int(before.reports),f'{before.support:.2%}',f'{after.support:.2%}',
            f'{after.subset_log_probability-before.subset_log_probability:+.6f}',
            f'{int(checks[checks.subject.eq(sid)].passed_registered_working_checks.sum())}/{len(checks[checks.subject.eq(sid)])}'])
    page = f'''# 未学会阶段的口述与内部状态：本轮结果

本轮找到了一处较清楚的局部对应：S306在一段短腿刺激中，口述的头—脖子比较与模型的错误规则分支相符。
S229的前半段内容预测也比简单表达参照好。S205仍有明显失败；每64题加入一条口述，对随后留出理由的帮助很小。
目前支持部分错误判断的开发性解释，尚未建立两名被试完整错误信念的可靠恢复。

建议先读[通俗解释](findings.md)，再看下列图和[逐题复核页](review.html)。
重点被试的[256条预定抽样](../diagnosis_v2/review_sample.csv)按试次均匀选取，人工栏保持待复核。

## 一、资料和本轮比较的具体做法

六人4288题全部保留，复用17组行为估计状态。原始转写、旧编码、参数、旧结果均保持原记录。
3031条完整数值编码理由进入原语义分析；其他1257条在表内保留并标注原因。
新读出的刺激选项参照另有2条无法构造，相关方法比较使用相同的3029条可评分试次；例如S205前半段为108条留出报告。

行为拟合继续使用每人的完整记录，状态逐题学习。口述按32题块交替分为调整块和留出块，
每块首尾4题为缓冲。这里只为汇总统一取记录前半段，学习状况另看正确率和规则变化。
主要表达参数用前半段调整块确定，全程调整方案另存于原始表。全部被试及历史口述已经被查看，
这次属于内部开发比较，口述还待最终人工核对。

“判断支持”沿用原标准：某条规则的该选项区域内，每个刺激都满足该判断，才计为支持。
这比“当前刺激碰巧符合这句话”要求更强。每条理由里的判断等权平均，所有规则按原状态概率加权。
“目录上限”允许逐题查看整个目录中最能支持理由的局部规则，仅用于判断表达能力。

{table(['被试','前半段可评分理由','目录上限','P1实际支持','刺激选项参照','编码全部符合当前刺激'],rows)}

最后一列用于复核编码的参照和精度。例如“稍短”附近的细小差别、未说明参照的“长短”可能与固定数值阈值不同。
这一列没有口述真假或心理状态真值的含义。

![六人总览]({img('overview')})

图a展示原来的判断支持及表达上限。图b展示在同一批前半段留出条目中，P1的内容预测比各参照好多少。
纵轴大于零表示提供了额外预测信息。这里没有把分数直接解释为“读懂被试的百分比”。

## 二、换成明确的口述概率后，改善来自哪里

本轮把规则转换成对69种可能判断的概率，68种来自原目录，另有OTHER接收词表外内容。
对实际说出的各项判断计算对数概率，再在每条理由内部取平均。因此长理由和短理由各占一条权重。
用词频率和判断是否符合当前刺激也进入表达模型，这些信息同样提供给参照。

下表为平均对数概率差，单位为nats。0表示平均没有增益，正数表示给实际内容分配了更高的概率。
“重新选候选”只在原有限工作集合内、用调整块口述选择。部位重合读出未提供清楚的独立收益，
语义读出的全部网格和结果保留在[readout_v1](../readout_v1/selected.csv)。

{table(['被试','同批留出理由','P1相对表达参照','选候选后相对表达参照','P1相对正确规则','P1相对打乱状态'],score_rows)}

S229的增益较大，S306有小幅增益；S205几乎没有改善。S129只有19条前半段可评分留出理由，
且参数仍未确立，其结果只作诊断。S110的早期规则往往只有一个局部判断，而理由常提到四个部位，
完整解释仍受表达范围限制。S317后段的躯干参照有较多未编码内容，图中的空白保留这个限制。

正确规则对照和原模型使用同一套表达设置。状态打乱仅在同一选项、同一64题段内进行，
用于检查精确逐题对应是否优于该段的一般规则倾向。它们都是诊断参照。

## 三、S306具体对应了什么，哪里没有对应

![S306头脖子关系]({img('S306_head_neck')})

图a只看刺激中腿短的题，图b看腿长的题；两图保留全程所有进入主编码的头—脖子方向比较。
蓝点是留出块，灰点为调整块或缓冲题。后段空白表示口述主编码中未再出现这一明确关系。

模型前期较常使用R66：短腿时比较头和脖子，长腿时看两者合起来是否超过1.5倍躯干长度。
任务正确规则R42则是短腿时按脖子长短、长腿时按头长短分类。
因此“短腿时头比脖子长”的对应提供了具体错误规则内容的线索。
长腿时被试也常说头和脖子的相对长短，R66的求和分支解释不好；模型转向正确规则后，
被试一段时间仍在报告头—脖子比较，这部分也没有得到好的对应。

该片段在查看结果后用于说明，完整图及逐题表同时保留。统计范围、三组参数和简单参照见
[S306关系审查表](S306_relation_audit.csv)，逐条解释见[通俗结果](findings.md)。

## 四、S205的“最长、最短”是否缺少规则表达

把“最长部位决定类别”和“最短部位决定类别”的所有标签安排列出来，共48个候选。
它们只参与表达上限诊断。另用调整块中的部位排序和选项频率，预测留出块选择，
检验这种排序信息是否至少与作答有关。该频率参照还没有反馈学习或规则搜索机制。

{table(['被试','原目录上限','加入排序规则的上限','最长部位：选择评分增益','最短部位：选择评分增益'],rank_table)}

S205的表达上限有明显提高，仍有较多理由无法由单条候选共同表达。
最长部位与选择的联系只有小幅增益，说明表达缺口需要认真处理，同时还需检验新的规则如何生成实际选择。
48个候选的最好逐题分数属于上限，认知模型尚未采用这些规则。

## 五、每64题加入一条口述，能否帮助相邻状态

S205固定11条、S306固定12条锚点，均按距离各段第16题最近的合格调整口述选择。
口述读出的表达参数先用调整块估计；每段状态加权只额外加入该段这一条锚点。
完整编码理由作为一个判断集合，使用精确归一化的集合概率；集合大小作为条件给定。
模型与表达背景各占一半。主语义强度为2，1和4两个敏感性值均保留。

重建使用截至该64题段末的行为，按原粒子祖先保留整条路径，再用锚点概率调整路径权重。
S205保留抽样执行身份；S306的零维持混合在粒子内部精确求和，并更新锚点的局部执行分布。
行为对照使用完全相同的段末和路径。各64题段分别重建，前段锚点不累积到后段。

{table(['被试','前半段留出理由','行为支持','加入口述后支持','完整理由评分增益','数值工作检查通过段数'],anchor_table)}

两人的新增收益都很小。主结果并列16次计算及两个8次半组，检查独立计算权重、锚点祖先多样性、
路径有效数量变化及半组评分差。逐段数值见[检查表](anchor_numerical_audit.csv)。
通过这些检查表示本次有限路径重算没有明显数值异常，原有潜在状态恢复边界继续保留。

![S205口述辅助]({img('S205_anchor')})

![S306口述辅助]({img('S306_anchor')})

图a的两条线分别是同段末行为重建和加入一条口述后的结果。图b仅比较后32题留出口述，
单位为每条完整编码理由的对数概率差，与前面的单项判断分数分别解释。
图c显示整个局部规则分布改变了多少；接近零表示额外口述没有明显改变这些试次的推断。

## 六、六人全程图

每人四行：学习正确率与正确规则概率；原语义支持及表达上限；固定64题段的留出预测增益；
正确规则支持与其他规则特有内容支持。连续线使用最近32题，至少8条可评分报告，缺失维持原位置。
第四行的橙线表示其他规则支持、正确任务规则的该选项分区未要求的内容。
它通常是对额外判断的支持，直接与正确规则矛盾的内容很少。原话说明的是决策理由，仍需区分判断与分类联系。

'''
    for sid in sorted(trials.subject.unique()):
        page += f'### S{sid}\n\n![S{sid}全程]({img(f"S{sid}_timecourse")})\n\n'
    page += '''## 七、计算完成范围与后续决定

本轮复用旧状态，新增认知粒子滤波0次。目录诊断复现21567项旧评分，读出比较检查7840370个归一化概率行，
口述辅助完成32个独立系统的存档重算，另有单系统试跑。原始数据和旧结果均保留。
初始脚本的一次返回值处理错误，以及S306混合执行的检查中断，均保留原目录和日志。
修正后的主结果使用diagnosis_v2、readout_v1、content_v1、anchor_v2。

按预定的两次追加上限停止继续搜索。S205缺口与S306分支差异已得到具体定位，少量口述辅助的有限方案没有推广收益。
下一步优先人工确认这些原话所指的比较和类别联系，再决定是否采用更丰富的理由模型或把排序规则接入学习模型。
本次没有校准口述噪声、重新拟合完整认知模型、开展新模拟状态恢复或独立被试验证。

源码、预算和停止规则见[执行协议](../../../../src/Bayesian_state/docs/model_architecture/model_1024_oral_development.md)。
实验登记、两次追加及判断记录在研究目录；最终文件核验、实际成本和本地提交另见[最终交付记录](../delivery.json)。
'''
    (output/'report.md').write_text(page)
    result = dict(status='complete', reports=len(trials), primary_reports=int(trials.eligible.sum()),
        matched_early_report_counts=comparisons[['subject','reports']].to_dict('records'),
        anchor_blocks=int(len(checks)), anchor_blocks_passing=int(checks.passed_registered_working_checks.sum()),
        inputs_sha256=inputs, source_sha256=digest(Path(__file__)),
        artifacts={f.name:digest(f) for f in output.iterdir() if f.is_file()})
    (output/'report_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.figures, args.output)
    print({k:v for k,v in result.items() if k not in ('inputs_sha256','artifacts')})
