"""Summarize the frozen single-subject pilot without changing its selection."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...metrics.autonomous_fit import behavior_features, energy_u

NAMES = {'model_1031': 'Model 1031', 'model_1024': 'Model 1024'}
PARTS = {'tail': '尾巴', 'neck': '脖子', 'leg': '腿', 'head': '头'}
GROUPS = ['尚未形成内容', '具体示例', '局部概括', '目录规则']


def read_job(path: Path) -> tuple[dict, dict]:
    record = json.loads(path.read_text())
    array_path = path.with_suffix('.npz')
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != record['array_sha256']:
        raise ValueError(f'Job array changed: {array_path}')
    with np.load(array_path, allow_pickle=False) as z:
        # Large particle contents remain in the original job archive. Summary
        # plotting uses every path and all marginal state probabilities.
        arrays = {key: z[key] for key in z.files if not key.startswith('particle_')}
    return record, arrays


def block_means(values: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(values, float)
    return np.stack([x[..., start:start + width].mean(axis=-1)
                     for start in range(0, x.shape[-1], width)], axis=-1)


def markdown_table(headers, rows) -> str:
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def run(study: Path, output: Path, figures: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    register = json.loads((study / 'registration.json').read_text())
    complete = json.loads((study / 'completion.json').read_text())
    ranking = json.loads((study / 'final_ranking.json').read_text())
    policy = register['policy']; n = register['trials']
    frame = pd.read_csv(study / 'subject_source.csv').reset_index(drop=True)
    with np.load(study / 'behavior_input.npz') as z:
        choices, categories, feedback = z['choices'], z['categories'], z['feedback']
    oral_mask = frame[[f'feature{i}_use' for i in range(1, 5)]].to_numpy(int)
    if not np.isin(oral_mask, [0, 1]).all():
        raise ValueError('Original feature-use coding must be binary')
    codes = oral_mask @ (2 ** np.arange(4))
    eligible = (codes > 0) & frame.text.notna().to_numpy()
    phases = [(start, min(start + policy['summaries']['phase_trials'], n))
              for start in range(0, n, policy['summaries']['phase_trials'])]
    mask_bits = (np.arange(16)[:, None] & (2 ** np.arange(4))) > 0
    metadata = dict(study=str(study), selected={}, summaries={}, phases=phases,
        feature_names=[frame.iloc[0][f'feature{i}_name'] for i in range(1, 5)],
        oral_eligible=int(eligible.sum()), oral_total=n,
        oral_unresolved_rows=(np.flatnonzero(~eligible) + 1).tolist(),
        oral_scope='original_feature_use_only; not_complete_content_or_report_probability',
        inference_timing='after_current_choice_before_current_feedback; full_record_selected_fixed_parameters',
        parameter_uncertainty='two_search_finalists_not_a_calibrated_region',
        source_commit=register['source_git_head'])
    plot = dict(trial=np.arange(1, n + 1), actual_feedback=feedback,
                oral_mask=oral_mask, oral_eligible=eligible, oral_feature_count=np.where(eligible, oral_mask.sum(axis=1), np.nan))
    point_rows, phase_rows, sensitivity_rows = [], [], []
    reference_features = behavior_features(choices[None, :], categories, n_categories=2)[0]
    final_points = {}
    for model in policy['models']:
        selected = next(r['candidate'] for r in ranking if r['model'] == model)
        metadata['selected'][model] = selected
        for row in [r for r in ranking if r['model'] == model]:
            candidate = row['candidate']
            files = sorted((study / 'jobs').glob(f'final_{model}_{candidate}_s*.json'))
            pairs = [read_job(path) for path in files]
            records, states = zip(*pairs)
            probabilities = np.stack([s['choice_probabilities'] for s in states])
            feature_sets = np.stack([s['feature_set_probability'] for s in states])
            groups = (np.stack([s['filtered_representation_execution'] for s in states])
                      if model == 'model_1031' else np.tile([0., 0., 0., 1.], (len(states), n, 1)))
            generation = [read_job(path)[1] for path in sorted((study / 'jobs').glob(f'generate_{model}_{candidate}_*.json'))
                          if not path.name.endswith('.paths.json')]
            generated = np.concatenate([a['feedback'] for a in generation], axis=0)
            expected_paths = policy['search']['generation_paths']
            if len(generated) != expected_paths:
                raise ValueError('Incomplete autonomous ensemble')
            # Under binary deterministic feedback, an error uniquely identifies
            # the other answer. This reconstructs choices for the shared metric.
            generated_choices = np.where(generated == 1., categories[None], 3 - categories[None])
            score = energy_u(behavior_features(generated_choices, categories, n_categories=2), reference_features)
            feature_mean = feature_sets.mean(axis=0)
            exact = np.where(eligible, feature_mean[np.arange(n), codes], np.nan)
            feature_probability = feature_mean @ mask_bits.astype(float)
            criterion = np.asarray([r['nll'] for r in records])
            lower = next(r for r in json.loads((study / 'confirm_ranking.json').read_text())
                         if r['model'] == model and r['candidate'] == candidate)
            precision_change = float((criterion.mean() - lower['mean_nll']) / n)
            final_points[(model, candidate)] = dict(probabilities=probabilities, groups=groups,
                feature_sets=feature_sets, generated=generated, exact=exact)
            point = dict(model=model, candidate=candidate, selected=candidate == selected,
                nll=float(criterion.mean()), nll_min=float(criterion.min()), nll_max=float(criterion.max()),
                nll_per_trial=float(criterion.mean() / n), nll_sd=float(criterion.std(ddof=1)),
                precision_change_per_trial=precision_change,
                working_precision_tolerance_met=abs(precision_change) <= policy['summaries']['precision_nll_per_trial'],
                energy=score['value'], energy_mc_se=score['mc_se'], paths=len(generated),
                first64_generated_mean=float(generated[:, :64].mean()), last64_generated_mean=float(generated[:, -64:].mean()),
                probability_last64_at_least_observed=float(np.mean(generated[:, -64:].mean(axis=1) >= feedback[-64:].mean())),
                early_exact_features=float(np.nanmean(exact[:64])), late_exact_features=float(np.nanmean(exact[-64:])),
                early_specific_execution=float(groups.mean(axis=0)[:64, 1].mean()),
                early_local_execution=float(groups.mean(axis=0)[:64, 2].mean()),
                late_rule_execution=float(groups.mean(axis=0)[-64:, 3].mean()),
                late_tail_only_execution=float(feature_mean[-64:, 1].mean()))
            point_rows.append(point)
            for a, b in phases:
                own = eligible[a:b]
                phase_rows.append(dict(model=model, candidate=candidate, selected=candidate == selected,
                    first_trial=a+1, last_trial=b, oral_eligible=int(own.sum()), oral_total=b-a,
                    actual_correct=float(feedback[a:b].mean()), autonomous_correct_mean=float(generated[:, a:b].mean()),
                    autonomous_correct_q10=float(np.quantile(generated[:, a:b].mean(axis=1), .1)),
                    autonomous_correct_q90=float(np.quantile(generated[:, a:b].mean(axis=1), .9)),
                    feature_exact=float(np.nanmean(exact[a:b])) if own.any() else None,
                    specific=float(groups[:, a:b, 1].mean()), local=float(groups[:, a:b, 2].mean()),
                    rule=float(groups[:, a:b, 3].mean()), tail_only=float(feature_mean[a:b, 1].mean())))
            sensitivity_rows.append(dict(model=model, candidate=candidate,
                nll_repeat_gap=float(abs(criterion[0]-criterion[1])),
                choice_probability_difference_q95=float(np.quantile(np.abs(probabilities[0]-probabilities[1]), .95)),
                feature_set_tv_mean=float((.5*np.abs(feature_sets[0]-feature_sets[1]).sum(axis=1)).mean()),
                feature_set_tv_q95=float(np.quantile(.5*np.abs(feature_sets[0]-feature_sets[1]).sum(axis=1), .95))))
            if candidate == selected:
                metadata['summaries'][model] = point
                plot[model + '_conditional_correct'] = probabilities.mean(axis=0)[np.arange(n), categories - 1]
                plot[model + '_generated'] = generated
                plot[model + '_group'] = groups.mean(axis=0)
                plot[model + '_feature_probability'] = feature_probability
                plot[model + '_feature_count'] = feature_probability.sum(axis=1)
                plot[model + '_feature_exact'] = exact
                plot[model + '_feature_sets'] = feature_mean
                plot[model + '_groups_by_repeat'] = groups
        own_candidates = [r['candidate'] for r in ranking if r['model'] == model]
        if len(own_candidates) == 2:
            first, second = [final_points[(model, c)] for c in own_candidates]
            a, b = first['feature_sets'].mean(axis=0), second['feature_sets'].mean(axis=0)
            metadata.setdefault('between_parameter_feature_tv', {})[model] = float((.5*np.abs(a-b).sum(axis=1)).mean())
            plot[model + '_alternative_group'] = second['groups'].mean(axis=0)
            plot[model + '_alternative_feature_count'] = (b @ mask_bits.astype(float)).sum(axis=1)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'plot_data.npz', **plot)
    pd.DataFrame(point_rows).to_csv(output / 'final_candidates.csv', index=False)
    pd.DataFrame(phase_rows).to_csv(output / 'phase_summary.csv', index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output / 'numerical_sensitivity.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    names = [PARTS.get(x, x) for x in metadata['feature_names']]
    def mask_label(code):
        return '、'.join(name for d, name in enumerate(names) if code & (1 << d)) or '未指明具体部位'
    review = frame[['iSub', 'iSession', 'iBlock', 'iTrial', 'choice', 'feedback', 'text']].copy()
    review.insert(1, 'trial', np.arange(1, n + 1))
    review['oral_eligible'] = eligible
    review['oral_features'] = [mask_label(c) for c in codes]
    for model in policy['models']:
        review[model + '_feature_exact'] = plot[model + '_feature_exact']
        review[model + '_modal_features'] = [mask_label(int(c)) for c in plot[model + '_feature_sets'].argmax(axis=1)]
    for i, name in enumerate(('uncommitted', 'specific', 'local', 'rule')):
        review['model_1031_' + name] = plot['model_1031_group'][:, i]
    review['review_note'] = np.where(eligible, '现有特征编码；长短方向另待检验', '保留原文：未指明部位的数量概括或操作说明')
    review.to_csv(output / 'trial_review.csv', index=False)
    html_rows = []
    for row in review.to_dict('records'):
        values = [row['trial'], f"{row['iSession']}/{row['iBlock']}/{row['iTrial']}", row['text'],
                  row['choice'], row['feedback'], row['oral_features'],
                  f"{row['model_1031_specific']:.1%} / {row['model_1031_local']:.1%} / {row['model_1031_rule']:.1%}",
                  row['model_1031_modal_features'],
                  '未评分' if not row['oral_eligible'] else f"{row['model_1031_feature_exact']:.1%}", row['review_note']]
        html_rows.append(f"<tr id='trial-{row['trial']}'>" + ''.join('<td>'+html.escape(str(v))+'</td>' for v in values) + '</tr>')
    headings = ['有效试次', '原始场/块/题', '原始理由', '选择', '反馈', '既有编码部位',
                '具体/局部/规则执行概率', '新模型最可能使用的部位', '部位集合吻合概率', '复核备注']
    (output / 'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>S118 初步复核</title>
<style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #ddd;vertical-align:top}th{position:sticky;top:0;background:#eef3f7}td:nth-child(3){min-width:300px}tr:nth-child(even){background:#fafafa}</style>
<h1>S118 逐题理由与行为推断</h1><p>口述参与选人；参数和状态只使用行为。状态为选择后、反馈前；参数由全记录选出。
部位吻合只评价特征集合，具体长短及整句内容另待检验。所有256条原文保留。</p><table><tr>'''
        + ''.join('<th>'+h+'</th>' for h in headings)+'</tr>'+''.join(html_rows)+'</table>')
    top_new, top_old = [metadata['summaries'][m] for m in ('model_1031', 'model_1024')]
    comparisons = [[NAMES[r['model']], r['candidate'], f"{r['nll']:.2f}", f"{r['nll_min']:.2f}–{r['nll_max']:.2f}",
                    f"{r['energy']:.4f}", f"{r['early_exact_features']:.1%}", f"{r['late_exact_features']:.1%}"] for r in point_rows]
    phase_compare = [r for r in phase_rows if r['model']=='model_1031' and r['selected']]
    phase_table = [[f"{r['first_trial']}–{r['last_trial']}", f"{r['actual_correct']:.1%}",
        f"{r['autonomous_correct_mean']:.1%}（{r['autonomous_correct_q10']:.1%}–{r['autonomous_correct_q90']:.1%}）",
        f"{r['specific']:.1%} / {r['local']:.1%} / {r['rule']:.1%}", f"{r['feature_exact']:.1%}", f"{r['oral_eligible']}/{r['oral_total']}"] for r in phase_compare]
    def relative(target: Path) -> str:
        import os
        return os.path.relpath(target, output)
    report = f'''# S118：新模型的首个个案试验

本轮已完成预定计算：两种模型各16个候选，分别经过32、128、512粒子检查；最终各两个候选，
每个候选生成64条自主路径，共256条完整路径。计算耗时{complete['elapsed_seconds']/60:.1f}分钟、
累计{complete['cpu_seconds']/3600:.2f} CPU小时。范围为单个开发案例，完整256题用于选参。

## 为什么选择S118

前64题有63条明确描述四个部位，后64题有63条只编码尾巴。第33题为“头短，脖子长，腿短，尾巴短”，
第113题为“三个部位都很短”，第177题为“尾巴比较长”。这呈现具体描述、部位数量概括及单特征理由的变化。
早期具体描述与示例记忆相容；“正在记忆例子”仍是待检验的心理解释。

本轮参数与状态只使用选择和反馈，口述用于选人及事后描述。口述已参与开发，原文与编码待人工复核。

## 行为与口述范围的结果

主展示候选为新模型{metadata['selected']['model_1031']}、旧模型{metadata['selected']['model_1024']}，分别按512粒子两次重复的平均选择NLL排序。
NLL越低表示真实选择得到更高概率。新减旧的NLL差为{top_new['nll']-top_old['nll']:.2f}，
每题差为{(top_new['nll']-top_old['nll'])/n:.4f}。两次重复范围表示计算变化，优化范围仅限本轮候选。
均匀猜测的NLL为{n*np.log(2):.2f}。

{markdown_table(['模型','候选','平均NLL','两次NLL范围','自主分布评分','前64题部位吻合','后64题部位吻合'], comparisons)}

自主分布评分为共同的行为特征 Energy 分数，越低越好；同时考虑32题正确率和按真实类别分组的回答频率。
评分使用全部64条路径，计算误差见`final_candidates.csv`的`energy_mc_se`。
部位吻合指执行内容的特征集合与现有口述特征编码完全相同的概率，具体长短判断和整句口述概率另待检验。

![行为与表示变化]({relative(figures / 'behavior_and_representation.png')})

图a是观察被试历史之后、下一次选择之前的正确类别概率；图b是模型沿自己的回答和反馈生成的路径。
两者均用最近32题平均；图b范围带包含同一参数下64条路径的中间80%，蓝线为平均值。
黑线为被试实际正确率。图c为本题选择后、反馈前的表示类型执行概率。
图d比较口述指明的部位数与模型执行内容涉及的维度数。数值编码没有指明具体部位的题目保留空缺。

## 分阶段看新模型

{markdown_table(['试次','实际正确率','自主均值（10%–90%）','具体/局部/规则执行','部位吻合','可评分口述'], phase_table)}

新模型在前64题的具体示例执行概率平均为{top_new['early_specific_execution']:.1%}，局部概括为
{top_new['early_local_execution']:.1%}；后64题的目录规则执行概率为{top_new['late_rule_execution']:.1%}，
只使用尾巴的概率为{top_new['late_tail_only_execution']:.1%}。
具体示例起步是模型预先设定的机制。早期四部位吻合中包含这个初始假设的作用，
需结合该状态保持时间、实际行为拟合及后续特征变化解释。

![特征使用和原始口述对照]({relative(figures / 'feature_use.png')})

图a保留原有口述特征编码；白色空列为未指明具体部位。图b是新模型推断的各部位参与执行的概率。
图c显示部位集合吻合概率，采用最近32题中可评分记录的平均；至少8条才绘制。
完整逐题结果见[口述复核表](review.html)，包含全部256条原文。

## 尚未解释的内容与计算限制

共有{int((~eligible).sum())}条口述缺少明确部位身份，主要是“两个／三个部位很长”，另有“按错了”。
这部分保留原文，未进入部位集合分数。当前删去维度的局部概括尚未显式表示“有几个部位很长”这一操作，
因此中间阶段的完整理由仍有机制缺口。

选择相近参数下的状态可以不同，`numerical_sensitivity.csv`分别记录同参数两次运行的分布差异，
`summary.json`记录两个最终候选之间的特征集合分布差异。两次重复及512粒子只提供本轮数值检查，
参数区域、状态恢复和独立口述预测尚待检验。自主路径使用固定刺激日程及256题长度，任务停止过程另待检验。

本轮全部候选与重复结果位于上级目录，两个最终候选均保留。图表和参数排序没有根据口述匹配度调整。
研究协议见[执行说明]({relative(Path('src/Bayesian_state/docs/model_architecture/model_1031_subject118_pilot.md'))})。
'''
    (output / 'report.md').write_text(report)
    print(json.dumps(dict(output=str(output), new=top_new, old=top_old), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output, args.figures)
