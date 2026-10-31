"""Describe prespecified catalogue rules in a completed behavior-only pilot.

The target IDs and windows are frozen in the study configuration. All finalists,
both numerical repeats, and every trial are retained. Feature scope and semantic
rule identity are separate quantities; no oral coding is modified here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from .model_1031_subject_pilot_report import markdown_table, read_job


def run(study: Path, report: Path, figures: Path) -> None:
    destinations = ['rule_diagnostics.npz', 'rule_diagnostics.json', 'rule_windows.csv',
                    'rule_trials.csv', 'rule_catalogue_windows.csv', 'rule_report.md']
    if any((report / name).exists() for name in destinations):
        raise FileExistsError('Preserve existing diagnostics; use a new report directory')
    register = json.loads((study / 'registration.json').read_text())
    summary = json.loads((report / 'summary.json').read_text())
    policy = register['policy']; targets = policy['rule_diagnostics']
    frame = pd.read_csv(study / 'subject_source.csv')
    names = summary['feature_names']; n = len(frame)
    with np.load(study / 'behavior_input.npz') as z:
        x, choices, categories = z['stimulus'], z['choices'], z['categories']
    catalogue = ContinuousPartition(4, 2)
    correct_id, wrong_id = targets['correct_rule_id'], targets['wrong_rule_id']
    neck, head = names.index('neck'), names.index('head')
    wrong_choices = np.where(x[:, head] > x[:, neck], 1, 2)
    np.testing.assert_array_equal(
        catalogue.get_category_assignments(correct_id, x, distance_mode='boundary') + 1, categories)
    unequal = x[:, head] != x[:, neck]
    np.testing.assert_array_equal(
        catalogue.get_category_assignments(wrong_id, x[unequal], distance_mode='boundary') + 1,
        wrong_choices[unequal])
    # Verify category orientation independently of observed choices and stimuli.
    probe = np.random.default_rng(0).uniform(.01, .99, (100, 4))
    np.testing.assert_array_equal(
        catalogue.get_category_assignments(wrong_id, probe, distance_mode='boundary') + 1,
        np.where(probe[:, head] > probe[:, neck], 1, 2))
    oral_mask = frame[[f'feature{i}_use' for i in range(1, 5)]].to_numpy(int)
    oral_codes = oral_mask @ (2 ** np.arange(4))
    pair_code = (1 << neck) | (1 << head)
    eligible = (oral_codes > 0) & frame.text.notna().to_numpy()
    session_starts = (np.flatnonzero(frame.iSession.ne(frame.iSession.shift()).to_numpy()) + 1).tolist()
    ranking = json.loads((study / 'final_ranking.json').read_text())
    plot = dict(trial=np.arange(1, n + 1),
        oral_head_neck_scope=np.where(eligible, oral_codes == pair_code, np.nan),
        oral_neck_scope=np.where(eligible, oral_codes == (1 << neck), np.nan))
    windows, trials, catalogue_rows, point_names, source_hashes = [], [], [], [], {}
    for index, point in enumerate(ranking):
        model, candidate = point['model'], point['candidate']
        key = f'{model}_{candidate}'
        files = sorted((study / 'jobs').glob(f'final_{key}_s*.json'))
        if len(files) != len(policy['search']['final_seeds']):
            raise ValueError('Incomplete numerical repeats')
        states = [read_job(path)[1] for path in files]
        for path in files:
            source_hashes[str(path)] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
        execution = np.stack([s['catalogue_execution_probability'] for s in states])
        feature = np.stack([s['feature_set_probability'] for s in states])
        assert execution.shape == (len(files), n, catalogue.length)
        assert np.isfinite(execution).all() and np.all(execution >= 0)
        if model == 'model_1024':
            np.testing.assert_allclose(execution.sum(axis=2), 1.)
        else:
            groups = np.stack([s['filtered_representation_execution'] for s in states])
            np.testing.assert_allclose(execution.sum(axis=2), groups[:, :, 3], atol=1e-12)
        np.testing.assert_array_less(execution[:, :, wrong_id] - 1e-12, feature[:, :, pair_code])
        selected = candidate == summary['selected'][model]
        point_names.append(dict(key=key, model=model, candidate=candidate, selected=selected))
        plot[key + '_wrong'] = execution[:, :, wrong_id]
        plot[key + '_correct'] = execution[:, :, correct_id]
        for a, b in targets['windows']:
            if not 1 <= a <= b <= n:
                raise ValueError('Invalid frozen diagnostic window')
            region = slice(a - 1, b)
            wrong = execution[:, region, wrong_id].mean(axis=1)
            correct = execution[:, region, correct_id].mean(axis=1)
            windows.append(dict(model=model, candidate=candidate, selected=selected,
                first_trial=a, last_trial=b, wrong=float(wrong.mean()),
                wrong_repeat_min=float(wrong.min()), wrong_repeat_max=float(wrong.max()),
                correct=float(correct.mean()), correct_repeat_min=float(correct.min()),
                correct_repeat_max=float(correct.max()),
                head_neck_scope=float(feature[:, region, pair_code].mean()),
                actual_accuracy=float((choices[region] == categories[region]).mean()),
                wrong_rule_choice_agreement=float((choices[region] == wrong_choices[region]).mean())))
            for rule_id, mass in enumerate(execution[:, region].mean(axis=(0, 1))):
                catalogue_rows.append(dict(model=model, candidate=candidate, first_trial=a,
                    last_trial=b, rule_id=rule_id, execution_probability=float(mass),
                    hyperplanes=json.dumps(catalogue.hypothesis_space[rule_id].hyperplanes)))
        for t in range(n):
            trials.append(dict(model=model, candidate=candidate, selected=selected, trial=t + 1,
                text=frame.iloc[t].text, choice=int(choices[t]), correct_category=int(categories[t]),
                comparison_choice=int(wrong_choices[t]),
                wrong=float(execution[:, t, wrong_id].mean()),
                wrong_repeat_min=float(execution[:, t, wrong_id].min()),
                wrong_repeat_max=float(execution[:, t, wrong_id].max()),
                correct=float(execution[:, t, correct_id].mean()),
                correct_repeat_min=float(execution[:, t, correct_id].min()),
                correct_repeat_max=float(execution[:, t, correct_id].max()),
                head_neck_scope=float(feature[:, t, pair_code].mean())))
    np.savez_compressed(report / 'rule_diagnostics.npz', **plot)
    pd.DataFrame(windows).to_csv(report / 'rule_windows.csv', index=False)
    pd.DataFrame(trials).to_csv(report / 'rule_trials.csv', index=False)
    pd.DataFrame(catalogue_rows).to_csv(report / 'rule_catalogue_windows.csv', index=False)
    metadata = dict(subject=policy['subject'], targets=targets, points=point_names,
        session_starts=session_starts,
        source_sha256=source_hashes, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='catalogue_execution_mass; templates_remain_separate; not_report_generation_probability',
        timing='after_current_choice_before_current_feedback; parameters_selected_on_full_behavior',
        windows='frozen_before_fit; oral_selected_development_descriptions',
        oral_scope='original_feature_codes; two_feature_scope_does_not_imply_a_comparison_rule')
    (report / 'rule_diagnostics.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    rows = [[r['model'].replace('model_', ''), r['candidate'] + ('*' if r['selected'] else ''),
        f"{r['first_trial']}—{r['last_trial']}",
        f"{r['wrong']:.1%}（{r['wrong_repeat_min']:.1%}—{r['wrong_repeat_max']:.1%}）",
        f"{r['correct']:.1%}（{r['correct_repeat_min']:.1%}—{r['correct_repeat_max']:.1%}）",
        f"{r['head_neck_scope']:.1%}"] for r in windows]
    figure_link = os.path.relpath(figures / 'wrong_and_correct_rules.png', report)
    text = f'''# S{policy['subject']}：错误规则与正确规则的具体检查

本页检查拟合前确定的两条规则。正确规则为脖子短选1、脖子长选2（目录{correct_id}号）；
错误比较规则为头长于脖子选1、头短于脖子选2（目录{wrong_id}号）。这两条规则的类别方向
已用实际刺激及独立生成的检查点核对。相等边界不用于比较方向核验。

纵轴是行为推断赋予“本题执行这条具体规则”的概率，条件为本题选择已知、当前反馈尚未知。
参数由全384题行为选择，状态按试次过滤；口述没有进入参数排序或粒子权重。
本例第321题开始第二场；模型沿记录连续更新，未额外估计场次间隔的状态变化。
具体模板、局部概括及其他规则可以给出同一个回答，因此这项量与选择预测概率分别解释。
它也只统计目录中的该条规则，局部模板与它在部分刺激上的一致性另待分析。

![错误规则与正确规则]({figure_link})

图a沿用原始部位编码，显示只提头和脖子、只提脖子的口述比例。头和脖子的共同出现只确定部位范围。
图b、c分别画错误比较规则与正确规则的执行概率，实线为各模型按行为选出的主候选，虚线为另一候选。
范围带为主候选两次数值重复的范围；全部曲线使用最近32题平均，开始时使用已有题数。
图a的窗口至少含8条可评分口述。灰底标记事先确定的89—115题与321—384题。

以下表格直接平均指定窗口内的逐题值。括号是两次数值重复的范围，*表示主展示参数。
“头与脖子范围”合计所有恰好使用这两个部位的内容，其中包括模板及其他判断形式。

{markdown_table(['模型','候选','试次','错误规则执行','正确规则执行','头与脖子范围'], rows)}

这些区间根据本例口述提前指定，属于开发性对照。中段还包含其他比较尝试；
第106题口述方向与刺激不一致，第194题提到腿与脖子，所有行为记录均保留。
完整384题、全部四个最终候选和两次重复汇总见`rule_trials.csv`，
全部29条目录规则的阶段概率见`rule_catalogue_windows.csv`。
固定参数的重复差异、不同参数的状态差异以及状态恢复分别决定后续解释范围。
本轮没有完成模拟状态恢复，完整口述内容的概率尚未评分。
'''
    (report / 'rule_report.md').write_text(text)
    print(json.dumps(dict(points=point_names, windows=len(windows), trials=len(trials))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.report, args.figures)
