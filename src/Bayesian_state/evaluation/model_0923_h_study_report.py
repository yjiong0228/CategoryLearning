"""Assemble the completed overnight H study without modifying source results.

Run from the repository root with --config and --output. All scientific inputs
and linked reports must exist before rendering; existing files are preserved.
The overview distinguishes development screens, chronological holdout scores,
finite-simulation errors and known-nuisance synthetic recovery.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from .model_0923_h_validation import digest, load


def build(config: Path, output: Path) -> None:
    settings = yaml.safe_load(config.read_text())
    if any((output / name).exists() for name in ('index.html', 'report_manifest.json')):
        raise FileExistsError('Use a new overview destination')
    sources = {key: Path(settings[key]) for key in
               ('summary', 'training_audit', 'profiles', 'profile_audit', 'recovery', 'pair_audit', 'uncertainty', 'criterion')}
    for key in ('profiles', 'profile_audit', 'recovery', 'pair_audit', 'uncertainty', 'criterion'):
        if load(sources[key] / 'completion.json')['status'] != 'complete':
            raise ValueError(f'Incomplete input: {key}')
    nomination_sha = load(sources['summary'] / 'summary_manifest.json')['nomination_sha256']
    if digest(sources['training_audit'] / 'nominees.json') != nomination_sha:
        raise ValueError('Overview sources use different training nominations')
    for key in ('profiles', 'profile_audit', 'recovery', 'uncertainty', 'criterion'):
        record = 'completion.json' if key in ('uncertainty', 'criterion') else 'protocol.json'
        if load(sources[key] / record)['nomination_sha256'] != nomination_sha:
            raise ValueError(f'Different training nominations in {key}')
    if load(sources['pair_audit'] / 'protocol.json')['source_protocol_sha256'] != digest(sources['recovery'] / 'protocol.json'):
        raise ValueError('Pair audit and recovery use different synthetic inputs')
    if load(sources['profile_audit'] / 'completion.json')['nominees_changed']:
        raise ValueError('Optimizer diagnostic changed the primary nominees')
    if load(sources['profile_audit'] / 'protocol.json')['profile_scores_sha256'] != digest(sources['profiles'] / 'profiles.csv'):
        raise ValueError('Optimizer diagnostic uses different training profiles')
    links = [(title, Path(path)) for title, path in settings['links']]
    for _, path in links:
        if not path.exists():
            raise FileNotFoundError(path)
    files = [sources['summary'] / 'scores.csv', sources['summary'] / 'aggregate_choice_scores.csv',
             sources['uncertainty'] / 'aggregate.csv', sources['recovery'] / 'recovered_decay.csv',
             sources['pair_audit'] / 'contrasts.csv', sources['profiles'] / 'profiles.csv',
             sources['profile_audit'] / 'contrasts.csv', sources['criterion'] / 'subjects.csv']
    scores, choices, uncertainty, recovery, pairs, profiles, optimizer, criterion = [pd.read_csv(path) for path in files]
    test = scores[scores.phase == 'test']
    columns = ['choice_nll', 'energy', 'crps', 'interval_coverage', 'interval_width', 'probability_mean_mae']
    mean = test.groupby('model', sort=False)[columns].mean().reset_index()
    mean = mean.rename(columns={'model': '模型', 'choice_nll': '逐题负对数评分', 'energy': '自主曲线energy score',
        'crps': '逐块CRPS', 'interval_coverage': '80%区间覆盖率', 'interval_width': '平均区间宽度',
        'probability_mean_mae': '自主均值绝对误差'})
    recovery_error = (recovery.candidate_decay - recovery.true_decay).abs()
    strong = pairs.difference_exceeds_two_mc_se.astype(bool)
    profile_minima = profiles.loc[profiles.groupby(['subject', 'model', 'profile']).delta_anchor_nll.idxmin(),
        ['subject', 'model', 'profile', 'first', 'second', 'delta_anchor_total_nll', 'delta_anchor_total_mc_se']]
    profile_minima = profile_minima.rename(columns={'first': 'first_coordinate', 'second': 'second_coordinate'})
    parameter_names = {'decay': '证据保持率', 'gamma': '记忆保持率', 'exit_rate': '退出率',
                       'stay_bias': '旧规则保留偏好', 'max_capacity': '容量', 'baseline_probability': '基础搜索概率'}
    profile_minima['profile'] = profile_minima.profile.map(
        lambda value: ' | '.join(parameter_names[part.strip().split('.')[-1]] for part in value.split('|')))
    table = lambda frame: '<div class="scroll">' + frame.to_html(index=False, float_format=lambda x: f'{x:.5f}') + '</div>'
    rel = lambda path: escape(os.path.relpath(path, output))
    body = [
        '<!doctype html><html lang="zh"><meta charset="utf-8">',
        f'<title>{escape(settings["title"])}</title>',
        '<style>body{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.7 sans-serif;color:#222}h1{font-size:28px}h2{font-size:22px;margin-top:32px}.decision{padding:16px;background:#edf3ed;border-left:4px solid #637e52}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid #ddd}a{color:#245d86}details{margin:16px 0}</style>',
        f'<h1>{escape(settings["title"])}</h1><p class="decision">{escape(settings["decision"])}</p>',
        '<p>15名既有被试用于四轮结构筛查。新增101、103、201、202、301、302在查看学习表现前按任务及编号选定；仅用前缀训练，冻结参数后评价832题留出选择，并从任务起点自主生成。六个结构采用相同参数搜索预算，结果对应有限搜索得到的参数点。</p>',
        '<p>ECM_S的自主生成较原模型改善，但逐题留出预测未形成稳定优势，202出现明确的生成退步。所有结构的80%预测区间覆盖率偏低。当前结果支持继续研究连续证据及容量竞争，并保留搜索控制、参数稳定性及个体差异的验证任务。</p>',
        '<h2>新增被试的主比较</h2><p>各列对6人等权平均。负对数评分、energy score与CRPS越低越好；覆盖率对应固定参数的自主路径分布。每组64条路径。</p>', table(mean),
        '<h2>逐题选择评分的两种汇总</h2><p>被试等权与全部试次求和回答不同加权问题。MCSE衡量滤波数值随机性，未包含参数估计及被试总体抽样误差。</p>', table(choices),
        '<h2>自主生成改善的数值精度</h2><p>差值为各结构减B；负值表示分数降低。每次删除整条自主路径，以共同种子配对计算jackknife标准误。各人的参数点保持固定。</p>', table(uncertainty),
        '<h2>达标停止与固定刺激日程</h2><p>原实验每64题检查一次：排除模糊刺激后，各类别精确正确率均达到0.9时结束；继续学习时根据表现调整下一块的类别比例。6名新增被试最后一块全部达标。当前自主路径沿实际刺激顺序和题数运行，完整实验中的学习时长与停止分布还需要联合生成刺激安排和达标事件。</p>',
        '<p>下表为64条路径中在记录范围内至少一次达标的条数。固定日程上的低达标率保留为诊断；末期曲线偏差中参数、机制和结束方式的贡献仍需分离。</p>',
        table(criterion.pivot(index='subject', columns='model', values='simulated_any_pass_count').reset_index()),
        '<h2>参数恢复与敏感性</h2>',
        f'<p>24条合成序列的八点保持率恢复中，{int((recovery_error == 0).sum())}条恢复到生成网格值，平均绝对误差为{recovery_error.mean():.4f}。全部其他参数固定为生成真值。</p>',
        f'<p>同一批序列仅比较0.6与0.95，增加至256粒子、8个新种子后，{int(pairs.correct.sum())}/{len(pairs)}条选择生成值；{int(strong.sum())}条的分数差超过两倍模拟标准误。该结果衡量两个指定值的区分能力；完整参数恢复与跨结构恢复仍待验证。</p>',
        table(pairs[['subject', 'true_retention', 'selected_retention', 'total_difference', 'total_mc_se', 'correct', 'exact_tie']]),
        '<details><summary>各参数剖面相对冻结点的最低训练分数</summary><p>其余参数固定，以64粒子、4个共同种子计算；负值表示当前坐标范围内出现更低训练分数。该检查用于诊断搜索与数值稳定性，正式留出参数保持原先冻结值。</p>' + table(profile_minima) + '</details>',
        '<h2>训练搜索的独立复核</h2>',
        f'<p>从每组训练剖面选择最低分数点，与原冻结点使用256粒子、8个新种子重评。{int(optimizer.lower_by_two_mc_se.sum())}/{len(optimizer)}组的替代点训练分数降低超过两倍模拟标准误。该复核在主留出结果读取后安排，替代点选择及重评仅使用训练序列；正式留出参数保持冻结。</p>',
        table(optimizer[['subject', 'model', 'anchor_nll', 'alternative_nll', 'total_delta', 'total_mc_se', 'lower_by_two_mc_se']]),
        '<h2>完整报告与复现资料</h2><ul>' + ''.join(f'<li><a href="{rel(path)}">{escape(title)}</a></li>' for title, path in links) + '</ul>',
        '<p>研究结果中保留全部候选、随机种子、源码校验值及环境记录。已记录的高精度训练复核平均使用约122.8个CPU核，每进程一个数值线程，128个工作进程。开发图形采用Python，PNG格式。</p>',
        '<p>结构、候选参数及指标在留出评分前冻结。后续由这些结果提出的新机制，需要新的验证资料。口述未用于本轮拟合或机制选择；完整联合参数恢复、模型恢复及独立口述验证尚未完成。</p></html>',
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'index.html').open('x') as handle:
        handle.write('\n'.join(body))
    with (output / 'report_manifest.json').open('x') as handle:
        json.dump(dict(status='complete', config_sha256=digest(config), source_sha256=digest(Path(__file__)),
                       inputs_sha256={str(path): digest(path) for path in files},
                       linked_reports=[str(path) for _, path in links],
                       nomination_sha256=nomination_sha,
                       interpretation='working_candidate_selected; final_mechanism_and_individual_parameter_interpretation_unresolved'), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.output)


if __name__ == '__main__':
    main()
