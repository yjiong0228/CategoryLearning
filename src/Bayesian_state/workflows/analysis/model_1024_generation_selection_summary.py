"""Audit and report frozen constrained-generation selection on all nine cases.

Use --evaluation COMPLETED --output NEW. Save full-path metrics and paired
differences, including unchanged estimates and failed training proposals.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from .model_1024_generation_selection import filter_arrays, sources, write
from .model_1024_trajectory_diagnosis import checked_new_paths, summarize_paths, probability_record
from .model_1022_belief_summary import digest, jackknife_se, nll, read
from .model_1022_belief_energy_guard import paired_energy_guard
from ..reports.model_1024_trajectory_report import table
from ...metrics.autonomous_fit import behavior_features, energy_u

ROOT = Path(__file__).resolve().parents[4]


def pareto_mask(nll_values: np.ndarray, energy_values: np.ndarray) -> np.ndarray:
    """Keep points not strictly dominated in both recorded training objectives."""
    values = np.column_stack([nll_values, energy_values])
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Need finite candidate objectives')
    return np.array([not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
                     for row in values])


def run(evaluation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new summary directory')
    done, protocol = read(evaluation/'completion.json'), read(evaluation/'protocol.json')
    if done['status'] != 'complete' or done['protocol_sha256'] != digest(evaluation/'protocol.json'):
        raise ValueError('Require completed, intact evaluation')
    training = Path(protocol['training'])
    frozen = read(training/'nominees.json')
    if (frozen['smoke'] or frozen['later_choices_used'] or protocol['selection_sources'] != sources()
            or digest(training/'nominees.json') != protocol['frozen_nominees_sha256']):
        raise ValueError('Frozen selection or model changed')
    p = protocol['policy']
    diagnostic_file = ROOT/'configs/shared/model1024_trajectory_diagnosis.yaml'
    diagnostic = yaml.safe_load(diagnostic_file.read_text())
    output.mkdir(parents=True)
    (output/'curves').mkdir()
    parameter_changes = []
    for row in frozen['nominees']:
        for name, old in row['original']['hyperparams'].items():
            new = row['candidate']['hyperparams'][name]
            if new != old:
                parameter_changes.append(dict(subject=row['subject'], mode=row['mode'], parameter=name,
                                              original=old, selected=new))
    pd.DataFrame(parameter_changes, columns=['subject', 'mode', 'parameter', 'original', 'selected']).to_csv(
        output/'parameter_changes.csv', index=False)
    results, path_rows, events, comparison, observed_rows = [], [], [], [], []
    ranking = pd.read_csv(training/'ranking_scores.csv')
    ranking['pareto'] = False
    for _, group in ranking.groupby(['subject', 'mode']):
        ranking.loc[group.index, 'pareto'] = pareto_mask(group.confirmation_nll.to_numpy(), group.training_energy.to_numpy())
    ranking.to_csv(output/'candidate_frontier.csv', index=False)
    for nominee in frozen['nominees']:
        sid, mode = nominee['subject'], nominee['mode']
        cases = [c for c in protocol['cases'] if c['subject'] == sid and c['mode'] == mode]
        folder = Path(cases[0]['folder'])
        meta = read(folder/'prepared.json')
        parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            if digest(folder/name) != meta['array_sha256'][name]:
                raise ValueError('Behavioral data changed after freezing')
            with np.load(folder/name) as z:
                parts.append({k: z[k] for k in z.files})
        obs = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
        cut, ncat = meta['training_trials'], 2 if meta['condition'] == 1 else 4
        geometry_file = ROOT/p['geometry']/f'subject_{sid}'/'audit.json'
        targets = read(geometry_file)['training_perfect_rules']
        bank = {}
        for case in cases:
            point = case['candidate']
            pred = filter_arrays(case['filter_jobs'])
            probabilities = pred['probabilities']
            if probabilities.shape != (p['evaluate']['seeds'], meta['trials'], ncat):
                raise ValueError('Evaluation probability dimensions changed')
            np.testing.assert_allclose(probabilities.sum(axis=2), 1., atol=1e-12, rtol=0)
            np.testing.assert_allclose(pred['observed_probability'],
                probabilities[:, np.arange(meta['trials']), obs['choices']-1], atol=1e-12, rtol=0)
            auto = checked_new_paths(case['autonomous_jobs'], dict(subject=sid, mode=mode, candidate=point), obs['categories'])
            if len(auto['choices']) != p['evaluation_paths']:
                raise ValueError('Missing full autonomous paths')
            sections = [('full', 0, None), ('training', 0, cut), ('late', cut, None)]
            features = {label: behavior_features(auto['choices'][:, start:stop], obs['categories'][start:stop],
                n_categories=ncat, **p['features']) for label, start, stop in sections}
            empirical = {label: behavior_features(obs['choices'][None, start:stop], obs['categories'][start:stop],
                n_categories=ncat, **p['features'])[0] for label, start, stop in sections}
            metrics, truth, event = summarize_paths(auto, obs, cut, ncat, targets, mode, diagnostic)
            point_result = dict(subject=sid, mode=mode, candidate=point['id'], condition=meta['condition'],
                observed_late_accuracy=truth['late_accuracy'], observed_late_feedback=float(obs['feedback'][cut:].mean()),
                training_nll=nll(pred['observed_probability'][:, :cut]),
                test_nll=nll(pred['observed_probability'][:, cut:]), paths=len(auto['choices']),
                **{k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in metrics.items()})
            for label in features:
                score = energy_u(features[label], empirical[label])
                point_result.update({label+'_energy': score['value'], label+'_energy_mcse': score['mc_se']})
            bank[point['id']] = dict(result=point_result, pred=pred, auto=auto, metrics=metrics,
                                     features=features, empirical=empirical, events=event)
        a, b = [bank[c['id']] for c in (nominee['original'], nominee['candidate'])]
        np.testing.assert_array_equal(a['pred']['seeds'], b['pred']['seeds'])
        np.testing.assert_array_equal(a['auto']['seeds'], b['auto']['seeds'])
        delta = dict(subject=sid, mode=mode, changed=nominee['original']['id'] != nominee['candidate']['id'])
        for label, start, stop in [('training_nll', 0, cut), ('test_nll', cut, None)]:
            old, new = [x['pred']['observed_probability'][:, start:stop] for x in (a, b)]
            deletes = np.array([nll(np.delete(new, i, 0))-nll(np.delete(old, i, 0)) for i in range(len(old))])
            delta.update({label+'_delta': nll(new)-nll(old), label+'_delta_mcse': jackknife_se(deletes)})
        for label in ('training', 'full', 'late'):
            record = paired_energy_guard(a['features'][label], b['features'][label], a['empirical'][label], p['mcse_multiplier'])
            delta.update({label+'_energy_delta': record['energy_difference'], label+'_energy_delta_mcse': record['difference_mcse']})
        for key in ('late_accuracy', 'late_target_present', 'late_target_contribution'):
            values = b['metrics'][key]-a['metrics'][key]
            delta.update({key+'_delta': float(values.mean()), key+'_delta_mcse': float(values.std(ddof=1)/np.sqrt(len(values)))})
        comparison.append(delta)
        if mode == 'MIX':
            observed_rows.append(dict(subject=sid, training_trials=cut, trials=meta['trials'], **truth))
        for role, case in [('NLL', a), ('JOINT', b)]:
            ident = dict(subject=sid, mode=mode, estimator=role, candidate=case['result']['candidate'])
            results.append(dict(case['result'], estimator=role))
            for i, seed in enumerate(case['auto']['seeds']):
                path_rows.append(dict(**ident, seed=int(seed), **{k: v[i].item() for k, v in case['metrics'].items()},
                                      **{k: bool(v[i]) for k, v in case['events'].items()}))
            for name, values in case['events'].items():
                events.append(dict(**ident, event=name, **probability_record(values)))
            block = p['features']['block_width']
            correct = case['auto']['choices'] == obs['categories'][None, :]
            curves = np.stack([correct[:, t:t+block].mean(axis=1) for t in range(0, meta['trials'], block)], axis=1)
            curve = np.array([(obs['choices'][t:t+block] == obs['categories'][t:t+block]).mean()
                              for t in range(0, meta['trials'], block)])
            np.savez_compressed(output/'curves'/f'S{sid}_{mode}_{role}.npz', curves=curves, observed=curve,
                end_trial=np.minimum(np.arange(1, len(curve)+1)*block, meta['trials']), cut=cut)
        print(f'S{sid} {mode}: paired frozen evaluation audited', flush=True)
    frames = dict(summary=results, paths=path_rows, events=events, comparisons=comparison, observed=observed_rows)
    for name, values in frames.items():
        pd.DataFrame(values).to_csv(output/(name+'.csv'), index=False)
    decisions = read(training/'decisions.json')['decisions']
    flat = [dict(subject=d['subject'], mode=d['mode'], reason=d['reason'], replaced=d['replaced'],
        original=d['original'], proposed=d['proposed'], selected=d['selected'],
        nll_delta=d['likelihood'].get('nll_difference', np.nan),
        nll_mcse=d['likelihood'].get('difference_mcse', np.nan), nll_upper=d['likelihood'].get('guard_value', np.nan),
        energy_delta=d['generation'].get('energy_difference', np.nan),
        energy_mcse=d['generation'].get('difference_mcse', np.nan), energy_upper=d['generation'].get('guard_value', np.nan)) for d in decisions]
    pd.DataFrame(flat).to_csv(output/'selection_decisions.csv', index=False)
    write(output/'completion.json', dict(status='complete', evaluation=str(evaluation.resolve()),
        training=str(training.resolve()), policy=p, groups=len(comparison), changed=sum(c['changed'] for c in comparison),
        unique_autonomous_paths=done['paths'], described_paths=len(path_rows),
        source_sha256=digest(Path(__file__)), diagnostic_config_sha256=digest(diagnostic_file),
        freeze_sha256=digest(training/'nominees.json'), artifacts_sha256={str(f.relative_to(output)): digest(f)
        for f in output.rglob('*') if f.is_file()}))


def report(summary: Path, figures: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new report directory')
    done = read(summary/'completion.json')
    for name, sha in done['artifacts_sha256'].items():
        if digest(summary/name) != sha:
            raise ValueError('Audited summary changed')
    training = Path(done['training'])
    complete_train = read(training/'completion.json')
    choices = pd.read_csv(summary/'selection_decisions.csv')
    changes = pd.read_csv(summary/'comparisons.csv')
    values = pd.read_csv(summary/'summary.csv')
    events = pd.read_csv(summary/'events.csv')
    paths = pd.read_csv(summary/'paths.csv')
    observed = pd.read_csv(summary/'observed.csv').set_index('subject')
    output.mkdir(parents=True)
    aggregates = changes.groupby('mode', sort=False).agg(
        cases=('subject', 'size'), replaced=('changed', 'sum'),
        mean_suffix_nll_delta=('test_nll_delta', 'mean'),
        mean_suffix_energy_delta=('late_energy_delta', 'mean'),
        mean_full_energy_delta=('full_energy_delta', 'mean'),
        mean_suffix_accuracy_delta=('late_accuracy_delta', 'mean')).reset_index()
    aggregates.to_csv(output/'outcome_by_mode.csv', index=False)
    error_rows = []
    for key, group in paths[paths['mode'] != 'MIX'].groupby(['subject', 'mode', 'estimator']):
        total = float(group.total_errors.sum())
        counts = [float(group[name].sum()) for name in
                  ('error_absent', 'error_present_other', 'error_correct_executed')]
        np.testing.assert_allclose(sum(counts), total, atol=0, rtol=0)
        weights = group.trials_correct_executed
        error_rows.append(dict(subject=key[0], mode=key[1], estimator=key[2], errors=int(total),
            absent_fraction=counts[0]/total if total else np.nan,
            other_execution_fraction=counts[1]/total if total else np.nan,
            correct_execution_fraction=counts[2]/total if total else np.nan,
            pooled_correct_probability_when_correct_executed=float(
                (group.correct_probability_correct_executed.fillna(0)*weights).sum()/weights.sum())
                if weights.sum() else np.nan))
    error_table = pd.DataFrame(error_rows)
    error_table.to_csv(output/'generating_error_sources.csv', index=False)
    replaced = changes[changes.changed]
    energy_better = int((replaced.late_energy_delta < 0).sum())
    energy_worse = int((replaced.late_energy_delta > 0).sum())
    nll_better = int((replaced.test_nll_delta < 0).sum())
    nll_worse = int((replaced.test_nll_delta > 0).sum())
    reasons = dict(both_guards_pass='两项确认通过', energy_guard_fail='生成改善未确认',
        likelihood_guard_fail='似然条件未通过', likelihood_and_energy_guard_fail='两项确认均未通过',
        original_is_energy_leader='原点已是生成最优')
    choices_display = choices[['subject', 'mode', 'replaced', 'reason', 'nll_delta', 'nll_mcse',
        'nll_upper', 'energy_delta', 'energy_mcse', 'energy_upper']].replace({'reason': reasons})
    choices_display = choices_display.rename(columns=dict(subject='被试', mode='机制', replaced='更换', reason='确认结果',
        nll_delta='NLL差', nll_mcse='NLL标准误', nll_upper='NLL上界',
        energy_delta='Energy差', energy_mcse='Energy标准误', energy_upper='Energy上界'))
    comparison_columns = ['subject', 'mode', 'changed', 'test_nll_delta', 'test_nll_delta_mcse',
        'late_energy_delta', 'late_energy_delta_mcse', 'full_energy_delta', 'full_energy_delta_mcse',
        'late_accuracy_delta', 'late_accuracy_delta_mcse']
    comparison_display = changes[comparison_columns].rename(columns=dict(subject='被试', mode='机制', changed='更换',
        test_nll_delta='后段NLL差', test_nll_delta_mcse='NLL标准误', late_energy_delta='后段Energy差',
        late_energy_delta_mcse='后段E标准误', full_energy_delta='全程Energy差', full_energy_delta_mcse='全程E标准误',
        late_accuracy_delta='后段正确率差', late_accuracy_delta_mcse='正确率标准误'))
    critical = events[(events.subject.isin([129, 229])) & (events.estimator == 'JOINT') &
                      (events.event == 'jump_and_maintenance')]
    shape_statement = ('129、229的最终估计在三种机制下均为0/128，预定最大陡升与保持的联合形态仍处于预测尾部。'
        if (critical['count'] == 0).all() else
        '129、229的最大陡升与保持联合形态在部分最终估计中出现，具体发生数和区间见下表。')
    link = lambda p: os.path.relpath(p, output)
    lines = ['# Model 1024：训练似然与自主分布共同选参检验', '',
        '完成日期：2026-10-25。范围：九名exp123开发被试。MIX依据规则信念加权作答，'
        'MAP执行最高信念规则，STICKY每题执行一条规则并具有跨题维持倾向。', '',
        f'本轮完成27组比较；{done["changed"]}组通过独立训练NLL与Energy确认，采用新的参数点。'
        '所有提议、未通过原因和原NLL估计均保留。认知机制沿用当前信念框架，变化来自参数估计规则。', '',
        f'更换的组中，后段Energy点估计改善{energy_better}组、变差{energy_worse}组；'
        f'后段NLL改善{nll_better}组、变差{nll_worse}组。这些计数描述差值方向，'
        '各差值的模拟误差见配对表。', '',
        '## 预定规则与实际预算', '',
        'NLL为每题负对数似然，Energy衡量完整自主路径的块正确率及类别条件反应频率分布，二者均以较低值为优。'
        '先在原训练NLL差不超过0.02的候选中，以64条新训练路径排序；每组提议一个Energy最优点。'
        '原点与提议点再接受128条新训练路径及六次512粒子过滤。只有NLL差加两倍MCSE不超过0.02、'
        '且Energy差加两倍MCSE小于0，才采用提议点。相同提名直接复用。MCSE表示蒙特卡洛标准误。', '',
        f'实际排序路径{complete_train["ranking_paths"]}条，确认路径{complete_train["confirmation_paths"]}条，'
        f'训练过滤{complete_train["filter_jobs"]}个独立种子任务；冻结后生成{done["unique_autonomous_paths"]}条完整路径。', '',
        '训练工作进程禁止打开后段数组。排序、两项确认和冻结后评价使用不同种子角色。'
        '参数在后段评价之前冻结。原候选已使用完整训练段开发，独立种子检验的是计算稳定性；'
        '这些真人后段此前已被查看，继续作为开发评价。选择规则沿用既有约束方案，两项训练分数来自同一记录。', '',
        f'协议：[固定规则与资料边界]({link(ROOT/"src/Bayesian_state/docs/model_architecture/model_1024_generation_selection.md")})。', '',
        '## 训练决定', '', table(choices_display), '',
        '新减原为差值方向。NLL上界小于等于0.02、Energy上界小于0是两个独立数值确认条件。'
        '原点已经是排序最优点的行保留空确认项，作为同点复用。全部合格点及二维分数前沿见`candidate_frontier.csv`。'
        '分数前沿保留没有被另一个候选同时改善NLL和Energy的点，其标记不包含参数概率权重。'
        '每个实际变化的参数及其原值、新值保存在`parameter_changes.csv`；参数点标识保留在冻结清单和决定表中。', '',
        '## 冻结后的配对评价', '',
        'NLL与JOINT分别表示原纯似然估计和本轮最终估计。每组两个估计采用共同的新种子及相同数值预算；'
        '同一点复用同一模拟。条件预测在本题选择前形成，之后用实际选择与反馈更新状态；'
        '自主生成沿自己的选择和反馈更新。', '',
        table(aggregates.rename(columns=dict(mode='机制', cases='被试数', replaced='更换数',
            mean_suffix_nll_delta='平均后段NLL差', mean_suffix_energy_delta='平均后段Energy差',
            mean_full_energy_delta='平均全程Energy差', mean_suffix_accuracy_delta='平均后段正确率差'))), '',
        '上表每种机制的九名被试等权，包括参数未变的组；九人均为开发个案，三个机制共享这些被试。', '',
        table(comparison_display), '',
        '后段NLL和Energy的负差表示改善，正确率的正差表示改善。各指标的分母和配对种子均保留。'
        '后段均值改善与完整路径、少数关键事件及内部状态准确性分别评价。', '',
        '同批冻结后自主路径的训练段Energy另作数值复核，见`comparisons.csv`中的`training_energy_delta`。'
        '它使用新的生成种子，参数继续采用已经冻结的决定。', '',
        f'![九人冻结评价]({link(figures/"selection_outcomes.png")})', '',
        '## 关键个案与内部过程', '',
        table(values[(values.subject.isin([129, 229, 317]))][['subject', 'mode', 'estimator', 'late_accuracy',
            'late_energy', 'late_target_present', 'late_target_contribution', 'late_target_precision']]), '',
        '在场率是正确规则属于活跃集合的试次比例；贡献在MIX中是作答权重，在MAP和STICKY中是执行比例。'
        '精度控制单条规则类别判断的确定程度。在场和贡献对全部生成路径平均；'
        '精度先在路径内以正确规则信念加权，再对有定义的路径平均。'
        '多参数同时变化时，过程指标提供后续隔离检验的线索。模拟状态是已知生成状态，真人状态推断准确性另行验收。', '',
        table(error_table[(error_table.subject.isin([129, 229, 317])) & (error_table['mode'] == 'STICKY')]), '',
        '上表分解全序列生成错误：正确规则缺席、在场但执行其他规则、执行正确规则仍答错，三类比例相加为1。'
        '最后一列按相应试次数加权，给出执行正确规则时的种类正确概率。条件3的未知配对也会影响该概率，'
        '解释时同时考虑配对学习和精度。MIX的离散执行身份记为不适用。', '',
        table(events[((events.subject.isin([129, 229])) & (events.event == 'jump_and_maintenance')) |
                     ((events.subject == 317) & (events.event == 'low_and_bias_adjusted'))][
                         ['subject', 'mode', 'estimator', 'event', 'count', 'paths', 'low', 'high']]), '',
        '陡升—保持事件沿用具有完整随访的最大相邻32题改善及其第二个窗口保持；低表现事件同时约束最低128题'
        '正确率及其相对类别—反应边际独立基准的差值。真人参照和窗口规则沿用上一轮。0/128保留小概率不确定性，'
        '单点95% Wilson区间上界约2.91%。这些事件比例是开发个案描述。', '',
        f'129的改善与保持参照分别为{observed.loc[129, "complete_gain"]:.4f}和'
        f'{observed.loc[129, "complete_maintenance"]:.4f}；229分别为'
        f'{observed.loc[229, "complete_gain"]:.4f}和{observed.loc[229, "complete_maintenance"]:.4f}。'
        f'317的最低窗口正确率为{observed.loc[317, "minimum_window_accuracy"]:.4f}，'
        f'该窗口的类别—反应边际独立基准为{observed.loc[317, "minimum_window_marginal_chance"]:.4f}，'
        '均匀猜测参照为0.25。边际参照将观察到的类别比例和反应偏好纳入比较。', '',
        shape_statement, '',
        f'![重点个案完整路径]({link(figures/"selection_focus_paths.png")})', '',
        '## 证据范围与下一步', '',
        '本轮检验有限确认候选库中的选参规则。训练分布约束会改变个人参数及生成过程，后段得失按上表保留。'
        '所有候选与原点继续可追溯；候选集合和二维前沿用于敏感性解释，完整参数后验仍待建立。', '',
        '后续过程干预从共同训练前段状态出发，分别检验正确规则发现/保持与精度校准对剩余行为缺口的作用。'
        '机制修订需重新估计规定参数，并检查模拟状态恢复及口述预测。当前“正确规则在场但执行其他规则”'
        '状态恢复的识别限制继续保留。新选参规则下的参数与状态恢复尚待检验，'
        '真人内部状态解释继续保留这项验证边界。', '',
        '## 复现与产物', '',
        f'- [完整数值表、路径和曲线]({link(summary)})。',
        f'- [训练候选与冻结记录]({link(training)})。',
        '- `validation.json`记录测试、训练访问限制和完整运行核查。', '',
        '从仓库根目录使用新输出目录运行协议中的train和evaluate，再执行：', '', '```bash',
        'python -m src.Bayesian_state.workflows.analysis.model_1024_generation_selection_summary --evaluation NEW_EVALUATION --output NEW_SUMMARY',
        'python -m CategoryLearning_codes.figures.fig3.model1024_generation_selection --summary NEW_SUMMARY --output NEW_FIGURES',
        'python -m src.Bayesian_state.workflows.analysis.model_1024_generation_selection_summary --summary NEW_SUMMARY --figures NEW_FIGURES --output NEW_REPORT',
        '```', '']
    (output/'report.md').write_text('\n'.join(lines))
    write(output/'completion.json', dict(status='complete', changed=done['changed'],
        summary_sha256=digest(summary/'completion.json'), report_sha256=digest(output/'report.md'),
        source_sha256=digest(Path(__file__)), independent_empirical_validation=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'summary', 'figures', 'output'):
        parser.add_argument('--'+name, type=Path, required=name == 'output')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.evaluation:
            run(args.evaluation, args.output)
        elif args.summary and args.figures:
            report(args.summary, args.figures, args.output)
        else:
            parser.error('Use --evaluation, or --summary with --figures')
