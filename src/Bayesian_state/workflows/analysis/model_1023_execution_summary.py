"""Audit frozen execution fits and compare saved conditional/autonomous behavior.

Use --study FIT --output NEW. This reads all nine development participants,
all saved filters and all autonomous paths. Differences are relative to MIX;
Monte Carlo standard errors describe simulation uncertainty only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_1022_belief_summary import read, digest, arrays, load_case, nll, jackknife_se
from .model_1023_execution_study import write, KAPPA
from ...metrics.autonomous_fit import behavior_features, energy_u


def trajectory_diagnostics(choices: np.ndarray, categories: np.ndarray, jump: int = 32,
                           low_window: int = 128, low_stride: int = 64,
                           n_categories: int | None = None) -> dict:
    """Apply identical whole-path summaries to observed and generated choices."""
    correct = np.asarray(choices) == np.asarray(categories)[None, :]
    if correct.ndim != 2 or correct.shape[1] < max(2 * jump, low_window):
        raise ValueError('Insufficient complete trials for registered path diagnostics')
    cumulative = np.pad(np.cumsum(correct, axis=1), ((0, 0), (1, 0)))
    centers = np.arange(jump, correct.shape[1] - jump + 1)
    gains = (cumulative[:, centers+jump] - 2*cumulative[:, centers] + cumulative[:, centers-jump]) / jump
    best = gains.argmax(axis=1)
    # Assess a complete second window after the improvement window. Peaks
    # near the end are right-censored for this maintenance statistic.
    maintenance_start = centers[best] + jump
    available = maintenance_start + jump <= correct.shape[1]
    maintenance = np.full(len(correct), np.nan)
    use = np.flatnonzero(available)
    maintenance[use] = (cumulative[use, maintenance_start[use]+jump] -
                        cumulative[use, maintenance_start[use]]) / jump
    starts = np.arange(0, correct.shape[1] - low_window + 1, low_stride)
    low = (cumulative[:, starts+low_window] - cumulative[:, starts]) / low_window
    worst_start = starts[low.argmin(axis=1)]
    indices = worst_start[:, None] + np.arange(low_window)[None, :]
    window_choices = np.take_along_axis(np.asarray(choices), indices, axis=1)
    window_categories = np.asarray(categories)[indices]
    count = n_categories or int(max(np.max(choices), np.max(categories)))
    # The product of observed label/response marginals describes accuracy
    # under their independence in this same selected window. It is a
    # descriptive bias check, not a new significance test after selection.
    marginal_chance = sum((window_choices == c).mean(axis=1) *
                         (window_categories == c).mean(axis=1) for c in range(1, count+1))
    return dict(max_adjacent_gain=gains[np.arange(len(correct)), best],
        jump_after_trial=centers[best], minimum_window_accuracy=low.min(axis=1),
        minimum_window_start_trial=worst_start+1,
        minimum_window_marginal_chance=marginal_chance,
        minimum_window_accuracy_minus_marginal_chance=low.min(axis=1)-marginal_chance,
        maintenance_accuracy=maintenance, maintenance_observed=available,
        full_accuracy=correct.mean(axis=1))


def execution_audits(score: Path, mode: str) -> list[dict]:
    """Verify each saved filter actually exercised its declared mechanisms."""
    metadata = read(score.with_suffix('.json'))
    rows = []
    for stem in metadata['chunk_files']:
        chunk = Path(stem)
        own = read(chunk.with_suffix('.json'))
        file = chunk.with_suffix('.mechanism.json')
        record = read(file)
        audit = record['audit']
        expected = own['trials'] * own['particles'] * len(own['seeds'])
        execution = audit['error_entry']['execution']
        outcome = audit['outcome_retention']
        if (audit['transitions'] != expected or audit['feedback_updates'] != expected
                or outcome['updates'] != expected
                or sum(outcome[k] for k in ('error', 'partial', 'correct')) != expected
                or execution['trials'] != (0 if mode == 'MIX' else expected)
                or execution['switches'] != execution['forced_switches']+execution['voluntary_switches']
                or not 0 <= execution['switches'] <= execution['trials']
                or record['oral_used_for_fitting']):
            raise ValueError('Saved execution/feedback audit does not match the declared filter')
        rows.append(dict(subject=own['subject'], model=mode, candidate=own['candidate']['id'],
            expected_particle_trials=expected, **execution, audit_sha256=digest(file)))
    return rows


def zero_inertia_controls(study: Path, stage: str) -> list[dict]:
    """Compare saved, paired evaluations of analytically equivalent models."""
    completion = study / (stage + '_completion.json')
    completed = read(completion)
    if completed['status'] != 'complete' or completed['stage'] != stage:
        raise ValueError('Numerical controls require a complete scoring stage')
    groups = {(g['subject'], g['model']): g['rows'] for g in completed['groups']}
    subjects = sorted({sid for sid, _ in groups})
    rows = []
    for sid in subjects:
        mixtures = {row['id']: row for row in groups[sid, 'MIX']}
        pairs = [row for row in groups[sid, 'STICKY']
                 if row['hyperparams'][KAPPA] == 0 and row['id'] in mixtures]
        if not pairs:
            raise ValueError('Common zero-maintenance anchor missing')
        for sticky in pairs:
            mixture = mixtures[sticky['id']]
            if mixture['hyperparams'] != sticky['hyperparams']:
                raise ValueError('Matched candidate identity has different parameters')
            paths = [Path(row['score_file']) for row in (mixture, sticky)]
            metadata = [read(path) for path in paths]
            for path, meta, mode in zip(paths, metadata, ('MIX', 'STICKY')):
                if meta['candidate']['hyperparams'] != sticky['hyperparams'] or meta['subject'] != sid:
                    raise ValueError('Saved control metadata differs from the selected candidate')
                execution_audits(path, mode)
            if metadata[0]['particles'] != metadata[1]['particles']:
                raise ValueError('Paired numerical controls use different particle counts')
            p, q = [arrays(path) for path in paths]
            np.testing.assert_array_equal(p['seeds'], q['seeds'])
            p, q = p['observed_probability'], q['observed_probability']
            if p.shape != q.shape or len(p) < 2:
                raise ValueError('Paired controls require aligned trials and repeated filters')
            deleted = np.array([nll(np.delete(q, i, axis=0))-nll(np.delete(p, i, axis=0))
                                for i in range(len(p))])
            rows.append(dict(subject=sid, stage=stage, candidate=sticky['id'],
                particles=metadata[0]['particles'], seeds=len(p), trials=p.shape[1],
                mixture_nll=nll(p), zero_maintenance_nll=nll(q), difference=nll(q)-nll(p),
                difference_mcse=jackknife_se(deleted), theoretical_difference=0.,
                maximum_mean_probability_difference=float(np.abs(q.mean(axis=0)-p.mean(axis=0)).max()),
                mixture_score_sha256=digest(paths[0]), sticky_score_sha256=digest(paths[1]),
                stage_completion_sha256=digest(completion)))
    return rows


def run(study: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new execution summary destination')
    completed = read(study / 'completion.json')
    protocol = read(study / 'protocol.json')
    frozen = read(study / 'nominees.json')
    if (completed['status'] != 'complete' or completed['smoke'] or frozen['later_choices_used']
            or frozen['status'] != 'frozen_before_test'
            or completed['nominees_sha256'] != digest(study / 'nominees.json')):
        raise ValueError('Require complete, frozen numerical comparisons')
    jobs = read(study / 'autonomous_jobs.json')['jobs']
    cases, energy, rows, diagnostics, audits = {}, {}, [], [], []
    for nominee in frozen['nominees']:
        sid, mode = nominee['subject'], nominee['arm']
        case = cases[sid, mode] = load_case(study, nominee, jobs)
        audits.extend(execution_audits(study / 'evaluate' / f'subject_{sid}' /
                                      mode / nominee['candidate']['id'], mode))
        meta, observed, prediction, auto = [case[k] for k in ('meta', 'observed', 'prediction', 'autonomous')]
        cut = meta['training_trials']; categories = observed['categories']
        cat_count = 2 if meta['condition'] == 1 else 4
        features = behavior_features(auto['choices'][:, cut:], categories[cut:], n_categories=cat_count)
        empirical = behavior_features(observed['choices'][None, cut:], categories[cut:], n_categories=cat_count)[0]
        score = energy_u(features, empirical)
        energy[sid, mode] = features, empirical
        probability = prediction['observed_probability'][:, cut:]
        row = dict(subject=sid, condition=meta['condition'], model=mode, candidate=nominee['candidate']['id'],
            kappa=nominee['candidate']['hyperparams'][KAPPA], training_trials=cut, test_trials=meta['test_trials'],
            filter_seeds=len(probability), autonomous_paths=len(auto['choices']),
            train_nll=nll(prediction['observed_probability'][:, :cut]), test_nll=nll(probability),
            test_nll_mcse=jackknife_se(np.array([nll(np.delete(probability, i, axis=0)) for i in range(len(probability))])),
            observed_test_accuracy=float((observed['choices'][cut:] == categories[cut:]).mean()),
            generated_test_accuracy=float((auto['choices'][:, cut:] == categories[None, cut:]).mean()),
            observed_test_reward=float(observed['feedback'][cut:].mean()),
            generated_test_reward=float(auto['feedback'][:, cut:].mean()),
            observed_full_reward=float(observed['feedback'].mean()),
            generated_full_reward=float(auto['feedback'].mean()),
            test_energy=score['value'], test_energy_mcse=score['mc_se'])
        observed_d = trajectory_diagnostics(observed['choices'][None], categories, n_categories=cat_count)
        generated_d = trajectory_diagnostics(auto['choices'], categories, n_categories=cat_count)
        for name in generated_d:
            values = np.asarray(generated_d[name], dtype=float)
            valid = values[np.isfinite(values)]
            row['observed_' + name] = float(observed_d[name][0])
            row['generated_' + name + '_available_paths'] = len(valid)
            row['generated_' + name + '_median'] = float(np.median(valid)) if len(valid) else np.nan
            row['generated_' + name + '_p10'] = float(np.quantile(valid, .1)) if len(valid) else np.nan
            row['generated_' + name + '_p90'] = float(np.quantile(valid, .9)) if len(valid) else np.nan
        for index, seed in enumerate(auto['seeds']):
            diagnostics.append(dict(subject=sid, model=mode, seed=int(seed),
                mean_reward=float(auto['feedback'][index].mean()),
                **{k: float(v[index]) for k, v in generated_d.items()}))
        rows.append(row)
    comparisons = []
    for sid, mode in sorted(cases):
        if mode == 'MIX':
            continue
        baseline, comparison = cases[sid, 'MIX'], cases[sid, mode]
        cut = baseline['meta']['training_trials']
        for source in ('prediction', 'autonomous'):
            np.testing.assert_array_equal(baseline[source]['seeds'], comparison[source]['seeds'])
        p, q = [c['prediction']['observed_probability'][:, cut:] for c in (baseline, comparison)]
        deleted = np.array([nll(np.delete(q, i, axis=0)) - nll(np.delete(p, i, axis=0)) for i in range(len(p))])
        a, observed = energy[sid, 'MIX']; b, _ = energy[sid, mode]
        energy_deleted = np.array([energy_u(np.delete(b, i, axis=0), observed)['value']
                                  - energy_u(np.delete(a, i, axis=0), observed)['value'] for i in range(len(a))])
        comparisons.append(dict(subject=sid, model=mode, test_nll_difference=nll(q)-nll(p),
            test_nll_difference_mcse=jackknife_se(deleted),
            test_energy_difference=energy_u(b, observed)['value']-energy_u(a, observed)['value'],
            test_energy_difference_mcse=jackknife_se(energy_deleted)))
    output.mkdir(parents=True)
    table, comparison_table = pd.DataFrame(rows), pd.DataFrame(comparisons)
    table.to_csv(output / 'summary.csv', index=False)
    comparison_table.to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output / 'autonomous_path_diagnostics.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'execution_filter_audits.csv', index=False)
    controls = [row for stage in ('screen', 'confirmation') for row in zero_inertia_controls(study, stage)]
    pd.DataFrame(controls).to_csv(output / 'zero_inertia_numerical_controls.csv', index=False)
    lines = ['# Model 1023 执行机制：开发样本比较', '',
        '各模型分别使用每位被试前约70%的选择拟合；参数冻结后评价后段选择，并从起点自主生成完整路径。',
        '全部九人为开发案例。差值按“候选模型减MIX”计算，负值表示候选模型评分更好。',
        'MCSE表示有限滤波或生成次数造成的蒙特卡洛标准误，个体差异及抽样不确定性需另行评价。', '',
        '| 被试 | 模型 | 后段NLL差值 ± MCSE | 后段Energy差值 ± MCSE |',
        '|---|---|---:|---:|']
    for row in comparisons:
        lines.append(f'| {row["subject"]} | {row["model"]} | {row["test_nll_difference"]:.4f} ± '
                     f'{row["test_nll_difference_mcse"]:.4f} | {row["test_energy_difference"]:.4f} ± '
                     f'{row["test_energy_difference_mcse"]:.4f} |')
    lines += ['', 'NLL评价逐题条件选择概率。Energy评价完整自主路径的32题正确率和类别条件反应特征分布。',
        '额外路径指标使用相邻32题窗口和128题低表现窗口，包含全部生成路径。',
        '提升后的保持以第二个32题窗口的正确率表示；结束前不足完整窗口时记录缺测及可评价路径数。',
        '完整正确率与平均反馈得分分别保存；条件3的半分反馈计入后者。',
        '最低正确率窗口另列类别比例与反应比例独立时的期望正确率，用于检查类别不平衡和反应偏好；该量为描述性对照。',
        '零维持参数与混合预测的配对数值对照单独保存，包含初筛和高精度确认中所有同参数、同种子的匹配点。',
        '这些点的理论差值为零；观察到的差值用于评估有限粒子近似对机制比较的影响。',
        '129、229、317的主要后期改善位于参数拟合段之后；本比较评价前段参数对后段行为的解释范围。',
        '机制归因还需结合参数、模型和状态恢复结果，以及选择后、反馈前的口述检验。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    write(output / 'completion.json', dict(status='complete', study=str(study.resolve()),
        source_sha256=digest(Path(__file__)), nominees_sha256=digest(study / 'nominees.json'),
        policy=protocol['policy'], cases=len(rows), later_choices_used_for_parameter_selection=False,
        uncertainty='Monte Carlo only; exploratory development cohort',
        artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    print(comparison_table.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
