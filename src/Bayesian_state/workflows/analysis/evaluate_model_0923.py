"""Execute versioned subject evaluation and require its complete report bundle.

Rendering is injected by the figure entry point; this workflow shares the model
and metrics implementation without depending on the journal figure package.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import inspect
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.evaluation.model_0923_fit_source import validate_fit_source
from src.Bayesian_state.evaluation.model_0923_followup import learning_summary
from src.Bayesian_state.evaluation.model_0923_postfit import resolve_subject
from src.Bayesian_state.evaluation.model_0923_protocol import (
    EXTENSIONS, PROTOCOL, PROTOCOL_V2, artifact_manifest, criterion_tables, load_protocol,
)
from src.Bayesian_state.metrics.task import category_learning_feedback, condition3_task_metric_summaries
from .analyze_model_0923_postfit import run as run_postfit, write_json


def preflight(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    settings = load_protocol(config_path, smoke=smoke)
    source = validate_fit_source(Path(settings['fit_run']))
    if not set(settings['subjects']).issubset(source['subjects']):
        raise ValueError('Requested subjects are absent from the completed source fit')
    if not smoke and source.get('smoke_only', False):
        raise ValueError('A smoke fit supports smoke evaluation only')
    spec = settings['task_criterion']
    if spec['mode'] == 'exp123':
        program = Path(spec['source_program'])
        text = program.read_text()
        threshold = re.search(r'accstop\s*=\s*([\d.]+)', text)
        size = re.search(r'nTpB\s*=\s*\[\s*\d+\s*,\s*(\d+)\s*\]', text)
        if (threshold is None or size is None or
                float(threshold[1]) != spec['accuracy_threshold'] or int(size[1]) != spec['block_size']):
            raise ValueError('Task criterion differs from the declared experiment program')
        source['input_sha256'][str(program.resolve())] = hashlib.sha256(program.read_bytes()).hexdigest()
        for sid in settings['subjects']:
            inputs = resolve_subject(Path(settings['fit_run']), sid,
                                     Path(settings['fit_run']) / 'resources_cache', 16 if smoke else None)
            if 'exp123' not in Path(inputs['paths']['learning_data']).parts:
                raise ValueError('exp123 criterion requires exp123 source data')
    if settings['protocol'] == PROTOCOL_V2 and settings['oral_alignment']['mode'] == 'predict':
        from src.Bayesian_state.evaluation.model_0923_oral_alignment import load_observed_reports
        oral_path = Path(settings['oral_alignment']['trials_path'])
        source['input_sha256'][str(oral_path)] = hashlib.sha256(oral_path.read_bytes()).hexdigest()
        for sid in settings['subjects']:
            inputs = resolve_subject(Path(settings['fit_run']), sid,
                                     Path(settings['fit_run']) / 'resources_cache', 16 if smoke else None)
            load_observed_reports(inputs, settings['oral_alignment'])
    return dict(protocol=settings['protocol'], settings=settings, fit_source=source,
                scope=('Frozen parameters; first 16 valid observed trials; workflow validation' if smoke else
                       'Frozen parameters; full observed history; in-sample prediction'), smoke=smoke)


def run(config_path: Path, output: Path | None = None, smoke: bool = False, *,
        figure_output: Path | None = None, dry_run: bool = False,
        plot_subject: Callable | None = None, plot_cohort: Callable | None = None,
        plot_diagnostics: Callable | None = None,
        plot_behavior_alignment: Callable | None = None,
        plot_oral_alignment: Callable | None = None) -> dict[str, Any]:
    """Produce every standard section, retaining unavailable extension statuses."""
    started = perf_counter()
    config_path = config_path.resolve()
    plan = preflight(config_path, smoke=smoke)
    settings, source = plan['settings'], plan['fit_source']
    output = (output or Path(settings['output_dir'])).resolve()
    figures = (figure_output or (output / 'figures' if smoke else Path(settings['figure_output_dir']))).resolve()
    if output.exists() or figures.exists():
        raise FileExistsError('Choose new evaluation and figure output directories')
    if not smoke and not figures.is_relative_to((Path(__file__).resolve().parents[4] / 'CategoryLearning_codes/figures/outputs').resolve()):
        raise ValueError('Standard draft PNGs belong under CategoryLearning_codes/figures/outputs')
    if dry_run:
        return dict(**plan, status='planned', output_dir=str(output), figure_directory=str(figures))
    callbacks = [plot_subject, plot_cohort, plot_diagnostics]
    if settings['protocol'] == PROTOCOL_V2:
        callbacks += [plot_behavior_alignment, plot_oral_alignment]
    if any(callback is None for callback in callbacks):
        raise ValueError('The v2 protocol requires behavior and oral alignment renderers' if settings['protocol'] == PROTOCOL_V2 else
                         'The v1 protocol requires all three report renderers')
    renderer_files = {Path(inspect.getsourcefile(callback)).resolve() for callback in
                      callbacks}
    source_hashes = dict(source['input_sha256'])
    source_hashes.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in renderer_files})
    postfit_settings = deepcopy(settings)
    postfit_settings['calibration_subjects'] = list(settings['subjects'])
    base = run_postfit(config_path, output, smoke, plot_subject, plot_cohort,
                      settings_override=postfit_settings, figure_output=figures)
    source_hashes.update(base.get('analysis_source_sha256', {}))
    # Completion requires every table and figure in the selected protocol.
    base['status'] = 'assembling_subject_protocol'
    write_json(output / 'manifest.json', base)
    write_json(output / 'progress.json', {'phase': 'assembling_subject_protocol'})
    evaluations = []
    fit = Path(settings['fit_run'])
    summary = pd.read_csv(output / 'subjects.csv')
    for sid in settings['subjects']:
        directory = output / f'subject_{sid}'
        inputs = resolve_subject(fit, sid, output / 'resources_cache', 16 if smoke else None)
        trials = pd.read_csv(directory / 'trials.csv')
        arrays = inputs['arrays']
        for column, values in [('choice', arrays.choices), ('category', arrays.categories), ('feedback', arrays.feedback)]:
            np.testing.assert_array_equal(trials[column].to_numpy(), values)
        with np.load(directory / 'autonomous_rollouts.npz', allow_pickle=False) as saved:
            choices, feedback, expected, seeds = [saved[key] for key in
                ('choices', 'feedback', 'expected_correct_probability', 'seeds')]
        task_feedback = np.asarray([[category_learning_feedback(int(y), int(c), condition=inputs['condition'])
            for y, c in zip(row, arrays.categories)] for row in choices])
        np.testing.assert_array_equal(task_feedback, feedback)
        segments, learning = learning_summary(arrays.feedback, feedback, expected,
            trials.predicted_correct_probability.to_numpy(), settings['segment_size'],
            settings['late_window'], settings['target_accuracy'])
        pd.DataFrame(segments).to_csv(directory / 'learning_segments.csv', index=False)
        pd.DataFrame([learning]).to_csv(directory / 'learning.csv', index=False)
        task_metrics = {'status': 'not_applicable', 'reason': 'Separate ternary-feedback metrics apply to condition 3'}
        if inputs['condition'] == 3:
            probabilities = trials[[f'probability_choice_{c}' for c in range(1, 5)]].to_numpy()
            mask = trials.scored.to_numpy(dtype=bool)
            task_metrics = condition3_task_metric_summaries(probabilities=probabilities,
                choices=arrays.choices, feedback=arrays.feedback, categories=arrays.categories, valid_trial_mask=mask)
            task_metrics['status'] = 'completed'
            task_metrics['autonomous_task_metrics'] = {}
            for name, values in [('species_accuracy', feedback[:, mask] == 1),
                                 ('family_accuracy', feedback[:, mask] > 0), ('mean_reward', feedback[:, mask])]:
                quantiles = np.quantile(values.mean(axis=1), [.1, .5, .9])
                task_metrics['autonomous_task_metrics'][name] = dict(zip(('q10', 'q50', 'q90'), quantiles.tolist()))
        write_json(directory / 'task_metrics.json', task_metrics)
        criterion, blocks, trajectories = criterion_tables(inputs['frame'], feedback, seeds,
            2 if inputs['condition'] == 1 else 4, settings['task_criterion'], smoke=smoke)
        write_json(directory / 'task_criterion.json', criterion)
        blocks.to_csv(directory / 'criterion_blocks.csv', index=False)
        trajectories.to_csv(directory / 'criterion_trajectories.csv', index=False)
        comparison = pd.read_csv(directory / 'budget_comparison.csv')
        plot_diagnostics(sid, inputs['condition'], trials, figures / f'subject_{sid}', budget_comparison=comparison)
        score_row = summary.loc[summary.subject == sid].iloc[0]
        evaluation = dict(protocol=settings['protocol'], subject=sid, condition=int(inputs['condition']),
            smoke=smoke, source_kind=source['source_kind'],
            fit_acceptance=source['subject_acceptance'][str(sid)],
            fit_issues=inputs.get('fit_issues', []),
            steps={
                'input_and_search': 'completed',
                'choice_prediction': 'completed_in_sample' if not smoke else 'smoke_only',
                'numerical_stability': 'measured_without_acceptance_gate',
                'internal_states': 'fixed_parameter_pointwise_marginals',
                'autonomous_learning': 'fixed_observed_schedule',
                'task_criterion': criterion['status'],
            },
            numerical=dict(full_minus_half_repeats_nll=float(score_row.full_minus_half_repeats_nll),
                paired_high_minus_base_nll=float(score_row.calibration_high_minus_base_nll),
                paired_mean_abs_probability_difference=float(score_row.calibration_mean_abs_probability_difference)),
            learning=learning, task_criterion=criterion, task_metrics=task_metrics,
            extensions={name: 'not_performed' for name in EXTENSIONS},
            uncertainty='Fixed-parameter state and trajectory variation; PF seeds describe integration noise')
        if settings['protocol'] == PROTOCOL_V2:
            from src.Bayesian_state.evaluation.model_0923_behavior_alignment import summarize_behavior_alignment
            from src.Bayesian_state.evaluation.model_0923_oral_alignment import load_observed_reports, summarize_oral_alignment
            behavior = summarize_behavior_alignment(inputs, trials, settings['window_size'], directory)
            oral_spec = settings['oral_alignment']
            reports = load_observed_reports(inputs, oral_spec) if oral_spec['mode'] == 'predict' else None
            seed_files = sorted((directory / 'seeds').glob('pf_*.npz'))
            oral = summarize_oral_alignment(inputs, seed_files, reports, directory, smoke=smoke, spec=oral_spec)
            write_json(directory / 'oral_alignment.json', oral)
            plot_behavior_alignment(sid, inputs['condition'], pd.read_csv(directory / 'behavior_alignment_curves.csv'),
                                    figures / f'subject_{sid}')
            plot_oral_alignment(sid, inputs['condition'], pd.read_csv(directory / 'oral_trials.csv'), oral,
                                figures / f'subject_{sid}')
            evaluation['behavior_alignment'] = behavior
            evaluation['oral_alignment'] = oral
            evaluation['steps'] = dict(
                input_and_search='completed', behavior_alignment='smoke_only' if smoke else 'completed_in_sample',
                oral_alignment=oral['status'], internal_states='fixed_parameter_pointwise_marginals',
                autonomous_learning='fixed_observed_schedule', numerical_stability='measured_without_acceptance_gate',
                task_criterion=criterion['status'])
            evaluation['extensions'].pop('oral_validation')
            evaluation['extensions']['oral_rule_state_alignment'] = 'not_performed'
            evaluation['extensions']['near_candidate_state_sensitivity'] = 'not_performed'
            evaluation['extensions']['autonomous_shape_analysis'] = 'not_performed'
            evaluation['conclusions'] = subject_alignment_conclusions(evaluation)
        write_json(directory / 'evaluation.json', evaluation)
        evaluations.append(evaluation)
    if settings['protocol'] == PROTOCOL_V2:
        write_alignment_report(output, figures, evaluations, settings, smoke)
    else:
        append_protocol_report(output, figures, evaluations, settings, smoke)
    # Keep the full effective protocol, including all subjects' paired budget checks.
    (output / 'effective_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    for path, digest in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Evaluation input changed during computation: {path}')
    manifest = dict(base, status='completed', protocol=settings['protocol'], smoke=smoke,
                    scope=plan['scope'], settings=settings, evaluations=evaluations,
                    elapsed_seconds=perf_counter() - started,
                    input_sha256=source_hashes, artifacts_sha256=artifact_manifest(output, figures, settings['subjects'], protocol=settings['protocol']))
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'progress.json', {'phase': 'completed', 'completed': len(evaluations), 'total': len(evaluations)})
    return manifest


def append_protocol_report(output: Path, figures: Path, evaluations: list[dict],
                           settings: dict, smoke: bool) -> None:
    """Write a complete report in the same five-step order for every subject."""
    summary = pd.read_csv(output / 'subjects.csv')

    def picture(path: Path) -> str:
        relative = html.escape(os.path.relpath(path, output))
        return f'<img src="{relative}" loading="lazy" alt="{html.escape(path.name)}">'

    labels = {
        'held_out_prediction': '留出预测', 'parameter_uncertainty': '参数统计区间',
        'parameter_recovery': '参数恢复', 'state_and_model_recovery': '状态与模型恢复',
        'mechanism_comparison': '机制比较', 'oral_validation': '口述验证',
        'adaptive_task_generation': '完整自适应实验生成', 'whole_internal_paths': '完整内部路径',
    }
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>Model 0923 逐被试评估 v1</title>',
        '<style>body{max-width:1100px;margin:32px auto;padding:0 24px;font:17px/1.7 sans-serif;color:#25313c}'
        'img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}'
        'td,th{padding:6px 9px;border-bottom:1px solid #ccd}h2{margin-top:40px}a{color:#236b9b}</style>',
        '<h1>Model 0923：逐被试评估 v1</h1>',
        '<p>入选参数在本次评估中固定。每位被试依次完成输入与搜索审查、选择预测、数值稳定性、'
        '内部状态、自主学习。完整观察序列参与参数估计，选择分数属于样本内描述。</p>',
        f'<p>基础预算{settings["particle_count"]}粒子×{settings["pf_repeats"]}组种子；'
        f'每人前{settings["calibration_repeats"]}组种子另用{settings["calibration_particle_count"]}粒子配对检查。'
        f'自主模拟{settings["autonomous_rollouts"]}条，滚动窗口{settings["window_size"]}题。'
        'PF种子描述数值积分波动；状态与自主轨迹范围以固定入选参数为条件。参数统计区间单独评价。</p>',
        '<p><a href="subjects.csv">逐人指标</a> · <a href="parameters.csv">参数表</a> · '
        '<a href="restarts.csv">搜索记录</a> · <a href="finalists.csv">候选库</a> · '
        '<a href="manifest.json">执行与产物清单</a></p>',
        '<h2>总体评分与参数审查</h2>', picture(figures / 'cohort_behavior.png'),
        picture(figures / 'parameter_audit.png'),
        summary.to_html(index=False, float_format=lambda value: f'{value:.4f}')]
    if smoke:
        parts.append('<p>本报告为首位配置被试的16题通路检查；原完整评分复现和实验达标保留未执行状态。</p>')
    for evaluation in evaluations:
        sid = evaluation['subject']
        row = summary.loc[summary.subject == sid].iloc[0]
        directory = figures / f'subject_{sid}'
        acceptance = html.escape(str(evaluation['fit_acceptance']))
        numerical = evaluation['numerical']
        learning = evaluation['learning']
        criterion = evaluation['task_criterion']
        parts += [f'<h2>S{sid}：条件{evaluation["condition"]}</h2>',
            '<h3>1. 输入与搜索审查</h3>',
            f'<p>源拟合格式：{html.escape(evaluation["source_kind"])}；原拟合验收状态：{acceptance}。'
            f'有效试次{int(row.modeled_trials)}题，计分{int(row.scored_trials)}题。程序完成与拟合验收分别记录。'
            '参数端点和影响通道关闭的情况保留在参数表中。</p>',
            '<p>原拟合未解决事项：' + (html.escape('；'.join(evaluation['fit_issues'])) if evaluation['fit_issues'] else '源记录未列出') + '。</p>',
            '<h3>2. 选择预测</h3>',
            f'<p>平均选择NLL {row.mean_choice_nll:.4f}，Brier分数{row.choice_brier:.4f}；'
            f'相对本条件均匀选择的NLL改善{row.nll_gain_over_uniform:.4f}。'
            '概率在当前选择之前取得，多PF种子先平均类别概率再计分；首条有效试次用于初始化。</p>',
            picture(directory / 'behavior.png'),
            '<h3>3. 数值稳定性</h3>',
            f'<p>全部重复相对前半重复的NLL差值{numerical["full_minus_half_repeats_nll"]:.6f}；'
            f'配对较高粒子预算相对基础预算的NLL差值{numerical["paired_high_minus_base_nll"]:.6f}，'
            f'平均类别概率绝对差值{numerical["paired_mean_abs_probability_difference"]:.6f}。'
            '原始重放按源记录的种子和粒子数核对，容差为1e−12；短检查仅检查通路。'
            '当前状态为“已测量，尚未设置共同验收门槛”。</p>', picture(directory / 'diagnostics.png'),
            '<h3>4. 内部状态</h3>',
            '<p>候选包含概率、规则信念与作答贡献分别描述是否考虑、可信程度和对回答的影响。'
            '表征图使用当前选择后、当前反馈前的状态；条件3的配对图显示反馈后信念。'
            '隐藏状态10%–90%范围为逐点范围，完整内部路径另行评价。</p>', picture(directory / 'representations.png'),
            '<h3>5. 自主学习与实验达标</h3>',
            f'<p>早期自主预期正确率中位数{learning["early_autonomous_expected_median"]:.3f}，'
            f'末期为{learning["late_autonomous_expected_median"]:.3f}；末期真实精确正确率'
            f'{learning["late_observed_accuracy"]:.3f}。描述性目标{settings["target_accuracy"]:.0%}用于分段比较。'
            '各自主轨迹使用自己的选择和反馈，沿用真实刺激顺序和记录长度。</p>']
        if evaluation['task_metrics']['status'] == 'completed':
            metrics = evaluation['task_metrics']
            observed = metrics['observed_task_metrics']
            parts.append(f'<p>条件3另报告：真实种正确率{observed["species_accuracy"]:.3f}，'
                f'科正确率{observed["family_accuracy"]:.3f}，平均得分{observed["mean_reward"]:.3f}。'
                '条件预测与自主生成的对应指标保留在task_metrics.json。</p>')
        parts.append(f'<p>实验达标检查状态：{html.escape(criterion["status"])}。</p>')
        if settings['task_criterion']['mode'] == 'exp123':
            spec = settings['task_criterion']
            parts.append(f'<p>每{spec["block_size"]}题正式区块排除模糊刺激，各类别精确正确率均达到'
                f'{spec["accuracy_threshold"]:.0%}时记为达标，统计包含区块首题。'
                '不完整区块保留明确状态。达标结果限定于真实刺激顺序和记录长度，完整自适应实验生成另行评价。</p>')
            if criterion.get('autonomous_any_pass_fraction') is not None:
                parts.append(f'<p>记录长度内自主轨迹曾达标比例{criterion["autonomous_any_pass_fraction"]:.1%}。</p>')
        else:
            parts.append(f'<p>{html.escape(settings["task_criterion"]["reason"])}</p>')
        parts += [f'<p><a href="subject_{sid}/trials.csv">逐题表</a> · '
            f'<a href="subject_{sid}/rule_states.csv">规则状态表</a> · '
            f'<a href="subject_{sid}/learning_segments.csv">分段学习表</a> · '
            f'<a href="subject_{sid}/criterion_blocks.csv">任务区块表</a> · '
            f'<a href="subject_{sid}/evaluation.json">完整步骤状态</a></p>',
            '<h3>专项检验状态</h3>', '<p>' + '；'.join(f'{labels[key]}（尚未执行）' for key in EXTENSIONS) + '。</p>']
    parts.append('</html>')
    (output / 'report.html').write_text('\n'.join(parts))


def write_alignment_report(output: Path, figures: Path, evaluations: list[dict],
                           settings: dict, smoke: bool) -> None:
    """Explain behavioral and oral validation before computational diagnostics."""
    subjects = pd.read_csv(output / 'subjects.csv')

    def number(value: Any, *, percent: bool = False, points: bool = False) -> str:
        if value is None:
            return '暂无可计算值'
        if value == 'infinity':
            return '∞（包含零概率报告）'
        if points:
            return f'{100 * float(value):.1f}个百分点'
        return f'{float(value):.1%}' if percent else f'{float(value):.4f}'

    def picture(path: Path) -> str:
        source = html.escape(os.path.relpath(path, output))
        return f'<img src="{source}" loading="lazy" alt="{html.escape(path.name)}">'

    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>Model 0923：行为与口述评估</title>',
        '<style>body{max-width:1100px;margin:32px auto;padding:0 24px;font:17px/1.7 sans-serif;color:#25313c}'
        'img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}'
        'td,th{padding:6px 9px;border-bottom:1px solid #ccd}h2{margin-top:40px}a{color:#236b9b}</style>',
        '<h1>Model 0923：行为与口述评估</h1>',
        '<p>评估固定每位被试已估计的行为参数。行为对齐检查模型能否预测回答和正确率的变化；'
        '口述对齐检查模型能否预测被试说出的判断依据。内部状态解释学习过程，自主模拟检查模型自己学习时的表现。'
        '参数和数值检查为这些比较提供计算依据。</p>',
        '<p>行为参数由完整作答序列估计，因此本报告的选择预测属于样本内评价。'
        '真实口述只进入预测评分，口述表达设置在比较前固定。</p>',
        '<p><a href="subjects.csv">逐人结果表</a> · <a href="parameters.csv">参数表</a> · '
        '<a href="effective_config.yaml">本次设置</a> · <a href="manifest.json">来源与文件清单</a></p>']
    if smoke:
        parts.append('<p>本报告检查一名被试最多16个有效试次，用于确认数据、计算和输出连接正确。正式指标需要完整序列。</p>')
    for evaluation in evaluations:
        sid = evaluation['subject']
        row = subjects.loc[subjects.subject == sid].iloc[0]
        behavior = evaluation['behavior_alignment']
        oral = evaluation['oral_alignment']
        directory = figures / f'subject_{sid}'
        parts.extend([f'<h2>S{sid}：条件{evaluation["condition"]}</h2>',
            f'<p>本次包含{int(row.modeled_trials)}个有效试次，其中{int(row.scored_trials)}题计分。首题用于初始化。</p>',
            '<h3>1. 行为对齐：回答和正确率</h3>',
            f'<p>真实精确正确率为{number(behavior["observed_accuracy"], percent=True)}，模型平均正确概率为'
            f'{number(behavior["conditional_correct_probability"], percent=True)}。'
            f'平均选择负对数似然（NLL）为{number(behavior["mean_choice_nll"])}，'
            f'均匀选择参考为{number(behavior["uniform_nll"])}，同段选择频率参考为'
            f'{number(behavior["constant_choice_frequency_nll"])}。NLL越小，模型给真实回答的概率越高；'
            '正确率比较使用真实类别的概率，两种比较各自保存。</p>', picture(directory / 'behavior_alignment.png')])
        rolling = behavior['rolling']
        if rolling['n_observations']:
            parts.append(f'<p>每{settings["window_size"]}题形成一个滚动窗口。正确率曲线平均绝对差为'
                f'{number(rolling["acc_mae"], points=True)}，波动比为{number(rolling["vol_ratio"])}。'
                '平均绝对差描述两条曲线相差多少；波动比描述模型变化幅度相对真实变化幅度的大小，接近1表示幅度接近。</p>')
        else:
            parts.append('<p>当前计分试次不足以形成完整滚动窗口，曲线指标保留为空。</p>')
        sequential = behavior['sequential']
        if sequential['status'] == 'completed_descriptive' and sequential['scored_pairs']:
            switch = sequential['switch']
            parts.append(f'<p>同一session内的连续计分试次中，真实换答比例为'
                f'{number(switch["switch_human"], percent=True)}，模型预测换答比例为'
                f'{number(switch["switch_model"], percent=True)}。正确后保持、错误后换答及过去8题的关系保留在行为指标表。'
                '换答指反应类别改变，规则变化由内部状态图说明。</p>')
        parts += [f'<p><a href="subject_{sid}/behavior_alignment.csv">行为指标</a> · '
                  f'<a href="subject_{sid}/behavior_alignment_curves.csv">正确率曲线源表</a></p>',
                  '<h3>2. 口述对齐：报告内容</h3>']
        if oral['status'] == 'completed':
            metrics = oral['metrics']
            parts.append(f'<p>{metrics["n_scored_trials"]}个计分试次中，'
                f'{metrics["n_scored_coded_reports"]}条口述可完整编码，'
                f'{metrics["n_scored_needs_review_reports"]}条待复核，'
                f'{metrics["n_scored_missing_reports"]}条缺失。完整报告中，模型词表覆盖'
                f'{number(metrics["vocabulary_coverage"], percent=True)}，得到正预测概率的比例为'
                f'{number(metrics["positive_support_coverage"], percent=True)}。</p>')
            if metrics['n_scored_coded_reports']:
                parts.append(f'<p>模型给实际完整报告的平均概率为{number(metrics["mean_report_probability"], percent=True)}。'
                    f'实际报告排在预测首位的比例为{number(metrics["top1_hit_rate"], percent=True)}，'
                    f'排在前三位的比例为{number(metrics["top3_hit_rate"], percent=True)}。'
                    f'严格口述NLL为{number(metrics["mean_report_nll"])}；'
                    f'{metrics["n_zero_probability_reports"]}条完整报告得到零概率。'
                    '词表外报告仍进入分母，缺失和待复核报告单列。</p>')
            else:
                parts.append('<p>目前没有可完整编码的计分报告，报告概率与命中率保留为空；原话和待复核原因均保存。</p>')
            parts.append('<p>口述预测在实际选择之后、当前反馈之前取得。编码需要人工复核。'
                '本次完成了完整报告预测；规则状态与口述几何的比较需要当前目录的对应关系，另列执行状态。</p>')
        else:
            parts.append(f'<p>{html.escape(oral["reason"])}</p>')
        parts += [picture(directory / 'oral_alignment.png'),
                  f'<p><a href="subject_{sid}/oral_trials.csv">原话与逐题预测</a> · '
                  f'<a href="subject_{sid}/oral_summary.csv">口述指标与分母</a></p>',
                  '<h3>3. 内部过程：规则怎样影响回答</h3>',
                  '<p>候选包含概率表示规则是否正在被考虑，信念表示它有多可信，作答贡献表示它对当前回答的影响。'
                  '候选数量、精度和搜索状态补充说明学习过程。图中的规则状态位于当前选择之后、反馈之前；'
                  '条件3配对信念另标明反馈后时点。状态范围以本次入选参数为条件。</p>',
                  picture(directory / 'representations.png'),
                  '<h3>4. 自主学习：模型自己能否学会</h3>',
                  f'<p>本次生成{settings["autonomous_rollouts"]}条自主轨迹。各轨迹沿真实刺激顺序自行抽取回答，'
                  '根据自己的回答获得反馈，再继续学习。图中同时展示真实学习曲线和自主轨迹的逐点范围。</p>',
                  picture(directory / 'behavior.png')]
        learning = evaluation['learning']
        parts.append(f'<p>自主预期正确率的早期中位数为{number(learning["early_autonomous_expected_median"], percent=True)}，'
                     f'末期为{number(learning["late_autonomous_expected_median"], percent=True)}；'
                     f'真实末期精确正确率为{number(learning["late_observed_accuracy"], percent=True)}。</p>'
                     '<p>完整曲线的代表轨迹、形态比例和持续掌握时间由另行配置的完整轨迹评价生成。</p>')
        criterion = evaluation['task_criterion']
        if criterion['status'] == 'completed':
            parts.append(f'<p>在记录长度内，自主轨迹曾达到实验标准的比例为'
                         f'{number(criterion["autonomous_any_pass_fraction"], percent=True)}。</p>')
        elif criterion['status'] == 'incomplete_blocks':
            parts.append('<p>记录含不完整区块。完整区块的达标结果逐项保存，总体曾达标比例保留为空。</p>')
        elif criterion['status'] == 'not_performed_smoke':
            parts.append('<p>短检查保留实验达标为未执行状态。</p>')
        else:
            parts.append(f'<p>实验达标标准：{html.escape(criterion.get("reason", "当前任务的结束标准尚待配置"))}。</p>')
        parts += ['<h3>5. 参数和数值检查：结果的依据</h3>',
                  '<p>原入选参数、搜索停止记录和未解决事项均保留。原种子评分重放、重复数与粒子数比较，'
                  '以及有效样本量ESS用于检查计算结果的敏感性。参数统计区间与恢复检验另行开展。</p>',
                  picture(directory / 'diagnostics.png'),
                  '<h3>6. 每人的结论与后续检验</h3>',
                  *[f'<p>{html.escape(text)}</p>' for text in evaluation['conclusions'].values()],
                  '<p>行为对齐、口述预测和自主学习分别报告。当前指标用于描述差异，统一的通过门槛需要共同验证方案。'
                  '留出预测、参数恢复、近优参数下的内部状态敏感性、完整内部路径，以及模型与机制比较逐项记录执行状态。</p>',
                  f'<p><a href="subject_{sid}/evaluation.json">各项结果与执行状态</a> · '
                  f'<a href="subject_{sid}/rule_states.csv">完整规则状态</a> · '
                  f'<a href="subject_{sid}/learning_segments.csv">学习分段表</a></p>']
    parts += ['<h2>总体参数与拟合评分</h2>', picture(figures / 'parameter_audit.png'),
              picture(figures / 'cohort_behavior.png'), '</html>']
    (output / 'report.html').write_text('\n'.join(parts))


def subject_alignment_conclusions(evaluation: dict[str, Any]) -> dict[str, str]:
    """Describe observed differences without inventing scientific pass thresholds."""
    behavior = evaluation['behavior_alignment']
    bias = behavior['conditional_correct_probability'] - behavior['observed_accuracy']
    direction = '高于' if bias >= 0 else '低于'
    behavioral = f'行为：模型平均正确概率{direction}真实正确率{100 * abs(bias):.1f}个百分点。'
    oral = evaluation['oral_alignment']
    if oral['status'] != 'completed':
        oral_text = f'口述：{oral["reason"]}'
    elif not oral['metrics']['n_scored_coded_reports']:
        oral_text = '口述：当前没有可完整编码的计分报告，原话和复核原因均保留。'
    else:
        metrics = oral['metrics']
        oral_text = (f'口述：{metrics["n_scored_coded_reports"]}条完整报告中，'
                     f'模型词表覆盖{metrics["vocabulary_coverage"]:.1%}，'
                     f'{metrics["n_zero_probability_reports"]}条得到零概率，'
                     f'首位命中率为{metrics["top1_hit_rate"]:.1%}。')
    learning = evaluation['learning']
    difference = learning['late_autonomous_expected_median'] - learning['late_observed_accuracy']
    direction = '高于' if difference >= 0 else '低于'
    autonomous = (f'自主学习：末期自主预期正确率中位数{direction}真实末期正确率'
                  f'{100 * abs(difference):.1f}个百分点，结果限定于真实刺激顺序和记录长度。')
    return {'behavior': behavioral, 'oral': oral_text, 'autonomous': autonomous,
            'acceptance': '上述数值描述各项差异；共同验收门槛、留出预测和恢复检验各自记录。'}
