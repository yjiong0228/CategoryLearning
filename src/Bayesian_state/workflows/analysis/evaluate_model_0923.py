"""Execute the fixed v1 subject evaluation and require its complete report bundle.

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
    EXTENSIONS, PROTOCOL, artifact_manifest, criterion_tables, load_protocol,
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
    return dict(protocol=PROTOCOL, settings=settings, fit_source=source,
                scope=('Frozen parameters; first 16 valid observed trials; workflow validation' if smoke else
                       'Frozen parameters; full observed history; in-sample prediction'), smoke=smoke)


def run(config_path: Path, output: Path | None = None, smoke: bool = False, *,
        figure_output: Path | None = None, dry_run: bool = False,
        plot_subject: Callable | None = None, plot_cohort: Callable | None = None,
        plot_diagnostics: Callable | None = None) -> dict[str, Any]:
    """Produce every standard section, retaining unavailable extension statuses."""
    started = perf_counter()
    config_path = config_path.resolve()
    plan = preflight(config_path, smoke=smoke)
    settings, source = plan['settings'], plan['fit_source']
    output = (output or Path(settings['output_dir'])).resolve()
    figures = (figure_output or (output / 'figures' if smoke else Path(settings['figure_output_dir']))).resolve()
    if output.exists() or figures.exists():
        raise FileExistsError('Choose new evaluation and figure output directories')
    if not smoke and not figures.is_relative_to(Path(__file__).resolve().parents[4] / 'CategoryLearning_codes/figures/outputs'):
        raise ValueError('Standard draft PNGs belong under CategoryLearning_codes/figures/outputs')
    if dry_run:
        return dict(**plan, status='planned', output_dir=str(output), figure_directory=str(figures))
    if any(callback is None for callback in (plot_subject, plot_cohort, plot_diagnostics)):
        raise ValueError('The v1 protocol requires all three report renderers')
    renderer_files = {Path(inspect.getsourcefile(callback)).resolve() for callback in
                      (plot_subject, plot_cohort, plot_diagnostics)}
    source_hashes = dict(source['input_sha256'])
    source_hashes.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in renderer_files})
    postfit_settings = deepcopy(settings)
    postfit_settings['calibration_subjects'] = list(settings['subjects'])
    base = run_postfit(config_path, output, smoke, plot_subject, plot_cohort,
                      settings_override=postfit_settings, figure_output=figures)
    source_hashes.update(base.get('analysis_source_sha256', {}))
    # Completion is published only after the mandatory v1 tables and panels exist.
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
        evaluation = dict(protocol=PROTOCOL, subject=sid, condition=int(inputs['condition']),
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
        write_json(directory / 'evaluation.json', evaluation)
        evaluations.append(evaluation)
    append_protocol_report(output, figures, evaluations, settings, smoke)
    # Keep the full effective protocol, including all subjects' paired budget checks.
    (output / 'effective_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    for path, digest in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Evaluation input changed during computation: {path}')
    manifest = dict(base, status='completed', protocol=PROTOCOL, smoke=smoke,
                    scope=plan['scope'], settings=settings, evaluations=evaluations,
                    elapsed_seconds=perf_counter() - started,
                    input_sha256=source_hashes, artifacts_sha256=artifact_manifest(output, figures, settings['subjects']))
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
