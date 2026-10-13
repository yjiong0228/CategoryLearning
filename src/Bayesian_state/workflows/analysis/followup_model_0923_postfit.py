"""Check saved Model 0923 finalists and autonomous learning; index all first-stage figures.

The source fits, simulations and plots are read-only. New candidates are scored
with the independent common PF seeds used in the first-stage postfit run.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.evaluation.model_0923_followup import compare_finalists, learning_summary
from src.Bayesian_state.evaluation.model_0923_postfit import resolve_subject, rule_catalog, score_probabilities
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.metrics.task import category_learning_feedback
from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
from src.Bayesian_state.utils.parallel import parallel_job_count, single_threaded_processes
from src.Bayesian_state.utils.paths import ROOT_DIR
from .analyze_model_0923_postfit import resolve, write_json


def run_finalist_task(task: Mapping[str, Any]) -> str:
    inputs = resolve_subject(Path(task['fit_run']), int(task['subject']), Path(task['cache']), task.get('max_trials'))
    engine = apply_fixed_hyperparams_to_engine_config(inputs['engine'], task['hyperparams'])
    np.random.seed(int(task['seed']))
    result = run_state_model_particle_filter(
        engine_config=engine, subject_id=inputs['subject'], condition=inputs['condition'],
        stimulus=inputs['arrays'].stimulus, choices=inputs['arrays'].choices,
        feedback=inputs['arrays'].feedback, particle_count=int(task['particles']),
        filter_seed=int(task['seed']), choice_readout_power=1.,
        processed_data_dir=inputs['paths']['processed_dir'], dataset_paths=inputs['paths'])
    probabilities = np.asarray(result.observation_probabilities['prior_t'])
    score = score_probabilities(probabilities, inputs['arrays'].choices)['mean_choice_nll']
    with Path(task['output']).open('xb') as stream:
        np.savez_compressed(stream, probabilities=probabilities, seed=np.asarray(task['seed']),
                            particles=np.asarray(task['particles']), mean_choice_nll=np.asarray(score))
    return str(task['output'])


def figure_gallery(postfit: Path, output: Path, subjects: list[int], figures: Path) -> int:
    def rel(path: Path) -> str:
        return html.escape(os.path.relpath(path, output))

    sections = [('总体：拟合与参数', [figures / 'cohort_behavior.png', figures / 'parameter_audit.png'])]
    sections += [(f'S{sid}：行为与内在表征', [figures / f'subject_{sid}' / 'behavior.png',
                                                     figures / f'subject_{sid}' / 'representations.png']) for sid in subjects]
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 0923 全部分析图</title>',
             '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#25313c}img{max-width:100%;height:auto}a{color:#236b9b}section{margin:40px 0}</style>',
             '<h1>Model 0923：第一阶段分析图</h1>',
             f'<p><a href="{rel(postfit / "report.html")}">完整图文报告</a> · '
             f'<a href="{rel(postfit / "findings.html")}">主要发现</a></p>',
             '<p>每人两张图：behavior.png显示行为预测与自主模拟；representations.png显示模型推断的候选规则、信念、作答贡献和精度。状态范围以原入选参数为条件。各图保留原输出文件。</p>',
             '<nav>' + ' · '.join(f'<a href="#s{sid}">S{sid}</a>' for sid in subjects) + '</nav>']
    count = 0
    for i, (title, paths) in enumerate(sections):
        anchor = 'overall' if i == 0 else f's{subjects[i-1]}'
        parts.append(f'<section id="{anchor}"><h2>{title}</h2>')
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            parts.append(f'<p><a href="{rel(path)}">打开 {path.name}</a></p><img src="{rel(path)}" loading="lazy" alt="{html.escape(title + " " + path.name)}">')
            count += 1
        parts.append('</section>')
    parts.append('</html>')
    (output / 'figures.html').write_text('\n'.join(parts))
    return count


def run(config_path: Path, output: Path | None = None, smoke: bool = False) -> dict[str, Any]:
    started = perf_counter()
    settings = yaml.safe_load(config_path.read_text())
    fit = resolve(config_path.parent, settings['fit_run'])
    postfit = resolve(config_path.parent, settings['postfit_run'])
    output = output or resolve(config_path.parent, settings['output_dir'])
    source = json.loads((postfit / 'manifest.json').read_text())
    if source['status'] != 'completed' or Path(source['source_fit']).resolve() != fit:
        raise ValueError('Need a completed postfit run from the configured source fit')
    fit_manifest = json.loads((fit / 'run_manifest.json').read_text())
    input_paths = [(fit / 'source' / name, digest) for name, digest in fit_manifest['snapshot_files_sha256'].items()]
    input_paths += [(fit / name, digest) for name, digest in fit_manifest['runtime_files_sha256'].items()]
    for path, digest in input_paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Archived input changed: {path}')
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    if any(sid not in source['settings']['subjects'] for sid in subjects):
        raise ValueError('Requested subject is absent from the source postfit run')
    repeats = 2 if smoke else int(settings['pf_repeats'])
    particles = 4 if smoke else int(settings['particle_count'])
    if repeats > source['settings']['pf_repeats'] or (not smoke and particles != source['settings']['particle_count']):
        raise ValueError('Reused PF seeds require matching particle and repeat budgets')
    max_trials, workers = (16, 1) if smoke else (None, int(settings['parallel_budget']))
    output.mkdir(parents=True, exist_ok=False)
    cache = output / 'resources_cache'
    shutil.copytree(postfit / 'resources_cache', cache)
    settings.update(subjects=subjects, pf_repeats=repeats, particle_count=particles,
                    parallel_budget=workers, max_trials=max_trials)
    (output / 'effective_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    figures = Path(source['figure_directory'])
    figure_count = figure_gallery(postfit, output, source['settings']['subjects'], figures)
    tracked_inputs = [config_path, postfit / 'manifest.json', fit / 'run_manifest.json',
                      *sorted((ROOT_DIR / 'src/Bayesian_state').rglob('*.py'))]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_inputs}
    analysis_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

    def progress(phase: str, completed: int = 0, total: int = 0) -> None:
        write_json(output / 'progress.json', dict(phase=phase, completed=completed, total=total,
                   updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
        print(phase, completed, total, flush=True)

    inputs = {sid: resolve_subject(fit, sid, cache, max_trials) for sid in subjects}
    banks, paths, selections, tasks = {}, {}, {}, []
    for sid, value in inputs.items():
        rule_catalog(value)
        bank_path = fit / 'hyper_cd' / f'subject_{sid}' / 'final_rescore.jsonl'
        tracked_inputs += [bank_path, postfit / f'subject_{sid}' / 'trials.csv',
                           postfit / f'subject_{sid}' / 'autonomous_rollouts.npz']
        bank = [json.loads(line) for line in bank_path.read_text().splitlines()]
        selected = next(i for i, row in enumerate(bank) if row['hyperparams'] == value['point'])
        banks[sid], selections[sid] = bank, selected
        paths[sid] = [[] for _ in bank]
        for i, row in enumerate(bank):
            directory = output / f'subject_{sid}' / f'candidate_{i}'
            directory.mkdir(parents=True)
            for repeat in range(repeats):
                source_seed = postfit / f'subject_{sid}' / 'seeds' / f'pf_{repeat:02d}.npz'
                tracked_inputs.append(source_seed)
                with np.load(source_seed) as saved:
                    seed = int(saved['seed'])
                if i == selected and not smoke:
                    paths[sid][i].append(source_seed)
                    continue
                path = directory / f'pf_{repeat:02d}.npz'
                paths[sid][i].append(path)
                tasks.append(dict(fit_run=str(fit), subject=sid, cache=str(cache), max_trials=max_trials,
                                  hyperparams=row['hyperparams'], seed=seed, particles=particles, output=str(path)))
        write_json(output / f'subject_{sid}' / 'candidate_bank.json', dict(selected_candidate=selected, candidates=bank))
    hashes.update({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_inputs})
    progress('fixed_candidate_pf', 0, len(tasks))
    with single_threaded_processes():
        generated = Parallel(n_jobs=parallel_job_count(workers, len(tasks)), batch_size=1,
                             return_as='generator_unordered')(delayed(run_finalist_task)(task) for task in tasks)
        for i, _ in enumerate(generated, 1):
            if i % 32 == 0 or i == len(tasks):
                progress('fixed_candidate_pf', i, len(tasks))

    finalists, rankings, learning, segments = [], [], [], []
    for sid, value in inputs.items():
        probabilities = []
        for candidate in paths[sid]:
            rows = []
            for path in candidate:
                with np.load(path) as saved:
                    rows.append(saved['probabilities'])
            probabilities.append(rows)
        rows, diagnostic = compare_finalists(np.asarray(probabilities), value['arrays'].choices, selections[sid])
        old = np.array([r['aggregated_error'] for r in banks[sid]])
        old_rank = np.argsort(np.argsort(old, kind='stable'), kind='stable') + 1
        for i, row in enumerate(rows):
            row.update(subject=sid, condition=value['condition'], original_rank=int(old_rank[i]),
                       original_nll=float(old[i]), source_seed_files=[str(p) for p in paths[sid][i]])
            finalists.append(row)
        diagnostic.update(subject=sid, condition=value['condition'])
        rankings.append(diagnostic)
        trials = pd.read_csv(postfit / f'subject_{sid}' / 'trials.csv')
        np.testing.assert_array_equal(trials.choice.to_numpy()[:len(value['arrays'].choices)], value['arrays'].choices)
        np.testing.assert_array_equal(trials.category.to_numpy()[:len(value['arrays'].choices)], value['arrays'].categories)
        np.testing.assert_array_equal(trials.feedback.to_numpy()[:len(value['arrays'].choices)], value['arrays'].feedback)
        n = len(value['arrays'].choices)
        with np.load(postfit / f'subject_{sid}' / 'autonomous_rollouts.npz') as saved:
            generated_choices, feedback, expected = [saved[k][:, :n] for k in ('choices', 'feedback', 'expected_correct_probability')]
        task_feedback = np.array([[category_learning_feedback(int(choice), int(category), condition=value['condition'])
                                  for choice, category in zip(choices, value['arrays'].categories)] for choices in generated_choices])
        np.testing.assert_array_equal(task_feedback, feedback)
        rows, summary = learning_summary(value['arrays'].feedback, feedback, expected,
                                        trials.predicted_correct_probability.to_numpy()[:n],
                                        int(settings['segment_size']), int(settings['late_window']),
                                        float(settings['target_accuracy']))
        for row in rows:
            row.update(subject=sid, condition=value['condition'])
            segments.append(row)
        summary.update(subject=sid, condition=value['condition'])
        learning.append(summary)
    tables = {'finalists': finalists, 'rankings': rankings, 'learning': learning, 'learning_segments': segments}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False)
    report(output, postfit, pd.DataFrame(rankings), pd.DataFrame(learning))
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
        raise ValueError('A source file changed during the follow-up')
    manifest = dict(status='completed', scope='saved_finalist_rescore_and_saved_autonomous_descriptions',
                    smoke=smoke, settings=settings, elapsed_seconds=perf_counter() - started,
                    source_fit=str(fit), source_postfit=str(postfit), figures_indexed=figure_count,
                    generated_pf_tasks=len(tasks), analysis_commit_at_start=analysis_commit,
                    input_and_source_sha256=hashes,
                    versions={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'joblib', 'PyYAML')},
                    seed_role='First-stage independent postfit seeds, common across all saved finalists',
                    limits=['Numerical seed uncertainty conditions on the observed data.',
                            'Autonomous behavior descriptions fix selected parameters and observed schedules/lengths.',
                            'Parameter recovery, held-out validation and causal mechanism checks remain pending.'])
    write_json(output / 'manifest.json', manifest)
    progress('completed', len(subjects), len(subjects))
    return manifest


def report(output: Path, postfit: Path, rankings: pd.DataFrame, learning: pd.DataFrame) -> None:
    original = html.escape(os.path.relpath(postfit / 'report.html', output))
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 0923 定向后分析</title>',
             '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#25313c}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{padding:7px;border-bottom:1px solid #ccd}a{color:#236b9b}</style>',
             '<h1>Model 0923：参数候选与自主学习的定向检查</h1>',
             f'<p><a href="figures.html">浏览第一阶段全部26张图</a> · <a href="{original}">第一阶段完整报告</a></p>',
             '<h2>固定候选的数值可区分性</h2>',
             '<p>每人保留原来的四组入围参数，使用第一阶段独立于参数搜索的共同粒子滤波种子。先跨种子平均逐题选择概率，再计算负对数似然（NLL）；NLL越低，观察选择的预测概率越高。新候选评分保存在本目录，原入选参数保留原记录。表中候选编号从0开始，见逐人candidate_bank.json。</p>',
             '<p>Jackknife逐次删除一组随机种子，并重新计算候选间的NLL差值，其标准误描述粒子滤波数值积分的波动。参数统计区间需要另做数据层面的不确定性评价。两半种子的赢家用于检查当前预算下的排序敏感性。</p>',
             rankings.to_html(index=False, float_format=lambda x: f'{x:.5f}'),
             '<p><a href="finalists.csv">各候选分数、配对数值误差与概率差异</a> · <a href="rankings.csv">逐人排名检查</a></p>',
             '<h2>自主学习差距出现在哪个阶段</h2>',
             '<p>使用原来的64条自主轨迹，分别统计整体和早期、晚期窗口。第一条有效记录用于初始化，统计从第二条开始。精确正确率将条件3的半分回答记为未完全正确。自主预期正确概率来自各模拟轨迹当时的预测分布；它有助于检查偏低表现是否持续存在于学习状态。</p>',
             '<p>以下窗口分位数描述固定参数下模拟轨迹的变化。窗口正确率达到配置阈值的比例描述局部表现；持续掌握、实验终止时间和群体统计推断需另行分析。</p>',
             learning[['subject', 'condition', 'overall_observed_accuracy', 'overall_autonomous_accuracy_median',
                       'overall_autonomous_accuracy_q90', 'early_autonomous_expected_median',
                       'late_observed_accuracy', 'late_autonomous_accuracy_median', 'late_autonomous_accuracy_q90',
                       'late_autonomous_expected_median', 'late_fraction_rollouts_at_or_above_target']].to_html(index=False, float_format=lambda x: f'{x:.3f}'),
             '<p><a href="learning.csv">完整阶段指标</a> · <a href="learning_segments.csv">逐段学习数据</a></p>',
             '<p>本次检查定位数值排序和行为差距。具体认知机制的归因、参数恢复及留出预测继续作为后续验收项目。</p></html>']
    (output / 'report.html').write_text('\n'.join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/exp123/specific_models/model_0923_selected12_followup.yaml'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke and args.output_dir is None:
        parser.error('--smoke requires a new --output-dir')
    with single_threaded_processes():
        result = run(args.config.resolve(), args.output_dir, args.smoke)
    print(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds']))


if __name__ == '__main__':
    main()
