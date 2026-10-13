"""Apply the documented task stopping rule to saved autonomous trajectories.

The stimulus schedule is fixed to the observed schedule. This evaluates
criterion attainment within that schedule; adaptive task generation is pending.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.evaluation.model_0923_task_criterion import block_criterion
from src.Bayesian_state.utils.paths import ROOT_DIR


def run(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    settings = yaml.safe_load(config_path.read_text())

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()

    postfit, program = resolve(settings['postfit_run']), resolve(settings['source_program'])
    output = output or resolve(settings['output_dir'])
    threshold, size = float(settings['accuracy_threshold']), int(settings['block_size'])
    source_text = program.read_text()
    match = re.search(r'accstop\s*=\s*([\d.]+)', source_text)
    block_match = re.search(r'nTpB\s*=\s*\[\s*\d+\s*,\s*(\d+)\s*\]', source_text)
    if match is None or float(match.group(1)) != threshold or block_match is None or int(block_match.group(1)) != size:
        raise ValueError('Configured criterion differs from the experiment program')
    manifest = json.loads((postfit / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Need completed first-stage simulations')
    fit = Path(manifest['source_fit'])
    data_path = fit / 'source/data/exp123/processed/Task2_processed.csv'
    data = pd.read_csv(data_path)
    keys = ['iSub', 'condition', 'iSession', 'iBlock', 'iTrial']
    output.mkdir(parents=True, exist_ok=False)
    inputs = [config_path, program, data_path, postfit / 'manifest.json', Path(__file__),
              ROOT_DIR / 'src/Bayesian_state/evaluation/model_0923_task_criterion.py']
    summaries, blocks, trajectories = [], [], []
    for sid in settings['subjects']:
        trials_path = postfit / f'subject_{sid}' / 'trials.csv'
        simulation_path = postfit / f'subject_{sid}' / 'autonomous_rollouts.npz'
        inputs += [trials_path, simulation_path]
        trials = pd.read_csv(trials_path)
        original_keys = trials[keys].copy()
        trials = trials.merge(data[keys + ['ambiguous']], on=keys, how='left', validate='one_to_one', sort=False)
        pd.testing.assert_frame_equal(trials[keys], original_keys)
        if trials.ambiguous.isna().any():
            raise ValueError('A valid trial lacks its ambiguity annotation')
        condition = int(trials.condition.iloc[0])
        n_categories = 2 if condition == 1 else 4
        with np.load(simulation_path) as saved:
            feedback, seeds = saved['feedback'], saved['seeds']
        if feedback.shape[1] != len(trials) or len(seeds) != len(feedback):
            raise ValueError('Saved trajectory dimensions disagree with the observed trial schedule')
        first_pass = np.full(len(feedback), -1, dtype=int)
        observed_passes = []
        for block_index, ((session, block), frame) in enumerate(trials.groupby(['iSession', 'iBlock'], sort=False)):
            idx = frame.index.to_numpy()
            if len(idx) != size:
                raise ValueError('The stopping audit requires complete formal task blocks')
            outcomes = np.vstack([frame.feedback.to_numpy()[None, :], feedback[:, idx]])
            accuracy, passed = block_criterion(outcomes, frame.category.to_numpy(), frame.ambiguous.to_numpy(), n_categories, threshold)
            first_pass[(first_pass < 0) & passed[1:]] = block_index
            observed_passes.append(bool(passed[0]))
            row = dict(subject=sid, condition=condition, block_index=block_index, iSession=int(session), iBlock=int(block),
                       last_valid_trial=int(frame.valid_trial.iloc[-1]), nonambiguous_trials=int((frame.ambiguous == 0).sum()),
                       observed_reached_criterion=bool(passed[0]), autonomous_pass_count=int(passed[1:].sum()),
                       autonomous_pass_fraction=float(passed[1:].mean()))
            row.update({f'observed_accuracy_category_{k+1}': float(accuracy[0, k]) for k in range(n_categories)})
            blocks.append(row)
        reached = first_pass >= 0
        summaries.append(dict(subject=sid, condition=condition, valid_trials=len(trials), task_blocks=len(observed_passes),
                              observed_final_block_passed=observed_passes[-1], observed_first_pass_block=(observed_passes.index(True)+1 if any(observed_passes) else None),
                              autonomous_rollouts=len(feedback), autonomous_any_pass_count=int(reached.sum()),
                              autonomous_any_pass_fraction=float(reached.mean()), autonomous_final_pass_count=int(passed[1:].sum())))
        for i, first in enumerate(first_pass):
            trajectories.append(dict(subject=sid, rollout=i, seed=int(seeds[i]), reached=bool(first >= 0),
                                     first_pass_block=(int(first)+1 if first >= 0 else None)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'subjects.csv', index=False)
    pd.DataFrame(blocks).to_csv(output / 'blocks.csv', index=False)
    pd.DataFrame(trajectories).to_csv(output / 'trajectories.csv', index=False)
    display = summary[['subject', 'condition', 'valid_trials', 'task_blocks',
                       'observed_final_block_passed', 'autonomous_any_pass_count',
                       'autonomous_rollouts', 'autonomous_final_pass_count']].rename(columns={
                           'subject': '被试', 'condition': '条件', 'valid_trials': '有效试次',
                           'task_blocks': '区块数', 'observed_final_block_passed': '真实末块达标',
                           'autonomous_any_pass_count': '模拟曾达标条数', 'autonomous_rollouts': '模拟总条数',
                           'autonomous_final_pass_count': '模拟末块达标条数'})
    link = lambda p: html.escape(os.path.relpath(p, output))
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 0923 学习达标检查</title>',
             '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#25313c}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ccd}a{color:#236b9b}</style>',
             '<h1>Model 0923：沿真实刺激序列的学习达标检查</h1>',
             f'<p><a href="{link(output.parent / "figures.html")}">浏览全部26张图</a> · <a href="{link(output.parent / "report.html")}">参数候选与阶段检查</a></p>',
             f'<p>实验程序以每个{size}题正式区块为单位，排除模糊刺激后计算各类别的精确正确率；所有类别均达到{threshold:.0%}时结束学习。程序另根据当前表现调整下一块的类别采样。<a href="{link(program)}">原实验程序</a>保留这两项规则。</p>',
             '<p>本检查将同一达标定义用于已保存的64条自主轨迹，统计它们在被试记录的有效长度内是否出现过达标区块。任务达标统计包含区块首题，选择似然评分则排除模型初始化题。实际区块、刺激类别与模糊标记沿用被试记录。</p>',
             display.to_html(index=False),
             f'<p>实际记录末块通过同一标准的被试为{int(summary.observed_final_block_passed.sum())}/{len(summary)}人。模拟达标比例描述固定参数、固定真实刺激顺序下的结果。完整实验生成还需根据每条模拟轨迹的表现调整后续刺激，并在达标时结束；本表对真实刺激顺序条件化。</p>',
             '<p><a href="subjects.csv">逐人结果</a> · <a href="blocks.csv">各区块的类别正确率与达标数量</a> · <a href="trajectories.csv">各轨迹首次达标区块</a></p></html>']
    (output / 'report.html').write_text('\n'.join(parts))
    result = dict(status='completed', settings=settings, source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  uncertainty='Saved independent autonomous trajectories at fixed selected parameters and observed stimulus schedules',
                  remaining='Adaptive stimulus generation and model-dependent task termination')
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/exp123/specific_models/model_0923_selected12_task_criterion.yaml'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(run(args.config.resolve(), args.output_dir)['status'])


if __name__ == '__main__':
    main()
