"""Compare frozen task-rule state timing with complete oral content.

The pre-existing geometry audit defines the task-rule identity. It is used only
to index saved states. Reports and task-rule identity do not change any state.
Whole filters remain the resampling unit for Monte Carlo standard errors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_decision_basis_replay import ROOT, digest, read, write
from ...evaluation.oral.decision_basis_scoring import pool_joint, pooled_mean_and_mcse


def run(alignment: Path, oral_choice: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new target-state timing directory')
    manifest, oral_manifest = read(alignment / 'manifest.json'), read(oral_choice / 'manifest.json')
    if manifest['status'] != 'complete' or manifest['smoke'] or oral_manifest['status'] != 'complete':
        raise ValueError('Need completed scientific alignment and oral-choice audit')
    for directory, audit in ((alignment, manifest), (oral_choice, oral_manifest)):
        if digest(directory / 'trials.csv') != audit['artifacts']['trials.csv']:
            raise ValueError('Audited trials changed')
    source = Path(manifest['source']); protocol = read(source / 'protocol.json')
    oral = pd.read_csv(oral_choice / 'trials.csv')
    aligned = pd.read_csv(alignment / 'trials.csv')
    trial_tables, windows = [], []
    for (subject, arm), group in aligned.groupby(['iSub', 'arm'], sort=True):
        group = group.sort_values('ordinal_trial').reset_index(drop=True)
        reports = oral[oral.subject.eq(subject)].sort_values('ordinal_trial').reset_index(drop=True)
        for left, right in [('iSession', 'session'), ('iTrial', 'trial'), ('ordinal_trial', 'ordinal_trial'), ('choice', 'choice')]:
            np.testing.assert_array_equal(group[left], reports[right])
        if reports.target_rule.nunique() != 1:
            raise ValueError('Task-rule identity must be frozen within participant')
        target = int(reports.target_rule.iloc[0]); numerators, probabilities, seeds = [], [], []
        for file in sorted((source / 'states').glob(f'evaluate_S{subject}_{arm}_*.json')):
            record = read(file); array = file.with_suffix('.npz')
            if (not record['exact_behavior_replay'] or digest(array) != record['array_sha256']
                    or record['array_sha256'] != manifest['state_arrays'][str(array.resolve())]):
                raise ValueError('Replay differs from the completed alignment')
            with np.load(array) as saved:
                np.testing.assert_array_equal(saved['choices'], reports.choice)
                probability = saved['choice_probability']
                active = saved['filtered_active'][:, target]
                contribution = saved['choice_component_joint_mass'][:, target].sum(axis=1)
                numerators.append(np.column_stack([active * probability, contribution]))
                probabilities.append(probability); seeds.extend(saved['seeds'].tolist())
        if len(seeds) != protocol['settings']['seeds_per_arm'] or len(set(seeds)) != len(seeds):
            raise ValueError('Incomplete independent filter group')
        numerators, probabilities = np.asarray(numerators), np.asarray(probabilities)
        values = pool_joint(numerators, probabilities)
        if np.any(values < -1e-12) or np.any(values > 1 + 1e-12) or np.any(values[:, 1] > values[:, 0] + 1e-12):
            raise ValueError('Task-rule contribution exceeds its posterior presence')
        result = reports.copy(); result['arm'] = arm
        result['target_presence'] = values[:, 0]; result['target_contribution'] = values[:, 1]
        result['whole_report_support'] = group.structural_support
        result['training_trials'] = group.training_trials
        trial_tables.append(result)
        ordinal = np.arange(len(result)); cut = int(group.training_trials.iloc[0])
        masks = [('training', ordinal < cut), ('test', ordinal >= cut)]
        masks += [(f'trials_{start+1}_{min(start+128,len(result))}',
                   (ordinal >= start) & (ordinal < start + 128)) for start in range(0, len(result), 128)]
        for name, mask in masks:
            point, error, _ = pooled_mean_and_mcse(numerators, probabilities, mask)
            selected = result[mask]; complete = selected[selected.complete]
            windows.append(dict(subject=int(subject), arm=arm, window=name, trials=len(selected),
                target_rule=target, target_presence=point[0], target_presence_mcse=error[0],
                target_contribution=point[1], target_contribution_mcse=error[1],
                complete_reports=len(complete), exact_task_branch_reports=int(complete.exact_task_branch.sum()),
                mean_whole_report_support=float(complete.whole_report_support.mean())))
    output.mkdir(parents=True)
    pd.concat(trial_tables, ignore_index=True).to_csv(output / 'trials.csv', index=False)
    summary = pd.DataFrame(windows); summary.to_csv(output / 'summary.csv', index=False)
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>任务规则状态与口述时间过程</title><h1>任务规则何时在场、何时影响作答</h1>'
        '<p>任务规则编号来自先前冻结的训练几何核对，当前仅索引已保存状态。'
        '存在概率与作答贡献先合并八次滤波的选择联合质量，再逐题归一化。误差为整次滤波的模拟标准误。</p>'
        '<p>完整报告数和其中恰好复现任务分支的报告数并列保留。省略、改述及角色未确定的内容均需另外核对。'
        '模型包含任务规则与被试明确报告该规则之间的时间差，提供表征诊断线索。</p>'
        + summary.to_html(index=False, float_format=lambda x: f'{x:.4f}') + '</html>')
    inputs = [alignment / 'manifest.json', alignment / 'trials.csv', oral_choice / 'manifest.json', oral_choice / 'trials.csv']
    write(output / 'manifest.json', dict(status='complete', subjects=manifest['subjects'], arms=manifest['arms'],
        inputs_sha256={str(p.resolve()): digest(p) for p in inputs},
        source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
            'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py': digest(ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py')},
        artifacts={p.name: digest(p) for p in output.iterdir()},
        interpretation='development descriptive state timing; no new state inference or parameter fitting'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'oral-choice', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); run(args.alignment, args.oral_choice, args.output)
