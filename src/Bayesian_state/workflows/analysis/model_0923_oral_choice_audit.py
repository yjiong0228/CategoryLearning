"""Describe exact task-branch reports alongside observed choice errors.

The criterion requires a completely parsed report with exactly the predicates
of one task-rule component containing the physical stimulus. All reports remain
in the denominator. This descriptive audit does not infer a motor-lapse rate or
establish that reported predicates were the participant's decision procedure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_decision_basis_replay import ROOT, digest, read, write
from .model_0923_oral_constraint_audit import catalogue
from ...model.modules.readout.report_space import component_tokens


def exact_applicable_branch(tokens: list[str], scope: str, components, stimulus: np.ndarray) -> bool:
    """Require complete predicate equality within one applicable component."""
    applicable = [c for c in components if np.all(c.A @ stimulus <= c.b + 1e-9)]
    if not applicable:
        raise ValueError('Physical stimulus has no component in its recorded task category')
    return scope == 'complete' and any(set(tokens) == set(component_tokens(c.A, c.b)) for c in applicable)


def run(audit: Path, diagnosis: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral-choice audit destination')
    manifest = read(audit / 'manifest.json')
    data_file = ROOT / 'data/exp123/processed/Task2_processed.csv'
    if (manifest['status'] != 'complete'
            or digest(audit / 'trials.csv') != manifest['artifacts']['trials.csv']
            or digest(data_file) != manifest['input_sha256'][str(data_file.relative_to(ROOT))]):
        raise ValueError('Oral audit or original physical table changed')
    oral, data = pd.read_csv(audit / 'trials.csv'), pd.read_csv(data_file)
    keys = ['iSub', 'iSession', 'iTrial']
    features = [f'feature{i}' for i in range(1, 5)]
    names = [f'feature{i}_name' for i in range(1, 5)]
    frame = oral.merge(data[keys + features + names + ['choice', 'category', 'ambiguous']],
        on=keys, how='left', validate='one_to_one', suffixes=('', '_physical'), indicator=True)
    if not frame['_merge'].eq('both').all():
        raise ValueError('Missing physical trial identities')
    for name in names + ['choice']:
        np.testing.assert_array_equal(frame[name], frame[name + '_physical'])
    inputs = [data_file, audit / 'manifest.json', audit / 'trials.csv']
    rows = []
    for subject, group in frame.groupby('iSub', sort=True):
        target_file = diagnosis / 'audit' / f'subject_{subject}' / 'audit.json'
        target = read(target_file); inputs.append(target_file)
        if target['selection'] != 'smallest_id_among_training_geometry_perfect_rules':
            raise ValueError('Expected previously frozen training-geometry target selection')
        space = catalogue(2 if target['condition'] == 1 else 4)
        rule = space[target['target']]
        if not group['condition'].eq(target['condition']).all():
            raise ValueError('Task condition mismatch')
        for item in group.to_dict('records'):
            stimulus = np.asarray([item[name] for name in features], dtype=float)
            if not np.isfinite(stimulus).all() or np.any((stimulus < 0) | (stimulus > 1)):
                raise ValueError('Invalid normalized physical stimulus')
            if item['ambiguous'] not in (0, 1):
                raise ValueError('Unexpected ambiguity flag')
            components = rule.categories[int(item['category']) - 1].components
            exact = exact_applicable_branch(json.loads(item['tokens']), item['scope'], components, stimulus)
            rows.append(dict(subject=subject, condition=target['condition'], session=item['iSession'],
                trial=item['iTrial'], ordinal_trial=item['ordinal_trial'], text=item['text'],
                scope=item['scope'], tokens=item['tokens'], target_rule=target['target'],
                choice=item['choice'], category=item['category'], ambiguous=bool(item['ambiguous']),
                correct=item['choice'] == item['category'], complete=item['scope'] == 'complete',
                exact_task_branch=exact, last128=item['ordinal_trial'] > group.ordinal_trial.max() - 128))
    trials = pd.DataFrame(rows); summaries = []
    for window, subset in [('all', trials), ('last128', trials[trials.last128]),
                           ('last128_nonambiguous', trials[trials.last128 & ~trials.ambiguous])]:
        for subject, group in subset.groupby('subject', sort=True):
            errors = ~group.correct
            summaries.append(dict(subject=subject, window=window, trials=len(group),
                complete=int(group.complete.sum()), errors=int(errors.sum()),
                complete_errors=int((errors & group.complete).sum()),
                exact_branch_errors=int((errors & group.exact_task_branch).sum()),
                exact_branch_correct=int((group.correct & group.exact_task_branch).sum())))
    summary = pd.DataFrame(summaries)
    output.mkdir(parents=True)
    trials.to_csv(output / 'trials.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>口述分支与选择错误</title><h1>完整复现任务分支的报告与选择</h1>'
        '<p>判据要求完整解析的整条报告，与包含当前物理刺激的一个正确类别分量具有完全相同的谓词。'
        '目标规则来自先前冻结的训练几何核对。全部报告保留在分母中，最后128题仅作描述性窗口。</p>'
        '<p>错误选择同时出现正确分支描述，可能涉及类别配对、知觉、表达或反应过程。'
        '这一计数尚不能确定口述内容是否用于作答，也不能估计运动失误率。'
        '未完全匹配的报告仍可能包含正确但省略或改述的判断。</p>'
        + summary.to_html(index=False) + '<p><a href="trials.csv">逐题原话与判定</a></p></html>')
    sources = [Path(__file__), ROOT / 'src/Bayesian_state/model/modules/readout/report_space.py',
               ROOT / 'src/Bayesian_state/workflows/analysis/model_0923_oral_constraint_audit.py']
    write(output / 'manifest.json', dict(status='complete', trials=len(trials),
        criterion='complete_report_exact_predicates_of_applicable_task_category_component',
        inputs_sha256={str(p.resolve()): digest(p) for p in inputs},
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        artifacts={p.name: digest(p) for p in output.iterdir()},
        data_use='development_descriptive_audit; no_model_weights_or_parameter_selection',
        limitation='report_role_unverified; exact_match_does_not_identify_cognitive_or_motor_process'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oral-audit', 'diagnosis', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); run(args.oral_audit, args.diagnosis, args.output)
