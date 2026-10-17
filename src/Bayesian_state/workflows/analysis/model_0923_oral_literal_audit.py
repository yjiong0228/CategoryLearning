"""Audit literal oral constraints against physical stimuli and fixed trial offsets.

Use --oral-audit FROZEN_AUDIT --output NEW_DIRECTORY. Retain every report in
trials.csv. Numerical agreement follows the frozen encoder's conventions;
body/half referents remain symbolic. Fixed offsets -2..2 use exactly the same
eligible reports and never cross sessions. This is a measurement audit: it
uses frozen numerical semantics and original trial identities. Report role,
perceptual calibration and isolated annotation discrepancies remain unresolved.
Model fitting and state inference continue under their existing protocols.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .model_0923_decision_basis_replay import ROOT, digest, read, write
from ...evaluation.oral.constraint_alignment import (
    InconsistentReportError, validate_report_constraints,
)

OFFSETS = (-2, -1, 0, 1, 2)
TOLERANCE = 1e-9


def literal_check(tokens: list[str], stimulus: np.ndarray) -> dict:
    """Check known predicates; retain symbolic referents without numerical anchors.

    Signed violation divides each affine inequality's residual by the norm of
    its normal vector. The maximum is a diagnostic of the most violated
    halfspace, not distance to the whole conjunction or a perceptual-noise SD.
    """
    point = np.asarray(stimulus, dtype=float)
    if point.shape != (4,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 1)):
        raise ValueError('Expected four normalized physical features')
    validate_report_constraints(tokens, n_dims=4, tolerance=TOLERANCE)
    numeric = [t for t in tokens if t.startswith('facet:')]
    margins = []; violated = []
    for token in numeric:
        values = np.asarray(json.loads(token[6:]), dtype=float)
        residual = (values[:, :4] @ point-values[:, 4]) / np.linalg.norm(values[:, :4], axis=1)
        margins.extend(residual.tolist())
        if np.any(residual > TOLERANCE):
            violated.append(token)
    return dict(numeric_predicates=len(numeric), symbolic_predicates=len(tokens)-len(numeric),
                numeric_holds=not violated if numeric else None,
                max_signed_halfspace_violation=max(margins) if margins else None,
                violated_tokens=violated)


def neighboring_keys(subject: int, session: int, trial: int, available: set) -> list[tuple] | None:
    """Return all fixed offsets, retaining original trial IDs and session boundaries."""
    keys = [(subject, session, trial+offset) for offset in OFFSETS]
    return keys if all(key in available for key in keys) else None


def run(audit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral-literal audit destination')
    manifest = read(audit/'manifest.json')
    data_file = ROOT/'data/exp123/processed/Task2_processed.csv'
    if (manifest['status'] != 'complete'
            or digest(audit/'trials.csv') != manifest['artifacts']['trials.csv']
            or digest(data_file) != manifest['input_sha256'][str(data_file.relative_to(ROOT))]):
        raise ValueError('Oral audit or physical data changed')
    inputs = {str(p.resolve()): digest(p) for p in (data_file, audit/'manifest.json', audit/'trials.csv')}
    oral = pd.read_csv(audit/'trials.csv')
    keys = ['iSub', 'iSession', 'iTrial']; features = [f'feature{i}' for i in range(1, 5)]
    names = [f'feature{i}_name' for i in range(1, 5)]
    data = pd.read_csv(data_file, usecols=keys+features+names)
    if data.duplicated(keys).any() or oral.duplicated(keys).any():
        raise ValueError('Duplicate trial identities')
    lookup = data.set_index(keys); available = set(lookup.index)
    rows = []; controls = []
    for item in oral.to_dict('records'):
        key = tuple(item[k] for k in keys)
        if key not in available:
            raise ValueError('Oral report lacks a physical trial')
        current = lookup.loc[key]
        if list(current[names]) != [item[n] for n in names]:
            raise ValueError('Feature-name order changed')
        tokens = json.loads(item['tokens'])
        try:
            result = literal_check(tokens, current[features].to_numpy(float)); inconsistent = False
        except InconsistentReportError:
            inconsistent = True
            result = dict(numeric_predicates=sum(t.startswith('facet:') for t in tokens),
                symbolic_predicates=sum(not t.startswith('facet:') for t in tokens), numeric_holds=None,
                max_signed_halfspace_violation=None, violated_tokens=[])
        complete_numeric = bool(item['scope'] == 'complete' and not inconsistent
                                and result['numeric_predicates'] and not result['symbolic_predicates'])
        neighbors = neighboring_keys(*key, available) if complete_numeric else None
        row = dict(subject=key[0], session=key[1], trial=key[2], ordinal_trial=item['ordinal_trial'],
            scope=item['scope'], text=item['text'], tokens=item['tokens'], inconsistent=inconsistent,
            complete_numeric=complete_numeric, lag_control_eligible=neighbors is not None,
            **result)
        row['violated_tokens'] = json.dumps(row['violated_tokens'], ensure_ascii=False)
        rows.append(row)
        if neighbors is not None:
            for offset, neighbor in zip(OFFSETS, neighbors):
                physical = lookup.loc[neighbor]
                if list(physical[names]) != list(current[names]):
                    raise ValueError('Neighbor feature ordering changed within session')
                checked = literal_check(tokens, physical[features].to_numpy(float))
                controls.append(dict(subject=key[0], session=key[1], trial=key[2],
                    ordinal_trial=item['ordinal_trial'], stimulus_trial_offset=offset,
                    stimulus_trial=neighbor[2], numeric_holds=checked['numeric_holds'],
                    max_signed_halfspace_violation=checked['max_signed_halfspace_violation']))
    trials = pd.DataFrame(rows); lag = pd.DataFrame(controls); summaries = []
    for subject, group in trials.groupby('subject', sort=True):
        numeric = group.loc[group.complete_numeric]
        summaries.append(dict(subject=subject, all_reports=len(group), complete_reports=int(group.scope.eq('complete').sum()),
            complete_numeric_reports=len(numeric), numeric_holds=int(numeric.numeric_holds.sum()),
            numeric_agreement=float(numeric.numeric_holds.mean()) if len(numeric) else None,
            complete_symbolic_reports=int((group.scope.eq('complete') & group.symbolic_predicates.gt(0)).sum()),
            inconsistent_reports=int(group.inconsistent.sum()), lag_reports=int(group.lag_control_eligible.sum()),
            numeric_reports_without_all_neighbors=int((group.complete_numeric & ~group.lag_control_eligible).sum())))
    summary = pd.DataFrame(summaries)
    lag_summary = lag.groupby(['subject', 'stimulus_trial_offset'], as_index=False).agg(
        reports=('numeric_holds', 'size'), numeric_holds=('numeric_holds', 'sum'),
        numeric_agreement=('numeric_holds', 'mean'))
    for subject, group in lag_summary.groupby('subject'):
        if set(group.stimulus_trial_offset) != set(OFFSETS) or group.reports.nunique() != 1:
            raise ValueError('Trial offsets must use identical report denominators')
    # The full trial table remains authoritative. This fixed-size excerpt only
    # prioritizes manual review of large discrepancies under the current coding.
    review = trials.loc[trials.complete_numeric & trials.numeric_holds.eq(False)].sort_values(
        ['max_signed_halfspace_violation', 'subject', 'ordinal_trial'], ascending=[False, True, True]).head(30)
    output.mkdir(parents=True)
    for name, frame in [('trials', trials), ('summary', summary), ('lag_controls', lag),
                        ('lag_summary', lag_summary), ('review_examples', review)]:
        frame.to_csv(output/f'{name}.csv', index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>口述数值约定与刺激一致性</title><h1>口述与物理刺激的测量审查</h1>'
        '<p>完整数值报告按已有编码约定检查。普通长短以0.5分界，近似等长沿用原编码范围；'
        '躯干与一半参照保留符号状态。全部报告保存在逐题表中。</p>'
        '<p>本检查描述固定约定下的字面一致性。偏差可能涉及知觉、转录、编码参照或报告用途，'
        '具体原因仍待逐例核实。报告与刺激一致也不能确定该内容参与了作答。</p>'
        + summary.to_html(index=False) + '<h2>固定试次偏移对照</h2>'
        '<p>偏移表示用报告所在题号加上该数值寻找刺激。各偏移使用相同报告，'
        '仅纳入同一会话内五个题号都存在的完整数值报告。原试次连接与编码保持固定。</p>'
        + lag_summary.to_html(index=False) + '<h2>人工复核例子</h2>'
        '<p>每项不等式以左侧值减去右侧界限，再除以系数向量的欧氏范数，得到归一化违约量。'
        '正值表示该不等式未满足。每份报告取其中最大值，下表按该值保留前30条。'
        '该量描述单项不等式的偏离；整条报告的距离和知觉噪声属于不同的测量对象。'
        '全部偏差记录保留在逐题表中。</p>'
        + review[['subject', 'session', 'trial', 'text', 'max_signed_halfspace_violation']].to_html(index=False)
        + '<p><a href="trials.csv">全部逐题结果</a> · <a href="lag_controls.csv">全部偏移对照</a></p></html>')
    for name, sha in inputs.items():
        if digest(Path(name)) != sha:
            raise ValueError('Input changed during audit')
    sources = [Path(__file__), ROOT/'src/Bayesian_state/evaluation/oral/constraint_alignment.py']
    for path in sources:
        target = output/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    write(output/'manifest.json', dict(status='complete', inputs_sha256=inputs,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        python=sys.version, versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas')},
        trials=len(trials), complete_numeric_reports=int(trials.complete_numeric.sum()),
        matched_lag_reports=int(trials.lag_control_eligible.sum()), offsets=list(OFFSETS), tolerance=TOLERANCE,
        data_use='development_measurement_audit; no_model_states_choices_feedback_or_parameter_selection',
        numerical_semantics='frozen_oral_tokens; symbolic_referents_unresolved',
        limitations=['Literal agreement does not identify report role', 'No report likelihood or fitted threshold',
                     'A high current-trial agreement does not exclude isolated annotation errors'],
        review_selection='30 largest normalized halfspace violations among complete numeric reports; all rows retained',
        artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oral-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.oral_audit, args.output)
