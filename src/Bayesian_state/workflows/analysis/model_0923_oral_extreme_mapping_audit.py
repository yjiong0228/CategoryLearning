"""Audit response labels for complete reports of one longest/shortest feature.

Run --oral-audit AUDIT --prepared-data DATA --output NEW_DIRECTORY. The existing
catalogue supplies the exact extreme-feature regions. All 24 bijections between
their four regions and responses are retained. Training-segment maxima and all
their ties are evaluated unchanged on later reports. This developmental coding
diagnostic supplies design evidence; it does not fit model parameters or replace
the model's oral predictions. All original reports retain a coverage record.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from itertools import permutations
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import model_0923_decision_basis_alignment as alignment
from ...hypothesis_space.spaces import continuous, regions, structural_0923

ROOT = alignment.ROOT


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extreme_signatures(space) -> dict:
    """Use actual category regions, including their fixed response labels."""
    signatures = {}
    for rule in space:
        if rule.family not in ('dimension_min', 'dimension_max'):
            continue
        for category, region in enumerate(rule.categories):
            if len(region.components) != 1:
                raise ValueError('Extreme category must have one connected region')
            part = region.components[0]; matrix = np.column_stack((part.A, part.b))
            if matrix.shape != (3, 5) or not np.isfinite(matrix).all():
                raise ValueError('Expected three comparisons among four features')
            signature = frozenset(map(tuple, matrix))
            if signature in signatures:
                raise ValueError('Duplicate extreme geometry in the frozen catalogue')
            signatures[signature] = (rule.family.removeprefix('dimension_'), category + 1, rule.index)
    if len(signatures) != 8:
        raise ValueError('Need all four minima and four maxima')
    return signatures


def classify_extreme(tokens: str, signatures: dict) -> tuple | None:
    """Require the entire encoded conjunction to equal one extreme region."""
    planes = []
    for token in json.loads(tokens):
        if not isinstance(token, str) or not token.startswith('facet:'):
            return None
        value = np.asarray(json.loads(token[6:]), dtype=float)
        if value.shape != (1, 5) or not np.isfinite(value).all():
            return None
        planes.append(tuple(value[0]))
    return signatures.get(frozenset(planes))


def compare_mappings(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select every training-best bijection; later reports never break ties."""
    matrices = {phase: np.zeros((4, 4), dtype=int) for phase in ('training', 'test')}
    for row in frame.itertuples(index=False):
        if row.phase not in matrices or row.dimension not in range(1, 5) or row.choice not in range(1, 5):
            raise ValueError('Invalid report phase, dimension or response')
        matrices[row.phase][int(row.dimension) - 1, int(row.choice) - 1] += 1
    rows = []
    for mapping in permutations(range(1, 5)):
        rows.append(dict(mapping=json.dumps(mapping), canonical=mapping == (1, 2, 3, 4),
            **{phase + '_matches': sum(matrix[i, response - 1] for i, response in enumerate(mapping))
               for phase, matrix in matrices.items()}))
    scores = pd.DataFrame(rows); train_n = int(matrices['training'].sum()); test_n = int(matrices['test'].sum())
    scores['training_best'] = (scores.training_matches.eq(scores.training_matches.max()) if train_n else False)
    winners = scores.loc[scores.training_best]; canonical = scores.loc[scores.canonical].iloc[0]
    summary = dict(training_reports=train_n, test_reports=test_n,
        canonical_training=int(canonical.training_matches), canonical_test=int(canonical.test_matches),
        best_training=int(winners.training_matches.iloc[0]) if train_n else None,
        training_best_count=len(winners),
        training_best_test_min=int(winners.test_matches.min()) if train_n else None,
        training_best_test_max=int(winners.test_matches.max()) if train_n else None,
        training_observed_dimensions=int(np.any(matrices['training'], axis=1).sum()))
    return scores, summary


def run(audit: Path, prepared: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new extreme-mapping audit directory')
    manifest = alignment.read(audit / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Need a completed oral constraint audit')
    inputs = {str((audit / 'manifest.json').resolve()): digest(audit / 'manifest.json')}
    for name, expected in manifest['input_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Audited source changed: {name}')
        inputs[str((ROOT / name).resolve())] = expected
    if digest(audit / 'trials.csv') != manifest['artifacts']['trials.csv']:
        raise ValueError('Audited reports changed')
    inputs[str((audit / 'trials.csv').resolve())] = digest(audit / 'trials.csv')
    oral = pd.read_csv(audit / 'trials.csv'); space = structural_0923.build_relational_0923_space()
    signatures = extreme_signatures(space); frames = []
    for subject in sorted(oral.iSub.unique()):
        folder = prepared / f'subject_{subject}'; meta = alignment.read(folder / 'prepared.json')
        count = int((oral.iSub == subject).sum())
        frame, cut = alignment.align_identities(oral, dict(folder=str(folder)), count)
        np.testing.assert_array_equal(frame.ordinal_trial, np.arange(1, count + 1))
        for name in ('prepared.json', 'training.npz', 'sealed_test.npz'):
            inputs[str((folder / name).resolve())] = digest(folder / name)
        data_path = Path(meta['paths']['learning_data']); inputs[str(data_path.resolve())] = digest(data_path)
        frame['phase'] = np.where(frame.state_trial < cut, 'training', 'test')
        frame['extreme_kind'] = None; frame['dimension'] = np.nan; frame['catalogue_rule'] = np.nan
        frame['feature_name'] = None; frame['canonical_match'] = False
        frame['audit_scope'] = 'other_complete_content'
        frame.loc[~frame.scope.eq('complete'), 'audit_scope'] = 'incomplete_encoding'
        frame.loc[frame.condition.eq(1), 'audit_scope'] = 'binary_task_outside_mapping_audit'
        for index, row in frame.loc[frame.scope.eq('complete') & frame.condition.ne(1)].iterrows():
            if row.catalogue_version != space.version:
                raise ValueError('Report and numerical catalogue versions differ')
            match = classify_extreme(row.tokens, signatures)
            if match is None:
                continue
            kind, dimension, rule = match
            canonical = int(row.choice) == dimension
            if bool(row.structural_support_any_component > 0) != canonical:
                raise ValueError('Frozen catalogue coverage differs from its extreme labels')
            frame.loc[index, ['extreme_kind', 'dimension', 'catalogue_rule', 'feature_name', 'canonical_match', 'audit_scope']] = [
                kind, dimension, rule, row[f'feature{dimension}_name'], canonical, 'exact_single_extreme']
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    selected = full.loc[full.audit_scope.eq('exact_single_extreme')].copy()
    mapping_tables, summaries, cross = [], [], []
    for (subject, kind), group in selected.groupby(['iSub', 'extreme_kind'], sort=True):
        table, summary = compare_mappings(group)
        table.insert(0, 'subject', subject); table.insert(1, 'kind', kind); mapping_tables.append(table)
        summaries.append(dict(subject=subject, kind=kind, **summary))
        for phase in ('training', 'test'):
            for dimension in range(1, 5):
                name = full.loc[full.iSub.eq(subject), f'feature{dimension}_name'].iloc[0]
                for response in range(1, 5):
                    count = int((group.phase.eq(phase) & group.dimension.eq(dimension) & group.choice.eq(response)).sum())
                    cross.append(dict(subject=subject, kind=kind, phase=phase, dimension=dimension,
                                      feature_name=name, choice=response, reports=count))
    coverage = []
    for (subject, phase), group in full.groupby(['iSub', 'phase'], sort=True):
        complete = group.scope.eq('complete'); exact = group.audit_scope.eq('exact_single_extreme')
        coverage.append(dict(subject=subject, phase=phase, reports=len(group), complete_reports=int(complete.sum()),
            catalogue_supported_complete=int((complete & group.structural_support_any_component.gt(0)).sum()),
            exact_single_extreme=int(exact.sum()), canonical_extreme_matches=int(group.loc[exact, 'canonical_match'].sum())))
    tables = dict(summary=pd.DataFrame(summaries), mappings=pd.concat(mapping_tables, ignore_index=True),
        crosstabs=pd.DataFrame(cross), coverage=pd.DataFrame(coverage),
        reports=full[['iSub', 'condition', 'iSession', 'iTrial', 'ordinal_trial', 'phase', 'choice', 'scope',
                      'audit_scope', 'extreme_kind', 'dimension', 'feature_name', 'catalogue_rule', 'canonical_match', 'text']])
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    caption = ('本诊断只识别完整表达单个部位最长或最短的报告，并保留全部报告的覆盖记录。'
        '当前目录把第i个几何区域固定对应第i个作答类别。表中枚举全部24种一一对应；'
        '训练段得分最高的所有对应均保留，后续段只核对这些对应。'
        '这些计数用于开发阶段的模型设计，口述未用于估计行为模型参数，也未替换模型的口述预测。'
        '报告可能包含刺激描述；报告与选择的关联也可能来自共同刺激。计数本身不能确定被试使用了该分类策略。'
        '参与者均为开发样本，后续段计数属于描述性核对。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>极值报告与类别对应</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}td,th{padding:5px}</style>'
        '<h1>极值报告与类别对应</h1><p>' + caption + '</p>' + tables['summary'].to_html(index=False)
        + '<p><a href="mappings.csv">全部24种对应</a> · <a href="crosstabs.csv">部位与选择交叉表</a> · '
        '<a href="coverage.csv">完整覆盖记录</a> · <a href="reports.csv">逐份报告</a></p></html>')
    (output / 'index.html').write_text(page)
    sources = [Path(__file__), Path(alignment.__file__), Path(continuous.__file__), Path(regions.__file__), Path(structural_0923.__file__)]
    archive = output / 'source'; archive.mkdir()
    for path in sources:
        shutil.copy2(path, archive / path.name)
    record = dict(status='complete', sample_role='development', reports=len(full), exact_single_extreme=len(selected),
        source_sha256={str(p.resolve().relative_to(ROOT)): digest(p) for p in sources}, inputs_sha256=inputs,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'threadpoolctl')},
        catalogue_version=space.version, catalogue_signature=space.signature,
        selection='all training-best ties retained; later reports never select or resolve mappings',
        data_use='oral-response descriptive diagnostic; no behavior-model fit and no model prediction replacement',
        artifacts={str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oral-audit', type=Path, required=True)
    parser.add_argument('--prepared-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.oral_audit, args.prepared_data, args.output)
