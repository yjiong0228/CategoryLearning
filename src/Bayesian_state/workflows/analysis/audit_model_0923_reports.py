"""Audit report structure and generate uncalibrated 0923 report kernels.

Reads the development inventory and original trial table, refuses overwrites,
and never fits parameters or changes raw data/old oral encodings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ...evaluation.oral import structure_0923
from ...evaluation.oral.structure_0923 import encode_report
from ...hypothesis_space.spaces import continuous
from ...hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from ...model import oral_report_space
from ...model.oral_report_space import (
    CatalogueReportSpace, ReportKernelParameters, report_code,
)
from .prepare_model_0923 import _boolean, _rows, _write_csv


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def audit_reports(*, inventory_dir: Path, trials_path: Path, config_path: Path,
                  output_dir: Path) -> dict[str, Any]:
    """Preserve trial identity and distinguish parsing from grammar support."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = yaml.safe_load(config_path.read_text())
    if config['model_id'] != 'model_0923' or config['status'] != 'development_audit_not_calibrated':
        raise ValueError('expected the uncalibrated 0923 audit config')
    if config['physical_body_reference'] != 0.75:
        raise ValueError('the confirmed physical body reference is 0.75')
    demo = dict(config['measurement_demo'])
    widths = demo.pop('body_reference_widths')
    if not widths or len(set(widths)) != len(widths):
        raise ValueError('body_reference_widths must be nonempty and unique')
    parameters = [ReportKernelParameters(**demo, body_reference_width=width) for width in widths]
    inventory_manifest = json.loads((inventory_dir / 'manifest.json').read_text())
    source_hash = sha256(trials_path.read_bytes()).hexdigest()
    if source_hash not in {item['sha256'] for item in inventory_manifest['sources']}:
        raise ValueError('trial source hash does not match the inventory')
    summaries = _rows(inventory_dir / 'subjects.csv')
    subjects = {int(row['subject']): row for row in summaries}
    if len(subjects) != len(summaries):
        raise ValueError('duplicate subject summary')
    trials = _rows(trials_path)
    if len(trials) != inventory_manifest['n_trials']:
        raise ValueError('trial count differs from inventory')
    keys, subject_trials = set(), Counter()
    for row in trials:
        sid, condition, choice = int(row['iSub']), int(row['condition']), int(row['choice'])
        if condition not in {1, 2, 3} or sid not in subjects or condition != int(subjects[sid]['condition']):
            raise ValueError('subject/condition mismatch')
        if not 1 <= choice <= (2 if condition == 1 else 4):
            raise ValueError('choice out of range')
        key = (sid, row['iSession'], row['iBlock'], row['iTrial'])
        if key in keys:
            raise ValueError('duplicate trial key')
        keys.add(key)
        subject_trials[sid] += 1
        if int(row['trial']) != subject_trials[sid]:
            raise ValueError('trial order mismatch')
    if set(subject_trials) != set(subjects) or any(subject_trials[sid] != int(subjects[sid]['n_trials']) for sid in subjects):
        raise ValueError('subject counts differ from inventory')

    spaces = {n: CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, n)) for n in (2, 4)}
    indices = {n: space.source_index() for n, space in spaces.items()}
    kernels: dict[str, np.ndarray] = {}
    kernel_checks = []
    for n, space in spaces.items():
        for scenario, params in enumerate(parameters):
            matrix = space.matrix(params)
            kernels[f'categories_{n}_scenario_{scenario}'] = matrix
            kernel_checks.append({'n_categories': n, 'scenario': scenario, 'shape': list(matrix.shape),
                                  'parameters': asdict(params),
                                  'max_row_sum_error': float(np.max(np.abs(matrix.sum(-1) - 1))),
                                  'minimum_probability': float(matrix.min())})
    cache, reports, trial_rows = {}, {}, []
    for row in trials:
        sid, condition, choice = int(row['iSub']), int(row['condition']), int(row['choice'])
        features = tuple(row[f'feature{i}_name'] for i in range(1, 5))
        raw_text = row['text']
        cache_key = (raw_text, features)
        if cache_key not in cache:
            cache[cache_key] = encode_report(raw_text, features)
        parsed = cache[cache_key]
        n = 2 if condition == 1 else 4
        code = report_code(parsed.tokens) if parsed.status == 'coded' else ''
        match = indices[n].get((code, choice - 1), {})
        alternatives = sorted(y + 1 for y in range(n) if (code, y) in indices[n] and y != choice - 1)
        if parsed.status != 'coded':
            coverage = parsed.status
        elif match.get('full_component'):
            coverage = 'full_component'
        elif match.get('omission'):
            coverage = 'omission'
        elif match.get('fuzzy_reference'):
            coverage = 'fuzzy_reference'
        elif alternatives:
            coverage = 'other_label_only'
        else:
            coverage = 'outside_report_grammar'
        # Include feature order in review identity: the previous raw-text
        # inventory may merge people whose anatomical features map differently.
        group_key = (condition, choice, features, raw_text)
        identifier = sha256(_json(group_key).encode()).hexdigest()[:16]
        if identifier not in reports:
            old_key = json.dumps((condition, choice, raw_text), ensure_ascii=False).encode()
            reports[identifier] = {
                'report_id': identifier, 'inventory_report_id': sha256(old_key).hexdigest()[:16] if parsed.status != 'missing' else '',
                'condition': condition, 'choice': choice, 'feature_order': _json(features), 'text': raw_text,
                'occurrences': 0, 'subjects': set(), 'first_subject': sid, 'first_trial': row['trial'],
                'parse_status': parsed.status, 'grammar_coverage': coverage, 'report_code': code,
                'relations': _json(parsed.relations), 'flags': _json(parsed.flags),
                'unparsed': _json(parsed.unparsed), 'review_suggestions': _json(parsed.review_suggestions),
                'source_hypotheses': _json(match), 'alternative_labels': _json(alternatives),
                'legacy_body_direction_disagreements': parsed.legacy_body_direction_disagreements,
                'review_status': 'automatic_requires_human_check',
            }
        reports[identifier]['occurrences'] += 1
        reports[identifier]['subjects'].add(sid)
        trial_rows.append({
            'iSub': sid, 'condition': condition, 'iSession': row['iSession'], 'iBlock': row['iBlock'],
            'iTrial': row['iTrial'], 'trial': row['trial'], 'choice': choice, 'report_id': identifier,
            'parse_status': parsed.status, 'grammar_coverage': coverage,
            'original_encoded_valid': _boolean(row['oral_valid']), 'original_scored': _boolean(row['scored']),
            'body_reference_uncertain': 'body_reference_uncertain' in parsed.flags,
            'legacy_body_direction_disagreement': parsed.legacy_body_direction_disagreements > 0,
            'physical_scale_required': 'physical_scale_required' in parsed.flags,
            'report_code': code,
        })
    report_rows = []
    for identifier in sorted(reports):
        report = reports[identifier]
        report['subjects'] = ';'.join(map(str, sorted(report['subjects'])))
        report_rows.append(report)
    subject_rows = []
    statuses = ('full_component', 'omission', 'fuzzy_reference', 'other_label_only',
                'outside_report_grammar', 'needs_review', 'missing')
    for sid, source in subjects.items():
        records = [row for row in trial_rows if row['iSub'] == sid]
        counts = Counter(row['grammar_coverage'] for row in records)
        subject_rows.append({
            'subject': sid, 'condition': source['condition'], 'n_trials': len(records),
            **{status: counts[status] for status in statuses},
            'body_reference_reports': sum(row['body_reference_uncertain'] for row in records),
            'legacy_direction_disagreement_reports': sum(row['legacy_body_direction_disagreement'] for row in records),
            'physical_scale_required_reports': sum(row['physical_scale_required'] for row in records),
            'sample_role': 'development',
        })
    source_paths = [trials_path, inventory_dir / 'manifest.json', inventory_dir / 'subjects.csv',
                    inventory_dir / 'report_review.csv', config_path, Path(__file__),
                    Path(structure_0923.__file__), Path(oral_report_space.__file__),
                    Path(continuous.__file__), Path('src/oral_coding.py')]
    manifest = {
        'model_id': 'model_0923', 'stage': config['stage'], 'encoder_version': structure_0923.VERSION,
        'n_subjects': len(subjects), 'n_trials': len(trials), 'n_feature_specific_reports': len(report_rows),
        'n_nonempty_feature_specific_reports': sum(row['parse_status'] != 'missing' for row in report_rows),
        'coverage_counts': dict(Counter(row['grammar_coverage'] for row in trial_rows)),
        'legacy_direction_disagreement_reports': sum(row['legacy_body_direction_disagreement'] for row in trial_rows),
        'physical_scale_required_reports': sum(row['physical_scale_required'] for row in trial_rows),
        'sources': [{'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()} for path in source_paths],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'pyyaml': yaml.__version__},
        'kernel_checks': kernel_checks,
        'interpretation': {
            'raw_data_and_legacy_encoding_unchanged': True, 'fit_performed': False,
            'kernel_calibrated': False, 'kernel_connected_to_particle_filter': False,
            'subject_specific_reference_parameter_added': False,
            'report_timing': 'choice_then_report_then_feedback',
            'outside_grammar_is_not_proof_of_outside_cognition': True,
            'missing_or_unresolved_is_not_other_or_empty': True,
            'vocabulary_source': 'catalogue_components_omissions_and_body_wording_before_text',
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / 'reports.csv', report_rows)
    _write_csv(output_dir / 'trials.csv', trial_rows)
    _write_csv(output_dir / 'subjects.csv', subject_rows)
    np.savez_compressed(output_dir / 'kernel_demo.npz', **kernels)
    (output_dir / 'vocabulary.json').write_text(_json({n: space.codes for n, space in spaces.items()}) + '\n')
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    table = ['| condition | 完整分量 | 省略 | 模糊躯干参照 | 仅其他标签 | 生成语法未覆盖 | 待复核 | 缺失 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in (1, 2, 3):
        counts = Counter(row['grammar_coverage'] for row in trial_rows if row['condition'] == condition)
        table.append('| ' + ' | '.join([str(condition), *(str(counts[status]) for status in statuses)]) + ' |')
    (output_dir / 'README.md').write_text(
        '# Model 0923：口述结构与报告概率审查\n\n'
        f"{len(subjects)} 人、{len(trials)} 题。按特征顺序进一步拆分为 {manifest['n_nonempty_feature_specific_reports']} 种非空报告记录；另保留缺失记录。\n\n"
        + '\n'.join(table) + '\n\n'
        f"旧解析与明确躯干比较方向不一致的报告：{manifest['legacy_direction_disagreement_reports']} 题。"
        f"保留加权和但仍需物理尺度核对：{manifest['physical_scale_required_reports']} 题。\n\n"
        '“完整分量”只表示所选类别的一个几何分量，不代表被试报告了整条分类规则。'
        '“模糊躯干参照”只表示生成语法允许，概率取决于尚未校准的参照宽度。'
        '“仅其他标签”不自动重标；“生成语法未覆盖”可能来自额外刺激描述、冗余关系或规则缺失，不能直接等同于认知规则在目录外。\n\n'
        '`reports.csv` 保留原话、特征映射、关系、未解析内容、旧方向差异与待人工检查标记；'
        '`trials.csv` 保留每题身份和原计分 mask；`subjects.csv` 提供个人汇总。'
        '`vocabulary.json` 与 `kernel_demo.npz` 是由目录生成的规范化报告矩阵。'
        '矩阵参数只用于归一化与敏感性检查，没有拟合或比较模型优劣。\n\n'
        '已确认物理躯干为 0.75；心理参照未被固定为此值。演示中心暂设 0.75，'
        '共用宽度取 0.05/0.10/0.20，仅显示表达概率如何随参照差距连续变化。'
        '没有增加个人阈值或改变行为规则；缺失和未解析报告不并入 OTHER/EMPTY。\n'
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-dir', type=Path, required=True)
    parser.add_argument('--trials-path', type=Path, required=True)
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser


def main() -> None:
    result = audit_reports(**vars(build_parser().parse_args()))
    print(json.dumps({key: result[key] for key in ('n_subjects', 'n_trials', 'coverage_counts',
                                                  'legacy_direction_disagreement_reports')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
