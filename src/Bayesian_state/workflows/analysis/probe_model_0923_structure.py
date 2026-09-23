"""Check a 12-rule feature-overlap candidate against the frozen R1 reports.

Measures structural support separately from support at the observed label.
Does not fit parameters, choose labels, change production rules or run a PF.
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
import scipy
import yaml
from scipy.optimize import linprog

from ...hypothesis_space.geometry.boundary import BoundaryGeometry
from ...hypothesis_space.spaces import continuous, structural_0923
from ...hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from ...hypothesis_space.spaces.structural_0923 import build_axis_pair_overlap_probe_space
from ...model import oral_observation, oral_report_space
from ...model.oral_observation import predict_choice_oral
from ...model.oral_report_space import CatalogueReportSpace, ReportKernelParameters
from .prepare_model_0923 import _boolean, _rows, _write_csv


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _memberships(hypothesis, points: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.logical_or.reduce([np.all(points @ part.A.T <= part.b, axis=1)
                              for part in category.components])
        for category in hypothesis.categories
    ])


def _rule_indices(index: dict, code: str, category: int) -> list[int]:
    groups = index.get((code, category), {})
    return sorted({int(h) for values in groups.values() for h in values})


def probe_structure(*, report_dir: Path, trials_path: Path, config_path: Path,
                    output_dir: Path) -> dict[str, Any]:
    """Compare representability on identical codes and verify geometry/readout."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = yaml.safe_load(config_path.read_text())
    required = {'model_id': 'model_0923', 'stage': 'R2_axis_pair_overlap_probe',
                'status': 'development_structure_probe_not_adopted', 'candidate': 'axis_pair_overlap_12',
                'label_policy': 'preserve_fixed_labels_no_permutations'}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('unsupported structural probe configuration')
    for name in ('geometry_seed', 'geometry_points', 'probability_trials_per_subject'):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'geometry_seed' else 1):
            raise ValueError(f'{name} must be an appropriate nonnegative/positive integer')
    beta = float(config['probability_beta'])
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('probability_beta must be nonnegative and finite')
    report_config_path = Path(config['report_config'])
    report_config = yaml.safe_load(report_config_path.read_text())
    demo = dict(report_config['measurement_demo'])
    widths = demo.pop('body_reference_widths')
    parameters = [ReportKernelParameters(**demo, body_reference_width=width) for width in widths]
    r1_manifest_path = report_dir / 'manifest.json'
    r1_manifest = json.loads(r1_manifest_path.read_text())
    if r1_manifest['stage'] != 'R1_report_structure_and_kernel':
        raise ValueError('expected an R1 report audit')
    source_hashes = {source['sha256'] for source in r1_manifest['sources']}
    if any(sha256(path.read_bytes()).hexdigest() not in source_hashes
           for path in (trials_path, report_config_path)):
        raise ValueError('source trials or report config changed since R1')
    reports = _rows(report_dir / 'reports.csv')
    coded_trials = _rows(report_dir / 'trials.csv')
    source_trials = _rows(trials_path)
    by_report = {row['report_id']: row for row in reports}
    if len(by_report) != len(reports) or len(source_trials) != len(coded_trials):
        raise ValueError('duplicate reports or mismatched trial count')
    if len(source_trials) != r1_manifest['n_trials']:
        raise ValueError('R1 trial count mismatch')
    seen, observed_counts = set(), Counter()
    for raw, coded in zip(source_trials, coded_trials):
        keys = ('iSub', 'condition', 'iSession', 'iBlock', 'iTrial', 'trial', 'choice')
        if any(raw[key] != coded[key] for key in keys):
            raise ValueError('source and R1 trial identity/order differ')
        trial_key = tuple(raw[key] for key in ('iSub', 'iSession', 'iBlock', 'iTrial'))
        if trial_key in seen:
            raise ValueError('duplicate trial key')
        seen.add(trial_key)
        report = by_report[coded['report_id']]
        features = [raw[f'feature{i}_name'] for i in range(1, 5)]
        if (report['text'] != raw['text'] or json.loads(report['feature_order']) != features
                or report['choice'] != raw['choice'] or report['condition'] != raw['condition']
                or report['report_code'] != coded['report_code'] or report['parse_status'] != coded['parse_status']):
            raise ValueError('report content/feature order differs from source trial')
        if (_boolean(raw['scored']) != _boolean(coded['original_scored'])
                or _boolean(raw['oral_valid']) != _boolean(coded['original_encoded_valid'])):
            raise ValueError('original masks were changed')
        observed_counts[coded['report_id']] += 1
    if any(observed_counts[key] != int(row['occurrences']) for key, row in by_report.items()):
        raise ValueError('report occurrence count mismatch')

    base = build_continuous_hypothesis_space(4, 4)
    candidate = build_axis_pair_overlap_probe_space()
    old_reports = CatalogueReportSpace.from_catalogue(base)
    candidate_reports = CatalogueReportSpace.from_catalogue(candidate)
    old_index, candidate_index = old_reports.source_index(), candidate_reports.source_index()
    # Enumerate both spaces before scanning report contents. Geometry orientation
    # and label order never adapt to a report's content, label or performance.
    reviews = []
    for row in reports:
        condition, choice = int(row['condition']), int(row['choice'])
        eligible = condition in (2, 3) and row['parse_status'] == 'coded'
        code = row['report_code']
        old_labels = [y + 1 for y in range(4) if (code, y) in old_index] if eligible else []
        new_labels = [y + 1 for y in range(4) if (code, y) in candidate_index] if eligible else []
        old_chosen, new_chosen = choice in old_labels, choice in new_labels
        structural_gain = eligible and not old_labels and bool(new_labels)
        selected_gain = eligible and not old_chosen and new_chosen
        new_sources = {y: [h for h in _rule_indices(candidate_index, code, y - 1) if h >= len(base)]
                       for y in new_labels}
        new_sources = {y: values for y, values in new_sources.items() if values}
        reviews.append({
            'report_id': row['report_id'], 'condition': condition, 'choice': choice,
            'feature_order': row['feature_order'], 'text': row['text'], 'subjects': row['subjects'],
            'occurrences': int(row['occurrences']), 'parse_status': row['parse_status'],
            'r1_grammar_coverage': row['grammar_coverage'], 'eligible_four_category_report': eligible,
            'old_supported_labels': _json(old_labels), 'candidate_supported_labels': _json(new_labels),
            'structural_gain': structural_gain, 'observed_label_gain': selected_gain,
            'structural_gain_but_label_unresolved': structural_gain and not new_chosen,
            'new_hypothesis_indices_by_label': _json(new_sources),
        })
    review_lookup = {row['report_id']: row for row in reviews}
    trial_rows = []
    for coded in coded_trials:
        review = review_lookup[coded['report_id']]
        trial_rows.append({
            **{key: coded[key] for key in ('iSub', 'condition', 'iSession', 'iBlock', 'iTrial', 'trial', 'choice', 'report_id')},
            'original_scored': _boolean(coded['original_scored']),
            'original_encoded_valid': _boolean(coded['original_encoded_valid']),
            **{key: review[key] for key in ('eligible_four_category_report', 'structural_gain',
                                            'observed_label_gain', 'structural_gain_but_label_unresolved')},
        })
    metrics = ('eligible_four_category_report', 'structural_gain', 'observed_label_gain',
               'structural_gain_but_label_unresolved')
    subjects = []
    for sid in sorted({int(row['iSub']) for row in trial_rows}):
        records = [row for row in trial_rows if int(row['iSub']) == sid]
        subjects.append({'subject': sid, 'condition': records[0]['condition'], 'n_trials': len(records),
                         **{key: sum(row[key] for row in records) for key in metrics},
                         'sample_role': 'development'})

    rng = np.random.default_rng(config['geometry_seed'])
    points = rng.uniform(0, 1, (config['geometry_points'], 4))
    old_signatures = {_memberships(h, points).argmax(1).tobytes() for h in base}
    new_signatures = set()
    rules, interior_checks = [], []
    for hypothesis in candidate.hypotheses[len(base):]:
        membership = _memberships(hypothesis, points)
        if not np.all(membership.sum(1) == 1):
            raise ValueError('candidate categories do not partition off-boundary points')
        signature = membership.argmax(1).tobytes()
        if signature in old_signatures or signature in new_signatures:
            raise ValueError('duplicate partition detected in geometry probe')
        new_signatures.add(signature)
        axis = int(hypothesis.parameters['axis_dimension'])
        pair = list(hypothesis.parameters['related_dimensions'])
        rules.append({'index': hypothesis.index, 'family': hypothesis.family, 'axis_dimension': axis,
                      'comparison_dimensions': _json(pair), 'threshold': 0.5,
                      'hyperplanes': _json(hypothesis.hyperplanes), 'label_permutation': _json(hypothesis.label_permutation)})
        for y, category in enumerate(hypothesis.categories):
            part = category.components[0]
            # Interior ball includes the cube constraints, so every category
            # has positive volume, not just a shared boundary witness.
            A = np.vstack((part.A, np.eye(4), -np.eye(4)))
            b = np.concatenate((part.b, np.ones(4), np.zeros(4)))
            solution = linprog([0, 0, 0, 0, -1], A_ub=np.column_stack((A, np.linalg.norm(A, axis=1))),
                               b_ub=b, bounds=[(0, 1)] * 5, method='highs')
            if not solution.success or solution.x[-1] <= 1e-8:
                raise ValueError('candidate has an empty/degenerate category')
            interior_checks.append({'hypothesis': hypothesis.index, 'category': y + 1,
                                    'radius': float(solution.x[-1]), 'witness': solution.x[:4].tolist()})

    # Embed old kernels into the same candidate vocabulary, keeping exactly
    # zero mass on previously unavailable codes. This checks numerical row
    # preservation only: refining R1 OTHER into named new codes needs a separate
    # measurement assumption before predictive scoring. No fit score is computed.
    vocabulary = candidate_reports.codes
    lookup = {code: index for index, code in enumerate(vocabulary)}
    arrays, kernel_checks = {}, []
    for scenario, params in enumerate(parameters):
        old_kernel = old_reports.matrix(params)
        new_kernel = candidate_reports.matrix(params)
        common_old = np.zeros((len(base), 4, len(vocabulary)))
        for source_index, code in enumerate(old_reports.codes):
            common_old[:, :, lookup[code]] = old_kernel[:, :, source_index]
        if not np.array_equal(common_old, new_kernel[:len(base)]):
            raise ValueError('appending rules changed the old report distributions')
        arrays[f'candidate_{scenario}'] = new_kernel
        arrays[f'base_common_vocabulary_{scenario}'] = common_old
        kernel_checks.append({'scenario': scenario, 'parameters': asdict(params),
                              'max_row_sum_error': float(abs(new_kernel.sum(-1) - 1).max()),
                              'old_rows_identical_in_common_vocabulary': True})
    # Tiny real-stimulus check of shared boundary geometry and the joint readout.
    # Uniform candidate weights are a numerical fixture, not participant fits.
    geometry = BoundaryGeometry(candidate, method=BoundaryGeometry.METHOD_KKT_ACTIVE_SET)
    additions = list(range(len(base), len(candidate)))
    prediction_checks = []
    for sid in sorted({int(row['iSub']) for row in source_trials if int(row['condition']) in (2, 3)}):
        raw = [row for row in source_trials if int(row['iSub']) == sid][:config['probability_trials_per_subject']]
        stimuli = np.array([[float(row[f'feature{i}']) for i in range(1, 5)] for row in raw])
        probabilities = np.stack([geometry.category_probabilities(h, stimuli, beta).T for h in additions])
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1, atol=1e-12, rtol=0):
            raise ValueError('candidate choice probabilities are invalid')
        max_joint_error = 0.0
        for t in range(len(raw)):
            prediction = predict_choice_oral(np.full(len(additions), 1 / len(additions)), probabilities[:, t, :], arrays['candidate_0'][additions])
            independent = np.einsum('hc,hco->co', probabilities[:, t, :], arrays['candidate_0'][additions]) / len(additions)
            max_joint_error = max(max_joint_error, float(abs(prediction.joint_probabilities - independent).max()))
            if not np.allclose(prediction.joint_probabilities, independent, atol=1e-12, rtol=0):
                raise ValueError('joint readout disagrees with independent marginalization')
        prediction_checks.append({'subject': sid, 'n_trials': len(raw), 'beta': beta,
                                  'max_joint_error': max_joint_error, 'fitted': False})
    sources = [config_path, report_config_path, trials_path, r1_manifest_path,
               report_dir / 'reports.csv', report_dir / 'trials.csv', Path(__file__),
               Path(structural_0923.__file__), Path(continuous.__file__),
               Path(oral_report_space.__file__), Path(oral_observation.__file__)]
    manifest = {
        'model_id': 'model_0923', 'stage': config['stage'], 'status': config['status'],
        'base_rules': len(base), 'candidate_rules': len(candidate), 'added_rules': len(candidate) - len(base),
        'n_subjects': len(subjects), 'n_trials': len(trial_rows),
        'counts': {key: sum(row[key] for row in trial_rows) for key in metrics},
        'geometry_check': {'seed': config['geometry_seed'], 'points': len(points),
                           'valid_categories': len(interior_checks), 'distinct_new_partitions_on_probe': len(new_signatures)},
        'kernel_checks': kernel_checks, 'real_stimulus_readout_checks': prediction_checks,
        'vocabulary_size': len(vocabulary),
        'sources': [{'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()} for path in sources],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                        'scipy': scipy.__version__, 'pyyaml': yaml.__version__},
        'interpretation': {'fit_performed': False, 'candidate_adopted': False, 'label_permutations_enumerated': False,
                           'labels_changed_to_match_reports': False, 'cognitive_search_mechanism_changed': False,
                           'coverage_gain_is_not_predictive_improvement': True, 'production_catalogue_unchanged': True,
                           'zero_extension_is_not_calibrated_observation_refinement': True},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('reports.csv', reviews), ('trials.csv', trial_rows), ('subjects.csv', subjects), ('candidate_rules.csv', rules)]:
        _write_csv(output_dir / filename, rows)
    (output_dir / 'geometry_witnesses.json').write_text(json.dumps(interior_checks, indent=2) + '\n')
    (output_dir / 'vocabulary.json').write_text(_json(vocabulary) + '\n')
    np.savez_compressed(output_dir / 'kernel_demo.npz', **arrays)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['| 被试 | condition | 新增结构覆盖 | 新增原标签匹配 | 新覆盖但标签仍不匹配 |', '|---|---:|---:|---:|---:|']
    for row in subjects:
        lines.append(f"| {row['subject']} | {row['condition']} | {row['structural_gain']} | {row['observed_label_gain']} | {row['structural_gain_but_label_unresolved']} |")
    counts = manifest['counts']
    (output_dir / 'README.md').write_text(
        '# Model 0923：重复特征结构的小规模检查\n\n'
        '候选四分类目录由 116 条增至 128 条，仅增加 12 条“阈值特征也参加两维比较”的规则。'
        '二分类目录仍为 29 条。没有枚举标签排列，没有修改正式 B0 配置。\n\n'
        f"新增结构可表达 {counts['structural_gain']} 题的口述；保留原选择标签后新增匹配 {counts['observed_label_gain']} 题；"
        f"其中 {counts['structural_gain_but_label_unresolved']} 题新增结构仍受标签限制。\n\n"
        + '\n'.join(lines) + '\n\n'
        '“结构覆盖”忽略标签只用于定位缺口，不用于预测评分；本题 choice、原计分 mask 与试次顺序均保持。'
        '比较方向沿用目录的特征顺序，未根据 S314 的类别交换比较方向。\n\n'
        '48 个新增类别区域均有正体积内点，固定随机点上覆盖完整且不重叠。'
        '矩阵嵌入共同代码表后，旧规则各已有代码的概率逐元素保持一致。'
        '这是数值零扩展检查；原 OTHER 如何分配到新增代码尚未定义，不能用这些矩阵直接做公平的口述评分比较。'
        f"{len(prediction_checks)} 名四分类被试、每人最多前 {config['probability_trials_per_subject']} 题，仅作共享几何/联合读出的数值检查；均匀规则权重和固定 beta 不是拟合结果。\n\n"
        '此次结果支持保留一个有限结构候选，并显示标签仍是主要未解决问题。'
        '不能据覆盖增加采用新版，也不据此加入更多结构或执行机制。'
        '需要先确定不扩张规则空间的类别关联方案，完成报告校准后，再比较真实行为和口述预测。\n'
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--trials-path', type=Path, required=True)
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser


def main() -> None:
    result = probe_structure(**vars(build_parser().parse_args()))
    print(json.dumps({key: result[key] for key in ('base_rules', 'candidate_rules', 'n_trials', 'counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
