"""Read-only audit of nested execution fits, prediction and simulation recovery.

Use --source STUDY --output NEW. --allow-incomplete creates a labeled preview.
The report does not change nominees, numerical thresholds or recorded results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from ..analysis import model_1024_individual_regions as regions
from ..analysis.model_1023_execution_summary import trajectory_diagnostics
from ...evaluation.execution_comparison import conditional_score, precision_check, score_difference
from ...metrics.autonomous_fit import behavior_features, energy_u

KEYS = regions.KEYS


def paired_energy_difference(a: np.ndarray, b: np.ndarray, observed: np.ndarray) -> dict:
    """Pair complete paths with the same registered seed; retain U-score scaling."""
    if a.shape != b.shape:
        raise ValueError('Energy comparison requires paired features')
    scores = [energy_u(x, observed) for x in (a, b)]
    deleted = []
    n = len(a)
    for x in (a, b):
        distance = cdist(x, x)
        np.fill_diagonal(distance, 0.)
        target = np.linalg.norm(x - observed, axis=1)
        deleted.append((target.sum() - target) / (n - 1)
                       - (distance.sum() / 2 - distance.sum(axis=1)) / ((n - 1) * (n - 2)))
    delta = deleted[0] - deleted[1]
    return dict(difference=scores[0]['value'] - scores[1]['value'],
                mcse=float(np.sqrt((n - 1) / n * np.square(delta - delta.mean()).sum())),
                paths=n, free=scores[0], zero=scores[1])


def numerical_direction(difference: float, mcse: float, *, complete: bool,
                        precise: bool, identical: bool = False) -> str:
    """Describe numerical separation, without a population significance claim."""
    if not complete:
        return 'incomplete'
    if not precise:
        return 'numerically_unresolved'
    if identical:
        return 'same_zero_nominee'
    if difference + 2 * mcse < 0:
        return 'free_lower'
    if difference - 2 * mcse > 0:
        return 'zero_lower'
    return 'not_numerically_separated'


def stage_coverage(case: dict, policy: dict, jobs: list[dict], ledger: dict) -> list[dict]:
    """Count registered and successful work, including stages never reached."""
    setting = policy['search'][case['group']]
    expected = {'screen_0': 2 * setting['initial_pairs'],
                **{f'screen_{i+1}': 2 * setting['round_pairs'] for i in range(setting['rounds'])},
                'confirm': 2 * setting['finalists_per_boundary'],
                'audit': 2 * min(setting['finalists_per_boundary'], policy['search']['audit_per_boundary'])}
    own = [j for j in jobs if j['comparison_case'] == case['id']]
    roles = list(dict.fromkeys([*expected, *(j['comparison_role'] for j in own)]))
    rows = []
    for role in roles:
        selected = [j for j in own if j['comparison_role'] == role]
        successful = [j for j in selected if ledger.get(j['name'], {}).get('status') == 'complete']
        candidates = {j['candidate']['id'] for j in selected}
        complete_candidates = {i for i in candidates if all(
            ledger.get(j['name'], {}).get('status') == 'complete'
            for j in selected if j['candidate']['id'] == i)}
        rows.append(dict(case=case['id'], group=case['group'], role=role,
                         registered_jobs=len(selected), successful_jobs=len(successful),
                         registered_candidates=len(candidates), complete_candidates=len(complete_candidates),
                         minimum_candidates=expected.get(role),
                         complete=bool(selected) and len(successful) == len(selected)
                         and len(complete_candidates) >= expected.get(role, 1)))
    return rows


def fit_complete(coverage: list[dict]) -> bool:
    fitting = [r for r in coverage if r['role'].startswith('screen_')
               or r['role'] in ('confirm', 'audit', 'escalation_all')]
    return bool(fitting) and all(r['complete'] for r in fitting)


class Audit:
    """Bind report inputs and read numerical artifacts through their receipts."""

    def __init__(self, source: Path):
        self.source = source
        self.inputs: dict[str, str] = {}

    def read(self, path: Path):
        self.inputs[str(path.resolve())] = regions.digest(path)
        return regions.read(path)

    def optional(self, name: str, default):
        path = self.source / name
        return self.read(path) if path.exists() else default

    def arrays(self, jobs: list[dict], key: str) -> np.ndarray | None:
        for job in jobs:
            target = Path(job['target'])
            metadata = target.with_suffix('.json')
            if metadata.exists():
                self.read(metadata)
            receipt = self.source / 'ledger' / (job['name'] + '.json')
            if not receipt.exists() or self.read(receipt)['status'] != 'complete':
                return None
        return regions.completed_arrays(jobs, key)


def parameter_rows(case: dict, frozen: dict) -> list[dict]:
    rows = []
    for row in frozen.get('candidates', []):
        point = row['candidate']
        labels = [label for label in ('zero', 'free', 'positive')
                  if frozen.get(label, {}).get('candidate', {}).get('id') == point['id']]
        rows.append(dict(case=case['id'], candidate=point['id'], selected_as=','.join(labels),
                         nll=row['score']['nll'], mcse=row['score']['mcse'],
                         numerical_pass=row.get('precision', {}).get('passed'),
                         **{k: point['hyperparams'][v] for k, v in KEYS.items()}))
    return rows


def evaluate_case(audit: Audit, case: dict, frozen: dict, coverage: list[dict],
                  policy: dict, evaluations: dict, low_evaluations: dict,
                  generation_jobs: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    row = dict(case=case['id'], subject=case['subject'], group=case['group'],
               status=frozen.get('status', 'not_frozen'), fit_jobs_complete=fit_complete(coverage))
    features, paths = [], []
    if row['status'] != 'frozen':
        return row, features, paths
    ids = {k: frozen[k]['candidate']['id'] for k in ('zero', 'free', 'positive')}
    row.update(fit_numerical_resolution=frozen['numerical_resolution'],
               identical_zero_nominee=ids['zero'] == ids['free'],
               free_kappa=frozen['free']['candidate']['hyperparams'][KEYS['inertia']],
               positive_kappa=frozen['positive']['candidate']['hyperparams'][KEYS['inertia']])
    fitted = {k: audit.arrays(frozen[k]['jobs'], 'observed_probability') for k in ('zero', 'free')}
    if all(x is not None for x in fitted.values()):
        difference = score_difference(fitted['free'], fitted['zero'])
        row.update({'fit_' + k: v for k, v in difference.items()})
    high = {r['candidate']['id']: r for r in evaluations.get(case['id'], [])}
    low = {r['candidate']['id']: r for r in low_evaluations.get(case['id'], [])}
    suffix_arrays, suffix_checks = {}, []
    for label in ('zero', 'free'):
        if ids[label] not in high or ids[label] not in low:
            continue
        a = audit.arrays(high[ids[label]]['jobs'], 'observed_probability')
        b = audit.arrays(low[ids[label]]['jobs'], 'observed_probability')
        if a is None or b is None:
            continue
        suffix_arrays[label] = a
        check = precision_check(conditional_score(b, case['cut']),
                                conditional_score(a, case['cut']), policy['search'])
        suffix_checks.append(check['passed'])
        row[label + '_suffix_precision'] = check['passed']
        row[label + '_suffix_particle_bound'] = check['upper_numerical_difference']
    row['suffix_numerical_resolution'] = (len(suffix_checks) == 2 and all(suffix_checks)) if case['group'] != 'full' else None
    if len(suffix_arrays) == 2:
        delta = score_difference(suffix_arrays['free'], suffix_arrays['zero'], case['cut'])
        row.update({'suffix_' + k: v for k, v in delta.items()})
        row['suffix_direction'] = numerical_direction(delta['difference'], delta['mcse'],
            complete=row['fit_jobs_complete'],
            precise=row['fit_numerical_resolution'] and row['suffix_numerical_resolution'],
            identical=row['identical_zero_nominee'])
    _, observed = regions.load_observed(Path(case['template']['folder']))
    kwargs = dict(n_categories=2 if case['condition'] == 1 else 4,
                  block_width=policy['generation']['block_width'],
                  temporal_weight=policy['generation']['temporal_weight'])
    all_choices, seed_banks = {}, {}
    for label in ('zero', 'free', 'positive'):
        own = [j for j in generation_jobs if j['comparison_case'] == case['id']
               and j['candidate']['id'] == ids[label]]
        choices = audit.arrays(own, 'choices')
        if choices is None:
            continue
        seed_banks[label] = [seed for j in own for seed in j['seeds']]
        expected = policy['recovery']['evaluation_paths'] if case['group'] == 'recovery' else policy['generation']['paths']
        row[label + '_paths'] = len(choices)
        row[label + '_paths_complete'] = len(choices) == expected
        all_choices[label] = choices
        for part, start in [('full', 0), ('suffix', case['cut'])]:
            target = behavior_features(observed['choices'][None, start:], observed['categories'][start:], **kwargs)[0]
            feature = behavior_features(choices[:, start:], observed['categories'][start:], **kwargs)
            energy = energy_u(feature, target)
            row.update({f'{label}_{part}_energy': energy['value'], f'{label}_{part}_energy_mcse': energy['mc_se']})
        if case['trials'] < 128:
            continue
        diagnostic = trajectory_diagnostics(choices, observed['categories'], n_categories=kwargs['n_categories'])
        actual = trajectory_diagnostics(observed['choices'][None, :], observed['categories'], n_categories=kwargs['n_categories'])
        for key, values in diagnostic.items():
            finite = np.isfinite(values)
            x, target = values[finite].astype(float), float(actual[key][0])
            features.append(dict(case=case['id'], model=label, feature=key, paths=len(values),
                available=int(finite.sum()), observed=target if np.isfinite(target) else None,
                mean=float(x.mean()) if len(x) else None,
                q05=float(np.quantile(x, .05)) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                q95=float(np.quantile(x, .95)) if len(x) else None,
                observed_midrank=float(((x < target).sum() + .5 * (x == target).sum()) / len(x))
                if len(x) and np.isfinite(target) else None))
        for index in range(len(choices)):
            paths.append(dict(case=case['id'], model=label, path=index, seed=seed_banks[label][index],
                **{k: v[index].item() if np.isfinite(v[index]) else None for k, v in diagnostic.items()}))
    if all(k in all_choices for k in ('zero', 'free')):
        if seed_banks['free'] != seed_banks['zero']:
            raise ValueError('Autonomous paths do not have aligned seeds')
        for part, start in [('full', 0), ('suffix', case['cut'])]:
            a, b = [behavior_features(all_choices[k][:, start:], observed['categories'][start:], **kwargs)
                    for k in ('free', 'zero')]
            target = behavior_features(observed['choices'][None, start:], observed['categories'][start:], **kwargs)[0]
            delta = paired_energy_difference(a, b, target)
            row.update({f'{part}_energy_difference': delta['difference'], f'{part}_energy_mcse': delta['mcse']})
    return row, features, paths


def recovery_audit(audit: Audit, case: dict, frozen: dict, truth: dict, checked: list[dict], comparison: dict) -> dict:
    """Keep primary prediction, numerical uncertainty and optimization distinct."""
    generating = truth['point']
    row = dict(case=case['id'], subject=case['subject'], generating_kappa=generating['hyperparams'][KEYS['inertia']],
               fitted_kappa=comparison.get('free_kappa'), suffix_direction=comparison.get('suffix_direction', 'incomplete'),
               fit_complete=comparison.get('fit_jobs_complete', False),
               fit_precision=comparison.get('fit_numerical_resolution', False),
               suffix_precision=comparison.get('suffix_numerical_resolution', False), optimization_check_complete=False)
    if frozen.get('status') != 'frozen':
        return row
    by_id = {r['candidate']['id']: r for r in checked}
    selected = frozen['zero' if row['generating_kappa'] == 0 else 'free']['candidate']
    if all(p['id'] in by_id for p in (selected, generating)):
        fitted, actual = [audit.arrays(by_id[p['id']]['jobs'], 'observed_probability') for p in (selected, generating)]
        if fitted is not None and actual is not None:
            difference = score_difference(fitted[:, :case['cut']], actual[:, :case['cut']])
            row.update(optimization_check_complete=True,
                       optimization_difference=difference['difference'], optimization_mcse=difference['mcse'],
                       generating_point_better_than_search=difference['difference'] > 2 * difference['mcse'])
    directions = {'free_lower': 'positive', 'zero_lower': 'zero', 'same_zero_nominee': 'zero'}
    row['recovered_class'] = directions.get(row['suffix_direction'], 'unresolved')
    if row['recovered_class'] != 'unresolved' and not row['optimization_check_complete']:
        row['recovered_class'] = 'optimization_unchecked'
    if row.get('generating_point_better_than_search'):
        row['recovered_class'] = 'search_unresolved'
    row['class_agreement'] = (row['recovered_class'] == ('zero' if row['generating_kappa'] == 0 else 'positive')
                              if row['recovered_class'] in ('zero', 'positive') else None)
    return row


def truth_events(audit: Audit, truth: dict) -> dict:
    """Describe how much persistence was actually exercised on a retained path."""
    executed = audit.arrays(truth['generation_jobs'], 'executed')
    rules = audit.arrays(truth['generation_jobs'], 'rule')
    forced = audit.arrays(truth['generation_jobs'], 'forced')
    if any(x is None for x in (executed, rules, forced)):
        return {}
    index = truth['path_index']
    sequence, active = executed[index], rules[index]
    available = np.any(active[1:] == sequence[:-1, None], axis=1)
    same = sequence[1:] == sequence[:-1]
    boundaries = np.r_[0, np.flatnonzero(~same) + 1, len(sequence)]
    return dict(truth_trials=len(sequence), truth_switches=int((~same).sum()),
                truth_forced_switches=int(forced[index, 1:].sum()),
                truth_previous_rule_available=int(available.sum()),
                truth_same_rule_given_available=float(same[available].mean()) if available.any() else None,
                truth_longest_execution_run=int(np.diff(boundaries).max()))


def group_averages(comparisons: list[dict], expected_subjects: int) -> list[dict]:
    """Equal-subject descriptive means; missing people stay visible."""
    rows = []
    for group in ('full', 'prefix'):
        own = [r for r in comparisons if r['group'] == group]
        for metric in ('fit', 'suffix', 'full_energy', 'suffix_energy'):
            selected = [r for r in own if metric + '_difference' in r]
            rows.append(dict(group=group, metric=metric, subjects=len(selected),
                expected_subjects=expected_subjects,
                available_mean=float(np.mean([r[metric + '_difference'] for r in selected])) if selected else None,
                all_subject_mean=float(np.mean([r[metric + '_difference'] for r in selected]))
                if len(selected) == expected_subjects else None,
                mean_mcse=float(np.sqrt(sum(r[metric + '_mcse'] ** 2 for r in selected)) / len(selected))
                if selected else None,
                complete_fits=sum(r['fit_jobs_complete'] for r in selected),
                precise_fits=sum(r.get('fit_numerical_resolution', False) for r in selected),
                precise_suffixes=sum(bool(r.get('suffix_numerical_resolution', False)) for r in selected)))
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    generate(args.source, args.output, args.allow_incomplete)


def generate(source: Path, output: Path, allow_incomplete: bool = False) -> dict:
    if output.exists():
        raise FileExistsError('Report output must be new')
    audit = Audit(source)
    completion = audit.optional('completion.json', {})
    if not completion and not allow_incomplete:
        raise ValueError('Queue has not finished; use --allow-incomplete for a preview')
    protocol = audit.read(source / 'protocol.json')
    if regions.digest(source / 'protocol.json') != audit.read(source / 'protocol_digest.json')['sha256']:
        raise ValueError('Protocol digest changed')
    policy = protocol['policy']
    cases = protocol['cases'] + audit.optional('recovery_cases.json', {'cases': []})['cases']
    jobs = [j for file in sorted((source / 'batches').glob('*.json'))
            if not file.name.endswith('.completion.json') for j in audit.read(file)['jobs']]
    ledger = {f.stem: audit.read(f) for f in (source / 'ledger').glob('*.json')}
    truths = {r['id']: r for r in audit.optional('recovery_truths.json', {'rows': []})['rows']}
    truth_checks = audit.optional('scores/recovery_truth_check.json', {})
    coverage, comparisons, parameters, features, paths, recovery = [], [], [], [], [], []
    for group in ('full', 'prefix', 'recovery'):
        frozen = audit.optional(f'frozen/{group}.json', {})
        evaluations = audit.optional(f'scores/{group}_evaluation.json', {})
        low = audit.optional(f'scores/{group}_evaluation_low.json', {})
        generation = audit.optional(f'generation_jobs/{group}.json', [])
        for case in [c for c in cases if c['group'] == group]:
            own_coverage = stage_coverage(case, policy, jobs, ledger)
            coverage.extend(own_coverage)
            result = frozen.get(case['id'], {})
            parameters.extend(parameter_rows(case, result))
            row, own_features, own_paths = evaluate_case(audit, case, result, own_coverage, policy, evaluations, low, generation)
            comparisons.append(row)
            features.extend(own_features)
            paths.extend(own_paths)
            if case['id'] in truths:
                checked = recovery_audit(audit, case, result, truths[case['id']], truth_checks.get(case['id'], []), row)
                checked.update(truth_events(audit, truths[case['id']]))
                recovery.append(checked)
    expected_recovery = len(policy['recovery']['subjects']) * len(policy['recovery']['kappas']) * policy['recovery']['replicates']
    averages = group_averages(comparisons, len(policy['subjects']))
    candidate_scores = []
    for file in sorted((source / 'scores').glob('*.json')):
        for cid, scores in audit.read(file).items():
            for row in scores:
                candidate_scores.append(dict(case=cid, stage=file.stem, candidate=row['candidate']['id'],
                    kappa=row['candidate']['hyperparams'][KEYS['inertia']], nll=row['score']['nll'],
                    mcse=row['score']['mcse'], repeats=row['score']['repeats'],
                    particles=row['jobs'][0]['particles'], maximum_system_weight=row['score']['maximum_system_weight']))
    summary = dict(queue_finished=bool(completion), queue_completion=completion,
        expected_human_cases=2 * len(policy['subjects']), expected_recovery_cases=expected_recovery,
        available_recovery_cases=sum(c['group'] == 'recovery' for c in cases),
        fit_cases_complete=sum(r['fit_jobs_complete'] for r in comparisons),
        worker_cpu_hours=sum(r['cpu_seconds'] for r in ledger.values()) / 3600,
        coordinator_cpu_hours=audit.optional('coordinator_usage.json', {}).get('cpu_seconds', 0) / 3600,
        continuation=audit.optional('continuation_validation.json', {}),
        unresolved_or_missing_recovery=expected_recovery - sum(r.get('class_agreement') is not None for r in recovery),
        evidence_scope='development humans; specified synthetic backgrounds; MCSE is numerical uncertainty')
    output.mkdir(parents=True)
    for name, rows in [('coverage', coverage), ('comparison', comparisons), ('parameters', parameters),
                       ('features', features), ('paths', paths), ('recovery', recovery),
                       ('group_averages', averages), ('candidate_scores', candidate_scores)]:
        save_csv(output / (name + '.csv'), rows)
    regions.write(output / 'audit.json', summary)
    regions.write(output / 'inputs.json', dict(files=audit.inputs,
                  report_source_sha256=regions.digest(Path(__file__))))
    lines = ['# 执行维持与零维持边界：结果审查', '',
        '队列已结束。' if completion else '队列仍在运行；本页为中间预览。', '',
        '真人均为已查看的开发资料。差值为自由维持减零维持；NLL与Energy较小表示评分较好。',
        '蒙特卡洛标准误描述计算误差。数值方向与人群统计显著性分别解释；参数集合没有置信覆盖含义。', '',
        f'完整执行拟合计划：{summary["fit_cases_complete"]}/{summary["expected_human_cases"] + expected_recovery}例。'
        f'已生成恢复案例：{summary["available_recovery_cases"]}/{expected_recovery}。', '',
        '| 案例 | 拟合任务完整 | κ | 拟合NLL差 | 后段NLL差 | 后段数值方向 |',
        '|---|---|---:|---:|---:|---|']
    for row in comparisons:
        def value(k):
            if k == 'suffix_difference' and row['group'] == 'full':
                return '未设此项'
            return f'{row[k]:.6f}' if k in row else '待完成'
        lines.append(f'| {row["case"]} | {row["fit_jobs_complete"]} | {row.get("free_kappa", "—")} | '
                     f'{value("fit_difference")} | {value("suffix_difference")} | {row.get("suffix_direction", "—")} |')
    lines += ['', 'coverage.csv列出各阶段登记、完成与缺失任务；parameters.csv保留最终审查的全部候选。',
        'candidate_scores.csv保留各阶段同分辨率下的候选评分；跨阶段的粒子数与种子不同，分别读取。',
        'group_averages.csv给出每名真人等权的描述性平均；缺少被试时，全体均值留空。',
        'comparison.csv保留选择预测与全部自主路径的Energy评分；features.csv和paths.csv保留行为特征分布及每条路径。',
        '自主路径均从任务起点生成；其后段评分未以真人前段状态为条件。条件后段NLL使用真人历史。',
        'recovery.csv保留数值未确定与搜索不足，未确定案例的class_agreement留空。',
        '生成点评分优于搜索结果时标记搜索不足；该诊断通过仅表示本次未发现这类不足。',
        '固定实际刺激日程和记录长度；实验自适应刺激及停止规则尚未纳入本轮。口述未参与本轮。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    main()
