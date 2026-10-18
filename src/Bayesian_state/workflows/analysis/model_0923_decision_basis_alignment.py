"""Align frozen behavior-only decision bases to the audited oral constraints.

Use --source REPLAY --oral-audit AUDIT --output NEW_DIRECTORY. Optional
--subjects selects a declared subset, whose complete seed groups must exist.
Whole-report scores use the unchanged complete-encoding mask. Fragment scores
remain identified in the trial table. Monte Carlo SE uses whole filter seeds.
--reference-model defaults to BASE; POOL supports the recalled-search comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...evaluation.oral.decision_basis import joint_availability
from ...evaluation.oral.decision_basis_scoring import (
    DOMAINS, LABELS, label_masses, pool_joint, pooled_mean_and_mcse,
    prefix_baseline, support_rule_masks,
)

ROOT = Path(__file__).resolve().parents[4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def make_labels(trials: pd.DataFrame, matches: pd.DataFrame, rules: int, components: int) -> np.ndarray:
    """Keep the entire observed conjunction and every catalogue component."""
    lookup = {}
    for query, group in matches.loc[matches.query_id.isin(trials.query_id.dropna())].groupby('query_id'):
        values = np.full((rules, components, 2), -1, dtype=np.int8)
        if group.duplicated(['rule', 'component']).any():
            raise ValueError('Duplicate catalogue match')
        for domain, name in enumerate(DOMAINS):
            codes = group[name].map(dict(zip(LABELS, range(3))))
            if codes.isna().any():
                raise ValueError('Unrecognized component verdict')
            values[group.rule.to_numpy(int), group.component.to_numpy(int), domain] = codes
        lookup[query] = values
    missing = np.full((rules, components, 2), -1, dtype=np.int8)
    return np.stack([lookup.get(key, missing) for key in trials.query_id])


def align_identities(oral: pd.DataFrame, job: dict, n_trials: int) -> tuple[pd.DataFrame, int]:
    """Verify choice, stimulus, feedback and original session/trial identities."""
    folder = Path(job['folder'])
    meta = read(folder / 'prepared.json')
    data_path = Path(meta['paths']['learning_data'])
    if digest(data_path) != meta['learning_data_sha256']:
        raise ValueError('Learning data changed since original fit')
    frame = pd.read_csv(data_path)
    frame = frame.loc[frame.iSub.eq(meta['subject'])]
    if 'miss' in frame:
        frame = frame.loc[frame['miss'].eq(0)]
    if frame.duplicated(['iSession', 'iTrial']).any():
        raise ValueError('Ambiguous original trial identity')
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        if digest(folder / name) != meta['array_sha256'][name]:
            raise ValueError('Prepared behavior changed')
        with np.load(folder / name) as values:
            parts.append({key: values[key] for key in ('stimulus', 'choices', 'feedback')})
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    np.testing.assert_array_equal(frame[[f'feature{i}' for i in range(1, 5)]].to_numpy(), arrays['stimulus'])
    np.testing.assert_array_equal(frame.choice.to_numpy(), arrays['choices'])
    np.testing.assert_array_equal(frame.feedback.to_numpy(), arrays['feedback'])
    identity = frame[['iSession', 'iTrial']].iloc[:n_trials].copy()
    identity['state_trial'] = np.arange(n_trials)
    selected = oral.loc[oral.iSub.eq(meta['subject'])]
    joined = identity.merge(selected, on=['iSession', 'iTrial'], how='left', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Oral report identities differ from behavior replay')
    np.testing.assert_array_equal(joined.choice.to_numpy(), arrays['choices'][:n_trials])
    return joined.drop(columns='_merge'), min(meta['training_trials'], n_trials)


def group_scores(source: Path, records: list[tuple[Path, dict]], oral: pd.DataFrame,
                 matches: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    records = sorted(records, key=lambda item: item[1]['seeds'][0])
    first_file, first = records[0]
    job = read(source / 'jobs' / first_file.name)
    trials, cut = align_identities(oral, job, first['trials'])
    numerator, denominator, prior, availability, seeds = [], [], [], [], []
    labels = None
    for file, record in records:
        target = file.with_suffix('.npz')
        if digest(target) != record['array_sha256'] or not record['exact_behavior_replay']:
            raise ValueError('Need unchanged exact-replay arrays')
        if record['catalogue_signature'] != first['catalogue_signature'] or record['trials'] != len(trials):
            raise ValueError('Incompatible seed trajectories')
        with np.load(target) as saved:
            np.testing.assert_array_equal(saved['choices'], trials.choice.to_numpy())
            joint = saved['choice_component_joint_mass']
            probability = saved['choice_probability']
            if labels is None:
                labels = make_labels(trials, matches, *joint.shape[1:])
                represented = trials.query_id.notna()
                if not trials.loc[represented, 'catalogue_version'].eq(record['catalogue_version']).all():
                    raise ValueError('Oral audit and model use different catalogues')
            numerator.append(joint)
            denominator.append(probability)
            prior.append(label_masses(saved['prior_component_mass'], labels))
            available = np.zeros((len(trials), 2, 2))
            for domain in range(2):
                for bound, mask in enumerate(support_rule_masks(labels, domain)):
                    available[:, domain, bound] = joint_availability(
                        saved['active_packed'], saved['conditional_particle_weights'], mask)
            availability.append(available)
            seeds.append(record['seeds'][0])
    joint, probability, prior, availability = map(np.asarray, (numerator, denominator, prior, availability))
    basis = pool_joint(joint, probability)
    score_joint = np.stack([label_masses(value, labels) for value in joint])
    valid = np.any(labels >= 0, axis=(1, 2, 3))
    score = np.full((len(trials), 2, 3), np.nan)
    score[valid] = pool_joint(score_joint[:, valid], probability[:, valid])
    availability_joint = availability * probability[:, :, None, None]
    pooled_availability = pool_joint(availability_joint, probability)
    pooled_availability[~valid] = np.nan
    static = prefix_baseline(basis, trials.choice.to_numpy(), cut)
    static_valid = np.isfinite(static).all(axis=(1, 2)) & valid
    static_score = np.full_like(score, np.nan)
    if static_valid.any():
        static_score[static_valid] = label_masses(static[static_valid], labels[static_valid])
    result = trials.copy()
    result['arm'] = first['arm']
    result['training_trials'] = cut
    result['phase'] = np.where(np.arange(len(trials)) < cut, 'training', 'test')
    result['filter_seeds'] = len(seeds)
    result['mean_choice_probability'] = probability.mean(axis=0)
    for d, domain in enumerate(DOMAINS):
        for k, label in enumerate(LABELS):
            key = f'{domain}_{label}'
            result[key] = score[:, d, k]
            result[f'prior_{key}'] = prior.mean(axis=0)[:, d, k]
            result[f'prefix_static_{key}'] = static_score[:, d, k]
        for b, bound in enumerate(('all_components', 'any_component')):
            result[f'{domain}_availability_{bound}'] = pooled_availability[:, d, b]
    ordinal = np.arange(len(trials))
    windows = dict(whole=np.ones(len(trials), bool), training=ordinal < cut, test=ordinal >= cut,
                   first64=ordinal < 64, last64=ordinal >= max(0, len(trials) - 64))
    summaries, jackknife = [], {}
    complete = trials.scope.eq('complete').to_numpy()
    for window, in_window in windows.items():
        for domain, name in enumerate(DOMAINS):
            for subset in ('complete', 'catalogue_supported'):
                mask = complete & in_window
                if subset == 'catalogue_supported':
                    mask &= np.any(labels[..., domain] == 0, axis=(1, 2))
                point, se, estimates = pooled_mean_and_mcse(score_joint[:, :, domain], probability, mask)
                available_point, available_se, _ = pooled_mean_and_mcse(availability_joint[:, :, domain], probability, mask)
                row = dict(subject=first['subject'], condition=first['condition'], arm=first['arm'],
                           window=window, domain=name, subset=subset, window_trials=int(in_window.sum()),
                           reports=int(mask.sum()), full_reports=int((complete & in_window).sum()),
                           encoded_fragments=int((trials.scope.eq('parsed_fragment').to_numpy() & in_window).sum()),
                           filter_seeds=len(seeds))
                for label, value, error in zip(LABELS, point, se):
                    row[label] = value; row[f'{label}_mcse'] = error
                for b, bound in enumerate(('all_components', 'any_component')):
                    row[f'availability_{bound}'] = available_point[b]
                    row[f'availability_{bound}_mcse'] = available_se[b]
                row['prior_support'] = float(prior[:, mask, domain, 0].mean()) if mask.any() else np.nan
                static_mask = mask & static_valid
                row['static_reports'] = int(static_mask.sum())
                row['prefix_static_support'] = float(static_score[static_mask, domain, 0].mean()) if static_mask.any() else np.nan
                row['dynamic_support_on_static_reports'] = float(score[static_mask, domain, 0].mean()) if static_mask.any() else np.nan
                summaries.append(row)
                jackknife[window, name, subset] = dict(seeds=np.array(seeds), values=estimates, point=point)
    return result, summaries, jackknife


def estimation_context(protocol: dict) -> tuple[dict, str]:
    """Carry parameter data use into oral reports while retaining time-window keys."""
    scope = protocol.get('parameter_estimation_scope', 'original_evaluation_protocol')
    use = protocol.get('evaluation_use', 'original_evaluation_protocol')
    metadata = dict(parameter_estimation_scope=scope, evaluation_use=use,
                    window_definition='original_chronological_boundary')
    if use == 'development_retrospective_comparison':
        if scope not in ('prefix', 'full'):
            raise ValueError('Retrospective oral replay needs a declared selection scope')
        selected = '完整行为序列' if scope == 'full' else '原前段行为序列'
        note = (f'<p>参数按{selected}的选择似然选定。候选筛选同时使用前段与全程目标，'
                '本次属于回顾性开发诊断。表中的training／test沿用原时间分段标记；'
                '后段行为属于本次已使用数据。口述内容仅用于检验冻结参数后的表征。</p>')
    else:
        note = ''
    return metadata, note


def run(source: Path, audit: Path, output: Path, subjects: list[int] | None = None,
        reference_model: str = 'BASE') -> None:
    if output.exists():
        raise FileExistsError('Use a new oral-state alignment directory')
    protocol = read(source / 'protocol.json')
    scope_metadata, scope_note = estimation_context(protocol)
    manifest = read(audit / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Oral audit is incomplete')
    for name, sha in manifest['input_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Oral source data changed')
    for name in ('trials.csv', 'component_matches.csv'):
        if digest(audit / name) != manifest['artifacts'][name]:
            raise ValueError('Audited report constraints changed')
    settings = protocol['settings']
    selected = subjects if subjects is not None else settings['subjects'][:1] if protocol['smoke'] else settings['subjects']
    arms = settings['arms'][:1] if protocol['smoke'] else settings['arms']
    if len(arms) > 1 and reference_model not in arms:
        raise ValueError('Reference model must be present in the declared comparison')
    if not set(selected) <= set(settings['subjects']) or len(set(selected)) != len(selected):
        raise ValueError('Subjects must be a unique subset of the declared replay cohort')
    groups = {}
    for file in (source / 'states').glob('*.json'):
        record = read(file)
        if record['subject'] in selected and record['arm'] in arms:
            groups.setdefault((record['subject'], record['arm']), []).append((file, record))
    for subject in selected:
        for arm in arms:
            records = groups.get((subject, arm), [])
            expected = 1 if protocol['smoke'] else settings['seeds_per_arm']
            if len(records) != expected or len({r['seeds'][0] for _, r in records}) != expected:
                raise ValueError(f'Incomplete or duplicate seed group: {subject} {arm}')
    oral = pd.read_csv(audit / 'trials.csv')
    matches = pd.read_csv(audit / 'component_matches.csv')
    output.mkdir(parents=True)
    trial_tables, summaries, jackknife, hashes = [], [], {}, {}
    for key, records in sorted(groups.items()):
        table, rows, estimates = group_scores(source, records, oral, matches)
        trial_tables.append(table); summaries.extend(rows); jackknife[key] = estimates
        hashes.update({str(file.with_suffix('.npz').resolve()): record['array_sha256'] for file, record in records})
        print(f'Aligned subject {key[0]} {key[1]}: {len(table)} trials, {len(records)} seeds', flush=True)
    contrasts = []
    for (subject, arm), estimates in jackknife.items():
        if arm == reference_model or (subject, reference_model) not in jackknife:
            continue
        for key, estimate in estimates.items():
            base = jackknife[subject, reference_model][key]
            np.testing.assert_array_equal(estimate['seeds'], base['seeds'])
            difference = estimate['values'] - base['values']
            n = len(difference)
            error = np.sqrt((n - 1) / n * np.sum((difference - difference.mean(axis=0)) ** 2, axis=0)) if n > 1 else np.full(3, np.nan)
            row = dict(subject=subject, arm=arm, reference=reference_model, window=key[0], domain=key[1], subset=key[2])
            for k, label in enumerate(LABELS):
                row[f'{label}_difference'] = estimate['point'][k] - base['point'][k]
                row[f'{label}_difference_mcse'] = error[k]
            contrasts.append(row)
    trials, summary = pd.concat(trial_tables, ignore_index=True), pd.DataFrame(summaries)
    trials.to_csv(output / 'trials.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(output / 'contrasts.csv', index=False)
    selected_summary = summary.loc[summary.subset.eq('complete') & summary.window.eq('test')]
    display = selected_summary[['subject', 'arm', 'domain', 'reports', 'support', 'support_mcse', 'conflict',
                                'undetermined', 'availability_all_components', 'availability_any_component',
                                'prefix_static_support']]
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述与作答依据</title>'
            '<style>body{font:16px sans-serif;max-width:1300px;margin:30px auto}td,th{padding:6px;border:1px solid #ccc}'
            'table{border-collapse:collapse}</style><h1>口述约束与行为条件作答依据</h1>'
            '<p>参数、状态和分量权重仅由行为模型确定。整组口述判断在同一局部依据内检验。'
            f'结果为内容支持质量；完整报告的生成概率仍需表达过程模型。本次{len(selected)}人均为开发样本。</p>'
            '<p>各次滤波先合并选择—依据联合质量，再按实际选择归一化。mcse为删除一次完整滤波重复所得的'
            'jackknife模拟标准误，未包含参数和被试抽样不确定性。完整解析报告保留共同分母。</p>'
            '<p>固定基准对训练段各类别的依据分布取试次平均，冻结后用于后段。动态与固定基准的差异同时涉及'
            '状态变化和刺激相关的局部读出，单项差异尚不足以定位其原因。</p>'
            + scope_note
            + ('<p><strong>短序列流程验证。</strong></p>' if protocol['smoke'] else '')
            + '<h2>后段结果</h2>' + display.to_html(index=False, float_format=lambda x: f'{x:.4f}')
            + '<ul>' + ''.join(f'<li><a href="{html.escape(name)}">{name}</a></li>' for name in ('trials.csv','summary.csv','contrasts.csv'))
            + '</ul></html>')
    (output / 'index.html').write_text(page)
    source_paths = [Path(__file__), ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis.py',
                    ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py']
    result = dict(status='complete', smoke=protocol['smoke'], subjects=selected, arms=arms, reference_model=reference_model,
                  source=str(source.resolve()), oral_audit=str(audit.resolve()), state_arrays=hashes,
                  source_protocol_sha256=digest(source / 'protocol.json'), oral_manifest_sha256=digest(audit / 'manifest.json'),
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                  artifacts={p.name: digest(p) for p in output.iterdir()},
                  full_replay_complete=(source / 'completion.json').exists(),
                  primary_denominator='unchanged_complete_reports_shared_by_all_arms',
                  secondary_denominator='complete_reports_with_catalogue_support_fixed_independently_of_fitted_states',
                  uncertainty='delete_one_whole_filter_seed_jackknife_mcse', **scope_metadata)
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--oral-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--reference-model', default='BASE')
    args = parser.parse_args()
    run(args.source, args.oral_audit, args.output, args.subjects, args.reference_model)


if __name__ == '__main__':
    main()
