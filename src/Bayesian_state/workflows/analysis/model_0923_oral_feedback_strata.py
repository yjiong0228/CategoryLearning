"""Split an already validated oral comparison by observed task feedback.

Run --comparison COMPLETED_COMPARISON --output NEW_DIRECTORY. Preserve all trial
identities and report complete-report means for the prefix, later trials and
whole sequence. Feedback 0, .5 and 1 remain separate, including empty strata.
Conditioned, current-choice-unweighted and their difference use the existing
joint-mass scoring and delete-one-filter MCSE. No parameters or oral readings
are selected by this analysis; all participants remain development cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_0923_oral_parameter_comparison as comparison
from ...evaluation.oral.decision_basis_scoring import (
    DOMAINS, LABELS, label_masses, pool_joint, pooled_mean_and_mcse,
)
from ...evaluation.oral.information_controls import jackknife_error

METRICS = ('conditioned', 'unweighted', 'current_choice_gain')
FEEDBACK = {'all': None, 'incorrect': 0., 'partial': .5, 'correct': 1.}


def stratified_estimates(joint: np.ndarray, probability: np.ndarray,
                        prior: np.ndarray, complete: np.ndarray,
                        feedback: np.ndarray, cut: int) -> dict:
    """Pool filters per trial, then average complete reports within each stratum."""
    joint, probability, prior = map(lambda a: np.asarray(a, float), (joint, probability, prior))
    complete, feedback = np.asarray(complete, bool), np.asarray(feedback, float)
    if (joint.ndim != 4 or joint.shape != prior.shape or joint.shape[2:] != (2, 3)
            or probability.shape != joint.shape[:2] or len(probability) < 2
            or complete.shape != (joint.shape[1],) or feedback.shape != complete.shape
            or not 0 < cut < len(complete) or not np.isin(feedback, (0., .5, 1.)).all()
            or not np.isfinite(probability).all() or np.any(probability <= 0)
            or np.any(probability > 1)):
        raise ValueError('Need aligned full-trial masses, feedback, prefix and multiple filters')
    if complete.any():
        if (not np.isfinite(joint[:, complete]).all() or not np.isfinite(prior[:, complete]).all()
                or np.any(joint[:, complete] < 0) or np.any(prior[:, complete] < 0)):
            raise ValueError('Complete reports need finite nonnegative labelled masses')
        np.testing.assert_allclose(joint[:, complete].sum(-1),
            np.repeat(probability[:, complete, None], 2, axis=2), rtol=0, atol=1e-12)
        np.testing.assert_allclose(prior[:, complete].sum(-1), 1., rtol=0, atol=1e-12)
    ordinal = np.arange(len(complete))
    windows = dict(whole=np.ones(len(complete), bool), training=ordinal < cut, test=ordinal >= cut)
    estimates = {}
    for window, in_window in windows.items():
        for stratum, value in FEEDBACK.items():
            selected = in_window if value is None else in_window & (feedback == value)
            mask = selected & complete
            conditioned = pooled_mean_and_mcse(joint, probability, mask)
            unweighted = pooled_mean_and_mcse(prior, np.ones_like(probability), mask)
            point = np.stack((conditioned[0], unweighted[0], conditioned[0] - unweighted[0]))
            deleted = np.stack((conditioned[2], unweighted[2], conditioned[2] - unweighted[2]), axis=1)
            estimates[window, stratum] = dict(point=point, deleted=deleted,
                mcse=jackknife_error(deleted), reports=int(mask.sum()),
                trials=int(selected.sum()), mask=mask)
    return estimates


def load_bank(directory: Path, subject: int, arm: str, scored: tuple) -> tuple[dict, dict]:
    """Bind feedback to the exact behavior trials used by the saved oral scores."""
    alignment = comparison.alignment
    manifest = alignment.read(directory / 'manifest.json')
    source, audit = Path(manifest['source']), Path(manifest['oral_audit'])
    matches = pd.read_csv(audit / 'component_matches.csv')
    trials = scored[0].sort_values('state_trial').reset_index(drop=True)
    records = [(p, alignment.read(p)) for p in sorted((source / 'states').glob('*.json'))]
    records = sorted([(p, r) for p, r in records if (r['subject'], r['arm']) == (subject, arm)],
                     key=lambda item: item[1]['seeds'][0])
    job = alignment.read(source / 'jobs' / records[0][0].name)
    folder = Path(job['folder'])
    meta = alignment.read(folder / 'prepared.json')
    behavior = []
    inputs = {}
    for name in ('training.npz', 'sealed_test.npz'):
        file = folder / name
        if alignment.digest(file) != meta['array_sha256'][name]:
            raise ValueError('Prepared behavior array changed')
        inputs[str(file.resolve())] = alignment.digest(file)
        with np.load(file) as saved:
            behavior.append({key: saved[key].copy() for key in ('choices', 'categories', 'feedback')})
    observed = {key: np.concatenate([b[key] for b in behavior]) for key in behavior[0]}
    np.testing.assert_array_equal(observed['choices'], trials.choice.to_numpy())
    np.testing.assert_array_equal(trials.state_trial, np.arange(len(trials)))
    cut = meta['training_trials']
    if not (trials.training_trials == cut).all():
        raise ValueError('Prefix boundary changed')
    if meta['condition'] in (1, 2):
        np.testing.assert_array_equal(observed['feedback'],
            (observed['choices'] == observed['categories']).astype(float))
    joint, probability, prior, seeds = [], [], [], []
    labels = None
    for file, record in records:
        target = file.with_suffix('.npz')
        if alignment.digest(target) != record['array_sha256']:
            raise ValueError('State arrays changed')
        with np.load(target) as saved:
            np.testing.assert_array_equal(saved['choices'], observed['choices'])
            if labels is None:
                labels = alignment.make_labels(trials, matches, *saved['choice_component_joint_mass'].shape[1:])
            joint.append(label_masses(saved['choice_component_joint_mass'], labels))
            prior.append(label_masses(saved['prior_component_mass'], labels))
            probability.append(saved['choice_probability'].copy())
        seeds.append(record['seeds'][0])
    joint, prior, probability = map(np.asarray, (joint, prior, probability))
    valid = np.isfinite(joint).all(axis=(0, 2, 3))
    scores = np.full((3, len(trials), 2, 3), np.nan)
    scores[0, valid] = pool_joint(joint[:, valid], probability[:, valid])
    scores[1] = prior.mean(axis=0)
    scores[2] = scores[0] - scores[1]
    for d, domain in enumerate(DOMAINS):
        for k, label in enumerate(LABELS):
            for m, prefix in ((0, ''), (1, 'prior_')):
                np.testing.assert_allclose(scores[m, :, d, k], trials[prefix + domain + '_' + label],
                                           atol=1e-12, rtol=0, equal_nan=True)
    estimates = stratified_estimates(joint, probability, prior, trials.scope.eq('complete').to_numpy(),
                                    observed['feedback'], cut)
    old_summary = pd.DataFrame(scored[1])
    for (window, stratum), values in estimates.items():
        if stratum != 'all':
            continue
        for d, domain in enumerate(DOMAINS):
            previous = old_summary[(old_summary.window == window) & (old_summary.domain == domain)
                                   & (old_summary.subset == 'complete')].iloc[0]
            if values['reports'] != previous.reports:
                raise ValueError('Complete-report denominator changed')
            np.testing.assert_allclose(values['point'][0, d], previous[list(LABELS)].to_numpy(float),
                                       atol=1e-12, rtol=0, equal_nan=True)
    return dict(trials=trials, feedback=observed['feedback'], categories=observed['categories'],
                scores=scores, estimates=estimates, seeds=np.asarray(seeds)), inputs


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral feedback-stratification directory')
    alignment = comparison.alignment
    original_manifest = alignment.read(source / 'manifest.json')
    if (original_manifest['status'] != 'complete'
            or not original_manifest['original_scores_reproduced']
            or not original_manifest['revised_scores_reproduced']):
        raise ValueError('Need a completed and reproduced oral comparison')
    inputs = dict(original_manifest['input_sha256'])
    inputs[str((source / 'manifest.json').resolve())] = alignment.digest(source / 'manifest.json')
    for name, sha in original_manifest['artifacts'].items():
        inputs[str((source / name).resolve())] = sha
    for name, sha in original_manifest['source_sha256'].items():
        inputs[str((alignment.ROOT / name).resolve())] = sha
    for name, sha in inputs.items():
        if alignment.digest(Path(name)) != sha:
            raise ValueError('Completed comparison dependency changed')
    subject, arm = original_manifest['subject'], original_manifest['arm']
    banks, identities = {}, {}
    for variant in ('original', 'revised'):
        directory = Path(original_manifest[variant])
        scored, identity, more = comparison.load_group(directory, subject, arm)
        if identity != original_manifest[variant + '_parameters']:
            raise ValueError('Comparison parameter or state identity changed')
        inputs.update(more)
        banks[variant], more = load_bank(directory, subject, arm, scored)
        inputs.update(more); identities[variant] = identity
    comparison.validate_comparison(identities['original'], identities['revised'], original_manifest['contrast_kind'])
    old, new = banks['original'], banks['revised']
    pd.testing.assert_frame_equal(old['trials'][comparison.IDENTITY], new['trials'][comparison.IDENTITY])
    for name in ('feedback', 'categories'):
        np.testing.assert_array_equal(old[name], new[name])
    relation = original_manifest['seed_relation']
    if relation == 'paired':
        np.testing.assert_array_equal(old['seeds'], new['seeds'])
    elif relation != 'independent' or set(old['seeds']) & set(new['seeds']):
        raise ValueError('Invalid relation between filter seed banks')
    rows, deleted = [], {}
    for key, a in old['estimates'].items():
        b = new['estimates'][key]
        np.testing.assert_array_equal(a['mask'], b['mask'])
        delta = b['point'] - a['point']
        error = (jackknife_error(b['deleted'] - a['deleted']) if relation == 'paired'
                 else np.hypot(a['mcse'], b['mcse']))
        window, stratum = key
        for variant, bank in banks.items():
            deleted[variant + '_' + window + '_' + stratum] = bank['estimates'][key]['deleted']
        for m, metric in enumerate(METRICS):
            for d, domain in enumerate(DOMAINS):
                row = dict(subject=subject, arm=arm, window=window, feedback_group=stratum,
                    trials=a['trials'], complete_reports=a['reports'], metric=metric, domain=domain,
                    filter_seeds=len(old['seeds']))
                for k, label in enumerate(LABELS):
                    row.update({f'original_{label}': a['point'][m, d, k],
                                f'revised_{label}': b['point'][m, d, k],
                                f'{label}_difference': delta[m, d, k],
                                f'{label}_difference_mcse': error[m, d, k]})
                rows.append(row)
    trials = old['trials'][comparison.IDENTITY].copy()
    trials['feedback'] = old['feedback']; trials['category'] = old['categories']
    for m, metric in enumerate(METRICS):
        for d, domain in enumerate(DOMAINS):
            for k, label in enumerate(LABELS):
                name = metric + '_' + domain + '_' + label
                trials['original_' + name] = old['scores'][m, :, d, k]
                trials['revised_' + name] = new['scores'][m, :, d, k]
                trials[name + '_difference'] = new['scores'][m, :, d, k] - old['scores'][m, :, d, k]
    summary = pd.DataFrame(rows)
    for name, sha in inputs.items():
        if alignment.digest(Path(name)) != sha:
            raise ValueError('Input changed during stratification')
    output.mkdir(parents=True)
    summary.to_csv(output / 'scores.csv', index=False)
    trials.to_csv(output / 'trials.csv', index=False)
    with (output / 'jackknife.npz').open('xb') as stream:
        np.savez_compressed(stream, **deleted, original_seeds=old['seeds'], revised_seeds=new['seeds'])
    note = ('每个时间范围按实际反馈分为错误、部分正确和正确；完整口述是各组评分分母，空组保留。'
        'conditioned使用本题选择条件下的依据质量；unweighted去掉本题选择似然加权，保留既往行为条件推断，'
        '两者都按实际所选类别确定局部依据。current_choice_gain为二者之差。'
        '差值为修订方案减原方案，模拟误差沿用原比较的完整滤波种子配对或独立关系。'
        '该分析用于开发诊断，行为参数和口述读法均保持已冻结值。')
    display = summary[(summary.domain == 'structural') & (summary.metric == 'conditioned')]
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>按反馈分层的口述比较</title><style>body{max-width:1400px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>按实际反馈分层的口述表征比较</h1><p>' + note + '</p>'
        + display.to_html(index=False, float_format=lambda x: f'{x:.6f}')
        + '<p><a href="scores.csv">全部分层评分</a> · <a href="trials.csv">全部试次</a> · <a href="jackknife.npz">整次滤波删除结果</a></p></html>')
    shutil.copy2(__file__, output / 'analysis_source.py')
    manifest = dict(status='complete', comparison=str(source.resolve()), subject=subject, arm=arm,
        seed_relation=relation, contrast_kind=original_manifest['contrast_kind'],
        data_use=original_manifest['data_use'], trials=len(trials), score_rows=len(summary),
        complete_reports=int(old['trials'].scope.eq('complete').sum()),
        original_and_revised_scores_reproduced=True, input_sha256=inputs,
        source_sha256={str(Path(__file__).relative_to(alignment.ROOT)): alignment.digest(Path(__file__))},
        artifacts={p.name: alignment.digest(p) for p in output.iterdir()})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison, args.output)
