"""Supplement a completed whole-report alignment with equal-predicate coverage.

Use --alignment COMPLETE_ALIGNMENT --output NEW_DIRECTORY. Existing reports,
whole scores, states, behavior parameters and reading conventions stay frozen.
Only complete reports receive the supplemental score. Every report contributes
equally after averaging its predicates. Full-filter deletions quantify Monte
Carlo error; paired model comparisons require identical filter seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd

from . import model_0923_decision_basis_alignment as alignment
from .model_0923_oral_constraint_audit import catalogue
from ...evaluation.oral.predicate_alignment import fractional_masses, predicate_fractions
from ...evaluation.oral.decision_basis_scoring import DOMAINS, LABELS, pool_joint, pooled_mean_and_mcse, prefix_baseline
from ...evaluation.oral.information_controls import jackknife_error

ROOT = Path(__file__).resolve().parents[4]


def run(anchor: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new predicate alignment output')
    inputs = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        path = path.resolve(); sha = alignment.digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Changed frozen input: ' + str(path))
        inputs[str(path)] = sha
        return path

    manifest = alignment.read(bind(anchor / 'manifest.json'))
    if manifest['status'] != 'complete' or manifest['smoke'] or not manifest['full_replay_complete']:
        raise ValueError('A completed full-sequence alignment is required')
    for name, sha in manifest['artifacts'].items():
        bind(anchor / name, sha)
    source, audit = Path(manifest['source']), Path(manifest['oral_audit'])
    protocol = alignment.read(bind(source / 'protocol.json', manifest['source_protocol_sha256']))
    if alignment.read(bind(source / 'completion.json'))['status'] != 'complete':
        raise ValueError('Incomplete state replay')
    coding = alignment.read(bind(audit / 'manifest.json', manifest['oral_manifest_sha256']))
    if coding['status'] != 'complete' or coding['config']['sample_role'] != 'development':
        raise ValueError('Only the declared development corpus is supported')
    for name, sha in coding['input_sha256'].items():
        bind(ROOT / name, sha)
    for name in ('trials.csv', 'component_matches.csv'):
        bind(audit / name, coding['artifacts'][name])
    oral = pd.read_csv(audit / 'trials.csv')
    matches = pd.read_csv(audit / 'component_matches.csv')
    original = pd.read_csv(anchor / 'trials.csv')
    tolerance = coding['config']['tolerance']
    groups = {}
    for path, sha in manifest['state_arrays'].items():
        file = Path(path).with_suffix('.json'); record = alignment.read(bind(file))
        if record['array_sha256'] != sha or not record['exact_behavior_replay']:
            raise ValueError('Require exact original behavior replay')
        bind(Path(path), sha)
        groups.setdefault((record['subject'], record['arm']), []).append((file, record))
    expected = {(s, a) for s in manifest['subjects'] for a in manifest['arms']}
    if set(groups) != expected:
        raise ValueError('State groups differ from the completed alignment')
    tables, summaries, estimates, component_rows, query_cache = [], [], {}, [], {}
    single_checks = 0
    coverage_checks = 0
    checked_queries = set()
    for (subject, arm), records in sorted(groups.items()):
        records.sort(key=lambda item: item[1]['seeds'][0])
        seeds = np.array([r['seeds'][0] for _, r in records])
        if len(seeds) != protocol['settings']['seeds_per_arm'] or len(set(seeds)) != len(seeds):
            raise ValueError('Incomplete or duplicate filter group')
        first = records[0][1]
        table = original[original.iSub.eq(subject) & original.arm.eq(arm)].sort_values('state_trial').reset_index(drop=True).copy()
        n = len(table); complete = table.scope.eq('complete').to_numpy()
        cut = int(table.training_trials.iloc[0]); ordinal = np.arange(n)
        if not n or n != first['trials'] or table.state_trial.tolist() != ordinal.tolist():
            raise ValueError('Incomplete ordered report sequence')
        joint, prior, probability, fractions = [], [], [], None
        for file, record in records:
            if any(record[key] != first[key] for key in ('candidate', 'particles', 'trials', 'catalogue_signature', 'component_policy', 'timing')):
                raise ValueError('Mixed scientific settings within a seed group')
            job = alignment.read(bind(source / 'jobs' / file.name))
            if job['candidate'] != record['candidate'] or job['seeds'] != record['seeds']:
                raise ValueError('Job and state identity differ')
            bind(Path(job['original_job']), job['original_job_sha256'])
            ref = bind(Path(job['reference']), job['reference_sha256'])
            task = Path(job['folder']); meta = alignment.read(bind(task / 'prepared.json'))
            bind(Path(meta['paths']['learning_data']), meta['learning_data_sha256'])
            for name, sha in meta['array_sha256'].items():
                bind(task / name, sha)
            checked, checked_cut = alignment.align_identities(oral, job, n)
            if cut != checked_cut:
                raise ValueError('Training boundary changed')
            pd.testing.assert_frame_equal(table[list(checked.columns)], checked, check_dtype=False)
            with np.load(file.with_suffix('.npz')) as saved:
                with np.load(ref) as reference:
                    np.testing.assert_array_equal(saved['prediction'], reference['probabilities'][0, :n])
                    np.testing.assert_array_equal(saved['choices'], reference['choices'][:n])
                mass = saved['choice_component_joint_mass']; p = saved['choice_probability']
                np.testing.assert_array_equal(saved['choices'], table.choice)
                np.testing.assert_allclose(mass.sum((1, 2)), p, atol=1e-12, rtol=0)
                np.testing.assert_allclose(saved['prior_component_mass'].sum((1, 2)), 1., atol=1e-12, rtol=0)
                if fractions is None:
                    fractions = np.full((*mass.shape, 2, 3), np.nan)
                    counts = np.full(n, np.nan)
                    space = catalogue(2 if first['condition'] == 1 else 4)
                    if space.version != record['catalogue_version']:
                        raise ValueError('Catalogue differs from frozen state replay')
                    whole_labels = alignment.make_labels(table, matches, *mass.shape[1:])
                    frozen_matches = {q:g.set_index(['rule','component']) for q,g in matches.groupby('query_id')}
                    for t in np.flatnonzero(complete):
                        row = table.iloc[t]; tokens = tuple(json.loads(row.tokens))
                        key = (int(first['condition']), int(row.choice), tokens)
                        if key not in query_cache:
                            values = np.full((*mass.shape[1:], 2, 3), np.nan)
                            for rule in space:
                                for c, component in enumerate(rule.categories[int(row.choice)-1].components):
                                    value = predicate_fractions(tokens, component.A, component.b, tolerance=tolerance)
                                    values[rule.index, c] = value
                                    component_rows.append(dict(query_id=row.query_id, condition=first['condition'], choice=row.choice,
                                        tokens=row.tokens, predicates=len(tokens), rule=rule.index, component=c,
                                        **{d+'_'+label: value[i,k] for i,d in enumerate(DOMAINS) for k,label in enumerate(LABELS)}))
                            query_cache[key] = values
                        fractions[t] = query_cache[key]; counts[t] = len(tokens)
                        if key not in checked_queries:
                            frozen = frozen_matches[row.query_id]
                            if frozen.index.has_duplicates:
                                raise ValueError('Duplicate frozen component identity')
                            r = frozen.index.get_level_values('rule').to_numpy(int)
                            c = frozen.index.get_level_values('component').to_numpy(int)
                            np.testing.assert_allclose(fractions[t,r,c,0,0], frozen.explicit_predicates/len(tokens), atol=1e-12, rtol=0)
                            for d, domain in enumerate(DOMAINS):
                                np.testing.assert_array_equal(fractions[t,r,c,d,0] == 1., frozen[domain].eq('support'))
                            coverage_checks += 3*len(frozen)
                            checked_queries.add(key)
                        if len(tokens) == 1:
                            covered = np.isfinite(fractions[t]).all((-1,-2))
                            for d in range(2):
                                for k in range(3):
                                    np.testing.assert_array_equal(fractions[t,...,d,k][covered], (whole_labels[t,...,d][covered] == k).astype(float))
                                    single_checks += int(covered.sum())
                    table['predicate_count'] = counts
                # Coverage validation occurs before pooling; zero-mass padding
                # cannot silently absorb a missing catalogue verdict.
                fractional_masses(mass, fractions)
                prior.append(fractional_masses(saved['prior_component_mass'], fractions))
                joint.append(mass.copy()); probability.append(p.copy())
        joint, prior, probability = map(np.asarray, (joint, prior, probability))
        numerator = np.stack([fractional_masses(value, fractions) for value in joint])
        scores = np.full((n,2,3), np.nan)
        scores[complete] = pool_joint(numerator[:,complete], probability[:,complete])
        prior_score = prior.mean(0)
        basis = pool_joint(joint, probability)
        static = prefix_baseline(basis, table.choice.to_numpy(), cut)
        static_valid = complete & np.isfinite(static).all((1,2))
        static_score = np.full_like(scores, np.nan)
        if static_valid.any():
            static_score[static_valid] = fractional_masses(static[static_valid], fractions[static_valid])
        for d, domain in enumerate(DOMAINS):
            for k, label in enumerate(LABELS):
                name = domain+'_'+label
                for prefix, value in [('predicate_', scores), ('predicate_prior_', prior_score), ('predicate_prefix_static_', static_score)]:
                    table[prefix+name] = value[:,d,k]
                    old_prefix = {'predicate_':'', 'predicate_prior_':'prior_', 'predicate_prefix_static_':'prefix_static_'}[prefix]
                    mask = table.predicate_count.eq(1).to_numpy()
                    np.testing.assert_allclose(value[mask,d,k], table.loc[mask,old_prefix+name], atol=1e-12, rtol=0, equal_nan=True)
                    single_checks += int(mask.sum())
        windows = dict(whole=np.ones(n,bool), training=ordinal<cut, test=ordinal>=cut,
                       first64=ordinal<64, last64=ordinal>=max(0,n-64))
        estimates[subject,arm] = {}
        for window, in_window in windows.items():
            mask = complete & in_window
            conditioned = pooled_mean_and_mcse(numerator, probability, mask)
            unweighted = pooled_mean_and_mcse(prior, np.ones_like(probability), mask)
            gain_point = conditioned[0]-unweighted[0]; gain_deleted = conditioned[2]-unweighted[2]
            metrics = dict(conditioned=conditioned, unweighted=unweighted,
                           current_choice_gain=(gain_point,jackknife_error(gain_deleted),gain_deleted))
            for metric, (point, error, deleted) in metrics.items():
                for d, domain in enumerate(DOMAINS):
                    row = dict(subject=subject,arm=arm,condition=first['condition'],window=window,domain=domain,metric=metric,
                        reports=int(mask.sum()),window_trials=int(in_window.sum()),filter_seeds=len(seeds),
                        single_predicate_reports=int((mask & table.predicate_count.eq(1).to_numpy()).sum()))
                    for k, label in enumerate(LABELS):
                        row[label] = point[d,k]; row[label+'_mcse'] = error[d,k]
                    static_mask = mask & static_valid
                    row['static_reports'] = int(static_mask.sum())
                    row['prefix_static_support'] = float(static_score[static_mask,d,0].mean()) if static_mask.any() else np.nan
                    row['dynamic_support_on_static_reports'] = float(scores[static_mask,d,0].mean()) if static_mask.any() else np.nan
                    summaries.append(row)
                    estimates[subject,arm][window,domain,metric] = dict(point=point[d],values=deleted[:,d],seeds=seeds,reports=int(mask.sum()))
        tables.append(table)
        print(f'Predicate alignment {subject} {arm}: {int(complete.sum())} complete reports', flush=True)
    contrasts = []
    reference = manifest['reference_model']
    for (subject,arm), own in estimates.items():
        if arm == reference:
            continue
        base = estimates[subject,reference]
        for key, estimate in own.items():
            old = base[key]
            np.testing.assert_array_equal(estimate['seeds'], old['seeds'])
            if estimate['reports'] != old['reports']:
                raise ValueError('Model comparison changed the report denominator')
            difference = estimate['point']-old['point']
            error = jackknife_error(estimate['values']-old['values'])
            contrasts.append(dict(subject=subject,arm=arm,reference=reference,window=key[0],domain=key[1],metric=key[2],reports=estimate['reports'],
                **{label+'_difference':difference[k] for k,label in enumerate(LABELS)},
                **{label+'_difference_mcse':error[k] for k,label in enumerate(LABELS)}))
    for path, sha in inputs.items():
        if alignment.digest(Path(path)) != sha:
            raise ValueError('Input changed during predicate evaluation')
    output.mkdir(parents=True)
    for name, table in [('trials',pd.concat(tables,ignore_index=True)),('summary',pd.DataFrame(summaries)),
                        ('contrasts',pd.DataFrame(contrasts)),('component_fractions',pd.DataFrame(component_rows))]:
        table.to_csv(output/(name+'.csv'),index=False)
    display = pd.DataFrame(summaries)
    display = display[display.window.eq('test') & display.metric.eq('conditioned')]
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐项口述内容覆盖</title>'
        '<style>body{max-width:1400px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>逐项口述内容覆盖</h1><p>每句话中的判断等权平均，再对完整报告等权平均。'
        '所有判断使用同一行为条件状态分布。原整句支持指标保存在逐题表中；逐项指标描述部分内容覆盖，'
        '报告生成概率需要单独的表达模型。全部完整报告进入主评价，缺失和编码矛盾保留原状态。</p>'
        '<p>状态取作答后、本题反馈前；参数仅按行为估计。两种读法沿用各自冻结编码。'
        'prior去掉本题选择加权；prefix_static使用前段按类别平均并冻结的依据分布。'
        'mcse为删除整次滤波重复的模拟标准误，仅描述有限模拟误差。开发样本和估参范围沿用来源报告。</p>'
        +display.to_html(index=False,float_format=lambda x:f'{x:.4f}')
        +'<p><a href="trials.csv">逐题表与原整句指标</a> · <a href="summary.csv">完整分段</a> · '
        '<a href="contrasts.csv">同口径方案差值</a> · <a href="component_fractions.csv">逐分量判断比例</a></p></html>')
    (output/'index.html').write_text(page)
    paths = [Path(__file__),ROOT/'src/Bayesian_state/evaluation/oral/predicate_alignment.py',
        ROOT/'src/Bayesian_state/evaluation/oral/constraint_alignment.py',Path(alignment.__file__),
        ROOT/'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py']
    (output/'source').mkdir()
    for path in paths:
        shutil.copy2(path,output/'source'/path.name)
    result = dict(status='complete',alignment=str(anchor.resolve()),subjects=manifest['subjects'],arms=manifest['arms'],
        primary_denominator='unchanged_complete_reports',metric='equal_predicates_within_report_then_equal_reports',
        single_predicate_identity_checks=single_checks,frozen_component_coverage_checks=coverage_checks,inputs_sha256=inputs,
        parameter_estimation_scope=manifest.get('parameter_estimation_scope'),evaluation_use=manifest.get('evaluation_use'),
        source_sha256={str(p.relative_to(ROOT)):alignment.digest(p) for p in paths},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        data_use='Frozen behavioral states and frozen oral content; no fitting or per-model reinterpretation.',
        artifacts={p.name:alignment.digest(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.alignment, args.output)
    print({k:v for k,v in result.items() if k not in ('inputs_sha256','source_sha256','artifacts')})
