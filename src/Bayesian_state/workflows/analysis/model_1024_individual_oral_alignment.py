"""Provisional oral compatibility against behavior-only individual states.

All encoded complete reports remain in the main denominator, including reports
with no supporting catalogue component. Fragments and unresolved reports retain
separate rows. Two existing text readings are scored without refitting states.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_1024_individual_states import read, write, digest
from .model_0923_decision_basis_alignment import make_labels, align_identities
from .model_0923_oral_constraint_audit import catalogue
from ...evaluation.oral.decision_basis_scoring import label_masses, DOMAINS, LABELS
from ...evaluation.oral.structure_0923 import FEATURE_NAME_TO_PART

FEATURES = ('脖子', '头', '腿', '尾巴')
SCOPE_LABELS = {
    'complete': '完整编码（含义待人工确认）',
    'parsed_fragment': '已解析片段',
    'needs_review': '待复核',
    'inconsistent_constraints': '编码约束矛盾',
    'missing': '缺失',
}


def constraint_text(A: np.ndarray, b: np.ndarray, features: tuple[str, ...] = FEATURES) -> str:
    """Render exact normalized inequalities for human review."""
    pieces = []
    for a, bound in zip(np.asarray(A), np.asarray(b)):
        use = np.flatnonzero(np.abs(a) > 1e-12)
        if len(use) == 1:
            d = use[0]; relation = '≤' if a[d] > 0 else '≥'
            pieces.append(f'{features[d]}{relation}{bound/a[d]:g}')
        elif len(use) == 2 and np.isclose(a[use].sum(), 0):
            positive = use[np.argmax(a[use])]; negative = use[np.argmin(a[use])]
            if abs(bound) < 1e-12:
                pieces.append(f'{features[positive]}≤{features[negative]}')
            else:
                pieces.append(f'{features[positive]}−{features[negative]}≤{bound/a[positive]:g}')
        else:
            terms = ''.join(('+' if a[d] > 0 and i else '')+f'{a[d]:g}×{features[d]}'
                            for i, d in enumerate(use))
            pieces.append(f'{terms}≤{bound:g}')
    return ' 且 '.join(pieces) or '整个刺激范围'


def tokens_text(value: str, features: tuple[str, ...] = FEATURES) -> str:
    pieces = []
    for token in json.loads(value):
        if token.startswith('facet:'):
            rows = np.asarray(json.loads(token[6:]))
            pieces.append(constraint_text(rows[:, :-1], rows[:, -1], features))
        elif token.startswith(('body:', 'half_unknown:')):
            kind, dim, direction = token.split(':')
            pieces.append(features[int(dim)]+('短于' if direction=='lt' else '长于')+
                          ('躯干（参照未定）' if kind=='body' else '一半（参照未定）'))
        else:
            pieces.append(token)
    return '；'.join(pieces)


def uniform_basis(choices: np.ndarray, counts: np.ndarray, components: int) -> np.ndarray:
    result = np.zeros((len(choices), len(counts), components))
    for t, choice in enumerate(choices):
        for h in range(len(counts)):
            count = int(counts[h, choice-1])
            result[t, h, :count] = 1./len(counts)/count
    return result


def static_basis(basis: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Complete-record category means; a descriptive time-invariant comparison."""
    result = np.zeros_like(basis)
    for choice in np.unique(choices):
        use = choices == choice
        result[use] = basis[use].mean(axis=0)
    return result


def subject_features(frame: pd.DataFrame) -> tuple[str, ...]:
    """Bind geometry dimensions to this participant's counterbalanced parts."""
    columns = [f'feature{i}_name' for i in range(1, 5)]
    if not frame[columns].nunique().eq(1).all():
        raise ValueError('Require a constant, explicit feature mapping within a participant')
    features = tuple(FEATURE_NAME_TO_PART[frame.iloc[0][c]] for c in columns)
    if len(set(features)) != 4:
        raise ValueError('Repeated feature mapping')
    return features


def top_components(basis: np.ndarray, choices: np.ndarray, space, labels: np.ndarray,
                   features: tuple[str, ...]) -> list[str]:
    descriptions = {}
    for h, rule in enumerate(space):
        for choice, category in enumerate(rule.categories):
            for c, component in enumerate(category.components):
                descriptions[h, choice+1, c] = constraint_text(component.A, component.b, features)
    result = []
    for t, mass in enumerate(basis):
        order = np.argsort(mass.ravel())[-3:][::-1]; text = []
        for index in order:
            h, c = np.unravel_index(index, mass.shape)
            if mass[h, c] <= 0:
                continue
            verdict = labels[t, h, c, 1]
            label = ('相容', '冲突', '无法判断')[verdict] if verdict >= 0 else '未评分'
            text.append(f'R{h}/分量{c}：{descriptions[h, int(choices[t]), c]} '
                        f'[{mass[h,c]:.3f}；{label}]')
        result.append(' | '.join(text))
    return result


def write_review_html(frame: pd.DataFrame, target: Path) -> None:
    columns = [('subject', '被试'), ('trial', '有效试次'), ('iSession', '场次'), ('iTrial', '原试次'),
        ('text', '口述原文'), ('scope', '编码状态'), ('decoded_constraints', '当前解释'),
        ('full_support_min', '相容下限'), ('full_support_max', '相容上限'),
        ('full_conflict_min', '冲突下限'), ('full_conflict_max', '冲突上限'),
        ('relative_support_change_min', '相对解释变化下限'),
        ('relative_support_change_max', '相对解释变化上限'),
        ('representative_components', '列示参数下最可能的内容'), ('flags', '编码提示'),
        ('unparsed', '未解析部分'), ('parameter_status', '参数状态'),
        ('state_numerical_status', '状态计算检查')]
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key, '')
            if pd.isna(value):
                value = ''
            elif key == 'scope':
                value = SCOPE_LABELS.get(value, value)
            elif key == 'parameter_status':
                value = {'diagnostic_unresolved': '诊断候选；参数问题待解决',
                         'finite_retained_candidates': '上轮保留的有限候选'}.get(value, value)
            elif isinstance(value, float):
                value = f'{value:.3f}'
            cells.append('<td>'+html.escape(str(value))+'</td>')
        body.append('<tr data-subject="'+str(row.subject)+'">'+''.join(cells)+'</tr>')
    heading = ''.join('<th>'+name+'</th>' for _, name in columns)
    content = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述逐条复核</title>
<style>body{font:14px sans-serif;margin:20px;color:#202020}table{border-collapse:collapse;width:100%}
th,td{padding:7px;border:1px solid #ddd;vertical-align:top;min-width:65px}th{position:sticky;top:0;background:#eaf1f6}
td:nth-child(5),td:nth-child(7),td:nth-child(14){min-width:270px}tr:nth-child(even){background:#fafafa}
input,select{font:inherit;padding:6px}header{padding-bottom:15px}</style><header>
<h1>口述逐条复核：暂定外部内容对照</h1>
<p>全部'''+str(len(frame))+'''条记录保留。相容与冲突为模型状态的概率质量；上下限来自同一人的候选参数。
未评分的行留空。列示内容取预先固定的第一组候选；S129为诊断候选。
状态计算检查列显示各候选的数值限制；受限候选的精确概率需要谨慎解释。
原文和编码均保留，人工修改请另记于CSV的复核列。</p>
<label>被试 <select id="subject"><option value="">全部</option>'''
    content += ''.join(f'<option>{s}</option>' for s in sorted(frame.subject.unique()))
    content += '''</select></label> <label>搜索 <input id="search" placeholder="原文、编码、规则内容"></label>
<span id="count"></span></header><table><thead><tr>'''+heading+'</tr></thead><tbody>'+''.join(body)+'''</tbody></table>
<script>const rows=[...document.querySelectorAll('tbody tr')];
function filter(){let s=document.getElementById('subject').value,q=document.getElementById('search').value.toLowerCase(),n=0;
for(const r of rows){let ok=(!s||r.dataset.subject===s)&&r.textContent.toLowerCase().includes(q);r.hidden=!ok;n+=ok;}
document.getElementById('count').textContent='显示 '+n+' 条';}
document.getElementById('subject').onchange=filter;document.getElementById('search').oninput=filter;filter();</script></html>'''
    target.write_text(content)


def run(source: Path, summary: Path, midpoint: Path, relative: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new alignment output directory')
    output.mkdir(parents=True)
    protocol = read(source/'protocol.json')
    cases = {c['id']: c for c in protocol['cases'] if c['kind']=='human'}
    selected = [s for s in read(summary/'selected.json') if 'file' in s]
    audits = {name: (pd.read_csv(path/'trials.csv'), pd.read_csv(path/'component_matches.csv'))
              for name, path in [('midpoint', midpoint), ('relative', relative)]}
    scores, metrics, representatives, trial_frames = [], [], {}, {}
    catalogue_rows = []
    for s in selected:
        case = cases[s['case']]; sid = case['subject']
        with np.load(s['file']) as z:
            arrays = {k: z[k] for k in ('filtered_component', 'full_component', 'component_counts')}
        counts = arrays['component_counts']; n, h, d = arrays['full_component'].shape
        space = catalogue(2 if case['condition']==1 else 4)
        for reading, (trials, matches) in audits.items():
            aligned, _ = align_identities(trials, case['template'], n)
            features = subject_features(aligned)
            choices = aligned.choice.to_numpy(int)
            labels = make_labels(aligned, matches, h, d)
            uniform = label_masses(uniform_basis(choices, counts, d), labels)
            if reading == 'midpoint' and sid not in trial_frames:
                trial_frames[sid] = aligned.copy()
                for rule_id, rule in enumerate(space):
                    for choice, region in enumerate(rule.categories):
                        for c, component in enumerate(region.components):
                            catalogue_rows.append(dict(subject=sid, rule=rule_id, choice=choice+1, component=c,
                                family=rule.family, description=constraint_text(component.A, component.b, features),
                                feature_mapping=';'.join(features)))
            for mode in ('filtered', 'full'):
                basis = arrays[mode+'_component']
                dynamic = label_masses(basis, labels)
                static = label_masses(static_basis(basis, choices), labels)
                for domain, domain_name in enumerate(DOMAINS):
                    frame = pd.DataFrame(dict(subject=sid, candidate=case['candidate']['id'], reading=reading,
                        mode=mode, domain=domain_name, trial=np.arange(1, n+1), scope=aligned.scope,
                        support=dynamic[:, domain, 0], conflict=dynamic[:, domain, 1],
                        undetermined=dynamic[:, domain, 2], uniform_support=uniform[:, domain, 0],
                        static_support=static[:, domain, 0]))
                    scores.append(frame)
                    for scope in ('complete', 'parsed_fragment'):
                        use = aligned.scope.eq(scope).to_numpy()
                        row = dict(subject=sid, candidate=case['candidate']['id'], reading=reading, mode=mode,
                            domain=domain_name, scope=scope, n=int(use.sum()), total_trials=n,
                            numerical_status=s['status'], parameter_status=case['parameter_status'])
                        for name, mass in [('dynamic', dynamic), ('uniform', uniform), ('static', static)]:
                            for k, label in enumerate(LABELS):
                                row[name+'_'+label] = float(np.mean(mass[use, domain, k])) if use.any() else np.nan
                        row['dynamic_minus_uniform_support'] = row['dynamic_support']-row['uniform_support']
                        row['dynamic_minus_static_support'] = row['dynamic_support']-row['static_support']
                        row['no_catalogue_support_count'] = int(np.sum(use & ~np.any(labels[..., domain]==0, axis=(1, 2))))
                        metrics.append(row)
                if reading=='midpoint' and mode=='full' and sid not in representatives:
                    representatives[sid] = dict(candidate=case['candidate']['id'],
                        components=top_components(basis, choices, space, labels, features))
        print(f'Oral alignment {s["case"]}', flush=True)
    all_scores = pd.concat(scores, ignore_index=True)
    all_scores.to_csv(output/'candidate_trial_scores.csv.gz', index=False, compression='gzip')
    pd.DataFrame(metrics).to_csv(output/'compatibility_summary.csv', index=False)
    pd.DataFrame(catalogue_rows).to_csv(output/'rule_content_catalogue.csv', index=False, encoding='utf-8-sig')
    reviews = []
    for sid, frame in trial_frames.items():
        own = all_scores.loc[all_scores.subject.eq(sid) & all_scores.domain.eq('functional')]
        frame = frame.copy(); frame['subject'] = sid; frame['trial'] = np.arange(1, len(frame)+1)
        features = subject_features(frame)
        frame['decoded_constraints'] = frame.tokens.apply(lambda value: tokens_text(value, features))
        frame['representative_candidate'] = representatives[sid]['candidate']
        frame['representative_components'] = representatives[sid]['components']
        frame['parameter_status'] = 'diagnostic_unresolved' if sid==129 else 'finite_retained_candidates'
        frame['state_numerical_status'] = '；'.join(
            s['candidate'][:6]+('：检查通过' if s['status']=='numerical_check_pass' else '：数值受限')
            for s in selected if s['subject']==sid)
        for mode in ('filtered', 'full'):
            selected_scores = own.loc[own.reading.eq('midpoint') & own['mode'].eq(mode)]
            for metric in ('support', 'conflict', 'undetermined'):
                grouped = selected_scores.groupby('trial')[metric].agg(['min', 'max'])
                for boundary in ('min', 'max'):
                    frame[mode+'_'+metric+'_'+boundary] = frame.trial.map(grouped[boundary])
        a = own.loc[own.reading.eq('midpoint') & own['mode'].eq('full'), ['candidate', 'trial', 'support']]
        b = own.loc[own.reading.eq('relative') & own['mode'].eq('full'), ['candidate', 'trial', 'support']]
        changes = a.merge(b, on=['candidate', 'trial'], validate='one_to_one', suffixes=('_a', '_b'))
        changes['change'] = changes.support_b-changes.support_a
        for boundary in ('min', 'max'):
            frame['relative_support_change_'+boundary] = frame.trial.map(changes.groupby('trial').change.agg(boundary))
        for column in ('manual_corrected_text', 'manual_intended_meaning', 'manual_notes'):
            frame[column] = ''
        frame.to_csv(output/f'S{sid}_review.csv', index=False, encoding='utf-8-sig')
        reviews.append(frame)
    review = pd.concat(reviews, ignore_index=True)
    review.to_csv(output/'all_subjects_review.csv', index=False, encoding='utf-8-sig')
    write_review_html(review, output/'review.html')
    write(output/'completion.json', dict(status='complete', reports=len(review), candidates=len(selected),
        scopes=review.scope.value_counts().to_dict(), oral_used_for_inference=False,
        raw_text_changed=False, encoder_changed=False, external_validation_status='provisional_development',
        timing_evidence='exp_program/exp123/Task2_learning.m: recording ends before feedback display',
        source_sha256=digest(Path(__file__)),
        audit_manifests={str(p): digest(p/'manifest.json') for p in (midpoint, relative)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'summary', 'midpoint', 'relative', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.summary, args.midpoint, args.relative, args.output)
