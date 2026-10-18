"""Run constraint-method examples and a read-only development-corpus audit.

Usage: python -m src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit
  --config configs/shared/model0923_oral_constraint_audit.yaml --output NEW_DIRECTORY

Catalogue counts describe representability, not fitted-model alignment. This
workflow never loads fitted weights, changes the parser, or fits oral data.
Optional YAML ``symbolic_half_reference: true`` preserves bare half references
with an unknown referent; its default is false and retains the original coding.
YAML ``half_reference: range_midpoint`` uses the independently confirmed length
range midpoint (normalized 0.5). These two half-reference modes are exclusive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import html
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
import yaml

from ...evaluation.oral.constraint_alignment import (
    VERSION, InconsistentReportError, evaluate_component, validate_report_constraints,
)
from ...evaluation.oral.structure_0923 import (
    encode_report, report_encoding_version,
)
from ...hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from ...hypothesis_space.spaces.structural_0923 import build_relational_0923_space

ROOT = Path(__file__).resolve().parents[4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def numerical_constraints(tokens: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not tokens or any(not token.startswith('facet:') for token in tokens):
        raise ValueError('Example rule must have complete numerical semantics')
    rows = np.asarray([row for token in tokens for row in json.loads(token[6:])])
    return rows[:, :-1], rows[:, -1]


def run_examples(config: dict) -> pd.DataFrame:
    rows = []
    for example in config['examples']:
        names = tuple(config['feature_order'])
        report = encode_report(example['report'], names,
                               symbolic_half_reference=config.get('symbolic_half_reference', False),
                               half_reference=config.get('half_reference'))
        rule = encode_report(example['rule'], names)
        if rule.status != 'coded' or report.status not in {'coded', 'missing'}:
            raise ValueError(f"Unresolved method example: {example['name']}")
        result = evaluate_component(report.tokens, *numerical_constraints(rule.tokens),
                                    tolerance=config['tolerance'])
        expected = example['expected']
        passed = [result.structural, result.functional] == expected and result.available == example.get('available', True)
        rows.append(dict(example, **asdict(result), passed=passed))
    frame = pd.DataFrame(rows)
    if not frame.passed.all():
        raise AssertionError(frame.loc[~frame.passed].to_dict('records'))
    return frame


def load_reports(config: dict, root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, str]]:
    """Join raw text and metadata one-to-one; never read correctness for coding."""
    source = root / config['processed']
    columns = ['iSub', 'condition', 'iSession', 'iTrial', 'choice', 'text',
               *[f'feature{i}_name' for i in range(1, 5)]]
    frame = pd.read_csv(source, usecols=columns)
    hashes, frames = {config['processed']: digest(source)}, []
    for subject in config['subjects']:
        sub = frame.loc[frame.iSub.eq(subject)].copy()
        if sub.empty or sub.condition.nunique() != 1:
            raise ValueError(f'Missing subject or inconsistent condition: {subject}')
        raw_path = root / config['raw_directory'] / f'Task2_{subject}_rec.csv'
        raw = pd.read_csv(raw_path)
        keys = ['iSession', 'iTrial']
        if sub.duplicated(keys).any() or raw.duplicated(keys).any():
            raise ValueError('Duplicate report identity')
        joined = sub.merge(raw[keys + ['text']], on=keys, how='outer',
                           suffixes=('_processed', ''), indicator=True, validate='one_to_one')
        if (not joined['_merge'].eq('both').all()
                or not joined.text.fillna('').eq(joined.text_processed.fillna('')).all()):
            raise ValueError(f'Raw/processed report mismatch: {subject}')
        condition = int(sub.condition.iloc[0])
        if condition not in {1, 2, 3}:
            raise ValueError('Unknown task condition')
        if not joined.choice.isin(range(1, (2 if condition == 1 else 4) + 1)).all():
            raise ValueError('Choice is outside the task category labels')
        joined = joined.drop(columns=['_merge', 'text_processed']).sort_values(keys)
        joined['ordinal_trial'] = np.arange(1, len(joined) + 1)
        frames.append(joined)
        hashes[str(raw_path.relative_to(root))] = digest(raw_path)
    return pd.concat(frames, ignore_index=True), hashes


def encode_observations(frame: pd.DataFrame, *, tolerance: float = 1e-9,
                        symbolic_half_reference: bool = False,
                        half_reference: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    report_encoding_version(symbolic_half_reference=symbolic_half_reference, half_reference=half_reference)
    cache, queries, rows = {}, {}, []
    for row in frame.itertuples(index=False):
        names = tuple(getattr(row, f'feature{i}_name') for i in range(1, 5))
        text = '' if pd.isna(row.text) else str(row.text)
        key = (text, names)
        if key not in cache:
            cache[key] = encode_report(text, names, symbolic_half_reference=symbolic_half_reference,
                                       half_reference=half_reference)
        report = cache[key]
        scope = ('complete' if report.status == 'coded' else 'parsed_fragment'
                 if report.tokens else report.status)
        query_id, constraint_issue = '', ''
        if report.tokens:
            try:
                validate_report_constraints(report.tokens, tolerance=tolerance)
            except InconsistentReportError as error:
                scope, constraint_issue = 'inconsistent_constraints', str(error)
        if report.tokens and not constraint_issue:
            payload = dict(condition=int(row.condition), choice=int(row.choice), tokens=list(report.tokens))
            query_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
            queries[query_id] = dict(query_id=query_id, **payload)
        rows.append(dict(row._asdict(), text=text, status=report.status, scope=scope,
                         constraint_issue=constraint_issue,
                         query_id=query_id, tokens=json.dumps(report.tokens, ensure_ascii=False),
                         flags=json.dumps(report.flags, ensure_ascii=False),
                         unparsed=json.dumps(report.unparsed, ensure_ascii=False),
                         n_predicates=len(report.tokens)))
    return pd.DataFrame(rows), list(queries.values())


@lru_cache(maxsize=2)
def catalogue(n_categories: int):
    return (build_continuous_hypothesis_space(4, 2) if n_categories == 2
            else build_relational_0923_space())


def audit_query(job: dict) -> tuple[dict, list[dict]]:
    space = catalogue(2 if job['condition'] == 1 else 4)
    matches, summary = [], dict(query_id=job['query_id'], condition=job['condition'],
                               choice=job['choice'], catalogue_version=space.version,
                               rules=len(space), tokens=json.dumps(job['tokens']))
    for domain in ('structural', 'functional'):
        for label in ('support', 'conflict', 'undetermined'):
            for bound in ('any_component', 'all_components'):
                summary[f'{domain}_{label}_{bound}'] = 0
    for h in space:
        results = []
        for i, component in enumerate(h.categories[job['choice'] - 1].components):
            match = evaluate_component(job['tokens'], component.A, component.b,
                                       tolerance=job['tolerance'])
            results.append(match)
            matches.append(dict(query_id=job['query_id'], rule=h.index, component=i,
                                **asdict(match)))
        for domain in ('structural', 'functional'):
            labels = [getattr(result, domain) for result in results]
            for label in ('support', 'conflict', 'undetermined'):
                summary[f'{domain}_{label}_any_component'] += int(label in labels)
                summary[f'{domain}_{label}_all_components'] += int(all(x == label for x in labels))
    return summary, matches


def summarize_windows(trials: pd.DataFrame, width: int) -> pd.DataFrame:
    rows = []
    for subject, group in trials.groupby('iSub', sort=True):
        group = group.sort_values('ordinal_trial')
        for label, window in [('whole', group), ('first', group.iloc[:width]), ('last', group.iloc[-width:])]:
            complete = window.loc[window.scope.eq('complete')]
            row = dict(subject=int(subject), window=label, trials=len(window),
                       coded=int(window.status.eq('coded').sum()),
                       needs_review=int(window.status.eq('needs_review').sum()),
                       missing=int(window.status.eq('missing').sum()),
                       inconsistent=int(window.scope.eq('inconsistent_constraints').sum()),
                       fragments=int(window.scope.eq('parsed_fragment').sum()))
            for domain in ('structural', 'functional'):
                field = f'{domain}_support_any_component'
                row[f'complete_with_{domain}_catalogue_support'] = (
                    int(complete[field].gt(0).sum()) if field in complete else 0)
            rows.append(row)
    return pd.DataFrame(rows)


def run(config_path: Path, output: Path, workers_override: int | None = None) -> None:
    if output.exists():
        raise FileExistsError('Use a new output directory')
    config = yaml.safe_load(config_path.read_text())
    if (config['sample_role'] != 'development' or len(set(config['subjects'])) != len(config['subjects'])
            or not config['subjects'] or int(config['window']) < 1):
        raise ValueError('Expected a unique development cohort and a positive window')
    requested = config['workers'] if workers_override is None else workers_override
    if requested < 1:
        raise ValueError('workers must be positive')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    examples = run_examples(config)
    frame, hashes = load_reports(config)
    symbolic_half = config.get('symbolic_half_reference', False)
    half_reference = config.get('half_reference')
    observations, queries = encode_observations(frame, tolerance=config['tolerance'],
                                               symbolic_half_reference=symbolic_half,
                                               half_reference=half_reference)
    jobs = [dict(query, tolerance=config['tolerance']) for query in queries]
    workers = min(requested, len(os.sched_getaffinity(0)), len(jobs)) if jobs else 0
    output.mkdir(parents=True)
    source_paths = [Path(__file__), ROOT / 'src/Bayesian_state/evaluation/oral/constraint_alignment.py',
                    ROOT / 'src/Bayesian_state/evaluation/oral/structure_0923.py', ROOT / 'src/oral_coding.py',
                    ROOT / 'src/Bayesian_state/model/modules/readout/report_space.py',
                    ROOT / 'src/Bayesian_state/hypothesis_space/spaces/continuous.py',
                    ROOT / 'src/Bayesian_state/hypothesis_space/spaces/structural_0923.py',
                    ROOT / 'src/Bayesian_state/hypothesis_space/spaces/regions.py']
    manifest = dict(status='running', protocol=VERSION,
                    encoder=report_encoding_version(symbolic_half_reference=symbolic_half,
                                                    half_reference=half_reference), config=config,
                    config_sha256=digest(config_path), input_sha256=hashes,
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    started_utc=datetime.now(timezone.utc).isoformat(), workers=workers, unique_queries=len(jobs),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, pandas=pd.__version__, pyyaml=yaml.__version__),
                    result_kind='catalogue_representability_not_fitted_alignment',
                    role_assumption='all_encoded_content_checked; cognitive_role_unassigned',
                    component_handling='report_each_component_and_all_any_counts; no_component_selection',
                    timing_for_future_state_alignment='after_choice_before_current_feedback')
    write_json(output / 'manifest.json', manifest)
    examples.to_csv(output / 'method_examples.csv', index=False)
    print(f'{len(observations)} trials; {len(jobs)} observed constraint queries; {workers} workers', flush=True)
    summaries, components = [], []
    if workers == 1:
        results = map(audit_query, jobs)
        for summary, rows in results:
            summaries.append(summary); components.extend(rows)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for i, (summary, rows) in enumerate(pool.map(audit_query, jobs, chunksize=1), 1):
                summaries.append(summary); components.extend(rows)
                if i % 100 == 0:
                    print(f'Completed {i}/{len(jobs)} queries', flush=True)
    summary_frame = pd.DataFrame(summaries)
    trials = observations.merge(summary_frame.drop(columns=['condition', 'choice', 'tokens']),
                                on='query_id', how='left', validate='many_to_one') if jobs else observations
    windows = summarize_windows(trials, config['window'])
    summary_frame.to_csv(output / 'constraint_queries.csv', index=False)
    pd.DataFrame(components).to_csv(output / 'component_matches.csv', index=False)
    trials.to_csv(output / 'trials.csv', index=False)
    windows.to_csv(output / 'windows.csv', index=False)
    # Keep review examples as descriptions, not automatic coding corrections.
    trials.loc[~trials.scope.eq('complete')].drop_duplicates(['iSub', 'text']).to_csv(
        output / 'review_examples.csv', index=False)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述约束方法检验</title>'
            '<style>body{font:16px sans-serif;max-width:1300px;margin:40px auto}td,th{padding:7px;'
            'border:1px solid #ccc}table{border-collapse:collapse}</style><h1>口述约束方法检验</h1>'
            '<p>本轮评价方法实例及真实口述在固定目录中的表达范围。目录计数不包含拟合状态概率，'
            '不用于判断BASE与NEW的优劣。所有编码内容暂按待检验的判断处理；附加描述的角色仍需确定。</p>'
            '<h2>方法实例</h2>' + examples[['name', 'report', 'rule', 'structural', 'functional', 'passed']].to_html(index=False)
            + '<h2>完整报告与目录支持</h2>' + windows.to_html(index=False)
            + '<p>any_component表示至少一个分量支持，all_components表示每个分量支持；两者保留分量选择的不确定性。'
            '表中目录支持按至少一个规则的至少一个分量计算。完整报告、解析片段、缺失分别保留。</p><ul>'
            + ''.join(f'<li><a href="{html.escape(p.name)}">{html.escape(p.name)}</a></li>'
                      for p in sorted(output.glob('*.csv'))) + '</ul></html>')
    (output / 'index.html').write_text(page)
    if any(digest(ROOT / path) != value for path, value in hashes.items()):
        raise RuntimeError('Input changed during audit')
    if any(digest(ROOT / path) != value for path, value in manifest['source_sha256'].items()):
        raise RuntimeError('Analysis source changed during audit')
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                    trials=len(trials), status_counts=trials.status.value_counts().to_dict(),
                    scope_counts=trials.scope.value_counts().to_dict(), method_examples_passed=len(examples),
                    artifacts={p.name: digest(p) for p in output.iterdir() if p.name != 'manifest.json'})
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(dict(status='complete', trials=len(trials), scopes=manifest['scope_counts']), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, help='Override the bounded process budget; use 1 for smoke checks')
    args = parser.parse_args()
    run(args.config, args.output, args.workers)


if __name__ == '__main__':
    main()
