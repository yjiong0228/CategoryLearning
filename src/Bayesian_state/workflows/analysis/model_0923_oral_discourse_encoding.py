"""Freeze text-only discourse content for a completed development oral audit.

Run --oral-audit DIRECTORY --output NEW_DIRECTORY [--workers 8]. Only report
identity, raw text, feature order and existing text coding are loaded. The new
layer retains original codes, spans, modifiers and incomplete clauses. It adds
a report-wide relative sensitivity reading for eligible bare long/short groups.
No stimulus, choice, behavior outcome or fitted state enters the transformation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from ...evaluation.oral import discourse_encoding as discourse
from ...evaluation.oral import structure_0923 as structure
from . import model_0923_oral_contrast_reference as reference
from . import model_0923_oral_reference_sensitivity as sensitivity

ROOT = Path(__file__).resolve().parents[4]
KEYS = ['iSub', 'iSession', 'iTrial']
FEATURES = [f'feature{k}_name' for k in range(1, 5)]
TEXT_COLUMNS = [*KEYS, 'ordinal_trial', 'text', *FEATURES,
                'status', 'scope', 'tokens', 'flags', 'unparsed']


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode_text(job: tuple[str, tuple[str, ...]]) -> dict:
    """One model-independent content coding, shared by duplicate report text."""
    text, features = job
    report = discourse.encode_discourse(text, features)
    reason, long, short = reference.contrast_dimensions(
        text, json.dumps(report.content.tokens),
        'complete' if report.content_complete else 'parsed_fragment')
    relative = sensitivity.relative_tokens(long, short) if reason == 'eligible' else ()
    return dict(**asdict(report), relative_reason=reason, relative_tokens=relative,
                relative_long_dimensions=long, relative_short_dimensions=short,
                relative_scope='whole_content_conjunction',
                reference_selection='none; preserve both eligible readings')


def run(audit: Path, output: Path, requested_workers: int) -> None:
    if output.exists():
        raise FileExistsError('Use a new output directory')
    if requested_workers < 1:
        raise ValueError('workers must be positive')
    manifest_path, trials_path = audit/'manifest.json', audit/'trials.csv'
    source = json.loads(manifest_path.read_text())
    if (source['status'] != 'complete' or source['config']['sample_role'] != 'development'
            or source.get('semantic_sensitivity') is not None):
        raise ValueError('Require a completed original development audit')
    if digest(trials_path) != source['artifacts']['trials.csv']:
        raise ValueError('Original text audit changed')
    table = pd.read_csv(trials_path, usecols=TEXT_COLUMNS, keep_default_na=False)
    if (table.duplicated(KEYS).any()
            or set(table.iSub.unique()) != set(source['config']['subjects'])):
        raise ValueError('Duplicate identities or different development participants')
    jobs = [(r['text'], tuple(r[name] for name in FEATURES))
            for r in table.to_dict('records')]
    unique = list(dict.fromkeys(jobs))
    workers = min(requested_workers, 128, len(os.sched_getaffinity(0)), len(unique))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    files = [Path(__file__), Path(discourse.__file__), Path(structure.__file__),
             Path(reference.__file__), Path(sensitivity.__file__), ROOT/'src/oral_coding.py',
             ROOT/'src/Bayesian_state/model/modules/readout/report_space.py']
    manifest = dict(status='running', encoder=discourse.VERSION,
        source_audit=str(audit.resolve()), sample_role='development',
        subjects=source['config']['subjects'], rows=len(table), distinct_texts=len(unique),
        input_columns=TEXT_COLUMNS, workers=workers,
        input_sha256={str(p.resolve()): digest(p) for p in (manifest_path, trials_path)},
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in files},
        started_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, scipy=scipy.__version__),
        model_states_read=False, responses_read=False, stimulus_values_read=False,
        role_assignment='unassigned', original_coding_changed=False,
        half_reference='range_midpoint', bare_long_short_reference='ambiguous',
        relative_reading='whole-content sensitivity; no automatic segment attribution',
        validation_scope='text coverage and provenance; no model agreement evaluated')
    output.mkdir(parents=True)

    def save_manifest() -> None:
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')

    save_manifest()
    print(f'{len(table)} rows; {len(unique)} distinct texts; {workers} workers', flush=True)
    if workers == 1:
        coded = list(map(encode_text, unique))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            coded = list(pool.map(encode_text, unique, chunksize=8))
    lookup = dict(zip(unique, coded))
    records = []
    with (output/'reports.jsonl').open('w') as stream:
        for row, job in zip(table.to_dict('records'), jobs):
            report = lookup[job]
            # Existing report coding is independently replayed, without any
            # change selected by downstream model or stimulus compatibility.
            for name in ('tokens', 'flags', 'unparsed'):
                if list(report['original'][name]) != json.loads(row[name]):
                    raise ValueError(f'Original {name} cannot be reproduced for {tuple(row[k] for k in KEYS)}')
            if row['status'] != report['original']['status']:
                raise ValueError('Original report status changed')
            identity = {name: row[name] for name in [*KEYS, 'ordinal_trial', *FEATURES]}
            stream.write(json.dumps(dict(**identity, original_scope=row['scope'], **report),
                                    ensure_ascii=False)+'\n')
            records.append(dict(**identity, text=row['text'], original_status=row['status'],
                original_scope=row['scope'], original_tokens=row['tokens'],
                content_status=report['content']['status'], content_complete=report['content_complete'],
                content_tokens=json.dumps(report['content']['tokens']),
                content_unparsed=json.dumps(report['content']['unparsed'], ensure_ascii=False),
                content_flags=json.dumps(report['content']['flags']),
                contrast_markers=len(report['contrast_markers']),
                concessions=sum(s['concession'] is not None for s in report['segments']),
                segments=len(report['segments']),
                coded_segments=sum(s['encoding']['status'] == 'coded' for s in report['segments']),
                relative_reason=report['relative_reason'],
                relative_tokens=json.dumps(report['relative_tokens']), role_status=report['role_status']))
    result = pd.DataFrame(records)
    result.to_csv(output/'trials.csv', index=False)
    summaries = []
    for subject, group in result.groupby('iSub', sort=True):
        summaries.append(dict(subject=int(subject), rows=len(group),
            nonempty=int(group.original_status.ne('missing').sum()),
            original_coded=int(group.original_status.eq('coded').sum()),
            content_complete=int(group.content_complete.sum()),
            new_complete=int((group.content_complete & group.original_status.ne('coded')).sum()),
            contrast_reports=int(group.contrast_markers.gt(0).sum()),
            contrast_complete=int((group.contrast_markers.gt(0) & group.content_complete).sum()),
            relative_eligible=int(group.relative_reason.eq('eligible').sum())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'summary.csv', index=False)
    for file, sha in manifest['input_sha256'].items():
        if digest(Path(file)) != sha: raise RuntimeError('Input changed during coding')
    for file, sha in manifest['source_sha256'].items():
        if digest(ROOT/file) != sha: raise RuntimeError('Source changed during coding')
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                    artifacts={name: digest(output/name) for name in ('reports.jsonl', 'trials.csv', 'summary.csv')})
    save_manifest()
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oral-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    run(args.oral_audit, args.output, args.workers)


if __name__ == '__main__':
    main()
