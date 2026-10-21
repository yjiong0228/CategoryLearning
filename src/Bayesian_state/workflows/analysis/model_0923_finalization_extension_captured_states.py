"""Validate original-filter captures and align them to fixed extension oral data.

Use --phase collect --summary SUMMARY --output NEW, then --phase align
--source STATES --oral-audit AUDIT --output NEW. Collection copies checked
state arrays from the original evaluation and runs zero particle filters.
Behavior identities, all registered seeds, chronological boundaries and the
original ONLINE/RETAIN execution audit are checked by the existing replay
workflow. Stored provenance distinguishes capture from independent replay.
Oral score mathematics are delegated unchanged to the shared alignment entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import yaml

from . import model_0923_finalization_extension_capture as capture
from . import model_0923_finalization_extension_states as states

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
ALIGNMENT = 'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_alignment.py'
read, write, digest = capture.read, capture.write, capture.digest


def validate_record(file: Path) -> dict:
    """Bind the original executed job, captured arrays and saved probabilities."""
    record = read(file)
    if (record.get('status') != 'complete' or record.get('exact_behavior_replay') is not False
            or record.get('exact_original_filter_capture') is not True or record.get('additional_filters') != 0
            or record.get('capture_source_sha256') != capture.sources()
            or record.get('timing') != 'after_choice_before_current_feedback'):
        raise ValueError('Require a complete, bound original-filter decision-basis capture')
    jobfile, score = Path(record['original_job']), Path(record['score_file'])
    if digest(jobfile) != record['original_job_sha256'] or digest(score) != record['score_sha256']:
        raise ValueError('Original captured job or behavioral probability reference changed')
    job = read(jobfile)
    capture.check_capture_job(job)
    origin = Path(job['decision_basis_target'])
    if (digest(origin.with_suffix('.npz')) != record['array_sha256']
            or digest(file.with_suffix('.npz')) != record['array_sha256']
            or read(origin.with_suffix('.json')) != record
            or Path(job['target']).with_suffix('.npz') != score
            or any(record[k] != job[k] for k in ('particles', 'seeds', 'candidate'))):
        raise ValueError('State capture differs from its original evaluation job')
    with np.load(file.with_suffix('.npz')) as z, np.load(score) as reference:
        np.testing.assert_array_equal(z['prediction'], reference['probabilities'][0])
        np.testing.assert_array_equal(z['choices'], reference['choices'])
        np.testing.assert_array_equal(z['seeds'], reference['seeds'])
        if len(z['choices']) != record['trials']:
            raise ValueError('Incomplete captured trajectory')
        np.testing.assert_allclose(z['prior_component_mass'].sum((1, 2)), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(z['choice_component_joint_mass'].sum((1, 2)),
                                   z['choice_probability'], atol=1e-12, rtol=0)
    return record


def source_jobs(settings: dict, source: Path, smoke: bool):
    if smoke:
        raise ValueError('Scientific captured-state collection requires the full extension')
    declaration = read(source / 'decision_basis_capture_protocol.json')
    if (declaration.get('source_sha256') != capture.sources() or declaration.get('additional_filters') != 0
            or declaration.get('scientific_settings_changed') is not False):
        raise ValueError('Missing unchanged evaluation capture declaration')
    old_check = states.check_filter

    def check(planned, actual, *args):
        capture.check_capture_job(actual)
        # Capture metadata is observational. Check every original scientific
        # field with the existing strict full-history filter identity check.
        old_check(planned, {k: v for k, v in actual.items() if k not in capture.CAPTURE_KEYS}, *args)

    states.check_filter = check
    try:
        files, metadata = states.source_jobs(settings, source, smoke)
    finally:
        states.check_filter = old_check
    bindings = metadata['refit_replay_inputs']
    paths = [source / 'decision_basis_capture_protocol.json', ROOT / ENTRY, ROOT / ALIGNMENT]
    for file in files:
        target = Path(read(file)['decision_basis_target'])
        validate_record(target.with_suffix('.json'))
        paths.extend(target.with_suffix(s) for s in ('.npz', '.json'))
    paths.extend(ROOT / p for p in capture.sources())
    for path in paths:
        bindings[str(path.resolve())] = digest(path)
    metadata.update(state_recording='original_evaluation_filter_capture', additional_filters=0)
    return files, metadata


def collect(summary: Path, output: Path) -> None:
    manifest = read(summary / 'manifest.json')
    settings = read(Path(manifest['fit']) / 'nominees.json')['settings']
    config = dict(protocol='model0923_finalization_extension_capture_v1', source_kind='registered_extension',
        source=manifest['evaluation'], summary=str(summary.resolve()), subjects=settings['subjects'],
        arms=list(settings['arms']), workers=1, seeds_per_arm=settings['budgets']['evaluate']['seeds'],
        sample_role='development')
    replay = states.cohort.separate.helpers().helpers().helpers()
    replay.source_jobs, replay.ENTRY = source_jobs, ENTRY
    with tempfile.TemporaryDirectory(prefix='extension_capture_collection_') as temporary:
        path = Path(temporary) / 'config.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        protocol, jobs = replay.prepare(path, output, False)
    directory = output / 'states'
    directory.mkdir()
    records = []
    for job in jobs:
        original = Path(job['decision_basis_target'])
        record = validate_record(original.with_suffix('.json'))
        if record['trials'] != read(Path(job['folder']) / 'prepared.json')['trials']:
            raise ValueError('Scientific collection cannot include short captures')
        write(output / 'jobs' / (job['name'] + '.json'), job)
        for suffix in ('.npz', '.json'):
            shutil.copyfile(original.with_suffix(suffix), Path(job['target']).with_suffix(suffix))
        records.append(record)
    states.estimation_context(protocol)
    write(output / 'completion.json', dict(status='complete', smoke=False, jobs=len(records),
        original_filter_captures=len(records), exact_behavior_replays=0, additional_filters=0,
        state_recording='original_evaluation_filter_capture',
        records=[dict(subject=r['subject'], arm=r['arm'], seed=r['seeds'][0], trials=r['trials']) for r in records]))


def align(source: Path, oral_audit: Path, output: Path) -> None:
    # Load an isolated orchestration module; shared alignment formulas and
    # existing bound sources remain untouched.
    alignment = capture.driver.load_module(
        'src.Bayesian_state.workflows.analysis._extension_capture_alignment', ROOT / ALIGNMENT)
    protocol, done = read(source / 'protocol.json'), read(source / 'completion.json')
    if (done.get('status') != 'complete' or done['smoke'] or protocol['smoke']
            or done.get('original_filter_captures') != protocol['jobs'] or done['jobs'] != protocol['jobs']
            or done.get('exact_behavior_replays') != 0 or done.get('additional_filters') != 0
            or protocol['source_sha256'].get(ENTRY) != digest(ROOT / ENTRY)
            or protocol.get('state_recording') != 'original_evaluation_filter_capture'):
        raise ValueError('Require all original extension filter captures before oral alignment')
    metadata, note = states.estimation_context(protocol)
    metadata.update(state_recording='original_evaluation_filter_capture', additional_filters=0)
    note = note.replace('状态重放逐值复现完整行为评价的选择概率。',
                        '内部状态随完整行为评价的原滤波同步保存，状态记录的选择概率与评价输出逐值一致。')
    original_read, original_dumps = alignment.read, alignment.json.dumps

    def checked_read(path):
        path = Path(path)
        if path.parent.resolve() == (source / 'states').resolve() and path.suffix == '.json':
            record = validate_record(path)
            # The legacy scorer uses this boolean solely as an array-validation
            # gate. Supply it only in memory after capture-specific verification.
            # Saved records and manifests explicitly retain zero replay counts.
            return dict(record, exact_behavior_replay=True)
        return original_read(path)

    def dumps(value, **kwargs):
        value = dict(value, full_replay_complete=False, full_capture_complete=True,
                     original_filter_captures=done['jobs'], additional_filters=0)
        value['source_sha256'] = dict(value['source_sha256'], **{ENTRY: digest(ROOT / ENTRY)})
        return original_dumps(value, **kwargs)

    alignment.read = checked_read
    alignment.estimation_context = lambda _: (metadata, note)
    alignment.json = SimpleNamespace(dumps=dumps, loads=alignment.json.loads)
    alignment.run(source, oral_audit, output, reference_model=protocol['settings']['arms'][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('collect', 'align'), required=True)
    for name in ('summary', 'source', 'oral-audit', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.phase == 'collect' and args.summary and args.output:
        collect(args.summary, args.output)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output:
        align(args.source, args.oral_audit, args.output)
    else:
        parser.error('Collect needs summary/output; align needs source/oral-audit/output')
