"""Score captured simplification states with the existing fixed oral constraints.

Use --evaluation EVALUATION --oral-audit AUDIT --output NEW. Support, conflict,
undetermined mass and complete-report denominators reuse group_scores. This
entry joins already frozen behavioral fits to the declared coding; it estimates
no model parameters and preserves capture provenance without claiming replay.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_0923_decision_basis_alignment as alignment
from . import model_1021_simplification_capture as capture

ROOT = Path(__file__).resolve().parents[4]
read, write, digest = capture.read, capture.write, capture.digest


def run(evaluation: Path, audit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new simplification oral directory')
    protocol = read(evaluation / 'protocol.json')
    fit = Path(protocol['fit'])
    frozen = read(fit / 'nominees.json')
    if (read(evaluation / 'completion.json')['status'] != 'complete' or frozen['smoke']
            or protocol['nomination_sha256'] != digest(fit / 'nominees.json')):
        raise ValueError('Require a completed frozen-parameter evaluation')
    manifest = read(audit / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete fixed oral coding')
    for name, sha in manifest['input_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Oral audit input changed')
    for name in ('trials.csv', 'component_matches.csv'):
        if digest(audit / name) != manifest['artifacts'][name]:
            raise ValueError('Oral coding changed')
    declaration = read(evaluation / 'decision_basis_capture_protocol.json')
    if declaration['source_sha256'] != capture.sources() or declaration['additional_filters'] != 0:
        raise ValueError('Capture source or filter accounting changed')
    settings = frozen['settings']
    nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
    expected = {(s, m) for s in settings['subjects'] for m in settings['arms']}
    groups, inputs = {}, {}
    for file in sorted((evaluation / 'state_capture').glob('*.json')):
        record = read(file)
        key = record['subject'], record['model']
        if (key not in expected or record['smoke'] or record['candidate'] != nominees[key]
                or record['capture_source_sha256'] != declaration['source_sha256']):
            raise ValueError('Captured state differs from its frozen model')
        alignment.validate_basis_record(file.with_suffix('.npz'), record)
        job = read(Path(record['original_job']))
        if job['source_sha256'] != frozen['source_sha256']:
            raise ValueError('Captured state uses different fitted source')
        groups.setdefault(key, []).append((file, record))
        inputs[str(file.resolve())] = digest(file)
        inputs[str(file.with_suffix('.npz').resolve())] = record['array_sha256']
    count = settings['budgets']['evaluate']['seeds']
    if set(groups) != expected or any(len(rows) != count or len({r['seeds'][0] for _, r in rows}) != count
                                      for rows in groups.values()):
        raise ValueError('Missing or duplicated captured evaluation seeds')
    oral, matches = [pd.read_csv(audit / name) for name in ('trials.csv', 'component_matches.csv')]
    tables, summaries, jackknives = [], [], {}
    for key, records in sorted(groups.items()):
        table, rows, estimates = alignment.group_scores(evaluation, records, oral, matches)
        tables.append(table)
        summaries.extend(rows)
        jackknives[key] = estimates
        print(f'S{key[0]} {key[1]}: {len(table)} trials, {len(records)} filter seeds', flush=True)
    contrasts = []
    for subject in settings['subjects']:
        for key, estimate in jackknives[subject, 'COMPACT'].items():
            baseline = jackknives[subject, 'REFERENCE'][key]
            np.testing.assert_array_equal(estimate['seeds'], baseline['seeds'])
            differences = estimate['values'] - baseline['values']
            n = len(differences)
            error = np.sqrt((n-1)/n * np.sum((differences-differences.mean(0))**2, axis=0))
            row = dict(subject=subject, window=key[0], domain=key[1], subset=key[2])
            for i, label in enumerate(alignment.LABELS):
                row[label + '_difference'] = estimate['point'][i] - baseline['point'][i]
                row[label + '_difference_mcse'] = error[i]
            contrasts.append(row)
    output.mkdir(parents=True)
    for name, frame in [('trials', pd.concat(tables, ignore_index=True)), ('summary', pd.DataFrame(summaries)),
                        ('contrasts', pd.DataFrame(contrasts))]:
        frame.to_csv(output / (name + '.csv'), index=False)
    for file in (audit / 'manifest.json', fit / 'nominees.json', evaluation / 'protocol.json',
                 evaluation / 'decision_basis_capture_protocol.json', Path(__file__), Path(alignment.__file__)):
        inputs[str(file.resolve())] = digest(file)
    write(output / 'completion.json', dict(status='complete', evaluation=str(evaluation.resolve()),
        oral_audit=str(audit.resolve()), subjects=settings['subjects'], inputs_sha256=inputs,
        state_recording='original_filter_capture', additional_filters=0,
        interpretation='Noisy content compatibility; not a probability of generating a report',
        development_sample=True, parameter_estimation='original training behavior only',
        primary_denominator='unchanged complete reports shared by both models',
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'oral-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.evaluation, args.oral_audit, args.output)
