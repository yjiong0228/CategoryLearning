"""Audit H autonomous paths with the exp123 per-category stopping criterion.

Uses existing saved paths and the shared criterion implementation. Stimulus
order and length remain those observed; adaptive experimental generation and
the distribution of actual stopping times are outside this diagnostic.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import json

import numpy as np
import pandas as pd

from .model_0923_h_validation import load, digest
from .model_0923_task_criterion import block_criterion


def run(summary: Path, program: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new task-criterion destination')
    source_text = program.read_text()
    threshold_match = re.search(r'accstop\s*=\s*([\d.]+)', source_text)
    size_match = re.search(r'nTpB\s*=\s*\[\s*\d+\s*,\s*(\d+)\s*\]', source_text)
    if threshold_match is None or size_match is None:
        raise ValueError('Task program lacks the documented stopping settings')
    threshold, size = float(threshold_match.group(1)), int(size_match.group(1))
    manifest = load(summary / 'summary_manifest.json')
    fit = load(Path(manifest['fit']) / 'nominees.json')
    generated = Path(manifest['autonomous'])
    rows = load(generated / 'evaluation.json')['rows']
    metadata = {}
    inputs = {str(program): digest(program), str(summary / 'summary_manifest.json'): digest(summary / 'summary_manifest.json')}
    for subject in manifest['subjects']:
        folder = Path(fit['data']) / f'subject_{subject}'
        prepared = load(folder / 'prepared.json')
        file = Path(prepared['paths']['learning_data'])
        if digest(file) != prepared['learning_data_sha256']:
            raise ValueError('Original trial metadata changed')
        frame = pd.read_csv(file)
        frame = frame[frame.iSub == subject].copy()
        if 'miss' in frame:
            frame = frame[frame['miss'] == 0]
        frame = frame.reset_index(drop=True)
        index = pd.MultiIndex.from_frame(frame[['iSession', 'iBlock', 'iTrial']])
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError('Original task trial order is ambiguous')
        parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            file_part = folder / name
            if digest(file_part) != prepared['array_sha256'][name]:
                raise ValueError('Prepared observations changed')
            with np.load(file_part) as saved:
                parts.append({key: saved[key] for key in ('choices', 'categories', 'feedback')})
        for key, column in [('choices', 'choice'), ('categories', 'category'), ('feedback', 'feedback')]:
            np.testing.assert_array_equal(np.concatenate([part[key] for part in parts]), frame[column])
        metadata[subject] = frame
        inputs[str(file)] = digest(file)
    summaries, blocks, paths = [], [], []
    for row in rows:
        subject, model = row['subject'], row['model']
        frame = metadata[subject]
        file = generated / 'evaluate' / f'subject_{subject}' / model / (row['candidate']['id'] + '.npz')
        if digest(file) != row['predictions_sha256']:
            raise ValueError('Autonomous predictions changed')
        with np.load(file) as saved:
            feedback, seeds = saved['feedback'], saved['seeds']
            np.testing.assert_array_equal(saved['categories'], frame.category)
        if feedback.shape != (len(seeds), len(frame)):
            raise ValueError('Autonomous paths and trial metadata do not align')
        first = np.full(len(seeds), -1, dtype=int)
        observed = []
        n_categories = 2 if int(frame.condition.iloc[0]) == 1 else 4
        for number, ((session, block), group) in enumerate(frame.groupby(['iSession', 'iBlock'], sort=False), 1):
            if len(group) != size:
                raise ValueError('Criterion audit requires complete original task blocks')
            outcomes = np.vstack([group.feedback.to_numpy()[None, :], feedback[:, group.index]])
            _, passed = block_criterion(outcomes, group.category.to_numpy(), group.ambiguous.to_numpy(), n_categories, threshold)
            first[(first < 0) & passed[1:]] = number
            observed.append(bool(passed[0]))
            blocks.append(dict(subject=subject, model=model, session=int(session), block=int(block),
                sequence_block=number, observed_pass=bool(passed[0]), simulated_pass_count=int(passed[1:].sum()),
                simulated_pass_fraction=float(passed[1:].mean())))
        summaries.append(dict(subject=subject, model=model, trials=len(frame), blocks=len(observed),
            observed_final_pass=observed[-1], observed_first_pass=(observed.index(True) + 1 if any(observed) else None),
            paths=len(seeds), simulated_any_pass_count=int((first > 0).sum()), simulated_any_pass_fraction=float((first > 0).mean()),
            simulated_final_pass_fraction=float(passed[1:].mean())))
        paths.extend(dict(subject=subject, model=model, seed=int(seed), first_pass_block=int(value) if value > 0 else None,
                          reached=bool(value > 0)) for seed, value in zip(seeds, first))
    output.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(output / 'subjects.csv', index=False)
    pd.DataFrame(blocks).to_csv(output / 'blocks.csv', index=False)
    pd.DataFrame(paths).to_csv(output / 'paths.csv', index=False)
    with (output / 'completion.json').open('x') as handle:
        json.dump(dict(status='complete', groups=len(summaries), paths=len(paths), block_size=size, accuracy_threshold=threshold,
            source_sha256=digest(Path(__file__)), criterion_source_sha256=digest(Path(__file__).with_name('model_0923_task_criterion.py')),
            inputs_sha256=inputs, nomination_sha256=manifest['nomination_sha256'],
            role='post_test_fixed_schedule_criterion_diagnostic; adaptive_stimuli_and_actual_stopping_time_distribution_unmodeled'), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--program', type=Path, default=Path('exp_program/exp123/Task2_learning.m'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.summary, args.program, args.output)


if __name__ == '__main__':
    main()
