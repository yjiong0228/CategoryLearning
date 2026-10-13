"""Versioned subject assessment contract for frozen Model 0923 parameters.

The protocol preserves trial order, scores pre-choice probabilities, and keeps
numerical integration checks separate from statistical parameter uncertainty.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .model_0923_task_criterion import block_criterion

PROTOCOL = 'model0923_subject_evaluation_v1'
REQUIRED_KEYS = {
    'protocol', 'fit_run', 'output_dir', 'figure_output_dir', 'subjects',
    'analysis_seed', 'particle_count', 'pf_repeats', 'parallel_budget',
    'window_size', 'autonomous_rollouts', 'calibration_particle_count',
    'calibration_repeats', 'segment_size', 'late_window', 'target_accuracy',
    'task_criterion',
}
SUBJECT_ARTIFACTS = (
    'summary.csv', 'trials.csv', 'rule_states.csv', 'rule_catalog.csv',
    'pooled_states.npz', 'seed_nll.csv', 'budget_comparison.csv',
    'learning.csv', 'learning_segments.csv', 'task_criterion.json', 'task_metrics.json',
    'criterion_blocks.csv', 'criterion_trajectories.csv', 'evaluation.json',
    'autonomous_rollouts.npz', 'original_seed_replay.npz', 'fixed_simulation.yaml',
)
SUBJECT_FIGURES = ('behavior.png', 'representations.png', 'diagnostics.png')
EXTENSIONS = (
    'held_out_prediction', 'parameter_uncertainty', 'parameter_recovery',
    'state_and_model_recovery', 'mechanism_comparison', 'oral_validation',
    'adaptive_task_generation', 'whole_internal_paths',
)


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_protocol(path: Path, *, smoke: bool = False) -> dict[str, Any]:
    """Reject silent omissions/unknown settings before creating output paths."""
    settings = yaml.safe_load(path.read_text())
    if not isinstance(settings, dict) or set(settings) != REQUIRED_KEYS:
        raise ValueError('Unknown or missing subject evaluation configuration keys')
    if settings['protocol'] != PROTOCOL:
        raise ValueError('Unsupported Model 0923 subject evaluation protocol')
    subjects = settings['subjects']
    if (not isinstance(subjects, list) or not subjects or
            any(type(s) is not int for s in subjects) or len(set(subjects)) != len(subjects)):
        raise ValueError('Need distinct integer subject IDs')
    for key in ('particle_count', 'pf_repeats', 'parallel_budget', 'window_size',
                'autonomous_rollouts', 'calibration_particle_count',
                'calibration_repeats', 'segment_size', 'late_window'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if settings['pf_repeats'] < 2 or not 2 <= settings['calibration_repeats'] <= settings['pf_repeats']:
        raise ValueError('Need at least two paired seed repeats within the base repeat budget')
    if settings['calibration_particle_count'] <= settings['particle_count']:
        raise ValueError('Calibration must use a larger particle budget')
    if type(settings['analysis_seed']) is not int or settings['analysis_seed'] < 0:
        raise ValueError('analysis_seed must be a nonnegative integer')
    if (type(settings['target_accuracy']) not in (int, float) or
            not np.isfinite(settings['target_accuracy']) or not 0 <= settings['target_accuracy'] <= 1):
        raise ValueError('Invalid descriptive accuracy target')
    criterion = settings['task_criterion']
    if not isinstance(criterion, dict) or criterion.get('mode') not in ('exp123', 'not_configured'):
        raise ValueError('Declare exp123 task criterion or not_configured')
    if criterion['mode'] == 'exp123':
        if set(criterion) != {'mode', 'source_program', 'block_size', 'accuracy_threshold'}:
            raise ValueError('Incomplete exp123 task criterion specification')
        if type(criterion['block_size']) is not int or criterion['block_size'] < 1:
            raise ValueError('Invalid task block size')
        if (type(criterion['accuracy_threshold']) not in (int, float) or
                not np.isfinite(criterion['accuracy_threshold']) or not 0 <= criterion['accuracy_threshold'] <= 1):
            raise ValueError('Invalid task accuracy threshold')
        criterion['source_program'] = str(resolve(path.parent, criterion['source_program']))
    elif set(criterion) != {'mode', 'reason'} or not criterion['reason']:
        raise ValueError('Explain why a task criterion is not configured')
    for key in ('fit_run', 'output_dir', 'figure_output_dir'):
        settings[key] = str(resolve(path.parent, settings[key]))
    if smoke:
        settings.update(subjects=subjects[:1], particle_count=4, pf_repeats=2,
                        parallel_budget=1, window_size=4, autonomous_rollouts=2,
                        calibration_particle_count=8, calibration_repeats=2,
                        segment_size=8, late_window=8)
    return settings


def criterion_tables(frame: pd.DataFrame, feedback: np.ndarray, seeds: np.ndarray,
                     n_categories: int, spec: dict[str, Any], *, smoke: bool = False
                     ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate complete original blocks, retaining explicit unavailable states.

    Criterion accuracy includes the task's first trial. Choice NLL excludes the
    model initialization trial. Missed/incomplete blocks are not reconstructed.
    """
    block_columns = ['block_index', 'iSession', 'iBlock', 'complete',
                     'observed_reached', 'autonomous_pass_fraction']
    trajectory_columns = ['rollout', 'seed', 'reached', 'first_pass_block']
    empty = (pd.DataFrame(columns=block_columns), pd.DataFrame(columns=trajectory_columns))
    if smoke or spec['mode'] == 'not_configured':
        status = 'not_performed_smoke' if smoke else 'not_configured'
        return dict(status=status, reason='Short pipeline check' if smoke else spec['reason']), *empty
    required = {'iSession', 'iBlock', 'ambiguous', 'category', 'feedback'}
    if not required.issubset(frame.columns):
        return dict(status='unavailable', reason='Missing original block or ambiguity annotations'), *empty
    if frame['ambiguous'].isna().any():
        raise ValueError('A task trial lacks its ambiguity annotation')
    outcomes = np.asarray(feedback, float)
    if outcomes.shape != (len(seeds), len(frame)):
        raise ValueError('Autonomous feedback does not match original trial identifiers')
    first_pass = np.full(len(seeds), -1, int)
    rows, observed_first = [], None
    ordered = frame.reset_index(drop=True)
    for index, ((session, block), group) in enumerate(ordered.groupby(['iSession', 'iBlock'], sort=False), 1):
        row = dict(block_index=index, iSession=int(session), iBlock=int(block),
                   complete=len(group) == spec['block_size'], observed_reached=None,
                   autonomous_pass_fraction=None)
        if row['complete']:
            indices = group.index.to_numpy()
            both = np.vstack((group.feedback.to_numpy()[None, :], outcomes[:, indices]))
            accuracy, reached = block_criterion(both, group.category.to_numpy(),
                group.ambiguous.to_numpy(), n_categories, spec['accuracy_threshold'])
            row.update(observed_reached=bool(reached[0]), autonomous_pass_fraction=float(reached[1:].mean()))
            row.update({f'observed_category_{c + 1}_accuracy': float(accuracy[0, c]) for c in range(n_categories)})
            first_pass[(first_pass < 0) & reached[1:]] = index
            if reached[0] and observed_first is None:
                observed_first = index
        rows.append(row)
    blocks = pd.DataFrame(rows)
    complete = bool(blocks.complete.all())
    trajectories = pd.DataFrame([dict(rollout=i, seed=int(seed), reached=bool(first_pass[i] >= 0),
        first_pass_block=int(first_pass[i]) if first_pass[i] >= 0 else None) for i, seed in enumerate(seeds)])
    summary = dict(status='completed' if complete else 'incomplete_blocks',
                   total_blocks=len(blocks), evaluated_blocks=int(blocks.complete.sum()),
                   observed_first_pass_block=observed_first,
                   autonomous_any_pass_fraction=float((first_pass >= 0).mean()) if complete else None,
                   scope='Original stimulus schedule and recorded length; all task trials in complete blocks')
    return summary, blocks, trajectories


def artifact_manifest(output: Path, figures: Path, subjects: list[int]) -> dict[str, str]:
    """Require the fixed report bundle and hash each portable relative artifact."""
    paths = [output / name for name in ('subjects.csv', 'parameters.csv', 'restarts.csv',
             'finalists.csv', 'search_summary.csv', 'original_replay_validation.json',
             'effective_config.yaml', 'report.html')]
    paths += [figures / name for name in ('cohort_behavior.png', 'parameter_audit.png')]
    for sid in subjects:
        paths += [output / f'subject_{sid}' / name for name in SUBJECT_ARTIFACTS]
        paths += [figures / f'subject_{sid}' / name for name in SUBJECT_FIGURES]
        paths += sorted((output / f'subject_{sid}' / 'seeds').glob('*.npz'))
        paths += [figures / f'subject_{sid}' / name for name in
                  ('behavior_plot_source.csv', 'diagnostics_plot_source.csv')]
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required evaluation artifact is missing: {path}')
        prefix, relative = ('report', path.relative_to(output)) if path.is_relative_to(output) else ('figures', path.relative_to(figures))
        result[f'{prefix}/{relative.as_posix()}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
