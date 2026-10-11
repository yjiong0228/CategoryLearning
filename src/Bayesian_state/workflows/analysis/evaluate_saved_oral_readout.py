"""Evaluate generated oral reports from saved Model 0826 behavioral fits.

Replay each selected parameter set once with the seed and particle count of its
saved state. Oral reports enter only the final scoring step. The historical
beta update is loaded from Git and checked against the original fit manifest.
Every replay must reproduce the saved choice probabilities before scoring.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ...evaluation.oral.structure_0923 import VERSION, encode_report
from ...model.modules.beta import BetaModule
from ...model.modules.readout.oral import OralReadoutParameters
from ...model.modules.readout.report_space import report_code
from ...optimization.adaptive_fit import make_context
from ...simulation.execution import evaluate_state_model_run
from ...simulation.parameters import apply_fixed_hyperparams_to_engine_config
from ...utils.parallel import parallel_job_count, single_threaded_processes

ROOT = Path(__file__).resolve().parents[4]
BETA_PATH = 'src/Bayesian_state/model/modules/beta.py'
DEFAULT_BETA_REF = 'a6a3b65a98dffb8644b7faf3e26fa10a32e65268'
PARAMETERS = OralReadoutParameters(
    mention_probability=.8, extra_probability=.2, other_probability=.02,
    body_naming_probability=.5, body_reference_center=.75,
    body_reference_width=.1, extra_selection='salience',
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def resolve_saved_fits(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve parameter nominees without selecting on oral performance."""
    subjects: dict[int, dict[str, Any]] = {}
    for directory in config['source_runs']:
        source = ROOT / directory
        fit = json.loads((source / 'fit_results.json').read_text())
        manifest = json.loads((source / 'manifest.json').read_text())
        for sid, result in fit['subjects'].items():
            sid = int(sid)
            if sid in subjects:
                raise ValueError(f'Duplicate fitted subject: {sid}')
            spec = next(s for s in manifest['subjects'] if s['subject'] == sid)
            subjects[sid] = dict(fit=result, spec=spec, fit_config=manifest['config'],
                                 source=source, source_manifest=manifest)
    if set(subjects) != set(config['subjects']):
        raise ValueError('Fit cohort and requested cohort differ')
    return [subjects[sid] for sid in config['subjects']]


def frozen_beta_source(ref: str) -> tuple[bytes, str]:
    """Retrieve the fitted version; Git arguments are passed without a shell."""
    revision = subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT).decode().strip()
    source = subprocess.check_output(['git', 'show', f'{revision}:{BETA_PATH}'], cwd=ROOT)
    return source, revision


@contextmanager
def saved_beta_update(source: bytes, manifest: dict[str, Any]):
    """Temporarily use the fitted beta method within one isolated worker.

    The surrounding model uses the shared core. Numerical agreement with the
    saved trajectory is checked after replay, including unchanged conditions.
    """
    recorded = [v for k, v in manifest['input_sha256'].items() if k.endswith('/' + BETA_PATH)]
    if recorded != [hashlib.sha256(source).hexdigest()]:
        raise ValueError('Historical beta source differs from the fitted version')
    historical = ModuleType(BetaModule.__module__.rsplit('.', 1)[0] + '._saved_fit_beta')
    historical.__package__ = BetaModule.__module__.rsplit('.', 1)[0]
    exec(compile(source, '<saved_fit_beta>', 'exec'), historical.__dict__)
    current = BetaModule._update_beta_probabilistic_feedback
    BetaModule._update_beta_probabilistic_feedback = historical.BetaModule._update_beta_probabilistic_feedback
    try:
        yield
    finally:
        BetaModule._update_beta_probabilistic_feedback = current


def _limit_arrays(arrays, count: int):
    fields = ('stimulus', 'choices', 'feedback', 'categories', 'target_probs', 'presskeys')
    return replace(arrays, **{key: None if getattr(arrays, key) is None else getattr(arrays, key)[:count]
                             for key in fields})


def replay_one(row: dict[str, Any], *, states: Path, output: Path, beta_source: bytes,
               smoke_trials: int | None = None) -> dict[str, Any]:
    sid = int(row['fit']['subject'])
    reference_path = states / f'S{sid}' / 'selected_00.npz'
    with np.load(reference_path, allow_pickle=False) as reference:
        seed, particles = int(reference['seed']), int(reference['particles'])
        if str(reference['point_id']) != row['fit']['selected']:
            raise ValueError(f'S{sid}: saved parameter nominee mismatch')
        expected = np.array(reference['pred_category_probs'])
        expected_choice = np.array(reference['observed_choice'])
        state_reference = {key: np.array(reference[key]) for key in (
            'marginal_prior', 'marginal_active_probability', 'marginal_executed_probability',
            'predictive_executed_beta', 'pairing_prior', 'pre_choice_ess', 'post_choice_ess',
        ) if key in reference}
    context = make_context(row['spec'], row['fit_config'], False)
    arrays = context['arrays']
    np.testing.assert_array_equal(arrays.choices, expected_choice, err_msg=f'S{sid} trial order')
    if smoke_trials is not None:
        arrays = _limit_arrays(arrays, smoke_trials)
        expected = expected[:smoke_trials]
    engine = apply_fixed_hyperparams_to_engine_config(context['engine'], row['fit']['hyperparams'])
    engine['inference']['particle_count'] = particles
    engine['oral_readout'] = {'method': 'rule_content_v1', 'kwargs': asdict(PARAMETERS)}
    began = perf_counter()
    with saved_beta_update(beta_source, row['source_manifest']):
        result = evaluate_state_model_run(
            subject_id=sid, condition=context['condition'], arrays=arrays, params={},
            engine_config_template=engine, processed_data_dir=context['processed_dir'],
            window_size=min(32, len(arrays.choices)-1), keep_logs=True, prediction_mode='prior_t',
            selection_prediction_mode='prior_t', loss_metric='choice_nll', trajectory_seed=seed,
        )
    metrics = result.metrics_by_mode['prior_t']
    predicted = np.asarray(metrics['pred_category_probs'])
    difference = float(np.max(np.abs(predicted - expected)))
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-12,
                               err_msg=f'S{sid}: choice replay differs from original saved fit')
    state = result.state_log
    state_differences = {}
    for key, saved in state_reference.items():
        saved = saved[:len(arrays.choices)]
        actual = np.asarray(state[key])
        np.testing.assert_allclose(actual, saved, rtol=0, atol=1e-12, equal_nan=True,
                                   err_msg=f'S{sid}: saved cognitive state {key}')
        finite = np.isfinite(saved)
        state_differences[key] = float(np.max(np.abs(actual[finite]-saved[finite]))) if finite.any() else 0.
    oral = np.asarray(state['oral_report_probabilities'])
    valid = np.asarray(state['oral_report_valid_mask'], dtype=bool)
    if not np.isfinite(oral[valid]).all() or np.any(oral[valid] < 0):
        raise ValueError('Invalid oral probability')
    np.testing.assert_allclose(oral[valid].sum(axis=1), 1, rtol=0, atol=1e-12)
    destination = output / f'S{sid}.npz'
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(destination, oral_report_codes=np.asarray(state['oral_report_codes']),
                        oral_report_probabilities=oral, oral_report_valid_mask=valid,
                        pred_category_probs=predicted, observed_choice=arrays.choices, stimulus=arrays.stimulus,
                        observed_feedback=arrays.feedback, seed=seed, particles=particles,
                        subject=sid, condition=context['condition'])
    receipt = dict(subject=sid, condition=context['condition'], trials=len(arrays.choices),
                   seed=seed, particles=particles, seconds=perf_counter()-began,
                   choice_max_absolute_difference=difference,
                   state_max_absolute_differences=state_differences,
                   vocabulary_size=oral.shape[1], state_source=_relative(reference_path),
                   state_source_sha256=_sha(reference_path),
                   fit_source=_relative(row['source']), point_id=row['fit']['selected'],
                   fit_status=row['fit']['status'], fit_issues=row['fit']['issues'])
    print(json.dumps(receipt), flush=True)
    return receipt


def score_replays(trials_path: Path, receipts: list[dict], replays: Path, output: Path) -> None:
    from ...evaluation.oral.generative_scoring import score_report_predictions, summarize_report_scores

    trials = pd.read_csv(trials_path, keep_default_na=False)
    cache, frames = {}, []
    for receipt in receipts:
        sid = receipt['subject']
        frame = trials.loc[trials.iSub == sid].copy().reset_index(drop=True)
        frame = frame.iloc[:receipt['trials']].copy()
        if not np.array_equal(frame.trial, np.arange(1, len(frame)+1)):
            raise ValueError(f'S{sid}: report trial order mismatch')
        parsed = []
        for row in frame.itertuples(index=False):
            features = tuple(getattr(row, f'feature{i}_name') for i in range(1, 5))
            key = (row.text, features)
            if key not in cache:
                cache[key] = encode_report(row.text, features)
            parsed.append(cache[key])
        frame['parse_status'] = [p.status for p in parsed]
        frame['report_code'] = [report_code(p.tokens) if p.status == 'coded' else '' for p in parsed]
        for field in ('relations', 'unparsed', 'flags', 'review_suggestions'):
            frame[field] = [json.dumps(getattr(p, field), ensure_ascii=False) for p in parsed]
        with np.load(replays / f'S{sid}.npz', allow_pickle=False) as run:
            np.testing.assert_array_equal(frame.choice, run['observed_choice'])
            np.testing.assert_array_equal(frame.feedback, run['observed_feedback'])
            if 'stimulus' in run:
                np.testing.assert_allclose(frame[[f'feature{i}' for i in range(1, 5)]].to_numpy(),
                                           run['stimulus'], rtol=0, atol=1e-14)
            score_mask = frame.scored.astype(str).str.lower().isin(['true', '1']).to_numpy()
            result = score_report_predictions(tuple(run['oral_report_codes']),
                                             run['oral_report_probabilities'], frame.report_code,
                                             frame.parse_status, score_mask)
        for key, values in result.items():
            frame[key] = values
        # Keep original oral text and trial identity, not old inverse-encoding metrics.
        keep = ['iSub', 'condition', 'iSession', 'iBlock', 'iTrial', 'trial', 'choice',
                'feedback', 'text', *[f'feature{i}_name' for i in range(1, 5)],
                'scored', 'parse_status', 'report_code', 'relations', 'unparsed', 'flags',
                'review_suggestions', *result]
        frames.append(frame.loc[:, list(dict.fromkeys(keep))])
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / 'trials.csv', index=False)
    summarize_report_scores(combined).to_csv(output / 'subjects.csv', index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-config', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--trials', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-beta-ref', default=DEFAULT_BETA_REF)
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--smoke-trials', type=int, help='Prefix validation, retaining original particles/seeds')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.cohort_config.read_text())
    rows = resolve_saved_fits(config)
    if args.subjects:
        if not set(args.subjects).issubset(config['subjects']):
            raise ValueError('Requested subject absent from saved cohort')
        rows = [row for row in rows if row['fit']['subject'] in args.subjects]
    if args.jobs < 1 or (args.smoke_trials is not None and args.smoke_trials < 2):
        raise ValueError('Positive jobs and at least two smoke trials are required')
    source, revision = frozen_beta_source(args.legacy_beta_ref)
    args.output.mkdir(parents=True)
    replays = args.output / 'replays'
    replays.mkdir()
    paths = [args.cohort_config, args.trials, Path(__file__)]
    for row in rows:
        paths.extend([row['source']/'fit_results.json', row['source']/'manifest.json', Path(row['spec']['engine'])])
    manifest = dict(
        evaluation='model_0826_saved_fit_current_section8_generated_report',
        oral_parameters=asdict(PARAMETERS), encoder_version=VERSION,
        behavior_beta_git_revision=revision, behavior_beta_sha256=hashlib.sha256(source).hexdigest(),
        timing='after_observed_choice_before_current_feedback',
        conditioning='pool_particle_joint_choice_report_then_divide_choice_probability',
        oral_use='external_prediction_scoring_only', sample_role='development',
        expression_parameters_selected_from='model_0923_section8_shared_demonstration_settings',
        repeats_per_subject=1, parameter_variant='original_selected_nominee',
        smoke_trials=args.smoke_trials, sources={_relative(p): _sha(p) for p in paths},
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
    )
    _json(args.output / 'manifest.json', manifest)
    workers = parallel_job_count(args.jobs, len(rows))
    with single_threaded_processes():
        receipts = Parallel(n_jobs=workers)(delayed(replay_one)(
            row, states=args.states, output=replays, beta_source=source, smoke_trials=args.smoke_trials)
            for row in rows)
    _json(args.output / 'replay_validation.json', receipts)
    score_replays(args.trials, receipts, replays, args.output)
    _json(args.output / 'completion.json', dict(subjects=len(rows), trials=sum(r['trials'] for r in receipts),
                                               workers=workers, choice_replay_verified=True))


if __name__ == '__main__':
    main()
