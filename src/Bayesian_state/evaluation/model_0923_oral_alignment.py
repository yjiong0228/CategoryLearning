"""Evaluate complete oral reports from fixed-parameter Model 0923 predictions.

Reports are joined to the original behavioral trial identities and scored only
after prediction. Each PF repeat already conditions its report distribution on
the actual choice; independent repeats are then averaged with equal weights.
The expression settings and report encoder remain fixed during this evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .oral.generative_scoring import score_report_predictions, summarize_report_scores
from .oral.structure_0923 import VERSION, encode_report
from ..model.modules.readout.oral import OralReadoutParameters
from ..model.modules.readout.report_space import report_code

TRIAL_KEYS = ('iSub', 'iSession', 'iBlock', 'iTrial')
FEATURE_NAMES = tuple(f'feature{i}_name' for i in range(1, 5))
FEATURE_VALUES = tuple(f'feature{i}' for i in range(1, 5))
READOUT_KEYS = {
    'mention_probability', 'extra_probability', 'other_probability',
    'body_naming_probability', 'body_reference_center', 'body_reference_width',
    'extra_selection',
}
WINDOW_SIZE = 64
MIN_CODED_REPORTS = 16
ARTIFACTS = ('oral_trials.csv', 'oral_summary.csv', 'oral_predictions.npz')


def validate_oral_spec(spec: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    """Validate an explicit prediction source or an explained unavailable source."""
    if not isinstance(spec, Mapping):
        raise ValueError('oral_alignment must be a mapping')
    if spec.get('mode') == 'not_available':
        if set(spec) != {'mode', 'reason'} or not isinstance(spec['reason'], str) or not spec['reason'].strip():
            raise ValueError('Unavailable oral observations require a reason')
        return dict(spec)
    if spec.get('mode') != 'predict' or set(spec) != {'mode', 'trials_path', 'readout'}:
        raise ValueError('Oral prediction requires mode, trials_path and readout')
    readout = spec['readout']
    if (not isinstance(readout, Mapping) or set(readout) != {'method', 'kwargs'}
            or not isinstance(readout['kwargs'], Mapping) or set(readout['kwargs']) != READOUT_KEYS):
        raise ValueError('Declare all seven common oral expression settings')
    if readout['kwargs']['extra_selection'] != 'salience':
        raise ValueError('This oral protocol requires salience for supplementary features')
    if any(type(readout['kwargs'][name]) not in (int, float) for name in READOUT_KEYS - {'extra_selection'}):
        raise ValueError('Oral expression parameters must be finite numerical values')
    OralReadoutParameters.from_config(readout)
    value = spec['trials_path']
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError('Declare an oral trials CSV')
    source = Path(value)
    source = source.resolve() if source.is_absolute() else (base_dir / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {'mode': 'predict', 'trials_path': str(source), 'readout': deepcopy(dict(readout))}


def _equal_numeric(actual: Any, expected: Any, name: str) -> None:
    first, second = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError(f'Oral source has invalid {name}')
    if not np.allclose(first, second, rtol=0, atol=1e-12):
        raise ValueError(f'Oral source differs from behavioral input: {name}')


def load_observed_reports(inputs: Mapping[str, Any], spec: Mapping[str, Any]) -> pd.DataFrame | None:
    """Join raw reports one-to-one by original subject/session/block/trial keys.

    Existing analysis tables can provide the original text and feature order.
    Their old model states, oral scores and fitted outcomes are never read.
    Additional source rows are allowed for a prefix smoke check; every retained
    behavioral trial must have exactly one source row, including empty reports.
    """
    if spec['mode'] == 'not_available':
        return None
    source_path = Path(spec['trials_path'])
    needed = set(TRIAL_KEYS + FEATURE_NAMES + FEATURE_VALUES + ('condition', 'choice', 'category', 'feedback', 'text'))
    source = pd.read_csv(source_path, keep_default_na=False, usecols=lambda name: name in needed)
    if not needed.issubset(source.columns):
        raise ValueError(f'Oral source lacks original identities or observations: {sorted(needed - set(source))}')
    subject = int(inputs['subject'])
    source = source.loc[pd.to_numeric(source.iSub, errors='raise').eq(subject)].copy()
    if source.duplicated(list(TRIAL_KEYS)).any():
        raise ValueError('Oral source contains duplicate original trial identities')
    original = inputs['frame'].copy().reset_index(drop=True)
    if 'iSub' not in original:
        original['iSub'] = subject
    if not set(TRIAL_KEYS).issubset(original):
        raise ValueError('Behavioral input lacks original trial identities for oral evaluation')
    if original.duplicated(list(TRIAL_KEYS)).any():
        raise ValueError('Behavioral input contains duplicate original trial identities')
    identity = original.loc[:, list(TRIAL_KEYS)].copy()
    identity['_oral_order'] = np.arange(len(identity))
    joined = identity.merge(source, how='left', on=list(TRIAL_KEYS), validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Oral source lacks a retained behavioral trial')
    joined = joined.sort_values('_oral_order', kind='stable').drop(columns=['_oral_order', '_merge']).reset_index(drop=True)
    if len(joined) != len(inputs['arrays'].choices):
        raise ValueError('Oral source and retained model trials differ in length')
    _equal_numeric(joined.condition, np.full(len(joined), int(inputs['condition'])), 'condition')
    _equal_numeric(joined.choice, inputs['arrays'].choices, 'choice')
    _equal_numeric(joined.feedback, inputs['arrays'].feedback, 'feedback')
    _equal_numeric(joined.loc[:, list(FEATURE_VALUES)], inputs['arrays'].stimulus, 'stimulus')
    categories = getattr(inputs['arrays'], 'categories', None)
    if categories is None:
        if 'category' not in original:
            raise ValueError('Behavioral input lacks category observations for oral source verification')
        categories = original.category
    _equal_numeric(joined.category, categories, 'category')
    for name in FEATURE_NAMES:
        if name in original and not np.array_equal(joined[name].to_numpy(), original[name].to_numpy()):
            raise ValueError(f'Oral source differs from original feature order: {name}')
    for row in joined.loc[:, list(FEATURE_NAMES)].itertuples(index=False, name=None):
        # Validate even missing reports, whose encoder otherwise returns early.
        encode_report('', tuple(row))
    joined['trial'] = np.arange(1, len(joined) + 1)
    joined.attrs['oral_spec'] = deepcopy(dict(spec))
    joined.attrs['source_sha256'] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return joined


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('infinity' if value > 0 else '-infinity')
    return value


def _write_artifacts(output: Path, trials: pd.DataFrame, metrics: pd.DataFrame,
                     predictions: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if (output / name).exists():
            raise FileExistsError(output / name)
    trials.to_csv(output / ARTIFACTS[0], index=False)
    metrics.to_csv(output / ARTIFACTS[1], index=False)
    with (output / ARTIFACTS[2]).open('xb') as stream:
        np.savez_compressed(stream, **predictions)


def summarize_oral_alignment(
    inputs: Mapping[str, Any], seedfiles: Sequence[Path], report_frame: pd.DataFrame | None,
    output: Path, *, smoke: bool = False, spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write fixed report metrics, original-trial curves and prediction provenance.

    Complete reports outside the vocabulary retain zero probability in the
    denominator. Exact NLL therefore retains infinity; JSON represents it as
    the string ``infinity``. Missing and unresolved reports remain separate.
    A trailing window uses 64 actual trials and at least 16 coded reports.
    Undefined current reports leave gaps in its plotted comparisons.
    """
    spec = dict(spec or (report_frame.attrs.get('oral_spec', {}) if report_frame is not None else inputs.get('oral_spec', {})))
    common = dict(subject=int(inputs['subject']), condition=int(inputs['condition']),
                  scope='smoke_only' if smoke else 'full_sequence', encoder_version=VERSION,
                  window_size=WINDOW_SIZE, min_coded_reports=MIN_CODED_REPORTS,
                  timing='after_observed_choice_before_current_feedback',
                  oral_use='external_prediction_scoring_only',
                  geometry_and_rule_state_alignment={'status': 'pending', 'reason':
                      'Versioned geometry mapping and rule-state comparison are required for the current catalogue.'})
    if report_frame is None:
        if spec.get('mode') != 'not_available':
            raise ValueError('Missing oral observations require an explicit not_available specification')
        summary = dict(common, status='not_available', reason=spec['reason'], metrics={})
        _write_artifacts(output, pd.DataFrame(columns=[*TRIAL_KEYS, 'trial', 'parse_status', 'scored', 'oral_score_mask',
                                                     'report_probability', 'report_top1_hit', 'report_top3_hit']),
                         pd.DataFrame([dict(iSub=common['subject'], condition=common['condition'],
                                            status='not_available', reason=spec['reason'])]),
                         dict(status=np.asarray('not_available'), reason=np.asarray(spec['reason']),
                              oral_report_codes=np.asarray([], dtype=str),
                              oral_report_probabilities=np.empty((0, 0)), oral_report_valid_mask=np.empty(0, bool)))
        return summary
    if not seedfiles or spec.get('mode') != 'predict':
        raise ValueError('Oral prediction requires independent seed files and its fixed specification')
    reports = report_frame.copy().reset_index(drop=True)
    if len(reports) != len(inputs['arrays'].choices) or len(reports) < 2:
        raise ValueError('Oral reports and retained behavioral trials differ in length')
    _equal_numeric(reports.iSub, np.full(len(reports), common['subject']), 'subject')
    _equal_numeric(reports.condition, np.full(len(reports), common['condition']), 'condition')
    _equal_numeric(reports.choice, inputs['arrays'].choices, 'choice')
    codes, valid, repeats, seeds = None, None, [], []
    metadata, sources = [], []
    for path in seedfiles:
        with np.load(path, allow_pickle=False) as run:
            current_codes = np.asarray(run['oral_report_codes'])
            probabilities = np.asarray(run['oral_report_probabilities'], dtype=float)
            current_valid = np.asarray(run['oral_report_valid_mask'])
            if (current_codes.ndim != 1 or len(current_codes) == 0 or current_valid.dtype.kind != 'b'
                    or current_valid.shape != (len(reports),)
                    or probabilities.shape != (len(reports), len(current_codes))):
                raise ValueError('Seed oral prediction arrays do not match retained trials')
            if codes is None:
                codes, valid = current_codes.copy(), current_valid.copy()
            elif not np.array_equal(codes, current_codes) or not np.array_equal(valid, current_valid):
                raise ValueError('Independent oral repeats disagree on vocabulary or validity mask')
            if (not np.isfinite(probabilities[current_valid]).all()
                    or np.any(probabilities[current_valid] < 0)
                    or not np.allclose(probabilities[current_valid].sum(axis=1), 1., rtol=1e-10, atol=1e-10)
                    or not np.isnan(probabilities[~current_valid]).all()):
                raise ValueError('Seed oral predictions must be normalized with NaN invalid rows')
            repeats.append(probabilities)
            seeds.append(int(run['seed']))
            if 'oral_readout_metadata_json' not in run:
                raise ValueError('Seed oral prediction lacks readout provenance')
            receipt = json.loads(str(run['oral_readout_metadata_json']))
            if (receipt.get('method') != spec['readout']['method']
                    or receipt.get('parameters') != spec['readout']['kwargs']
                    or receipt.get('timing') != common['timing']):
                raise ValueError('Seed oral prediction differs from the frozen readout specification')
            metadata.append(receipt)
        sources.append({'file': Path(path).name, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    if len(set(seeds)) != len(seeds):
        raise ValueError('Oral repeats must use distinct independent seeds')
    probabilities = np.mean(np.stack(repeats), axis=0)
    scored = np.ones(len(reports), dtype=bool)
    scored[0] = False
    if not valid[scored].all():
        raise ValueError('A behavioral scoring trial lacks an oral prediction')
    parsed = [encode_report(row.text, tuple(getattr(row, name) for name in FEATURE_NAMES))
              for row in reports.itertuples(index=False)]
    reports['parse_status'] = [report.status for report in parsed]
    reports['report_code'] = [report_code(report.tokens) if report.status == 'coded' else '' for report in parsed]
    for name in ('relations', 'unparsed', 'flags', 'review_suggestions'):
        reports[name] = [json.dumps(getattr(report, name), ensure_ascii=False) for report in parsed]
    reports['scored'] = scored
    reports['oral_prediction_valid'] = valid
    # Invalid initialization rows are unscored; the scorer's normalized-row
    # contract is satisfied locally while original prediction NaNs are retained.
    scoring_probabilities = probabilities.copy()
    scoring_probabilities[~valid] = 1. / len(codes)
    scores = score_report_predictions(tuple(codes.tolist()), scoring_probabilities,
                                     reports.report_code, reports.parse_status, scored)
    for name, values in scores.items():
        reports[name] = values
    for raw, rolling in (('report_probability', 'rolling_report_probability'),
                         ('report_top1_hit', 'rolling_top1_hit'), ('report_top3_hit', 'rolling_top3_hit')):
        reports[rolling] = reports[raw].where(reports.oral_score_mask).rolling(
            WINDOW_SIZE, min_periods=MIN_CODED_REPORTS).mean().where(reports.oral_score_mask)
    reports['rolling_positive_support'] = reports.report_has_positive_probability.astype(float).where(
        reports.oral_score_mask).rolling(WINDOW_SIZE, min_periods=MIN_CODED_REPORTS).mean().where(reports.oral_score_mask)
    reports['rolling_n_coded_reports'] = reports.oral_score_mask.astype(int).rolling(WINDOW_SIZE, min_periods=1).sum()
    metrics = summarize_report_scores(reports)
    metrics['n_out_of_vocabulary_reports'] = metrics.n_scored_coded_reports - metrics.n_vocabulary_reports
    metrics['n_in_vocabulary_zero_reports'] = metrics.n_vocabulary_reports - metrics.n_positive_probability_reports
    metrics['status'] = 'completed'
    summary = dict(common, status='completed', metrics=_json_safe(metrics.iloc[0].to_dict()),
                   n_pf_repeats=len(seeds), aggregation='equal_mean_of_independent_choice_conditioned_pf_repeats',
                   readout=spec['readout'], source_sha256=report_frame.attrs.get('source_sha256'),
                   seed_sources=sources, encoding_review_status='automatic_encoding_requires_manual_review',
                   score_denominator='all_scored_completely_coded_reports_including_zero_probability',
                   coverage_denominators={'parser_coverage': 'all_behaviorally_scored_trials',
                                          'vocabulary_and_positive_support': 'all_scored_completely_coded_reports'},
                   exact_nll_zero_probability='infinity; positive-only NLL and its report count are separate')
    _write_artifacts(output, reports, metrics, dict(
        oral_report_codes=codes, oral_report_probabilities=probabilities,
        oral_report_valid_mask=valid, seeds=np.asarray(seeds, dtype=int),
        oral_readout_metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)),
        evaluation_metadata_json=np.asarray(json.dumps(summary, ensure_ascii=False, allow_nan=False))))
    return summary
