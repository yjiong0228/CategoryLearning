"""Score a fixed, normalized distribution over complete oral report codes.

The caller obtains report codes with ``structure_0923.encode_report`` and
``report_space.report_code``. Complete coded reports outside the prediction
vocabulary receive zero probability. Missing and unresolved reports have no
score. Prediction distributions and expression parameters remain unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def score_report_predictions(
    codes: Sequence[str],
    probabilities: np.ndarray,
    report_codes: Sequence[str],
    statuses: Sequence[str],
    score_mask: Sequence[bool] | np.ndarray,
    *,
    top_k: Sequence[int] = (1, 3),
    tie_rtol: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Return trialwise scores, coverage masks and exact negative log scores.

    ``probabilities[t, o]`` predicts a report conditional on trial ``t``'s
    actual choice, before feedback. All rows must be finite and normalized.
    Only trials with ``score_mask`` true and status ``coded`` enter report
    score denominators. Their complete codes retain zero probability when
    absent from ``codes``. A zero probability has infinite exact NLL.

    Top-k hits use the expected result of random tie breaking: if ``a`` codes
    rank strictly higher and ``b`` share the observed code's probability, the
    hit is ``clip((k - a) / b, 0, 1)``. Zero-probability codes have zero hit.
    ``tie_rtol`` is a relative numerical tolerance with zero absolute tolerance.
    """
    if isinstance(codes, str) or len(codes) == 0 or any(not isinstance(code, str) or not code for code in codes):
        raise ValueError("codes must be a nonempty sequence of nonempty strings")
    lookup = {code: index for index, code in enumerate(codes)}
    if len(lookup) != len(codes):
        raise ValueError("prediction vocabulary contains duplicate codes")
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != len(codes):
        raise ValueError("probabilities must have shape [trials, report_codes]")
    n_trials = probability.shape[0]
    if (not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=1), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError("prediction rows must be finite, nonnegative and normalized")
    observed = np.asarray(report_codes, dtype=object)
    status = np.asarray(statuses, dtype=object)
    mask = np.asarray(score_mask)
    if observed.shape != (n_trials,) or status.shape != (n_trials,) or mask.shape != (n_trials,):
        raise ValueError("report codes, statuses and score mask must align with trials")
    if mask.dtype.kind != 'b':
        raise ValueError("score_mask must contain booleans")
    if any(not isinstance(code, str) for code in observed):
        raise ValueError("observed report codes must be strings")
    if any(value not in {'coded', 'missing', 'needs_review'} for value in status):
        raise ValueError("statuses must be coded, missing or needs_review")
    coded = status == 'coded'
    if np.any(coded & (observed == '')):
        raise ValueError("a coded report requires a complete report code")
    ks = tuple(top_k)
    if (not ks or len(set(ks)) != len(ks)
            or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1 for k in ks)):
        raise ValueError("top_k must contain distinct positive integers")
    if not np.isfinite(tie_rtol) or tie_rtol < 0:
        raise ValueError("tie_rtol must be finite and nonnegative")

    eligible = mask & coded
    in_vocabulary = coded & np.array([code in lookup for code in observed])
    observed_probability = np.full(n_trials, np.nan)
    observed_probability[eligible] = 0.
    for trial in np.flatnonzero(eligible & in_vocabulary):
        observed_probability[trial] = probability[trial, lookup[observed[trial]]]
    positive = eligible & (observed_probability > 0)
    nll = np.full(n_trials, np.nan)
    nll[eligible & ~positive] = np.inf
    nll[positive] = -np.log(observed_probability[positive])
    finite_nll = np.full(n_trials, np.nan)
    finite_nll[positive] = nll[positive]
    result = {
        'report_coded': coded,
        'oral_score_mask': eligible,
        'report_in_vocabulary': in_vocabulary,
        'report_probability': observed_probability,
        'report_has_positive_probability': positive,
        'report_zero_probability': eligible & ~positive,
        'report_nll': nll,
        'report_finite_nll': finite_nll,
    }
    for k in ks:
        values = np.full(n_trials, np.nan)
        values[eligible] = 0.
        result[f'report_top{k}_hit'] = values
    for trial in np.flatnonzero(positive):
        row = probability[trial]
        mass = observed_probability[trial]
        tied = np.isclose(row, mass, rtol=tie_rtol, atol=0.)
        above = int(np.sum((row > mass) & ~tied))
        ties = int(tied.sum())
        for k in ks:
            result[f'report_top{k}_hit'][trial] = np.clip((min(k, len(codes)) - above) / ties, 0., 1.)
    return result


def summarize_report_scores(
    trials: pd.DataFrame,
    *,
    subject_column: str = 'iSub',
    condition_column: str = 'condition',
    score_column: str = 'scored',
    status_column: str = 'parse_status',
) -> pd.DataFrame:
    """Summarize report predictions per subject with explicit denominators.

    Exact probability and top-k rates use all scored, completely coded reports.
    Vocabulary and positive support coverage use that same denominator.
    ``mean_positive_report_nll`` uses only positive-probability reports; the
    corresponding count is ``n_positive_probability_reports``. Exact mean NLL
    retains infinity whenever an eligible report receives zero probability.
    Parsing coverage uses all scored trials, including missing reports.
    """
    required = {
        subject_column, condition_column, score_column, status_column,
        'oral_score_mask', 'report_coded', 'report_in_vocabulary',
        'report_has_positive_probability', 'report_zero_probability',
        'report_probability', 'report_nll', 'report_finite_nll',
    }
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"missing report score columns: {sorted(missing)}")
    top_columns = [column for column in trials.columns
                   if column.startswith('report_top') and column.endswith('_hit')]
    records: list[dict[str, object]] = []

    def fraction(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else np.nan

    def average(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else np.nan

    for (subject, condition), group in trials.groupby([subject_column, condition_column], sort=True, dropna=False):
        for column in (score_column, 'oral_score_mask', 'report_coded', 'report_in_vocabulary',
                       'report_has_positive_probability', 'report_zero_probability'):
            if group[column].dtype.kind != 'b':
                raise ValueError(f"{column} must contain booleans")
        scored = group[score_column].to_numpy()
        coded = group['report_coded'].to_numpy()
        eligible = group['oral_score_mask'].to_numpy()
        if not np.array_equal(coded, group[status_column].eq('coded').to_numpy()):
            raise ValueError("report_coded must match the complete encoding status")
        if not np.array_equal(eligible, scored & coded):
            raise ValueError("oral_score_mask must equal scored & report_coded")
        positive = group['report_has_positive_probability'].to_numpy()
        if np.any(positive & ~eligible):
            raise ValueError("positive probability mask includes an ineligible report")
        n_scored, n_eligible = int(scored.sum()), int(eligible.sum())
        n_positive = int(positive.sum())
        n_vocab = int((eligible & group['report_in_vocabulary'].to_numpy()).sum())
        n_zero = int((eligible & group['report_zero_probability'].to_numpy()).sum())
        if not np.array_equal(group['report_zero_probability'].to_numpy(), eligible & ~positive):
            raise ValueError("zero probability mask must cover eligible reports without positive mass")
        record = {
            subject_column: subject, condition_column: condition,
            'n_trials': len(group), 'n_scored_trials': n_scored,
            'n_coded_reports': int(coded.sum()), 'n_scored_coded_reports': n_eligible,
            'n_scored_missing_reports': int((scored & group[status_column].eq('missing').to_numpy()).sum()),
            'n_scored_needs_review_reports': int((scored & group[status_column].eq('needs_review').to_numpy()).sum()),
            'n_vocabulary_reports': n_vocab,
            'n_positive_probability_reports': n_positive,
            'n_zero_probability_reports': n_zero,
            'parser_coverage': fraction(n_eligible, n_scored),
            'vocabulary_coverage': fraction(n_vocab, n_eligible),
            'positive_support_coverage': fraction(n_positive, n_eligible),
            'zero_probability_fraction': fraction(n_zero, n_eligible),
            'mean_report_probability': average(group.loc[eligible, 'report_probability'].to_numpy()),
            'mean_report_nll': average(group.loc[eligible, 'report_nll'].to_numpy()),
            'mean_positive_report_nll': average(group.loc[positive, 'report_finite_nll'].to_numpy()),
        }
        for column in top_columns:
            record[column.removeprefix('report_').removesuffix('_hit') + '_hit_rate'] = average(group.loc[eligible, column].to_numpy())
        records.append(record)
    return pd.DataFrame.from_records(records)
