"""Compare generated and observed complete oral reports in their report space.

Read the completed 12-subject Model 0826 oral evaluation. Both trailing-window
distributions use the same scored, completely coded trials. Similarities use
the complete union of model vocabulary and observed report codes; heatmaps
combine small rows only for display. Run from the repository root with a new
output directory. All figures are PNG files.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig2.build_generated_oral_alignment import (
    CONDITION_COLORS, DPI, MIN_PERIODS, WIDTH_MM, WINDOW, read_source, sha256, style,
)
from src.Bayesian_state.metrics.oral_distribution_alignment import (
    distribution_overlap, js_similarity, rolling_report_distributions,
)


ROOT = Path(__file__).resolve().parents[3]
DISPLAY_REPORTS = 24
MISSING_COLOR = '#E5E8EB'
FEATURE_COLUMNS = [f'feature{i}_name' for i in range(1, 5)]
IDENTITY_COLUMNS = ['iSub', 'condition', 'iSession', 'iBlock', 'iTrial', 'trial']
SPECIAL_LABELS = {'EMPTY': 'Empty description', 'OTHER': 'Other modeled report'}


@dataclass
class SubjectDistribution:
    """One participant's complete probabilities and their fixed display rows."""

    subject: int
    condition: int
    trials: pd.DataFrame
    codes: tuple[str, ...]
    vocabulary_size: int
    model: np.ndarray
    observed_indices: np.ndarray
    score_mask: np.ndarray
    rolling: dict[str, np.ndarray]
    row_groups: tuple[np.ndarray, ...]
    row_labels: tuple[str, ...]
    vocabulary_table: pd.DataFrame
    row_table: pd.DataFrame
    mean_model: np.ndarray
    observed_frequency: np.ndarray


def _number(value: float) -> str:
    """Use compact decimal labels without changing a constraint's value."""
    return f'{value:.12g}'


def _linear_term(coefficients: np.ndarray, features: tuple[str, ...]) -> str:
    parts = []
    for coefficient, feature in zip(coefficients, features):
        if coefficient == 0:
            continue
        magnitude = abs(float(coefficient))
        term = feature if magnitude == 1 else f'{_number(magnitude)}*{feature}'
        if not parts:
            parts.append(('-' if coefficient < 0 else '') + term)
        else:
            parts.append((' - ' if coefficient < 0 else ' + ') + term)
    return ''.join(parts) or '0'


def report_label(code: str, features: tuple[str, ...]) -> str:
    """Translate a canonical report using the participant's actual feature order.

    Facets retain A*x <= b. A single .5 threshold has the familiar short/long
    wording. Body-reference tokens stay distinct from absolute thresholds.
    """
    if code in SPECIAL_LABELS:
        return SPECIAL_LABELS[code]
    tokens = json.loads(code)
    if not isinstance(tokens, list) or not tokens or any(not isinstance(token, str) for token in tokens):
        raise ValueError(f'Unsupported complete report code: {code}')
    labels = []
    for token in tokens:
        if token.startswith('body:'):
            _, dimension, direction = token.split(':')
            index = int(dimension)
            if index not in range(len(features)) or direction not in {'gt', 'lt'}:
                raise ValueError(f'Invalid body-reference predicate: {token}')
            labels.append(f'{features[index]} {">" if direction == "gt" else "<"} body ref')
        elif token.startswith('facet:'):
            constraints = np.asarray(json.loads(token[6:]), dtype=float)
            if (constraints.ndim != 2 or constraints.shape[1] != len(features) + 1
                    or len(constraints) == 0 or not np.isfinite(constraints).all()):
                raise ValueError(f'Invalid facet predicate: {token}')
            # Opposing bounds on one feature form one closed interval.
            if (len(constraints) == 2
                    and np.array_equal(constraints[0, :-1], -constraints[1, :-1])
                    and len(np.flatnonzero(constraints[0, :-1])) == 1):
                index = int(np.flatnonzero(constraints[0, :-1])[0])
                upper_row = constraints[0] if constraints[0, index] > 0 else constraints[1]
                lower_row = constraints[1] if constraints[0, index] > 0 else constraints[0]
                upper = float(upper_row[-1] / upper_row[index])
                lower = float(lower_row[-1] / lower_row[index])
                labels.append(f'{_number(lower)} <= {features[index]} <= {_number(upper)}')
                continue
            facet_labels = []
            for constraint in constraints:
                coefficients, bound = constraint[:-1], float(constraint[-1])
                dimensions = np.flatnonzero(coefficients)
                if len(dimensions) == 1:
                    index = int(dimensions[0])
                    coefficient = float(coefficients[index])
                    threshold = bound / coefficient
                    if threshold == .5:
                        facet_labels.append(f'{features[index]} {"short" if coefficient > 0 else "long"}')
                    else:
                        sign = '<=' if coefficient > 0 else '>='
                        facet_labels.append(f'{features[index]} {sign} {_number(threshold)}')
                elif (len(dimensions) == 2 and bound == 0
                      and coefficients[dimensions[0]] == -coefficients[dimensions[1]]):
                    positive = int(dimensions[coefficients[dimensions] > 0][0])
                    negative = int(dimensions[coefficients[dimensions] < 0][0])
                    facet_labels.append(f'{features[positive]} <= {features[negative]}')
                else:
                    facet_labels.append(f'{_linear_term(coefficients, features)} <= {_number(bound)}')
            labels.append(' & '.join(facet_labels))
        else:
            raise ValueError(f'Unsupported report predicate: {token}')
    return ' & '.join(labels)


def _display_label(label: str, maximum: int = 46) -> str:
    return label if len(label) <= maximum else label[:maximum - 3].rstrip() + '...'


def read_lineage(manifest: dict) -> tuple[pd.DataFrame, Path]:
    """Read the frozen trial table used to encode reports and check its hash."""
    candidates = [key for key in manifest['sources'] if key.endswith('/base_analysis/trials.csv')]
    if len(candidates) != 1:
        raise ValueError('The evaluation manifest must identify one original cohort trial table')
    path = ROOT / candidates[0]
    if sha256(path) != manifest['sources'][candidates[0]]:
        raise ValueError('Original cohort trial table differs from the evaluation manifest')
    return pd.read_csv(path, keep_default_na=False), path


def _compare_lineage(group: pd.DataFrame, lineage: pd.DataFrame, subject: int) -> tuple[str, ...]:
    original = lineage.loc[lineage.iSub.eq(subject)].sort_values('trial').reset_index(drop=True)
    columns = [*IDENTITY_COLUMNS, 'choice', 'feedback', *FEATURE_COLUMNS]
    if len(original) != len(group):
        raise ValueError(f'S{subject}: original and evaluated histories differ in length')
    for column in columns:
        if not np.array_equal(group[column].to_numpy(), original[column].to_numpy()):
            raise ValueError(f'S{subject}: {column} differs from the frozen original trial table')
    features = tuple(str(group[column].iloc[0]) for column in FEATURE_COLUMNS)
    if (len(set(features)) != 4 or set(features) != {'neck', 'head', 'leg', 'tail'}
            or any(group[column].nunique(dropna=False) != 1 for column in FEATURE_COLUMNS)):
        raise ValueError(f'S{subject}: feature order must contain four fixed, distinct body features')
    return features


def _display_rows(codes: tuple[str, ...], vocabulary_size: int, mean_model: np.ndarray,
                  observed_frequency: np.ndarray, features: tuple[str, ...],
                  subject: int, condition: int) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Select fixed named rows and preserve every remaining probability mass."""
    weight = .5 * (mean_model + observed_frequency)
    eligible = [index for index in range(len(codes))
                if codes[index] not in SPECIAL_LABELS and weight[index] > 0]
    selected = sorted(eligible, key=lambda index: (-weight[index], codes[index]))[:DISPLAY_REPORTS]
    lookup = {code: index for index, code in enumerate(codes)}
    if any(code not in lookup for code in SPECIAL_LABELS):
        raise ValueError('Model vocabulary requires EMPTY and OTHER outcomes')
    groups = [np.asarray([index], dtype=int) for index in selected]
    complete_labels = [report_label(codes[index], features) for index in selected]
    display_labels = [f'O{position + 1:02d}{"*" if index >= vocabulary_size else ""}  {_display_label(label)}'
                      for position, (index, label) in enumerate(zip(selected, complete_labels))]
    kinds = ['complete_report_oov' if index >= vocabulary_size else 'complete_report' for index in selected]
    outside_flags = [index >= vocabulary_size for index in selected]
    selected_set = set(selected)
    for code, label in SPECIAL_LABELS.items():
        groups.append(np.asarray([lookup[code]], dtype=int))
        complete_labels.append(label)
        display_labels.append(label)
        kinds.append(code)
        outside_flags.append(False)
    rest = np.asarray([index for index in range(vocabulary_size)
                       if index not in selected_set and codes[index] not in SPECIAL_LABELS], dtype=int)
    outside = np.asarray([index for index in range(vocabulary_size, len(codes))
                          if index not in selected_set], dtype=int)
    groups.extend([rest, outside])
    complete_labels.extend(['Rest of model vocabulary', 'Rest of outside model vocabulary'])
    display_labels.extend(['Rest vocabulary', 'Rest outside vocabulary'])
    kinds.extend(['rest_vocabulary', 'outside_vocabulary'])
    outside_flags.extend([False, True])
    membership = np.full(len(codes), -1, dtype=int)
    for row, indices in enumerate(groups):
        if np.any(membership[indices] != -1):
            raise ValueError('Display groups overlap')
        membership[indices] = row
    if np.any(membership < 0):
        raise ValueError('Display groups do not partition the full report space')
    rows = pd.DataFrame({
        'iSub': subject, 'condition': condition, 'display_row': np.arange(len(groups)),
        'row_kind': kinds, 'out_of_vocabulary': outside_flags,
        'display_label': display_labels, 'full_label': complete_labels,
        'n_codes': [len(indices) for indices in groups],
        'model_mean': [mean_model[indices].sum() for indices in groups],
        'observed_frequency': [observed_frequency[indices].sum() for indices in groups],
    })
    vocabulary = pd.DataFrame({
        'iSub': subject, 'condition': condition, 'code_index': np.arange(len(codes)),
        'code': codes, 'label': [report_label(code, features) for code in codes],
        'display_row': membership, 'out_of_vocabulary': np.arange(len(codes)) >= vocabulary_size,
        'model_mean': mean_model, 'observed_frequency': observed_frequency,
        'selection_weight': weight,
    })
    return tuple(groups), tuple(display_labels), vocabulary, rows


def load_subject(source: Path, trials: pd.DataFrame, lineage: pd.DataFrame,
                 subject: int, condition: int, receipt: dict) -> SubjectDistribution:
    """Validate one saved replay and build its full union-space comparison."""
    group = trials.loc[trials.iSub.eq(subject)].sort_values('trial').reset_index(drop=True)
    features = _compare_lineage(group, lineage, subject)
    replay_path = source / 'replays' / f'S{subject}.npz'
    with np.load(replay_path, allow_pickle=False) as replay:
        if (int(replay['subject']) != subject or int(replay['condition']) != condition
                or int(replay['particles']) != receipt['particles']
                or int(replay['seed']) != receipt['seed']):
            raise ValueError(f'S{subject}: replay identity differs from its receipt')
        model_codes = tuple(str(code) for code in replay['oral_report_codes'])
        probability = np.asarray(replay['oral_report_probabilities'], dtype=float)
        valid = np.asarray(replay['oral_report_valid_mask'])
        np.testing.assert_array_equal(group.choice.to_numpy(), replay['observed_choice'])
        np.testing.assert_array_equal(group.feedback.to_numpy(), replay['observed_feedback'])
    if (probability.shape != (len(group), len(model_codes))
            or len(group) != receipt['trials'] or len(model_codes) != receipt['vocabulary_size']
            or len(set(model_codes)) != len(model_codes)):
        raise ValueError(f'S{subject}: replay lengths or vocabulary are invalid')
    if valid.dtype.kind != 'b' or valid.shape != (len(group),) or not valid.all():
        raise ValueError(f'S{subject}: every saved model report prediction must be valid')
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError(f'S{subject}: report probabilities must be finite and nonnegative')
    np.testing.assert_allclose(probability.sum(axis=1), 1., atol=1e-12, rtol=0)
    coded = group.report_coded.to_numpy(dtype=bool)
    if not np.array_equal(coded, group.parse_status.eq('coded')):
        raise ValueError(f'S{subject}: report coding statuses differ from their mask')
    observed_codes = group.report_code.fillna('').astype(str).to_numpy()
    if np.any(coded & (observed_codes == '')):
        raise ValueError(f'S{subject}: complete reports require a report code')
    outside = sorted(set(observed_codes[coded]) - set(model_codes))
    codes = (*model_codes, *outside)
    model = np.zeros((len(group), len(codes)), dtype=float)
    model[:, :len(model_codes)] = probability
    lookup = {code: index for index, code in enumerate(codes)}
    observed_indices = np.asarray([lookup[code] if is_coded else -1
                                   for code, is_coded in zip(observed_codes, coded)], dtype=int)
    score_mask = group.oral_score_mask.to_numpy(dtype=bool)
    expected = np.zeros(int(score_mask.sum()))
    scored_indices = observed_indices[score_mask]
    expected[:] = model[np.flatnonzero(score_mask), scored_indices]
    np.testing.assert_allclose(expected, group.loc[score_mask, 'report_probability'], atol=1e-12, rtol=0,
                               err_msg=f'S{subject}: replay probability differs from prior exact scores')
    if score_mask[0] or not score_mask.any():
        raise ValueError(f'S{subject}: scored reports must exclude the initial trial and be nonempty')
    mean_model = model[score_mask].mean(axis=0)
    observed_frequency = np.bincount(observed_indices[score_mask], minlength=len(codes)) / score_mask.sum()
    rolling = rolling_report_distributions(model, observed_indices, score_mask,
                                          window=WINDOW, min_reports=MIN_PERIODS)
    groups, labels, vocabulary, rows = _display_rows(
        codes, len(model_codes), mean_model, observed_frequency, features, subject, condition)
    return SubjectDistribution(subject, condition, group, tuple(codes), len(model_codes), model,
                               observed_indices, score_mask, rolling, groups, labels,
                               vocabulary, rows, mean_model, observed_frequency)


def display_distribution(matrix: np.ndarray, groups: tuple[np.ndarray, ...]) -> np.ndarray:
    """Sum each disjoint display group, retaining NaN columns as missing data."""
    values = np.asarray(matrix, dtype=float)
    result = np.column_stack([values[:, indices].sum(axis=1) for indices in groups])
    invalid = ~np.isfinite(values).all(axis=1)
    result[invalid] = np.nan
    valid = ~invalid
    if valid.any():
        np.testing.assert_allclose(result[valid].sum(axis=1), 1., atol=1e-10, rtol=0)
    return result


def raw_display(case: SubjectDistribution) -> tuple[np.ndarray, np.ndarray]:
    model = display_distribution(case.model, case.row_groups)
    model[~case.score_mask] = np.nan
    observation = np.full_like(model, np.nan)
    member = np.full(len(case.codes), -1, dtype=int)
    for row, indices in enumerate(case.row_groups):
        member[indices] = row
    valid_trials = np.flatnonzero(case.score_mask)
    observation[valid_trials] = 0.
    observation[valid_trials, member[case.observed_indices[valid_trials]]] = 1.
    return model, observation


def report_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list('oral_probability', ['#FFFFFF', '#487DA8', '#174568'])
    cmap.set_bad(MISSING_COLOR)
    return cmap


def _heatmap(ax: plt.Axes, matrix: np.ndarray, *, labels: tuple[str, ...] | None = None):
    n_trials, n_rows = matrix.shape
    im = ax.imshow(matrix.T, aspect='auto', interpolation='nearest', origin='upper',
                   cmap=report_cmap(), vmin=0, vmax=1,
                   extent=(.5, n_trials + .5, n_rows - .5, -.5), rasterized=True)
    if labels is not None:
        ax.set_yticks(np.arange(n_rows), labels, fontsize=5.3)
    else:
        ax.set_yticks([])
    ax.tick_params(axis='y', length=0, pad=4)
    ax.tick_params(axis='x', length=2)
    ax.spines['left'].set_visible(False)
    for boundary in (n_rows - 4.5, n_rows - 2.5):
        ax.axhline(boundary, color='#D1D5D8', linewidth=.4)
    ax.set_xlim(.5, n_trials + .5)
    return im


def subject_figure(case: SubjectDistribution, output: Path, *, raw: bool = False) -> None:
    """Use identical named report rows and a fixed scale for both heatmaps."""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 180 / 25.4))
    left, width = .325, .575
    model_ax = fig.add_axes([left, .565, width, .265])
    observed_ax = fig.add_axes([left, .225, width, .265])
    if raw:
        model, observed = raw_display(case)
        subtitle = 'Actual trials: model probabilities and one observed complete report per coded trial'
        titles = ['Model: complete-report probability', 'Observed: current report (one-hot)']
    else:
        model = display_distribution(case.rolling['model'], case.row_groups)
        observed = display_distribution(case.rolling['observed'], case.row_groups)
        subtitle = f'Trailing {WINDOW} actual trials; at least {MIN_PERIODS} coded reports; matched trial subsets'
        titles = ['Model: mean complete-report probability', 'Observed: complete-report frequency']
    for ax, matrix, title in zip((model_ax, observed_ax), (model, observed), titles):
        im = _heatmap(ax, matrix, labels=case.row_labels)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=7, pad=6)
    model_ax.tick_params(labelbottom=False)
    observed_ax.set_xticks([1, len(case.trials) // 2, len(case.trials)])
    observed_ax.set_xlabel('Actual trial index', labelpad=3)
    colorbar = fig.colorbar(im, cax=fig.add_axes([.928, .225, .011, .605]), ticks=[0, .5, 1])
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label('Probability / frequency', fontsize=6, labelpad=3)
    curve_ax = fig.add_axes([left, .093, width, .061])
    trial = case.trials.trial.to_numpy()
    curve_ax.plot(trial, case.rolling['overlap'], color=CONDITION_COLORS[case.condition], linewidth=.95)
    curve_ax.set(xlim=(1, len(trial)), ylim=(0, 1), yticks=[0, .5, 1],
                 xticks=[1, len(trial) // 2, len(trial)], ylabel='Overlap', xlabel='Actual trial index')
    curve_ax.grid(axis='y', color='#E7E7E7', linewidth=.5)
    curve_ax.set_title('Full-space overlap: shared report probability mass', loc='left', fontsize=7, pad=5)
    fig.text(.03, .955, f'S{case.subject} · Condition {case.condition} · Complete oral-report distributions',
             fontsize=9, fontweight='bold', color=CONDITION_COLORS[case.condition])
    fig.text(.03, .917, subtitle, fontsize=6.9)
    fig.text(.03, .882, f'{int(case.score_mask.sum())} scored coded reports; {len(case.codes)} codes in the full comparison space',
             fontsize=6.5, color='#47515A')
    footer = ('Gray: initial, unparsed or missing report. Overlap below uses matched trailing windows.' if raw else
              f'Gray: fewer than {MIN_PERIODS} coded reports in the window. Both panels use the same reports.')
    fig.text(.03, .028, footer, fontsize=6.3)
    fig.text(.03, .008, '*: outside model vocabulary. Rest rows combine remaining codes; overlap uses every complete code.',
             fontsize=6.2)
    suffix = 'raw_trials' if raw else 'rolling_distribution'
    fig.savefig(output / f'S{case.subject}_{suffix}.png', dpi=DPI, facecolor='white')
    plt.close(fig)


def cohort_figure(cases: list[SubjectDistribution], output: Path) -> None:
    """Give all participants the same small-panel layout and probability scale."""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 220 / 25.4))
    grid = fig.add_gridspec(3, 4, left=.09, right=.89, bottom=.115, top=.865,
                           hspace=.38, wspace=.22)
    image = None
    for slot, case in zip(grid, cases):
        sub = slot.subgridspec(3, 1, height_ratios=[1., 1., .38], hspace=.23)
        model_ax = fig.add_subplot(sub[0])
        observed_ax = fig.add_subplot(sub[1], sharex=model_ax)
        curve_ax = fig.add_subplot(sub[2], sharex=model_ax)
        model = display_distribution(case.rolling['model'], case.row_groups)
        observed = display_distribution(case.rolling['observed'], case.row_groups)
        image = _heatmap(model_ax, model)
        _heatmap(observed_ax, observed)
        model_ax.set_title(f'S{case.subject} · n={int(case.score_mask.sum())}',
                           loc='left', color=CONDITION_COLORS[case.condition], fontsize=7, fontweight='bold', pad=4)
        model_ax.tick_params(labelbottom=False)
        observed_ax.tick_params(labelbottom=False)
        curve_ax.plot(case.trials.trial, case.rolling['overlap'], linewidth=.75,
                      color=CONDITION_COLORS[case.condition])
        curve_ax.set(ylim=(0, 1), yticks=[0, 1], xticks=[1, len(case.trials)], xlim=(1, len(case.trials)))
        curve_ax.tick_params(axis='both', labelsize=5.5, length=2)
        if slot.colspan.start == 0:
            model_ax.set_ylabel(f'Condition {case.condition}\nModel', fontsize=6)
            observed_ax.set_ylabel('Observed', fontsize=6)
            curve_ax.set_ylabel('Overlap', fontsize=5.5)
        if slot.rowspan.start == 2:
            curve_ax.set_xlabel('Actual trial', fontsize=6)
    colorbar = fig.colorbar(image, cax=fig.add_axes([.921, .27, .012, .38]), ticks=[0, .5, 1])
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label('Probability / frequency', fontsize=6)
    fig.text(.09, .957, 'Predicted and observed oral distributions across learning', fontsize=9, fontweight='bold')
    fig.text(.09, .925, f'Trailing {WINDOW} actual trials · at least {MIN_PERIODS} coded reports · matched report subsets', fontsize=7)
    fig.text(.09, .895, 'Each participant: model heatmap, observed-frequency heatmap, full-space overlap curve', fontsize=6.5)
    fig.legend(handles=[Patch(facecolor=MISSING_COLOR, label=f'Fewer than {MIN_PERIODS} coded reports'),
                        Line2D([], [], color='#487DA8', linewidth=1, label='Overlap = 1 - total variation')],
               loc='lower left', bbox_to_anchor=(.09, .06), ncol=2, fontsize=6.5, columnspacing=1.4)
    fig.text(.09, .032, 'Rows are fixed within each participant. *: outside model vocabulary; individual figures show their content.', fontsize=6.4)
    fig.text(.09, .013, 'All overlaps use complete codes, including outside-vocabulary reports. n = scored, completely coded reports.', fontsize=6.3)
    fig.savefig(output / 'oral_distribution_overview.png', dpi=DPI, facecolor='white')
    plt.close(fig)


def save_case(case: SubjectDistribution, output: Path) -> tuple[dict, pd.DataFrame]:
    """Export full-space sources, display mappings and both similarity measures."""
    directory = output / 'source_data'
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / f'S{case.subject}.npz',
                        codes=np.asarray(case.codes), trial=case.trials.trial.to_numpy(),
                        observed_indices=case.observed_indices, oral_score_mask=case.score_mask,
                        rolling_model=case.rolling['model'], rolling_observed=case.rolling['observed'],
                        rolling_n_reports=case.rolling['n_reports'], rolling_valid=case.rolling['valid'],
                        rolling_overlap=case.rolling['overlap'], rolling_js_similarity=case.rolling['js_similarity'],
                        model_mean=case.mean_model, observed_frequency=case.observed_frequency)
    case.vocabulary_table.to_csv(directory / f'S{case.subject}_vocabulary.csv', index=False)
    summary = {
        'iSub': case.subject, 'condition': case.condition, 'n_trials': len(case.trials),
        'n_scored_coded_reports': int(case.score_mask.sum()), 'vocabulary_size': case.vocabulary_size,
        'union_space_size': len(case.codes), 'n_observed_outside_codes': len(case.codes) - case.vocabulary_size,
        'outside_vocabulary_frequency': float(case.observed_frequency[case.vocabulary_size:].sum()),
        'full_sequence_overlap': float(distribution_overlap(case.mean_model, case.observed_frequency)),
        'full_sequence_js_similarity': float(js_similarity(case.mean_model, case.observed_frequency)),
        'mean_rolling_overlap': float(np.nanmean(case.rolling['overlap'])),
        'mean_rolling_js_similarity': float(np.nanmean(case.rolling['js_similarity'])),
        'n_valid_rolling_windows': int(case.rolling['valid'].sum()),
    }
    trajectory = case.trials[[*IDENTITY_COLUMNS, 'scored', 'parse_status', 'oral_score_mask',
                              'report_in_vocabulary']].copy()
    trajectory['rolling_n_reports'] = case.rolling['n_reports']
    trajectory['rolling_valid'] = case.rolling['valid']
    trajectory['rolling_overlap'] = case.rolling['overlap']
    trajectory['rolling_js_similarity'] = case.rolling['js_similarity']
    trajectory['window_start_trial'] = np.maximum(1, trajectory.trial - WINDOW + 1)
    return summary, trajectory


def write_notes(output: Path, manifest: dict) -> None:
    """Explain the comparison space, display compression and data boundaries."""
    summary = pd.read_csv(output / 'summary_source.csv').sort_values(['condition', 'iSub'])
    navigation = [
        '## Participant files', '',
        'Full-sequence overlap compares the two pooled distributions across all scored, completely coded reports for each participant. It is the full-sequence summary metric.', '',
        '| Participant | Condition | Full-sequence overlap | Window distributions | Actual trials | Complete vocabulary |',
        '|---|---:|---:|---|---|---|',
    ]
    for row in summary.itertuples(index=False):
        subject = int(row.iSub)
        navigation.append(
            f'| S{subject} | {int(row.condition)} | {row.full_sequence_overlap:.3f} | '
            f'[PNG](individual/S{subject}_rolling_distribution.png) | '
            f'[PNG](individual/S{subject}_raw_trials.png) | '
            f'[CSV](source_data/S{subject}_vocabulary.csv) |')
    navigation_text = '\n'.join(navigation) + '\n'
    (output / 'README.md').write_text(f'''# Complete oral-report distribution comparison

The horizontal axis retains actual trials. Rows represent complete oral descriptions, such as “neck short & tail long”. The model and observed heatmaps share the same rows within each participant. Their vocabulary tables translate the participant's actual feature order.

The main figures use trailing windows of {WINDOW} actual trials, requiring {MIN_PERIODS} scored, completely coded reports. Each window averages model report probabilities over those reports and divides their observed code counts by the same report count. A report supplies one observation on its own trial; the window gives an empirical distribution. The first trial, missing reports and reports requiring review do not enter either side's window average. A window remains defined at a missing current report when its earlier trials provide enough coded reports.

The full comparison space is the union of the frozen model vocabulary and every complete observed code. Observed codes outside the model vocabulary receive zero model probability. EMPTY and OTHER remain model outcomes. Missing and unparsed reports are recorded separately and are excluded from these coded-report comparisons. The displayed empirical distribution describes successfully coded reports.

Each participant has up to {DISPLAY_REPORTS} named complete-code rows, selected from the complete union space by one half of the sum of mean model probability and observed frequency over all scored coded reports. This selection is fixed for both heatmaps and every trial. An asterisk after a row number marks a complete report outside the model vocabulary. Four further rows show EMPTY, OTHER, all remaining vocabulary mass, and all remaining outside-vocabulary mass. Summing rows preserves total probability. Vocabulary CSVs contain the complete code, feature-aware label, display-row mapping and selection weight. Abbreviated figure labels retain their full text in those tables.

Overlap is the shared mass sum_o min(P_o,Q_o), equal to one minus total variation. Its range is 0–1; 1 means the two distributions are identical. The source tables also give Jensen–Shannon similarity as 1 - JSD, using base-2 logarithms. Both measures are calculated over every complete code before rows are combined for display. `full_sequence_*` compares distributions pooled across the complete scored sequence; `mean_rolling_*` averages the defined, overlapping trailing windows.

Window similarity compares the frequencies of complete descriptions within a window. Trialwise report scores evaluate the prediction assigned to the actual report at its own time point.

`oral_distribution_overview.png` includes all 12 participants. `individual/S*_rolling_distribution.png` gives each participant's two window distributions and full-space overlap. `individual/S*_raw_trials.png` shows model probabilities and observed one-hot reports on their actual trials, using the same display rows; excluded columns are gray. Its lower curve gives the matched-window overlap. `source_data/S*.npz` stores complete, uncompressed rolling distributions and codes. The original replay contains the raw model probabilities.

The saved behavioral parameters, seeds and 128-particle replays are reused. Oral readout is conditioned on the observed choice before current feedback. Oral reports are used for prediction evaluation. The cohort contributed to model and report-grammar development; automatic coding remains subject to review. The figures describe one seed per participant. Expression settings: {json.dumps(manifest['oral_parameters'], ensure_ascii=False)}.

{navigation_text}
''')
    (output / 'legend.md').write_text(f'''# Figure legend

**Complete oral-report distributions in 12 participants.** Each participant's upper heatmap shows the mean model probability of each complete oral description; the lower heatmap shows its observed frequency over the same coded trials. Windows retain the most recent {WINDOW} actual trials including the current trial and require {MIN_PERIODS} scored coded reports. The first trial, missing reports and unresolved reports are excluded from both averages. Both heatmaps share a 0–1 scale and fixed rows within a participant. Gray indicates insufficient coded reports.

Up to {DISPLAY_REPORTS} complete-code rows are selected from the complete union space by the average of full-sequence predicted probability and observed frequency. An asterisk marks a selected report outside the model vocabulary. EMPTY, OTHER, remaining vocabulary mass and remaining outside-vocabulary mass have separate rows. Full-code tables accompany each figure. The curve below gives overlap = sum_o min(P_o,Q_o) = 1 - total variation, calculated in the complete union space before display grouping. Larger values indicate more similar report distributions. The full-space Jensen–Shannon similarity is provided in the source tables.

Raw-trial companions show current model probabilities and a one-hot observed report; the first trial and missing or unresolved reports remain gray in both panels. Oral predictions pool particle joint choice–report mass before conditioning on the observed choice. Reports enter evaluation after prediction. The cohort is a development sample, report coding is automatic, and each participant contributes one fixed-seed 128-particle replay.
''')
    (output / 'figure_contract.md').write_text(f'''# Figure contract

Question: Do the generated complete oral-report distributions resemble the observed report distributions, and how does that similarity change during learning?

Comparison: full union of frozen model vocabulary and all complete observed codes; both sides use identical scored coded trials. Primary similarity: overlap = 1 - total variation. Secondary source metric: 1 - Jensen–Shannon divergence using base-2 logarithms. The model predicts reports after choice and before feedback. The empirical distribution summarizes coded reports in the same windows.

Backend: Python/matplotlib. Individual figures: {WIDTH_MM} × 180 mm; overview: {WIDTH_MM} × 220 mm; PNG only, {DPI} dpi. Common white–blue 0–1 probability/frequency scale. Gray denotes absent comparisons. Original trial order, including omitted reports, is retained.

Display: fixed participant-level selection of up to {DISPLAY_REPORTS} complete codes from the model-and-observed union space plus EMPTY, OTHER, remaining vocabulary mass and remaining outside-vocabulary mass. An asterisk marks a named report outside the model vocabulary. Complete-code metrics are calculated before display grouping. Full source arrays and row mappings accompany all figures.

Evidence: saved choice-only fits with shared fixed oral expression settings; one validated 128-particle replay per participant; development cohort and automatically encoded reports. Window summaries condition on report coding availability and are descriptive. Adjacent trailing windows share trials.
''')
    (output / 'QA.md').write_text('''# Figure QA

Data checks performed by the builder: cohort has four subjects per condition; replay receipts agree with the source summary; every sequential trial index is retained; subject, condition, seed, particles, vocabulary size, choices and feedback agree with their saved replay. Original source hash and trial identifiers, choices, feedback and all four feature names agree with the frozen lineage. Every model report distribution is finite, nonnegative and normalized. Replay probabilities for observed complete reports reproduce the existing exact scores.

The metric implementation validates matched trial masks, complete union-space probability conservation, finite normalized empirical distributions and bounded full-space overlap and Jensen–Shannon similarity. Display rows partition every code once; aggregated rows retain normalization. All observed outside-vocabulary codes retain zero model probability. Raw missing and unparsed columns remain gray.

Rendering uses native matplotlib artists, nearest-neighbor heatmaps, a shared 0–1 scale, original trial indices, and PNG-only exports at 600 dpi. Individual displays show row labels, a shared colorbar, and a full-space overlap curve. The overview includes all participants at equal panel sizes.

Source preflight: the shared plot module supplies the 600-dpi and 183-mm constants. Publication-safe fonts and editable-text settings are explicit in this source. The repository export contract is PNG only; the generic validator's vector and TIFF requirements are recorded export-policy exceptions.

Visual inspection is recorded after generation in final_qa.json.
''')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Completed saved-fit oral evaluation directory')
    parser.add_argument('--output', type=Path, required=True, help='New figure directory; existing paths are refused')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trials, subjects, receipts, manifest = read_source(args.source)
    lineage, lineage_path = read_lineage(manifest)
    receipt_lookup = {receipt['subject']: receipt for receipt in receipts}
    cases = [load_subject(args.source, trials, lineage, int(row.iSub), int(row.condition), receipt_lookup[row.iSub])
             for row in subjects.itertuples(index=False)]
    args.output.mkdir(parents=True)
    individual = args.output / 'individual'
    individual.mkdir()
    style()
    # Explicit font declarations also make the standalone source preflight auditable.
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    summary_records, trajectories = [], []
    for case in cases:
        summary, trajectory = save_case(case, args.output)
        summary_records.append(summary)
        trajectories.append(trajectory)
        subject_figure(case, individual)
        subject_figure(case, individual, raw=True)
    cohort_figure(cases, args.output)
    pd.DataFrame(summary_records).to_csv(args.output / 'summary_source.csv', index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(args.output / 'trajectory_source.csv', index=False)
    pd.concat([case.row_table for case in cases], ignore_index=True).to_csv(args.output / 'row_source.csv', index=False)
    write_notes(args.output, manifest)
    inputs = [args.source / name for name in ('trials.csv', 'subjects.csv', 'manifest.json', 'replay_validation.json')]
    inputs.extend(args.source / 'replays' / f'S{case.subject}.npz' for case in cases)
    inputs.append(lineage_path)
    code_paths = [Path(__file__), Path(__file__).with_name('build_generated_oral_alignment.py'),
                  ROOT / 'src/Bayesian_state/metrics/oral_distribution_alignment.py']
    source_hashes = {str(path.resolve().relative_to(ROOT)): sha256(path) for path in [*inputs, *code_paths]}
    output_hashes = {str(path.relative_to(args.output)): sha256(path)
                     for path in args.output.rglob('*') if path.is_file()}
    provenance = {
        'backend': 'python', 'dpi': DPI, 'window': WINDOW, 'minimum_coded_reports': MIN_PERIODS,
        'display_complete_report_rows': DISPLAY_REPORTS, 'subjects': [case.subject for case in cases],
        'comparison_space': 'model_vocabulary_union_all_complete_observed_codes',
        'trial_subset': 'identical_scored_and_completely_coded_trials_for_both_distributions',
        'metric_space': 'full_uncompressed_union_space', 'replays': receipts,
        'source_hashes': source_hashes, 'output_hashes': output_hashes,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'subjects': len(cases),
                      'individual_pngs': 2 * len(cases), 'window': WINDOW,
                      'minimum_coded_reports': MIN_PERIODS}), flush=True)


if __name__ == '__main__':
    main()
