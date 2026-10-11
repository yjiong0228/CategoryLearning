"""Plot complete-report predictions from fixed Model 0826 behavior replays.

Run from the repository root with a completed --source and a new --output
directory. This entry point reads the evaluated report tables and writes PNG
figures, source tables, legends, a figure contract and provenance hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CONDITION_COLORS = {1: '#3B6FB6', 2: '#C9823B', 3: '#2A918F'}
COVERAGE = (
    ('n_positive_probability_reports', 'Positive probability', '#567A98'),
    ('n_in_vocabulary_zero_reports', 'In vocabulary, zero probability', '#B4C6D6'),
    ('n_out_of_vocabulary_reports', 'Outside vocabulary', '#D49A5C'),
    ('n_scored_needs_review_reports', 'Unparsed', '#BCB8B3'),
    ('n_scored_missing_reports', 'Missing', '#ECE8E4'),
)
WINDOW = 64
MIN_PERIODS = 16
DPI = 600
WIDTH_MM = 183


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool_column(frame: pd.DataFrame, name: str) -> None:
    values = frame[name]
    if values.dtype.kind == 'b':
        return
    normalized = values.astype(str).str.lower()
    if not normalized.isin(['true', 'false', '1', '0']).all():
        raise ValueError(f'{name} contains values outside the boolean schema')
    frame[name] = normalized.isin(['true', '1'])


def read_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    """Validate cohort, counting partitions and behavioral replay agreement."""
    trials = pd.read_csv(source / 'trials.csv')
    subjects = pd.read_csv(source / 'subjects.csv').sort_values(['condition', 'iSub']).reset_index(drop=True)
    receipts = json.loads((source / 'replay_validation.json').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    for name in ('scored', 'oral_score_mask', 'report_coded', 'report_in_vocabulary',
                 'report_has_positive_probability', 'report_zero_probability'):
        _bool_column(trials, name)
    if (len(subjects) != 12 or subjects.iSub.duplicated().any()
            or subjects.groupby('condition').size().to_dict() != {1: 4, 2: 4, 3: 4}):
        raise ValueError('This figure requires four subjects from each of the three conditions')
    if set(trials.iSub.unique()) != set(subjects.iSub):
        raise ValueError('Subject summary and trial table differ')
    if (len(receipts) != 12 or len({r['subject'] for r in receipts}) != 12
            or {r['subject'] for r in receipts} != set(subjects.iSub)):
        raise ValueError('Replay receipts and subject table differ')
    if any(r['particles'] != 128 or r['choice_max_absolute_difference'] > 1e-12 for r in receipts):
        raise ValueError('Figure requires validated 128-particle behavior replays')
    if manifest['repeats_per_subject'] != 1 or manifest['oral_use'] != 'external_prediction_scoring_only':
        raise ValueError('Unexpected oral prediction protocol')
    if manifest['oral_parameters']['extra_selection'] != 'salience':
        raise ValueError('The current section 8 prediction uses salience-based supplementary descriptions')
    subjects['n_in_vocabulary_zero_reports'] = (
        subjects.n_vocabulary_reports - subjects.n_positive_probability_reports)
    subjects['n_out_of_vocabulary_reports'] = (
        subjects.n_scored_coded_reports - subjects.n_vocabulary_reports)
    if np.any(subjects[[name for name, _, _ in COVERAGE]].to_numpy() < 0):
        raise ValueError('Coverage partition has negative counts')
    if not np.array_equal(subjects[[name for name, _, _ in COVERAGE]].sum(axis=1), subjects.n_scored_trials):
        raise ValueError('Coverage partition must include every scored trial')
    for row in subjects.itertuples(index=False):
        group = trials.loc[trials.iSub.eq(row.iSub)].sort_values('trial')
        if not np.array_equal(group.trial, np.arange(1, len(group) + 1)):
            raise ValueError(f'S{row.iSub}: trials must retain their complete ordered index')
        if (len(group) != row.n_trials or int(group.scored.sum()) != row.n_scored_trials
                or int(group.oral_score_mask.sum()) != row.n_scored_coded_reports):
            raise ValueError(f'S{row.iSub}: denominator mismatch')
        eligible = group.oral_score_mask
        if not np.array_equal(eligible, group.scored & group.report_coded):
            raise ValueError(f'S{row.iSub}: oral scoring mask differs from complete coded trials')
        for column, summary in [('report_top1_hit', row.top1_hit_rate),
                                ('report_top3_hit', row.top3_hit_rate),
                                ('report_probability', row.mean_report_probability)]:
            values = group.loc[eligible, column]
            if values.isna().any() or not values.between(0, 1).all():
                raise ValueError(f'S{row.iSub}: invalid {column}')
            np.testing.assert_allclose(values.mean(), summary, rtol=0, atol=1e-12)
    return trials, subjects, receipts, manifest


def style() -> None:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })


def percent_limit(values: np.ndarray, minimum: float) -> float:
    maximum = float(np.max(values)) if len(values) else 0.
    step = 5. if maximum >= 10. else 1.
    return min(100., max(minimum, np.ceil(maximum * 1.25 / step) * step))


def summary_figure(subjects: pd.DataFrame, output: Path) -> None:
    """Use equal subject rows and distinct denominators for scores and coverage."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM / 25.4, 148 / 25.4), sharey=True,
                             gridspec_kw={'width_ratios': [1., .85, 1.5]})
    fig.subplots_adjust(left=.205, right=.975, bottom=.205, top=.835, wspace=.3)
    y = np.arange(len(subjects))
    colors = [CONDITION_COLORS[c] for c in subjects.condition]
    top1, top3 = subjects.top1_hit_rate.to_numpy() * 100, subjects.top3_hit_rate.to_numpy() * 100
    probability = subjects.mean_report_probability.to_numpy() * 100
    for row, color in zip(y, colors):
        axes[0].plot([top1[row], top3[row]], [row, row], color='#AEB5BC', lw=.8, zorder=1)
        axes[0].scatter(top1[row], row, color=color, s=17, zorder=3, clip_on=False)
        axes[0].scatter(top3[row], row, facecolors='white', edgecolors=color, s=23,
                        linewidths=1, zorder=2, clip_on=False)
        axes[1].scatter(probability[row], row, color=color, marker='D', s=15, zorder=3, clip_on=False)
        axes[1].annotate(f'{probability[row]:.2f}', (probability[row], row),
                         xytext=(4, 0), textcoords='offset points', va='center', fontsize=6)
    left = np.zeros(len(subjects))
    for name, label, color in COVERAGE:
        values = subjects[name].to_numpy() / subjects.n_scored_trials.to_numpy() * 100
        axes[2].barh(y, values, left=left, height=.58, color=color, edgecolor='white', linewidth=.3, label=label)
        left += values
    axes[0].set_xlim(0, percent_limit(top3, 5.))
    axes[1].set_xlim(0, percent_limit(probability, 1.))
    axes[2].set_xlim(0, 100)
    axes[0].set_yticks(y, [f'S{r.iSub} (n={r.n_scored_coded_reports})' for r in subjects.itertuples()])
    axes[0].invert_yaxis()
    for row, color in zip(axes[0].get_yticklabels(), colors):
        row.set_color(color)
    for ax, title, xlabel in zip(axes,
                                ['a  Exact report hit rate', 'b  Report probability', 'c  Report coverage'],
                                ['Coded reports (%)', 'Mean probability (%)', 'All scored trials (%)']):
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
        ax.set_xlabel(xlabel)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E7E7E7', linewidth=.5)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        for boundary in (3.5, 7.5):
            ax.axhline(boundary, color='#D1D4D6', lw=.6)
    axes[2].set_xticks([0, 50, 100])
    fig.text(.205, .953, 'Complete oral reports predicted from saved behavioral fits',
             fontsize=9, fontweight='bold')
    fig.text(.205, .913, 'One replay per subject · 128 particles · shared fixed expression parameters', fontsize=7)
    point_handles = [Line2D([], [], marker='o', linestyle='', color='#59636D', label='Top 1', markersize=4),
                     Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
                            markeredgecolor='#59636D', label='Top 3', markersize=4)]
    condition_handles = [Patch(facecolor=color, label=f'Condition {condition}')
                         for condition, color in CONDITION_COLORS.items()]
    fig.legend(handles=point_handles + condition_handles, loc='lower left',
               bbox_to_anchor=(.205, .105), ncol=5, columnspacing=1.4, handlelength=1.2)
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in COVERAGE],
               loc='lower left', bbox_to_anchor=(.205, .047), ncol=3, columnspacing=1.3, handlelength=1.2)
    fig.text(.205, .025, 'n = scored, completely coded reports; coverage includes every scored trial.', fontsize=6.5)
    fig.savefig(output / 'oral_alignment_summary.png', dpi=DPI, facecolor='white')
    plt.close(fig)


def rolling_source(trials: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing windows on actual trials; unresolved trials break curves."""
    frames = []
    for _, group in trials.groupby(['condition', 'iSub'], sort=True):
        group = group.sort_values('trial').copy()
        valid = group.oral_score_mask
        hit = group.report_top3_hit.where(valid)
        support = group.report_has_positive_probability.astype(float).where(valid)
        group['rolling_top3_hit'] = hit.rolling(WINDOW, min_periods=MIN_PERIODS).mean().where(valid)
        group['rolling_positive_support'] = support.rolling(WINDOW, min_periods=MIN_PERIODS).mean().where(valid)
        group['rolling_n_coded_reports'] = valid.astype(int).rolling(WINDOW, min_periods=1).sum()
        frames.append(group[['iSub', 'condition', 'trial', 'oral_score_mask', 'parse_status',
                             'rolling_top3_hit', 'rolling_positive_support', 'rolling_n_coded_reports']])
    return pd.concat(frames, ignore_index=True)


def trajectory_figure(rolling: pd.DataFrame, subjects: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH_MM / 25.4, 145 / 25.4), sharey=True)
    fig.subplots_adjust(left=.095, right=.975, bottom=.165, top=.84, hspace=.68, wspace=.24)
    for ax, row in zip(axes.flat, subjects.itertuples()):
        group = rolling.loc[rolling.iSub.eq(row.iSub)]
        color = CONDITION_COLORS[row.condition]
        ax.plot(group.trial, 100 * group.rolling_top3_hit, color=color, lw=1,
                label='Top 3 hit rate', zorder=4, clip_on=False)
        ax.plot(group.trial, 100 * group.rolling_positive_support, color='#6F7880',
                lw=.9, linestyle=(0, (3, 2)), label='Positive report support', zorder=3, clip_on=False)
        ax.set_title(f'S{row.iSub} · n={row.n_scored_coded_reports}', color=color, fontweight='bold', loc='left')
        ax.set_xlim(1, row.n_trials)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([1, row.n_trials])
        ax.spines['bottom'].set_position(('outward', 2))
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E7E7E7', lw=.5)
    for row_axes, condition in zip(axes, (1, 2, 3)):
        row_axes[0].set_ylabel(f'Condition {condition}\nCoded reports (%)')
    for ax in axes[-1]:
        ax.set_xlabel('Actual trial index')
    fig.text(.095, .956, 'Oral prediction across learning', fontsize=9, fontweight='bold')
    fig.text(.095, .913, 'Trailing 64 trials · at least 16 coded reports · unresolved and missing trials leave gaps', fontsize=7)
    handles = [Line2D([], [], color='#59636D', lw=1, label='Top 3 exact report hit rate'),
               Line2D([], [], color='#6F7880', lw=.9, linestyle=(0, (3, 2)), label='Observed report has positive probability')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.095, .065), ncol=2, columnspacing=1.8)
    fig.text(.095, .031, 'n = scored, completely coded reports. Each curve uses one fixed-parameter, 128-particle replay.', fontsize=6.5)
    fig.savefig(output / 'oral_alignment_trajectories.png', dpi=DPI, facecolor='white')
    plt.close(fig)


def write_notes(output: Path, manifest: dict, subjects: pd.DataFrame) -> None:
    total = int(subjects.n_scored_coded_reports.sum())
    scored = int(subjects.n_scored_trials.sum())
    legend = f'''# Figure legend

**Complete oral report prediction in 12 participants.** Each participant is represented by one replay of the saved selected behavioral parameters and seed, with 128 particles. The replay reproduces the saved choice probabilities. Oral predictions combine the joint choice–report mass across particles before conditioning on the observed choice. Reports are evaluated after this prediction, with shared fixed expression settings.

**Summary figure.** a, Top-1 and top-3 complete-report hit rates. Numerical ties receive the expected hit from random tie breaking. b, Mean probability assigned to each observed complete report. Both panels include all scored, completely coded reports, including codes outside the prediction vocabulary ({total} reports in total). c, Coverage of all {scored} scored trials: positive-probability reports, vocabulary entries with zero probability, complete reports outside the vocabulary, reports requiring parsing review, and missing reports. Subject labels give the coded-report denominator.

**Trajectory figure.** Solid lines show the top-3 complete-report hit rate; dashed gray lines show the proportion of coded reports assigned positive probability. Values are trailing means over 64 actual trials, requiring at least 16 scored coded reports in the window. Missing and unresolved current reports leave gaps. The abscissa retains the original trial index.

The displayed cohort contributed to model and report-grammar development. Reports use automatic structural coding and remain subject to human review. The figures describe a single seed per participant; interval estimates and hypothesis tests are absent. Fixed expression parameters: {json.dumps(manifest['oral_parameters'], ensure_ascii=False)}.
'''
    (output / 'legend.md').write_text(legend)
    (output / 'figure_contract.md').write_text('''# Figure contract

Core question: How much probability does the section 8 observation model assign to complete oral reports from the 12 saved behavioral fits?

Archetype: quantitative grid. The summary compares exact prediction scores and explains their coverage. The trajectory grid shows changes during each participant's full learning sequence.

Backend: Python/matplotlib. Summary: 183 × 148 mm; trajectories: 183 × 145 mm. PNG only, 600 dpi. Participants use the same plotting scales within each panel family, condition colors, and metric definitions.

Score denominator: all scored, completely coded reports. Out-of-vocabulary reports retain zero probability and zero top-k hit. Coverage denominator: every scored trial. Missing and unresolved reports remain visible in coverage and unscored in exact metrics. Full histories remain in trajectory tables; trailing windows ignore missing scores and leave gaps at unresolved current trials.

Evidence boundary: one fixed-parameter, 128-particle replay per participant; automatic report coding; development cohort. This comparison evaluates the oral observation layer attached to saved 0826 states.
''')
    (output / 'QA.md').write_text('''# Figure QA

Data checks: 12 distinct subjects, four per condition; replay receipts match the cohort; 128 particles per replay; saved choice probabilities agree to an absolute tolerance of 1e-12; source and summary denominators match; complete coded scores lie in [0, 1]; five coverage counts partition all scored trials; complete trial order is retained.

Rendering: Python/matplotlib, PNG only, 600 dpi. Each subject has equal space in both figures. Condition is encoded by color and row grouping; top-k and coverage categories also have labels or marker styles. Quantitative source tables and hashes accompany the figures.

Statistical reporting: exact metrics use all scored coded reports; coverage uses all scored trials; rolling windows require 16 coded reports in 64 actual trials. Each participant contributes one seed. Figure legends state the automatic coding and development-sample roles.

Visual inspection: pending inspection of the actual PNGs at final size.

Export choice: PNG follows the repository's requested export policy. The skill's generic source validator requires SVG/PDF and also flags absent TIFF; those format findings are documented exceptions under this PNG-only contract. Editable vector exports are outside this output contract.
''')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trials, subjects, receipts, manifest = read_source(args.source)
    rolling = rolling_source(trials)
    args.output.mkdir(parents=True)
    style()
    summary_figure(subjects, args.output)
    trajectory_figure(rolling, subjects, args.output)
    subjects.to_csv(args.output / 'summary_source.csv', index=False)
    rolling.to_csv(args.output / 'trajectory_source.csv', index=False)
    write_notes(args.output, manifest, subjects)
    source_files = [args.source / name for name in ('trials.csv', 'subjects.csv', 'manifest.json', 'replay_validation.json')]
    source_files.append(Path(__file__))
    provenance = {
        'backend': 'python', 'dpi': DPI, 'rolling_window': WINDOW,
        'minimum_coded_reports': MIN_PERIODS, 'subjects': subjects.iSub.tolist(),
        'replays': receipts, 'source_hashes': {path.name: sha256(path) for path in source_files},
        'figure_hashes': {path.name: sha256(path) for path in args.output.glob('*.png')},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'subjects': len(subjects), 'scored_coded_reports':
                      int(subjects.n_scored_coded_reports.sum())}), flush=True)


if __name__ == '__main__':
    main()
