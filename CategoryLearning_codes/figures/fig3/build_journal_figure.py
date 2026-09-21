"""Journal-style learning-dynamics figure using the existing 12-person analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .bottleneck_analysis import ROOT, sha256
from .build_abstract_story import read_story
from .build_nine_subject_figures import COLORS, TASK_COLORS

MARKERS = {1: 'o', 2: '^', 3: 's'}


def journal_style() -> None:
    """Set physical-size typography shared by the two journal revisions."""
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 6.7, 'axes.labelsize': 6.7, 'axes.titlesize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .55, 'xtick.major.width': .55, 'ytick.major.width': .55,
        'xtick.major.size': 2.3, 'ytick.major.size': 2.3,
        'legend.frameon': False, 'legend.fontsize': 6,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })


def panel_label(fig, letter: str, x: float, y: float) -> None:
    fig.text(x, y, letter, fontsize=8, fontweight='bold', ha='left', va='top')


def support_episodes(trials: pd.DataFrame, threshold: float = .5,
                     minimum: int = 16) -> pd.DataFrame:
    """Return within-session episodes of sustained model support, without bridging gaps."""
    records = []
    for (subject, session), group in trials.groupby(['iSub', 'iSession'], sort=False):
        group = group.sort_values('trial')
        start = previous = None
        for trial, support in zip(group.trial.astype(int), group.belief):
            above = np.isfinite(support) and support > threshold
            contiguous = previous is not None and trial == previous + 1
            if start is not None and (not above or not contiguous):
                if previous - start + 1 >= minimum:
                    records.append((subject, session, start, previous, previous-start+1))
                start = None
            if above and start is None:
                start = trial
            previous = trial
        if start is not None and previous - start + 1 >= minimum:
            records.append((subject, session, start, previous, previous-start+1))
    return pd.DataFrame(records, columns=['subject', 'session', 'start', 'stop', 'length'])


def save_bundle(fig, source: Path, output: Path, filename: str, code: Path,
                legend: str, tables: dict[str, pd.DataFrame] | None = None) -> None:
    """Export a PNG and hash-traceable figure-specific source tables, never overwrite."""
    fig.savefig(output / filename, dpi=450, facecolor='white')
    dimensions = [float(value * 25.4) for value in fig.get_size_inches()]
    plt.close(fig)
    (output / 'source_data').mkdir()
    for filename_source in ['subjects.csv', 'trials.csv', 'jump_windows.csv', 'manifest.json']:
        shutil.copy2(source / filename_source, output / 'source_data' / filename_source)
    for name, table in (tables or {}).items():
        table.to_csv(output / 'source_data' / name, index=False)
    (output / 'code').mkdir()
    sources = [code.resolve(), Path(__file__).resolve(), code.with_name('JOURNAL_DESIGN.md').resolve(),
               Path(__file__).with_name('build_abstract_story.py').resolve(),
               Path(__file__).with_name('build_nine_subject_figures.py').resolve(),
               Path(__file__).with_name('bottleneck_analysis.py').resolve()]
    for item in dict.fromkeys(sources):
        shutil.copy2(item, output / 'code' / (item.parent.name + '_' + item.name))
    (output / 'legend.md').write_text(legend)
    receipt = {
        'figure': filename, 'source': str(source.resolve().relative_to(ROOT)),
        'dimensions_mm': dimensions, 'dpi': 450, 'format': 'PNG',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
        'source_sha256': {item.name: sha256(item) for item in (output/'source_data').iterdir()},
        'code_sha256': {str(item.relative_to(ROOT)): sha256(item) for item in dict.fromkeys(sources)},
        'excluded_participants': [], 'new_model_fits': 0, 'new_particle_replays': 0,
    }
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')


LEGEND = """# Fig. 3 | Performance and reconstructed belief follow distinct learning trajectories.

**a–c,** Behavior-selected examples: earliest behavioral criterion (S122), and latest criterion among clearly trend-preferring (S206; ΔBIC ≤ −6) or step-preferring learners (S215; ΔBIC ≥ 6). Upper plots show observed accuracy (gray) and predicted accuracy (dashed blue), averaged over trailing 32 trials. Lower plots show unsmoothed pre-choice target-rule consideration (light blue), belief (teal) and execution (ochre). Execution is undefined for S122's mixture readout. Model curves average eight fixed-parameter particle-filter replays; teal bands show belief ± one replay standard deviation, clipped to [0, 1], indicating numerical variation rather than confidence intervals. Dotted lines mark first trailing-64 accuracy > .9. In c, beige shading spans 32 trials either side of the behavior-only split after trial 650 (dashed line). Chance is .5 for Task 1 and .25 for Tasks 2/3.

**d,** All 12 participants (four/task). ΔBIC = BIC(trend) − BIC(step); positive values favor a step. Shading spans ±6. A constant model also competes when selecting the preferred shape. Labels identify a–c.

**e,** Gray lines show complete records, ordered by behavioral criterion. Teal intervals mark Q > .5 for ≥16 consecutive within-session trials, including episodes followed by support loss. Diamonds mark first behavioral criterion.

All 7,936 trials are retained; task targets are H0 (Task 1) and H42 (Tasks 2/3). Parameters use complete choice records. S122's weak verbal agreement limits interpretation. These are descriptive trajectories, not established subtypes or causal effects. Sensitivity: Fig. S2. Source data accompany the figure.
"""


def build(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    trials, subjects, manifest = read_story(source)
    assert len(subjects) == 12 and len(trials) == 7936
    episodes = support_episodes(trials)
    journal_style()
    fig = plt.figure(figsize=(183 / 25.4, 192 / 25.4))
    xs = [.080, .405, .730]
    examples = list(manifest['examples'].values())
    headings = ['Earlier improvement', 'Gradual improvement', 'Abrupt improvement']
    handles = [Line2D([], [], color=COLORS[key], lw=1.15, ls=linestyle, label=name)
               for key, name, linestyle in [('observed', 'Observed', '-'),
                 ('predicted', 'Predicted', '--'), ('available', 'Considered', '-'),
                 ('belief', 'Belief', '-'), ('executed', 'Executed', '-')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, 1.005),
               ncol=5, columnspacing=1.8, handlelength=2.1)
    for column, (subject, heading) in enumerate(zip(examples, headings)):
        x = xs[column]
        row = subjects.loc[subject]
        group = trials.loc[trials.iSub.eq(subject)].sort_values('trial')
        task = int(row.task)
        panel_label(fig, chr(97 + column), x-.045, .951)
        fig.text(x, .951, heading, fontsize=7.1, va='top')
        fig.text(x, .929, f'S{subject}  ·  Task {task}', color=TASK_COLORS[task], fontsize=6.2, va='top')
        upper = fig.add_axes([x, .724, .235, .181])
        lower = fig.add_axes([x, .481, .235, .200])
        upper.plot(group.trial, group.correct_w32, lw=.9, color=COLORS['observed'])
        upper.plot(group.trial, group.predicted_w32, lw=.95, ls='--', color=COLORS['predicted'])
        upper.axhline(.5 if task == 1 else .25, color='.84', lw=.5, zorder=0)
        lower.fill_between(group.trial,
                           np.clip(group.belief-group.belief_seed_sd, 0, 1),
                           np.clip(group.belief+group.belief_seed_sd, 0, 1),
                           color=COLORS['belief'], alpha=.14, lw=0, zorder=1)
        for state in ['available', 'belief', 'executed']:
            if group[state].notna().any():
                lower.plot(group.trial, group[state], color=COLORS[state],
                           lw=.75 if state == 'available' else .9)
        for axis in [upper, lower]:
            axis.set(xlim=(1, len(group)), ylim=(-.035, 1.035), yticks=[0, .5, 1])
            axis.axvline(row.criterion, lw=.65, ls=':', color='.60', zorder=0)
            axis.tick_params(pad=2)
            if column == 2:
                axis.axvspan(row.split-31.5, row.split+32.5,
                             color='#EDE8DF', alpha=.8, zorder=0, lw=0)
                axis.axvline(row.split+.5, color='#8C806F', lw=.7, ls='--', zorder=1)
        upper.set_xticks([])
        lower.set(xticks=[1, int(len(group)/2), len(group)], xlabel='Trial')
        if column == 0:
            upper.set_ylabel('Accuracy')
            lower.set_ylabel('Target-rule state')
        else:
            upper.tick_params(labelleft=False)
            lower.tick_params(labelleft=False)
        if column == 2:
            upper.text(.03, .06, 'Change at trial 650', transform=upper.transAxes,
                       fontsize=6, color='#7C705F')

    # Cohort panels use different horizontal scales but retain the true trial unit.
    ax = fig.add_axes([.100, .083, .315, .284])
    panel_label(fig, 'd', .035, .410)
    fig.text(.100, .409, 'Timing and shape of improvement', fontsize=7, va='top')
    ax.axhspan(-6, 6, lw=0, color='#F0F1F1', zorder=0)
    ax.axhline(0, color='.8', lw=.5, zorder=0)
    offsets = {122: (7, 6), 206: (-25, 6), 215: (6, 4)}
    for subject, row in subjects.iterrows():
        task = int(row.task)
        ax.scatter(row.criterion, row.delta_bic, marker=MARKERS[task],
                   color=TASK_COLORS[task], s=24 if subject in examples else 18,
                   edgecolor='white', lw=.5, zorder=3)
        if subject in examples:
            ax.annotate(f'S{subject}', (row.criterion, row.delta_bic),
                        xytext=offsets[subject], textcoords='offset points',
                        color=TASK_COLORS[task], fontsize=6)
    ax.set(xlim=(0, 1460), ylim=(-58, 31), xticks=[0, 700, 1400],
           yticks=[-50, -25, 0, 25], xlabel='Trial of behavioral criterion',
           ylabel='Step preference (ΔBIC)')
    ax.text(.98, .94, 'Step', transform=ax.transAxes, ha='right', fontsize=6, color='.45')
    ax.text(.98, .045, 'Gradual', transform=ax.transAxes, ha='right', fontsize=6, color='.45')
    ax.legend(handles=[Line2D([], [], marker=MARKERS[task], ls='',
                     color=TASK_COLORS[task], markersize=3.7, label=f'Task {task}')
                     for task in [1, 2, 3]], loc='lower left', bbox_to_anchor=(-.15, -.32),
              ncol=3, handletextpad=.25, columnspacing=.85)

    ax = fig.add_axes([.577, .083, .381, .284])
    panel_label(fig, 'e', .505, .410)
    fig.text(.577, .409, 'Episodes of sustained belief', fontsize=7, va='top')
    order = subjects.sort_values('criterion').index
    for index, subject in enumerate(order):
        row = subjects.loc[subject]
        ax.plot([1, row.n], [index, index], color='#E6E8E9', lw=2.2,
                solid_capstyle='butt', zorder=0)
        for event in episodes.loc[episodes.subject.eq(subject)].itertuples():
            ax.plot([event.start-.5, event.stop+.5], [index, index],
                    color=COLORS['belief'], lw=2.8, solid_capstyle='butt', zorder=1)
        ax.scatter(row.criterion, index, marker='D', color=COLORS['observed'],
                   s=12, edgecolors='white', lw=.4, zorder=3)
    ax.set(xlim=(0, 1460), ylim=(11.65, -.65), xticks=[0, 700, 1400],
           yticks=range(12), yticklabels=[f'S{subject}' for subject in order],
           xlabel='Trial')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=4)
    for tick, subject in zip(ax.get_yticklabels(), order):
        tick.set_color(TASK_COLORS[int(subjects.loc[subject].task)])
    ax.legend(handles=[Line2D([], [], lw=2.8, color=COLORS['belief'], label='Sustained support'),
                       Line2D([], [], marker='D', ls='', color=COLORS['observed'],
                              markersize=3.5, label='Behavioral criterion')],
              loc='lower left', bbox_to_anchor=(-.02, -.32), ncol=2,
              handlelength=1.8, columnspacing=1.1)
    save_bundle(fig, source, output, 'Figure3.png', Path(__file__), LEGEND,
                {'support_episodes.csv': episodes})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    main()
