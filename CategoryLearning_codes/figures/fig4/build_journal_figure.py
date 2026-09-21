"""Journal-style comparison of latent transitions and fitted learning mechanisms."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..fig3.build_abstract_story import read_story
from ..fig3.build_nine_subject_figures import COLORS, TASK_COLORS
from ..fig3.build_journal_figure import MARKERS, journal_style, panel_label, save_bundle


def feedback_search(trials: pd.DataFrame) -> pd.DataFrame:
    """Pair each feedback with next-trial search without crossing sessions or gaps."""
    records = []
    for subject, group in trials.groupby('iSub', sort=True):
        group = group.sort_values('trial')
        next_search = group.search.shift(-1)
        valid = (group.iSession.eq(group.iSession.shift(-1)) &
                 group.trial.add(1).eq(group.trial.shift(-1)) & next_search.notna())
        for feedback in [1., .5, 0.]:
            mask = valid & group.feedback.eq(feedback)
            values = next_search.loc[mask]
            records.append({'subject': int(subject), 'task': int(group.task.iloc[0]),
                            'feedback': feedback, 'n_pairs': int(mask.sum()),
                            'mean_next_search': values.mean()})
    return pd.DataFrame(records)


def case_window_accuracy(trials: pd.DataFrame, subject: int, split: int,
                         window: int = 32) -> tuple[float, float]:
    """Observed accuracy in exactly the existing replay-comparison windows."""
    group = trials.loc[trials.iSub.eq(subject)].set_index('trial')
    before = group.loc[split-window+1:split, 'correct']
    after = group.loc[split+1:split+window, 'correct']
    assert len(before) == window and len(after) == window
    return float(before.mean()), float(after.mean())


def paired_replays(axis, data: pd.DataFrame, color: str, offset: float = 0.,
                   dashed: bool = False) -> None:
    """Draw mean window changes with numerical min–max spread, not a confidence interval."""
    values = data[['before', 'after']].to_numpy(float)
    assert len(values) == 8 and np.isfinite(values).all()
    mean = values.mean(axis=0)
    axis.vlines(np.array([0., 1.])+offset, values.min(axis=0), values.max(axis=0),
                lw=2.3, color=color, alpha=.23, zorder=2)
    axis.plot(np.array([0., 1.])+offset, mean, lw=1.1, marker='o', ms=3.1,
              color=color, ls='--' if dashed else '-', zorder=3)


LEGEND = """# Fig. 4 | Abrupt behavioral gains can accompany different latent transitions.

**a,** All four participants with a preferred step model and BIC(trend) − BIC(step) ≥ 6. Pairs compare 32 trials immediately before and after behavior-only splits after trials 367, 650, 447 and 593 for S102, S215, S307 and S328, respectively. Upper plots show observed accuracy (gray) and model P(correct) (dashed blue); lower plots show pre-choice target-rule consideration (light blue), belief (teal) and execution (ochre). Model points average eight fixed-parameter particle-filter repeats; pale ranges are minima–maxima of repeat-specific window means, indicating numerical variability rather than confidence intervals. Observed accuracy has no replay range. Execution is undefined for S102/S328's mixture readout. The cue summarizes state definitions, not an established sequence of psychological events.

**b,** Each of 12 participants contributes mean model search probabilities on trial t + 1 following correct (1), partial (.5) or incorrect (0) feedback on trial t. Pairs are consecutive and within-session. Partial feedback occurs only in Task 2; other tasks connect applicable endpoints. Color/shape identify task. Unequal trial counts are provided in source data. Contrasts combine policy, history and stimulus effects.

**c,** Twelve participants' fitted evidence retention γ versus mean belief reallocation over the first 128 trials. Reallocation is total variation between adjacent complete belief distributions, calculated per replay then averaged. Labels identify Fig. 3 cases. Neither variable independently measures memory or psychological resetting.

Parameters use complete choice records. No group test or causal compensation effect is asserted; particle ranges do not measure parameter uncertainty. Further sensitivity appears in Fig. S2. Source data accompany the figure.
"""


def build(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    trials, subjects, manifest = read_story(source)
    assert len(subjects) == 12 and len(trials) == 7936
    jumps = pd.read_csv(source / 'jump_windows.csv')
    cases = subjects.loc[subjects.delta_bic.ge(6) & subjects.preferred.eq('step')]
    assert list(cases.index) == [102, 215, 307, 328]
    search = feedback_search(trials)
    journal_style()
    fig = plt.figure(figsize=(183 / 25.4, 186 / 25.4))
    panel_label(fig, 'a', .030, .977)
    fig.text(.075, .977, 'Latent transitions at abrupt improvement', fontsize=7.1, va='top')
    cue = fig.add_axes([.115, .884, .79, .065])
    cue.set(xlim=(0, 1), ylim=(0, 1))
    cue.axis('off')
    for x, state, symbol, title in [(0.08, 'available', 'A', 'Rule considered'),
                                   (.41, 'belief', 'Q', 'Belief in rule'),
                                   (.74, 'executed', 'E', 'Rule executed')]:
        cue.scatter([x], [.58], s=140, color=COLORS[state], alpha=.20, edgecolors='none')
        cue.text(x, .58, symbol, color=COLORS[state], fontsize=7, ha='center', va='center', weight='bold')
        cue.text(x+.041, .58, title, fontsize=6.4, ha='left', va='center')
    for left, right in [(.294, .373), (.628, .703)]:
        cue.annotate('', xy=(right, .58), xytext=(left, .58),
                     arrowprops={'arrowstyle': '->', 'lw': .55, 'color': '.5'})
    behavior_legend = [Line2D([], [], color=COLORS[key], lw=1, ls=style, label=name)
                       for key, style, name in [('observed', '-', 'Observed'),
                                               ('predicted', '--', 'Predicted')]]
    fig.legend(handles=behavior_legend, loc='upper right', bbox_to_anchor=(.96, .991),
               ncol=2, columnspacing=1.1, handlelength=1.7, fontsize=5.8)

    xs = [.085, .32, .555, .79]
    summary = []
    for x, (subject, row) in zip(xs, cases.iterrows()):
        task = int(row.task)
        fig.text(x, .855, f'S{subject}  ·  Task {task}', color=TASK_COLORS[task], fontsize=6.7, va='top')
        upper = fig.add_axes([x, .699, .167, .127])
        lower = fig.add_axes([x, .457, .167, .199])
        before, after = case_window_accuracy(trials, subject, int(row.split))
        upper.plot([0, 1], [before, after], marker='o', color=COLORS['observed'], lw=1.05, ms=3.3)
        selected = jumps.loc[jumps.subject.eq(subject) & jumps.window.eq(32) & jumps.variant.eq('selected')]
        paired_replays(upper, selected.loc[selected.measure.eq('predicted')], COLORS['predicted'], dashed=True)
        for state, offset in [('available', -.045), ('belief', 0.), ('executed', .045)]:
            values = selected.loc[selected.measure.eq(state)]
            if len(values):
                paired_replays(lower, values, COLORS[state], offset)
                summary.append({'subject': subject, 'task': task, 'split': int(row.split),
                                'measure': state, 'before_mean': values.before.mean(),
                                'after_mean': values.after.mean(),
                                'before_min': values.before.min(), 'before_max': values.before.max(),
                                'after_min': values.after.min(), 'after_max': values.after.max(),
                                'observed_before': before, 'observed_after': after})
        for axis in [upper, lower]:
            axis.set(xlim=(-.16, 1.16), ylim=(-.04, 1.04), yticks=[0, .5, 1])
            axis.tick_params(pad=2)
        upper.set_xticks([])
        lower.set(xticks=[0, 1], xticklabels=['Before', 'After'])
        if subject == cases.index[0]:
            upper.set_ylabel('Accuracy')
            lower.set_ylabel('Target-rule state')
        else:
            upper.tick_params(labelleft=False)
            lower.tick_params(labelleft=False)
        if not row.chi:
            lower.text(.5, .06, 'Mixture readout', transform=lower.transAxes,
                       ha='center', fontsize=5.7, color='.5')

    panel_label(fig, 'b', .030, .368)
    panel_label(fig, 'c', .523, .368)
    fig.text(.085, .368, 'Feedback and subsequent search', fontsize=7, va='top')
    fig.text(.59, .368, 'Evidence retention and belief updating', fontsize=7, va='top')
    ax = fig.add_axes([.10, .083, .337, .226])
    for subject, group in search.groupby('subject', sort=True):
        valid = group.loc[group.n_pairs.gt(0)].sort_values('feedback', ascending=False)
        task = int(valid.task.iloc[0])
        xpos = valid.feedback.map({1.: 0., .5: 1., 0.: 2.})
        ax.plot(xpos, valid.mean_next_search, color=TASK_COLORS[task], alpha=.78,
                marker=MARKERS[task], ms=3.3, lw=.7, mec='white', mew=.3)
    ax.set(xlim=(-.15, 2.15), ylim=(0, .575), xticks=[0, 1, 2],
           xticklabels=['Correct', 'Partial', 'Incorrect'], yticks=[0, .2, .4],
           xlabel='Feedback on trial t', ylabel='Search probability on trial t + 1')

    ax = fig.add_axes([.60, .083, .356, .226])
    offsets = {122: (-24, 7), 206: (-25, 6), 215: (6, 4)}
    examples = list(manifest['examples'].values())
    for subject, row in subjects.iterrows():
        task = int(row.task)
        ax.scatter(row.gamma, row.early_reallocation, color=TASK_COLORS[task],
                   marker=MARKERS[task], s=21, edgecolors='white', lw=.4)
        if subject in examples:
            ax.annotate(f'S{subject}', (row.gamma, row.early_reallocation),
                        xytext=offsets[subject], textcoords='offset points',
                        fontsize=6, color=TASK_COLORS[task])
    ax.set(xlim=(.32, 1.02), ylim=(.055, .245), xticks=[.4, .7, 1], yticks=[.1, .15, .2],
           xlabel='Evidence retention (γ)', ylabel='Early belief reallocation')
    fig.legend(handles=[Line2D([], [], marker=MARKERS[task], color=TASK_COLORS[task],
                    markersize=3.8, ls='', label=f'Task {task}') for task in [1, 2, 3]],
               loc='lower center', bbox_to_anchor=(.52, .005), ncol=3,
               handletextpad=.4, columnspacing=2.5)
    save_bundle(fig, source, output, 'Figure4.png', Path(__file__), LEGEND,
                {'feedback_search.csv': search, 'abrupt_state_windows.csv': pd.DataFrame(summary)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    main()
