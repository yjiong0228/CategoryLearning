"""Publication supplement: fitted resources and the resolution of decoded beliefs.

Reads existing results only. No fitting, model simulation or oral re-encoding.
Use a fresh output directory. All exported figures are PNG at 450 dpi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from src.Bayesian_state.optimization.model_0826 import extract_model_0826_parameters

ROOT = Path(__file__).resolve().parents[3]
ORDER = [102, 104, 118, 122, 307, 314, 315, 328, 206, 215, 221, 222]
TASK_COLORS = {1: '#487DA8', 2: '#A46A8A', 3: '#648B76'}
Q_COLOR, NEAR_COLOR = '#167D8D', '#7F8185'
PARAMETERS = ['M', 'gamma', 'chi', 'beta_0', 'eta_plus', 'eta_minus',
              'E_C', 'E_E', 'g_0', 'c_A', 'c_G']
PARAM_LABELS = [r'$M$', r'$\gamma$', r'$\chi$', r'$\beta_0$',
                r'$\eta_+$', r'$\eta_-$', r'$E_C$', r'$E_E$',
                r'$g_0$', r'$c_A$', r'$c_G$']
SOURCE = 'results/model_0826/fig34_twelve_subjects_20260921_v1'
ORAL_SOURCE = 'CategoryLearning_codes/figures/outputs/fig2/belief_validation_20260921_v2'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_probabilities(values: np.ndarray) -> np.ndarray:
    """Require a replay × trial × state probability array without exclusions."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or not np.isfinite(array).all() or np.any(array < -1e-12):
        raise ValueError('Expected finite, nonnegative replay/trial/state probabilities')
    np.testing.assert_allclose(array.sum(axis=-1), 1, atol=1e-7)
    return array


def pairwise_distance(values: np.ndarray) -> float:
    """Mean total variation over distinct replay pairs and aligned trials."""
    array = checked_probabilities(values)
    if len(array) < 2 or not array.shape[1]:
        raise ValueError('At least two repeats and one trial are required')
    return float(np.mean([.5 * np.abs(array[i] - array[j]).sum(axis=-1).mean()
                          for i, j in combinations(range(len(array)), 2)]))


def candidate_distance(selected: np.ndarray, alternative: np.ndarray) -> float:
    """Distance between repeat means; not a measure of parameter uncertainty."""
    first, second = checked_probabilities(selected), checked_probabilities(alternative)
    if first.shape[1:] != second.shape[1:]:
        raise ValueError('Parameter points must share trial and state identities')
    return float(.5 * np.abs(first.mean(axis=0) - second.mean(axis=0)).sum(axis=-1).mean())


def collect(source: Path, oral_source: Path, output: Path) -> dict:
    manifest_path = source / 'states/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['smoke'] or manifest['config']['oral_sigma'] != .05:
        raise ValueError('Need complete states and the existing oral encoder')
    cohort = {int(row['subject']): row for row in manifest['cohort']}
    if set(cohort) != set(ORDER):
        raise ValueError('Unexpected fitted cohort')
    people = pd.read_csv(source / 'analysis/subjects.csv').set_index('subject').loc[ORDER]
    seeds = pd.read_csv(oral_source / 'seeds.csv')
    change = pd.read_csv(oral_source / 'change_sensitivity.csv')
    jump = pd.read_csv(source / 'analysis/jump_windows.csv')
    inputs = [manifest_path, source / 'analysis/subjects.csv',
              oral_source / 'seeds.csv', oral_source / 'change_sensitivity.csv',
              source / 'analysis/jump_windows.csv', Path(__file__).resolve(),
              Path(__file__).with_name('JOURNAL_SUPPLEMENT_DESIGN.md')]
    metrics, parameters, traces = [], [], []
    trajectories = {}
    for sid in ORDER:
        info, identity = people.loc[sid], cohort[sid]
        fit_path = ROOT / identity['source'] / 'subjects' / str(sid) / 'fit_result.json'
        fit = json.loads(fit_path.read_text()); inputs.append(fit_path)
        alternate = next(item for item in fit['candidate_bank']
                         if item['id'] == identity['alternative'])
        selected_parameters = extract_model_0826_parameters(fit['hyperparams'])
        alternate_parameters = extract_model_0826_parameters(alternate['hyperparams'])
        for name in PARAMETERS:
            np.testing.assert_allclose(selected_parameters[name], info[name], atol=1e-10)
            parameters.append({'subject': sid, 'task': int(info.task), 'parameter': name,
                               'selected': selected_parameters[name],
                               'alternative': alternate_parameters[name],
                               'changed': not np.isclose(selected_parameters[name], alternate_parameters[name])})
        variants, common_mask, identity_arrays = {}, None, None
        for variant, n_replays in [('selected', 8), ('alternative', 4)]:
            q, p, seeds_seen = [], [], []
            for repeat in range(n_replays):
                path = source / f'states/S{sid}/{variant}_{repeat:02d}.npz'
                inputs.append(path)
                with np.load(path, allow_pickle=False) as state:
                    if (int(state['subject']) != sid or int(state['particles']) != 128
                            or str(state['point_id']) != identity[variant]):
                        raise ValueError(f'State identity mismatch for S{sid}')
                    mask = state['valid_trial_mask'].astype(bool) & state['score_trial_mask'].astype(bool)
                    observed = [state[key] for key in ('observed_choice', 'observed_feedback', 'true_category_index')]
                    if common_mask is None:
                        common_mask, identity_arrays = mask, observed
                    np.testing.assert_array_equal(mask, common_mask)
                    for current, reference in zip(observed, identity_arrays):
                        np.testing.assert_array_equal(current, reference)
                    q.append(state['marginal_prior']); p.append(state['pred_category_probs'])
                    seeds_seen.append(int(state['seed']))
            if len(set(seeds_seen)) != n_replays:
                raise ValueError('Independent repeats require unique seeds')
            variants[variant] = (checked_probabilities(q), checked_probabilities(p))
        q, p = variants['selected']; alt_q, alt_p = variants['alternative']
        if len(common_mask) != int(info.n) or common_mask.sum() != int(info.n) - 1:
            raise ValueError('Unexpected trial mask')
        metrics.append({'subject': sid, 'task': int(info.task), 'trials': int(info.n),
                        'scored_trials': int(common_mask.sum()),
                        'repeat_choice_tv': pairwise_distance(p[:, common_mask]),
                        'repeat_belief_tv': pairwise_distance(q[:, common_mask]),
                        'candidate_choice_tv': candidate_distance(p[:, common_mask], alt_p[:, common_mask]),
                        'candidate_belief_tv': candidate_distance(q[:, common_mask], alt_q[:, common_mask])})
        target = int(info.target)
        trajectories[sid] = {'selected': q[:, :, target], 'alternative': alt_q[:, :, target]}
        for variant, array in trajectories[sid].items():
            for t, values in enumerate(array.T, start=1):
                traces.append({'subject': sid, 'task': int(info.task), 'variant': variant,
                               'trial': t, 'target': target, 'mean': values.mean(),
                               'min': values.min(), 'max': values.max()})
    metrics = pd.DataFrame(metrics); parameters = pd.DataFrame(parameters)
    for name, frame in [('stability', metrics), ('parameters', parameters), ('trajectories', pd.DataFrame(traces)),
                        ('oral_replays', seeds), ('report_change_sensitivity', change), ('breakthrough_windows', jump)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    metadata = {'source': str(source.relative_to(ROOT)), 'oral_source': str(oral_source.relative_to(ROOT)),
                'input_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in dict.fromkeys(inputs)},
                'subjects': ORDER, 'participant_n': 12, 'trials': int(metrics.trials.sum()),
                'scored_trials': int(metrics.scored_trials.sum()), 'particle_count': 128,
                'selected_repeats': 8, 'alternative_repeats': 4,
                'numerical_ranges': 'observed minimum to maximum across particle replays; not confidence intervals',
                'fits': 'complete-choice-sequence fitted parameters; all strict stopping diagnostics unresolved',
                'new_model_execution': False, 'oral_sigma': .05,
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return {'people': people, 'stability': metrics, 'parameters': parameters, 'oral': seeds,
            'change': change, 'jump': jump, 'trajectories': trajectories, 'metadata': metadata}


def style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 6.7, 'axes.titlesize': 7, 'axes.labelsize': 6.5,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                         'xtick.major.width': .6, 'ytick.major.width': .6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'lines.linewidth': 1, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def heading(ax, letter: str, title: str, x: float = -.12, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    ax.text(0, y, title, transform=ax.transAxes, fontsize=7, va='bottom')


def parameter_panel(ax, frame: pd.DataFrame, people: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list('parameter_tint', ['#FFFFFF', '#C5DDE1'])
    ax.set_xlim(-1.12, len(PARAMETERS)); ax.set_ylim(12.13, -1.2); ax.axis('off')
    for j, name in enumerate(PARAMETERS):
        part = frame[frame.parameter.eq(name)].set_index('subject').loc[ORDER]
        low, high = part[['selected', 'alternative']].min().min(), part[['selected', 'alternative']].max().max()
        for i, (sid, row) in enumerate(part.iterrows()):
            shade = (row.selected - low) / (high - low) if high > low else .0
            ax.add_patch(Rectangle((j + .025, i + .025), .95, .95, facecolor=cmap(shade),
                                   edgecolor='#A47649' if row.changed else '#FFFFFF',
                                   linewidth=.85 if row.changed else .3))
            value = f'{row.selected:.4g}' if name.startswith('eta') else f'{row.selected:.3g}'
            ax.text(j + .5, i + .52, value, ha='center', va='center', fontsize=6.2)
        ax.text(j + .5, -.25, PARAM_LABELS[j], ha='center', va='center', fontsize=7)
    for i, sid in enumerate(ORDER):
        ax.text(-.19, i + .5, f'S{sid}', ha='right', va='center', fontsize=6.2,
                color=TASK_COLORS[int(people.loc[sid, 'task'])])
    for start, end, title in [(0, 2, 'Resources'), (2, 6, 'Choice'), (6, 11, 'Search')]:
        ax.plot([start + .05, end - .05], [-.62, -.62], color='#83888A', lw=.5)
        ax.text((start + end) / 2, -.94, title, ha='center', va='center', fontsize=6.5)
    for boundary in [4, 8]:
        ax.axhline(boundary, color='white', lw=.5)
    heading(ax, 'a', 'Resource, choice and search parameters', x=-.04, y=1.02)


def stability_panel(ax, frame: pd.DataFrame, prefix: str, letter: str, title: str) -> None:
    ax.plot([0, .25], [0, .25], color='#C2C5C7', lw=.7, ls='--', zorder=0)
    for row in frame.itertuples():
        x, y = getattr(row, f'{prefix}_choice_tv'), getattr(row, f'{prefix}_belief_tv')
        ax.scatter(x, y, s=18, c=TASK_COLORS[row.task], edgecolors='white', linewidths=.4, zorder=3)
        if row.subject in [122, 215, 307, 328]:
            offset = (3, 2) if row.subject != 307 else (-20, 3)
            ax.annotate(f'S{row.subject}', (x, y), xytext=offset, textcoords='offset points', fontsize=5.4)
    ax.set(xlim=(0, .25), ylim=(0, .72), xticks=[0, .1, .2], yticks=[0, .2, .4, .6],
           xlabel='Choice distance (TV)', ylabel='Belief distance (TV)')
    heading(ax, letter, title, x=-.23)


def oral_panel(ax, frame: pd.DataFrame) -> None:
    ax.plot([0, .7], [0, .7], color='#C2C5C7', lw=.7, ls='--', zorder=0)
    for sid in ORDER:
        group = frame[frame.subject.eq(sid)]; task = int(group.task.iloc[0])
        selected = group[group.variant.eq('selected')].compatibility.to_numpy()
        alternate = group[group.variant.eq('alternative')].compatibility.to_numpy()
        x, y = selected.mean(), alternate.mean()
        ax.errorbar(x, y, xerr=[[x - selected.min()], [selected.max() - x]],
                    yerr=[[y - alternate.min()], [alternate.max() - y]], fmt='o', markersize=3.3,
                    color=TASK_COLORS[task], elinewidth=.65, markeredgecolor='white', markeredgewidth=.35, zorder=3)
        if sid in [215, 328, 122, 314]:
            offsets = {215: (4, -8), 328: (-23, 5), 122: (2, 12), 314: (7, 1)}
            leader = {'arrowstyle': '-', 'color': '#8A8D90', 'lw': .4} if sid == 122 else None
            ax.annotate(f'S{sid}', (x, y), xytext=offsets[sid], textcoords='offset points',
                        fontsize=5.4, arrowprops=leader)
    ax.set(xlim=(-.015, .68), ylim=(-.015, .68), xticks=[0, .2, .4, .6], yticks=[0, .2, .4, .6],
           xlabel='Selected parameter point', ylabel='Nearby parameter point')
    heading(ax, 'd', 'Oral compatibility', x=-.23)


def change_panel(ax, frame: pd.DataFrame) -> None:
    ax.axhline(0, color='#B5B9BB', lw=.7, ls='--')
    for sid in ORDER:
        g = frame[frame.subject.eq(sid)].sort_values('oral_change_threshold')
        color = TASK_COLORS[int(g.task.iloc[0])]
        ax.plot(g.oral_change_threshold, g.change_difference, '-o', c=color, ms=2.7, lw=.8, alpha=.88)
        if sid in [118, 104, 122, 215, 328]:
            last = g.iloc[-1]
            offsets = {118: (4, 0), 104: (4, -3), 122: (4, 6), 215: (4, 2), 328: (4, -5)}
            ax.annotate(f'S{sid}', (last.oral_change_threshold, last.change_difference),
                        xytext=offsets[sid], textcoords='offset points', fontsize=5.6, color=color)
    ax.set(xlim=(.2, .90), xticks=[.25, .5, .75], xlabel='Report-change threshold (TV)',
           ylabel='Excess belief change per trial')
    heading(ax, 'e', 'Report-change correspondence', x=-.14)


def breakthrough_panel(ax, jump: pd.DataFrame, people: pd.DataFrame) -> None:
    cases = [sid for sid in ORDER if people.loc[sid, 'delta_bic'] >= 6]
    if cases != [102, 307, 328, 215]:
        raise ValueError('Behavioral case-selection source changed')
    for i, sid in enumerate(cases):
        color = TASK_COLORS[int(people.loc[sid, 'task'])]
        for window, marker, window_offset in [(32, 'o', -.13), (64, 's', .13)]:
            for variant, shift in [('selected', -.045), ('alternative', .045)]:
                g = jump[jump.subject.eq(sid) & jump.window.eq(window) &
                         jump.variant.eq(variant) & jump.measure.eq('belief')]
                values = g.before.to_numpy(); mean = values.mean()
                ax.errorbar(i + window_offset + shift, mean,
                            yerr=[[mean - values.min()], [values.max() - mean]], fmt=marker,
                            color=color, markerfacecolor=color if variant == 'selected' else 'white',
                            markersize=3.3, elinewidth=.6, capsize=1.4, markeredgewidth=.7)
    ax.set(xlim=(-.5, 3.6), ylim=(-.03, 1.03), xticks=np.arange(4),
           xticklabels=[f'S{sid}' for sid in cases], yticks=[0, .5, 1],
           ylabel='Target belief before improvement')
    handles = [Line2D([], [], color='#656B6E', marker='o', ms=3.3, ls='', label='32 trials'),
               Line2D([], [], color='#656B6E', marker='s', ms=3.3, ls='', label='64 trials'),
               Line2D([], [], color='#656B6E', marker='o', mfc='white', ms=3.3, ls='', label='Nearby point')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.02, 1.025), ncol=3,
              columnspacing=.8, handletextpad=.25, borderaxespad=0, frameon=False, fontsize=5.3)
    heading(ax, 'f', 'Pre-improvement belief', x=-.14)


def draw_main(bundle: dict, output: Path) -> Path:
    fig = plt.figure(figsize=(183 / 25.4, 220 / 25.4))
    parameter_panel(fig.add_axes([.07, .625, .895, .30]), bundle['parameters'], bundle['people'])
    stability_panel(fig.add_axes([.09, .348, .235, .19]), bundle['stability'],
                    'repeat', 'b', 'Particle replay')
    stability_panel(fig.add_axes([.405, .348, .235, .19]), bundle['stability'],
                    'candidate', 'c', 'Nearby parameters')
    oral_panel(fig.add_axes([.72, .348, .235, .19]), bundle['oral'])
    change_panel(fig.add_axes([.09, .077, .36, .175]), bundle['change'])
    breakthrough_panel(fig.add_axes([.58, .077, .375, .175]), bundle['jump'], bundle['people'])
    legend = [Line2D([], [], ls='', marker='o', ms=4, color=color, label=f'Task {task}')
              for task, color in TASK_COLORS.items()]
    legend.append(Line2D([], [], ls='', marker='s', ms=5, color='#A47649', mfc='white',
                         label='Parameter differs at nearby point'))
    fig.legend(handles=legend, loc='center', bbox_to_anchor=(.52, .578), ncol=4,
               columnspacing=1.3, handletextpad=.4, frameon=False, fontsize=6)
    path = output / 'FigureS2_model_resolution.png'
    fig.savefig(path, dpi=450); plt.close(fig)
    return path


def draw_atlas(bundle: dict, output: Path) -> Path:
    fig = plt.figure(figsize=(183 / 25.4, 210 / 25.4))
    cols = [[102, 104, 118, 122], [307, 314, 315, 328], [206, 215, 221, 222]]
    for c, subjects in enumerate(cols):
        left = .08 + .318 * c
        fig.text(left + .119, .968, f'Task {c + 1}', ha='center', color=TASK_COLORS[c + 1],
                 fontsize=7, fontweight='bold')
        for r, sid in enumerate(subjects):
            ax = fig.add_axes([left, .742 - .218 * r, .238, .152])
            arrays = bundle['trajectories'][sid]; n = arrays['selected'].shape[1]
            x = np.arange(1, n + 1)
            for variant, color, linestyle in [('alternative', NEAR_COLOR, '--'), ('selected', Q_COLOR, '-')]:
                values = arrays[variant]
                ax.fill_between(x, values.min(0), values.max(0), color=color,
                                alpha=.13 if variant == 'alternative' else .19, lw=0)
                ax.plot(x, values.mean(0), color=color, lw=.8, ls=linestyle)
            ax.set(xlim=(1, n), ylim=(-.015, 1.025), yticks=[0, .5, 1],
                   xticks=[1, n // 2, n], xlabel='Trial' if r == 3 else '')
            ax.text(0, 1.08, f'S{sid}', transform=ax.transAxes, fontsize=6.6)
            if c == 0:
                ax.set_ylabel('Target-rule belief')
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.text(-.18, 1.08, chr(ord('a') + c), transform=ax.transAxes,
                        fontsize=8, fontweight='bold')
    handles = [Line2D([], [], color=Q_COLOR, lw=1, label='Selected point · 8 replays'),
               Line2D([], [], color=NEAR_COLOR, lw=1, ls='--', label='Nearby point · 4 replays')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.52, .925), ncol=2,
               frameon=False, fontsize=6.3, columnspacing=2)
    path = output / 'FigureS2_belief_trajectories.png'
    fig.savefig(path, dpi=450); plt.close(fig)
    return path


def write_caption(bundle: dict, output: Path) -> None:
    main_legend = """# Supplementary Fig. 2 | Fitted cognitive resources and the resolution of belief reconstruction

**a**, Selected resource, choice and search parameters for 12 participants (four
per task). Shading is scaled within parameter; outlined cells differ at a nearby
search finalist. **b**, Participant-level numerical variability: mean
total-variation (TV) distance between full choice distributions versus full
rule-belief distributions, averaged over scored trials and 28 pairs of eight
independent particle-filter replays. **c**, Corresponding distances between the
repeat means at the selected and nearby parameter points; dashed lines in b–c
denote equal distance. **d**, Current-report compatibility at both points,
calculated from pre-choice belief and the fixed oral encoder's relative report
likelihood. Reports were not fitted. Points are replay means; bars span replay
minima and maxima. **e**, Excess belief change per trial for changed versus stable
reports across three report-change thresholds. Lines denote participants;
comparisons use category-equivalent beliefs and exact-gap matching within
participant, session and chosen category. **f**, Target belief before improvement
for all four participants with step-over-trend BIC advantage ≥6. Circles and
squares use 32 and 64 preceding trials; filled and open marks indicate selected
and nearby parameters. Centres are replay means; bars span replay ranges.
Improvement locations were selected using behavior alone.

Analyses include 7,924 scored trials from 7,936 observations. Selected and nearby
points use eight and four replays, respectively, with 128 particles each. Ranges
represent numerical variability, not confidence intervals; comparisons are
descriptive. Parameters were fitted to complete choice sequences. Nearby points
are neither model ablations nor posterior samples. Strict fitting diagnostics
remain unresolved. Source data and metric definitions accompany this figure.
"""
    atlas_legend = """# Supplementary trajectory atlas | Complete belief trajectories and numerical variability

**a–c**, Full unsmoothed pre-choice target-rule beliefs for all 12 participants,
grouped by task. Teal solid lines show selected-parameter means across eight
particle-filter replays; grey dashed lines show nearby-candidate means across four
replays. Shaded bands span replay minima and maxima, not confidence intervals.
Each replay uses 128 particles. Parameters were fitted to the complete choice
sequence; these are observed-history reconstructions, not held-out predictions.
The atlas retains the first trial, which is excluded from scored summary
distances. Nearby candidates provide a finite sensitivity check, not parameter
recovery. Source data are available in trajectories.csv.
"""
    methods = """## Methods notes and complete definitions

**a**, Selected parameter values for all 12 fitted participants. Columns correspond
to the resource, choice and search components of the same model shown in Fig2a.
M is hypothesis capacity; gamma is evidence retention; chi identifies mixture (0)
or persistent-rule (1) choice; beta0 is initial choice precision; eta+ and eta− are
its increase and decrease rates. EC and EE are event probabilities after correct
and incorrect feedback, g0 is baseline global-search probability, cA is accumulated
failure modulation of search events, and cG is failure modulation of global search.
Shading is scaled separately to the observed range of each parameter across both
parameter points; numeric entries are the actual selected values. Outlined cells
change at the nearby candidate; all nearby values are in parameters.csv. Colour
indicates task and does not assign learners to cognitive subtypes.

**b**, Numerical repeat variability: mean total-variation distance between full
choice distributions versus between full rule-belief distributions. Each point is
one participant. Distances are averaged over scored trials and all 28 distinct
pairs of eight independent particle-filter replays at fixed parameters. **c**,
Corresponding distances between the repeat-mean trajectories at the selected point
(eight replays) and one documented nearby search finalist (four replays). The same
probability-distance axes are used in b and c; dashed lines denote equal distance.
The nearby point is not a posterior draw or a model ablation.

**d**, Current-report compatibility at the two parameter points. Centres are means
across replays; bars span observed replay minima and maxima, not confidence
intervals. Compatibility is the expectation, under the pre-choice rule belief,
of the fixed oral encoder's relative report likelihood, O(h)/max O. It preserves
ambiguity when one reported category is compatible with multiple full rules.
The same valid reports, choices and sigma=0.05 encoder enter both comparisons;
reports were not used to fit parameters. The oral catalogue itself uses report
information, so these scores do not establish unrestricted semantic recovery.

**e**, Report-change correspondence across three oral-change thresholds (total
variation 0.25, 0.50, 0.75). Each line is one participant. The y-axis is the excess
model-belief change per trial for changed versus stable reports. Reports are paired
only within session and chosen category; a missing report breaks that category's
chain; the maximum gap is 32 trials. Beliefs are projected into the oral encoder's
category-equivalence groups. Changed and stable pairs are matched on exact trial
gap within participant. The stable threshold remains TV <= 0.10. These are
descriptive associations, not event-time prediction accuracy or intervention effects.

**f**, Target belief immediately before a behaviorally fitted improvement for all
four participants whose step model exceeds the monotonic trend model by at least
6 BIC units: S102, S307, S328 and S215. Circles use the 32 preceding trials; squares
use 64. Filled marks show selected parameters; open marks show the nearby candidate.
Centres are replay means and bars are replay ranges. The improvement locations are
selected using behavior alone. Large ranges limit precise claims about latent
belief being established before improvement.

**Companion trajectory atlas.** Full unsmoothed target-rule belief trajectories for
all 12 participants, arranged by task. Teal solid lines are selected-point means;
grey dashed lines are nearby-point means. Bands are replay minima to maxima (eight
and four replays, respectively), not confidence intervals. Q is pre-choice;
parameters were fitted using the complete choice sequence. The atlas includes the
first trial; summary distances omit the first trial under the saved scoring mask.

## Scope and reproduction

All panels reuse completed outputs: 12 people, 7,936 trials, 7,924 scored trials;
128 particles per replay. No new fits, simulations, oral recoding or participant
exclusions were performed. Particle repeats and parameter points are technical
comparisons, not independent participants. No inferential tests or population
confidence intervals are claimed. Strict fit stopping diagnostics remain unresolved.
No compatible completed parameter recovery, alternative-model comparison or module
ablation is supplied by these results; this figure does not claim one.

The previous subject-first recovery run failed numerical calibration, so it is not
combined with these current fits. Historic S129 optimizer probes are also excluded.
The manifest records source hashes and actual package versions. Source CSVs preserve
all numeric values behind the figure. See the source-adjacent design note for the
panel argument and data boundaries.

```bash
MPLCONFIGDIR=/tmp/model0826_journal_mpl python -m CategoryLearning_codes.figures.figS2.build_journal_supplement \\
  --output CategoryLearning_codes/figures/outputs/figS2/journal_NEW
```
"""
    counts = {'main_legend_words': len(main_legend.split()),
              'atlas_legend_words': len(atlas_legend.split())}
    if counts['main_legend_words'] > 300 or counts['atlas_legend_words'] > 120:
        raise ValueError('Publication caption word budget exceeded')
    (output / 'legend_main.md').write_text(main_legend)
    (output / 'legend_atlas.md').write_text(atlas_legend)
    (output / 'caption_counts.json').write_text(json.dumps(counts, indent=2) + '\n')
    (output / 'README.md').write_text(main_legend + '\n' + atlas_legend + '\n' + methods)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--oral-source', default=ORAL_SOURCE)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    bundle = collect((ROOT / args.source).resolve(), (ROOT / args.oral_source).resolve(), output)
    style(); paths = [draw_main(bundle, output), draw_atlas(bundle, output)]
    write_caption(bundle, output)
    print(json.dumps({'figures': [str(path.relative_to(ROOT)) for path in paths],
                      'participants': len(ORDER), 'new_model_execution': False}, indent=2))


if __name__ == '__main__':
    main()
