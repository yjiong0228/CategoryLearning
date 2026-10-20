"""Show all16 previously diagnosed tail-max error cases and their next cue opportunity.

Contract: participant207 often repeats a local response/reason when the cue next
occurs after an error. Panel a shows three overlapping descriptive counts;
panel b retains every qualifying pair, elapsed trials, literal next report and
next choice/feedback. Quantitative grid, with individual pairs as the main panel.
Python,183x178mm,PNG only at300dpi. No intervals or tests:16 dependent events
from one development participant, not independent replicates. Other subjects
and all literal origins remain in the linked analysis tables. Build anew; no
statistical logic inherited from a prior figure. No latent-rule retention claim.

Use --source COMPLETED_CUE_PERSISTENCE --output NEW_DIRECTORY.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write


def plot(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new cue-persistence figure directory')
    done = load(source/'completion.json')
    if done['status'] != 'complete' or 207 not in done['subjects']:
        raise ValueError('Require the completed, unfiltered development chronology')
    inputs = {str(Path(__file__).resolve()): digest(Path(__file__)),
              str((source/'completion.json').resolve()): digest(source/'completion.json')}
    for name, h in done['artifacts'].items():
        path = source/name
        if digest(path) != h: raise ValueError('Chronology source table changed')
        inputs[str(path.resolve())] = h
    events = pd.read_csv(source/'events.csv')
    selection = ((events.subject == 207) & (events.phase == 'training') & (events.text == '尾巴最长。')
                 & (events.source_choice == 2) & (events.source_feedback == 0))
    cases = events[selection & (events.event == 'next_weak_cue')].sort_values('source_ordinal').copy()
    strict = events[selection & (events.event == 'next_strict_cue')].sort_values('source_ordinal')
    if len(cases) != 16 or not cases.status.eq('observed').all():
        raise ValueError('Require every one of the16 previously declared error cases')
    np.testing.assert_array_equal(cases.target_ordinal, strict.target_ordinal)
    if cases.source_ordinal.duplicated().any(): raise ValueError('Repeated source identity')
    same = cases.target_choice.eq(cases.source_choice)
    exact = cases.target_text.eq(cases.text)
    phrase = cases.target_text.str.contains('尾巴最长', regex=False)
    for name, expected in [('same_choice', same), ('same_wording', exact), ('mentions_literal_phrase', phrase)]:
        np.testing.assert_array_equal(cases[name].astype(bool), expected)
    counts = pd.DataFrame(dict(metric=['相同选项（选2）', '原句再次出现', '仍包含“尾巴最长”'],
                               count=[int(same.sum()), int(exact.sum()), int(phrase.sum())], denominator=len(cases)))
    font_manager.findfont('Noto Sans CJK JP', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
        'font.size': 7, 'axes.linewidth': .65, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(183/25.4, 178/25.4))
    fig.text(.10, .965, '错误反馈后，局部理由仍会再次出现', fontsize=11, weight='bold')
    fig.text(.10, .927, '被试207 · 训练段全部16组“尾巴最长→2”的错误回答 · 后续试次由刺激确定', fontsize=7)
    ax = fig.add_axes([.29, .73, .59, .145])
    ax.barh(np.arange(3), counts['count']/counts.denominator, height=.48,
            color=['#597FA0', '#88A7BD', '#A9BDC9'])
    ax.set_yticks(np.arange(3), counts.metric); ax.invert_yaxis(); ax.set_xlim(0, 1.10)
    ax.set_xticks([0, .5, 1], ['0', '50%', '100%']); ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=7)
    for i, row in enumerate(counts.itertuples()):
        ax.text(row.count/row.denominator+.025, i, f'{row.count}/{row.denominator}', va='center', fontsize=7)
    fig.text(.10, .887, 'a  下一次尾巴最长时的行为与原话', fontsize=8, weight='bold')
    fig.text(.10, .680, 'b  全部前后试次：每行保留一组真实记录', fontsize=8, weight='bold')
    gaps = fig.add_axes([.13, .215, .28, .425])
    wording = fig.add_axes([.46, .215, .50, .425])
    n = len(cases); y = np.arange(n)
    gaps.set_ylim(n-.5, -.5); wording.set_ylim(n-.5, -.5)
    gaps.set_xlim(-.25, 8.75); gaps.set_xticks([0, 2, 4, 6, 8]); gaps.set_xlabel('距错误回答的试次数', labelpad=5)
    source_labels = [str(int(r.source_ordinal))+('†' if r.source_cue_status == 'false' else '') for r in cases.itertuples()]
    gaps.set_yticks(y, source_labels); gaps.set_ylabel('原错误试次', labelpad=5)
    gaps.tick_params(axis='y', length=0, pad=5); gaps.spines['left'].set_visible(False)
    wording.set_xlim(0, 1); wording.axis('off')
    wording.text(0, 1.035, '后续试次与原话', transform=wording.transAxes, fontsize=7, weight='bold')
    wording.text(.82, 1.035, '选项／反馈', transform=wording.transAxes, fontsize=7, weight='bold')
    for i, row in enumerate(cases.itertuples()):
        if i % 2 == 0:
            gaps.axhspan(i-.5, i+.5, color='#F3F5F6', zorder=0)
            wording.axhspan(i-.5, i+.5, color='#F3F5F6', zorder=0)
        colour = '#597FA0' if row.target_choice == 2 else '#777777'
        gaps.plot([0, row.gap_trials], [i, i], color=colour, lw=.85)
        gaps.scatter([0], [i], s=9, marker='s', facecolors='white', edgecolors='#666666', linewidths=.7, zorder=3)
        gaps.scatter([row.gap_trials], [i], s=13, color=colour, zorder=3)
        wording.text(0, i, f'{int(row.target_ordinal)}  {row.target_text}', va='center', fontsize=7,
                     weight='bold' if row.same_wording else 'normal')
        outcome = '对' if row.target_feedback == 1 else '错'
        wording.text(.84, i, f'{int(row.target_choice)}／{outcome}', va='center', fontsize=7, color=colour)
    fig.text(.10, .145, '后续机会：同一session及训练阶段内，首次再次满足“尾巴最长”的刺激。', fontsize=7)
    fig.text(.10, .112, '三项计数可重叠；该组严格最长与允许并列最长的目标相同。† 3条原刺激的尾巴并非最长。', fontsize=6.6)
    fig.text(.10, .079, '这是1位开发被试的描述性记录；重复口述和选项的出现与内部规则持续保留分别解释。', fontsize=6.6)
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output/'plot_source.py')
    cases.to_csv(output/'cases.csv', index=False); counts.to_csv(output/'counts.csv', index=False)
    fig.savefig(output/'cue_persistence.png', dpi=300, facecolor='white'); plt.close(fig)
    for p, h in inputs.items():
        if digest(Path(p)) != h: raise ValueError('Figure input changed')
    write(output/'completion.json', dict(status='complete', subject=207, paired_events=n, source_origins=done['origins'],
        source_event_rows=len(events), selection='207,training,tail-max literal,choice2,feedback0,next weak cue; all16 retained',
        statistics='Observed counts from dependent events; no interval or causal test',
        backend='python', width_mm=183, height_mm=178, dpi=300, formats=['PNG'], inputs_sha256=inputs,
        artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); plot(args.source, args.output)
