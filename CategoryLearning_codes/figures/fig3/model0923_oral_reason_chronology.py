"""Show observed wording and inferred reference-rule use on one trial axis.

Contract: in one development participant, explicit branch wording becomes common
later than the model's high reference-rule presence. Three distinct measurements
remain in separate panels: observed accuracy, literal wording frequencies, and
behavior-only marginal states. Quantitative grid with the state panel emphasized;
Python, 183 x 170 mm, PNG only at 300 dpi. All 960 trials and both proposal arms
are retained in fixed 64-trial blocks. State ribbons are one whole-filter Monte
Carlo standard error from eight filters, not participant uncertainty. Templates
are descriptive exact wording; no learning transition is inferred from them.

Use --source COMPLETED_CHRONOLOGY --output NEW_DIRECTORY.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new chronology figure directory')
    done = load(source / 'completion.json')
    if done['status'] != 'complete' or done['subject'] != 207 or done['reports'] != 960:
        raise ValueError('Require the completed 207 chronology')
    inputs = {str(Path(__file__).resolve()):digest(Path(__file__)),
              str((source / 'completion.json').resolve()):digest(source / 'completion.json')}
    for name, sha in done['artifacts_sha256'].items():
        path = source / name
        if digest(path) != sha:
            raise ValueError('Chronology source table changed')
        inputs[str(path.resolve())] = sha
    blocks = pd.read_csv(source / 'blocks.csv'); trials = pd.read_csv(source / 'trials.csv', keep_default_na=False)
    states = pd.read_csv(source / 'state_blocks.csv')
    if (len(blocks) != 15 or blocks.trials.sum() != 960 or not blocks.trials.eq(64).all()
            or set(states.model) != {'MIX'} or set(states.arm) != {'POOL', 'SEARCH'}):
        raise ValueError('Require all blocks and both completed MIX references')
    for row in blocks.itertuples():
        group = trials[trials.block64 == row.block64]
        assert len(group) == row.trials
        np.testing.assert_allclose(row.correct_rate, group.correct.mean(), atol=1e-12, rtol=0)
        assert row.template_reports == group.template_matched.sum()
        assert row.longest_word_reports == group.contains_longest_word.sum()
    x = (blocks.first_trial + blocks.last_trial).to_numpy() / 2
    font_manager.findfont('Noto Sans CJK JP', fallback_to_default=False)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
        'font.size':7, 'axes.linewidth':.6, 'axes.spines.top':False, 'axes.spines.right':False,
        'legend.frameon':False, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, axes = plt.subplots(3, 1, figsize=(183/25.4, 170/25.4), sharex=True,
                             gridspec_kw={'height_ratios':[.7, 1., 1.25]})
    fig.subplots_adjust(left=.115, right=.975, bottom=.19, top=.90, hspace=.50)
    fig.suptitle('207：口述表达与模型状态的时间进程', x=.115, y=.975, ha='left', fontsize=11, weight='bold')
    fig.text(.115, .936, '三个面板分别呈现行为、已观察措辞和模型状态；每点汇总64题。', fontsize=7)
    axes[0].plot(x, blocks.correct_rate, color='#555555', marker='o', markersize=3, linewidth=1.3)
    axes[0].set_ylabel('正确率'); axes[0].set_title('a  被试行为', loc='left', fontsize=8, weight='bold')
    axes[1].plot(x, blocks.template_reports/blocks.trials, color='#8C629B', marker='o', markersize=3,
                 linewidth=1.4, label='四种固定分支措辞')
    axes[1].plot(x, blocks.longest_word_reports/blocks.trials, color='#999999', marker='s', markersize=3,
                 linewidth=1., label='包含“最长”一词')
    axes[1].set_ylabel('试次比例'); axes[1].set_title('b  口述文字特征', loc='left', fontsize=8, weight='bold')
    axes[1].legend(loc='upper left', fontsize=7)
    axes[2].set_title('c  参考规则42的条件状态', loc='left', fontsize=8, weight='bold')
    colors = {'reference_presence':'#427A9B', 'reference_response_share':'#B87F50'}
    labels = {'reference_presence':'在场概率', 'reference_response_share':'作答份额'}
    for metric in colors:
        for arm, style in [('POOL', '--'), ('SEARCH', '-')]:
            group = states[(states.metric == metric) & (states.arm == arm)].sort_values('block64')
            if group.block64.tolist() != blocks.block64.tolist():
                raise ValueError('State blocks do not match all observed trials')
            values, error = group.value.to_numpy(), group.mc_se.to_numpy()
            axes[2].plot(x, values, color=colors[metric], linestyle=style, linewidth=1.3,
                         label=f'{labels[metric]} · {arm}')
            axes[2].fill_between(x, values-error, values+error, color=colors[metric], alpha=.12, linewidth=0)
    axes[2].set_ylabel('概率／份额'); axes[2].set_xlabel('原试次序号')
    axes[2].legend(loc='upper center', bbox_to_anchor=(.5,-.27), ncol=4, fontsize=6.3,
                   columnspacing=1.2, handlelength=2.5)
    for ax in axes:
        ax.set_xlim(0,960); ax.set_ylim(-.025,1.025); ax.set_yticks([0,.5,1])
        ax.axvline(672,color='#777777',linestyle=':',linewidth=.8)
        ax.tick_params(length=3, width=.6, labelsize=7)
    axes[0].text(678,.08,'训练截止：672',fontsize=6.4,color='#666666')
    axes[2].set_xticks([0,192,384,576,768,960])
    fig.text(.115,.075,'措辞示例：“脖子短，尾巴长”。模板频率描述文字形式；“长／短”的参照仍保留歧义。',fontsize=6.5)
    fig.text(.115,.047,'1名开发被试，960题；模型每方案8次完整滤波。色带：±1个完整滤波模拟标准误。',fontsize=6.5)
    fig.text(.115,.020,'规则在场与作答份额是本题选择后的边际状态。口述改变与内部规则学习的关系仍需检验。',fontsize=6.5)
    output.mkdir(parents=True)
    fig.savefig(output / 'chronology.png', dpi=300, facecolor='white')
    plt.close(fig)
    blocks.to_csv(output / 'observed_plot_source.csv', index=False)
    states.to_csv(output / 'state_plot_source.csv', index=False)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Bound figure input changed')
    write(output / 'validation.json', dict(status='complete', inputs_sha256=inputs, backend='python',
        dimensions_mm=[183,170], dpi=300, png_only=True, trials=960, excluded_trials=0,
        blocks=15, state_block_rows=len(states), report_templates='Exact wording; all other reports remain in trial denominator',
        scope='One development participant, no population test; state uncertainty is finite-filter Monte Carlo error',
        outputs_sha256={p.name:digest(p) for p in output.iterdir()}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); plot(args.source,args.output)
