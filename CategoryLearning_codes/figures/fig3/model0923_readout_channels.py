"""Plot fixed-state error allocation for complete saved autonomous banks.

Run --source READOUT_DIAGNOSIS --output NEW_DIRECTORY. The upper panels partition
expected errors. Lower panels show every path before and after instantaneous
target-only readout where the target already exists. Both sides use the same
saved states; these are not new autonomous simulations. Output is PNG only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load

CHANNELS = [('missing_target_error', 'Target absent', '#8294AB'),
            ('target_emission_error', 'Target emission', '#C69462'),
            ('other_rules_error', 'Other rules', '#6B9E91')]
ARMS = ('POOL', 'SEARCH')


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    completion, protocol = load(source/'completion.json'), load(source/'protocol.json')
    if completion['status'] != 'complete' or completion['smoke']:
        raise ValueError('Need completed scientific readout diagnostics')
    for name, expected in completion['outputs_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError('Readout diagnostic output changed')
    for name, expected in protocol['input_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError('Autonomous source changed')
    full = pd.read_csv(source/'path_windows.csv')
    frame = full[full.window.eq('late128')]
    subjects = sorted(frame.subject.unique())
    if frame.duplicated(['subject', 'model', 'seed', 'metric']).any():
        raise ValueError('Duplicated path score')
    observed, data_hashes = {}, {}
    for subject in subjects:
        job = load(source/'jobs'/f'S{subject}_POOL.json')
        folder = Path(job['folder'])
        meta = load(folder/'prepared.json')
        parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            if digest(folder/name) != meta['array_sha256'][name]:
                raise ValueError('Prepared choices changed')
            data_hashes[str(folder/name)] = digest(folder/name)
            with np.load(folder/name) as saved:
                parts.append(saved['choices'] == saved['categories'])
        observed[subject] = np.concatenate(parts)[-128:].mean()
    output.mkdir(parents=True)
    contract = dict(conclusion='The saved paths distinguish target absence, target-emission '
        'errors and competing-rule errors; target-only readout gains vary across subjects.',
        panels='Upper: error mass by target availability and rule emission; '
               'lower: every path under the original and fixed-state target-only readout.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=133,
        export='PNG only, 300 dpi, discussion draft', n_definition='independent generated paths',
        center='mean across every path', spread='1 Monte Carlo standard error of whole paths',
        tests='none; descriptive mechanism diagnosis', source_rows=len(full), displayed_rows=len(frame),
        window='last 128 trials; complete whole-sequence scores remain in source',
        limits='Fixed states; instantaneous post-hoc target selection; no learning feedback propagated',
        table_sha256=digest(source/'path_windows.csv'), observed_data_sha256=data_hashes)
    (output/'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    frame.to_csv(output/'plot_source.csv', index=False)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':7, 'axes.titlesize':8, 'axes.spines.right':False, 'axes.spines.top':False,
        'axes.linewidth':.7, 'legend.frameon':False, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, len(subjects), figsize=(183/25.4, 133/25.4), sharey='row', squeeze=False)
    fig.subplots_adjust(left=.09, right=.99, top=.84, bottom=.17, wspace=.27, hspace=.40)
    gains = []
    for column, subject in enumerate(subjects):
        part = frame[frame.subject.eq(subject)]
        for a, arm in enumerate(ARMS):
            wide = part[part.model.eq(arm)].pivot(index='seed', columns='metric', values='value')
            if not np.isfinite(wide.to_numpy()).all():
                raise ValueError('Incomplete path metrics')
            ax = axes[0, column]
            bottom = 0.
            for metric, label, color in CHANNELS:
                mean = wide[metric].mean()
                ax.bar(a, mean, bottom=bottom, color=color, width=.55,
                       edgecolor='white', linewidth=.4)
                bottom += mean
            np.testing.assert_allclose(bottom, wide.expected_error.mean(), atol=1e-12, rtol=0)
            ax.errorbar(a, bottom, yerr=wide.expected_error.std(ddof=1)/np.sqrt(len(wide)),
                        fmt='none', color='#333333', lw=.8, capsize=2)
            ax.text(a, bottom+.035, f'{bottom:.3f}', ha='center', fontsize=6)
            ax = axes[1, column]
            x = np.array([a*3, a*3+1])
            values = wide[['expected_accuracy', 'force_present_target_correct']].to_numpy()
            ax.plot(x, values.T, color='#8294AB', alpha=.12, lw=.4)
            ax.errorbar(x, values.mean(axis=0), yerr=values.std(axis=0, ddof=1)/np.sqrt(len(values)),
                        color='#222222', fmt='o-', ms=3, lw=1., capsize=2)
            gain = values[:, 1]-values[:, 0]
            gains.append(dict(subject=subject, model=arm, paths=len(gain),
                delta=float(gain.mean()), mc_se=float(gain.std(ddof=1)/np.sqrt(len(gain))),
                observed_late_accuracy=float(observed[subject])))
        axes[0, column].set(xticks=[0,1], xticklabels=ARMS, ylim=(0,.67),
                            title=f'{chr(97+column)}  Subject {subject}')
        axes[1, column].set(xticks=[0,1,3,4], xticklabels=['Saved','Target','Saved','Target'],
            ylim=(0,1.03), title=f'{chr(100+column)}  Same states; n={len(wide)} per arm')
        axes[1, column].axhline(observed[subject], color='#555555', ls='--', lw=.8)
        for a, arm in enumerate(ARMS):
            axes[1, column].text(a*3+.5, -.18, arm, ha='center', va='top', fontsize=7,
                                transform=axes[1, column].get_xaxis_transform())
    axes[0,0].set_ylabel('Expected error probability')
    axes[1,0].set_ylabel('Expected correct probability')
    fig.suptitle('Why do autonomous paths still make late errors?', y=.97, fontsize=9)
    fig.legend(handles=[Patch(facecolor=color, label=label) for _,label,color in CHANNELS],
               loc='upper center', bbox_to_anchor=(.5,.93), ncol=3, fontsize=7)
    fig.text(.09,.035, 'Last 128 trials. Thin lines: all paths. Bars: 1 simulation SE. Dashed line: observed accuracy.', fontsize=6)
    fig.text(.09,.010, 'Target: use the correct rule alone when present; preserve the original response distribution when absent.', fontsize=6)
    fig.savefig(output/'readout_channels.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(gains).to_csv(output/'instantaneous_gains.csv', index=False)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>规则读出诊断</title>'
        '<h1>规则在场、作答贡献与预测精度</h1><p>上排：末128题预期错误的来源。下排：每条路径的原预测，'
        '以及在正确规则已在场时完全采用该规则的即时计算。该计算保持认知状态固定，尚未改变后续反馈与学习。</p>'
        '<p>各组全部自主路径均显示，路径数量见图和表；误差条为整条路径均值的模拟标准误。横虚线为被试末128题正确率。'
        '原参数只使用行为训练段估计；本图用于开发诊断。</p><img src="readout_channels.png" style="max-width:100%">'
        +pd.DataFrame(gains).to_html(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == '__main__':
    main()
