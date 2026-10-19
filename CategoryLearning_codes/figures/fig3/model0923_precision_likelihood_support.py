"""Plot every fixed-H S107 precision candidate and its point likelihood support.

Contract: show how the nominated precision coordinates compare with alternatives
while retaining the distinction between numerical ranking and parameter support.
Quantitative grid; Python/matplotlib only; 183 x 154 mm; PNG at300dpi for review.
All80 original screen points and eight confirmed points are displayed. Heatmap
values use the original two-seed,128-particle training screen; outlines identify
points checked at four seeds,512particles. Confirmed ratios are reported below.
No uncertainty interval or population inference is attached to these grids.
Relative likelihood is scaled to the best point within each cell, not to a sum
of one. H/memory coordinates remain fixed; other parameter uncertainty is open.

Run --source COMPLETED_SUPPORT_AUDIT --output NEW_DIRECTORY.
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
import numpy as np
import pandas as pd


CELLS = ('MIX_ADD10', 'MIX_LOG25', 'SELECT_ADD10', 'SELECT_LOG25')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new precision support figure directory')
    done = json.loads((source / 'completion.json').read_text())
    if done['status'] != 'complete':
        raise ValueError('Require the completed point likelihood audit')
    inputs = {str((source / 'completion.json').resolve()): digest(source / 'completion.json'),
              str(Path(__file__).resolve()): digest(Path(__file__))}
    for name, sha in done['artifacts'].items():
        if digest(source / name) != sha:
            raise ValueError('Changed support table')
        inputs[str((source / name).resolve())] = sha
    table = pd.read_csv(source / 'point_support.csv')
    families = pd.read_csv(source / 'family_contrasts.csv')
    screen = table[table.stage == 'screen']
    confirmed = table[table.stage == 'confirmation']
    if (len(screen) != 80 or len(confirmed) != 8 or set(table.cell) != set(CELLS)
            or not table.training_trials.eq(128).all() or table.duplicated(['stage', 'cell', 'candidate']).any()
            or not np.isfinite(table.select_dtypes('number').to_numpy()).all()):
        raise ValueError('Candidate coverage or original boundary changed')
    np.testing.assert_allclose(table.relative_sequence_likelihood, np.exp(-table.delta_total_nll), atol=1e-14, rtol=0)
    if not table.relative_sequence_likelihood.between(0, 1).all():
        raise ValueError('Relative point likelihood lies outside its range')
    initials = [.1, .5, 1., 5.]
    rates = [.01, .03, .1, .3, 1.]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
        'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2047244, 6.0629921))  # 183 x 154 mm
    fig.subplots_adjust(left=.11, right=.88, bottom=.26, top=.81, wspace=.35, hspace=.55)
    ratios = {}
    for label, cell, ax in zip('abcd', CELLS, axes.flat):
        group = screen[screen.cell == cell]
        if len(group) != 20 or set(group.beta_init) != set(initials) or set(group.configured_rate) != set(rates):
            raise ValueError('Declared common precision grid changed')
        matrix = group.pivot(index='beta_init', columns='configured_rate', values='relative_sequence_likelihood').loc[initials, rates]
        im = ax.imshow(matrix.to_numpy(), cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        selected = group[group.final_nominee]
        local = confirmed[confirmed.cell == cell]
        if len(selected) != 1 or len(local) != 2 or local.final_nominee.sum() != 1:
            raise ValueError('Missing unique frozen nominee or confirmed alternative')
        ratios[cell] = float(local.loc[~local.final_nominee, 'relative_sequence_likelihood'].iloc[0])
        for y, initial in enumerate(initials):
            for x, rate in enumerate(rates):
                value = matrix.loc[initial, rate]
                text = '<0.001' if value < .001 else f'{value:.3f}'
                row = group[(group.beta_init == initial) & (group.configured_rate == rate)].iloc[0]
                if row.final_nominee:
                    text += '*'
                ax.text(x, y, text, ha='center', va='center', fontsize=6.5,
                        color='white' if value > .5 else '#333333')
                if row.candidate in set(local.candidate):
                    ax.add_patch(plt.Rectangle((x-.48, y-.48), .96, .96, fill=False,
                                               edgecolor='#222222', linewidth=1.1))
        ax.set(xticks=np.arange(5), xticklabels=[f'{x:g}' for x in rates],
               yticks=np.arange(4), yticklabels=[f'{x:g}' for x in initials],
               xlabel='配置学习率', ylabel='精度初值')
        ax.set_title(cell.replace('_', ' · '), loc='left', fontsize=8, pad=9)
        ax.text(-.22, 1.09, label, transform=ax.transAxes, weight='bold', fontsize=9)
    colorbar = fig.colorbar(im, cax=fig.add_axes([.915, .30, .018, .51]))
    colorbar.set_label('相对训练序列似然', fontsize=7)
    fig.suptitle('107：固定H后，各精度参数点如何胜出', x=.11, ha='left', y=.97, fontsize=10)
    fig.text(.11, .91, '每格保留全部20点；相对似然 = exp[−128 ×（本点平均NLL − 该格最小平均NLL）]', fontsize=7)
    fig.text(.11, .875, '颜色与数字：128粒子 × 2种子的初筛；黑框：512粒子 × 4种子确认；*：最终提名', fontsize=6.5)
    line = '；'.join(f"{cell.replace('_', ' ')} {ratios[cell]:.3f}" for cell in CELLS)
    whole = families[families.window == 'whole'].set_index('policy')
    family_line = '；'.join(f"{p} {whole.loc[p, 'sequence_likelihood_ratio_LOG25_to_ADD10']:.2f}∶1" for p in ('MIX', 'SELECT'))
    fig.text(.11, .17, '高粒子确认后，次优点／提名点的训练序列似然：\n' + line + '\n'
             '冻结参数后，全192题的LOG25／ADD10近似点似然比：' + family_line + '\n'
             'ADD10的有效加法学习率为配置值的10倍；LOG25使用原配置学习率。\n'
             '初筛小于0.001的数值使用统一文字精度，所有原值保存在源表。\n'
             '这些数值比较固定参数点；参数后验、机制证据和联合恢复需要各自的估计。', fontsize=6.2, va='top')
    output.mkdir(parents=True)
    figure = output / 'Figure3_s107_precision_likelihood_support.png'
    fig.savefig(figure, dpi=300, facecolor='white')
    plt.close(fig)
    for name in done['artifacts']:
        (output / name).write_bytes((source / name).read_bytes())
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Plot source changed')
    record = dict(status='complete', figure_contract=__doc__, inputs_sha256=inputs,
        screen_points=80, confirmed_points=8, exclusions='none within the original precision study',
        confirmed_relative_likelihood=ratios, reported_family_ratio='Independent frozen evaluation; whole192 is development data',
        formatting='Values below0.001 use text<0.001; color retains the original value on0to1 scale',
        export_exceptions='Repository PNG-only instruction; research-review draft at300dpi',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
        artifacts={path.name: digest(path) for path in output.iterdir() if path.is_file()})
    (output / 'manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
