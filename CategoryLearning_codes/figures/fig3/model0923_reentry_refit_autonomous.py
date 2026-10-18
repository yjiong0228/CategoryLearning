"""Show all S208 paths after refitting retained rule precision.

Use --source COMPLETED_AUTONOMOUS_PART --validation INDEPENDENT_AUDIT
--output NEW_DIRECTORY. The source contains original RESET and refitted RETAIN
for POOL/SEARCH. All 128 paths per cell and all late accuracies are displayed.
Conditional choice filtering and oral evaluation can finish separately.
Exports are discussion PNGs at 183 mm and 300 dpi, with source tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def plot(source: Path, validation: Path, output: Path) -> None:
    """Preserve every matched path and require its independent score audit."""
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    manifest = read(source / 'manifest.json')
    audit = read(validation / 'validation.json')
    if (manifest['status'] != 'complete_autonomous_part' or manifest['subject'] != 208
            or audit['status'] != 'complete' or audit['paths'] != 512):
        raise ValueError('Need the complete four-cell autonomous comparison and audit')
    inputs = {}
    for record in (manifest, audit):
        for field in ('inputs_sha256', 'source_sha256'):
            for name, expected in record[field].items():
                if digest(Path(name)) != expected:
                    raise ValueError('Source or independent audit input changed: ' + name)
                if name in inputs and inputs[name] != expected:
                    raise ValueError('Source and audit disagree')
                inputs[name] = expected
    for directory, name in ((source, 'manifest.json'), (validation, 'validation.json')):
        inputs[str((directory / name).resolve())] = digest(directory / name)
    for name, expected in manifest['artifacts'].items():
        file = source / name
        if digest(file) != expected or audit['inputs_sha256'].get(str(file.resolve())) != expected:
            raise ValueError('Unaudited plot table')
    tables = {name: pd.read_csv(source / (name + '.csv'))
              for name in ('scores', 'paths', 'path_blocks', 'contrasts')}
    scores, paths, blocks = (tables[name] for name in ('scores', 'paths', 'path_blocks'))
    if (len(scores) != 12 or len(paths) != 512 or len(blocks) != 9216
            or blocks.duplicated(['variant', 'model', 'seed', 'trial']).any()):
        raise ValueError('Missing or duplicate score, path or trial-block rows')
    cells = {(variant, arm) for variant in ('RESET', 'RETAIN') for arm in ('POOL', 'SEARCH')}
    if set(zip(paths.variant, paths.model)) != cells:
        raise ValueError('Need every original and refitted proposal cell')
    contract = dict(
        conclusion='Refitted precision retention leaves a substantial late learning gap in S208.',
        evidence='All autonomous trajectories, their means, and all late-accuracy empirical distributions.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=145,
        dpi=300, formats=['PNG'], participant_n=1, computational_paths_per_cell=128,
        panels='Rows: POOL/SEARCH; columns: RESET paths, RETAIN paths, late-accuracy distributions.',
        mapping='path_blocks: seed -> curve, trial -> x, accuracy -> y, observed -> participant; paths: late_accuracy -> ECDF.',
        center='Arithmetic mean of all realized path accuracies in nonoverlapping 32-trial blocks.',
        uncertainty='All fixed-parameter path variation; no parameter or population uncertainty.',
        comparison='RETAIN includes mechanism change and behavioral parameter re-estimation; matching path seeds.',
        boundary='First 384 of 576 choices estimate parameters; all paths generate their own choices and feedback from task start.',
        sample_role='Development participant; oral excluded; conditional choice scores evaluated separately.',
        hypothesis_tests='None', multiple_comparison_correction='Not applicable',
        exclusions='None: all 512 paths and 9216 block rows retained.',
        adaptation='Style-only inheritance from existing Fig3 trajectories; new data and provenance checks.',
        export_exceptions={'EXPORT-VECTOR': 'Repository PNG-only instruction',
                           'EXPORT-RASTER': 'Repository PNG-only instruction',
                           'RASTER-DPI': '300 dpi discussion figure'})
    output.mkdir(parents=True)
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    for name in tables:
        shutil.copy2(source / (name + '.csv'), output / (name + '.csv'))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
        'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = {'RESET': '#68778B', 'RETAIN': '#AA7655'}
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 145 / 25.4),
                             gridspec_kw={'width_ratios': [1, 1, .83]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.14, top=.84, wspace=.38, hspace=.55)
    for row, arm in enumerate(('POOL', 'SEARCH')):
        for col, variant in enumerate(('RESET', 'RETAIN')):
            group = blocks[(blocks.model == arm) & (blocks.variant == variant)]
            matrix = group.pivot(index='seed', columns='trial', values='accuracy').sort_index()
            observed = group.groupby('trial')['observed'].first().sort_index()
            if (matrix.shape != (128, 18) or not np.array_equal(matrix.columns, np.arange(32, 577, 32))
                    or (group.groupby('trial')['observed'].nunique() != 1).any()
                    or not (group.training_trials == 384).all()):
                raise ValueError('Invalid full-trajectory block geometry')
            data = matrix.to_numpy()
            if not np.isfinite(data).all() or np.any((data < 0) | (data > 1)):
                raise ValueError('Invalid realized path accuracy')
            ax = axes[row, col]
            ax.plot(matrix.columns, data.T, color=colors[variant], alpha=.12, lw=.35)
            ax.axvline(384, color='#777777', ls=':', lw=.8)
            ax.plot(matrix.columns, data.mean(0), color=colors[variant], lw=1.6)
            ax.plot(observed.index, observed.to_numpy(), color='#181818', lw=1., ls='--')
            own = scores[(scores.model == arm) & (scores.variant == variant) & (scores.phase == 'whole')].iloc[0]
            ax.text(.03, .045, f'Energy {own.energy:.3f}\nLate rule contribution {own.target_late_contribution:.3f}',
                    transform=ax.transAxes, fontsize=6,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1.5))
            ax.set(title=f'{arm} | {variant}', xlim=(0, 576), ylim=(0, 1.02),
                   xticks=[0, 192, 384, 576], yticks=[0, .5, 1], xlabel='Trial')
            if col == 0:
                ax.set_ylabel('32-trial accuracy')
        ax = axes[row, 2]
        for variant in ('RESET', 'RETAIN'):
            own = paths[(paths.model == arm) & (paths.variant == variant)].sort_values('seed')
            if len(own) != 128 or own.seed.nunique() != 128:
                raise ValueError('Missing full-path late accuracies')
            ax.step(np.sort(own.late_accuracy.to_numpy()), np.arange(1, 129) / 128,
                    where='post', color=colors[variant], lw=1.2, label=variant)
        human = scores[(scores.model == arm) & (scores.phase == 'whole')].observed_late
        if human.nunique() != 1:
            raise ValueError('Observed late accuracy changed between variants')
        ax.axvline(human.iloc[0], color='#181818', ls='--', lw=1.)
        ax.set(title=f'{arm} | final 128 trials', xlim=(0, 1.02), ylim=(0, 1.03),
               xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel='Accuracy', ylabel='Path fraction ≤ x')
        ax.legend(loc='upper left', frameon=False)
    for i, ax in enumerate(axes.flat):
        ax.text(-.17, 1.05, chr(97 + i), transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', clip_on=False)
    fig.suptitle('Participant 208: autonomous learning after precision-retention refitting', fontsize=9, y=.97)
    fig.text(.5, .92, 'Every 128-path distribution is retained; behavioral parameters were estimated from the first 384 choices.',
             ha='center', fontsize=6.5)
    handles = [Line2D([], [], color='#555555', lw=.5, label='Individual paths'),
               Line2D([], [], color=colors['RETAIN'], lw=1.6, label='Generated mean'),
               Line2D([], [], color='#181818', ls='--', lw=1, label='Participant'),
               Line2D([], [], color='#777777', ls=':', lw=.8, label='Fit boundary')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .055))
    fig.text(.5, .025, 'One development participant; 512 paths in total. RETAIN changes both storage and fitted parameters.',
             ha='center', fontsize=6.5)
    fig.savefig(output / 'autonomous_paths.png', dpi=300)
    plt.close(fig)
    whole = scores[scores.phase == 'whole']
    caption = ('每格显示全部128条自主路径，细线是各条路径，粗线是均值，黑色虚线是被试。'
        '竖点线标出前384题的选参边界；右列保留每条路径的末128题正确率。'
        '两轮拟合均只使用前段选择，差值包含精度保持和重新估参的共同作用。'
        '每条路径在原刺激顺序上按自身回答获得反馈。该被试用于开发，口述未进入选参。'
        '条件选择评分与口述检验分别汇总。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>208精度保持重拟合自主路径</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}td,th{padding:6px}</style>'
        '<h1>208：精度保持重新估参后的自主学习</h1><p>' + caption + '</p>'
        '<img src="autonomous_paths.png" alt="全部自主路径与末段正确率分布">'
        + whole[['variant', 'model', 'energy', 'late_accuracy', 'observed_late',
                 'target_late_present', 'target_late_contribution']].to_html(index=False, float_format=lambda x: f'{x:.4f}')
        + '<p><a href="path_blocks.csv">全部路径源表</a> · <a href="contrasts.csv">分段Energy与CRPS差值及配对模拟误差</a></p></html>')
    (output / 'index.html').write_text(page)
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('Figure input changed during rendering')
    shutil.copy2(__file__, output / 'plot_source.py')
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', input_sha256=inputs,
        source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))},
        artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'validation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.validation, args.output)
