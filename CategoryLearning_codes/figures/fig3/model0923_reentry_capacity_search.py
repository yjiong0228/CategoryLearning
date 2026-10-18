"""Display every path in the fixed RETAIN capacity x global-proposal crossing.

Use --source SUMMARY --validation INDEPENDENT_AUDIT --output NEW_DIRECTORY.
Rows show POOL/SEARCH; columns show first correct-rule appearance, late rule
contribution and late accuracy. Every cell contains its 128 actual path values.
Horizontal displacement separates points without changing their measurements.
PNG output is 183 x 125 mm at 300 dpi, with all source tables and provenance.
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
import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(source: Path, validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new capacity-search figure destination')
    inputs = {}
    records = []
    for folder, name in ((source, 'manifest.json'), (validation, 'validation.json')):
        path = folder / name
        record = json.loads(path.read_text())
        inputs[str(path.resolve())] = digest(path)
        for field in ('inputs_sha256', 'source_sha256'):
            for name, expected in record[field].items():
                if digest(Path(name)) != expected:
                    raise ValueError('Changed input: ' + name)
                inputs[name] = expected
        records.append(record)
    if any(record['status'] != 'complete' or record['paths'] != 1024 for record in records):
        raise ValueError('Require the full independently checked eight-cell crossing')
    tables = {}
    for name, sha in records[0]['artifacts'].items():
        path = source / name
        if digest(path) != sha or records[1]['inputs_sha256'].get(str(path.resolve())) != sha:
            raise ValueError('Unaudited table: ' + name)
        inputs[str(path.resolve())] = sha
        tables[path.stem] = pd.read_csv(path)
    paths, scores = tables['paths'], tables['scores']
    if len(paths) != 1024 or len(scores) != 24 or not paths.target_ever.all():
        raise ValueError('Need all paths and explicitly observed discovery times')
    contract = dict(
        conclusion='At these S208 parameters, higher capacity reduces correct-rule use; lower global proposal probability advances discovery.',
        evidence='Every path discovery time, late contribution and late accuracy in both proposal arms.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=125, dpi=300, formats=['PNG'],
        mapping='capacity -> x; global_proposal -> color and offset; path metric -> y; model -> row.',
        participant_n=1, computational_paths_per_cell=128, cells=8,
        center='Arithmetic mean of all 128 path values; late means use final128trials.',
        uncertainty='Error bars: +/- one Monte Carlo SE of path mean; points: all actual path values.',
        split='Original 384/192 task boundary retained; fixed-parameter intervention across all576trials.',
        scope='Development; capacity4/6 and baseline global proposal.2/.8 crossed; other12parameters fixed; RETAIN everywhere.',
        estimation='No refitting; original behavioral nominees unchanged; oral excluded.',
        exclusions='None: all1024paths used for each plotted metric; all18432curve points retained in source table.',
        transformations='Deterministic horizontal displacement only; all vertical coordinates unchanged.',
        hypothesis_tests='None', multiple_comparison_correction='Not applicable',
        adaptation='Style-only inheritance from Fig3; factorial process comparison and all-path point displays.',
        export_exceptions={'EXPORT-VECTOR': 'Repository PNG-only instruction',
                           'EXPORT-RASTER': 'Repository PNG-only instruction',
                           'RASTER-DPI': '300 dpi discussion output'})
    output.mkdir(parents=True)
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    for name in tables:
        shutil.copy2(source / (name + '.csv'), output / (name + '.csv'))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6,
        'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = {0.2: '#AA7655', 0.8: '#68778B'}
    metrics = [('first_target_or_after_end', 'First correct-rule trial', (0, 576)),
               ('target_late_contribution', 'Late correct-rule contribution', (0, 1)),
               ('late_accuracy', 'Late accuracy', (0, 1))]
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 125 / 25.4))
    fig.subplots_adjust(left=.085, right=.98, bottom=.18, top=.83, hspace=.60, wspace=.40)
    plot_rows = []
    for row, arm in enumerate(('POOL', 'SEARCH')):
        for col, (metric, label, limits) in enumerate(metrics):
            ax = axes[row, col]
            for g, offset in ((.2, -.10), (.8, .10)):
                means, errors = [], []
                for x, capacity in enumerate((4, 6)):
                    own = paths[(paths.model == arm) & (paths.capacity == capacity)
                                & (paths.global_proposal == g)].sort_values('seed')
                    values = own[metric].to_numpy(float)
                    if (len(values) != 128 or own.seed.nunique() != 128 or not np.isfinite(values).all()
                            or np.any((values < limits[0]) | (values > limits[1]))):
                        raise ValueError('Incomplete or out-of-axis path values')
                    mean, error = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
                    means.append(mean); errors.append(error)
                    ax.scatter(x + offset + np.linspace(-.055, .055, len(values)), values,
                               color=colors[g], s=2, alpha=.23, linewidths=0)
                    plot_rows.append(dict(model=arm,capacity=capacity,global_proposal=g,metric=metric,
                                          mean=mean,mc_se=error,paths=len(values)))
                ax.errorbar(np.arange(2) + offset, means, yerr=errors, color=colors[g],
                            fmt='o-', markersize=3, lw=1.1, capsize=2, label=f'Global proposal g={g:.1f}')
            if metric == 'late_accuracy':
                human = scores[(scores.model == arm) & (scores.phase == 'whole')].observed_late
                if human.nunique() != 1:
                    raise ValueError('Observed accuracy changed between cells')
                ax.axhline(human.iloc[0], color='#222222', ls='--', lw=.8)
                ax.text(.97, .94, 'Participant', ha='right', transform=ax.transAxes, fontsize=6)
            ax.set(title=f'{arm} | ' + ['discovery', 'rule use', 'performance'][col],
                   ylabel=label, xlabel='Capacity', xticks=[0, 1], xticklabels=['4', '6'],
                   xlim=(-.35, 1.35), ylim=limits)
    for i, ax in enumerate(axes.flat):
        ax.text(-.10, 1.04, chr(97+i), transform=ax.transAxes, fontsize=8, fontweight='bold')
    fig.suptitle('Participant 208: separate effects of capacity and global proposal probability', fontsize=9, y=.97)
    fig.text(.5, .92, 'All other 12 parameters are fixed; every rule retains its last active precision.', ha='center', fontsize=6.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .065))
    fig.text(.5, .025, 'All 128 paths per cell are shown. Lines: means; bars: one Monte Carlo SE. Late = final 128 trials.',
             ha='center', fontsize=6.5)
    fig.savefig(output / 'capacity_search.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(output / 'plot_means.csv', index=False)
    caption = ('固定其余12个参数，交叉改变容量4／6与基础全局提议概率0.2／0.8，两种提议方案都保持规则精度。'
        '每格全部128条路径均以细点显示，横向位移仅用于减少重叠；实线为均值，误差线为一个模拟标准误。'
        '左列为正确规则首次在场题号，中列和右列为末128题的贡献和正确率。'
        '全部1024条路径均曾发现正确规则。原行为选参结果保持冻结，口述未参与本轮比较。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>容量与全局提议的交叉对照</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}td,th{padding:6px}</style>'
        '<h1>208：容量与全局提议分别影响什么</h1><p>' + caption + '</p>'
        '<img src="capacity_search.png" alt="全部路径的发现、贡献和正确率交叉对照">'
        + scores[scores.phase == 'whole'][['model','capacity','global_proposal','energy','late_accuracy',
            'target_late_present','target_late_contribution','active_rules_late']].to_html(index=False,float_format=lambda x:f'{x:.4f}')
        + '<p><a href="effects.csv">Energy和CRPS的配对效应</a> · <a href="path_effects.csv">过程指标的配对效应</a></p></html>')
    (output / 'index.html').write_text(page)
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError('Source changed during plotting')
    shutil.copy2(__file__, output / 'plot_source.py')
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete',input_sha256=inputs,
        source_sha256={str(Path(__file__).resolve()):digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','validation','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    plot(args.source,args.validation,args.output)
