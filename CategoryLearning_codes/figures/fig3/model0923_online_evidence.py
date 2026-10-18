"""Show complete autonomous paths and paired effects of evidence-based belief.

Use --source SUMMARY --validation INDEPENDENT_AUDIT --output NEW_DIRECTORY.
The trajectory panels show every baseline, common-recall and online-evidence
path in the four fixed parameter cells. ENTRY remains in the source tables;
the figure isolates the two every-trial alternatives. PNG: 183 x 125 mm, 300 dpi.
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


def plot(source: Path, validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new online-evidence figure destination')
    inputs, records = {}, []
    for folder, name in ((source, 'manifest.json'), (validation, 'validation.json')):
        path = folder / name
        record = json.loads(path.read_text())
        inputs[str(path.resolve())] = digest(path)
        for field in ('inputs_sha256', 'source_sha256'):
            for name, expected in record[field].items():
                if digest(Path(name)) != expected:
                    raise ValueError('Changed bound input: ' + name)
                inputs[name] = expected
        records.append(record)
    if any(r['status'] != 'complete' or r['paths'] != 1024 or r['cells'] != 16 for r in records):
        raise ValueError('Require the complete independently checked sixteen-cell comparison')
    tables = {}
    for name, expected in records[0]['artifacts'].items():
        path = source / name
        if digest(path) != expected or records[1]['inputs_sha256'].get(str(path.resolve())) != expected:
            raise ValueError('Unaudited table: ' + name)
        inputs[str(path.resolve())] = expected
        tables[path.stem] = pd.read_csv(path)
    contract = dict(
        conclusion='At fixed S208 parameters, existing online H evidence increases correct-rule use and late autonomous accuracy.',
        evidence='Four panels show all paths through learning; paired Energy and contribution effects quantify the change.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=125, dpi=300, formats=['PNG'],
        mapping='trial -> x,32-trial accuracy -> y,seed -> faint path,schedule -> color,model/capacity -> panel.',
        participant_n=1, computational_paths_per_cell=64, displayed_cells=12,
        center='Arithmetic means over all64paths; error bars are one paired whole-path Monte Carlo SE.',
        split='Original384/192 chronological boundary; all576trials generated autonomously; development subject.',
        scope='RETAIN, global proposal.2, capacity4/6, other12parameters fixed; no refit or oral input.',
        exclusions='Display BASE/common-every-trial/online-every-trial: all768paths. ENTRY256paths remain in all source tables.',
        transformations='Non-overlapping32-trial accuracy blocks. No smoothing, seed sampling or path selection.',
        hypothesis_tests='None', multiple_comparison_correction='Not applicable',
        adaptation='Style-only inheritance from Fig3; new all-path trajectories and paired contrasts.',
        export_exceptions={'EXPORT-VECTOR':'Repository PNG-only instruction',
                           'EXPORT-RASTER':'Repository PNG-only instruction',
                           'RASTER-DPI':'300dpi discussion output'})
    output.mkdir(parents=True)
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    for name in tables:
        shutil.copy2(source / (name + '.csv'), output / (name + '.csv'))
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':7, 'axes.titlesize':7, 'axes.labelsize':7, 'xtick.labelsize':6,
        'ytick.labelsize':6, 'legend.fontsize':6, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.6, 'svg.fonttype':'none', 'pdf.fonttype':42})
    colors = {'off':'#68778B', 'every_trial':'#B27B58', 'online':'#257B89'}
    labels = {'off':'BASE', 'every_trial':'Common recall', 'online':'Online evidence'}
    styles = {'off':'--', 'every_trial':':', 'online':'-'}
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 125 / 25.4))
    fig.subplots_adjust(left=.075, right=.975, bottom=.19, top=.83, hspace=.60, wspace=.55)
    plot_rows = []
    for row, arm in enumerate(('POOL', 'SEARCH')):
        for col, capacity in enumerate((4, 6)):
            ax = axes[row, col]
            own = tables['path_blocks'].query('model==@arm and capacity==@capacity')
            observed = own.groupby('trial')['observed'].agg(['min', 'max'])
            np.testing.assert_array_equal(observed['min'], observed['max'])
            for schedule, color in colors.items():
                points = own[own.schedule == schedule].pivot(index='seed', columns='trial', values='accuracy')
                values = points.to_numpy(float)
                if values.shape != (64, 18) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError('Incomplete or invalid complete-path display')
                ax.plot(points.columns, values.T, color=color, alpha=.045, lw=.4)
                ax.plot(points.columns, values.mean(0), color=color, ls=styles[schedule], lw=1.3)
                plot_rows.extend(dict(model=arm,capacity=capacity,schedule=schedule,trial=int(t),mean=float(m),paths=64)
                                 for t, m in zip(points.columns, values.mean(0)))
            ax.plot(observed.index, observed['min'], color='#222222', lw=1.0)
            ax.axvline(384, color='#999999', lw=.6, ls='--')
            ax.set(title=f'{arm} | capacity {capacity}', ylabel='Accuracy', xlabel='Trial',
                   xlim=(0, 576), ylim=(0, 1), xticks=[0, 192, 384, 576])
    cells = [('POOL',4), ('POOL',6), ('SEARCH',4), ('SEARCH',6)]
    effect_rows = []
    for row, (table, metric, error, label) in enumerate((
            (tables['effects'].query('phase=="whole"'), 'delta_energy', 'delta_energy_mc_se', 'Energy difference'),
            (tables['path_effects'].query('metric=="target_late_contribution"'), 'delta', 'paired_mc_se', 'Late contribution difference'))):
        ax = axes[row, 2]
        for schedule, offset, marker in [('every_trial', -.12, '^'), ('online', .12, 'o')]:
            means, errors = [], []
            for arm, capacity in cells:
                part = table.query('model==@arm and capacity==@capacity and current==@schedule and reference=="off"')
                if len(part) != 1:
                    raise ValueError('Missing paired fixed-parameter contrast')
                m, e = float(part.iloc[0][metric]), float(part.iloc[0][error])
                means.append(m); errors.append(e)
                effect_rows.append(dict(model=arm,capacity=capacity,schedule=schedule,metric=label,mean=m,mc_se=e))
            ax.errorbar(means, np.arange(4) + offset, xerr=errors, fmt=marker, color=colors[schedule],
                        markersize=3, lw=.8, capsize=2)
        ax.axvline(0, color='#999999', ls='--', lw=.6)
        ax.set(yticks=np.arange(4), yticklabels=[f'{arm} / {k}' for arm,k in cells],
               xlabel=label, title='Difference from BASE', ylim=(3.5,-.5))
    for i, ax in enumerate(axes.flat):
        ax.text(-.10, 1.06, chr(97+i), transform=ax.transAxes, fontsize=8, fontweight='bold')
    fig.suptitle('Participant 208: using accumulated evidence to sustain rule use', fontsize=9, y=.97)
    fig.text(.5,.92,'Fixed parameters; independent autonomous choices and feedback; retained rule precision.',ha='center',fontsize=6.5)
    handles = [Line2D([],[],color=colors[s],ls=styles[s],lw=1.3,label=labels[s]) for s in colors]
    handles.append(Line2D([],[],color='#222222',lw=1,label='Participant'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5,.07))
    fig.text(.5,.025,'Faint lines: all 64 paths per cell. Bold lines: means. Bars: paired Monte Carlo SE. Late = final 128 trials.',
             ha='center',fontsize=6.3)
    fig.savefig(output / 'online_evidence.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(output / 'plot_curves.csv',index=False)
    pd.DataFrame(effect_rows).to_csv(output / 'plot_effects.csv',index=False)
    caption = ('每格保留相同64个随机种子，背景显示全部路径，粗线显示均值，黑线为被试。每点为32题正确率。'
        'BASE沿用原信念；Common recall每题按初始精度回顾历史；Online evidence使用H已有的逐题证据。'
        '右列为相对BASE的全程Energy差和末128题正确规则贡献差，误差线为配对完整路径的一个模拟标准误。'
        'Energy越低越好。前384题与后192题的分界仅表示时间；该被试已经用于机制开发。'
        '表格另保留仅入场回顾的ENTRY对照。参数继续冻结，口述未进入本轮计算。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>连续证据与规则使用</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}td,th{padding:6px}</style>'
        '<h1>208：让已有证据影响规则使用</h1><p>' + caption + '</p>'
        '<img src="online_evidence.png" alt="全部自主路径及证据使用的配对效果">'
        + tables['scores'].query('phase=="whole"')[['model','capacity','schedule','energy','late_accuracy',
            'target_late_present','target_late_contribution','late_reaches_observed']].to_html(index=False,float_format=lambda x:f'{x:.4f}')
        + '<p><a href="effects.csv">分时段生成评分</a> · <a href="path_effects.csv">配对过程指标</a></p></html>')
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
    args = parser.parse_args()
    plot(args.source,args.validation,args.output)
