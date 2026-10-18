"""Show behavioral fit and autonomous sensitivity of frozen RETAIN finalists.

Use --source SUMMARY --validation INDEPENDENT_AUDIT --confirmation DIAGNOSIS
--output NEW_DIRECTORY. Panels retain both finalists in each proposal arm,
all 512 autonomous paths in the late-accuracy distributions, and all 384 choices
in the training-loss decomposition. Export: 183 x 125 mm discussion PNG, tables,
figure contract, caption and provenance. No parameter nominations are changed.
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


def plot(source: Path, validation: Path, confirmation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new finalist figure output')
    inputs = {}
    records = []
    for folder, name in ((source, 'manifest.json'), (validation, 'validation.json'),
                         (confirmation, 'manifest.json')):
        file = folder / name
        record = json.loads(file.read_text())
        inputs[str(file.resolve())] = digest(file)
        for field in ('inputs_sha256', 'input_sha256', 'source_sha256'):
            for path, expected in record.get(field, {}).items():
                if digest(Path(path)) != expected:
                    raise ValueError('Changed source: ' + path)
                inputs[path] = expected
        records.append(record)
    if (records[0]['status'] != 'complete' or records[1]['status'] != 'complete'
            or records[1]['paths'] != 512 or not records[1]['selected_reproduces_previous']):
        raise ValueError('Require complete independently checked finalist results')
    tables = {}
    for name in ('scores', 'paths', 'path_blocks', 'contrasts', 'path_contrasts'):
        file = source / (name + '.csv')
        sha = digest(file)
        if (records[0]['artifacts'][file.name] != sha
                or records[1]['inputs_sha256'].get(str(file.resolve())) != sha):
            raise ValueError('Unaudited source table')
        inputs[str(file.resolve())] = sha
        tables[name] = pd.read_csv(file)
    loss_file = confirmation / 'contrasts.csv'
    if records[2]['artifacts']['contrasts.csv'] != digest(loss_file):
        raise ValueError('Changed confirmation contrast table')
    inputs[str(loss_file.resolve())] = digest(loss_file)
    loss = pd.read_csv(loss_file).set_index('model')
    scores = tables['scores']; paths = tables['paths']
    if len(scores) != 12 or len(paths) != 512 or len(tables['path_blocks']) != 9216:
        raise ValueError('Incomplete finalist bank')
    contract = dict(
        conclusion='The training-choice winner generates poorer late learning than its other frozen finalist in S208.',
        evidence='Training loss by actual response; full-trajectory distribution score; every late path accuracy.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=125,
        formats=['PNG'], dpi=300, participant_n=1, paths_per_cell=128,
        panels='Rows: POOL and SEARCH; columns: selected-minus-alternative training NLL, full-path Energy, late-accuracy ECDF.',
        mapping='confirmation correct/incorrect/total loss -> bars; scores energy/energy_mc_se -> point/error bar; paths late_accuracy -> ECDF.',
        center='Empirical equal-weight path distribution; cumulative NLL on 384 actual choices.',
        uncertainty='NLL total: +/- one paired whole-filter Monte Carlo SE (8 seeds); Energy: +/- one whole-path Monte Carlo SE; ECDF: all 128 fixed-parameter paths.',
        training='First 384 of 576 choices; eight confirmation filter seeds; oral excluded.',
        interpretation='Development parameter-point sensitivity; candidate bank is not a posterior distribution.',
        comparison='Same RETAIN mechanism and paired seeds; capacity 6/global proposal .2 versus capacity 4/global proposal .8.',
        exclusions='None: all four cells, all 512 paths, all 384 training choices retained.',
        hypothesis_tests='None', multiple_comparison_correction='Not applicable',
        adaptation='Style-only inheritance from existing Fig3 diagnostics; new panel meanings and source checks.',
        export_exceptions={'EXPORT-VECTOR': 'Repository PNG-only instruction',
                           'EXPORT-RASTER': 'Repository PNG-only instruction',
                           'RASTER-DPI': '300 dpi discussion output'})
    output.mkdir(parents=True)
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    for name in tables:
        shutil.copy2(source / (name + '.csv'), output / (name + '.csv'))
    shutil.copy2(loss_file, output / 'training_contrasts.csv')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6,
        'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = {'SELECTED': '#AA7655', 'ALTERNATIVE': '#68778B'}
    labels = {'SELECTED': 'Selected: K6, g=.2', 'ALTERNATIVE': 'Alternative: K4, g=.8'}
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 125 / 25.4))
    fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.81, wspace=.52, hspace=.65)
    for row, arm in enumerate(('POOL', 'SEARCH')):
        own_loss = loss.loc[arm]
        whole = scores[(scores.model == arm) & (scores.phase == 'whole')].set_index('variant')
        if (whole.loc['SELECTED', 'candidate'] != own_loss.selected
                or whole.loc['ALTERNATIVE', 'candidate'] != own_loss.reference):
            raise ValueError('Training and generation candidates disagree')
        ax = axes[row, 0]
        values = [own_loss.correct_total_delta, own_loss.incorrect_total_delta, own_loss.total_delta]
        np.testing.assert_allclose(values[0] + values[1], values[2], atol=1e-12, rtol=0)
        ax.axhline(0, color='#888888', lw=.6)
        ax.bar([0, 1, 2], values, width=.65, color=['#AAA39C', '#768695', '#AA7655'])
        ax.errorbar(2, values[2], yerr=own_loss.total_mc_se, fmt='none', color='#222222', lw=.8, capsize=2)
        ax.set(title=f'{arm} | training choice fit', ylabel='Cumulative NLL difference',
               xticks=[0, 1, 2], xticklabels=['Correct\n245', 'Incorrect\n139', 'All\n384'], ylim=(-10, 9))
        ax.text(.04, .95, f'Estimated likelihood ratio: {own_loss.estimated_likelihood_ratio:.2f}',
                transform=ax.transAxes, va='top', fontsize=6)
        ax = axes[row, 1]
        for i, variant in enumerate(('SELECTED', 'ALTERNATIVE')):
            own = whole.loc[variant]
            ax.errorbar(i, own.energy, yerr=own.energy_mc_se, fmt='o', color=colors[variant],
                        markersize=4, lw=.9, capsize=2)
        ax.set(title=f'{arm} | whole-path generation', ylabel='Energy score (lower is better)',
               xlim=(-.5, 1.5), ylim=(.075, .115), xticks=[0, 1], xticklabels=['Selected', 'Alternative'])
        ax = axes[row, 2]
        for variant in ('SELECTED', 'ALTERNATIVE'):
            own = paths[(paths.model == arm) & (paths.variant == variant)]
            if (len(own) != 128 or own.seed.nunique() != 128
                    or not np.isfinite(own.late_accuracy).all()
                    or not own.late_accuracy.between(0, 1).all()):
                raise ValueError('Incomplete late-accuracy distribution')
            ax.step(np.sort(own.late_accuracy), np.arange(1, 129) / 128, where='post',
                    color=colors[variant], lw=1.2, label=labels[variant])
        assert whole.observed_late.nunique() == 1
        ax.axvline(whole.observed_late.iloc[0], color='#222222', ls='--', lw=.9)
        ax.set(title=f'{arm} | final 128 trials', ylabel='Path fraction ≤ x', xlabel='Accuracy',
               xlim=(0, 1), ylim=(0, 1.03), xticks=[0, .5, 1], yticks=[0, .5, 1])
        ax.text(.97, .06, 'Participant', transform=ax.transAxes, ha='right', fontsize=6)
    for i, ax in enumerate(axes.flat):
        ax.text(-.10, 1.04, chr(97 + i), transform=ax.transAxes, fontsize=8, fontweight='bold')
    fig.suptitle('Participant 208: the behavior-selected point and its other frozen finalist', fontsize=9, y=.97)
    fig.text(.5, .91, 'Both points retain precision. Negative training-loss differences favor the selected point.', ha='center', fontsize=6.5)
    handles, names = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .065))
    fig.text(.5, .025, 'One development participant; 128 paired paths per cell. Error bars show one Monte Carlo SE.', ha='center', fontsize=6.5)
    fig.savefig(output / 'finalist_sensitivity.png', dpi=300)
    plt.close(fig)
    caption = ('两行分别为POOL和SEARCH。左列将最优点减另一候选的训练累计NLL差分解到正确与错误回答；'
        '负值表示最优点更能解释真实选择，总差误差线按8个完整滤波种子配对计算。中列为全程自主轨迹的Energy，越低表示分布越接近被试；'
        '误差线为一个完整路径模拟标准误。右列保留每组全部128条路径的末128题正确率。'
        '两候选仅容量和基础全局提议概率不同，其余12个参数相同。两者均在行为确认阶段已冻结，口述未参与选参。'
        '图中似然比比较指定参数点，候选的后验概率尚未估计。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>208候选参数敏感性</title>'
            '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}td,th{padding:6px}</style>'
            '<h1>208：行为选参结果与生成过程的参数敏感性</h1><p>' + caption + '</p>'
            '<img src="finalist_sensitivity.png" alt="行为似然、自主分布与末段正确率">'
            + scores[scores.phase == 'whole'][['variant', 'model', 'energy', 'late_accuracy',
                'target_late_present', 'target_late_contribution']].to_html(index=False, float_format=lambda x: f'{x:.4f}')
            + '<p><a href="path_blocks.csv">全部路径源表</a> · <a href="contrasts.csv">配对生成差值</a></p></html>')
    (output / 'index.html').write_text(page)
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError('Figure input changed')
    shutil.copy2(__file__, output / 'plot_source.py')
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', input_sha256=inputs,
        source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))},
        artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'validation', 'confirmation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.validation, args.confirmation, args.output)
