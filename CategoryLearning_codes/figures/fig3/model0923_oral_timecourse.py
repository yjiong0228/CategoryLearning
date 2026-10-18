"""Show oral coverage, decision support and rule availability over learning.

Read a completed decision-basis alignment with --alignment and export to a new
--output directory. Fixed 32-trial display blocks preserve complete-report
denominators. Blocks without complete reports remain missing. PNG only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model0923_decision_basis import COLORS, digest


def blocks(frame: pd.DataFrame, width: int = 32) -> pd.DataFrame:
    """Keep all trials for coverage and the same complete reports for each arm."""
    rows = []
    for (subject, arm), group in frame.groupby(['iSub', 'arm'], sort=True):
        group = group.sort_values('ordinal_trial')
        np.testing.assert_array_equal(group.ordinal_trial, np.arange(1, len(group) + 1))
        cut = group.training_trials.unique()
        if len(cut) != 1 or int(cut[0]) % width:
            raise ValueError('Display blocks must preserve the frozen training boundary')
        for index, part in group.groupby((group.ordinal_trial - 1) // width, sort=True):
            complete = part[part.scope.eq('complete')]
            row = dict(subject=int(subject), arm=arm, block=int(index),
                start=int(part.ordinal_trial.min()), end=int(part.ordinal_trial.max()),
                midpoint=float(part.ordinal_trial.mean()), trials=len(part), reports=len(complete),
                catalogue_reports=int(complete.structural_support_any_component.gt(0).sum()),
                training_trials=int(cut[0]))
            for name in ['structural_support', 'structural_conflict', 'structural_undetermined',
                         'structural_availability_all_components', 'structural_availability_any_component',
                         'prefix_static_structural_support']:
                row[name] = float(complete[name].mean())
            if len(complete):
                np.testing.assert_allclose(sum(row[n] for n in ['structural_support', 'structural_conflict',
                    'structural_undetermined']), 1., atol=1e-10, rtol=0)
            rows.append(row)
    return pd.DataFrame(rows)


def plot(alignment: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral-timecourse output directory')
    manifest = json.loads((alignment / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['smoke']:
        raise ValueError('Need completed scientific oral alignment')
    if digest(alignment / 'trials.csv') != manifest['artifacts']['trials.csv']:
        raise ValueError('Aligned trials changed')
    retrospective = manifest.get('evaluation_use') == 'development_retrospective_comparison'
    estimation_scope = manifest.get('parameter_estimation_scope', 'original_evaluation_protocol')
    boundary_label = 'original time split' if retrospective else 'training boundary'
    order, reference = manifest['arms'], manifest.get('reference_model', 'BASE')
    subjects = sorted(manifest['subjects'])
    if reference not in order or not set(order) <= COLORS.keys():
        raise ValueError('Unsupported comparison')
    frame = pd.read_csv(alignment / 'trials.csv')
    if set(map(tuple, frame[['iSub', 'arm']].drop_duplicates().to_numpy())) != {(s, a) for s in subjects for a in order}:
        raise ValueError('Missing declared subject-arm groups')
    # Identity and coverage must agree before comparing any state probabilities.
    identity = ['ordinal_trial', 'choice', 'scope', 'tokens', 'training_trials']
    for subject in subjects:
        anchor = frame[frame.iSub.eq(subject) & frame.arm.eq(reference)][identity].reset_index(drop=True)
        for arm in order:
            pd.testing.assert_frame_equal(anchor,
                frame[frame.iSub.eq(subject) & frame.arm.eq(arm)][identity].reset_index(drop=True))
    source = blocks(frame)
    output.mkdir(parents=True)
    source.to_csv(output / 'plot_source.csv', index=False)
    contract = dict(claim='Time-resolved oral support distinguishes changing report coverage, supporting-rule availability and decision contribution.',
        backend='python', archetype='individual longitudinal diagnosis', subjects=subjects, arms=order,
        reference_model=reference, block_width=32, width_mm=183, height_mm=145, dpi=300, formats=['PNG'],
        parameter_estimation_scope=estimation_scope, evaluation_use=manifest.get('evaluation_use', 'original_evaluation_protocol'),
        panels=['complete and catalogue-supported report counts', 'whole-report decision support with frozen-prefix reference',
                'posterior supporting-rule availability: all-component to any-component bounds'],
        statistics={'center': 'mean of pooled eight-filter support over complete reports in each display block',
                    'sample_size': 'report counts in top panel; eight filters per fitted arm',
                    'bands': 'component ambiguity bounds; not confidence intervals',
                    'tests': 'none', 'uncertainty': 'no interval estimate in this descriptive time plot'},
        limits=['automatic parsing requires semantic review', 'reported content can change between blocks',
                'availability is conditional on observed choices', 'prefix reference also removes trial-specific stimulus readout',
                'display blocks are not a cognitive memory window'],
        source_preflight_exceptions={'EXPORT-VECTOR': 'Repository PNG-only instruction.',
                                     'EXPORT-RASTER': 'TIFF omitted under PNG-only instruction.',
                                     'RASTER-DPI': '300 dpi discussion figure.'},
        inputs_sha256={str(p.resolve()): digest(p) for p in [alignment / 'manifest.json', alignment / 'trials.csv']})
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    colors = dict(COLORS)
    if order == ['POOL', 'SEARCH']:
        colors['POOL'] = '#68778B'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7,
        'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 6, 'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    for subject in subjects:
        own = source[source.subject.eq(subject)]
        baseline = own[own.arm.eq(reference)]
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(183 / 25.4, 145 / 25.4))
        fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.86, hspace=.52)
        axes[0].bar(baseline.midpoint, baseline.reports, width=29, color='#DBDFE4', label='Complete reports')
        axes[0].bar(baseline.midpoint, baseline.catalogue_reports, width=29, color='#68778B', label='Catalogue supports report')
        axes[0].set(ylabel='Reports per block', ylim=(0, 34), yticks=[0, 16, 32])
        axes[0].legend(frameon=False, loc='upper left',
                       bbox_to_anchor=(0, 1.30 if retrospective else 1.44), ncol=2)
        for arm in order:
            part = own[own.arm.eq(arm)]
            axes[1].plot(part.midpoint, part.structural_support, color=colors[arm], linewidth=1.1, label=arm)
            low, high = part.structural_availability_all_components, part.structural_availability_any_component
            axes[2].fill_between(part.midpoint, low, high, color=colors[arm], alpha=.12, linewidth=0)
            axes[2].plot(part.midpoint, high, color=colors[arm], linewidth=1.1)
        axes[1].plot(baseline.midpoint, baseline.prefix_static_structural_support,
            color='#333333', linewidth=.9, linestyle=':', label=f'{reference}: frozen prefix')
        axes[1].legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.45), ncol=len(order) + 1)
        axes[1].set(ylabel='Decision support', ylim=(-.03, 1.03), yticks=[0, .5, 1])
        axes[2].set(ylabel='Rule availability', xlabel='Trial', ylim=(-.03, 1.03), yticks=[0, .5, 1])
        cut = int(baseline.training_trials.iloc[0])
        for index, ax in enumerate(axes):
            ax.axvline(cut + .5, color='#999999', linestyle='--', linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2.5, width=.6)
            ax.text(-.10, 1.03, 'abc'[index], transform=ax.transAxes, weight='bold', fontsize=8)
            ax.set_xlim(.5, int(baseline.end.max()) + .5)
        axes[0].text(cut + .5, 35, 'Later trials →', ha='left', va='bottom', fontsize=6)
        fig.suptitle(f'Subject {subject}: oral alignment through learning', x=.12, ha='left', fontsize=9, y=.98)
        fig.text(.12, .045, '32-trial display blocks; complete reports only in b/c. Shading in c denotes component bounds.', fontsize=6)
        fig.text(.12, .02, f'Dashed line: {boundary_label}. Dotted line: category-specific basis frozen at that boundary.', fontsize=6)
        if retrospective:
            fig.text(.12, .94, f'{estimation_scope.capitalize()}-scope parameter selection; retrospective development diagnostic.', fontsize=6)
        fig.savefig(output / f'subject_{subject}.png', dpi=300)
        plt.close(fig)
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述对齐时间过程</title>'
        '<h1>口述覆盖、作答支持和规则存在概率</h1><p>每32题汇总。上图显示完整报告数及其中目录可支持的报告数。'
        '中图显示完整报告的平均显式支持质量，下图显示支持规则存在概率的分量上下界。竖虚线表示原时间划分边界。</p>'
        '<p>八次滤波先合并联合质量，再按实际选择概率归一化。中下图只评价完整报告；空窗口保留空缺。'
        '阴影表示分量歧义，数值模拟误差请查完整汇总。训练段静态基准同时冻结状态与试次刺激读出。</p>'
        + ('<p>本轮候选筛选使用完整行为序列，后段为回顾性诊断。口述仅用于冻结参数后的表征检验。</p>' if retrospective else '')
        + ''.join(f'<h2>{s}</h2><img src="subject_{s}.png" style="max-width:100%;width:900px">' for s in subjects))
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', subjects=subjects, arms=order,
        plotting_source_sha256=digest(Path(__file__)), artifacts={p.name: digest(p) for p in output.iterdir()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); plot(args.alignment, args.output)
