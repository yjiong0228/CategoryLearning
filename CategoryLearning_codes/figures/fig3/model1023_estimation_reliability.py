"""Show independent search checks and finite-candidate parameter sensitivity.

Figure contract: independent scores test whether the bounded search improved
the estimate; parameter and state recovery answer separate questions. The
quantitative grid shows every audited candidate, both known generating-point
comparisons and all free parameter values. Per-case parameter heatmaps display
grid positions with actual values printed in cells. Exact inactive coordinates
are marked; they do not provide evidence about a participant's trait.

Use --summary COMPLETED_PHASE_SUMMARY --output NEW [--allow-smoke]. Python is
the saved backend. Exports are 183-mm, 300-dpi PNG review figures, source CSVs
and an HTML reading guide. Error bars are two paired Monte Carlo standard
errors at fixed parameters. State plots reuse the established full-trial
recovery layout and weighted 5--95% intervals exactly. Smoke figures carry an
explicit software-only label and never enter scientific summaries.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1023_estimation_reliability import read, write, digest
from . import model1022_compact_recovery as state_figure

PARAMETER_LABELS = {
    'search_controller.baseline_probability': 'Baseline search',
    'search_controller.error_gain': 'Failure sensitivity',
    'search_controller.failure_decay': 'Failure retention',
    'search_controller.global_search': 'Global search',
    'max_capacity': 'Capacity',
    'memory_mod.kwargs.gamma': 'Pairing retention',
    'beta_mod.kwargs.beta_init': 'Initial precision',
    'beta_mod.kwargs.learning_rate': 'Precision learning',
    'intervention.decay': 'Evidence retention',
}
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.spines.top': False,
         'axes.spines.right': False, 'axes.linewidth': .7, 'legend.frameon': False,
         'svg.fonttype': 'none', 'pdf.fonttype': 42}


def parameter_label(key: str) -> str:
    matches = [label for suffix, label in PARAMETER_LABELS.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('Unmapped free parameter: '+key)
    return matches[0]


def candidate_label(row) -> str:
    if row.selected and row.reference:
        return 'Selected / reference'
    return 'Selected' if row.selected else ('Reference' if row.reference else 'Alternative')


def render(summary: Path, output: Path, allow_smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new estimation-figure directory')
    done = read(summary/'completion.json')
    if done['status'] != 'complete' or (done['smoke'] and not allow_smoke):
        raise ValueError('Complete scientific phase required; smoke needs explicit --allow-smoke')
    inputs = {str((summary/'completion.json').resolve()): digest(summary/'completion.json'),
              str(Path(__file__).resolve()): digest(Path(__file__))}
    for name, sha in done['inputs_sha256'].items():
        if digest(Path(name)) != sha:
            raise ValueError('Underlying phase-summary input changed: '+name)
        inputs[name] = sha
    tables = {}
    for name, sha in done['artifacts_sha256'].items():
        file = summary/name
        if digest(file) != sha:
            raise ValueError('Phase-summary table changed: '+name)
        inputs[str(file.resolve())] = sha
        tables[file.stem] = pd.read_csv(file)
    audit = tables['audit_candidates']
    parameters = tables['audit_parameters']
    case_ids = sorted(audit.case_id.unique())
    if len(case_ids) != done['cases'] or set(parameters.case_id) != set(case_ids):
        raise ValueError('Figure cases differ from the audited phase')
    output.mkdir(parents=True)
    for name in done['artifacts_sha256']:
        shutil.copyfile(summary/name, output/name)
    smoke_title = 'SOFTWARE CHECK ONLY\n' if done['smoke'] else ''
    files = []
    with plt.rc_context(STYLE):
        recovery = 'recovery_likelihood_comparison' in tables
        fig, axes = plt.subplots(1, 2 if recovery else 1,
                                 figsize=(183/25.4, max(100, 15*len(case_ids)+45)/25.4), squeeze=False)
        fig.subplots_adjust(left=.19, right=.97, bottom=.33, top=.75, wspace=.66)
        ax = axes[0, 0]
        for i, case_id in enumerate(case_ids):
            own = audit[audit.case_id == case_id].sort_values(['selected', 'reference'], ascending=False)
            for offset, row in zip(np.linspace(-.2, .2, len(own)), own.itertuples()):
                marker = 'o' if row.selected else ('s' if row.reference else '^')
                color = '#3B8C85' if row.selected else ('#527697' if row.reference else '#9B775D')
                ax.errorbar(row.delta_to_selected, i+offset, xerr=2*row.delta_mc_se,
                            marker=marker, color=color, ms=4, ls='', capsize=2)
        ax.axvline(0, color='#777777', ls=':', lw=.8)
        ax.set_yticks(range(len(case_ids)), case_ids)
        ax.set_ylim(len(case_ids)-.5, -.5)
        ax.set_xlabel('NLL / trial relative to selected\nNegative: candidate fits better')
        ax.set_title('a  Independent candidate audit', loc='left', fontweight='bold', fontsize=8)
        if recovery:
            values = tables['recovery_likelihood_comparison'].set_index('case_id').loc[case_ids]
            if values.index.duplicated().any():
                raise ValueError('Repeated generating-point contrast')
            ax = axes[0, 1]
            ax.errorbar(values.estimate_minus_truth, np.arange(len(case_ids)),
                        xerr=2*values.paired_mc_se, fmt='o', color='#3B8C85', ms=4, capsize=2)
            ax.axvline(0, color='#777777', ls=':', lw=.8)
            ax.set_yticks(range(len(case_ids)), case_ids)
            ax.set_ylim(len(case_ids)-.5, -.5)
            ax.set_xlabel('Selected minus generating-point\nNLL / trial (positive: search gap)')
            ax.set_title('b  Generating-point comparison', loc='left', fontweight='bold', fontsize=8)
        fig.suptitle(smoke_title+'Model 1022 · estimation '+done['phase'], y=.98, fontsize=9)
        fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls='', label=label) for color, marker, label in
                   [('#3B8C85', 'o', 'Selected'), ('#527697', 's', 'Reference'), ('#9B775D', '^', 'Alternative')]],
                   loc='upper center', bbox_to_anchor=(.5, .87), ncol=3, fontsize=7)
        fig.text(.5, .15, 'Selection precedes this audit; candidates remain fixed. Bars: ±2 paired Monte Carlo SE.',
                 ha='center', fontsize=6.5)
        fig.text(.5, .095, 'Numerical uncertainty at fixed parameters; individual panels use their own x scales.',
                 ha='center', fontsize=6.5)
        fig.text(.5, .04, 'When selected and reference coincide, the selected marker represents both.', ha='center', fontsize=6.5)
        name = 'independent_audit.png'
        fig.savefig(output/name, dpi=300, facecolor='white')
        plt.close(fig)
        files.append(name)
        for case_id in case_ids:
            own = parameters[(parameters.case_id == case_id) & parameters.fitted]
            candidates = audit[audit.case_id == case_id].sort_values(['selected', 'reference'], ascending=False)
            keys = list(own.parameter.drop_duplicates())
            ids = list(candidates.candidate)
            matrix = own.pivot(index='candidate', columns='parameter', values='normalized_grid_position').loc[ids, keys]
            if not np.isfinite(matrix).all().all() or np.any((matrix < 0) | (matrix > 1)):
                raise ValueError('Invalid parameter grid positions')
            fig, ax = plt.subplots(figsize=(183/25.4, 100/25.4))
            fig.subplots_adjust(left=.2, right=.89, top=.75, bottom=.36)
            im = ax.imshow(matrix, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(keys)), [parameter_label(k) for k in keys], rotation=38, ha='right')
            ax.set_yticks(range(len(ids)), [candidate_label(row) for row in candidates.itertuples()])
            for i, candidate in enumerate(ids):
                for j, key in enumerate(keys):
                    row, = own[(own.candidate == candidate) & (own.parameter == key)].itertuples()
                    label = f'{row.value:g}'+('*' if row.known_zero_effect else '')
                    ax.text(j, i, label, ha='center', va='center', fontsize=7,
                            color='white' if matrix.iloc[i, j] > .55 else '#222222')
            cax = fig.add_axes([.91, .36, .014, .39])
            bar = fig.colorbar(im, cax=cax, ticks=[0, 1])
            bar.ax.set_yticklabels(['Low', 'High'], fontsize=6.5)
            fig.suptitle(smoke_title+case_id+' · fixed candidate parameters', y=.98, fontsize=9)
            fig.text(.5, .84, 'Cell numbers: actual parameter values. Color: ordered position in the search grid.',
                     ha='center', fontsize=6.5)
            fig.text(.5, .09, '* Exact inactive coordinate at this point; individual-trait interpretation is unavailable.',
                     ha='center', fontsize=6.5)
            fig.text(.5, .04, 'All audited candidates shown. Fixed common coordinates remain in the source table.',
                     ha='center', fontsize=6.5)
            name = case_id+'_parameters.png'
            fig.savefig(output/name, dpi=300, facecolor='white')
            plt.close(fig)
            files.append(name)
    if recovery and not done['smoke']:
        state_figure.render(Path(done['study'])/'recovery_states', output/'states')
        inputs[str(Path(state_figure.__file__).resolve())] = digest(Path(state_figure.__file__))
        files.extend('states/'+case_id+'.png' for case_id in case_ids)
    page = '<!doctype html><meta charset="utf-8"><title>Model 1022估参检验</title>'
    page += '<style>body{max-width:1100px;margin:32px auto;font:16px sans-serif}img{width:100%}table{font-size:13px}</style>'
    page += '<h1>'+html.escape(smoke_title)+'Model 1022估参检验</h1>'
    page += '<p>独立复核检查选中参数是否优于其他候选。误差条表示模拟计算误差，参数保持冻结。'
    page += '恢复案例还与已知生成参数比较；生成点用于检查搜索缺口。有限行为序列上的最优参数可能偏离生成值。</p>'
    page += '<p>参数图保留全部受检候选。颜色表示共同搜索范围中的有序位置，格内数字给出参数实际值。'
    page += '星号标出在当前组合下已知不起作用的参数。候选集合用于检查解释的敏感性。</p>'
    page += '<p>内部状态图逐题比较已知轨迹与推断轨迹，显示作答之后、当前反馈之前的状态。'
    page += '阴影表示固定参数下90%的状态区间；规则存在时才定义其精度。</p>'
    page += ''.join('<img src="'+html.escape(name)+'">' for name in files)
    page += '<h2>独立复核数据</h2>'+audit.to_html(index=False)
    (output/'index.html').write_text(page, encoding='utf-8')
    shutil.copyfile(Path(__file__), output/'source_snapshot.py')
    write(output/'completion.json', dict(status='complete', smoke=done['smoke'], phase=done['phase'],
        summary=str(summary.resolve()), inputs_sha256=inputs, backend='python',
        archetype='quantitative_grid', dimensions_mm=[183, 100], dpi=300, format='PNG',
        selection='All audited cases and candidates; fixed coordinates retained in source CSVs',
        uncertainty='Two paired Monte Carlo SE; state plots retain 5th–95th weighted intervals at fixed parameters',
        figures={name: digest(output/name) for name in files}, visual_qa='pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    render(args.summary, args.output, args.allow_smoke)
