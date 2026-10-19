"""Show fixed-parameter SEPARATE behavior, rule use and external oral evidence.

Run --probe PROBE --oral-audit AUDIT --output NEW_DIRECTORY. All six cells,
192 autonomous paths and both fixed oral readings are retained. This Python
discussion figure uses the completed participant 208 fixed-parameter screen;
the separate full refit remains a later comparison. Export PNG, source CSVs,
provenance and an HTML explanation. Errors describe simulation, not participants.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load

PARAMETERS = ('RETAIN', 'ONLINE', 'NEAR')
MECHANISMS = ('ONLINE', 'SEPARATE')
LABELS = ['RETAIN-fit\nγ = 0', 'ONLINE-fit\nγ = 1', 'Near candidate\nγ = 1']
COLORS = {'ONLINE': '#4678A5', 'SEPARATE': '#CC8152'}


def draw(probe: Path, oral: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new joint-evidence figure directory')
    proof, oral_proof = load(probe / 'completion.json'), load(oral / 'completion.json')
    if (proof['status'] != 'complete' or proof['smoke'] or proof['cells'] != 6
            or proof['parameters_changed'] is not False or oral_proof['status'] != 'complete'
            or oral_proof['checked_contrasts'] != 120):
        raise ValueError('Need the complete six-cell screen and paired oral audit')
    inputs = dict(proof['inputs_sha256']); inputs.update(oral_proof['inputs_sha256'])
    for path, sha in inputs.items():
        if digest(Path(path)) != sha: raise ValueError('Completed evidence input changed')
    def bind(path):
        path = Path(path).resolve(); inputs[str(path)] = digest(path); return path
    for directory in (probe, oral): bind(directory / 'completion.json')
    if digest(bind(oral / 'run.py')) != oral_proof['source_sha256']:
        raise ValueError('Independent oral audit implementation changed')
    protocol = load(bind(probe / 'protocol.json'))
    if digest(probe / 'protocol.json') != proof['protocol_sha256']:
        raise ValueError('Probe protocol changed')
    tables = {name: pd.read_csv(bind(probe / (name + '.csv'))) for name in ('scores', 'contrasts', 'paths')}
    for name, sha in proof['artifacts'].items():
        if digest(probe / name) != sha: raise ValueError('Probe table changed')
    all_oral = pd.read_csv(bind(oral / 'feedback_contrasts.csv'))
    checked = pd.read_csv(bind(oral / 'checked_contrasts.csv'))
    oral_scores = all_oral[(all_oral.feedback == 'all') & (all_oral.metric == 'conditioned')].copy()
    checks = 0
    for _, row in oral_scores.iterrows():
        other = checked[(checked.reading == row.reading) & (checked.parameter == row.parameter)
            & (checked.window == row.window) & (checked.domain == row.domain) & (checked.subset == 'complete')]
        if len(other) != 1: raise ValueError('Oral report set lacks a unique paired comparison')
        for prefix in ('online', 'separate'):
            np.testing.assert_allclose(sum(row[prefix + '_' + field] for field in ('support', 'conflict', 'undetermined')),
                                       1., rtol=0, atol=1e-12); checks += 1
        for field in ('support', 'conflict', 'undetermined'):
            for suffix in ('_difference', '_difference_mcse'):
                np.testing.assert_allclose(row[field + suffix], other.iloc[0][field + suffix], rtol=0, atol=1e-12)
                checks += 1
    paths, summaries, parameters = tables['paths'], [], []
    if paths.duplicated(['cell', 'seed']).any() or len(paths) != 192:
        raise ValueError('Expected all 192 distinct within-cell autonomous paths')
    expected_cells = {m + '_' + p for p in PARAMETERS for m in MECHANISMS}
    if set(paths.cell) != expected_cells or {b['cell'] for b in protocol['banks']} != expected_cells:
        raise ValueError('Six fixed-parameter cells required')
    human = None
    for bank in protocol['banks']:
        parts = []
        for item in bank['roles']['autonomous']:
            job = item['job']; path = Path(job['target']).with_suffix('.npz')
            record = load(bind(path.with_suffix('.json')))
            if record['candidate'] != bank['candidate'] or digest(bind(path)) != record['predictions_sha256']:
                raise ValueError('Autonomous parameter identity or array changed')
            with np.load(path) as z: parts.append({k: z[k].copy() for k in z.files})
        for part in parts[1:]: np.testing.assert_array_equal(part['categories'], parts[0]['categories'])
        arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0] if key != 'categories'}
        categories = parts[0]['categories']
        if len(categories) != 576 or len(set(arrays['seeds'])) != 32:
            raise ValueError('Need every full 32-path bank')
        folder = Path(bank['roles']['autonomous'][0]['job']['folder'])
        prepared = load(bind(folder / 'prepared.json')); choices = []
        for name in ('training.npz', 'sealed_test.npz'):
            with np.load(bind(folder / name)) as z: choices.append(z['choices'].copy())
        if prepared['subject'] != 208 or prepared['training_trials'] != 384:
            raise ValueError('Original participant and trial windows changed')
        observed = np.concatenate(choices) == categories
        current = float(observed[-128:].mean())
        if human is not None: np.testing.assert_array_equal(current, human)
        human = current
        saved = paths[paths.cell == bank['cell']].set_index('seed').loc[arrays['seeds']]
        values = dict(late_accuracy=(arrays['choices'][:, -128:] == categories[-128:]).mean(1),
                      late_present=arrays['present'][:, -128:].mean(1),
                      late_contribution=arrays['contribution'][:, -128:].mean(1))
        for key, value in values.items():
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12); checks += len(value)
        summaries.append(dict(cell=bank['cell'], **{k: v.mean() for k, v in values.items()},
                              human_late_accuracy=human, paths=32))
        point = bank['candidate']['hyperparams']
        parameters.append(dict(cell=bank['cell'], parameter=bank['parameter'], candidate=bank['candidate']['id'],
            gamma=point['engine.modules.memory_mod.kwargs.gamma'],
            delta=point['engine.modules.hypo_transitions_mod.kwargs.intervention.decay']))
    parameter_frame = pd.DataFrame(parameters)
    for p in PARAMETERS:
        group = parameter_frame[parameter_frame.parameter == p]
        if len(group) != 2 or group.candidate.nunique() != 1 or group.gamma.nunique() != 1:
            raise ValueError('Paired mechanisms must use identical frozen parameters')
        np.testing.assert_array_equal(group.gamma, [0., 0.] if p == 'RETAIN' else [1., 1.])
    summary = pd.DataFrame(summaries).set_index('cell')
    later_oral = oral_scores[(oral_scores.window == 'test') & (oral_scores.domain == 'functional')]
    if len(later_oral) != 6 or not (later_oral.reports == 119).all():
        raise ValueError('Need the same 119 complete later reports in both readings')
    for path, sha in inputs.items():
        if digest(Path(path)) != sha: raise ValueError('Figure input changed during verification')
    output.mkdir(parents=True)
    contract = dict(conclusion='At two gamma1 fixed points, separate belief persistence improves later generation and oral support while early choice likelihood worsens; the gamma0 point changes in the opposite direction.',
        archetype='quantitative grid', backend='python', dimensions_mm=[183, 176], dpi=300, export=['PNG'],
        panels=dict(a='Training and later choice-NLL changes', b='All32 autonomous paths in every cell',
                    c='Presence and contribution of the target rule', d='Later functional oral-support change under both readings'),
        participant_count=1, simulated_paths_per_cell=32, filters_per_cell=2, particles_per_filter=128,
        complete_reports=391, later_reports=119, estimation_unchanged=True,
        intervals='One paired simulation standard error in panels a and d; path dots in b are individual simulations',
        exclusions='No within-cell paths or complete reports excluded; a shows training/later, b/c final128trials, d later functional support. Full phase and oral-mass tables retained.',
        style_reuse='Existing project typography and mechanism colors; panel statistics implemented for this evidence',
        status='Development discussion draft; post-refit and participant generalization remain separate')
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 176 / 25.4))
    fig.subplots_adjust(left=.11, right=.97, bottom=.17, top=.85, wspace=.38, hspace=.52)
    titles = ['Choice prediction has a phase trade-off', 'Every autonomous path is shown',
              'A rule can be present but used less', 'Both oral readings support later gains']
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, fontsize=8, loc='left', pad=10)
        ax.text(-.19, 1.08, 'abcd'[i], transform=ax.transAxes, weight='bold', fontsize=10)
        ax.set_xticks(range(3), LABELS); ax.tick_params(labelsize=7)
        ax.set_xlim(-.5, 2.5)
    ax = axes[0, 0]; ax.axhline(0., color='.7', lw=.8)
    for phase, name, offset, color, marker in [('training', 'Training', -.09, '#424A53', 'o'),
                                              ('test', 'Later', .09, '#879B72', 's')]:
        data = tables['contrasts']; data = data[data.phase == phase].set_index('parameter').loc[list(PARAMETERS)]
        ax.errorbar(np.arange(3) + offset, data.delta_choice_nll, yerr=data.delta_choice_nll_mc_se,
                    fmt=marker, color=color, ms=5, capsize=2, lw=1., label=name)
    ax.set_ylabel('Δ choice NLL (lower is better)'); ax.legend(fontsize=7, loc='lower left')
    ax = axes[0, 1]
    for i, parameter in enumerate(PARAMETERS):
        for mechanism, offset in [('ONLINE', -.14), ('SEPARATE', .14)]:
            cell = mechanism + '_' + parameter; data = paths[paths.cell == cell].sort_values('seed')
            ax.scatter(i + offset + np.linspace(-.075, .075, len(data)), data.late_accuracy,
                       s=8, color=COLORS[mechanism], alpha=.55, linewidths=0)
            ax.plot([i + offset - .09, i + offset + .09], [data.late_accuracy.mean()] * 2,
                    color=COLORS[mechanism], lw=2.)
    ax.axhline(human, color='.25', ls=':', lw=.9)
    ax.text(2.4, 1.0, 'Participant: 90.6%', ha='right', fontsize=7)
    ax.set(ylabel='Accuracy in final 128 trials', ylim=(0., 1.04))
    ax = axes[1, 0]
    for i, parameter in enumerate(PARAMETERS):
        for mechanism, offset in [('ONLINE', -.14), ('SEPARATE', .14)]:
            row = summary.loc[mechanism + '_' + parameter]; x = i + offset; color = COLORS[mechanism]
            ax.plot([x, x], [row.late_contribution, row.late_present], color=color, lw=1.2)
            ax.plot(x, row.late_present, 'o', mfc='white', mec=color, ms=5)
            ax.plot(x, row.late_contribution, 'o', color=color, ms=5)
    ax.set(ylabel='Mean in final 128 trials', ylim=(0., 1.07))
    ax.legend(handles=[Line2D([], [], marker='o', mfc='white', color='.3', ls='', label='Presence'),
                       Line2D([], [], marker='o', color='.3', ls='', label='Contribution')], loc='lower left', fontsize=7)
    ax = axes[1, 1]; ax.axhline(0., color='.7', lw=.8)
    for reading, name, offset, color, marker in [('midpoint', 'Midpoint', -.09, '#424A53', 'o'),
                                                ('relative', 'Relative', .09, '#879B72', 's')]:
        data = later_oral[later_oral.reading == reading].set_index('parameter').loc[list(PARAMETERS)]
        ax.errorbar(np.arange(3) + offset, data.support_difference, yerr=data.support_difference_mcse,
                    fmt=marker, color=color, ms=5, capsize=2, lw=1., label=name)
    ax.set_ylabel('Δ functional oral support\n(higher is better)'); ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Separating belief persistence restores later rule use at γ = 1', fontsize=10, y=.98)
    fig.text(.5, .94, 'Participant 208 · three frozen parameter points · SEPARATE compared with ONLINE',
             ha='center', fontsize=8)
    fig.legend(handles=[Line2D([], [], color=COLORS[m], marker='o', ls='', label=m) for m in MECHANISMS],
               loc='upper center', bbox_to_anchor=(.5, .92), ncol=2, fontsize=8)
    fig.text(.11, .055, 'All differences: SEPARATE − ONLINE. Error bars: ±1 paired simulation SE.\n'
             '32 paths and 2 × 128-particle filters per cell; oral: 119 complete later reports.\n'
             'Fixed-parameter development comparison; full behavioral re-estimation is evaluated separately.', fontsize=7)
    fig.savefig(output / 'separate_belief_joint.png', dpi=300, facecolor='white'); plt.close(fig)
    exports = dict(tables, oral_full_strata=all_oral, oral_complete=oral_scores,
                   late_path_summary=summary.reset_index(), parameter_identity=parameter_frame)
    for name, frame in exports.items(): frame.to_csv(output / (name + '.csv'), index=False)
    oral_columns = ['reading', 'parameter', 'reports', 'online_support', 'separate_support',
                    'support_difference', 'support_difference_mcse', 'online_conflict', 'separate_conflict',
                    'online_undetermined', 'separate_undetermined']
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>208固定参数联合证据</title>
<style>body{max-width:1120px;margin:32px auto;font:16px/1.65 sans-serif;color:#222}img{width:100%}
table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}section{overflow-x:auto}</style>
<h1>信念保持与候选管理分开后的联合证据</h1>
<p>208的两个γ＝1固定参数点改善了后段自主表现和口述功能支持，训练段选择预测仍有代价。
γ＝0的旧参数点方向相反。每组保留全部32条自主路径；参数、两种口述读法及384／192试次分段保持固定。</p>
<img src="separate_belief_joint.png" alt="固定参数下选择、路径、规则使用与口述的四面板对照">
<p>a、d误差棒为两个完整滤波重复成对计算的一个模拟标准误，表示固定参数下的数值波动。
b每点是一条自主路径，短横线为均值；c空心点为正确规则在场率，实心点为其作答贡献，均以全部路径和末128题为分母。
d展示本题选择后、反馈前的局部依据对整句口述的功能支持；支持、冲突和未确定质量同时保存在下表。</p>
<p>RETAIN-fit与ONLINE-fit分别是旧RETAIN和ONLINE机制从训练行为选出的参数；Near是已确认的接近候选。
每个参数点的两种机制使用完全相同的参数。正式SEPARATE行为重新估参及外部口述验证另外进行。</p>
<h2>后段双读法：完整报告的三种质量</h2><section>'''
    html += later_oral[oral_columns].to_html(index=False, float_format=lambda x: f'{x:.6f}')
    html += '</section><h2>完整分段行为评分</h2><section>' + tables['scores'].to_html(index=False, float_format=lambda x: f'{x:.6f}')
    html += '</section><p>源数据：' + ' · '.join(f'<a href="{n}.csv">{n}</a>' for n in exports) + '</p></html>'
    (output / 'index.html').write_text(html)
    shutil.copy2(__file__, output / 'analysis_source.py')
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', inputs_sha256=inputs,
        analysis_sha256=digest(Path(__file__)), numeric_checks=checks,
        outputs_sha256={p.name: digest(p) for p in output.iterdir()},
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'matplotlib')},
        data_use='Fixed behavior-selected parameters; oral externally evaluated; all6cells and192paths included'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'oral-audit', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); draw(args.probe, args.oral_audit, args.output)
