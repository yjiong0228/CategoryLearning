"""Show when behavior-inferred states support the recorded oral reports.

Run from the repository root with --source COMPLETED_SUMMARY --output NEW.
The input is the same frozen six-person evaluation used by model1022_compact_refit.
Python/matplotlib quantitative grids show all complete-report trial support dots,
32-trial means of support/conflict/undetermined mass, and catalogue coverage.
Columns compare COMPACT and TRANSFER; cohorts, domains and readings stay separate.
Trial coordinates and train/test cuts match the behavioral trajectories. A block
never crosses the cut. Missing or incomplete reports remain in source tables and
are marked below the axes; empty blocks stay undefined. Means weight reports
equally, following the existing complete-report summary. No refitting, recoding,
new model simulations, uncertainty estimates or hypothesis tests are performed.

Figure contract: temporal support can differ from a whole-phase mean. Preserve
the complete-report denominator and distinguish catalogue coverage from model
support. Style-only inheritance from the existing comparison, with new temporal
aggregation validated against its summary. Export: 183-mm PNG review figures at
300 dpi, source CSVs, input hashes, interpretation notes and an HTML index.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ARMS = ('COMPACT', 'TRANSFER')
READINGS = ('midpoint', 'relative')
DOMAINS = ('functional', 'structural')
LABELS = ('support', 'conflict', 'undetermined')
COLORS = {'support': '#538B84', 'conflict': '#C58B60', 'undetermined': '#929BA6'}
STYLES = {'support': '-', 'conflict': '--', 'undetermined': ':'}
NAMES = {'COMPACT': 'Compact', 'TRANSFER': 'Model 1022 · precision transfer'}
COHORTS = {'development': 'Development', 'extension': 'Extension · prior influence unknown'}
BLOCK_WIDTH = 32


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_blocks(trials: pd.DataFrame, width: int = BLOCK_WIDTH) -> pd.DataFrame:
    """Average complete reports on the behavioral time grid, preserving gaps."""
    if width < 1 or int(width) != width:
        raise ValueError('Block width must be a positive integer')
    rows = []
    for (subject, arm), source in trials.groupby(['iSub', 'arm'], sort=True):
        group = source.sort_values('ordinal_trial')
        if group.training_trials.nunique() != 1:
            raise ValueError('Inconsistent training boundary')
        n, cut = len(group), int(group.training_trials.iloc[0])
        if not 0 < cut < n or cut % width:
            raise ValueError('Block width must preserve the training boundary')
        if not np.array_equal(group.ordinal_trial, np.arange(1, n + 1)):
            raise ValueError('Trial coordinates must be unique and consecutive')
        complete = group.scope.eq('complete')
        columns = [f'{d}_{label}' for d in DOMAINS for label in LABELS]
        mass = group.loc[complete, columns].to_numpy(float).reshape(-1, 2, 3)
        if not np.isfinite(mass).all() or np.any(mass < -1e-12) or np.any(mass > 1 + 1e-12):
            raise ValueError('Invalid probability mass on complete reports')
        np.testing.assert_allclose(mass.sum(2), 1, atol=1e-10, rtol=0)
        for start in range(0, n, width):
            block = group.iloc[start:start + width]
            used = block.loc[block.scope.eq('complete')]
            end = start + len(block)
            row = dict(subject=int(subject), arm=arm, trial=end, start_trial=start + 1,
                       training_trials=cut, phase='training' if end <= cut else 'test',
                       trials=len(block), reports=len(used), unresolved=len(block) - len(used))
            for domain in DOMAINS:
                for label in LABELS:
                    row[f'{domain}_{label}'] = used[f'{domain}_{label}'].mean()
                covered = used[f'{domain}_support_any_component'].gt(0)
                row[f'{domain}_covered_reports'] = int(covered.sum())
                row[f'{domain}_coverage'] = covered.mean() if len(used) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def check_summary(trials: pd.DataFrame, blocks: pd.DataFrame, summary: pd.DataFrame) -> int:
    """Reconstruct archived means by report counts, including unequal blocks."""
    checks = 0
    for _, target in summary.query('subset == "complete" and window in ["whole", "training", "test"]').iterrows():
        raw = trials.loc[trials.iSub.eq(target.subject) & trials.arm.eq(target.arm)]
        binned = blocks.loc[blocks.subject.eq(target.subject) & blocks.arm.eq(target.arm)]
        if target.window != 'whole':
            raw = raw.loc[raw.phase.eq(target.window)]
            binned = binned.loc[binned.phase.eq(target.window)]
        reports = raw.loc[raw.scope.eq('complete')]
        assert len(raw) == target.window_trials == binned.trials.sum()
        assert len(reports) == target.reports == binned.reports.sum()
        for label in LABELS:
            column = f'{target.domain}_{label}'
            direct = reports[column].mean()
            nonempty = binned.loc[binned.reports.gt(0)]
            weighted = (np.average(nonempty[column], weights=nonempty.reports)
                        if len(nonempty) else np.nan)
            np.testing.assert_allclose([direct, weighted], target[label], atol=1e-10,
                                       rtol=0, equal_nan=True)
            checks += 1
    return checks


def load_data(source: Path) -> dict:
    done = json.loads((source / 'completion.json').read_text())
    if done['status'] != 'complete' or len(done['subjects']) != 6:
        raise ValueError('Require the completed six-person comparison')
    bindings = {str((source / 'completion.json').resolve()): digest(source / 'completion.json')}

    def read(name: str) -> pd.DataFrame:
        file = source / name
        sha = digest(file)
        if sha != done['artifacts_sha256'][name]:
            raise ValueError('Archived source changed: ' + str(file))
        bindings[str(file.resolve())] = sha
        return pd.read_csv(file)

    curves, scores = read('curves.csv'), read('scores.csv')
    meta = scores.query('phase == "whole"').copy()
    if meta.duplicated(['subject', 'model']).any() or len(meta) != 12:
        raise ValueError('Require one full trajectory per subject and mechanism')
    trials, blocks, counts = {}, {}, []
    summary_checks = 0
    identity_columns = ['iSub', 'ordinal_trial', 'state_trial', 'iSession', 'iTrial',
                        'condition', 'choice', 'text', 'scope', 'training_trials', 'phase']
    baseline = None
    for reading in READINGS:
        table = read(f'oral_{reading}/trials.csv')
        summary = read(f'oral_{reading}/summary.csv')
        if set(table.arm) != set(ARMS) or set(table.iSub) != set(done['subjects']):
            raise ValueError('Missing subject or mechanism in oral trials')
        for arm in ARMS:
            identities = table.loc[table.arm.eq(arm), identity_columns].sort_values(
                ['iSub', 'ordinal_trial']).reset_index(drop=True)
            if baseline is None:
                baseline = identities
            else:
                pd.testing.assert_frame_equal(identities, baseline)
        for _, case in meta.iterrows():
            raw = table.loc[table.iSub.eq(case.subject) & table.arm.eq(case.model)].sort_values('ordinal_trial')
            curve = curves.loc[curves.subject.eq(case.subject) & curves.model.eq(case.model)].sort_values('trial')
            if (len(raw) != case.trials or not raw.condition.eq(case.condition).all()
                    or not raw.training_trials.eq(case.training_trials).all()
                    or not raw.filter_seeds.eq(8).all()):
                raise ValueError('Oral/behavior identities, trial counts or filter repetitions differ')
            np.testing.assert_array_equal(raw.ordinal_trial, raw.state_trial + 1)
            np.testing.assert_array_equal(raw.phase, np.where(raw.ordinal_trial <= case.training_trials,
                                                            'training', 'test'))
            np.testing.assert_array_equal(curve.trial, np.minimum(
                np.arange(BLOCK_WIDTH, int(case.trials) + BLOCK_WIDTH, BLOCK_WIDTH), case.trials))
            if not curve.training_trials.eq(case.training_trials).all():
                raise ValueError('Behavioral training cut differs')
            for phase, own in raw.groupby('phase'):
                for scope, n in own.groupby('scope', dropna=False).size().items():
                    counts.append(dict(reading=reading, subject=int(case.subject), arm=case.model,
                                       phase=phase, scope=scope, trials=int(n)))
        binned = time_blocks(table)
        summary_checks += check_summary(table, binned, summary)
        trials[reading], blocks[reading] = table, binned
    return dict(trials=trials, blocks=blocks, counts=pd.DataFrame(counts), meta=meta,
                bindings=bindings, summary_checks=summary_checks)


def plot_grid(data: dict, cohort: str, reading: str, domain: str, output: Path) -> str:
    """Display raw support and fixed-block trends without interpolating gaps."""
    subjects = sorted(data['meta'].query('cohort == @cohort').subject.unique())
    if len(subjects) != 3:
        raise ValueError('Each cohort must contain three subjects')
    fig, axes = plt.subplots(3, 2, figsize=(183 / 25.4, 192 / 25.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.16, hspace=.70, wspace=.19)
    for row, subject in enumerate(subjects):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            raw = data['trials'][reading].query('iSub == @subject and arm == @arm').sort_values('ordinal_trial')
            block = data['blocks'][reading].query('subject == @subject and arm == @arm').sort_values('trial')
            complete = raw.scope.eq('complete')
            cut, n = int(raw.training_trials.iloc[0]), len(raw)
            ax.axvspan(cut, n, color='#F3F4F6', zorder=0)
            ax.scatter(raw.loc[complete, 'ordinal_trial'], raw.loc[complete, f'{domain}_support'],
                       color=COLORS['support'], s=3, alpha=.19, edgecolors='none', zorder=2)
            # A NaN mean in an empty block breaks each trend line.
            for label in LABELS:
                ax.plot(block.trial, block[f'{domain}_{label}'], color=COLORS[label],
                        ls=STYLES[label], lw=1.25 if label == 'support' else .9,
                        marker='o' if label == 'support' else None, ms=2, zorder=3)
            ax.plot(block.trial, block[f'{domain}_coverage'], color='#527697', ls='-.', lw=.85, zorder=2)
            ax.plot(raw.loc[~complete, 'ordinal_trial'], np.full((~complete).sum(), -.22),
                    '|', color='#6B7077', ms=2, mew=.55, alpha=.5,
                    transform=ax.get_xaxis_transform(), clip_on=False)
            ax.axvline(cut, color='#747B84', ls=':', lw=.8)
            ax.set(xlim=(0, n), ylim=(-.025, 1.025), yticks=[0, .5, 1.])
            ax.tick_params(labelsize=7)
            ax.set_title(f'S{subject} · {NAMES[arm]}\n{complete.sum()} complete reports / {n} trials',
                         fontsize=7.1, pad=5)
            if col == 0:
                ax.set_ylabel('Content compatibility mass', fontsize=7)
            if row == 2:
                ax.set_xlabel('Trial (means in 32-trial blocks)', fontsize=7, labelpad=15)
    handles = [Line2D([], [], color=COLORS[x], ls=STYLES[x], lw=1.3, label=x.capitalize()) for x in LABELS]
    handles += [Line2D([], [], color=COLORS['support'], marker='o', ls='', ms=2.5, alpha=.5,
                       label='Trial support'),
                Line2D([], [], color='#527697', ls='-.', lw=.9, label='Catalogue coverage')]
    fig.suptitle(f'{COHORTS[cohort]}\n{domain.capitalize()} oral alignment · {reading} reading', y=.985, fontsize=10)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, fontsize=7)
    fig.text(.5, .065, 'Lines: means over complete reports. Dots: every complete-report support value.',
             ha='center', fontsize=6.7)
    fig.text(.5, .043, 'Bottom ticks: incomplete / missing reports. Empty blocks remain undefined.', ha='center', fontsize=6.7)
    fig.text(.5, .021, 'Grey background: later behavior excluded from fitting. Coverage: catalogue can support the report.',
             ha='center', fontsize=6.5)
    name = f'oral_trajectories_{cohort}_{domain}_{reading}.png'
    fig.savefig(output / name, dpi=300, facecolor='white')
    plt.close(fig)
    return name


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    data = load_data(source)
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / 'source_snapshot.py')
    own = output / 'source_data'
    own.mkdir()
    for reading in READINGS:
        # Retain every trial, including fragments, invalid encodings and missing reports.
        data['trials'][reading].to_csv(own / f'{reading}_trials.csv', index=False)
        data['blocks'][reading].to_csv(own / f'{reading}_blocks.csv', index=False)
    data['counts'].to_csv(own / 'report_scope_counts.csv', index=False)
    data['meta'].to_csv(own / 'behavior_identity.csv', index=False)
    images = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        for reading in READINGS:
            for domain in DOMAINS:
                for cohort in COHORTS:
                    images.append(plot_grid(data, cohort, reading, domain, output))
    notes = (
        '本图显示行为条件下推断的当前作答依据与当题口述的兼容性。口述带有噪声，未进入参数拟合。'
        '左列为COMPACT，右列为当前Model 1022（精度继承）。每个点是一份完整编码报告的支持概率质量，'
        '即与报告兼容的作答依据所占的权重；'
        '三条趋势线分别为支持、冲突与未确定质量，使用与行为图相同的32题分块，在块末端标记等权均值。'
        '横轴是跨场次连续的有效试次；灰底为后期行为，竖线是参数训练分界。后期状态推断仍逐题使用观察到的行为。'
        '蓝色点划线是完整报告中规则目录可支持的比例，反映表达覆盖。底部短刻度标记未完整编码或缺失的报告；'
        '它们保留在逐题源表，主曲线继续使用原统计图的完整报告分母。空块保持缺测，线段在空块处断开。'
        '本图复用八次滤波汇总的逐题结果，未增加拟合或重新编码，也未计算新的误差区间。'
    )
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 1022逐题口述轨迹</title>'
            '<style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:32px auto;padding:0 20px}'
            'img{width:100%}nav{display:flex;gap:18px;flex-wrap:wrap}section{margin-top:40px}</style>'
            '<h1>Model 1022：口述支持随试次的变化</h1><p>' + notes + '</p><nav>')
    for reading in READINGS:
        for domain in DOMAINS:
            label = ('中点读法' if reading == 'midpoint' else '相对读法') + ' · ' + ('功能' if domain == 'functional' else '结构')
            page += f'<a href="#{reading}_{domain}">{label}</a>'
    page += '</nav>'
    for reading in READINGS:
        for domain in DOMAINS:
            title = ('中点读法' if reading == 'midpoint' else '相对读法') + '：' + ('功能支持' if domain == 'functional' else '结构支持')
            page += f'<section id="{reading}_{domain}"><h2>{title}</h2>'
            for cohort in COHORTS:
                file = f'oral_trajectories_{cohort}_{domain}_{reading}.png'
                page += f'<a href="{file}"><img src="{file}" alt="{html.escape(title + " " + cohort)}"></a>'
            page += '</section>'
    page += '<h2>源表与核对</h2><p>逐题重算及按报告数加权的分块汇总均与原统计结果一致。</p><ul>'
    page += ''.join(f'<li><a href="source_data/{p.name}">{p.name}</a></li>' for p in sorted(own.glob('*.csv')))
    (output / 'index.html').write_text(page + '</ul></html>')
    (output / 'reading_notes.txt').write_text(notes)
    manifest = dict(status='complete', source=str(source.resolve()), source_sha256=digest(Path(__file__)),
                    inputs_sha256=data['bindings'], summary_scalar_checks=data['summary_checks'],
                    backend='python', block_width=BLOCK_WIDTH, trial_axis='ordinal_trial = state_trial + 1',
                    panels=dict(readings=READINGS, domains=DOMAINS, variants=ARMS, cohorts=list(COHORTS)),
                    plot_denominator='scope == complete; identical to existing phase summary',
                    exclusions=data['counts'].query('scope != "complete"').to_dict('records'),
                    empty_block_policy='NaN; do not fill or bridge', all_source_rows_retained=True,
                    uncertainty='No new uncertainty intervals; existing pooled eight-filter point estimates',
                    export='183-mm PNG review figures at 300 dpi; repository PNG-only policy',
                    artifacts_sha256={p.name: digest(p) for p in output.glob('*.png')},
                    data_sha256={p.name: digest(p) for p in own.glob('*.csv')})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(dict(output=str(output), figures=len(images), summary_scalar_checks=data['summary_checks'])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == '__main__':
    main()
