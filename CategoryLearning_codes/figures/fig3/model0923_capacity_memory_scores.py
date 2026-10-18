"""Compare choice likelihood and autonomous learning at the same grid points.

Run --source COMPLETED_CHOICE_SCORES --output NEW_DIRECTORY. The score protocol
identifies its completed autonomous probe. Every parameter cell, scoring phase
and trial remains in the source tables. --smoke permits the eight-trial choice
fixture and marks its different choice/autonomous lengths as layout validation.
Scientific outputs require matching full sequence lengths. Export: PNG only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASES = ('prefix', 'later', 'full')
IDENTITY = ['subject', 'point', 'anchor', 'memory', 'capacity']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def prepare_tables(choice: pd.DataFrame, trials: pd.DataFrame,
                   autonomous: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require identical cells and verify scores against every ordered trial."""
    cells = autonomous[IDENTITY + ['paths', 'energy', 'energy_mcse']].copy()
    if (cells.subject.nunique() != 1 or cells.point.duplicated().any()
            or cells.duplicated(['memory', 'capacity']).any()
            or len(cells) != cells.memory.nunique()*cells.capacity.nunique()
            or cells.anchor.sum() != 1):
        raise ValueError('Require one complete crossed parameter grid with one anchor')
    points = set(cells.point)
    expected = {(point, phase) for point in points for phase in PHASES}
    if (choice.duplicated(['point', 'phase']).any()
            or set(map(tuple, choice[['point', 'phase']].to_numpy())) != expected
            or set(trials.point) != points or trials.duplicated(['point', 'trial']).any()
            or contrasts.point.duplicated().any() or set(contrasts.point) != points):
        raise ValueError('Incomplete or duplicate cells, phases or trials')
    anchor = cells.loc[cells.anchor, 'point'].item()
    if not contrasts.reference.eq(anchor).all():
        raise ValueError('Autonomous contrasts use a different reference')
    for frame in (choice, trials):
        merged = frame[IDENTITY].drop_duplicates().merge(cells[IDENTITY], on=IDENTITY,
                                                        how='outer', indicator=True)
        if not merged['_merge'].eq('both').all() or len(merged) != len(cells):
            raise ValueError('Choice and autonomous parameter identities differ')
    if not choice.model.eq('SEARCH').all() or not trials.model.eq('SEARCH').all():
        raise ValueError('Require the same SEARCH mechanism')
    for frame, metrics in [(choice, ['choice_nll', 'choice_nll_mc_se', 'delta_choice_nll', 'delta_choice_nll_mc_se']),
                           (trials, ['observed_probability', 'choice_nll', 'delta_choice_nll']),
                           (cells, ['energy', 'energy_mcse']),
                           (contrasts, ['delta_energy', 'delta_energy_mcse'])]:
        if not np.isfinite(frame[metrics].to_numpy(float)).all():
            raise ValueError('Non-finite comparison values')
        if any(frame[col].lt(0).any() for col in metrics if col.endswith(('mc_se', 'mcse'))):
            raise ValueError('Negative simulation standard error')
    if not trials.observed_probability.between(0, 1).all() or not trials.correct.isin([0, 1]).all():
        raise ValueError('Invalid choice probabilities or observed correctness')
    for name in ('particles', 'seeds'):
        if choice[name].nunique() != 1 or trials[name].nunique() != 1 or choice[name].iloc[0] != trials[name].iloc[0]:
            raise ValueError('Choice numerical budgets differ across cells')
    original = trials[trials.point.eq(anchor)].sort_values('trial')
    count = len(original)
    cut = int(choice.loc[choice.point.eq(anchor) & choice.phase.eq('prefix'), 'trials'].item())
    if not 0 < cut < count:
        raise ValueError('Require an internal choice time split')
    original_scores = choice[choice.point.eq(anchor)].set_index('phase')
    ordered = []
    for point, part in trials.groupby('point'):
        part = part.sort_values('trial').copy()
        np.testing.assert_array_equal(part.trial, np.arange(1, count+1))
        np.testing.assert_array_equal(part.correct, original.correct)
        np.testing.assert_array_equal(part.phase, np.where(part.trial <= cut, 'prefix', 'later'))
        loss = -np.log(np.maximum(part.observed_probability.to_numpy(), np.finfo(float).tiny))
        np.testing.assert_allclose(part.choice_nll, loss, rtol=0, atol=1e-12)
        np.testing.assert_allclose(part.delta_choice_nll, loss-original.choice_nll.to_numpy(), rtol=0, atol=1e-12)
        for phase, start, stop in [('prefix', 0, cut), ('later', cut, count), ('full', 0, count)]:
            row = choice[choice.point.eq(point) & choice.phase.eq(phase)].iloc[0]
            if row.trials != stop-start:
                raise ValueError('Choice score length differs from trial table')
            np.testing.assert_allclose(row.choice_nll, loss[start:stop].mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(row.delta_choice_nll,
                row.choice_nll-original_scores.loc[phase, 'choice_nll'], rtol=0, atol=1e-12)
        part['cumulative_delta_choice_nll'] = part.delta_choice_nll.cumsum()
        ordered.append(part)
    cells = cells.merge(contrasts, on=['subject', 'point'], validate='one_to_one', how='outer', indicator=True)
    if not cells['_merge'].eq('both').all():
        raise ValueError('Autonomous contrast subject differs from its parameter cell')
    cells = cells.drop(columns='_merge')
    np.testing.assert_allclose(cells.delta_energy,
        cells.energy-cells.loc[cells.anchor, 'energy'].item(), rtol=0, atol=1e-12)
    table = choice.merge(cells, on=IDENTITY, validate='many_to_one')
    return table.sort_values(['memory', 'capacity', 'phase']), pd.concat(ordered, ignore_index=True)


def load_tables(source: Path, smoke: bool) -> tuple[dict, dict]:
    completion, protocol = read(source/'completion.json'), read(source/'protocol.json')
    if completion['status'] != 'complete' or completion['smoke'] != smoke or protocol['smoke'] != smoke:
        raise ValueError('Need completed choice scores; fixtures require --smoke')
    probe = Path(protocol['probe'])
    done, summary = read(probe/'completion.json'), read(probe/'summary.json')
    if done['status'] != 'complete' or done['smoke'] or digest(probe/'summary.json') != done['summary_sha256']:
        raise ValueError('Need the completed scientific autonomous probe')
    inputs = {str(source/name): digest(source/name) for name in ('completion.json', 'protocol.json')}
    for name, expected in completion['artifacts_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError('Choice score table changed')
        inputs[str(source/name)] = expected
    for name in ('protocol.json', 'completion.json', 'summary.json', 'scores.csv', 'contrasts.csv', 'curves.csv'):
        file = probe/name
        if digest(file) != protocol['input_sha256'][str(file.resolve())]:
            raise ValueError('Autonomous input differs from the scored parameter comparison')
        if name.endswith('.csv') and digest(file) != summary['artifacts'][name]:
            raise ValueError('Autonomous table changed')
        inputs[str(file)] = digest(file)
    choice = pd.read_csv(source/'scores.csv'); trial = pd.read_csv(source/'trial_scores.csv')
    autonomous = pd.read_csv(probe/'scores.csv'); contrasts = pd.read_csv(probe/'contrasts.csv')
    table, cumulative = prepare_tables(choice, trial, autonomous, contrasts)
    curves = pd.read_csv(probe/'curves.csv')
    counts = curves.groupby('point').trial.max()
    if counts.nunique() != 1 or set(counts.index) != set(table.point):
        raise ValueError('Autonomous sequences have different lengths or cells')
    n = int(cumulative.trial.max()); generated_n = int(counts.iloc[0])
    if n != completion['trials'] or (not smoke and n != generated_n):
        raise ValueError('Choice and autonomous sequence lengths differ')
    if not smoke:
        observed = cumulative[cumulative.anchor].sort_values('trial').correct.to_numpy()
        for _, group in curves.groupby('point'):
            start = 0
            for row in group.sort_values('trial').itertuples():
                np.testing.assert_allclose(row.observed, observed[start:int(row.trial)].mean(), rtol=0, atol=1e-12)
                start = int(row.trial)
    context = dict(subject=int(table.subject.iloc[0]), anchor=table.loc[table.anchor, 'point'].iloc[0],
        choice_trials=n, autonomous_trials=generated_n, prefix_trials=completion['training_trials'],
        particles=int(table.particles.iloc[0]), filter_seeds=int(table.seeds.iloc[0]),
        paths_per_cell=sorted(int(x) for x in table.paths.unique()), smoke=smoke,
        inputs_sha256=inputs, source_rows=dict(choice=len(choice), trials=len(trial), autonomous=len(autonomous), contrasts=len(contrasts)))
    return dict(scores=table, trials=cumulative), context


def draw(tables: dict, context: dict, output: Path) -> None:
    scores, trials = tables['scores'], tables['trials']
    cells = scores[scores.phase.eq('full')].sort_values(['memory', 'capacity'])
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':7,
        'axes.labelsize':7, 'axes.titlesize':7, 'xtick.labelsize':6, 'ytick.labelsize':6,
        'legend.fontsize':6, 'legend.frameon':False, 'axes.linewidth':.6,
        'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig = plt.figure(figsize=(183/25.4, 145/25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], left=.18, right=.97, bottom=.23, top=.86,
                          wspace=.38, hspace=.65)
    ax_a, ax_b, ax_c = fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1]), fig.add_subplot(grid[1,:])
    phase_style = {'prefix':('#8B8B8B', 'o', -.2), 'later':('#B97845', '^', 0), 'full':('#376E88', 's', .2)}
    for phase in PHASES:
        color, marker, offset = phase_style[phase]
        part = scores[scores.phase.eq(phase)].set_index('point').loc[cells.point]
        ax_a.errorbar(part.delta_choice_nll, np.arange(len(cells))+offset, xerr=part.delta_choice_nll_mc_se,
            fmt=marker, color=color, ms=3, elinewidth=.7, capsize=1.5, label=phase.capitalize())
    ax_b.errorbar(cells.delta_energy, np.arange(len(cells)), xerr=cells.delta_energy_mcse,
        fmt='o', color='#4C8275', ms=3.5, elinewidth=.8, capsize=2)
    labels = [f'γ={row.memory:g}, K={int(row.capacity)}'+('  original' if row.anchor else '') for row in cells.itertuples()]
    for ax in (ax_a, ax_b):
        ax.axvline(0, color='#AAAAAA', lw=.6, zorder=0)
        ax.set_ylim(len(cells)-.5, -.5); ax.set_yticks(np.arange(len(cells)))
    ax_a.set_yticklabels(labels); ax_b.set_yticklabels([])
    ax_a.set(xlabel='Δ mean choice NLL', title='a  Observed-choice prediction')
    ax_b.set(xlabel='Δ autonomous Energy score', title='b  Whole learning-curve distribution')
    ax_a.legend(loc='upper center', bbox_to_anchor=(.5, -.27), ncol=3, columnspacing=.8, handletextpad=.3)
    levels = sorted(cells.memory.unique())
    palette = ['#777777', '#39758B', '#A57248']
    colors = dict(zip(levels, palette if len(levels) <= 3 else plt.cm.viridis(np.linspace(.15,.85,len(levels)))))
    capacities = sorted(cells.capacity.unique())
    for row in cells.itertuples():
        part = trials[trials.point.eq(row.point)].sort_values('trial')
        ax_c.plot(part.trial, part.cumulative_delta_choice_nll, color=colors[row.memory],
            ls='-' if row.capacity == max(capacities) else '--', lw=1.,
            label=f'γ={row.memory:g}, K={int(row.capacity)}'+(' (original)' if row.anchor else ''))
    ax_c.axhline(0, color='#BBBBBB', lw=.6, zorder=0)
    ax_c.axvline(context['prefix_trials']+.5, color='#999999', lw=.7, ls=':')
    ax_c.set(xlabel='Trial', ylabel='Cumulative Δ choice NLL', xlim=(1,context['choice_trials']),
             title='c  When each parameter point gains or loses choice likelihood')
    ax_c.legend(loc='upper center', bbox_to_anchor=(.5,-.29), ncol=3, columnspacing=1., handlelength=2.3)
    title=f'Subject {context["subject"]}: likelihood and autonomous learning at identical parameter points'
    if context['smoke']:
        title=f'SMOKE LAYOUT: {context["choice_trials"]} choice trials; {context["autonomous_trials"]} autonomous trials'
    fig.suptitle(title,fontsize=9,y=.98)
    fig.text(.5,.925,'All differences relative to the original point; negative values indicate improvement.',ha='center',fontsize=7)
    fig.text(.5,.065,f'{context["filter_seeds"]} filter seeds × {context["particles"]} particles; '
             f'{"/".join(map(str,context["paths_per_cell"]))} autonomous paths per cell. Bars: 1 paired Monte Carlo SE.',ha='center',fontsize=6)
    fig.text(.5,.025,'Retrospective development comparison; oral reports excluded. γ: belief retention; K: active rule capacity.',ha='center',fontsize=6)
    fig.savefig(output/'capacity_memory_scores.png',dpi=300);plt.close(fig)


def plot(source: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new parameter comparison figure directory')
    tables, context = load_tables(source, smoke)
    output.mkdir(parents=True)
    contract = dict(context, claim='Compare which parameter points explain observed choices and generate plausible learning curves.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=145, dpi=300, formats=['PNG'],
        panels=['all cells and all three choice phases', 'all cells autonomous Energy differences', 'every trial cumulative choice-loss difference'],
        statistics='Mean NLL pools filter probabilities before logging; Energy scores complete sampled-choice curve distributions.',
        uncertainty='1 paired whole-filter or whole-path jackknife Monte Carlo SE; parameter and participant uncertainty excluded',
        data_use='retrospective development; all other parameters fixed; no oral input; all cells and trials retained',
        smoke_boundary='Choice fixture and autonomous probe have different lengths; layout validation only.' if smoke else None,
        reuse='Style only from existing longitudinal comparison figures; table matching and transformations checked independently.',
        source_preflight_exceptions={'EXPORT-VECTOR':'Repository PNG-only instruction.', 'EXPORT-RASTER':'PNG-only instruction.',
                                     'RASTER-DPI':'300 dpi discussion figure.'})
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    tables['scores'].to_csv(output/'plot_scores.csv',index=False)
    tables['trials'].to_csv(output/'plot_trials.csv',index=False)
    draw(tables,context,output)
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>保持率、容量与行为似然</title>',
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{max-width:100%}td,th{padding:5px}</style>',
        '<h1>同一组参数的行为似然与自主学习</h1>',
        '<p>每行对应一组保持率γ和容量K。所有差值均为该组减去原参数的得分，负值表示改善。'
        'a保留前段、后段及全程选择负对数似然；b比较整条自主正确率曲线分布的Energy评分；'
        'c逐题累加选择损失差，向下表示该段实际选择获得更高概率。虚线标出原训练范围的末尾。</p>',
        '<p>误差棒为固定参数下的一个模拟标准误，分别按完整滤波或自主路径配对计算。'
        '完整行为参与这项回顾性开发比较，口述未参与评分。所有参数格、时间范围和逐题数据保留在源表中。</p>']
    if smoke:
        page.append('<p><strong>短序列绘图验证：行为评分只覆盖前8题，自主评分覆盖完整序列；本图用于检查实现。</strong></p>')
    page.append('<img src="capacity_memory_scores.png" alt="六格参数的行为和自主评分对照">')
    page.append('<p><a href="plot_scores.csv">完整评分源表</a> · <a href="plot_trials.csv">全部逐题源表</a></p>')
    columns=['memory','capacity','phase','trials','choice_nll','delta_choice_nll','delta_choice_nll_mc_se','energy','delta_energy','delta_energy_mcse']
    page.append(tables['scores'][columns].to_html(index=False,float_format=lambda x:f'{x:.5f}'))
    (output/'index.html').write_text(''.join(page))
    (output/'manifest.json').write_text(json.dumps(dict(status='complete',smoke=smoke,
        source=str(source.resolve()),source_sha256=digest(Path(__file__)),
        artifacts={p.name:digest(p) for p in output.iterdir()}),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();plot(args.source,args.output,args.smoke)
