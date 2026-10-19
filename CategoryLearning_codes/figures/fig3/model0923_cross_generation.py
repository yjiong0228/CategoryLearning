"""Plot the completed five-cell autonomous component of the SEARCH cross.

Run from the repository root with --cross CROSS --generation-check CHECK
--output NEW_DIRECTORY. All 640 paths are retained. The figure shows late
accuracy distributions, mechanism/parameter interaction, target-rule presence
versus contribution, and phase-specific Energy. Intervals describe the central
80% of simulated paths, not uncertainty about participants. Outputs are PNG,
source CSVs, provenance, and an HTML discussion page. Conditional and oral
results are separate pending components; this figure does not select parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_online_parameter_cross_audit import distribution

CELLS = ('RETAIN_RETAIN', 'ONLINE_RETAIN', 'RETAIN_ONLINE', 'ONLINE_ONLINE', 'ONLINE_NEAR')
NAMES = {'RETAIN_RETAIN':'RETAIN / old', 'ONLINE_RETAIN':'ONLINE / old',
         'RETAIN_ONLINE':'RETAIN / new', 'ONLINE_ONLINE':'ONLINE / new', 'ONLINE_NEAR':'ONLINE / near'}
COLORS = {'RETAIN':'#4678A5', 'ONLINE':'#CC8152'}
STYLES = {'RETAIN':'-', 'ONLINE':'--', 'NEAR':':'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(cross: Path, check: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new cross-generation figure directory')
    proof = json.loads((check / 'completion.json').read_text())
    if proof['status'] != 'complete' or proof['cells'] != 5 or proof['paths'] != 640:
        raise ValueError('Need all completed own-choice paths')
    inputs = dict(proof['inputs_sha256'])
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Generation input changed: ' + path)
    protocol_file = cross / 'protocol.json'
    if inputs[str(protocol_file.resolve())] != digest(protocol_file):
        raise ValueError('Generation check belongs to another cross')
    protocol = json.loads(protocol_file.read_text())
    for file in (check / 'completion.json', check / 'scores.csv', check / 'paths.csv', Path(__file__)):
        inputs[str(file.resolve())] = digest(file)
    scores = pd.read_csv(check / 'scores.csv').set_index('cell').loc[list(CELLS)]
    paths = pd.read_csv(check / 'paths.csv')
    if set(paths.cell) != set(CELLS) or paths.duplicated(['cell', 'seed']).any():
        raise ValueError('Missing or duplicate path identity')
    folder = Path(protocol['banks'][0]['roles']['evaluate'][0]['job']['folder'])
    prepared = json.loads((folder / 'prepared.json').read_text())
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        with np.load(folder / name) as z: parts.append(z['choices'].copy())
    human_choices = np.concatenate(parts); cut = prepared['training_trials']
    if prepared['subject'] != 208 or cut != 384 or len(human_choices) != 576:
        raise ValueError('This figure uses the declared participant208 trial schedule')
    width = protocol['settings']['features']['block_width']
    diagnostics, intervals = [], []; checks = 0
    for bank in protocol['banks']:
        cell = bank['cell']; chunks = []
        for item in bank['roles']['autonomous']:
            file = Path(item['job']['target']).with_suffix('.npz')
            if inputs[str(file.resolve())] != digest(file):
                raise ValueError('Path array differs from the completed generation check')
            with np.load(file) as z: chunks.append({k:z[k].copy() for k in z.files})
        arrays = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0] if k != 'categories'}
        categories = chunks[0]['categories']; observed = human_choices == categories
        generated = arrays['choices'] == categories; present = arrays['present'].astype(bool)
        p_error = 1 - arrays['correct_probability']; n = len(categories)
        data = paths[paths.cell == cell].set_index('seed').loc[arrays['seeds']]
        if len(data) != 128 or len(set(arrays['seeds'])) != 128:
            raise ValueError('Need every original independent trajectory seed')
        late = generated[:, -128:].mean(1)
        for key, values in dict(late_accuracy=late, target_late_presence=present[:, -128:].mean(1),
                                target_late_contribution=arrays['contribution'][:, -128:].mean(1)).items():
            np.testing.assert_allclose(data[key], values, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores.loc[cell, key], values.mean(), atol=1e-12, rtol=0)
            checks += len(values) + 1
        low, high = np.quantile(late, [.1, .9])
        intervals.append(dict(cell=cell, paths=len(late), mean=late.mean(), lower80=low, upper80=high,
                              human_late_accuracy=observed[-128:].mean()))
        for phase, start, stop in [('training', 0, cut), ('test', cut, n), ('whole', 0, n), ('late128', n-128, n)]:
            active = present[:, start:stop]; errors = p_error[:, start:stop]
            contribution = arrays['contribution'][:, start:stop]
            row = dict(cell=cell, phase=phase, trials=stop-start, paths=len(late),
                expected_error=float(errors.mean()),
                error_with_target_present=float((errors * active).mean()),
                error_with_target_absent=float((errors * ~active).mean()),
                target_presence=float(active.mean()),
                target_contribution=float(contribution.mean()),
                target_contribution_given_presence=float(contribution[active].mean()),
                target_beta_given_presence=float(arrays['beta'][:, start:stop][active].mean()))
            np.testing.assert_allclose(row['expected_error'], row['error_with_target_present'] + row['error_with_target_absent'], atol=1e-12)
            diagnostics.append(row); checks += 1
            if phase != 'late128':
                x = np.stack([generated[:, i:min(i+width, stop)].mean(1) for i in range(start, stop, width)], axis=1)
                y = np.array([observed[i:min(i+width, stop)].mean() for i in range(start, stop, width)])
                np.testing.assert_allclose(scores.loc[cell, phase+'_energy'], distribution(x, y)['energy'], atol=1e-12, rtol=0)
                checks += 1
    interval = pd.DataFrame(intervals).set_index('cell').loc[list(CELLS)]
    output.mkdir(parents=True)
    contract = dict(conclusion='Belief mechanism effects on autonomous learning reverse across frozen parameter points; target-rule presence alone does not ensure use.',
        archetype='quantitative grid', backend='python', dimensions_mm=[183, 174], dpi=300, export=['PNG'],
        scope='Discussion draft: completed autonomous component; conditional and oral components pending',
        panels={'a':'Same-parameter mechanism effects and central 80% path range',
                'b':'Empirical late-accuracy distributions of all paths',
                'c':'Target presence versus unconditional contribution', 'd':'Training, later and whole-sequence Energy'},
        biological_replicates=1, simulated_paths_per_cell=128, exclusions=0, block_width=width,
        intervals='10th to 90th percentiles of simulated paths; not confidence intervals',
        inference='Descriptive development comparison; no hypothesis test or participant generalization',
        original_nominations_changed=False)
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    matplotlib.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':8,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7, 'legend.frameon':False,
        'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 174/25.4))
    fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.20, wspace=.32, hspace=.48)
    titles = ['Mechanism effect depends on parameters', 'Every generated path retained',
              'Presence and contribution separate', 'Learning phases show different trade-offs']
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, loc='left', fontsize=8, pad=9)
        ax.text(-.16, 1.06, 'abcd'[i], transform=ax.transAxes, weight='bold', fontsize=10)
        ax.tick_params(labelsize=7)
    accuracy_floor = max(0., min(.2, paths.late_accuracy.min()-.02))
    ax = axes[0, 0]
    for mechanism in ('RETAIN', 'ONLINE'):
        for parameter, x in [('RETAIN',0), ('ONLINE',1), ('NEAR',2)]:
            cell = mechanism + '_' + parameter
            if cell not in interval.index: continue
            row = interval.loc[cell]; offset = -.045 if mechanism == 'RETAIN' else .045
            ax.vlines(x+offset, row.lower80, row.upper80, color=COLORS[mechanism], lw=1.5, alpha=.7)
            ax.plot(x+offset, row['mean'], 'o' if mechanism == 'RETAIN' else 's', color=COLORS[mechanism], ms=5)
        ax.plot([-.045 if mechanism=='RETAIN' else .045, .955 if mechanism=='RETAIN' else 1.045],
                [interval.loc[mechanism+'_RETAIN','mean'], interval.loc[mechanism+'_ONLINE','mean']],
                color=COLORS[mechanism], lw=1.)
    human = interval.human_late_accuracy.iloc[0]
    ax.axhline(human, color='.25', ls=':', lw=1)
    ax.text(1.95, human+.018, 'Participant', ha='right', fontsize=7)
    ax.set(xticks=[0,1,2], xticklabels=['Old parameters','New parameters','Near candidate'],
           ylabel='Late accuracy', ylim=(accuracy_floor,1.02), xlim=(-.3,2.25))
    ax.text(.02,.02,'Dots: means; lines: central 80% of paths',transform=ax.transAxes,fontsize=6.5)
    ax = axes[0, 1]
    for cell in CELLS:
        mechanism, parameter = cell.split('_'); values = np.sort(paths.loc[paths.cell == cell, 'late_accuracy'])
        ax.step(values, np.arange(1,len(values)+1)/len(values), where='post',
                color=COLORS[mechanism], ls=STYLES[parameter], lw=1.4)
    ax.axvline(human, color='.25', ls=':', lw=1)
    ax.set(xlim=(accuracy_floor,1.), ylim=(0,1.02), xlabel='Late accuracy', ylabel='Fraction of paths at or below accuracy')
    ax = axes[1, 0]
    for x, cell in enumerate(CELLS):
        color = COLORS[cell.split('_')[0]]; row = scores.loc[cell]
        ax.plot([x,x], [row.target_late_contribution,row.target_late_presence], color=color, lw=1.2)
        ax.plot(x,row.target_late_presence,'o',mfc='white',mec=color,ms=5)
        ax.plot(x,row.target_late_contribution,'o',color=color,ms=5)
    ax.set(xticks=range(5), xticklabels=['R\nold','O\nold','R\nnew','O\nnew','O\nnear'],
           ylim=(0,1.07), ylabel='Late mean across paths and trials')
    ax.legend(handles=[Line2D([],[],marker='o',mfc='white',color='.3',ls='',label='Presence'),
                       Line2D([],[],marker='o',color='.3',ls='',label='Contribution')],loc='lower left',fontsize=7)
    ax = axes[1, 1]
    for cell in CELLS:
        mechanism, parameter = cell.split('_')
        ax.plot(range(3), scores.loc[cell,['training_energy','test_energy','whole_energy']].to_numpy(float),
                color=COLORS[mechanism], ls=STYLES[parameter], marker='o', ms=3, lw=1.2)
    ax.set(xticks=range(3), xticklabels=['Training','Later','Whole'], ylabel='Energy (lower is better)', ylim=(0,.23))
    handles = [Line2D([],[],color=COLORS[c.split('_')[0]],ls=STYLES[c.split('_')[1]],label=NAMES[c],lw=1.5) for c in CELLS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.075), ncol=3, fontsize=7.5)
    fig.suptitle('SEARCH cross: mechanism and parameters interact', fontsize=11, y=.97)
    fig.text(.5,.925,'Participant 208 | 128 paths per cell | Fixed behavior-selected parameter points',ha='center',fontsize=8)
    fig.text(.5,.025,'Autonomous component complete; matched conditional and oral comparisons pending.',ha='center',fontsize=7)
    fig.savefig(output / 'cross_generation.png', dpi=300)
    plt.close(fig)
    interval.reset_index().to_csv(output / 'accuracy_intervals.csv', index=False)
    paths.to_csv(output / 'source_paths.csv', index=False)
    scores.reset_index().to_csv(output / 'source_scores.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output / 'error_decomposition.csv', index=False)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SEARCH机制与参数交叉：自主生成</title>'
        '<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto;line-height:1.6}img{width:100%}</style>'
        '<h1>相同机制在不同参数下产生不同学习过程</h1><p>被试208；每格128条完整自主路径，全部640条均保留。'
        'R表示RETAIN，O表示ONLINE。旧参数、新参数和接近候选均由既有训练行为估计或确认获得，口述用于后续检验。</p>'
        '<img src="cross_generation.png" alt="五组自主路径的准确率分布、规则在场与贡献，以及分段Energy">'
        '<p>a：均值及路径的第10至90百分位；连接线表示固定参数下的机制比较。'
        'b：全部路径的经验累积分布，竖线为被试末128题正确率90.625%。'
        'c：正确规则的在场比例与平均作答贡献；未在场时贡献记为0。'
        f'd：前384题、后192题和全部576题的Energy，按{width}题分块计算，越低越好。</p>'
        '<p>区间表示模型生成的路径差异。这里只有一名开发被试，图中未进行假设检验。'
        '条件预测与口述对照仍在计算。完整参数评价需结合这些证据以及训练阶段的预测代价。</p>'
        '<p>error_decomposition.csv进一步把期望错误概率分解为正确规则在场与缺席两部分；两部分使用相同总试次数作分母，和等于总体期望错误概率。</p>'
        '<p><a href="source_paths.csv">全部路径</a> · <a href="source_scores.csv">分段得分</a> · '
        '<a href="accuracy_intervals.csv">路径区间</a> · <a href="error_decomposition.csv">错误分解</a></p></html>')
    (output / 'index.html').write_text(page)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha: raise ValueError('Figure input changed during rendering')
    manifest = dict(status='rendered_pending_visual_review', numerical_checks=checks, cells=5, paths=640,
        source_sha256=digest(Path(__file__)), inputs_sha256=inputs, original_nominations_changed=False,
        versions={n:importlib.metadata.version(n) for n in ('numpy','pandas','matplotlib')},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(dict(status=manifest['status'], numerical_checks=checks, paths=640), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cross','generation-check','output'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); draw(args.cross,args.generation_check,args.output)
