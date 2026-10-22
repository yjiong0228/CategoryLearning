"""Show where frozen parameter combinations recover the same latent path.

Figure contract: behavioral fit, rule membership, rule use and precision have
different recovery properties. A quantitative grid compares every audited point
and the generating point against the same known trajectory. Both inference
timings, all trials and all three recorded state quantities are retained.
Columns identify parameters; rows identify state quantities. Style-only reuse
of the established Python figures preserves their typography and role colors.

Use --summary COMPLETE_RECOVERY_CANDIDATE_SUMMARY --output NEW. All values and
weighted 5--95% latent-state intervals are read from the original summaries.
Precision is conditional on rule presence; absence stays undefined. No smoothing,
simulation, fitting or participant-level hypothesis test is performed. Exports
are 183-mm, 300-dpi PNG review figures, source CSVs and an HTML reading guide.
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1023_estimation_reliability import read, write, digest
from .model1023_estimation_reliability import STYLE

PHASES = dict(predictive='Before current choice and feedback',
              filtered='After current choice, before feedback')


def point_labels(scores: pd.DataFrame) -> list[dict]:
    if (len(scores) != 4 or scores.selected.sum() != 1 or scores.reference.sum() != 1
            or scores.generating.sum() != 1 or scores.duplicated(['candidate', 'generating']).any()):
        raise ValueError('Require three audited points plus one generating-point evaluation')
    ordered = scores.sort_values(['generating', 'selected', 'reference', 'candidate'],
                                 ascending=[True, False, False, True])
    rows, alternative = [], 0
    for row in ordered.to_dict('records'):
        if row['generating']:
            label, color = 'Generating parameters', '#7B6BA5'
        elif row['selected']:
            label, color = ('Selected / reference' if row['reference'] else 'Selected'), '#3B8C85'
        elif row['reference']:
            label, color = 'Reference', '#527697'
        else:
            alternative += 1
            label, color = 'Alternative '+str(alternative), '#9B775D'
        rows.append(dict(row, label=label, color=color))
    return rows


def load_data(summary: Path) -> tuple:
    done = read(summary/'completion.json')
    if done['status'] != 'complete' or done['smoke'] or done['additional_filters']:
        raise ValueError('Require a full original-filter recovery comparison')
    inputs = {str((summary/'completion.json').resolve()): digest(summary/'completion.json')}
    for name, sha in done['inputs_sha256'].items():
        if digest(Path(name)) != sha:
            raise ValueError('Underlying state-summary input changed: '+name)
        inputs[name] = sha
    case_file = (Path(done['study'])/'cases.json').resolve()
    if str(case_file) not in inputs:
        raise ValueError('Case identities must be bound to the recovery summary')
    cases = {row['case_id']: row for row in read(case_file)['cases']}
    tables = {}
    for name, sha in done['artifacts_sha256'].items():
        p = summary/name
        if digest(p) != sha:
            raise ValueError('State-summary table changed: '+name)
        inputs[str(p.resolve())] = sha
        tables[p.stem] = pd.read_csv(p)
    scores, trials = tables['scores'], tables['trials']
    if (scores.case_id.nunique() != done['cases'] or set(scores.phase) != set(PHASES)
            or set(trials.case_id) != set(scores.case_id)
            or not scores.filters.eq(8).all() or not scores.particles.eq(1024).all()):
        raise ValueError('Incomplete recovery cases or inference timings')
    for case_id, group in scores.groupby('case_id'):
        truth = None
        for phase in PHASES:
            for row in point_labels(group[group.phase == phase]):
                own = trials[(trials.case_id == case_id) & (trials.phase == phase)
                             & (trials.candidate == row['candidate']) & (trials.generating == row['generating'])].sort_values('trial')
                np.testing.assert_array_equal(own.trial, np.arange(1, row['trials']+1))
                actual = own[['truth_present', 'truth_contribution', 'truth_beta']].to_numpy()
                if truth is not None:
                    np.testing.assert_array_equal(actual, truth)
                truth = actual
                probabilities = own[['presence_probability', 'contribution_mean']].to_numpy()
                if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1+1e-12)):
                    raise ValueError('Invalid full-trial state probabilities')
                if not np.array_equal(np.isfinite(own.conditional_beta_mean), own.presence_probability > 0):
                    raise ValueError('Precision availability differs from rule presence')
                np.testing.assert_allclose(((own.presence_probability-own.truth_present)**2).mean(), row['membership_brier'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(np.sqrt(((own.contribution_mean-own.truth_contribution)**2).mean()), row['contribution_rmse'], atol=1e-12, rtol=0)
                mask = own.truth_present.astype(bool) & np.isfinite(own.conditional_beta_mean)
                if int(mask.sum()) != row['conditional_beta_scored_trials']:
                    raise ValueError('Conditional precision trial denominator differs')
                precision_error = np.sqrt(((own.loc[mask, 'conditional_beta_mean']-own.loc[mask, 'truth_beta'])**2).mean()) if mask.any() else np.nan
                np.testing.assert_allclose(precision_error, row['conditional_beta_rmse'], atol=1e-12, rtol=0, equal_nan=True)
                for key in ('contribution', 'conditional_beta'):
                    q = own[[key+'_q05', key+'_q50', key+'_q95']].to_numpy()
                    valid = np.isfinite(q).all(1)
                    if (np.any(np.isfinite(q).any(1) != valid) or np.any(q[valid] < 0)
                            or np.any(np.diff(q[valid], axis=1) < 0)):
                        raise ValueError('Invalid latent-state interval')
    inputs[str(Path(__file__).with_name('model1023_estimation_reliability.py').resolve())] = digest(Path(__file__).with_name('model1023_estimation_reliability.py'))
    return tables, inputs, cases


def render(summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new candidate-recovery figure directory')
    tables, inputs, cases = load_data(summary)
    output.mkdir(parents=True)
    scores, trials = tables['scores'], tables['trials']
    files = []
    with plt.rc_context(STYLE):
        for case_id in sorted(scores.case_id.unique()):
            case_trials = trials[trials.case_id == case_id]
            upper = 1.05*max(1, case_trials.conditional_beta_q95.max(), case_trials.truth_beta.max())
            for phase, timing in PHASES.items():
                points = point_labels(scores[(scores.case_id == case_id) & (scores.phase == phase)])
                fig, axes = plt.subplots(3, 4, figsize=(183/25.4, 180/25.4), sharex=True, sharey='row')
                fig.subplots_adjust(left=.10, right=.985, bottom=.18, top=.79, wspace=.20, hspace=.30)
                for col, point in enumerate(points):
                    own = case_trials[(case_trials.phase == phase) & (case_trials.candidate == point['candidate'])
                                      & (case_trials.generating == point['generating'])].sort_values('trial')
                    actual_beta = own.truth_beta.where(own.truth_present.astype(bool))
                    for row, (mean, actual, label) in enumerate([
                        ('presence_probability', own.truth_present, 'Rule presence probability'),
                        ('contribution_mean', own.truth_contribution, 'Rule contribution'),
                        ('conditional_beta_mean', actual_beta, 'Precision given presence'),
                    ]):
                        ax = axes[row, col]
                        if row:
                            prefix = 'contribution' if row == 1 else 'conditional_beta'
                            ax.fill_between(own.trial, own[prefix+'_q05'], own[prefix+'_q95'],
                                            color=point['color'], alpha=.18, linewidth=0)
                        ax.plot(own.trial, actual, color='#222222', lw=.8,
                                drawstyle='steps-post' if row == 0 else 'default',
                                marker='.' if row == 2 else None, markersize=1.5, zorder=3)
                        ax.plot(own.trial, own[mean], color=point['color'], lw=1.0, zorder=4)
                        ax.set(xlim=(1, len(own)), ylim=(-.03, 1.03) if row < 2 else (0, upper),
                               xticks=[1, len(own)//2, len(own)])
                        ax.text(-.12, 1.04, chr(97+4*row+col), transform=ax.transAxes, fontsize=8, fontweight='bold')
                        if row < 2:
                            ax.set_yticks([0, .5, 1])
                        if col == 0:
                            ax.set_ylabel(label)
                        if row == 2:
                            ax.set_xlabel('Trial')
                    axes[0, col].set_title(point['label']+f'\nNLL / trial = {point["choice_nll"]:.4f}', fontsize=6.8, pad=8)
                target = cases[case_id]['settings']['recovery_target_rule']
                fig.suptitle(case_id+f' · target rule {target} · fixed parameter combinations', y=.985, fontsize=9)
                fig.text(.5, .935, timing+f' · {int(points[0]["filters"])} filters × {int(points[0]["particles"])} particles per point', ha='center', fontsize=7)
                fig.legend(handles=[Line2D([], [], color='#222222', lw=.8, label='Known simulated path'),
                    Line2D([], [], color='#3B8C85', lw=1, label='Inferred mean'),
                    Patch(facecolor='#527697', alpha=.18, edgecolor='none', label='Pointwise 5–95% state interval (rows 2–3)')],
                    loc='upper center', bbox_to_anchor=(.5, .90), ncol=3, fontsize=6.5)
                fig.text(.5, .09, 'All trials and fixed candidates shown; intervals condition on each parameter point.', ha='center', fontsize=6.5)
                fig.text(.5, .055, 'Precision is defined when the rule is present. Parameters were estimated from the full simulated sequence.', ha='center', fontsize=6.5)
                present = int(own.truth_present.sum())
                fig.text(.5, .025, f'Known target present on {present} / {len(own)} trials; black precision marks appear only on those trials.', ha='center', fontsize=6.5)
                name = case_id+'_'+phase+'.png'
                fig.savefig(output/name, dpi=300, facecolor='white')
                plt.close(fig)
                files.append(name)
    for name, table in tables.items():
        table.to_csv(output/(name+'.csv'), index=False)
    page = '<!doctype html><meta charset="utf-8"><title>固定参数的规则状态恢复</title>'
    page += '<style>body{max-width:1100px;margin:32px auto;font:16px/1.7 sans-serif}img{width:100%}</style>'
    page += '<h1>固定参数的规则状态恢复</h1><p>每列使用一组固定参数，各列对照同一条真实模拟路径。'
    page += '三行依次显示目标规则的存在概率、作答贡献和存在条件下的精度。黑线为真实路径，彩线为推断均值。</p>'
    page += '<p>第一行的概率直接表示对规则在场的判断。第二、三行的阴影为固定参数下逐题5%至95%的状态分位区间。'
    page += '规则经常缺席时，贡献分布可在零点集中，均值与分位区间可以不对称。规则缺席时，真实精度保留为空。'
    page += '生成参数列显示已知参数下的状态推断。它与黑线的差异反映行为数据对隐藏路径的分辨限制及数值误差。</p>'
    page += '<p>每条序列分别呈现当前选择之前，以及选择之后、当前反馈之前的状态。参数由完整模拟行为序列估计。'
    page += '全部受检候选和全部试次均保留，有限候选集合用于比较解释的敏感性。</p>'
    page += ''.join('<h2>'+html.escape(name.removesuffix('.png'))+'</h2><img src="'+name+'">' for name in files)
    (output/'index.html').write_text(page, encoding='utf-8')
    shutil.copyfile(Path(__file__), output/'source_snapshot.py')
    write(output/'completion.json', dict(status='complete', summary=str(summary.resolve()),
        source_sha256=digest(Path(__file__)), inputs_sha256=inputs, backend='python', format='PNG',
        width_mm=183, height_mm=180, dpi=300, archetype='quantitative_grid',
        all_candidates_and_timings_and_trials_retained=True, visual_qa='pending',
        uncertainty='Pointwise weighted 5th–95th latent-state quantiles at fixed parameters; finite candidates shown separately',
        versions=dict(matplotlib=matplotlib.__version__, numpy=np.__version__, pandas=pd.__version__),
        figures={name: digest(output/name) for name in files},
        source_tables={name+'.csv': digest(output/(name+'.csv')) for name in tables}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.output)
