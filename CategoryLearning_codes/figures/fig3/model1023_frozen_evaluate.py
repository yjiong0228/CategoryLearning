"""Compare frozen parameter choices through behavior, states and noisy reports.

Contract: behavior-score improvements and stable psychological interpretations
are assessed separately. Quantitative grids retain all three audited parameter
points per participant, every autonomous path and both fixed oral readings.
Conditional target-rule states come directly from original filters. Oral curves
reuse the established 32-trial aggregation and its complete-report denominator.
Development and extension participants remain separate. This is style-level
adaptation of the Model 1022 figures; parameter points replace mechanism arms.

Use --source COMPLETE_EXTERNAL_SUMMARY --states COMPLETE_STATE_SENSITIVITY
--output NEW. Python/matplotlib exports 183-mm, 300-dpi PNG review figures with
source tables, hashes and a reading guide. No fitting, new model simulations,
oral recoding or participant-level significance tests occur here.
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
from .model1023_estimation_reliability import STYLE
from . import model1022_oral_trajectories as oral_time

COLORS = ('#3B8C85', '#527697', '#9B775D')
COHORTS = dict(development='Development', extension='Extension · prior influence unknown')


def ordered_points(meta: pd.DataFrame, subject: int) -> list[dict]:
    points = meta[meta.subject == subject].sort_values(['selected', 'reference', 'candidate'], ascending=[False, False, True])
    if len(points) != 3 or points.selected.sum() != 1 or points.reference.sum() != 1:
        raise ValueError('Require three fixed points and unique selected/reference identities')
    result, alternative = [], 0
    for row in points.to_dict('records'):
        if row['selected']:
            label = 'Selected / reference' if row['reference'] else 'Selected'
        elif row['reference']:
            label = 'Reference'
        else:
            alternative += 1
            label = 'Alternative '+str(alternative)
        result.append(dict(row, label=label, color=COLORS[len(result)]))
    return result


def state_blocks(trials: pd.DataFrame, width: int = 32) -> pd.DataFrame:
    """Preserve time boundaries and presence-weight the block precision."""
    if width < 1 or int(width) != width:
        raise ValueError('Block width must be a positive integer')
    rows = []
    for (subject, candidate), group in trials.groupby(['subject', 'candidate']):
        group = group.sort_values('trial')
        np.testing.assert_array_equal(group.trial, np.arange(1, len(group)+1))
        if group.training_trials.nunique() != 1 or int(group.training_trials.iloc[0]) % width:
            raise ValueError('State time blocks cross the parameter-training boundary')
        for start in range(0, len(group), width):
            part = group.iloc[start:start+width]
            weights = part.target_present.to_numpy()
            precision = part.target_precision_given_presence.to_numpy()
            if np.any(~np.isfinite(precision[weights > 0])):
                raise ValueError('Precision undefined while the target has positive mass')
            positive = weights > 0
            value = np.dot(weights[positive], precision[positive])/weights.sum() if weights.sum() else np.nan
            rows.append(dict(subject=subject, candidate=candidate, trial=int(part.trial.iloc[-1]),
                training_trials=int(part.training_trials.iloc[0]), target_present=weights.mean(),
                target_contribution=part.target_contribution.mean(), target_precision_given_presence=value))
    return pd.DataFrame(rows)


def load_data(source: Path, states: Path) -> dict:
    inputs, tables = {}, {}
    for folder, prefix in ((source, ''), (states, 'states/')):
        done = read(folder/'completion.json')
        if done['status'] != 'complete':
            raise ValueError('Require complete external and state summaries')
        inputs[str((folder/'completion.json').resolve())] = digest(folder/'completion.json')
        for name, sha in done['inputs_sha256'].items():
            if digest(Path(name)) != sha:
                raise ValueError('Figure source changed: '+name)
            inputs[name] = sha
        for name, sha in done['artifacts_sha256'].items():
            path = folder/name
            if digest(path) != sha:
                raise ValueError('Figure table or state array changed: '+str(path))
            inputs[str(path.resolve())] = sha
            if path.suffix == '.csv':
                tables[prefix+name.removesuffix('.csv')] = pd.read_csv(path)
    a, b = read(source/'completion.json'), read(states/'completion.json')
    if Path(a['evaluation']).resolve() != Path(b['evaluation']).resolve() or a['participants'] != 6:
        raise ValueError('State and behavior evaluation scopes differ')
    meta = tables['scores'].query('phase == "whole"').copy()
    if len(meta) != 18 or meta.duplicated(['subject', 'candidate']).any():
        raise ValueError('Require all eighteen participant/parameter combinations')
    points = {int(s): ordered_points(meta, s) for s in meta.subject.unique()}
    tables['states/blocks'] = state_blocks(tables['states/trials'])
    summary_checks = 0
    for reading in oral_time.READINGS:
        raw, scores = tables['oral_'+reading+'/trials'], tables['oral_'+reading+'/summary']
        blocks = oral_time.time_blocks(raw)
        summary_checks += oral_time.check_summary(raw, blocks, scores)
        tables['oral_'+reading+'/blocks'] = blocks
    for row in meta.itertuples():
        own = tables['curves'].query('subject == @row.subject and candidate == @row.candidate').sort_values('trial')
        paths = tables['path_blocks'].query('subject == @row.subject and candidate == @row.candidate')
        matrix = paths.pivot(index='seed', columns='trial', values='accuracy')
        if matrix.shape != (128, len(own)):
            raise ValueError('Missing autonomous paths or blocks')
        np.testing.assert_array_equal(matrix.columns, own.trial)
        np.testing.assert_allclose(matrix.mean(0), own.autonomous, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.quantile(matrix, [.1, .9], axis=0), own[['lower', 'upper']].to_numpy().T, atol=1e-12, rtol=0)
        identity = None
        for reading in oral_time.READINGS:
            raw = tables['oral_'+reading+'/trials'].query('subject == @row.subject and candidate == @row.candidate').sort_values('ordinal_trial')
            if len(raw) != row.trials or not raw.training_trials.eq(row.training_trials).all():
                raise ValueError('Oral and behavior trial boundaries differ')
            columns = ['ordinal_trial', 'iSession', 'iTrial', 'choice', 'scope']
            if identity is not None:
                pd.testing.assert_frame_equal(raw[columns].reset_index(drop=True), identity)
            identity = raw[columns].reset_index(drop=True)
    inputs[str(Path(oral_time.__file__).resolve())] = digest(Path(oral_time.__file__))
    style_source = Path(__file__).with_name('model1023_estimation_reliability.py')
    inputs[str(style_source.resolve())] = digest(style_source)
    return dict(tables=tables, points=points, meta=meta, inputs=inputs, oral_summary_checks=summary_checks)


def save(fig, output: Path, name: str) -> str:
    fig.savefig(output/name, dpi=300, facecolor='white')
    plt.close(fig)
    return name


def behavior(data: dict, cohort: str, output: Path) -> str:
    subjects = sorted(data['meta'].query('cohort == @cohort').subject.unique())
    fig, axes = plt.subplots(3, 3, figsize=(183/25.4, 183/25.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.12, hspace=.42, wspace=.23)
    for row, subject in enumerate(subjects):
        for col, point in enumerate(data['points'][int(subject)]):
            candidate = point['candidate']
            curve = data['tables']['curves'].query('subject == @subject and candidate == @candidate').sort_values('trial')
            samples = data['tables']['path_blocks'].query('subject == @subject and candidate == @candidate').pivot(index='seed', columns='trial', values='accuracy')
            ax = axes[row, col]
            ax.axvspan(point['training_trials'], point['trials'], color='#F3F4F6')
            ax.plot(curve.trial, samples.to_numpy().T, color=point['color'], alpha=.06, lw=.35)
            ax.plot(curve.trial, curve.autonomous, color=point['color'], lw=1.5)
            ax.plot(curve.trial, curve.conditional, color=point['color'], lw=1, ls='--')
            ax.plot(curve.trial, curve.observed, color='#222222', marker='o', ms=1.8, lw=.9)
            ax.axvline(point['training_trials'], color='#777777', ls=':', lw=.7)
            ax.set(xlim=(0, point['trials']), ylim=(0, 1.02), yticks=[0, .5, 1])
            ax.set_title(f'S{subject} · '+point['label'], fontsize=7.5)
            if col == 0:
                ax.set_ylabel('Accuracy')
            if row == 2:
                ax.set_xlabel('Trial (32-trial blocks)')
    fig.suptitle('Model 1022 · '+COHORTS[cohort], y=.985, fontsize=10)
    handles = [Line2D([], [], color='#222222', marker='o', ms=2, lw=1, label='Participant'),
               Line2D([], [], color='#527697', lw=1.5, label='Autonomous mean'),
               Line2D([], [], color='#527697', lw=1, ls='--', label='Conditional prediction'),
               Line2D([], [], color='#527697', lw=.5, alpha=.4, label='All 128 autonomous paths')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=2, fontsize=7)
    fig.text(.5, .045, 'Shading: later choices excluded from parameter fitting. All three fixed candidates are shown.', ha='center', fontsize=6.5)
    return save(fig, output, 'behavior_'+cohort+'.png')


def state_trajectories(data: dict, subject: int, output: Path) -> str:
    points = data['points'][subject]
    fig, axes = plt.subplots(3, 2, figsize=(183/25.4, 165/25.4), sharex=True, sharey='row')
    fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.81, hspace=.25, wspace=.22)
    metrics = [('target_present', 'Target membership'), ('target_contribution', 'Choice contribution'),
               ('target_precision_given_presence', 'Precision given presence')]
    for col, key in enumerate(('autonomous_states', 'states/blocks')):
        for point in points:
            candidate = point['candidate']
            own = data['tables'][key].query('subject == @subject and candidate == @candidate').sort_values('trial')
            if len(own) == 0:
                raise ValueError('Missing candidate state trajectory')
            for row, (metric, label) in enumerate(metrics):
                ax = axes[row, col]
                ax.plot(own.trial, own[metric], color=point['color'], lw=1.1, label=point['label'])
                if col == 0:
                    ax.set_ylabel(label)
                if row < 2:
                    ax.set_ylim(-.02, 1.02)
                if row == 2:
                    ax.set_xlabel('Trial (32-trial blocks)')
        for ax in axes[:, col]:
            ax.axvspan(points[0]['training_trials'], points[0]['trials'], color='#F3F4F6', zorder=-5)
            ax.axvline(points[0]['training_trials'], color='#777777', ls=':', lw=.7)
            ax.set_xlim(0, points[0]['trials'])
        axes[0, col].set_title('Autonomous generation' if col == 0 else 'After observed choice, before feedback', fontsize=8)
    fig.suptitle(f'S{subject} · parameter sensitivity of inferred learning', y=.985, fontsize=10)
    fig.legend(handles=[Line2D([], [], color=p['color'], lw=1.2, label=p['label']) for p in points],
               loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, fontsize=7)
    fig.text(.5, .09, 'Left: 128 independent paths. Right: original eight-filter joint-state summaries.', ha='center', fontsize=6.5)
    fig.text(.5, .055, 'Lines compare fixed parameter points; precision is weighted by target-rule presence within each block.', ha='center', fontsize=6.5)
    fig.text(.5, .02, 'Behavior scores and state differences are listed separately; these candidates are not a confidence region.', ha='center', fontsize=6.5)
    return save(fig, output, f'S{subject}_states.png')


def oral_trajectories(data: dict, cohort: str, reading: str, domain: str, output: Path) -> str:
    subjects = sorted(data['meta'].query('cohort == @cohort').subject.unique())
    raw, blocks = data['tables']['oral_'+reading+'/trials'], data['tables']['oral_'+reading+'/blocks']
    fig, axes = plt.subplots(3, 3, figsize=(183/25.4, 183/25.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.81, bottom=.14, hspace=.45, wspace=.23)
    for row, subject in enumerate(subjects):
        for col, point in enumerate(data['points'][int(subject)]):
            candidate = point['candidate']
            own = raw.query('subject == @subject and candidate == @candidate').sort_values('ordinal_trial')
            curve = blocks.query('subject == @subject and arm == @candidate').sort_values('trial')
            complete = own.scope.eq('complete')
            ax = axes[row, col]
            ax.axvspan(point['training_trials'], point['trials'], color='#F3F4F6')
            ax.scatter(own.loc[complete, 'ordinal_trial'], own.loc[complete, domain+'_support'],
                       color=oral_time.COLORS['support'], s=2.5, alpha=.22, linewidths=0)
            for label in oral_time.LABELS:
                ax.plot(curve.trial, curve[domain+'_'+label], color=oral_time.COLORS[label],
                        ls=oral_time.STYLES[label], lw=1.1)
            ax.plot(curve.trial, curve[domain+'_coverage'], color='#527697', ls='-.', lw=.8)
            ax.plot(own.loc[~complete, 'ordinal_trial'], np.full((~complete).sum(), -.045),
                    '|', color='#777777', ms=2.5, markeredgewidth=.5)
            ax.axvline(point['training_trials'], color='#777777', ls=':', lw=.7)
            ax.set(xlim=(0, point['trials']), ylim=(-.07, 1.04), yticks=[0, .5, 1])
            ax.set_title(f'S{subject} · '+point['label']+f'\n{int(complete.sum())} complete reports', fontsize=7.2)
            if col == 0:
                ax.set_ylabel('Content compatibility mass')
            if row == 2:
                ax.set_xlabel('Trial')
    fig.suptitle(COHORTS[cohort]+f'\n{domain.capitalize()} oral support · {reading} reading', y=.99, fontsize=9)
    handles = [Line2D([], [], color=oral_time.COLORS[k], ls=oral_time.STYLES[k], lw=1.1, label=k.capitalize()) for k in oral_time.LABELS]
    handles.append(Line2D([], [], color='#527697', ls='-.', lw=.8, label='Catalogue coverage'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .915), ncol=4, fontsize=6.5)
    fig.text(.5, .075, 'Dots: every complete report. Lines: equal-report means in 32-trial blocks; empty blocks stay undefined.', ha='center', fontsize=6.5)
    fig.text(.5, .04, 'Bottom marks: incomplete or missing reports. Later states use observed choices; oral reports remain external.', ha='center', fontsize=6.5)
    return save(fig, output, f'oral_{cohort}_{domain}_{reading}.png')


def render(source: Path, states: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new paired-parameter figure directory')
    data = load_data(source, states)
    output.mkdir(parents=True)
    files = []
    with plt.rc_context(STYLE):
        for cohort in COHORTS:
            files.append(behavior(data, cohort, output))
            for reading in oral_time.READINGS:
                for domain in oral_time.DOMAINS:
                    files.append(oral_trajectories(data, cohort, reading, domain, output))
        for subject in sorted(data['points']):
            files.append(state_trajectories(data, subject, output))
    sources = output/'source_data'
    sources.mkdir()
    for name, table in data['tables'].items():
        table.to_csv(sources/(name.replace('/', '_')+'.csv'), index=False)
    page = '<!doctype html><meta charset="utf-8"><title>Model 1022参数选择与学习解释</title>'
    page += '<style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:32px auto}img{width:100%}table{font-size:12px}</style>'
    page += '<h1>参数选择与学习过程解释</h1><p>同一模型下比较三组已冻结参数。选中点来自本轮联合搜索，参考点为原训练段估计。'
    page += '两者重合时共用一列，其余列保留不同候选。所有候选均由前段行为确定，口述仅用于外部检验。</p>'
    page += '<p>行为图的细线包含全部128条自主轨迹。内部状态图比较自主生成和行为条件下的推断，'
    page += '精度只在目标规则存在时定义。三组参数的独立行为分差与内部状态差异并列保留，用于判断解释的敏感性。</p>'
    page += '<p>口述图保留中点与相对两种既定读法，绿色、橙色和灰色分别表示支持、冲突和未确定质量，'
    page += '蓝线表示目录覆盖。点为完整报告的逐题支持质量，趋势线为32题内的等报告权重均值。'
    page += '口述带有噪声，图中兼容性只提供对模型解释的支持程度。</p>'
    page += ''.join('<h2>'+html.escape(name.removesuffix('.png'))+'</h2><img src="'+name+'">' for name in files)
    columns = ['subject', 'candidate', 'selected', 'reference', 'window', 'audit_delta_nll',
               'audit_delta_mc_se', 'basis_total_variation', 'target_presence_absolute_change', 'target_precision_absolute_change']
    page += '<h2>行为评分与内部解释差异</h2>'+data['tables']['states/contrasts'][columns].to_html(index=False)
    (output/'index.html').write_text(page, encoding='utf-8')
    shutil.copyfile(Path(__file__), output/'source_snapshot.py')
    write(output/'completion.json', dict(status='complete', source=str(source.resolve()), states=str(states.resolve()),
        inputs_sha256=data['inputs'], source_sha256=digest(Path(__file__)), backend='python', format='PNG', dpi=300,
        width_mm=183, all_autonomous_paths_retained=True, all_audited_candidates_retained=True,
        versions=dict(matplotlib=matplotlib.__version__, numpy=np.__version__, pandas=pd.__version__),
        oral_summary_checks=data['oral_summary_checks'], state_aggregation='Precision weighted by target presence',
        figures={name: digest(output/name) for name in files}, visual_qa='pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'states', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.states, args.output)
