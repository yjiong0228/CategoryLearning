"""Show behavior, autonomous learning and oral evidence after behavioral refitting.

Use --comparison precision|online --output NEW_DIRECTORY. Both proposals and
both fixed oral readings are required. Production input must have a completed
independent audit. PNG-only discussion figures; no model or score is changed.
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
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASES = ('training', 'test', 'whole')
ARMS = ('POOL', 'SEARCH')
LABELS = ('support', 'conflict', 'undetermined')
PROFILES = {
    'precision': dict(labels=('RESET', 'RETAIN'), kind='precision_retention_refit',
        behavior='reentry_precision_refit_s208_comparison_20261018_v2',
        behavior_audit='reentry_precision_refit_comparison_validation_20261018_v1',
        oral_audit='reentry_precision_refit_oral_full_audit_20261019_v1',
        oral='reentry_precision_refit_oral_{reading}_{arm}_comparison_20261018_v1',
        summary_keys=('reset_summary', 'retained_summary')),
    'online': dict(labels=('RETAIN', 'ONLINE'), kind='online_evidence_refit',
        behavior='online_evidence_refit_behavior_comparison_20261019_v1',
        behavior_audit='online_evidence_refit_behavior_audit_20261019_v1',
        oral_audit='online_evidence_refit_oral_full_audit_20261019_v1',
        oral='online_evidence_refit_oral_{reading}_{arm}_comparison_20261019_v1',
        summary_keys=('retained_summary', 'online_summary')),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_cells(table: pd.DataFrame, columns: list[str], expected: set) -> None:
    """Reject omitted or repeated scientific cells before reshaping for display."""
    cells = list(table[columns].itertuples(index=False, name=None))
    if len(cells) != len(set(cells)) or set(cells) != expected:
        raise ValueError('Missing, duplicated or unexpected figure cells')


def audit_hashes(record: dict) -> dict:
    """Normalize the two existing audit schemas without weakening bindings."""
    mappings = [record[k] for k in ('inputs_sha256', 'input_sha256') if k in record]
    if not mappings:
        raise ValueError('Audit has no bound inputs')
    result = {}
    for mapping in mappings:
        for path, sha in mapping.items():
            name = str((ROOT/Path(path)).resolve())
            if name in result and result[name] != sha:
                raise ValueError('Conflicting audit hashes for one input')
            result[name] = sha
    return result


def load_inputs(comparison: str) -> tuple[dict, dict, dict, int]:
    profile = PROFILES[comparison]
    result = ROOT/'results/model_0923'
    inputs, tables, identities = {}, {}, {}
    checks = 0

    def bind(path, expected=None):
        path = Path(path).resolve(); sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Figure source differs from its audited version: '+str(path))
        inputs[str(path)] = sha
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    def audited_manifest(folder, audit):
        path = (folder/'manifest.json').resolve()
        manifest = read(path, audit_hashes(audit)[str(path)])
        if manifest['status'] != 'complete':
            raise ValueError('Incomplete comparison')
        return manifest

    audits = {key: read(result/profile[key]/'validation.json') for key in ('behavior_audit', 'oral_audit')}
    if any(a['status'] != 'complete' for a in audits.values()):
        raise ValueError('Complete independent numerical audits are required')
    folder = result/profile['behavior']
    manifest = audited_manifest(folder, audits['behavior_audit'])
    if manifest['subjects'] != [208] or set(manifest['arms']) != set(ARMS):
        raise ValueError('Require both proposals for the declared development participant')
    for filename, sha in manifest['tables_sha256'].items():
        tables['behavior_'+filename[:-4]] = pd.read_csv(bind(folder/filename, sha))
    nominees = {}
    for variant, key in zip(profile['labels'], profile['summary_keys']):
        summary_path = Path(manifest[key])/'manifest.json'
        summary = read(summary_path, audit_hashes(manifest)[str(summary_path.resolve())])
        point_path = Path(summary['fit'])/'nominees.json'
        frozen = read(point_path, audit_hashes(manifest)[str(point_path.resolve())])
        if frozen['status'] != 'frozen_before_test' or frozen['smoke']:
            raise ValueError('Figure requires full frozen behavioral nominations')
        for arm in ARMS:
            rows = [row for row in frozen['nominees'] if (row['subject'], row['model']) == (208, arm)]
            if len(rows) != 1:
                raise ValueError('Missing or duplicated behavioral nominee')
            nominees[variant, arm] = rows[0]['candidate']
    behavior = tables['behavior_contrasts']; scores = tables['behavior_scores']
    require_cells(behavior, ['subject', 'model', 'phase'], {(208, a, p) for a in ARMS for p in PHASES})
    require_cells(scores, ['subject', 'model', 'phase', 'variant'],
                  {(208, a, p, v) for a in ARMS for p in PHASES for v in profile['labels']})
    for row in behavior.itertuples():
        selected = scores[scores.model.eq(row.model)&scores.phase.eq(row.phase)].set_index('variant')
        for metric in ('choice_nll', 'energy', 'crps'):
            a, b = (selected.loc[v, metric] for v in profile['labels'])
            np.testing.assert_allclose(b-a, getattr(row, 'delta_'+metric), rtol=0, atol=1e-12)
            if getattr(row, 'delta_'+metric+'_mc_se') < 0:
                raise ValueError('Negative simulation uncertainty')
            checks += 1
    oral = []
    for reading in ('midpoint', 'relative'):
        for arm in ARMS:
            directory = result/profile['oral'].format(reading=reading, arm=arm.lower())
            record = audited_manifest(directory, audits['oral_audit'])
            if (record['subject'] != 208 or record['arm'] != arm or record['contrast_kind'] != profile['kind']
                    or record['seed_relation'] != 'paired' or not record['original_scores_reproduced']
                    or not record['revised_scores_reproduced']):
                raise ValueError('Oral comparison does not match the declared refit')
            for key, variant in zip(('original_parameters', 'revised_parameters'), profile['labels']):
                identity = record[key]; candidate = identity['candidate']
                if any(candidate[k] != nominees[variant, arm][k] for k in ('id', 'hyperparams')):
                    raise ValueError('Behavior and oral panels use different parameter points')
                previous = identities.setdefault((variant, arm), identity)
                for field in ('candidate', 'scientific', 'mechanism', 'task_sha256'):
                    if previous[field] != identity[field]:
                        raise ValueError('Oral readings use different model or task states')
            a, b = (record[key] for key in ('original_parameters', 'revised_parameters'))
            if a['scientific'] != b['scientific'] or a['task_sha256'] != b['task_sha256']:
                raise ValueError('Before/after task identity differs')
            if a['oral_manifest_sha256'] != b['oral_manifest_sha256']:
                raise ValueError('Before/after oral encoding differs')
            scientific = a['scientific']; cut = scientific['training_trials']; length = scientific['trials']
            scored = pd.read_csv(bind(directory/'scores.csv', record['artifacts']['scores.csv']))
            trials = pd.read_csv(bind(directory/'trials.csv', record['artifacts']['trials.csv']))
            if len(trials) != length or not np.array_equal(trials.state_trial.to_numpy(), np.arange(length)):
                raise ValueError('Missing or reordered oral trials')
            selected = scored[scored.subset.eq('complete')&scored.window.isin(PHASES)]
            require_cells(selected, ['subject', 'arm', 'window', 'domain'],
                          {(208, arm, p, d) for p in PHASES for d in ('structural', 'functional')})
            for row in selected.itertuples():
                start, stop = {'training': (0, cut), 'test': (cut, length), 'whole': (0, length)}[row.window]
                mask = trials.scope.eq('complete')&trials.state_trial.between(start, stop-1)
                if row.reports != mask.sum() or row.window_trials != stop-start:
                    raise ValueError('Figure changed oral report coverage')
                behavior_row = behavior[behavior.model.eq(arm)&behavior.phase.eq(row.window)].iloc[0]
                if behavior_row.trials != stop-start or row.filter_seeds != 8 or scientific['particles'] != 512:
                    raise ValueError('Behavior/oral boundaries or numerical budgets differ')
                for label in LABELS:
                    for prefix in ('original', 'revised'):
                        value = trials.loc[mask, prefix+'_'+row.domain+'_'+label].mean()
                        np.testing.assert_allclose(value, getattr(row, prefix+'_'+label), rtol=0, atol=1e-12)
                        checks += 1
                    np.testing.assert_allclose(getattr(row, 'revised_'+label)-getattr(row, 'original_'+label),
                        getattr(row, label+'_difference'), rtol=0, atol=1e-12)
                    checks += 1
                for prefix in ('original', 'revised'):
                    np.testing.assert_allclose(sum(getattr(row, prefix+'_'+label) for label in LABELS), 1., rtol=0, atol=1e-12)
                    checks += 1
            oral.append(scored.assign(reading=reading))
            tables[f'oral_{reading}_{arm.lower()}_trials'] = trials
    tables['oral_scores'] = pd.concat(oral, ignore_index=True)
    numeric = behavior.filter(regex='^delta_').to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Non-finite behavior contrasts')
    selected = tables['oral_scores'].query('subset == "complete" and window in @PHASES')
    numeric = selected.filter(regex='(_difference|_difference_mcse)$').to_numpy(float)
    if not np.isfinite(numeric).all() or (selected.filter(regex='_mcse$') < 0).any().any():
        raise ValueError('Invalid oral contrast or simulation uncertainty')
    if not (scores.particles.eq(512).all() and scores.filter_seeds.eq(8).all() and scores.paths.eq(128).all()):
        raise ValueError('Figure caption differs from actual filter/path budgets')
    return tables, inputs, profile, checks


def draw(tables: dict, labels: tuple[str, str], output: Path) -> None:
    """Render all declared phases, proposals and oral mass components."""
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = {'POOL': '#407B94', 'SEARCH': '#A77949'}
    fig, axes = plt.subplots(2, 2, figsize=(7.204724, 6.653543), gridspec_kw={'height_ratios': [1, 1.8]})
    fig.subplots_adjust(left=.16, right=.97, bottom=.17, top=.87, hspace=.6, wspace=.65)
    phase_names = {'training': 'Training', 'test': 'Later', 'whole': 'Whole'}
    for ax, metric, title in zip(axes[0], ('choice_nll', 'energy'), ('Choice prediction', 'Autonomous learning')):
        for i, phase in enumerate(PHASES):
            for arm, shift in [('POOL', -.12), ('SEARCH', .12)]:
                row = tables['behavior_contrasts'].query('model == @arm and phase == @phase').iloc[0]
                ax.errorbar(row['delta_'+metric], i+shift, xerr=row['delta_'+metric+'_mc_se'],
                    fmt='o' if arm == 'POOL' else 's', color=colors[arm], markersize=3.5, capsize=2, lw=.8)
        ax.set_yticks(range(3), [phase_names[p] for p in PHASES]); ax.set_ylim(2.5, -.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(('NLL' if metric == 'choice_nll' else 'Energy score')+' difference\nLower is better')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    oral = tables['oral_scores']; display = oral[oral.domain.eq('structural')&oral.subset.eq('complete')&oral.window.isin(PHASES)]
    for ax, reading, title in zip(axes[1], ('midpoint', 'relative'), ('Absolute oral reading', 'Relative oral reading')):
        for i, phase in enumerate(PHASES):
            for j, label in enumerate(LABELS):
                y = i*3.6+j
                for arm, shift in [('POOL', -.12), ('SEARCH', .12)]:
                    row = display.query('reading == @reading and arm == @arm and window == @phase').iloc[0]
                    ax.errorbar(row[label+'_difference'], y+shift, xerr=row[label+'_difference_mcse'],
                        fmt='o' if arm == 'POOL' else 's', color=colors[arm], markersize=3.1, capsize=1.8, lw=.7)
        ticks = [i*3.6+j for i in range(3) for j in range(3)]
        text = [f'{phase_names[p]}: support' if j == 0 else ('conflict' if j == 1 else 'undetermined')
                for p in PHASES for j in range(3)]
        ax.set_yticks(ticks, text, fontsize=6); ax.set_ylim(10, -.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Whole-report mass difference')
    span = max(abs(v) for ax in axes[1] for v in ax.get_xlim())
    for ax in axes[1]: ax.set_xlim(-span, span)
    for letter, ax in zip('abcd', axes.flat):
        ax.axvline(0, color='#7F858C', lw=.6, ls='--'); ax.tick_params(axis='y', length=0)
        ax.text(-.34, 1.12, letter, transform=ax.transAxes, weight='bold', fontsize=9)
    fig.suptitle(f'Subject 208: {labels[1]} minus {labels[0]} after behavioral refitting', y=.98, fontsize=9)
    fig.legend([Line2D([0], [0], marker=m, color=colors[a], ls='none', markersize=4) for a,m in [('POOL','o'),('SEARCH','s')]],
        ARMS, loc='upper center', bbox_to_anchor=(.59,.95), ncol=2, fontsize=7)
    counts = []
    for phase in PHASES:
        values = display[display.window.eq(phase)].reports.unique()
        if len(values) != 1: raise ValueError('Both readings and proposals must have the same displayed report counts')
        counts.append(str(int(values[0])))
    fig.text(.16,.08,'Complete oral reports (training / later / whole): '+' / '.join(counts)+'. Both fixed readings retained.', fontsize=6)
    fig.text(.16,.055,'Bars: paired simulation SE; 8 filters (512 particles), 128 autonomous paths per proposal.', fontsize=6)
    fig.text(.16,.03,'One development participant; later data are a temporal diagnostic. Oral is excluded from fitting.', fontsize=6)
    fig.savefig(output/'joint_evidence.png', dpi=300, facecolor='white'); plt.close(fig)


def run(comparison: str, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new joint-evidence figure directory')
    tables, inputs, profile, checks = load_inputs(comparison)
    output.mkdir(parents=True)
    contract = dict(backend='python', archetype='quantitative grid', dimensions_mm=[183,169], export='PNG only, 300 dpi discussion draft',
        question='Does the behavior-selected mechanism improve choice prediction, autonomous learning and oral representation together?',
        evidence_chain='Top: all phase-specific choice NLL and autonomous Energy differences. Bottom: support, conflict and undetermined mass under both fixed oral readings.',
        contrast=f"{profile['labels'][1]} minus {profile['labels'][0]}; mechanism and re-estimation jointly change",
        sample='One development participant, both proposals. All complete oral reports within each original phase; full trials and all other metrics remain in source tables.',
        uncertainty='Audited paired complete-filter/path simulation standard errors at frozen parameters; not participant confidence intervals.',
        statistics='Descriptive differences; no significance tests or multiplicity claims.',
        interpretation='Lower NLL/Energy is better; more support and less conflict favor content agreement. Undetermined mass indicates unexpressed or uncommitted content.',
        adaptation='Build anew; local palette and typography conventions only.',
        limitations='Whole-report structural readout; prior-readout and predicate diagnostics remain in their original reports. No parameter uncertainty or independent participant validation.')
    (output/'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    for name, table in tables.items(): table.to_csv(output/(name+'.csv'), index=False)
    draw(tables, profile['labels'], output)
    links=' · '.join(f'<a href="{name}.csv">{name}</a>' for name in tables)
    note=('差值同时包含机制变化与行为重新估参。参数只由原训练选择确定；口述仅用于检验冻结参数后的表征。'
        '前段、后段和全程相互重叠，POOL与SEARCH属于同一被试的两种提议方案。误差线表示配对模拟标准误。'
        'NLL与Energy越低越好；口述支持增加、冲突减少分别表示内容对应的改善，未确定质量单独保留。'
        '裸长短的两种读法统一用于全部方案，所有完整报告均计入；源表另保留功能域、目录条件子集及首末窗口。')
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>重估后的联合证据</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>'
        '<h1>行为重估后的预测、生成与口述证据</h1><p>'+note+'</p><img src="joint_evidence.png"><p>'+links+'</p></html>')
    shutil.copy2(__file__, output/'figure_source.py')
    for path, sha in inputs.items():
        if digest(Path(path)) != sha: raise ValueError('Figure source changed while rendering')
    (output/'validation.json').write_text(json.dumps(dict(status='complete',comparison=comparison,source_checks=checks,
        input_sha256=inputs,source_sha256=digest(Path(__file__)),artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()},
        qa_exceptions=['PNG-only repository convention','300 dpi discussion draft']), indent=2)+'\n')
    print(dict(status='complete', comparison=comparison, source_checks=checks))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', choices=PROFILES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args();run(args.comparison,args.output)
