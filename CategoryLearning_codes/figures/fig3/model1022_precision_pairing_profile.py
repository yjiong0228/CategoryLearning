"""Audit and plot the completed retrospective precision/pairing profile.

Use --study COMPLETE_PROFILE --output NEW. The two-column quantitative grid
shows every coarse candidate and every high-precision finalist for both
registered sequences. This optimizer diagnostic holds seven other coordinates
fixed at their previous estimates. The generating point can differ on those
coordinates. Paired Monte Carlo errors describe numerical precision only.
Python/matplotlib exports a parameter grid and two state figures as 183-mm,
300-dpi PNG review figures, source tables and an HTML explanation. No
additional fitting or state simulation occurs.
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

from src.Bayesian_state.workflows.analysis import model_1022_precision_pairing_profile as profile
from src.Bayesian_state.evaluation.model_0923_h_validation import nll, jackknife_scores, jackknife_se
from . import model1022_compact_recovery as recovery_figure

read, write, digest = profile.read, profile.write, profile.digest


def render(study: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new profile-review directory')
    inputs = {}

    def bind(file, expected=None):
        file = Path(file).resolve()
        value = digest(file)
        if expected is not None and value != expected:
            raise ValueError('Profile review input changed: '+str(file))
        inputs[str(file)] = value
        return file

    done = read(bind(study/'completion.json'))
    if done['status'] != 'complete' or done['smoke'] or done['cases'] != 2:
        raise ValueError('Complete both scientific profile cases first')
    protocol = read(bind(study/'protocol.json'))
    for name, sha in protocol['source_sha256'].items():
        bind(study/'source'/name, sha)
    plan = read(bind(study/'cases.json'))
    for name, sha in plan['inputs_sha256'].items():
        bind(name, sha)
    fit_done = read(bind(study/'fit/completion.json'))
    bind(study/'fit/nominees.json', fit_done['nomination_sha256'])
    truth_done = read(bind(study/'truth_scores/completion.json'))
    if not truth_done['truths_opened_after_freeze']:
        raise ValueError('New generating-point scoring preceded estimate freezing')
    grid, confirmed, filters, changes = [], [], [], []
    panels = []
    for case in plan['cases']:
        case_id = case['case_id']; fit = Path(case['output'])
        frozen = read(bind(fit/'nominees.json'))
        finish = read(bind(fit/'completion.json'))
        bind(fit/'nominees.json', finish['nomination_sha256'])
        nominee = frozen['nominee']; selected = nominee['candidate']
        bank, required = profile.profile_bank(case['space'], case['required'])
        if bank != case['starts'] or len(bank) != 56:
            raise ValueError('The registered full grid or incumbent changed')
        if selected['mean_nll'] != min(r['mean_nll'] for r in nominee['finalists']):
            raise ValueError('Selected point is not the best confirmed candidate')
        counts = dict(screen=112, selection=16, confirmation=24)
        stage_seeds = {k:set() for k in counts}
        for phase, count in counts.items():
            jobs = list((fit/'jobs').glob(phase+'_*.json'))
            if len(jobs) != count:
                raise ValueError('Actual profile budget differs from registration')
            for file in jobs:
                job = read(bind(file)); score = read(bind(Path(job['target']).with_suffix('.json')))
                variant = read(bind(Path(job['target']).with_suffix('.variant.json')))
                if (job['compact_variant'] != 'TRANSFER' or job['entry_control'] != 'PRECISION'
                        or variant['compact_variant'] != 'TRANSFER' or len(job['seeds']) != 1
                        or job['particles'] != case['settings']['budgets'][phase]['particles']
                        or any(p['num_threads'] != 1 for p in score['numeric_pools'])):
                    raise ValueError('Incorrect profile mechanism or numerical execution')
                bind(Path(job['target']).with_suffix('.npz'), score['predictions_sha256'])
                stage_seeds[phase].update(job['seeds'])
                filters.append(dict(case_id=case_id, stage=phase, candidate=job['candidate']['id'],
                                    seed=job['seeds'][0], particles=job['particles']))
        if any(len(stage_seeds[s]) != case['settings']['budgets'][s]['seeds'] for s in counts):
            raise ValueError('Incorrect independent seed count')
        if sum(map(len, stage_seeds.values())) != len(set.union(*stage_seeds.values())):
            raise ValueError('Profile stage seeds overlap')
        for candidate in bank:
            file = fit/'screen/subject_306/COMPACT'/(candidate['id']+'.json')
            score = read(bind(file)); bind(file.with_suffix('.npz'), score['predictions_sha256'])
            with np.load(file.with_suffix('.npz')) as z:
                value = nll(z['observed_probability'])
                if set(z['seeds']) != stage_seeds['screen']:
                    raise ValueError('Grid candidates have different screening seeds')
            np.testing.assert_allclose(value, score['mean_nll'], atol=1e-12, rtol=0)
            grid.append(dict(case_id=case_id, candidate=candidate['id'],
                beta_init=candidate['hyperparams'][profile.BETA], gamma=candidate['hyperparams'][profile.GAMMA],
                nll=value, previous=candidate['id']==required['id'], selected=candidate['id']==selected['id']))
        truth_folder = study/'truth_scores'/case_id
        truth_files = list((truth_folder/'confirmation/subject_306/COMPACT').glob('*.json'))
        truth_files = [p for p in truth_files if not p.name.endswith('.variant.json')]
        if len(truth_files) != 1:
            raise ValueError('Expected one combined generating-point score')
        truth_file = truth_files[0]; truth = read(bind(truth_file))
        bind(truth_file.with_suffix('.npz'), truth['predictions_sha256'])
        with np.load(truth_file.with_suffix('.npz')) as z:
            truth_seeds = z['seeds'].copy(); probabilities = z['observed_probability'].copy()
        truth_nll, truth_jk = nll(probabilities), jackknife_scores(probabilities)
        for file in (truth_folder/'jobs').glob('*.json'):
            job = read(bind(file)); score = read(bind(Path(job['target']).with_suffix('.json')))
            if (job['compact_variant'] != 'TRANSFER' or job['entry_control'] != 'PRECISION'
                    or job['particles'] != 1024 or len(job['seeds']) != 1
                    or any(p['num_threads'] != 1 for p in score['numeric_pools'])):
                raise ValueError('Incorrect generating-point scorer')
            bind(Path(job['target']).with_suffix('.npz'), score['predictions_sha256'])
            filters.append(dict(case_id=case_id, stage='truth', candidate=job['candidate']['id'],
                                seed=job['seeds'][0], particles=1024))
        if len(nominee['finalists']) != 3 or required['id'] not in {r['id'] for r in nominee['finalists']}:
            raise ValueError('Missing high-precision incumbent or finalists')
        values = {}
        for candidate in nominee['finalists']:
            file = Path(candidate['score_file']); score = read(bind(file))
            bind(file.with_suffix('.npz'), score['predictions_sha256'])
            with np.load(file.with_suffix('.npz')) as z:
                np.testing.assert_array_equal(z['seeds'], truth_seeds)
                value = nll(z['observed_probability']); jk = jackknife_scores(z['observed_probability'])
            np.testing.assert_allclose(value, candidate['mean_nll'], atol=1e-12, rtol=0)
            values[candidate['id']] = value, jk
            confirmed.append(dict(case_id=case_id, candidate=candidate['id'],
                beta_init=candidate['hyperparams'][profile.BETA], gamma=candidate['hyperparams'][profile.GAMMA],
                nll=value, truth_nll=truth_nll, delta_truth=value-truth_nll,
                paired_mc_se=jackknife_se(jk-truth_jk), previous=candidate['id']==required['id'],
                selected=candidate['id']==selected['id']))
        old, old_jk = values[required['id']]; new, new_jk = values[selected['id']]
        changes.append(dict(case_id=case_id, previous_id=required['id'], selected_id=selected['id'],
            previous_nll=old, selected_nll=new, delta_previous=new-old,
            paired_mc_se_previous=jackknife_se(new_jk-old_jk), truth_nll=truth_nll,
            remaining_gap=new-truth_nll, paired_mc_se_truth=jackknife_se(new_jk-truth_jk)))
        panels.append(case_id)
    if len(filters) != 320 or len(grid) != 112 or len(confirmed) != 6:
        raise ValueError('Incomplete bounded profile')
    output.mkdir(parents=True)
    tables = dict(grid=pd.DataFrame(grid), confirmation=pd.DataFrame(confirmed),
                  changes=pd.DataFrame(changes), filters=pd.DataFrame(filters))
    for name, table in tables.items():
        table.to_csv(output/(name+'.csv'), index=False)
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':7, 'axes.spines.top':False,
                         'axes.spines.right':False, 'legend.frameon':False}):
        fig, axes = plt.subplots(2,2,figsize=(183/25.4,155/25.4))
        fig.subplots_adjust(left=.15,right=.89,bottom=.17,top=.84,hspace=.70,wspace=.70)
        vmin = tables['grid'].nll.min(); vmax = tables['grid'].nll.max()
        for col, case_id in enumerate(panels):
            group = tables['grid'].query('case_id==@case_id')
            matrix = group.pivot(index='beta_init', columns='gamma', values='nll')
            ax = axes[0,col]; im = ax.imshow(matrix, origin='lower', aspect='auto', cmap='viridis_r', vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(matrix.columns)), [f'{v:g}' for v in matrix.columns], rotation=45)
            ax.set_yticks(range(len(matrix.index)), [f'{v:g}' for v in matrix.index])
            ax.set(xlabel='Pairing evidence retention', ylabel='Initial precision')
            ax.set_title(case_id+'\nScreen: 56 points, 2 × 128 particles', fontsize=7)
            for flag, marker in [('previous','o'),('selected','*')]:
                row = group[group[flag]].iloc[0]
                ax.scatter(matrix.columns.get_loc(row.gamma),matrix.index.get_loc(row.beta_init),
                           marker=marker,s=48,facecolors='none',edgecolors='#111111',linewidths=1.1)
            ax = axes[1,col]
            values = tables['confirmation'].query('case_id==@case_id').sort_values('nll')
            labels = []
            for i, row in enumerate(values.itertuples()):
                tag = 'Selected' if row.selected else 'Other'
                if row.previous: tag += ' / previous'
                labels.append(f'{tag}\nβ₀={row.beta_init:g}, γ={row.gamma:g}')
                ax.errorbar(row.delta_truth, i, xerr=2*row.paired_mc_se, fmt='o', ms=4,
                            color='#3B8C85' if row.selected else '#527697', capsize=2)
            ax.axvline(0,color='#666666',ls=':',lw=.8)
            ax.set_yticks(range(3),labels,fontsize=6.5);ax.invert_yaxis();ax.set_ylim(2.6,-.6)
            ax.set_xlabel('NLL / trial minus generating point\nPositive values: worse fit',fontsize=7)
            ax.set_title('Confirmation: 8 × 1024 particles\nBars: ±2 paired Monte Carlo SE',fontsize=7)
        cax=fig.add_axes([.905,.56,.015,.24]);fig.colorbar(im,cax=cax,label='Screen NLL / trial')
        fig.suptitle('Initial precision × pairing retention · optimizer diagnostic',y=.985,fontsize=10)
        fig.legend(handles=[Line2D([],[],marker='o',mfc='none',color='#555555',ls='',label='Previous estimate'),
                            Line2D([],[],marker='*',mfc='none',color='#555555',ls='',label='Selected after confirmation')],
                   loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,fontsize=7)
        fig.text(.5,.075,'Seven other coordinates fixed at previous estimates; generating points may differ on them.',ha='center',fontsize=6.5)
        fig.text(.5,.035,'Retrospective diagnostic; confirmation x scales differ. Bars show Monte Carlo error only.',ha='center',fontsize=6.5)
        fig.savefig(output/'precision_pairing_profile.png',dpi=300,facecolor='white');plt.close(fig)
    # Reuse the established state figure; only its estimate label changes to
    # disclose the retrospective status of this additional search.
    state_labels = dict(recovery_figure.LABELS, estimate='Retrospective profile estimate')
    original_labels = recovery_figure.LABELS
    recovery_figure.LABELS = state_labels
    try:
        recovery_figure.render(study/'states', output/'states')
    finally:
        recovery_figure.LABELS = original_labels
    bind(Path(recovery_figure.__file__))
    page='<!doctype html><meta charset="utf-8"><title>精度与配对证据联合检查</title><style>body{max-width:1100px;margin:32px auto;font:16px sans-serif}img{width:100%}</style>'
    page+='<h1>精度与配对证据联合检查</h1><p>针对两个已知搜索缺口的事后优化诊断。每例检查56点，其他七个参数固定，原估计点始终进入高精度确认。生成点的其他参数可能不同。误差条表示计算误差。</p>'
    page+='<img src="precision_pairing_profile.png">'
    page+='<h2>追加搜索后的内部状态</h2><p>与已知生成参数的推断共同呈现。灰蓝区间只包含固定参数下的潜在状态不确定性。</p>'
    page+=''.join('<img src="states/'+html.escape(case_id)+'.png">' for case_id in panels)
    for name in ('changes','confirmation'):
        page+='<h2>'+html.escape(name)+'</h2>'+tables[name].to_html(index=False)
    (output/'index.html').write_text(page,encoding='utf-8')
    shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    bind(Path(__file__))
    write(output/'completion.json',dict(status='complete',inputs_sha256=inputs,filters=len(filters),
        scope=profile.SCOPE,extra_filters=0,format='PNG',dpi=300,width_mm=183,state_plot_labels=state_labels,
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.suffix in ('.csv','.png','.html')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.study,args.output)
