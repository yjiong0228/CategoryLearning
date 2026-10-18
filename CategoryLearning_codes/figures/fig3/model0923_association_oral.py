"""Audit and display oral evidence for the fixed-catalogue association study.

Use --source ORAL_RESULTS --output NEW_DIRECTORY. The loader independently sums
saved component masses and checks both scoring domains before plotting. Two PNG
discussion figures show report coverage over time and the behavioral/oral tradeoff.
All trial rows remain in the source tables; complete reports define plotted means.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis import model_0923_association_oral as analysis

ARMS = ('FIXED_SHARED', 'LEARN_SHARED', 'FIXED_STABLE', 'LEARN_STABLE')
COLORS = ('#758399', '#B18358', '#758399', '#277F8E')
STYLES = (':', '-', '--', '-')
NAMES = ('Fixed / shared', 'Learn / shared', 'Fixed / stable', 'Learn / stable')
DOMAINS, VERDICTS = analysis.DOMAINS, analysis.LABELS
read, digest = analysis.read, analysis.digest


def time_blocks(frame: pd.DataFrame, width: int = 32) -> pd.DataFrame:
    """Use identical report identities in every arm; retain empty display blocks."""
    rows = []
    identity = ['ordinal_trial', 'choice', 'scope', 'tokens', 'training_trials']
    for subject, own in frame.groupby('iSub', sort=True):
        if set(own.arm_name) != set(ARMS):
            raise ValueError('Need all four association arms.')
        anchor = None
        for arm in ARMS:
            group = own.loc[own.arm_name.eq(arm)].sort_values('ordinal_trial')
            np.testing.assert_array_equal(group.ordinal_trial, np.arange(1, len(group) + 1))
            current = group[identity].reset_index(drop=True)
            if anchor is not None:
                pd.testing.assert_frame_equal(current, anchor)
            anchor = current
            cuts = group.training_trials.unique()
            if len(cuts) != 1 or int(cuts[0]) % width:
                raise ValueError('Display blocks must preserve the training boundary.')
            for index, part in group.groupby((group.ordinal_trial - 1) // width, sort=True):
                complete = part.loc[part.scope.eq('complete')]
                rows.append(dict(subject=int(subject), arm_name=arm, block=int(index),
                    midpoint=float(part.ordinal_trial.mean()), end=int(part.ordinal_trial.max()),
                    trials=len(part), reports=len(complete), fragments=int(part.scope.eq('parsed_fragment').sum()),
                    unresolved=int((~part.scope.isin(['complete', 'parsed_fragment'])).sum()),
                    training_trials=int(cuts[0]),
                    **{domain+'_'+v:float(complete[domain+'_'+v].mean())
                       for domain in DOMAINS for v in VERDICTS}))
    return pd.DataFrame(rows)


def collect(source: Path) -> tuple[dict, dict, dict]:
    """Check provenance, frozen predictions and every report's mass arithmetic."""
    done, pro = read(source/'completion.json'), read(source/'protocol.json')
    state_done = read(source/'behavior_states_complete.json')
    if (done['status'] != 'complete' or done['smoke']
            or digest(source/'protocol.json') != done['protocol_sha256']
            or digest(source/'behavior_states_complete.json') != done['behavior_states_sha256']
            or state_done['oral_tables_loaded'] or state_done['status'] != 'complete'):
        raise ValueError('Need completed behavior-first oral evaluation.')
    inputs = dict(done['input_sha256'])
    analysis.verify(inputs)
    for name, sha in pro['source_sha256'].items():
        if digest(source/'source'/name) != sha:
            raise ValueError('Archived scientific source changed.')
    for name, sha in done['artifacts'].items():
        if digest(source/name) != sha:
            raise ValueError('Oral result changed.')
        inputs[str(source/name)] = sha
    for name in ('completion.json', 'protocol.json', 'behavior_states_complete.json'):
        inputs[str(source/name)] = digest(source/name)
    frame = pd.read_csv(source/'trial_scores.csv')
    summary = pd.read_csv(source/'summary.csv')
    contrasts = pd.read_csv(source/'contrasts.csv')
    blocks = time_blocks(frame)
    nominees = {(n['subject'], n['arm_name']): n for n in pro['nominees']}
    if set(nominees) != set(map(tuple, frame[['iSub', 'arm_name']].drop_duplicates().to_numpy())):
        raise ValueError('Nomination groups differ from oral groups.')
    queries = pd.read_csv(source/'queries.csv')
    labels = {}
    for row in queries.itertuples():
        file = source/'queries'/row.query_id
        job = source/'jobs'/('Q_'+row.query_id+'.json')
        record = read(file.with_suffix('.json'))
        if record['job_sha256'] != digest(job) or record['arrays_sha256'] != digest(file.with_suffix('.npz')):
            raise ValueError('Query identity changed.')
        inputs[str(job)] = digest(job)
        with np.load(file.with_suffix('.npz')) as z:
            labels[analysis.content_key(row.tokens), row.region] = z['labels'].copy()
    audit = analysis.ROOT/pro['settings']['oral_audit']
    old_queries = pd.read_csv(audit/'constraint_queries.csv')
    old_matches = pd.read_csv(audit/'component_matches.csv').set_index(['query_id', 'rule', 'component'])
    comparisons = pd.read_csv(source/'original_query_checks.csv')
    old_lookup = old_queries.set_index('query_id')
    for row in comparisons.itertuples():
        old = old_lookup.loc[row.original_query_id]
        new = labels[analysis.content_key(old.tokens), int(old.choice)-1]
        for index, match in old_matches.loc[row.original_query_id].iterrows():
            for d, domain in enumerate(DOMAINS):
                if new[*index, d] != VERDICTS.index(match[domain]):
                    raise ValueError('Original oral label changed.')
    maximum_error = 0.
    for (subject, arm), group in frame.groupby(['iSub', 'arm_name'], sort=True):
        group = group.sort_values('ordinal_trial'); name = f'S{subject}_{arm}'
        job_file = source/'jobs'/(name+'.json'); job = read(job_file)
        record = read(source/'states'/(name+'.json')); file = source/'states'/(name+'.npz')
        if (record['job_sha256'] != digest(job_file) or record['arrays_sha256'] != digest(file)
                or job['sources'] != pro['source_sha256']
                or any(job[k] != nominees[subject, arm][k] for k in ('beta', 'gamma'))):
            raise ValueError('Behavior state identity changed.')
        inputs[str(job_file)] = digest(job_file)
        with np.load(file) as z: arrays = {k:z[k].copy() for k in z.files}
        with np.load(job['prediction_reference']) as z: reference = z['probabilities']
        residual = float(np.max(np.abs(arrays['prediction'] - reference)))
        np.testing.assert_allclose(arrays['prediction'], reference, rtol=0, atol=1e-12)
        np.testing.assert_allclose(residual, record['max_prediction_error'], rtol=0, atol=0)
        maximum_error = max(maximum_error, residual)
        np.testing.assert_array_equal(arrays['choices'], group.choice)
        p = arrays['prediction'][np.arange(len(group)), arrays['choices']-1]
        np.testing.assert_allclose(arrays['prediction'].sum(axis=1), 1., rtol=0, atol=1e-12)
        np.testing.assert_allclose(arrays['choice_probability'], p, rtol=0, atol=1e-12)
        np.testing.assert_allclose(group.choice_probability, p, rtol=0, atol=1e-12)
        prior, joint = arrays['prior_component_mass'], arrays['choice_component_joint_mass']
        for mass, total in ((prior, np.ones(len(group))), (joint, p)):
            if not np.isfinite(mass).all() or np.any(mass < 0): raise ValueError('Invalid component mass.')
            np.testing.assert_allclose(mass.sum(axis=(1,2,3)), total, rtol=0, atol=1e-12)
        for t, row in enumerate(group.itertuples()):
            for d, domain in enumerate(DOMAINS):
                for k, verdict in enumerate(VERDICTS):
                    if pd.isna(row.query_id):
                        expected, unconditioned = np.nan, np.nan
                    else:
                        tokens = analysis.content_key(row.tokens)
                        mask = np.stack([labels[tokens, g][..., d] == k for g in range(4)], axis=1)
                        expected = float(joint[t][mask].sum()/p[t])
                        unconditioned = float(prior[t][mask].sum())
                    for prefix, value in (('', expected), ('prior_', unconditioned)):
                        np.testing.assert_allclose(getattr(row, prefix+domain+'_'+verdict), value,
                                                   rtol=0, atol=1e-12, equal_nan=True)
        ordinal = group.ordinal_trial.to_numpy(); cut = group.training_trials.iloc[0]
        windows = dict(whole=ordinal>0, training=ordinal<=cut, test=ordinal>cut,
                       first64=ordinal<=64, last64=ordinal>len(group)-64)
        for row in summary.loc[summary.subject.eq(subject)&summary.arm_name.eq(arm)].itertuples():
            selected = group.loc[windows[row.window]&group.scope.eq(row.scope)]
            if len(selected) != row.reports: raise ValueError('Summary report count changed.')
            for prefix in ('', 'prior_'):
                for v in VERDICTS:
                    np.testing.assert_allclose(getattr(row, prefix+v), selected[prefix+row.domain+'_'+v].mean(),
                                               rtol=0, atol=1e-12)
    keys = ['subject','arm_name','window','scope','domain']
    indexed = summary.set_index(keys)
    for row in contrasts.itertuples():
        left = indexed.loc[(row.subject,row.arm_name,row.window,row.scope,row.domain)]
        right = indexed.loc[(row.subject,row.reference,row.window,row.scope,row.domain)]
        if left.reports != right.reports or left.reports != row.reports:
            raise ValueError('Comparison report masks differ.')
        np.testing.assert_allclose(row.delta_support, left.support-right.support, rtol=0, atol=1e-12)
    auto = analysis.ROOT/pro['settings']['source']; auto_pro = read(auto/'protocol.json')
    fit = analysis.ROOT/auto_pro['settings']['source']
    for parent in (auto, fit):
        complete = read(parent/'completion.json')
        if complete['status'] != 'complete' or complete['smoke']:
            raise ValueError('Parent study incomplete.')
        for name, sha in complete['artifacts'].items():
            if digest(parent/name) != sha: raise ValueError('Parent artifact changed.')
            inputs[str(parent/name)] = sha
        inputs[str(parent/'completion.json')] = digest(parent/'completion.json')
    behavior = pd.read_csv(fit/'summary.csv').set_index(['subject','arm_name'])
    autonomous = pd.read_csv(auto/'contrasts.csv').set_index(['subject','arm_name','phase'])
    gains = []
    primary = contrasts.loc[contrasts.window.eq('test')&contrasts.scope.eq('complete')&contrasts.domain.eq('structural')]
    for row in primary.itertuples():
        a = behavior.loc[(row.subject,row.arm_name)]; b = behavior.loc[(row.subject,row.reference)]
        c = autonomous.loc[(row.subject,row.arm_name,'test')]
        gains.append(dict(subject=row.subject, arm_name=row.arm_name, reference=row.reference, reports=row.reports,
            gain_test_nll=float(b.test_nll-a.test_nll), gain_test_energy_u=-float(c.delta_energy_u),
            energy_mc_se=float(c.delta_energy_u_mc_se), gain_test_support=row.delta_support))
    checks = dict(trial_rows=len(frame), groups=len(nominees), queries=len(queries),
                  original_query_checks=len(comparisons), max_prediction_error=maximum_error,
                  component_mass_scores='all trials, both domains, prior and choice-conditioned',
                  source_files=len(pro['source_sha256']), passed=True)
    return dict(trials=frame, blocks=blocks, summary=summary, contrasts=contrasts, gains=pd.DataFrame(gains)), inputs, checks


def plot(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new association oral figure directory.')
    tables, inputs, checks = collect(source)
    output.mkdir(parents=True)
    for name, table in tables.items(): table.to_csv(output/(name+'.csv'), index=False)
    contract = dict(claim='Learning label associations improves some generated behavior while losing explicit oral support.',
        archetype='quantitative grid', backend='python', adaptation='style-only; new association-state validation',
        formats=['PNG'], dpi=300, timecourse_mm=[183,180], comparison_mm=[183,90],
        panels=['complete/fragment/unresolved report counts', 'explicit support', 'undetermined mass',
                'later-choice NLL gain', 'later autonomous U Energy gain', 'later explicit support gain'],
        statistics=dict(timecourse='complete-report means in nonoverlapping 32-trial display blocks',
            replicates='three development participants; 32 autonomous paths per arm',
            uncertainty='one paired path-jackknife Monte Carlo SE for U Energy only; no inferential test',
            parameters='training choices only; each arm uses its own frozen nominee'),
        limits=['support is representation correspondence, not a report likelihood',
            'display blocks are not a cognitive memory window', 'all 152 geometries always available',
            'deterministic state estimates omit parameter uncertainty', 'incomplete reports retained separately'],
        inputs_sha256=inputs, checks=checks,
        preflight_exceptions={'EXPORT-VECTOR':'PNG-only repository instruction',
                              'EXPORT-RASTER':'PNG-only repository instruction',
                              'RASTER-DPI':'300 dpi discussion figures'})
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'], 'font.size':7,
        'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    width_inches, height_inches = np.array([183.,180.])/25.4
    fig, axes = plt.subplots(3,3,figsize=(width_inches,height_inches))
    for column, (subject, own) in enumerate(tables['blocks'].groupby('subject',sort=True)):
        fixed = own.loc[own.arm_name.eq(ARMS[0])]
        ax = axes[0,column]
        bottom = np.zeros(len(fixed))
        for key, color, label in [('reports','#73899B','Complete'),('fragments','#B9C3CC','Fragment'),('unresolved','#E6E6E6','Unresolved')]:
            ax.bar(fixed.midpoint,fixed[key],bottom=bottom,width=29,color=color,label=label)
            bottom += fixed[key].to_numpy()
        ax.set(title=f'Subject {subject}',ylim=(0,34),yticks=[0,16,32])
        for arm, color, style, name in zip(ARMS,COLORS,STYLES,NAMES):
            part = own.loc[own.arm_name.eq(arm)]
            for row, key in ((1,'structural_support'),(2,'structural_undetermined')):
                axes[row,column].plot(part.midpoint,part[key],color=color,linestyle=style,linewidth=1.2,label=name)
        for row in range(3):
            ax = axes[row,column]
            ax.axvline(int(fixed.training_trials.iloc[0])+.5,color='#555555',linestyle='--',linewidth=.6)
            ax.set_xlim(.5,int(fixed.end.max())+.5)
            ax.spines[['top','right']].set_visible(False)
            ax.text(-.14,1.05,chr(97+row*3+column),transform=ax.transAxes,weight='bold',fontsize=8)
            if row: ax.set(ylim=(-.025,1.025),yticks=[0,.5,1])
            if row==2: ax.set_xlabel('Trial')
            if column==0: ax.set_ylabel(['Reports per block','Explicit support','Undetermined mass'][row])
    axes[0,0].legend(loc='upper left',bbox_to_anchor=(0,1.37),ncol=3,fontsize=6)
    handles, names = axes[1,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=4,bbox_to_anchor=(.5,.055),fontsize=6)
    fig.suptitle('Oral correspondence after learning category associations',fontsize=10,y=.985)
    fig.text(.5,.02,'32-trial blocks; complete-report means. Dashed vertical line: training boundary.',ha='center',fontsize=6)
    fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.85,hspace=.35,wspace=.28)
    fig.savefig(output/'oral_timecourse.png',dpi=300);plt.close(fig)
    width_inches, height_inches = np.array([183.,90.])/25.4
    fig, axes = plt.subplots(1,3,figsize=(width_inches,height_inches),sharey=True)
    gains = tables['gains'].sort_values(['subject','arm_name']).reset_index(drop=True)
    for column, key in enumerate(['gain_test_nll','gain_test_energy_u','gain_test_support']):
        ax = axes[column]
        for i, row in enumerate(gains.itertuples()):
            color = COLORS[1] if row.arm_name=='LEARN_SHARED' else COLORS[3]
            ax.errorbar(getattr(row,key),i,xerr=row.energy_mc_se if column==1 else None,
                        marker='o',color=color,markersize=4,capsize=2,linewidth=.8)
        ax.axvline(0,color='#777777',linewidth=.6)
        ax.set(title=['Later-choice prediction','Autonomous later behavior','Later oral correspondence'][column],
               xlabel=['NLL improvement','U Energy improvement','Support improvement'][column])
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.12,1.08,chr(97+column),transform=ax.transAxes,weight='bold',fontsize=8)
    axes[0].set_yticks(np.arange(len(gains)),[f'S{r.subject} / '+('shared' if r.arm_name=='LEARN_SHARED' else 'stable') for r in gains.itertuples()])
    axes[0].invert_yaxis()
    fig.suptitle('Behavioral gains do not imply improved oral correspondence',fontsize=10,y=.98)
    fig.text(.5,.08,'Positive values favor learned associations; each comparison uses its matching fixed-association arm.',ha='center',fontsize=6)
    fig.text(.5,.025,'U Energy bars: one Monte Carlo SE across 32 paired paths. Training-selected parameters; three development cases.',ha='center',fontsize=6)
    fig.subplots_adjust(left=.15,right=.985,top=.78,bottom=.25,wspace=.34)
    fig.savefig(output/'evidence_comparison.png',dpi=300);plt.close(fig)
    caption=('四方案各自使用训练行为选定的参数。口述仅在行为状态重放完成后评价；上图保存全部报告的解析范围，'
        '中下图在相同完整报告上计算显式支持与未确定质量。完整原话的所有谓词共同参与评价，支持质量描述表征对应程度。'
        '比较图三项均以正值表示关联学习的改善；U Energy误差线为32条配对路径的一个模拟标准误，'
        '行为似然与口述状态为确定性计算，参数不确定性尚未纳入。Shared取η=γ，Stable取η=1。'
        '目录中的152条几何规则始终可用，当前结果用于机制开发。')
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}img{width:100%}</style>'
        '<h1>类别关联学习：行为与口述证据</h1><p>'+caption+'</p>'
        '<img src="evidence_comparison.png"><img src="oral_timecourse.png">'
        +gains.to_html(index=False)+'<p><a href="trials.csv">全部逐题记录</a> · <a href="summary.csv">两种评价与解析范围</a></p></html>')
    shutil.copy2(__file__,output/'figure_source.py')
    (output/'manifest.json').write_text(json.dumps(dict(status='complete',checks=checks,
        plotting_source_sha256=digest(Path(__file__)),artifacts={p.name:digest(p) for p in output.iterdir()}),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); plot(args.source,args.output)
