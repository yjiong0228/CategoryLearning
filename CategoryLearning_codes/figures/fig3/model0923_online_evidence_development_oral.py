"""Plot audited fixed-parameter ONLINE evidence in three development participants.

All complete oral reports enter non-overlapping 32-trial means; all choices enter
the cumulative likelihood difference. Use --results ROOT --output NEW. The
183 x 185 mm PNG is a development diagnostic, with both oral readings retained.
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
import numpy as np
import pandas as pd


SUBJECTS = {107: (192, 128, 75), 207: (960, 672, 626), 306: (768, 512, 636)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(results: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    inputs, copied, audits = {}, {}, {}
    folders = {'oral': results / 'online_evidence_development_oral_full_audit_20261018_v1'}
    folders.update({str(s): results / f'online_evidence_development_scores_{s}_audit_20261018_v1'
                    for s in SUBJECTS})
    for key, folder in folders.items():
        path = folder / 'validation.json'
        audit = json.loads(path.read_text())
        if audit['status'] != 'complete' or audit['smoke']:
            raise ValueError('Full independent audits required')
        audits[key] = audit
        inputs[str(path.resolve())] = digest(path)
        for field in ('inputs_sha256', 'source_sha256'):
            for name, sha in audit[field].items():
                if name in inputs and inputs[name] != sha:
                    raise ValueError('Incompatible audit provenance')
                inputs[name] = sha
    if (audits['oral']['comparisons'], audits['oral']['unique_states']) != (6, 24):
        raise ValueError('Expected six oral comparisons from 24 state replays')
    for name, sha in inputs.items():
        if digest(Path(name)) != sha:
            raise ValueError('Changed audited input: ' + name)

    def table(folder: Path, name: str, label: str, artifacts: dict | None = None) -> pd.DataFrame:
        path = folder / name
        expected = artifacts[name] if artifacts is not None else inputs.get(str(path.resolve()))
        if expected is None or digest(path) != expected:
            raise ValueError('Unaudited table: ' + str(path))
        inputs[str(path.resolve())] = expected
        copied[label] = path
        return pd.read_csv(path)

    feedback = table(folders['oral'], 'feedback_scores.csv', 'feedback_scores.csv', audits['oral']['artifacts'])
    blocks, summaries, cumulatives, coverage, choice_summaries, checks = [], [], [], [], [], 0
    for subject, (trials_n, cutoff, reports_n) in SUBJECTS.items():
        audit = audits[str(subject)]
        if (audit['filters'], audit['particles'], audit['trials'], audit['training_trials']) != (8, 128, trials_n, cutoff):
            raise ValueError('Changed participant evaluation')
        choices = table(folders[str(subject)], 'verified_scores.csv', f'{subject}_choice_scores.csv', audit['artifacts'])
        choice_summaries.append(choices.assign(subject=subject))
        trial_choices = table(results / f'online_evidence_development_scores_20261018_v1/subject_{subject}',
                              'trial_scores.csv', f'{subject}_choice_trials.csv').sort_values('trial')
        np.testing.assert_array_equal(trial_choices.trial, np.arange(1, trials_n + 1))
        probabilities = trial_choices[['baseline_probability', 'online_probability']].to_numpy(float)
        if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities > 1)):
            raise ValueError('Choice logarithms require probabilities in (0, 1]')
        delta = np.log(probabilities[:, 0]) - np.log(probabilities[:, 1])
        np.testing.assert_allclose(delta, trial_choices.delta_choice_nll, atol=2e-12)
        checks += trials_n
        for phase in ('prefix', 'later', 'full'):
            mask = np.ones(trials_n, bool) if phase == 'full' else (trial_choices.phase == phase).to_numpy()
            np.testing.assert_allclose(delta[mask].mean(), choices.loc[choices.phase == phase, 'delta_nll'].iloc[0], atol=2e-12)
            checks += 1
        cumulatives.extend(dict(subject=subject, trial=int(t), cumulative_delta=float(d))
                           for t, d in zip(trial_choices.trial, delta.cumsum()))
        for reading in ('midpoint', 'relative'):
            folder = results / f'online_evidence_development_oral_{subject}_{reading}_20261018_v1'
            trials = table(folder, 'trials.csv', f'{subject}_{reading}_trials.csv')
            summary = table(folder, 'summary.csv', f'{subject}_{reading}_summary.csv')
            summaries.append(summary.assign(reading=reading))
            for arm in ('BASE', 'ONLINE'):
                own = trials.loc[trials.arm == arm].sort_values('state_trial')
                np.testing.assert_array_equal(own.state_trial, np.arange(trials_n))
                complete = own.loc[own.scope == 'complete'].copy()
                if len(complete) != reports_n:
                    raise ValueError('Changed complete-report denominator')
                values = complete[['structural_support', 'structural_conflict', 'structural_undetermined']].to_numpy(float)
                if not np.isfinite(values).all() or np.any((values < -2e-12) | (values > 1 + 2e-12)):
                    raise ValueError('Invalid oral masses')
                np.testing.assert_allclose(values.sum(axis=1), 1, atol=2e-12)
                checks += reports_n
                for window in ('whole', 'training', 'test'):
                    subset = complete if window == 'whole' else complete.loc[complete.phase == window]
                    row = summary.query('arm == @arm and window == @window and domain == "structural" and subset == "complete"').iloc[0]
                    if int(row.reports) != len(subset):
                        raise ValueError('Changed window report counts')
                    for field, column in [('support', 'structural_support'), ('prior_support', 'prior_structural_support'),
                                          ('availability_all_components', 'structural_availability_all_components')]:
                        np.testing.assert_allclose(row[field], subset[column].mean(), atol=2e-12)
                        checks += 1
                if arm == 'BASE':
                    coverage.append(dict(subject=subject, reading=reading, total_trials=trials_n, complete_reports=reports_n,
                        catalogue_supported=int((complete.structural_support_all_components > 0).sum())))
                complete['block'] = complete.state_trial.astype(int) // 32
                for block, group in complete.groupby('block'):
                    blocks.append(dict(subject=subject, reading=reading, arm=arm, block=int(block),
                        trial=min((int(block) + 1) * 32, trials_n), reports=len(group), support=group.structural_support.mean()))
    block_table, cumulative = pd.DataFrame(blocks), pd.DataFrame(cumulatives)
    summary, choices, coverage = pd.concat(summaries, ignore_index=True), pd.concat(choice_summaries), pd.DataFrame(coverage)
    contract = dict(conclusion='Late oral support improves for 207 and 306 under both readings, while early choice loss increases.',
        evidence='Columns: all three participants. Rows: absolute oral, relative oral, all cumulative choice costs.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=185, dpi=300, formats=['PNG'],
        mapping='state_trial is zero-based; displayed trial/block ends are one-based; support is a mass in [0,1]; NLL is in nats.',
        participant_n=3, participants=list(SUBJECTS), filter_seeds=4, particles=128,
        center='Mean over complete reports after pooling joint state masses across filters and normalizing each trial.',
        uncertainty='Time plots show means, no interval; linked table MCSE uses paired whole-filter deletion.',
        split={str(s):dict(trials=v[0], training_trials=v[1], complete_reports=v[2]) for s,v in SUBJECTS.items()},
        data_use='Fixed behavioral parameters; oral never fits parameters; both bare-long/short readings fixed across variants.',
        exclusions='Oral primary plot requires scope=complete; all remaining reports retained in trial tables. Every choice retained.',
        transformations='Non-overlapping32-trial oral means and cumulative choice NLL; no sampling or smoothing.',
        hypothesis_tests='None', multiple_comparison_correction='Not applicable',
        adaptation='Style-only inheritance; participant columns and input verification implemented for this comparison.',
        limits='Development participants, old fixed parameters, 128 particles. 107 catalogue support covers only1or2of75complete reports.',
        export_exceptions={'EXPORT-VECTOR':'Repository PNG-only instruction','EXPORT-RASTER':'Repository PNG-only instruction','RASTER-DPI':'300dpi discussion output'})
    output.mkdir(parents=True)
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    for name, path in copied.items():
        shutil.copy2(path, output / name)
    for name, frame in [('plot_blocks',block_table),('plot_summary',summary),('plot_cumulative_nll',cumulative),
                        ('plot_choice_scores',choices),('report_coverage',coverage)]:
        frame.to_csv(output / (name + '.csv'), index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
    colors = {'BASE':'#68778B', 'ONLINE':'#257B89'}
    fig, axes = plt.subplots(3, 3, figsize=(183/25.4,185/25.4))
    fig.subplots_adjust(left=.085,right=.975,top=.875,bottom=.15,wspace=.43,hspace=.5)
    for col, (subject, (trials_n, cutoff, reports_n)) in enumerate(SUBJECTS.items()):
        for row, reading in enumerate(('midpoint','relative')):
            ax = axes[row,col]
            for arm in ('BASE','ONLINE'):
                own = block_table.query('subject == @subject and reading == @reading and arm == @arm')
                ax.plot(own.trial,own.support,color=colors[arm],ls='-' if arm=='ONLINE' else '--',lw=1.1)
            ax.set(ylim=(-.035,1.035),ylabel='Structural support mass',
                   title=f'{subject}: '+(f'absolute (n={reports_n})' if row==0 else 'relative reading'))
            if subject == 107:
                n = int(coverage.query('subject == @subject and reading == @reading').catalogue_supported.iloc[0])
                ax.text(.04,.82,f'Catalogue covers {n}/75\ncomplete reports',transform=ax.transAxes,fontsize=6.2)
        own = cumulative.loc[cumulative.subject == subject]
        ax = axes[2,col]
        ax.plot(own.trial,own.cumulative_delta,color='#B27737',lw=1.1)
        ax.axhline(0,color='#AAAAAA',lw=.6)
        ax.set(title=f'{subject}: cumulative choice cost',ylabel='ONLINE − BASE NLL')
        for ax in axes[:,col]:
            ax.axvline(cutoff,color='#999999',ls=':',lw=.7)
            ax.set(xlabel='Trial',xlim=(0,trials_n),xticks=[0,cutoff,trials_n])
    for i,ax in enumerate(axes.flat):
        ax.text(-.16,1.07,chr(97+i),transform=ax.transAxes,fontsize=8,fontweight='bold')
    fig.suptitle('Online evidence: cross-condition oral gains and early choice costs',fontsize=8.5,y=.975)
    fig.text(.5,.94,'Fixed old parameters | SEARCH | 4 paired filters × 128 particles per variant',ha='center',fontsize=6.5)
    handles=[Line2D([],[],color=colors[a],ls='-' if a=='ONLINE' else '--',lw=1.2,label=a) for a in ('BASE','ONLINE')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.text(.5,.045,'Oral: all complete reports, 32-trial means. Choice: every trial; positive cumulative NLL = higher loss.',ha='center',fontsize=6.2)
    fig.text(.5,.018,'Dotted lines mark original training boundaries. All participants are development cases; no population inference.',ha='center',fontsize=6.2)
    fig.savefig(output / 'online_evidence_development_oral.png',dpi=300)
    plt.close(fig)
    late=feedback.query('window == "test" and domain == "structural" and metric == "conditioned" and feedback_group == "all"')
    partial=feedback.query('subject == 306 and window != "whole" and domain == "structural" and metric == "conditioned"')
    fields=['subject','reading','complete_reports','baseline_support','online_support','delta_support','delta_support_mcse']
    caption=('三列依次为107、207和306；前两行保留绝对与相对两种读法，每点使用32题内全部完整报告。'
        '107的75份完整报告中，当前目录能支持的报告分别仅1份和2份。其接近零的分数首先提示目录覆盖与对齐关系需要检查。'
        '第三行累计全部选择的负对数似然差；正值表示ONLINE预测损失更高，曲线下降表示该阶段提供收益。'
        '竖线为原参数估计的前段边界。107和306沿用此前POOL参数后应用SEARCH；207沿用SEARCH联合重估参数。'
        '口述用于外部表征检验，两种读法在模型间统一固定。结构支持质量度量模型作答依据对编码内容的支持；口述句子的生成概率尚未建模。'
        '下表误差为四个配对滤波的模拟标准误，参数不确定性和有限粒子偏差尚未计入。')
    page=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>三名开发被试：在线证据与口述</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}td,th{padding:6px}</style>'
        '<h1>累积反馈证据改善规则使用，早期回答仍有代价</h1><p>'+caption+'</p>'
        '<img src="online_evidence_development_oral.png" alt="三被试双读法口述支持及完整选择损失">'
        '<h2>后段全部完整报告</h2>'+late[fields].to_html(index=False,float_format=lambda x:f'{x:.6f}')
        +'<h2>306按实际反馈分层</h2>'+partial[['reading','window','feedback_group']+fields[2:]].to_html(index=False,float_format=lambda x:f'{x:.6f}')
        +'<p><a href="feedback_scores.csv">全部反馈分层及加权前结果</a> · <a href="plot_summary.csv">完整口述评分及在场概率</a> · '
        '<a href="plot_choice_scores.csv">选择评分</a> · <a href="report_coverage.csv">报告覆盖</a> · '
        '<a href="figure_contract.json">图示范围与误差定义</a></p></html>')
    (output/'index.html').write_text(page)
    for name,sha in inputs.items():
        if digest(Path(name))!=sha:
            raise ValueError('Scientific input changed during plotting')
    shutil.copy2(__file__,output/'plot_source.py')
    (output/'manifest.json').write_text(json.dumps(dict(status='complete',numeric_checks=checks,
        input_sha256=inputs,source_sha256={str(Path(__file__).resolve()):digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plot(args.results,args.output)
