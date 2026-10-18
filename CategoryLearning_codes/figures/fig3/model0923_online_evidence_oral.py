"""Plot the audited S208 fixed-parameter ONLINE oral and choice comparison.

Use --results ROOT --oral-validation AUDIT --choice-validation AUDIT --output NEW.
Both oral readings and both proposal schemes use the same frozen behavioral
parameters. All complete reports enter 32-trial blocks; all choices enter the
cumulative likelihood comparison. Export: PNG only, 183 x 150 mm, 300 dpi.
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(results: Path, oral_validation: Path, choice_validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    inputs, audits = {}, []
    for folder in (oral_validation, choice_validation):
        path = folder / 'validation.json'
        record = json.loads(path.read_text())
        if record['status'] != 'complete' or record['smoke']:
            raise ValueError('Complete independent audits required')
        inputs[str(path.resolve())] = digest(path)
        for field in ('inputs_sha256', 'source_sha256'):
            for name, sha in record[field].items():
                if name in inputs and inputs[name] != sha:
                    raise ValueError('Incompatible audit provenance')
                inputs[name] = sha
        audits.append(record)
    if audits[0]['comparisons'] != 4 or audits[0]['unique_states'] != 16:
        raise ValueError('Expected four oral comparisons from sixteen state replays')
    if (audits[1]['filters'], audits[1]['particles'], audits[1]['trials']) != (16, 128, 576):
        raise ValueError('Expected original fixed-parameter choice screen')
    for name, sha in inputs.items():
        if digest(Path(name)) != sha:
            raise ValueError('Changed audited input: ' + name)

    copied = {}

    def table(folder: Path, name: str, label: str, artifact: dict | None = None) -> pd.DataFrame:
        path = folder / name
        expected = artifact[name] if artifact is not None else inputs.get(str(path.resolve()))
        if expected is None or digest(path) != expected:
            raise ValueError('Unaudited table: ' + str(path))
        inputs[str(path.resolve())] = expected
        copied[label] = path
        return pd.read_csv(path)

    feedback = table(oral_validation, 'feedback_scores.csv', 'feedback_scores.csv', audits[0]['artifacts'])
    choices = table(choice_validation, 'verified_scores.csv', 'choice_scores.csv', audits[1]['artifacts'])
    choice_trials = table(results / 'online_evidence_scores_20261018_v2', 'trial_scores.csv', 'choice_trials.csv')
    blocks, summaries, checks = [], [], 0
    for proposal in ('pool', 'search'):
        for reading in ('midpoint', 'relative'):
            folder = results / f'online_evidence_oral_{proposal}_{reading}_20261018_v1'
            trials = table(folder, 'trials.csv', f'{proposal}_{reading}_trials.csv')
            summary = table(folder, 'summary.csv', f'{proposal}_{reading}_summary.csv')
            summary = summary.assign(proposal=proposal, reading=reading)
            summaries.append(summary)
            for arm in ('BASE', 'ONLINE'):
                own = trials[trials.arm == arm].copy()
                if len(own) != 576 or own.state_trial.nunique() != 576:
                    raise ValueError('Missing trial identities')
                np.testing.assert_array_equal(np.sort(own.state_trial), np.arange(576))
                complete = own[own.scope == 'complete'].copy()
                if len(complete) != 391:
                    raise ValueError('Changed primary report denominator')
                values = complete[['structural_support', 'structural_conflict', 'structural_undetermined']].to_numpy(float)
                if not np.isfinite(values).all() or np.any((values < -2e-12) | (values > 1+2e-12)):
                    raise ValueError('Invalid oral mass')
                np.testing.assert_allclose(values.sum(axis=1), 1, atol=2e-12)
                checks += len(complete)
                for window, subset in [('whole', complete), ('training', complete[complete.phase == 'training']),
                                       ('test', complete[complete.phase == 'test'])]:
                    row = summary.query('arm == @arm and window == @window and domain == "structural" and subset == "complete"').iloc[0]
                    if int(row.reports) != len(subset):
                        raise ValueError('Changed report counts')
                    for field, column in [('support', 'structural_support'), ('prior_support', 'prior_structural_support'),
                                          ('availability_all_components', 'structural_availability_all_components')]:
                        np.testing.assert_allclose(row[field], subset[column].mean(), atol=2e-12)
                        checks += 1
                # State observer indexes from zero; displayed block ends are one-based.
                complete['block'] = complete.state_trial.astype(int) // 32
                for block, group in complete.groupby('block'):
                    blocks.append(dict(proposal=proposal, reading=reading, arm=arm, block=int(block),
                        trial=(int(block)+1)*32, reports=len(group), support=group.structural_support.mean()))
    summary = pd.concat(summaries, ignore_index=True)
    block_table = pd.DataFrame(blocks)
    cumulative = []
    for proposal in ('POOL', 'SEARCH'):
        own = choice_trials[choice_trials.model == proposal].sort_values('trial')
        if len(own) != 576 or own.trial.nunique() != 576:
            raise ValueError('Missing choices')
        probabilities = own[['baseline_probability', 'online_probability']].to_numpy(float)
        if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities > 1)):
            raise ValueError('Choice logarithms require probabilities in (0, 1]')
        delta = np.log(own.baseline_probability) - np.log(own.online_probability)
        np.testing.assert_allclose(delta, own.delta_choice_nll, atol=2e-12)
        for phase in ('prefix', 'later', 'full'):
            selected = own if phase == 'full' else own[own.phase == phase]
            score = choices.query('model == @proposal and phase == @phase').iloc[0]
            np.testing.assert_allclose(score.delta_nll, selected.delta_choice_nll.mean(), atol=2e-12)
            checks += 1
        cumulative.extend(dict(proposal=proposal, trial=int(t), cumulative_delta=float(d))
                          for t, d in zip(own.trial, delta.cumsum()))
        checks += len(own)
    cumulative = pd.DataFrame(cumulative)
    contract = dict(conclusion='Online H evidence increases late oral support under both readings while early choice loss increases.',
        evidence='a/b: all complete reports across time; c: all choice costs; d/e: support and availability; f: both feedback strata.',
        archetype='quantitative grid', backend='python', width_mm=183, height_mm=150, dpi=300, formats=['PNG'],
        mapping='Time to x; structural support to upper-left y; summed ONLINE-minus-BASE NLL to upper-right y; fixed metrics to lower x.',
        participant_n=1, participant=208, filter_seeds=4, particles=128, complete_reports=391, late_complete_reports=119,
        center='Mean across complete reports after pooling joint masses over seeds and normalizing each trial.',
        uncertainty='Panel f: paired whole-filter deletion Monte Carlo SE. Other panels show means; no population intervals.',
        split='First384choices were used to estimate frozen old parameters; last192 are development extrapolation.',
        data_use='Oral reports never enter parameter estimation. Bare long/short has no prescribed reference; both readings are fixed across models.',
        exclusions='Primary oral plot uses391complete reports;185remaining reports remain in full trial tables. All576choices used.',
        transformations='Non-overlapping32-trial oral means; exact cumulative NLL; no smoothing or sampling.',
        hypothesis_tests='None', multiple_comparison_correction='Not applicable',
        adaptation='Style-only inheritance; new data mapping and panel logic.',
        limits='Fixed old parameters, one development participant; Monte Carlo SE excludes finite-particle bias and parameter uncertainty.',
        export_exceptions={'EXPORT-VECTOR':'Repository PNG-only instruction','EXPORT-RASTER':'Repository PNG-only instruction','RASTER-DPI':'300dpi discussion output'})
    output.mkdir(parents=True)
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    for name, path in copied.items():
        shutil.copy2(path, output / name)
    block_table.to_csv(output / 'plot_blocks.csv', index=False)
    summary.to_csv(output / 'plot_summary.csv', index=False)
    cumulative.to_csv(output / 'plot_cumulative_nll.csv', index=False)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':7,
        'axes.titlesize':7, 'axes.labelsize':7, 'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6, 'svg.fonttype':'none', 'pdf.fonttype':42})
    colors = {'BASE':'#68778B', 'ONLINE':'#257B89'}
    styles = {'pool':'-', 'search':'--'}
    fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 150/25.4))
    fig.subplots_adjust(left=.085, right=.975, top=.84, bottom=.22, wspace=.43, hspace=.54)
    for col, reading in enumerate(('midpoint', 'relative')):
        for proposal in ('pool', 'search'):
            for arm in ('BASE', 'ONLINE'):
                rows = block_table.query('proposal == @proposal and reading == @reading and arm == @arm')
                axes[0, col].plot(rows.trial, rows.support, color=colors[arm], ls=styles[proposal], lw=1.15)
        axes[0, col].axvline(384, color='#999999', ls=':', lw=.7)
        axes[0, col].set(title='Absolute reading' if col == 0 else 'Relative reading', xlabel='Trial',
            ylabel='Structural support mass', xlim=(0,576), ylim=(0,1), xticks=[0,192,384,576])
        ax = axes[1, col]
        fields = ['support', 'prior_support', 'availability_all_components']
        for i, field in enumerate(fields):
            for proposal, offset, marker in [('pool', -.07, 'o'), ('search', .07, 's')]:
                own = summary.query('proposal == @proposal and reading == @reading and window == "test" and domain == "structural" and subset == "complete"').set_index('arm')
                y = own.loc[['BASE', 'ONLINE'], field].to_numpy(float)
                ax.plot([i+offset-.022, i+offset+.022], y, color='#AAAAAA', lw=.8)
                for k, arm in enumerate(('BASE', 'ONLINE')):
                    ax.scatter(i+offset+(-.022 if k == 0 else .022), y[k], color=colors[arm], marker=marker, s=17, zorder=3)
        ax.set(title='Late reports: support and availability', ylabel='Mean mass / probability',
            xticks=range(3), xticklabels=['Choice\nconditioned', 'Before current\nchoice', 'Support-rule\npresence'], ylim=(0,1), xlim=(-.4,2.4))
    for proposal, color in [('POOL', '#555555'), ('SEARCH', '#B27737')]:
        own = cumulative[cumulative.proposal == proposal]
        axes[0,2].plot(own.trial, own.cumulative_delta, label=proposal, color=color, lw=1.1, ls=styles[proposal.lower()])
    axes[0,2].axhline(0, color='#AAAAAA', lw=.6)
    axes[0,2].axvline(384, color='#999999', ls=':', lw=.7)
    axes[0,2].set(title='Behavior: early cost, later gain', xlabel='Trial', ylabel='Cumulative NLL difference',
        xlim=(0,576), xticks=[0,192,384,576])
    axes[0,2].legend(frameon=False, loc='upper left')
    ax = axes[1,2]
    for j, (reading, proposal) in enumerate((('midpoint','pool'),('midpoint','search'),('relative','pool'),('relative','search'))):
        for group, offset, color, marker in [('correct', -.1, '#257B89', 'o'), ('incorrect', .1, '#B27737', 's')]:
            row = feedback.query('proposal == @proposal and reading == @reading and window == "test" and domain == "structural" and metric == "conditioned" and feedback_group == @group').iloc[0]
            ax.errorbar(j+offset, row.delta_support, yerr=row.delta_support_mcse, color=color, fmt=marker, ms=3.4, capsize=2, lw=.7)
    ax.axhline(0, color='#AAAAAA', lw=.6)
    ax.set(title='Late gain by actual feedback', ylabel='ONLINE − BASE support', xticks=range(4),
        xticklabels=['Abs.\nPOOL','Abs.\nSEARCH','Rel.\nPOOL','Rel.\nSEARCH'], ylim=(-.015,.225))
    ax.text(.03,.45,'Correct: 108 reports\nIncorrect: 11 reports', transform=ax.transAxes, va='top', fontsize=6)
    for i, ax in enumerate(axes.flat):
        ax.text(-.16,1.07,chr(97+i),transform=ax.transAxes,fontsize=8,fontweight='bold')
    fig.suptitle('Online evidence: oral representation improves; early choice errors remain costly', fontsize=8.5, y=.975)
    fig.text(.5,.925,'Participant 208 | fixed K=6 parameters | 4 paired filters × 128 particles per variant and proposal',ha='center',fontsize=6.5)
    handles = [Line2D([],[],color=colors[a],lw=1.3,label=a) for a in ('BASE','ONLINE')]
    handles += [Line2D([],[],color='#555555',ls=styles[p],marker=m,lw=.8,label=p.upper()) for p,m in [('pool','o'),('search','s')]]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5,.11))
    fig.text(.5,.078,'a–b: all 391 complete reports, 32-trial means. d–f: all 119 complete late reports. No oral selection or refitting.',ha='center',fontsize=6.2)
    fig.text(.5,.045,'f: teal circles = correct, ochre squares = incorrect; bars = paired Monte Carlo SE, often smaller than symbols.',ha='center',fontsize=6.2)
    fig.text(.5,.015,'Positive NLL difference = worse choice prediction. Presence of a supporting rule and its contribution are distinct.',ha='center',fontsize=6.2)
    fig.savefig(output / 'online_evidence_oral.png', dpi=300)
    plt.close(fig)
    caption = ('208的两种提议方案使用同一冻结参数，每组四个128粒子滤波。上排前两图汇总全部391份完整报告，'
        '每点为32题区间内完整报告的平均结构支持质量；竖线为第384题。第三图累计全部选择的负对数似然差，'
        '正值表示ONLINE预测更差。下排前两图使用后192题的119份完整报告，分别显示本题选择加权后的支持、'
        '加权前支持及严格支持规则在场概率。下排末图按实际正确／错误反馈分层，误差棒为配对滤波模拟标准误。'
        '两种口述读法在全部方案统一固定。185份其他报告保留在完整源表中，部分解析片段单独评价。'
        '口述支持表示模型作答依据对报告内容的支持，口述句子本身的生成概率尚未建模。')
    late = feedback.query('window == "test" and domain == "structural" and metric == "conditioned" and feedback_group == "all"')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>在线证据的口述与行为对照</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}td,th{padding:6px}</style>'
        '<h1>规则已经在场时，反馈证据能否改善实际使用</h1><p>'+caption+'</p>'
        '<img src="online_evidence_oral.png" alt="双读法口述支持、选择代价、在场和使用的比较">'
        +late[['proposal','reading','complete_reports','baseline_support','online_support','delta_support','delta_support_mcse']].to_html(index=False,float_format=lambda x:f'{x:.6f}')
        +'<p><a href="feedback_scores.csv">全部反馈分层</a> · <a href="choice_scores.csv">全部选择评分</a> · '
        '<a href="plot_summary.csv">两种读法的完整评分</a> · <a href="figure_contract.json">图示范围与误差定义</a></p></html>')
    (output / 'index.html').write_text(page)
    for name, sha in inputs.items():
        if digest(Path(name)) != sha:
            raise ValueError('Scientific input changed during plotting')
    shutil.copy2(__file__, output / 'plot_source.py')
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', numeric_checks=checks,
        input_sha256=inputs, source_sha256={str(Path(__file__).resolve()):digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results','oral-validation','choice-validation','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    plot(args.results, args.oral_validation, args.choice_validation, args.output)
