"""Plot audited fixed-parameter ONLINE predicate coverage; PNG discussion draft.

Run with --output NEW_DIRECTORY from the repository root. Both fixed readings,
all complete reports and all four development participants are retained. The
two proposal arms for participant 208 are sensitivity views of the same person.
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CASES = [(107,'SEARCH'),(207,'SEARCH'),(208,'POOL'),(208,'SEARCH'),(306,'SEARCH')]


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new predicate figure directory')
    inputs, summaries, differences, identities = {}, [], [], []
    checks = 0

    def bind(path: Path, expected: str | None = None) -> None:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError('Changed audited figure input')
        inputs[str(path.resolve())] = sha

    for subject, proposal in CASES:
        for reading in ('midpoint','relative'):
            middle = proposal.lower()+'_' if subject == 208 else ''
            name = f'predicate_online_{subject}_{middle}{reading}'
            version = 2 if (subject,reading) == (107,'midpoint') else 1
            folder = ROOT/'results/model_0923'/f'{name}_20261019_v{version}'
            audit = folder.with_name(name+'_audit_20261019_v1')/'validation.json'
            bind(audit); validation = json.loads(audit.read_text())
            if validation['status'] != 'complete':
                raise ValueError('Independent raw-state validation required')
            bind(folder/'manifest.json',validation['input_sha256'][str((folder/'manifest.json').resolve())])
            manifest = json.loads((folder/'manifest.json').read_text())
            for file in ('summary.csv','contrasts.csv','trials.csv'):
                bind(folder/file,manifest['artifacts'][file])
            summary = pd.read_csv(folder/'summary.csv'); contrast = pd.read_csv(folder/'contrasts.csv')
            trials = pd.read_csv(folder/'trials.csv')
            for row in summary.itertuples():
                if row.metric != 'conditioned':
                    continue
                group = trials[(trials.arm == row.arm)&trials.scope.eq('complete')]
                n = trials[trials.arm == row.arm].shape[0];cut = int(group.training_trials.iloc[0])
                if row.window == 'training':group = group[group.state_trial < cut]
                elif row.window == 'test':group = group[group.state_trial >= cut]
                elif row.window == 'first64':group = group[group.state_trial < 64]
                elif row.window == 'last64':group = group[group.state_trial >= max(0,n-64)]
                if len(group) != row.reports:
                    raise ValueError('Figure denominator changed')
                for label in ('support','conflict','undetermined'):
                    np.testing.assert_allclose(group['predicate_'+row.domain+'_'+label].mean(),getattr(row,label),atol=1e-12,rtol=0)
                    checks += 1
            for row in contrast.itertuples():
                same = summary[(summary.window == row.window)&(summary.domain == row.domain)&(summary.metric == row.metric)].set_index('arm')
                for label in ('support','conflict','undetermined'):
                    np.testing.assert_allclose(same.loc['ONLINE',label]-same.loc['BASE',label],getattr(row,label+'_difference'),atol=1e-12,rtol=0)
                    checks += 1
            summaries.append(summary.assign(reading=reading,proposal=proposal))
            differences.append(contrast.assign(reading=reading,proposal=proposal))
            identities.append(dict(subject=subject,proposal=proposal,reading=reading,reports=int((trials.arm.eq('BASE')&trials.scope.eq('complete')).sum()),
                filter_seeds=int(summary.filter_seeds.iloc[0]),parameter_scope=manifest['parameter_estimation_scope']))
    summary = pd.concat(summaries,ignore_index=True); difference = pd.concat(differences,ignore_index=True)
    output.mkdir(parents=True)
    summary.to_csv(output/'summary_source.csv',index=False);difference.to_csv(output/'difference_source.csv',index=False)
    pd.DataFrame(identities).to_csv(output/'analysis_identity.csv',index=False)
    contract = dict(conclusion='ONLINE improves later predicate coverage in 207, 208 and 306, while early tradeoffs and unexplained content remain.',
        archetype='quantitative grid',backend='python',dimensions_mm=[183,180],export='PNG only; repository discussion draft, 300 dpi',
        panels='Top: later support/conflict/undetermined shares. Bottom: early and later paired support changes.',
        sample='Four development participants; 208 shown under both proposal arms; all complete reports per reading retained.',
        center='Predicates averaged within report, then reports equally; behavior-conditioned state mass pooled before normalization.',
        uncertainty='Bottom error bars: paired delete-one-whole-filter MCSE; four seeds and 128 particles; no participant confidence interval or p value.',
        data_use='Existing behavior-only parameters fixed; no oral fitting; original chronological boundaries; joint-report scores remain separately available.',
        limitations='Predicate coverage is partial content evidence; unknown content retained; readings are common sensitivity conventions; development only.')
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig, axes = plt.subplots(2,2,figsize=(7.204724,7.086614),gridspec_kw={'height_ratios':[1.5,1]})
    fig.subplots_adjust(left=.16,right=.98,bottom=.09,top=.91,hspace=.46,wspace=.47)
    colors = ['#5F958F','#C98379','#DBDFE3']
    for col, reading in enumerate(('midpoint','relative')):
        ax = axes[0,col]; labels=[];positions=[]
        for i,(subject,proposal) in enumerate(CASES):
            selected = summary[(summary.subject == subject)&(summary.proposal == proposal)&(summary.reading == reading)&
                summary.window.eq('test')&summary.domain.eq('structural')&summary.metric.eq('conditioned')].set_index('arm')
            for offset,(arm,short) in zip((-.32,.32),(('BASE','B'),('ONLINE','O'))):
                y=i*1.8+offset;left=0.
                for k,label in enumerate(('support','conflict','undetermined')):
                    value=selected.loc[arm,label]
                    ax.barh(y,value,left=left,height=.53,color=colors[k],edgecolor='none',label=label.capitalize() if i==0 and arm=='BASE' else None)
                    left+=value
                np.testing.assert_allclose(left,1.,atol=1e-12,rtol=0);checks+=1
                positions.append(y);labels.append(f'{subject} {proposal} · {short}')
        ax.set_yticks(positions,labels,fontsize=6);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.set_xticks([0,.5,1]);ax.set_xlabel('Later reported content (fraction)')
        ax.set_title(('Absolute reading','Relative reading')[col],fontsize=8,pad=10)
        ax.tick_params(axis='y',length=0)
        ax.text(-.35,1.08,'ab'[col],transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax = axes[1,col]
        for i,(subject,proposal) in enumerate(CASES):
            selected = difference[(difference.subject == subject)&(difference.proposal == proposal)&(difference.reading == reading)&
                difference.domain.eq('structural')&difference.metric.eq('conditioned')].set_index('window')
            for offset,window,marker,color,label in [(-.16,'training','o','#7C8289','Earlier'),(.16,'test','s','#407B94','Later')]:
                row=selected.loc[window]
                ax.errorbar(row.support_difference,i+offset,xerr=row.support_difference_mcse,fmt=marker,color=color,
                    markersize=3.5,capsize=2,elinewidth=.7,label=label if i==0 else None)
        ax.axvline(0,color='#9BA1A6',lw=.7,zorder=0)
        ax.set_yticks(range(len(CASES)),[f'{s} {p}' for s,p in CASES],fontsize=6.5);ax.invert_yaxis()
        ax.set_xlim(-.03,.26);ax.set_xticks([0,.1,.2]);ax.set_xlabel('Support change: ONLINE − BASE')
        ax.text(-.35,1.08,'cd'[col],transform=ax.transAxes,fontweight='bold',fontsize=9)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.56,.99),ncol=3,fontsize=7)
    axes[1,1].legend(loc='lower right',fontsize=6.5)
    fig.text(.16,.44,'B: BASE   O: ONLINE   |   Later reports: 35, 264, 119, 119, 255 (top to bottom)',fontsize=6)
    fig.savefig(output/'predicate_coverage.png',dpi=300,facecolor='white');plt.close(fig)
    page=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐判断口述对应</title>'
        '<style>body{max-width:1150px;margin:30px auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>'
        '<h1>逐判断支持、冲突和未确定内容</h1><p>上排为后段内容质量，三部分合计为1；下排为同口径ONLINE减BASE的支持差。'
        '两种读法保持共同约定，全部完整报告保留。前段与后段均来自开发数据；208两行对应同一人的两种提议方案。'
        '从上至下，后段完整报告数为35、264、119、119、255，前段为40、362、272、272、381，两种读法相同。'
        '误差线是4次完整滤波删除得到的模拟标准误，每次128粒子。行为参数保持原值，口述未用于估参。</p>'
        '<img src="predicate_coverage.png"><p>逐判断覆盖描述模型能解释的部分内容，整句支持继续单独保留。'
        '107后段的三分之一支持同时受二分类目录限制；306的支持与少量冲突都上升。'
        '完整功能支持、去掉本题选择加权的对照、前段冻结基准和所有窗口见源表。</p>'
        '<p><a href="summary_source.csv">全部评分源表</a> · <a href="difference_source.csv">全部配对差值</a> · '
        '<a href="analysis_identity.csv">样本与滤波数量</a></p></html>')
    (output/'index.html').write_text(page);shutil.copy2(__file__,output/'figure_source.py')
    for path,sha in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise ValueError('Figure input changed during rendering')
    (output/'validation.json').write_text(json.dumps(dict(status='complete',checks=checks,input_sha256=inputs,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qa_exceptions=['PNG-only repository convention','300 dpi discussion draft; no submission claim']),indent=2)+'\n')
    print(f'Created predicate figure with {checks} source checks')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
