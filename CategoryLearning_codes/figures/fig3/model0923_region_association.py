"""Show conditional gains and complete autonomous distributions together.

Figure contract: independent association learning improves later predictions
and generated learning in two development participants, with remaining late
accuracy gaps. Quantitative grid: two small NLL panels above four trajectory
panels. Every one of512 paths appears, aggregated in declared32-trial blocks.
Python only;183 x185mm;600dpi PNG only under the repository export policy.
NLL has no Monte Carlo error here. Faint lines are independent paths, bold
colored lines are their equal-weight means, black is the observed sequence.
Training candidates per subject:20 fixed versus60 learned. These selected
parameters are a fixed-catalogue diagnostic; H search/capacity are absent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import load,digest,block_paths
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write
from src.Bayesian_state.workflows.analysis.model_0923_region_association_diagnostic import ROOT,verify


def plot(conditional: Path, autonomous: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new figure directory')
    cd=load(conditional/'completion.json');ad=load(autonomous/'completion.json')
    ap=load(autonomous/'protocol.json');cp=load(conditional/'protocol.json')
    if cd['status']!='complete' or ad['status']!='complete' or cd['smoke'] or ad['smoke']:
        raise ValueError('Require complete scientific comparisons')
    assert digest(autonomous/'protocol.json')==ad['protocol_sha256']
    assert ap['parent_nomination_sha256']==cd['nomination_sha256']
    assert ap['paths_per_group']==128 and ap['settings']['block_width']==32
    bindings={str((conditional/n).resolve()):cd['artifacts'][n] for n in ['summary.csv']}
    bindings.update(ad['artifacts']);verify(bindings)
    summary=pd.read_csv(conditional/'summary.csv')
    assert len(summary)==4 and set(summary.subject)=={207,208} and set(summary.arm)=={'FIXED','ROW'}
    nll=[];path_rows=[];banks={};metadata={}
    for s in [207,208]:
        folder=ROOT/cp['settings']['data']/f'subject_{s}';meta=load(folder/'prepared.json');metadata[s]=meta
        observed=[];labels=[]
        for name in ['training.npz','sealed_test.npz']:
            bindings[str(folder/name)]=meta['array_sha256'][name]
            with np.load(folder/name) as z:observed.append(z['choices']);labels.append(z['categories'])
        labels=np.concatenate(labels);observed=np.concatenate(observed)==labels
        human=block_paths(observed[None],0,len(labels),32)[0]
        for arm in ['FIXED','ROW']:
            predictions=[];seeds=[]
            for file in sorted((autonomous/'paths').glob(f'S{s}_{arm}_C*.npz')):
                with np.load(file) as z:
                    np.testing.assert_array_equal(z['categories'],labels)
                    predictions.append(z['choices']==labels[None]);seeds.extend(map(int,z['seeds']))
            assert seeds==ap['seed_banks'][str(s)]
            values=block_paths(np.concatenate(predictions),0,len(labels),32)
            assert values.shape==(128,len(human))
            banks[s,arm]=(values,human)
            for i,seed in enumerate(seeds):
                for b in range(len(human)):
                    path_rows.append(dict(subject=s,arm=arm,path=i,seed=seed,block=b+1,
                        first_trial=32*b+1,last_trial=min(32*(b+1),len(labels)),accuracy=values[i,b],observed=human[b]))
            row=summary[(summary.subject==s)&(summary.arm==arm)].iloc[0]
            for phase in ['training','test']:nll.append(dict(subject=s,arm=arm,phase=phase,nll=row[phase+'_nll']))
    verify(bindings);output.mkdir(parents=True)
    pd.DataFrame(nll).to_csv(output/'nll_source.csv',index=False)
    pd.DataFrame(path_rows).to_csv(output/'paths_source.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6,'ytick.labelsize':6,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    width_mm,height_mm=183,185
    fig,axes=plt.subplots(3,2,figsize=(width_mm/25.4,height_mm/25.4),
        gridspec_kw={'height_ratios':[.8,1.15,1.15]})
    fig.subplots_adjust(left=.095,right=.98,bottom=.085,top=.91,hspace=.43,wspace=.29)
    colors={'FIXED':'#858F99','ROW':'#31688E'}
    for col,s in enumerate([207,208]):
        ax=axes[0,col];ax.set_title(f'Subject {s}',fontweight='bold',pad=9)
        for i,phase in enumerate(['training','test']):
            points=[next(r['nll'] for r in nll if r['subject']==s and r['arm']==arm and r['phase']==phase) for arm in ['FIXED','ROW']]
            ax.plot([i-.10,i+.10],points,color='#B9B9B9',lw=.8)
            for offset,arm,value in zip([-.10,.10],['FIXED','ROW'],points):
                ax.scatter(i+offset,value,color=colors[arm],s=18,zorder=3)
                ax.annotate(f'{value:.3f}',(i+offset,value),xytext=(-7 if offset<0 else 7,4),textcoords='offset points',
                    ha='right' if offset<0 else 'left',fontsize=6)
        ax.set_xticks([0,1],['Training','Later']);ax.set_xlim(-.5,1.5);ax.set_ylim(.25,1.15)
        ax.set_ylabel('Mean choice NLL')
        for row,arm in enumerate(['FIXED','ROW'],1):
            ax=axes[row,col];values,human=banks[s,arm];n=metadata[s]['trials'];cut=metadata[s]['training_trials']
            centers=np.array([(32*b+1+min(32*(b+1),n))/2 for b in range(len(human))])
            ax.plot(centers,values.T,color=colors[arm],alpha=.075,lw=.4)
            ax.plot(centers,values.mean(0),color=colors[arm],lw=1.5)
            ax.plot(centers,human,color='black',lw=1.25)
            ax.axvline(cut+.5,color='#777777',lw=.7,ls='--')
            ax.set_ylim(0,1.03);ax.set_xlim(0,n);ax.set_yticks([0,.25,.5,.75,1])
            ax.set_xticks(np.linspace(0,n,4,dtype=int));ax.set_ylabel('Block accuracy')
            ax.set_title('Fixed category labels' if arm=='FIXED' else 'Learned independent associations',loc='left',pad=5)
            if row==2:ax.set_xlabel('Trial')
    for letter,ax in zip('abcdef',axes.flat):ax.text(-.15,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=8,va='bottom')
    fig.legend(handles=[Line2D([0],[0],color=colors['FIXED'],lw=1.5,label='Fixed labels'),
        Line2D([0],[0],color=colors['ROW'],lw=1.5,label='Learned associations'),
        Line2D([0],[0],color='black',lw=1.25,label='Observed')],loc='upper center',bbox_to_anchor=(.54,.985),ncol=3,fontsize=7)
    fig.text(.5,.018,'128 equal-weight paths per arm; faint lines show every path; dashed line marks the training boundary.',ha='center',fontsize=6)
    fig.savefig(output/'region_association.png',dpi=600,facecolor='white');plt.close(fig)
    caption='''<p>上排：每题选择负对数似然（NLL，越低越好），参数只按训练段选择。207保持672／288题，208保持384／192题。
    每人固定类别组20个候选，独立关联组60个候选。后段在冻结参数下继续接收被试历史选择和反馈。</p>
    <p>中、下排：各组全部128条自主路径，从第一题开始根据自身选择获得任务反馈；每32题计算正确率，全部试次保留。
    浅线为单条路径，粗彩线为等权平均，黑线为被试。虚线为训练与后段边界。路径分布描述模拟随机性。
    该对照中152个几何结构始终可用，β固定；H搜索、容量和精度学习另行接入。两名被试均属于开发样本。</p>'''
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><h1>独立类别关联的行为检验</h1>'
        '<img src="region_association.png" style="max-width:100%;width:850px">'+caption+summary.to_html(index=False))
    verify(bindings)
    write(output/'validation.json',dict(status='complete',backend='python',dimensions_mm=[183,185],dpi=600,
        paths=512,path_block_rows=len(path_rows),nll_points=len(nll),excluded_paths=0,excluded_trials=0,
        inputs_sha256=bindings,plot_source_sha256=digest(Path(__file__)),
        export_policy='PNG only per repository instruction; editable plotting source and numeric CSV retained',
        artifacts={p.name:digest(p) for p in [output/'region_association.png',output/'nll_source.csv',output/'paths_source.csv',output/'index.html']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditional',type=Path,required=True);parser.add_argument('--autonomous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plot(args.conditional,args.autonomous,args.output)
