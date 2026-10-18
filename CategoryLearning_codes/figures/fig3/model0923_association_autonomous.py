"""Show every autonomous path from the frozen association diagnostic.

Run --source AUTONOMOUS_RESULTS --output NEW_DIRECTORY. This quantitative grid
compares four controlled association variants in three development participants.
All paths remain visible with equal weight. Plotting rechecks raw probabilities,
seeds, candidate identity and all block-distribution scores. Discussion export:
183 by 180 mm, PNG only at 300 dpi, following the repository format instruction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.Bayesian_state.workflows.analysis import model_0923_association_autonomous as analysis
from src.Bayesian_state.evaluation.model_0923_h_validation import block_paths, jackknife_se
from src.Bayesian_state.metrics.task import category_learning_feedback

ARMS = ('FIXED_SHARED','LEARN_SHARED','FIXED_STABLE','LEARN_STABLE')
LABELS = ('Fixed labels\nShared retention','Learn labels\nShared retention',
          'Fixed labels\nStable retention','Learn labels\nStable retention')
COLORS = {'FIXED_SHARED':'#758399','FIXED_STABLE':'#758399',
          'LEARN_SHARED':'#B18358','LEARN_STABLE':'#277F8E'}
read, digest = analysis.read, analysis.digest


def collect(source: Path) -> tuple[dict, dict, dict]:
    done, pro = read(source/'completion.json'), read(source/'protocol.json')
    if done['status']!='complete' or done['smoke'] or digest(source/'protocol.json')!=done['protocol_sha256']:
        raise ValueError('Need a complete scientific autonomous diagnostic.')
    inputs = {str(source/name):digest(source/name) for name in ('completion.json','protocol.json')}
    for name, sha in done['artifacts'].items():
        if digest(source/name)!=sha: raise ValueError('Summary artifact changed.')
        inputs[str(source/name)] = sha
    analysis.verify(done['input_sha256']); inputs.update(done['input_sha256'])
    nominees = {(n['subject'],n['arm_name']):n for n in pro['nominees']}
    subjects = sorted({s for s,_ in nominees})
    if len(subjects)!=3 or set(nominees)!={(s,a) for s in subjects for a in ARMS}:
        raise ValueError('Need all three participants and all four diagnostic arms.')
    source_fit = analysis.ROOT/pro['settings']['source']; fitted = read(source_fit/'protocol.json')
    if digest(source_fit/'nominees.json')!=pro['parent_nomination_sha256']:
        raise ValueError('Parent nominees changed.')
    data = analysis.ROOT/fitted['settings']['data']
    groups = defaultdict(list)
    for job_file in sorted((source/'jobs').glob('*.json')):
        j = read(job_file)
        if j['phase']!='autonomous': continue
        key = j['subject'],j['arm_name']; nominee = nominees[key]
        file = Path(j['target']).with_suffix('.npz'); r = read(file.with_suffix('.json'))
        if (r['job_sha256']!=digest(job_file) or r['arrays_sha256']!=digest(file)
                or j['beta']!=nominee['beta'] or j['gamma']!=nominee['gamma']
                or j['arm']!=fitted['settings']['arms'][j['arm_name']]
                or j['sources']!=pro['source_sha256']):
            raise ValueError('Path differs from the frozen parameter and mechanism identity.')
        with np.load(file) as z: arrays={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(arrays['seeds'],j['seeds'])
        groups[key].append((j,arrays))
        for f in (job_file,file.with_suffix('.json'),file): inputs[str(f)]=digest(f)
    if set(groups)!=set(nominees): raise ValueError('Missing autonomous groups.')
    scores=pd.read_csv(source/'scores.csv'); contrasts=pd.read_csv(source/'contrasts.csv')
    curves=pd.read_csv(source/'curves.csv'); path_rows=[]; path_summaries=[]; bank={}; cuts={}
    for (s,arm), chunks in sorted(groups.items()):
        meta=read(data/f'subject_{s}/prepared.json'); cut=meta['training_trials']; cuts[s]=cut
        human_parts=[]
        for name in ('training.npz','sealed_test.npz'):
            file=data/f'subject_{s}'/name
            if digest(file)!=meta['array_sha256'][name]: raise ValueError('Prepared data changed.')
            with np.load(file) as z: human_parts.append({k:z[k].copy() for k in ('choices','categories')})
        categories=np.concatenate([p['categories'] for p in human_parts]); n=len(categories)
        human=np.concatenate([p['choices'] for p in human_parts])==categories
        seeds=np.concatenate([a['seeds'] for _,a in chunks]); order=np.argsort(seeds)
        seeds=seeds[order]
        if len(seeds)!=pro['paths_per_group'] or len(set(seeds))!=len(seeds):
            raise ValueError('Missing or duplicated path seeds.')
        np.testing.assert_array_equal(seeds,sorted(pro['seed_banks'][str(s)]))
        x=np.concatenate([a['choices'] for _,a in chunks])[order]
        probability=np.concatenate([a['probabilities'] for _,a in chunks])[order]
        feedback=np.concatenate([a['feedback'] for _,a in chunks])[order]
        if (x.shape!=(len(seeds),n) or probability.shape!=(len(seeds),n,4)
                or not np.isin(x,range(1,5)).all() or not np.isfinite(probability).all() or np.any(probability<0)):
            raise ValueError('Invalid path arrays.')
        np.testing.assert_allclose(probability.sum(axis=-1),1.,atol=1e-12,rtol=0)
        for j,a in chunks:
            np.testing.assert_array_equal(a['categories'],categories)
            if j['condition']!=meta['condition']: raise ValueError('Feedback condition differs.')
        expected=np.array([[category_learning_feedback(y,c,condition=meta['condition'])
                            for y,c in zip(row,categories)] for row in x])
        np.testing.assert_array_equal(feedback,expected)
        correct=x==categories[None,:]; width=pro['settings']['block_width']
        for phase,first,last in [('whole',0,n),('training',0,cut),('test',cut,n)]:
            samples=block_paths(correct,first,last,width); target=block_paths(human[None,:],first,last,width)[0]
            metrics,jack=analysis.block_scores(samples,target); bank[s,arm,phase]=(metrics,jack)
            saved=scores.loc[scores.subject.eq(s)&scores.arm_name.eq(arm)&scores.phase.eq(phase)]
            if len(saved)!=1: raise ValueError('Missing or duplicated score row.')
            for key,value in metrics.items(): np.testing.assert_allclose(saved.iloc[0][key],value,rtol=0,atol=1e-12)
        samples=block_paths(correct,0,n,width); target=block_paths(human[None,:],0,n,width)[0]
        frame=curves.loc[curves.subject.eq(s)&curves.arm_name.eq(arm)].sort_values('block')
        np.testing.assert_array_equal(frame.block,np.arange(1,samples.shape[1]+1))
        np.testing.assert_allclose(frame['mean'],samples.mean(axis=0),rtol=0,atol=1e-14)
        np.testing.assert_allclose(frame.observed,target,rtol=0,atol=1e-14)
        for i,seed in enumerate(seeds):
            path_summaries.append(dict(subject=s,arm_name=arm,seed=int(seed),late_accuracy=float(correct[i,-128:].mean())))
            path_rows.extend(dict(subject=s,arm_name=arm,seed=int(seed),block=b+1,
                trial=min((b+1)*width,n),accuracy=float(samples[i,b])) for b in range(samples.shape[1]))
    for row in contrasts.itertuples():
        a,j=bank[row.subject,row.arm_name,row.phase];b,k=bank[row.subject,row.reference,row.phase]
        for metric in ('energy_u','energy','crps'):
            np.testing.assert_allclose(getattr(row,'delta_'+metric),a[metric]-b[metric],rtol=0,atol=1e-12)
            np.testing.assert_allclose(getattr(row,'delta_'+metric+'_mc_se'),jackknife_se(j[metric]-k[metric]),rtol=0,atol=1e-12)
    return dict(scores=scores,contrasts=contrasts,curves=curves,path_blocks=pd.DataFrame(path_rows),
                paths=pd.DataFrame(path_summaries)), inputs, dict(subjects=subjects,cuts=cuts,nominees=pro['nominees'],
                paths_per_group=pro['paths_per_group'],block_width=pro['settings']['block_width'])


def plot(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new association figure directory.')
    with threadpool_limits(limits=1): tables,inputs,details=collect(source)
    output.mkdir(parents=True)
    for name,table in tables.items(): table.to_csv(output/f'{name}.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    fig,axes=plt.subplots(3,4,figsize=(183/25.4,180/25.4),squeeze=False)
    nominees={(n['subject'],n['arm_name']):n for n in details['nominees']}
    score=tables['scores'].set_index(['subject','arm_name','phase'])
    for i,s in enumerate(details['subjects']):
        for j,arm in enumerate(ARMS):
            ax=axes[i,j]; points=tables['path_blocks'].query('subject==@s and arm_name==@arm')
            matrix=points.pivot(index='seed',columns='trial',values='accuracy')
            curve=tables['curves'].query('subject==@s and arm_name==@arm').sort_values('block')
            x=matrix.columns.to_numpy(); n=nominees[s,arm]; record=score.loc[(s,arm,'whole')]
            ax.plot(x,matrix.to_numpy().T,color='#777777',alpha=.20,linewidth=.4,zorder=1)
            ax.axvline(details['cuts'][s],color='#A57C59',linestyle='--',linewidth=.7,zorder=2)
            ax.plot(x,curve['mean'],color=COLORS[arm],linewidth=1.7,zorder=3)
            ax.plot(x,curve.observed,color='#161616',marker='o',markersize=1.7,markerfacecolor='white',
                    markeredgewidth=.5,linewidth=.9,zorder=4)
            ax.set(title=f"S{s} | β={n['beta']:g}, γ={n['gamma']:g}",xlim=(0,x.max()),ylim=(0,1.02),yticks=[0,.5,1])
            ax.text(.04,.04,f'U Energy {record.energy_u:.3f}',transform=ax.transAxes,fontsize=6,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
            ax.text(-.12,1.07,chr(97+i*4+j),transform=ax.transAxes,fontsize=8,fontweight='bold',clip_on=False)
            ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2.5,width=.6)
            if j==0: ax.set_ylabel('Block accuracy')
            if i==2: ax.set_xlabel('Trial')
    fig.suptitle('Autonomous learning with fixed or learned category associations',y=.985,fontsize=9)
    fig.text(.5,.949,f"All {details['paths_per_group']} equal-weight paths per group; training-selected parameters",ha='center',fontsize=7)
    for j,label in enumerate(LABELS): fig.text(.17+j*.223,.907,label,ha='center',va='center',fontsize=7,fontweight='bold')
    handles=[Line2D([],[],color='#161616',marker='o',markersize=3,markerfacecolor='white',label='Participant'),
             Line2D([],[],color='#277F8E',linewidth=1.7,label='Generated mean'),
             Line2D([],[],color='#777777',linewidth=.5,label='Individual path'),
             Line2D([],[],color='#A57C59',linestyle='--',label='Fit boundary')]
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,.012),fontsize=6)
    fig.subplots_adjust(left=.075,right=.985,bottom=.105,top=.845,hspace=.38,wspace=.27)
    fig.savefig(output/'autonomous_paths.png',dpi=300);plt.close(fig)
    caption=('每格显示全部32条自主路径，黑线为被试，彩线为模型均值。虚线标出训练段边界；'
             '参数由边界前的选择数据确定，每条路径从第一题开始根据自身回答接收反馈。'
             '曲线为互不重叠的32题块，路径等权。保持设置Shared取η=γ，Stable取η=1。'
             'U Energy比较块正确率分布，排除路径自身配对项，越低越好。'
             '全部几何规则始终可用，H搜索、容量和知觉噪声尚待接入；三人均为开发样本。')
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}img{width:100%}</style>'
        '<h1>类别关联方案的完整自主路径</h1><p>'+caption+'</p><img src="autonomous_paths.png">'
        +tables['scores'].query("phase=='whole'")[["subject","arm_name","energy_u","late_accuracy","observed_late"]].to_html(index=False)
        +'<p><a href="path_blocks.csv">全部路径</a> · <a href="contrasts.csv">配对差值与模拟误差</a></p></html>')
    shutil.copy2(__file__,output/'figure_source.py')
    manifest=dict(source=str(source.resolve()),source_sha256=digest(Path(__file__)),inputs=inputs,details=details,
        claim='Retained associations improve selected development cases; late autonomous accuracy remains below participants.',
        archetype='quantitative grid',adaptation='style-only; new four-arm loader and U-statistic scoring',
        paths_retained=len(tables['paths']),paths_excluded=0,backend='python',export='PNG only, 300dpi discussion figure',
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.source,args.output)
