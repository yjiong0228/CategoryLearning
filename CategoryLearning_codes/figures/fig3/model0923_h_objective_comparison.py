"""Plot all paths and shared scores for the frozen NLL/GEN objective comparison.

Run from the repository root. Modes primary, recovery and task accept their
completed analysis folders and a fresh --output folder. PNG-only scientific
review figures retain every participant and path; quantitative sources and the
figure contract are saved beside each export. Recovery is labelled synthetic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import load, digest, checked_arrays, block_paths

COLORS = {'NLL': '#0072B2', 'GEN': '#D55E00'}
MARKERS = {'NLL': 'o', 'GEN': 's'}
ARMS = ('NLL', 'GEN')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'legend.fontsize': 6, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
    'svg.fonttype': 'none', 'pdf.fonttype': 42})


def destination(output: Path, mode: str, inputs: dict) -> dict:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    output.mkdir(parents=True)
    contract = dict(mode=mode, archetype='quantitative grid', backend='python', width_mm=183,
        conclusion='Compare fitting objectives through held-out distributions and joint recovery under one fixed H mechanism',
        evidence={'primary': 'all paths and four common held-out scores',
            'recovery': 'all 14 jointly estimated coordinates, 6 generating points with 2 independent sequences each',
            'task': 'full first-passage distributions, including all administrative censoring'},
        sources=inputs, source_sha256=digest(Path(__file__)), reuse_level='build anew',
        exclusions='No display-driven participant or path selection; task exclusions follow the audited task cohort',
        uncertainty='predictive 10th to 90th percentiles for trajectories; Monte Carlo SE for scores; no population inference',
        tests='No significance tests or p values; fixed-parameter predictive comparison',
        export='PNG only, 300 dpi scientific review, per repository instruction; editable source retained',
        static_export_exceptions=['vector and TIFF omitted under repository PNG-only instruction',
            '300 dpi review export; submission export requirements have not been claimed'])
    if mode=='task':
        contract['policy_scope']='CDF shows all three policies with the same 20-block horizon. Fixed-schedule paths have participant-specific censoring horizons and remain in the full task summary table.'
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2))
    shutil.copy2(__file__,output/'plot_source.py')
    return contract


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def panel(ax, letter: str) -> None:
    ax.text(-.12,1.045,letter,transform=ax.transAxes,fontweight='bold',fontsize=8)


def trajectory(ax, x, paths, observed, conditional, arm, cut=None) -> tuple[np.ndarray, np.ndarray]:
    if paths.ndim!=2 or paths.shape[1]!=len(x) or not np.isfinite(paths).all():
        raise ValueError('Invalid full path bank')
    low,high=np.quantile(paths,[.1,.9],axis=0)
    ax.fill_between(x,low,high,color=COLORS[arm],alpha=.15,linewidth=0)
    ax.plot(x,paths.T,color=COLORS[arm],alpha=.08,lw=.35)
    ax.plot(x,paths.mean(axis=0),color=COLORS[arm],lw=1.4)
    ax.plot(x,conditional,color='#8064A2',lw=1,ls='--')
    ax.plot(x,observed,color='#222222',lw=1.2,marker='.',ms=3)
    if cut is not None:ax.axvline(cut+.5,color='#777777',ls=':',lw=.8)
    ax.set(ylim=(-.03,1.03),xlabel='Trial (32-trial block centres)',ylabel='Accuracy',yticks=[0,.25,.5,.75,1])
    ax.grid(axis='y',lw=.35,color='#E5E5E5')
    return low,high


def primary(summary: Path, features: Path, forecast: Path, prefix_scores: Path,
            uncertainty: Path, output: Path) -> None:
    meta=load(summary/'summary_manifest.json');completed=load(forecast/'forecast.json')
    assert meta['status']==completed['status']=='complete' and not completed['smoke']
    assert meta['models']==list(ARMS)
    for folder,file in [(features,'completion.json'),(prefix_scores,'comparison_manifest.json'),(uncertainty,'completion.json')]:
        manifest=load(folder/file)
        assert manifest['nomination_sha256']==meta['nomination_sha256']
    destination(output,'primary',{k:str(v.resolve()) for k,v in dict(summary=summary,features=features,
        forecast=forecast,prefix_scores=prefix_scores,uncertainty=uncertainty).items()})
    conditional,autonomous=Path(meta['conditional']),Path(meta['autonomous'])
    crows={(r['subject'],r['model']):r for r in load(conditional/'evaluation.json')['rows']}
    arows={(r['subject'],r['model']):r for r in load(autonomous/'evaluation.json')['rows']}
    frows={(r['subject'],r['model']):r for r in completed['rows']}
    source_rows=[];summary_rows=[];path_counts=[];hashes={}
    for subject in meta['subjects']:
        fig,axes=plt.subplots(2,2,figsize=(183/25.4,142/25.4))
        fig.subplots_adjust(left=.09,right=.98,bottom=.21,top=.9,hspace=.38,wspace=.27)
        for column,arm in enumerate(ARMS):
            key=subject,arm;c=checked_arrays(conditional,crows[key]);a=checked_arrays(autonomous,arows[key]);row=frows[key]
            file=Path(row['target']).with_suffix('.npz');assert digest(file)==row['predictions_sha256'];hashes[str(file)]=digest(file)
            with np.load(file) as saved:f={k:saved[k] for k in saved.files}
            end=len(c['choices']);cut=crows[key]['training_trials']
            np.testing.assert_array_equal(f['categories'],c['categories'][cut:])
            for line,start in enumerate(('task_start','filtered_prefix')):
                first=0 if line==0 else cut
                observed=block_paths((c['choices']==c['categories'])[None,:],first,end)[0]
                conditional_curve=block_paths(c['correct_probability'],first,end).mean(axis=0)
                if line==0:
                    bank=a;path_ids=a['seeds'];trials_first=0
                else:
                    bank=f;path_ids=f['branch_seeds'];trials_first=0
                paths=block_paths((bank['choices']==bank['categories'][None,:]).astype(float),trials_first,end-first)
                x=np.array([(t+1+min(t+32,end))/2 for t in range(first,end,32)])
                low,high=trajectory(axes[line,column],x,paths,observed,conditional_curve,arm,cut if line==0 else None)
                label='From task start' if line==0 else 'After observed training prefix'
                axes[line,column].set_title(f'{arm} · {label}',loc='left');panel(axes[line,column],'abcd'[line*2+column])
                path_counts.append(dict(subject=subject,model=arm,start=start,paths=len(paths),blocks=len(x),
                    independent_units=len(paths) if line==0 else len(f['filter_seed'])))
                for j,t in enumerate(x):
                    summary_rows.append(dict(subject=subject,model=arm,start=start,trial=t,observed=observed[j],
                        conditional_mean=conditional_curve[j],generated_mean=paths[:,j].mean(),q10=low[j],q90=high[j]))
                    source_rows.extend(dict(subject=subject,model=arm,start=start,path_seed=int(seed),trial=t,accuracy=paths[i,j])
                                       for i,seed in enumerate(path_ids))
        fig.suptitle(f'Participant {subject} · all generated paths',fontsize=10,y=.97)
        fig.legend(handles=[Line2D([],[],color='#222222',label='Observed'),
            Line2D([],[],color='#8064A2',ls='--',label='Conditional prediction'),
            Line2D([],[],color='#777777',lw=1.4,label='Generated mean'),
            Line2D([],[],color='#777777',alpha=.3,lw=5,label='Central 80% prediction interval')],
            loc='lower center',bbox_to_anchor=(.5,.065),ncol=2)
        fig.text(.5,.025,'Thin lines: all 128 sampled paths. Vertical dotted line: training boundary. Prefix: 8 filters × 16 branches.',ha='center',fontsize=6)
        save(fig,output,f'subject_{subject}')
    pd.DataFrame(source_rows).to_csv(output/'trajectory_paths.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(output/'trajectory_summary.csv',index=False)
    scores=pd.read_csv(summary/'scores.csv').query("phase == 'test'")
    feature=pd.read_csv(features/'feature_scores.csv').query("phase == 'test'")
    prefix=pd.read_csv(prefix_scores/'scores.csv').query("start == 'filtered_prefix'")
    errors=pd.read_csv(uncertainty/'primary.csv')
    metrics=[(scores,'choice_nll','choice_nll_mc_se','Future-choice NLL'),
        (feature,'energy_u','mc_se','Autonomous feature energy (U)'),
        (errors,'energy','energy_mc_se','Autonomous curve energy (V)'),
        (prefix,'energy','energy_mc_se','Prefix-forecast curve energy (V)')]
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,122/25.4));fig.subplots_adjust(left=.1,right=.98,bottom=.19,top=.91,hspace=.42,wspace=.3)
    plotted=[]
    for index,(frame,value,error,title) in enumerate(metrics):
        ax=axes.flat[index];panel(ax,'abcd'[index]);ax.set_title(title,loc='left')
        for i,subject in enumerate(meta['subjects']):
            rows=frame[frame.subject==subject].set_index('model')
            ax.plot([i-.08,i+.08],[rows.loc[arm,value] for arm in ARMS],color='#999999',lw=.7)
            for offset,arm in zip((-.08,.08),ARMS):
                score,se=float(rows.loc[arm,value]),float(rows.loc[arm,error])
                ax.errorbar(i+offset,score,yerr=se,fmt=MARKERS[arm],color=COLORS[arm],ms=3.5,lw=.7,capsize=2)
                plotted.append(dict(subject=subject,model=arm,metric=title,value=score,mc_se=se))
        ax.set(xticks=np.arange(len(meta['subjects'])),xticklabels=meta['subjects'],xlabel='Participant',ylabel='Score (lower is better)')
    fig.legend(handles=[Line2D([],[],color=COLORS[a],marker=MARKERS[a],lw=0,label=a) for a in ARMS],loc='lower center',bbox_to_anchor=(.5,.055),ncol=2)
    fig.text(.5,.025,'Error bars: Monte Carlo SE at fixed parameters. Six reserved participants; metrics retain their own scales.',ha='center',fontsize=6)
    save(fig,output,'common_scores');pd.DataFrame(plotted).to_csv(output/'common_scores.csv',index=False)
    (output/'completion.json').write_text(json.dumps(dict(status='complete',subjects=meta['subjects'],path_counts=path_counts,
        nomination_sha256=meta['nomination_sha256'],forecast_paths_sha256=hashes),indent=2))


def recovery(source: Path, output: Path) -> None:
    frame=pd.read_csv(source/'parameters.csv');done=load(source/'completion.json');assert done['status']=='complete'
    destination(output,'recovery',dict(summary=str(source.resolve())))
    parameters=list(frame.parameter.unique());assert len(parameters)==14
    fig,axes=plt.subplots(4,4,figsize=(183/25.4,183/25.4));fig.subplots_adjust(left=.08,right=.98,bottom=.08,top=.92,hspace=.65,wspace=.55)
    for i,parameter in enumerate(parameters):
        ax=axes.flat[i];data=frame[frame.parameter==parameter];panel(ax,chr(97+i))
        values=np.r_[data.truth,data.estimate];lo,hi=values.min(),values.max();pad=(hi-lo)*.08 or .1
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#999999',lw=.7,ls=':')
        for arm in ARMS:
            group=data[data.model==arm]
            assert len(group)==done['cases']
            ax.scatter(group.truth,group.estimate,color=COLORS[arm],marker=MARKERS[arm],s=14,alpha=.65,linewidths=0)
        name=parameter.rsplit('.',1)[-1].replace('_',' ')
        if name=='decay':name='evidence decay'
        ax.set(title=textwrap.fill(name,18),xlabel='Generating value',ylabel='Fitted value',xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad))
        ax.locator_params(nbins=3)
    for ax in axes.flat[len(parameters):]:ax.axis('off')
    axes.flat[-2].legend(handles=[Line2D([],[],color=COLORS[a],marker=MARKERS[a],lw=0,label=a) for a in ARMS],loc='center')
    axes.flat[-1].text(.05,.75,'Synthetic recovery\n6 generating points\n2 sequences per point\n14 coordinates fitted\nEstimates frozen',va='top',fontsize=7,linespacing=1.6)
    fig.suptitle('Joint parameter recovery · bounded pilot',fontsize=10,y=.98)
    save(fig,output,'parameter_recovery');frame.to_csv(output/'parameter_recovery_source.csv',index=False)
    (output/'completion.json').write_text(json.dumps(dict(status='complete',cases=done['cases'],coordinates=14,
        source_parameters_sha256=digest(source/'parameters.csv'),overlap='Every estimate drawn; coincident values overlap without jitter'),indent=2))


def task(source: Path, stopping: Path, output: Path) -> None:
    frame=pd.read_csv(source/'paths.csv');protocol=load(source/'protocol.json');observed=pd.read_csv(stopping/'observed.csv')
    destination(output,'task',dict(task=str(source.resolve()),stopping=str(stopping.resolve())))
    subjects=sorted(frame.subject.unique());horizon=protocol['settings']['max_blocks'];rows=[]
    fig,axes=plt.subplots(3,2,figsize=(183/25.4,173/25.4));fig.subplots_adjust(left=.09,right=.98,bottom=.13,top=.93,hspace=.48,wspace=.28)
    for i,subject in enumerate(subjects):
        ax=axes.flat[i];panel(ax,chr(97+i));ax.set_title(f'Participant {subject}',loc='left')
        for arm in ARMS:
            for policy,style in [('adaptive','-'),('adaptive_repeat',':'),('balanced','--')]:
                group=frame[(frame.subject==subject)&(frame.model==arm)&(frame.policy==policy)]
                assert len(group)==protocol['settings']['paths']
                times=np.where(group.passed,group.blocks,horizon+1)
                x=np.arange(0,horizon+2);cdf=(times[:,None]<=x[None,:]).mean(axis=0)
                ax.step(x,cdf,where='post',color=COLORS[arm],ls=style,lw=1.2)
                rows.extend(dict(subject=subject,model=arm,policy=policy,block=int(t),cdf=float(p),paths=len(group)) for t,p in zip(x,cdf))
        target=observed.loc[observed.subject==subject,'first_passing_block'].item()
        ax.axvline(target,color='#222222',ls='-.',lw=.8)
        ticks=list(range(0,horizon,5))+[horizon+1]
        ax.set(xlim=(0,horizon+1.5),ylim=(-.02,1.02),xlabel='First-passing block',ylabel='Cumulative probability',
            xticks=ticks,xticklabels=[str(x) for x in ticks[:-1]]+[f'>{horizon}'])
    for ax in axes.flat[len(subjects):]:ax.axis('off')
    fig.legend(handles=[Line2D([],[],color=COLORS[a],label=a) for a in ARMS]+[
        Line2D([],[],color='#777777',ls='-',label='Adaptive'),Line2D([],[],color='#777777',ls=':',label='Session RNG reset'),
        Line2D([],[],color='#777777',ls='--',label='Balanced'),
        Line2D([],[],color='#222222',ls='-.',label='Observed first pass')],loc='lower center',bbox_to_anchor=(.5,.035),ncol=3)
    fig.suptitle('Task-level first-passage distributions · all paths retained',fontsize=10,y=.98)
    save(fig,output,'stopping_cdf');pd.DataFrame(rows).to_csv(output/'stopping_cdf_source.csv',index=False)
    (output/'completion.json').write_text(json.dumps(dict(status='complete',subjects=[int(s) for s in subjects],
        exclusions=protocol['excluded_subjects'],horizon=horizon,source_paths_sha256=digest(source/'paths.csv')),indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['primary','recovery','task'],required=True)
    for name in ('summary','features','forecast','prefix-scores','uncertainty','source','stopping','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.mode=='primary':primary(args.summary,args.features,args.forecast,args.prefix_scores,args.uncertainty,args.output)
    elif args.mode=='recovery':recovery(args.source,args.output)
    else:task(args.source,args.stopping,args.output)


if __name__=='__main__':main()
