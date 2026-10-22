"""Plot the six-person compact-mechanism comparison from frozen evaluations.

Figure contract: assess whether precision transfer preserves later choices and
improves autonomous learning, and show the support available from noisy oral
reports. Quantitative grids separate development and extension participants.
Every autonomous path, both fixed oral readings and all selection/confirmation
candidates are retained. Python/matplotlib produces 183-mm PNG review figures; errors are
two paired Monte Carlo SE at fitted parameters. The existing figure contributes
style and its validated selection-score reader, with explicit variant mapping.
Use --source SUMMARY --output NEW after the full six-person evaluation finishes.
"""
from __future__ import annotations

import argparse
import html
import inspect
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import block_paths, checked_arrays, digest, load
from src.Bayesian_state.workflows.analysis.model_1021_simplification import io
from .model1021_simplification import selection_candidates

COLORS = dict(COMPACT='#527697', TRANSFER='#3B8C85')
NAMES = dict(COMPACT='Compact', TRANSFER='Precision transfer')
MASS = dict(support='#538B84', conflict='#C58B60', undetermined='#D9DDE2')
COHORTS = dict(development='Development', extension='Extension · prior influence unknown')


def load_data(source: Path) -> dict:
    done = load(source/'completion.json')
    if done['status']!='complete' or len(done['subjects'])!=6:
        raise ValueError('Require the completed six-person comparison')
    study, evaluation = Path(done['study']),Path(done['evaluation'])
    bindings = {}

    def bind(file: Path, expected=None):
        sha = digest(file)
        if expected is not None and sha!=expected:
            raise ValueError('Figure input changed: '+str(file))
        bindings[str(file.resolve())] = sha
        return file

    bind(source/'completion.json')
    bind(Path(inspect.getfile(selection_candidates)))
    for name,sha in done['inputs_sha256'].items():
        bind(Path(name),sha)
    tables = {name.removesuffix('.csv'):pd.read_csv(bind(source/name,sha))
              for name,sha in done['artifacts_sha256'].items()}
    cases = [c for c in load(bind(study/'cases.json'))['cases'] if c['kind']=='human']
    paths, counts, selections, parameters = {}, {}, [], []
    for case in cases:
        subject,variant = case['subject'],case['variant']
        fit = Path(case['output']); frozen = load(bind(fit/'nominees.json'))
        selected,coords = selection_candidates(fit,frozen,bind)
        selections.append(selected.assign(model=variant,cohort=case['cohort']))
        parameters.append(coords.assign(model=variant,cohort=case['cohort']))
        confirmed=tables['finalists'].query('subject == @subject and model == @variant')
        if (len(confirmed)!=frozen['settings']['confirmation_candidates']
                or confirmed.candidate.duplicated().any() or confirmed.selected.sum()!=1
                or not set(confirmed.candidate)<=set(selected.candidate)
                or confirmed.loc[confirmed.selected,'candidate'].iloc[0]!=selected.loc[selected.selected,'candidate'].iloc[0]):
            raise ValueError('Confirmation candidates differ from the selected parameter bank')
        if (not np.isfinite(confirmed[['delta_nll','paired_mc_se']].to_numpy()).all()
                or (confirmed.delta_nll < -1e-12).any() or (confirmed.paired_mc_se < 0).any()):
            raise ValueError('Invalid confirmation differences or paired uncertainty')
        own = evaluation/case['case_id']/'autonomous'
        record, = load(bind(own/'evaluation.json'))['rows']
        arrays = checked_arrays(own,record)
        file = own/'evaluate'/f'subject_{subject}'/'COMPACT'/(record['candidate']['id']+'.npz')
        bind(file,record['predictions_sha256'])
        samples = block_paths(arrays['choices']==arrays['categories'][None,:],0,len(arrays['categories']),
                              frozen['settings']['features']['block_width'])
        curves = tables['curves']
        curve = curves[(curves.subject==subject)&(curves.model==variant)].sort_values('trial')
        if samples.shape!=(frozen['settings']['autonomous_paths'],len(curve)):
            raise ValueError('Autonomous path count or time grid differs')
        np.testing.assert_allclose(samples.mean(0),curve.autonomous,atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.quantile(samples,[.1,.9],axis=0),curve[['lower','upper']].to_numpy().T,
                                   atol=1e-12,rtol=0)
        paths[subject,variant] = samples,curve,arrays['seeds']
        counts[subject,variant] = sum(len(v)>1 for v in case['space'].values())
    tables['selection_candidates'] = pd.concat(selections,ignore_index=True)
    tables['selection_parameters'] = pd.concat(parameters,ignore_index=True)
    oral = pd.concat([tables['oral_'+r+'/summary'].assign(reading=r) for r in ('midpoint','relative')])
    complete = oral.query('window == "test" and subset == "complete"')
    if complete.duplicated(['subject','arm','domain','reading']).any():
        raise ValueError('Duplicate oral comparison cell')
    for _,rows in complete.groupby(['subject','domain','reading']):
        if set(rows.arm)!=set(COLORS) or rows.reports.nunique()!=1:
            raise ValueError('Missing oral variant or unequal denominator')
        values = rows[list(MASS)].to_numpy()
        if rows.reports.iloc[0]:
            if not np.isfinite(values).all():
                raise ValueError('Undefined oral mass for available reports')
            np.testing.assert_allclose(values.sum(1),1,atol=1e-12,rtol=0)
        elif not np.isnan(values).all():
            raise ValueError('Absent reports must retain undefined scores')
    return dict(tables=tables,paths=paths,counts=counts,oral=complete,
                coverage=oral.query('window == "test" and subset == "catalogue_supported"'),bindings=bindings,
                tolerances=done['working_tolerances'])


def save(fig, output: Path, name: str):
    fig.savefig(output/name,dpi=300,facecolor='white')
    plt.close(fig)


def trajectories(data: dict, subjects: list[int], cohort: str, output: Path):
    fig,axes = plt.subplots(3,2,figsize=(183/25.4,176/25.4),sharey=True)
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.83,hspace=.38,wspace=.18)
    for row,s in enumerate(subjects):
        for col,variant in enumerate(COLORS):
            ax=axes[row,col]; samples,curve,_=data['paths'][s,variant]
            cut=int(curve.training_trials.iloc[0])
            ax.axvspan(cut,curve.trial.max(),color='#F3F4F6')
            ax.plot(curve.trial,samples.T,color=COLORS[variant],alpha=.075,lw=.35)
            ax.plot(curve.trial,curve.autonomous,color=COLORS[variant],lw=1.6)
            ax.plot(curve.trial,curve.conditional,color=COLORS[variant],lw=1,ls='--')
            ax.plot(curve.trial,curve.observed,color='#222222',marker='o',ms=2,lw=1)
            ax.axvline(cut,color='#7B8087',ls=':',lw=.7)
            ax.set(xlim=(0,curve.trial.max()),ylim=(0,1.02),yticks=[0,.5,1.])
            ax.set_title(f'S{s} · {NAMES[variant]} · {data["counts"][s,variant]} parameters',fontsize=8)
            if col==0: ax.set_ylabel('Accuracy')
            if row==2: ax.set_xlabel('Trial (32-trial blocks)')
    handles=[Line2D([],[],color='#222222',marker='o',ms=2,lw=1,label='Participant'),
             Line2D([],[],color=COLORS['COMPACT'],lw=1.6,label='Autonomous mean'),
             Line2D([],[],color=COLORS['COMPACT'],lw=1,ls='--',label='Conditional prediction'),
             Line2D([],[],color=COLORS['COMPACT'],lw=.5,alpha=.4,label='All 128 autonomous paths')]
    fig.suptitle(COHORTS[cohort],y=.985,fontsize=10)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,fontsize=7)
    fig.text(.5,.025,'Shading: later trials excluded from parameter fitting.',ha='center',fontsize=7)
    save(fig,output,'trajectories_'+cohort+'.png')


def behavior(data: dict, output: Path):
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,147/25.4))
    fig.subplots_adjust(left=.20,right=.98,bottom=.18,top=.91,hspace=.55,wspace=.60)
    for row,cohort in enumerate(COHORTS):
        rows=data['tables']['contrasts'].query('phase == "test" and cohort == @cohort').sort_values('subject')
        mean=data['tables']['aggregate'].query('phase == "test" and cohort == @cohort').iloc[0]
        labels=[f'S{s}' for s in rows.subject]+['Equal-person mean']
        for col,(metric,title) in enumerate([('choice_nll','Choice NLL / trial'),('energy','Autonomous Energy score')]):
            ax=axes[row,col]; values=np.r_[rows['delta_'+metric],mean['delta_'+metric]]
            errors=2*np.r_[rows['delta_'+metric+'_mc_se'],mean['delta_'+metric+'_mc_se']]
            ax.axvline(0,color='#80868E',lw=.8)
            ax.axvline(data['tolerances'][metric],color='#B08D65',ls=':',lw=.8)
            ax.errorbar(values,np.arange(4),xerr=errors,fmt='o',color=COLORS['TRANSFER'],ms=4,capsize=2,lw=1)
            ax.set_yticks(np.arange(4),labels,fontsize=7)
            ax.set_ylim(3.5,-.5)
            ax.set_title(COHORTS[cohort].split(' ·')[0]+' · '+title,fontsize=8)
            ax.set_xlabel('Transfer − compact; lower favors transfer',fontsize=7)
    fig.text(.5,.085,'Bars: ±2 paired MCSE at fitted parameters. Dotted line: working loss tolerance.',ha='center',fontsize=7)
    fig.text(.5,.04,'Three development and three extension participants; extension prior influence is unknown.',ha='center',fontsize=7)
    save(fig,output,'later_behavior.png')


def oral(data: dict, subjects: list[int], cohort: str, output: Path):
    fig,axes=plt.subplots(3,2,figsize=(183/25.4,170/25.4),sharex=True)
    fig.subplots_adjust(left=.24,right=.98,top=.86,bottom=.15,hspace=.46,wspace=.82)
    order=[(reading,variant) for reading in ('midpoint','relative') for variant in COLORS]
    for row,s in enumerate(subjects):
        for col,domain in enumerate(('functional','structural')):
            ax=axes[row,col]
            table=data['oral'].query('subject == @s and domain == @domain').set_index(['reading','arm']).loc[order]
            covered=data['coverage'].query('subject == @s and domain == @domain').set_index(['reading','arm']).loc[order]
            offset=np.zeros(4)
            for key,color in MASS.items():
                values=np.nan_to_num(table[key].to_numpy())
                ax.barh(np.arange(4),values,left=offset,color=color,height=.65);offset+=values
            labels=[f'{NAMES[v]} · {r}\n{int(table.loc[(r,v),"reports"])} reports · '
                    f'{int(covered.loc[(r,v),"reports"])} covered' for r,v in order]
            ax.set_yticks(range(4),labels,fontsize=6.4);ax.invert_yaxis()
            ax.set(xlim=(0,1),xticks=[0,.5,1]);ax.set_title(f'S{s} · {domain.capitalize()}',fontsize=8)
            for i,count in enumerate(table.reports):
                if count==0: ax.text(.5,i,'No complete reports',ha='center',va='center',fontsize=7)
            if row==2: ax.set_xlabel('Content compatibility mass')
    fig.suptitle(COHORTS[cohort],y=.99,fontsize=10)
    fig.legend(handles=[Patch(facecolor=c,label=k.capitalize()) for k,c in MASS.items()],
               loc='upper center',ncol=3,bbox_to_anchor=(.6,.95),fontsize=8)
    fig.text(.5,.04,'Covered: reports supported by at least one catalogue rule, independent of fitted states.',ha='center',fontsize=7)
    fig.text(.5,.015,'Bars use all complete reports. Grey mass retains ambiguity and catalogue limits.',ha='center',fontsize=7)
    save(fig,output,'oral_'+cohort+'.png')


def candidates(data: dict, subjects: list[int], cohort: str, output: Path):
    fig,axes=plt.subplots(3,2,figsize=(183/25.4,180/25.4))
    fig.subplots_adjust(left=.20,right=.98,top=.86,bottom=.14,hspace=.50,wspace=.70)
    for row,s in enumerate(subjects):
        for col,variant in enumerate(COLORS):
            ax=axes[row,col]
            points=data['tables']['selection_candidates'].query('subject == @s and model == @variant').sort_values('selection_rank')
            confirmed=data['tables']['finalists'].query('subject == @s and model == @variant').set_index('candidate')
            labels=[]
            for i,p in enumerate(points.itertuples()):
                labels.append(('Chosen' if p.selected else f'Candidate {p.selection_rank}')+(' · start' if p.warm_start else ''))
                ax.errorbar(p.delta_nll,i-.14,xerr=2*p.paired_mc_se,fmt='o',color='#8A9098',
                            markerfacecolor='white',ms=3,capsize=2,lw=.8)
                if p.candidate in confirmed.index:
                    q=confirmed.loc[p.candidate]
                    ax.errorbar(q.delta_nll,i+.14,xerr=2*q.paired_mc_se,fmt='D',color=COLORS[variant],
                                ms=3.2,capsize=2,lw=.9)
            ax.axvline(0,color='#80868E',lw=.7);ax.set_yticks(range(len(points)),labels,fontsize=7)
            ax.set_ylim(len(points)-.5,-.5);ax.set_title(f'S{s} · {NAMES[variant]}',fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3))
            if row==2: ax.set_xlabel('NLL / trial − chosen candidate',fontsize=7)
    fig.suptitle(COHORTS[cohort],y=.98,fontsize=10)
    fig.legend(handles=[Line2D([],[],marker='o',mfc='white',color='#8A9098',ls='',ms=3,
                              label='Selection: 512 particles × 4 filters'),
                        Line2D([],[],marker='D',color='#444444',ls='',ms=3.2,
                              label='Confirmation: 1024 particles × 8 filters')],
               loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,fontsize=6.5)
    fig.text(.5,.06,'All six selection candidates; four receive confirmation scores. Bars: ±2 paired MCSE.',ha='center',fontsize=7)
    fig.text(.5,.025,'Training scores relative to confirmed nominee; each panel has its own x scale.',ha='center',fontsize=7)
    save(fig,output,'candidates_'+cohort+'.png')


def run(source: Path, output: Path):
    if output.exists(): raise FileExistsError('Use a new figure directory')
    data=load_data(source);output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False}):
        behavior(data,output)
        for cohort in COHORTS:
            subjects=sorted(data['tables']['scores'].query('cohort == @cohort').subject.unique().tolist())
            if len(subjects)!=3: raise ValueError('Each declared cohort contains three people')
            trajectories(data,subjects,cohort,output);oral(data,subjects,cohort,output);candidates(data,subjects,cohort,output)
    own=output/'source_data';own.mkdir()
    for name,frame in data['tables'].items(): frame.to_csv(own/(name.replace('/','_')+'.csv'),index=False)
    rows=[]
    for (s,v),(samples,curve,seeds) in data['paths'].items():
        rows.extend(dict(subject=s,model=v,seed=int(seed),trial=int(t),accuracy=float(x))
                    for seed,path in zip(seeds,samples) for t,x in zip(curve.trial,path))
    pd.DataFrame(rows).to_csv(own/'autonomous_path_blocks.csv',index=False)
    images=['later_behavior.png']+[f'{kind}_{cohort}.png' for cohort in COHORTS for kind in ('trajectories','oral','candidates')]
    page='<!doctype html><meta charset="utf-8"><title>简化机制重拟合比较</title><style>body{font:16px sans-serif;max-width:1100px;margin:32px auto}img{width:100%}</style>'
    page+='<h1>简化机制与精度继承</h1><p>两种机制使用相同拟合预算。原开发组与扩展组分别显示；扩展组既往开发影响未确定。参数仅由训练行为选择。细线包含全部自主路径。口述保留两种读法及未确定质量，表示内容兼容性。误差条为固定参数下的模拟误差。</p>'
    page+='<p>候选图逐行对应同一参数点：灰色圆圈为选择阶段成绩，实心菱形为独立确认成绩。全部六个选择候选均显示，其中四个接受确认。每个阶段的差值均相对于最终选中点在该阶段的成绩计算；各面板使用自己的横轴范围。</p>'
    page+=''.join(f'<p><a href="{html.escape(n)}"><img src="{html.escape(n)}"></a></p>' for n in images)
    (output/'index.html').write_text(page)
    io.write(output/'manifest.json',dict(status='complete',source=str(source.resolve()),input_sha256=data['bindings'],
        source_sha256=digest(Path(__file__)),all_autonomous_paths_retained=True,exclusions=[],
        export='183-mm PNG at 300 dpi; review figures',artifacts_sha256={n:digest(output/n) for n in images},
        uncertainty='Two MCSE at fitted parameters; parameter and population uncertainty excluded'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
