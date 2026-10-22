"""Plot early-fit prediction and full-choice reconstruction side by side.

Quantitative grids ask whether additional behavioral information changes the
autonomous learning distribution and external oral support in two development
participants. Every saved path and both oral readings are retained. Both
functional and structural compatibility and all time windows are available.
The primary visual comparison uses full-report denominators; catalogue coverage
is written next to each bar. Python produces 183-mm PNG review figures, not
submission vector assets. No population or parameter confidence is implied.
Use --source COMPLETED_SCOPE_SUMMARY --output NEW.
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1022_compact_scope_control import read,write,digest

COLORS={'EARLY':'#527697','FULL':'#3B8C85'}
NAMES={'EARLY':'Early-choice fit','FULL':'Full-choice fit'}
MASS={'support':'#538B84','conflict':'#C58B60','undetermined':'#D9DDE2'}


def run(source: Path,output: Path):
    if output.exists():raise FileExistsError('Use a new scope-figure directory')
    done=read(source/'completion.json')
    if done['status']!='complete':raise ValueError('Complete the descriptive scope summary first')
    bindings=dict(done['inputs_sha256']);bindings[str((source/'completion.json').resolve())]=digest(source/'completion.json')
    for name,sha in done['artifacts_sha256'].items():bindings[str((source/name).resolve())]=sha
    for name,sha in bindings.items():
        if digest(Path(name))!=sha:raise ValueError('Figure input changed: '+name)
    curves=pd.read_csv(source/'curves.csv');paths=pd.read_csv(source/'path_blocks.csv')
    if sorted(curves.subject.unique())!=[205,306] or set(curves.scope)!=set(COLORS):raise ValueError('Expected both participants and scopes')
    output.mkdir(parents=True);own=output/'source_data';own.mkdir()
    for file in source.rglob('*.csv'):
        target=own/file.relative_to(source);target.parent.mkdir(exist_ok=True);shutil.copyfile(file,target)
    files=[]

    def save(fig,name):
        fig.savefig(output/name,dpi=300,facecolor='white');plt.close(fig);files.append(name)

    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False}):
        fig,axes=plt.subplots(2,2,figsize=(183/25.4,140/25.4),sharey=True)
        fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.83,hspace=.40,wspace=.18)
        for row,subject in enumerate((205,306)):
            for col,scope in enumerate(COLORS):
                ax=axes[row,col];curve=curves.query('subject==@subject and scope==@scope').sort_values('trial')
                sample=paths.query('subject==@subject and scope==@scope').pivot(index='seed',columns='trial',values='accuracy')
                if len(sample)!=128 or sample.isna().any().any():raise ValueError('Missing autonomous paths')
                np.testing.assert_array_equal(sample.columns,curve.trial)
                np.testing.assert_allclose(sample.mean(0),curve.autonomous,atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.quantile(sample.to_numpy(),[.1,.9],axis=0),curve[['lower','upper']].to_numpy().T,atol=1e-12,rtol=0)
                cut=int(curve.original_training_trials.iloc[0]);last=curve.trial.max()
                if scope=='EARLY':ax.axvspan(cut,last,color='#F3F4F6')
                ax.axvline(cut,color='#7B8087',ls=':',lw=.7)
                ax.plot(curve.trial,sample.to_numpy().T,color=COLORS[scope],alpha=.075,lw=.35)
                ax.plot(curve.trial,curve.autonomous,color=COLORS[scope],lw=1.6)
                ax.plot(curve.trial,curve.conditional,color=COLORS[scope],lw=1,ls='--')
                ax.plot(curve.trial,curve.observed,color='#222222',marker='o',ms=2,lw=1)
                ax.set(xlim=(0,last),ylim=(0,1.02),yticks=[0,.5,1.])
                ax.set_title(f'S{subject} · {NAMES[scope]}',fontsize=8)
                if col==0:ax.set_ylabel('Accuracy')
                if row==1:ax.set_xlabel('Trial (32-trial blocks)')
        fig.suptitle('TRANSFER · two development participants',y=.985,fontsize=10)
        fig.legend(handles=[Line2D([],[],color='#222222',marker='o',ms=2,lw=1,label='Participant'),
            Line2D([],[],color=COLORS['EARLY'],lw=1.6,label='Autonomous mean'),
            Line2D([],[],color=COLORS['EARLY'],lw=1,ls='--',label='Conditional prediction'),
            Line2D([],[],color=COLORS['EARLY'],lw=.5,alpha=.4,label='All 128 autonomous paths')],
            loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,fontsize=7)
        fig.text(.5,.065,'Left shading: choices excluded from early fitting. Right: all choices used to fit.',ha='center',fontsize=7)
        fig.text(.5,.025,'Full-choice fit is descriptive reconstruction; the dotted line retains the original time boundary.',ha='center',fontsize=6.5)
        save(fig,'behavior_scope.png')
        oral=pd.concat([pd.read_csv(source/f'oral_{r}/summary.csv').assign(reading=r) for r in ('midpoint','relative')])
        for domain in ('functional','structural'):
            fig,axes=plt.subplots(2,3,figsize=(183/25.4,140/25.4),sharex=True)
            fig.subplots_adjust(left=.22,right=.98,bottom=.17,top=.82,hspace=.65,wspace=1.30)
            order=[(r,s) for r in ('midpoint','relative') for s in COLORS]
            for row,subject in enumerate((205,306)):
                for col,(window,title) in enumerate([('training','Earlier trials'),('test','Later trials'),('whole','Whole sequence')]):
                    ax=axes[row,col]
                    group=oral.query('subject==@subject and domain==@domain and window==@window')
                    full=group.query('subset=="complete"').set_index(['reading','arm']).loc[order]
                    coverage=group.query('subset=="catalogue_supported"').set_index(['reading','arm']).loc[order]
                    if full.reports.nunique()!=1:raise ValueError('Report denominator changed')
                    values=full[list(MASS)].to_numpy()
                    np.testing.assert_allclose(values.sum(1),1,atol=1e-12,rtol=0)
                    base=np.zeros(4)
                    for key,color in MASS.items():
                        ax.barh(range(4),full[key],left=base,color=color,height=.65);base+=full[key].to_numpy()
                    labels=[f'{"Early" if s=="EARLY" else "Full"} · {r}\n'
                            f'{int(coverage.loc[(r,s),"reports"])} / {int(full.loc[(r,s),"reports"])} covered' for r,s in order]
                    ax.set_yticks(range(4),labels,fontsize=6.2);ax.invert_yaxis()
                    ax.set(xlim=(0,1),xticks=[0,.5,1]);ax.set_title(f'S{subject} · {title}',fontsize=7)
                    if row==1:ax.set_xlabel('Compatibility mass',fontsize=6.8)
            fig.suptitle(domain.capitalize()+' oral support · all complete reports',y=.985,fontsize=10)
            fig.legend(handles=[Patch(facecolor=c,label=k.capitalize()) for k,c in MASS.items()],loc='upper center',
                       bbox_to_anchor=(.5,.95),ncol=3,fontsize=7)
            fig.text(.5,.07,'Early / Full: parameter-estimation scope. Both oral readings remain fixed.',ha='center',fontsize=7)
            fig.text(.5,.03,'Covered / complete report counts accompany every bar. Oral content is a noisy external check.',ha='center',fontsize=6.5)
            save(fig,'oral_'+domain+'_scope.png')
    page='<!doctype html><meta charset="utf-8"><title>完整行为重建对照</title><style>body{max-width:1100px;margin:32px auto;font:16px sans-serif}img{width:100%}</style>'
    page+='<h1>估计阶段对照：原早期参数与完整行为参数</h1><p>两名开发被试，均使用TRANSFER机制。完整行为估计用于事后重建，口述继续作为带噪声的外部检验。原后期预测成绩保留独立解释。全部128条自主路径和两种固定口述读法均呈现。</p>'
    page+=''.join('<p><img src="'+html.escape(n)+'"></p>' for n in files)
    for name in ('scores','contrasts','parameters','finalists'):
        page+='<h2>'+name+'</h2>'+pd.read_csv(source/(name+'.csv')).to_html(index=False)
    (output/'index.html').write_text(page,encoding='utf-8');shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    write(output/'completion.json',dict(status='complete',inputs_sha256=bindings,script_sha256=digest(Path(__file__)),
        source=str(source.resolve()),figures={n:digest(output/n) for n in files},format='PNG',dpi=300,width_mm=183,
        exclusions=[],all_paths_retained=True,scope='Review figures; full-choice fits have no held-out choices'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
