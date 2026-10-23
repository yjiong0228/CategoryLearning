"""Show dynamic effects of response certainty at every frozen candidate.

Figure contract: quantitative grids test whether final response sharpening
changes complete learning and continuation, rather than just a saved-state
probability. Behavior is the main evidence; predictive distributions show
variability and paired score differences retain simulation uncertainty.
Python only, 183-mm 300-dpi PNG, all eighteen candidates and all 128 paths.
Bands are central 80% predictive ranges. Error bars are one paired MC standard
error, not participant confidence intervals. Style-only inheritance; new data
mapping uses exponent, start, candidate and trial as explicit grouping keys.

Use --source COMPLETE_DYNAMIC_SUMMARY --output NEW. Raw tables are copied as
source data; there is no fitting, filtering by performance or path subsampling.
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

from src.Bayesian_state.workflows.analysis.model_1023_response_dynamics import read,write,digest,verify

STYLE={'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
       'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
       'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'}
COLORS={1:'#527697',2:'#C47742'}
LABELS={1:'Original response',2:'Squared response'}
FIG_WIDTH_MM=183


def save(fig,output,name):
    fig.savefig(output/name,dpi=300,facecolor='white');plt.close(fig);return name


def points(tables,subject):
    rows=tables['scores'].query('subject==@subject and start=="task_start" and exponent==1')
    rows=rows.sort_values(['selected','roles','candidate'],ascending=[False,True,True])
    if len(rows)!=3 or rows.selected.sum()!=1:raise ValueError('Need all three candidates')
    result=[];alternative=0
    for r in rows.itertuples():
        if r.selected:label='Selected'
        elif 'reference' in r.roles:label='Reference'
        else:
            alternative+=1;label=f'Alternative {alternative}'
        result.append((r.candidate,label))
    return result


def trajectories(tables,subject,output):
    fig,axes=plt.subplots(2,3,figsize=(FIG_WIDTH_MM/25.4,132/25.4),sharey=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.82,hspace=.39,wspace=.20)
    for col,(candidate,label) in enumerate(points(tables,subject)):
        for row,start in enumerate(('task_start','prefix')):
            ax=axes[row,col]
            for exponent in (1,2):
                curve=tables['curves'].query('subject==@subject and candidate==@candidate and start==@start and exponent==@exponent').sort_values('trial')
                paths=tables['path_blocks'].query('subject==@subject and candidate==@candidate and start==@start and exponent==@exponent')
                matrix=paths.pivot(index='seed',columns='trial',values='accuracy')
                if len(matrix)!=128:raise ValueError('Missing paths')
                np.testing.assert_array_equal(matrix.columns,curve.trial)
                np.testing.assert_allclose(matrix.mean(0),curve['mean'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.quantile(matrix,[.1,.9],axis=0),curve[['q10','q90']].to_numpy().T,atol=1e-12,rtol=0)
                ax.plot(curve.trial,matrix.to_numpy().T,color=COLORS[exponent],alpha=.035,lw=.3)
                ax.fill_between(curve.trial,curve.q10,curve.q90,color=COLORS[exponent],alpha=.11,lw=0)
                ax.plot(curve.trial,curve['mean'],color=COLORS[exponent],lw=1.6)
            ax.plot(curve.trial,curve.observed,color='#222222',marker='o',ms=2,lw=1.1)
            cut=curve.training_trials.iloc[0]
            if start=='task_start':ax.axvline(cut,color='#999999',ls=':',lw=.7)
            ax.set(xlim=(0 if start=='task_start' else cut,curve.trial.max()),ylim=(-.02,1.02),yticks=[0,.5,1],xlabel='Trial (32-trial blocks)')
            if row==0:ax.set_title(label,fontsize=8)
    axes[0,0].set_ylabel('From task start\nAccuracy');axes[1,0].set_ylabel('From inferred prefix\nAccuracy')
    handles=[Line2D([],[],color=COLORS[q],lw=1.6,label=LABELS[q]) for q in (1,2)]
    handles.append(Line2D([],[],color='#222222',marker='o',ms=3,label='Participant'))
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.92))
    fig.suptitle(f'Subject {subject}: dynamic response comparison',fontsize=10,y=.98)
    fig.text(.09,.035,'All 128 paths per arm; shaded central 80% predictive range. Same frozen parameters and paired seeds.\nEach mechanism infers its own prefix state. Dotted line: original estimation boundary.',fontsize=7)
    return save(fig,output,f'S{subject}_response_dynamics.png')


def contrasts(tables,output):
    fig,axes=plt.subplots(2,3,figsize=(FIG_WIDTH_MM/25.4,130/25.4),sharex=True)
    fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.86,hspace=.35,wspace=.35)
    for ax,subject in zip(axes.flat,(108,205,306,110,209,309)):
        own=tables['contrasts'].query('subject==@subject')
        labels=[]
        for i,(candidate,label) in enumerate(points(tables,subject)):
            labels.append(label)
            for start,offset,color in [('task_start',-.13,'#527697'),('prefix',.13,'#C47742')]:
                row=own[(own.candidate==candidate)&(own.start==start)].iloc[0]
                ax.errorbar(row.delta_energy,i+offset,xerr=row.delta_energy_mc_se,fmt='o',ms=4,
                    color=color,mfc=color if label=='Selected' else 'white',lw=.9,capsize=2)
        ax.axvline(0,color='#999999',lw=.7,ls=':');ax.set_yticks(range(3),labels);ax.invert_yaxis()
        ax.set_title(f'Subject {subject}',fontsize=8);ax.set_xlabel('Energy score change')
    fig.suptitle('Squared response minus original response',fontsize=10,y=.98)
    fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='',label=l) for c,l in
        [('#527697','Complete learning'),('#C47742','Prefix continuation')]],ncol=2,loc='upper center',bbox_to_anchor=(.5,.94))
    fig.text(.12,.035,'Negative values favor squared response. Bars: one paired simulation standard error.\nTask-start scores use the entire sequence; prefix scores use only later trials. These are development comparisons.',fontsize=7)
    return save(fig,output,'paired_energy_changes.png')


def distributions(source,tables,output):
    fig,axes=plt.subplots(2,3,figsize=(FIG_WIDTH_MM/25.4,130/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.83,hspace=.36,wspace=.25)
    for ax,subject in zip(axes.flat,(108,205,306,110,209,309)):
        candidate=points(tables,subject)[0][0]
        rows=tables['scores'].query('subject==@subject and candidate==@candidate')
        for start,style in [('task_start','-'),('prefix','--')]:
            for exponent in (1,2):
                file=source/'arrays'/f'S{subject}_TRANSFER_{candidate}_{start}_Q{exponent}.npz'
                with np.load(file) as z:values=(z['choices'][:,-128:]==z['categories'][-128:]).mean(1)
                values=np.sort(values);ax.step(np.r_[values[0],values],np.r_[0,np.arange(1,129)/128],
                    where='post',color=COLORS[exponent],ls=style,lw=1.3)
        ax.axvline(rows.late_observed.iloc[0],color='#222222',lw=1.,ls=':')
        ax.set(title=f'Subject {subject}',xlabel='Last-128 accuracy',xlim=(0,1),ylim=(0,1.02))
    for ax in axes[:,0]:ax.set_ylabel('Cumulative path fraction')
    fig.suptitle('Selected parameters: complete late-performance distributions',fontsize=10,y=.98)
    handles=[Line2D([],[],color=COLORS[q],lw=1.4,label=LABELS[q]) for q in (1,2)]
    handles.extend([Line2D([],[],color='#555555',ls=s,label=l) for s,l in [('-','Task start'),('--','Prefix')]])
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,.94))
    fig.text(.09,.045,'Each curve retains all 128 paths. Dotted vertical line: participant accuracy.\nContinuation branches share eight prefix filters; no parameter or path is selected using this display.',fontsize=7)
    return save(fig,output,'selected_late_distributions.png')


def run(source: Path,output: Path):
    if output.exists():raise FileExistsError('Use a new figure directory')
    done=read(source/'completion.json')
    if done['status']!='complete' or done['candidates']!=18:raise ValueError('Require the complete dynamic comparison')
    verify(done['inputs_sha256']);verify({str(source/k):v for k,v in done['artifacts_sha256'].items()})
    tables={k:pd.read_csv(source/(k+'.csv')) for k in ('scores','curves','path_blocks','states','contrasts')}
    output.mkdir(parents=True);(output/'source_data').mkdir()
    for name in tables:shutil.copy2(source/(name+'.csv'),output/'source_data'/(name+'.csv'))
    pd.DataFrame([dict(subject=s,candidate=c,label=label) for s in (108,205,306,110,209,309)
                  for c,label in points(tables,s)]).to_csv(output/'source_data'/'candidate_labels.csv',index=False)
    with plt.rc_context(STYLE):
        figures=[trajectories(tables,s,output) for s in (108,205,306,110,209,309)]
        figures.extend([contrasts(tables,output),distributions(source,tables,output)])
    body='<h1>作答确定性的动态对照</h1><p>每人保留三个参数候选。蓝色为原读出，橙色为平方读出，黑色为被试。上排从任务起点生成，下排从各机制推断的前段状态继续生成。细线包含全部路径，色带为中央80%的预测范围。</p>'
    body+='<p>口述尚未进入这一阶段；参数保持冻结。评分变化图的横线为配对模拟标准误，不能解释为被试总体的置信区间。</p>'
    body+=''.join(f'<h2>{html.escape(f)}</h2><img src="{f}" style="max-width:1100px;width:100%">' for f in figures)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>动态作答对照</title><body>'+body+'</body></html>')
    write(output/'completion.json',dict(status='complete',source=str(source.resolve()),figures=figures,
        source_completion_sha256=digest(source/'completion.json'),plot_source_sha256=digest(Path(__file__)),
        export='183 mm, 300 dpi, PNG only',exclusions=0,
        inputs_sha256={str(source/(n+'.csv')):digest(source/(n+'.csv')) for n in tables},
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
