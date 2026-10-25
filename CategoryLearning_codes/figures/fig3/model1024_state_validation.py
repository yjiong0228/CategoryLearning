"""Render the registered state-validation handoff as PNG research drafts.

Use --panel COMPLETE_SUMMARY --history COMPLETE_SUMMARY --output NEW.
All registered paths and particle systems are retained in the source tables.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
WIDTH=7.2
COLORS={'filtered':'#587a9e','lag':'#c18c55','full':'#5e9085','equal':'#9b9b9b','evidence':'#587a9e'}
NAMES={'filtered':'Filtering','lag':'32-trial lag','full':'Full smoothing'}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def table(folder: Path,name: str,bindings: dict) -> pd.DataFrame:
    done=json.loads((folder/'completion.json').read_text());file=folder/(name+'.csv')
    if done['status']!='complete' or done['files'][file.name]!=digest(file):raise ValueError('Incomplete or changed figure source')
    bindings[str(file)]=digest(file);return pd.read_csv(file)


def calibration_figure(data: pd.DataFrame,output: Path):
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,5.1),layout='constrained')
    states=[('present','Correct rule present'),('high_belief','Correct rule: belief ≥ 0.5'),
        ('target_execution','Correct rule executed'),('error_available_other','Error: present, other rule executed')]
    for letter,ax,(state,title) in zip('abcd',axes.flat,states):
        ax.plot([0,1],[0,1],color='#b8b8b8',lw=.8,ls='--',zorder=0)
        for mode in ('filtered','lag','full'):
            rows=data[(data.scope=='all')&(data.state==state)&(data['mode']==mode)&(data.trials>0)].sort_values('bin')
            ax.plot(rows.mean_prediction,rows.observed_frequency,color=COLORS[mode],lw=.8,label=NAMES[mode])
            for row in rows.itertuples():
                ax.plot([row.mean_prediction]*2,[row.frequency_lower,row.frequency_upper],color=COLORS[mode],lw=.8)
                ax.scatter([row.mean_prediction],[row.observed_frequency],s=18,marker='o',
                    facecolors=COLORS[mode] if row.sufficient else 'white',edgecolors=COLORS[mode],linewidths=.8,zorder=3)
        ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='Mean predicted probability',ylabel='Observed event frequency',title=title)
        ax.text(-.14,1.08,letter,transform=ax.transAxes,weight='bold',fontsize=9)
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1])
    axes[0,0].legend(loc='upper left',fontsize=6)
    fig.savefig(output/'calibration.png',dpi=300);plt.close(fig)


def history_figure(curves: pd.DataFrame,functionals: pd.DataFrame,history: Path,output: Path,bindings: dict):
    bindings[str(history/'bank_diagnostics.json')]=digest(history/'bank_diagnostics.json')
    diagnostics=json.loads((history/'bank_diagnostics.json').read_text())['cases']
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,4.3),layout='constrained',gridspec_kw={'width_ratios':[1.5,1,1]})
    for row,cid in enumerate(('C006','C013')):
        own=curves[(curves.case_id==cid)&(curves['mode']=='full')]
        truth=own[own.method=='evidence'];ax=axes[row,0]
        ax.fill_between(truth.trial,0,truth.true_present.astype(float),color='#e3e3e3',step='mid',label='True presence')
        for method,label in [('equal','Equal systems'),('evidence','Evidence weighted')]:
            values=own[own.method==method];ax.plot(values.trial,values.present,color=COLORS[method],lw=.9,label=label)
        ax.set(xlabel='Trial',ylabel='Presence probability',ylim=(-.03,1.03),title=cid+' · full history')
        if row==0:ax.legend(loc='lower right',fontsize=6)
        file=history/(cid+'_outer.npz');bindings[str(file)]=digest(file)
        with np.load(file) as z:w=np.sort(z['full_weights'])[::-1]
        diag=next(d for d in diagnostics if d['case_id']==cid);ax=axes[row,1]
        ax.plot(np.arange(1,len(w)+1),w,color=COLORS['evidence'],marker='.',ms=3,lw=.7)
        ax.axhline(1/len(w),color='#9b9b9b',ls=':',lw=.8)
        ax.set(xlabel='System rank',ylabel='System weight',ylim=(0,max(.12,float(w.max())*1.12)),
            title=f"{len(w)} systems · ESS {diag['outer_ess']:.1f}")
        ax=axes[row,2]
        selected=functionals[(functionals.case_id==cid)&(functionals['mode']=='full')&(functionals.function=='presence_trials')]
        for index,(method,label) in enumerate([('equal','Equal'),('evidence','Evidence')]):
            value=selected[selected.method==method].iloc[0]
            ax.plot([value.q05,value.q95],[index]*2,color=COLORS[method],lw=2)
            ax.scatter([value['mean']],[index],color=COLORS[method],s=20)
        value=selected.iloc[0];ax.axvline(value.true_value,color='#333333',ls='--',lw=.8)
        ax.set(yticks=[0,1],yticklabels=['Equal','Evidence'],ylim=(-.7,1.7),xlabel='Number of presence trials',title='Mean and 90% interval')
        for col in range(3):axes[row,col].text(-.18,1.07,'abcdef'[row*3+col],transform=axes[row,col].transAxes,weight='bold',fontsize=9)
    fig.savefig(output/'history.png',dpi=300);plt.close(fig)


def run(panel: Path,history: Path,output: Path):
    if output.exists():raise FileExistsError('Use a new figure directory')
    output.mkdir(parents=True);bindings={}
    calibration_data=table(panel,'calibration',bindings);curves=table(history,'curves',bindings);functionals=table(history,'functionals',bindings)
    calibration_figure(calibration_data,output);history_figure(curves,functionals,history,output,bindings)
    qa=dict(status='rendered',visual_review_record='visual_review.json',backend='python',width_inches=WIDTH,dpi=300,format='PNG',
        independent_paths=48,calibration_interval='pointwise 95% whole-path stratified bootstrap; 2000 draws',
        functional_interval='90% common-endpoint approximate coherent-path posterior',
        calibration_population='all registered contexts pooled; condition-specific acceptance remains in source CSV',
        exclusions='prefix rows and individual conditions remain in source tables; plotted modes/states set in figure contract',
        all_paths_retained=True,oral_used=False,sources=bindings,script_sha256=digest(Path(__file__)),
        files={f.name:digest(f) for f in output.glob('*.png')})
    (output/'QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--history',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    run(a.panel,a.history,a.output)
