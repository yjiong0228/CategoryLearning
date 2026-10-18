"""Display all paired paths for the three-condition online-evidence extension.

Use --source SUMMARY --validation INDEPENDENT_AUDIT --output NEW_DIRECTORY.
Top panels show every autonomous learning curve and its mean; bottom panels
show empirical cumulative distributions of late correct-rule contribution.
All 384 paths are retained. Parameters and their original provenance remain
fixed. Export is PNG only, 183 x 125 mm at 300 dpi.
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


def plot(source: Path, validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new development figure destination')
    inputs, records = {}, []
    for folder, name in ((source,'manifest.json'),(validation,'validation.json')):
        file = folder / name
        record = json.loads(file.read_text())
        inputs[str(file.resolve())] = digest(file)
        for field in ('inputs_sha256','source_sha256'):
            for path, sha in record[field].items():
                if digest(Path(path)) != sha:
                    raise ValueError('Changed scientific input: '+path)
                inputs[path] = sha
        records.append(record)
    if any(r['status'] != 'complete' or r['paths'] != 384 or r['subjects'] != [107,207,306] for r in records):
        raise ValueError('Require all independently checked development paths')
    tables = {}
    for name, sha in records[0]['artifacts'].items():
        file = source / name
        if digest(file) != sha or records[1]['inputs_sha256'].get(str(file.resolve())) != sha:
            raise ValueError('Unaudited figure table: '+name)
        inputs[str(file.resolve())] = sha
        tables[file.stem] = pd.read_csv(file)
    contract = dict(conclusion='Online H evidence increases late rule use in207/306;107 changes little, and early learning can fit less well.',
        evidence='All learning paths and observed curves above; all late contribution values in cumulative distributions below.',
        archetype='quantitative grid',backend='python',width_mm=183,height_mm=125,dpi=300,formats=['PNG'],
        mapping='subject -> column,trial -> upper x,32-trial accuracy -> upper y,schedule -> color/line; late contribution -> lower x,empirical fraction -> lower y.',
        participant_n=3,computational_paths_per_cell=64,cells=6,
        center='Arithmetic mean over all64paths; cumulative distributions retain every path including zero contribution.',
        uncertainty='Whole-Energy annotation is online minus baseline +/- one paired whole-path Monte Carlo SE.',
        split='Original chronological boundaries128/192,672/960,512/768; all are development comparisons.',
        parameter_provenance='107/306 use earlier POOL points applied to SEARCH;207 uses its joint SEARCH refit point.',
        data_use='No parameter fitting, oral input or reserved subjects. Both variants retain learned precision.',
        exclusions='None: all384paths and7680curve points used.',
        transformations='Non-overlapping32-trial accuracy blocks and exact empirical CDF; no smoothing or downsampling.',
        hypothesis_tests='None',multiple_comparison_correction='Not applicable',
        adaptation='Style-only inheritance from Fig3; subject-specific trajectories and all-path contribution distributions.',
        export_exceptions={'EXPORT-VECTOR':'Repository PNG-only instruction','EXPORT-RASTER':'Repository PNG-only instruction','RASTER-DPI':'300dpi discussion output'})
    output.mkdir(parents=True)
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    for name in tables:
        shutil.copy2(source/(name+'.csv'),output/(name+'.csv'))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
    colors={'off':'#68778B','online':'#257B89'}
    styles={'off':'--','online':'-'}
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,125/25.4))
    fig.subplots_adjust(left=.085,right=.98,bottom=.22,top=.82,hspace=.55,wspace=.40)
    means,cdf=[] ,[]
    for col,subject in enumerate((107,207,306)):
        scores=tables['scores'].query('subject==@subject')
        n=int(scores.query('phase=="whole"').trials.iloc[0])
        cut=int(scores.query('phase=="training"').trials.iloc[0])
        condition=int(scores.condition.iloc[0])
        own=tables['path_blocks'].query('subject==@subject')
        observed=own.groupby('trial')['observed'].agg(['min','max'])
        np.testing.assert_array_equal(observed['min'],observed['max'])
        for schedule,color in colors.items():
            frame=own[own.schedule==schedule].pivot(index='seed',columns='trial',values='accuracy')
            values=frame.to_numpy(float)
            if values.shape!=(64,n//32) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
                raise ValueError('Incomplete or out-of-range path matrix')
            axes[0,col].plot(frame.columns,values.T,color=color,lw=.4,alpha=.07)
            axes[0,col].plot(frame.columns,values.mean(0),color=color,lw=1.3,ls=styles[schedule])
            means.extend(dict(subject=subject,schedule=schedule,trial=int(t),mean=float(v),paths=64)
                         for t,v in zip(frame.columns,values.mean(0)))
            paths=tables['paths'].query('subject==@subject and schedule==@schedule').sort_values('target_late_contribution')
            x=paths.target_late_contribution.to_numpy(float)
            if len(x)!=64 or paths.seed.nunique()!=64 or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
                raise ValueError('Incomplete contribution distribution')
            y=np.arange(1,65)/64
            axes[1,col].step(np.r_[0,x,1],np.r_[0,y,1],where='post',color=color,lw=1.3,ls=styles[schedule])
            cdf.extend(dict(subject=subject,schedule=schedule,seed=int(seed),contribution=value,cumulative=fraction)
                       for seed,value,fraction in zip(paths.seed,x,y))
        axes[0,col].plot(observed.index,observed['min'],color='#222222',lw=1.0)
        axes[0,col].axvline(cut,color='#999999',lw=.6,ls='--')
        axes[0,col].set(title=f'Participant {subject} | condition {condition}',xlabel='Trial',ylabel='Accuracy',
                        xlim=(0,n),ylim=(0,1),xticks=[0,n//2,n])
        effect=tables['effects'].query('subject==@subject and phase=="whole"').iloc[0]
        axes[1,col].set(title=f'Energy Δ={effect.delta_energy:+.4f} ± {effect.delta_energy_mc_se:.4f}',
                        xlabel='Late correct-rule contribution',ylabel='Fraction of paths',xlim=(0,1),ylim=(0,1))
    for i,ax in enumerate(axes.flat):
        ax.text(-.11,1.06,chr(97+i),transform=ax.transAxes,fontsize=8,fontweight='bold')
    fig.suptitle('Online evidence belief across three existing development participants',fontsize=9,y=.97)
    fig.text(.5,.92,'Parameters and precision storage are fixed within participant; all 64 paired paths per variant are shown.',ha='center',fontsize=6.5)
    handles=[Line2D([],[],color=colors[s],ls=styles[s],lw=1.3,label=label) for s,label in [('off','BASE'),('online','Online evidence')]]
    handles.append(Line2D([],[],color='#222222',lw=1,label='Participant'))
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.105))
    fig.text(.5,.070,'Faint paths and bold means: 32-trial accuracy. Lower panels: all late contribution values; late = final 128 trials.',ha='center',fontsize=6.3)
    fig.text(.5,.030,'107/306: earlier POOL parameters applied to SEARCH. 207: SEARCH refit. No oral data or parameter selection.',ha='center',fontsize=6.3)
    fig.savefig(output/'online_evidence_development.png',dpi=300)
    plt.close(fig)
    pd.DataFrame(means).to_csv(output/'plot_means.csv',index=False)
    pd.DataFrame(cdf).to_csv(output/'plot_cdf.csv',index=False)
    caption=('每人两方案使用同一组64个种子，参数与精度保持机制固定。上排背景保留全部自主路径，粗线为均值，黑线为被试；'
        '每点汇总32题，竖线表示原时间分界。下排显示末128题正确规则贡献的全部路径累积分布：横坐标越大，规则贡献越高。'
        '标题的Energy差为ONLINE减BASE及一个配对模拟标准误，负值表示整段生成分布更接近被试。'
        '107、306沿用早期POOL参数应用到SEARCH的点，207沿用SEARCH联合重拟合的点。三人均属于开发样本。')
    page=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>在线证据的三条件扩展</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}td,th{padding:6px}</style>'
        '<h1>在线证据如何影响三名被试的自主过程</h1><p>'+caption+'</p>'
        '<img src="online_evidence_development.png" alt="三人全部轨迹及正确规则贡献分布">'
        +tables['scores'].query('phase=="whole"')[['subject','schedule','energy','late_accuracy','observed_late',
            'target_late_present','target_late_contribution']].to_html(index=False,float_format=lambda x:f'{x:.4f}')
        +'<p><a href="effects.csv">分时段配对评分</a> · <a href="path_effects.csv">发现与保留过程</a> · <a href="parameters.csv">全部固定参数</a></p></html>')
    (output/'index.html').write_text(page)
    for path,sha in inputs.items():
        if digest(Path(path))!=sha:raise ValueError('Scientific input changed during drawing')
    shutil.copy2(__file__,output/'plot_source.py')
    (output/'manifest.json').write_text(json.dumps(dict(status='complete',input_sha256=inputs,
        source_sha256={str(Path(__file__).resolve()):digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','validation','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();plot(args.source,args.validation,args.output)
