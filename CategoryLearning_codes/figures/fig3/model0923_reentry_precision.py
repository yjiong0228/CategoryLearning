"""Plot every autonomous path and paired precision-retention effects.

Run --sources COMPLETED_208_PROBE COMPLETED_DEVELOPMENT_CHECK --output NEW_DIR.
Each fixed parameter point is retained in the overview and its subject panels.
All original and retained-precision paths are shown, with shared observed curves.
Error bars are one paired whole-path MCSE, not population confidence intervals.
PNG only, 183 mm width and 300 dpi for research discussion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_recalled_evidence_study import digest, read
from src.Bayesian_state.evaluation.model_0923_h_validation import distribution_scores, jackknife_se
from src.Bayesian_state.evaluation.model_0923_h_generation_uncertainty import path_jackknife


def load_sources(sources: list[Path]) -> tuple[dict[str, pd.DataFrame], dict]:
    tables={name:[] for name in ('scores','contrasts','curves','path_windows')};inputs={}
    seen=set()
    for source in sources:
        done=read(source/'completion.json');protocol=read(source/'protocol.json')
        if done['status']!='complete' or done.get('smoke') is not False or protocol['settings']['block_width']!=32:
            raise ValueError('Require completed scientific 32-trial-block comparisons')
        for file in (source/'completion.json',source/'protocol.json'):
            inputs[str(file.resolve())]=digest(file)
        for name,sha in done['artifacts'].items():
            if digest(source/name)!=sha:raise ValueError('Source summary changed')
            inputs[str((source/name).resolve())]=sha
        frames={name:pd.read_csv(source/f'{name}.csv') for name in tables}
        keys=set(zip(frames['scores'].subject,frames['scores'].point))
        if seen & keys:raise ValueError('Repeated subject/parameter point across sources')
        seen.update(keys)
        for name,frame in frames.items():tables[name].append(frame)
    tables={name:pd.concat(parts,ignore_index=True) for name,parts in tables.items()}
    curves,scores,contrasts=[tables[name] for name in ('curves','scores','contrasts')]
    for (subject,point),frame in curves.groupby(['subject','point']):
        if set(frame.variant)!={'RESET','RETAIN'}:raise ValueError('Need both mechanisms for every point')
        banks={};energies={};deletes={};late={}
        score=scores[(scores.subject==subject)&(scores.point==point)]
        contrast=contrasts[(contrasts.subject==subject)&(contrasts.point==point)]
        for variant,rows in frame.groupby('variant'):
            bank=rows.pivot(index='seed',columns='trial',values='accuracy').sort_index()
            if len(bank)!=64 or not np.isfinite(bank.to_numpy()).all() or np.any((bank<0)|(bank>1)):
                raise ValueError('Need all 64 valid autonomous paths')
            np.testing.assert_array_equal(bank.columns,np.arange(32,int(bank.columns.max())+1,32))
            observed=rows.groupby('trial')['observed']
            if not observed.nunique().eq(1).all():raise ValueError('Observed curve differs across paths')
            human=observed.first().loc[bank.columns].to_numpy()
            values=bank.to_numpy();energies[variant]=distribution_scores(values,human)['energy']
            deletes[variant]=path_jackknife(values,human)['energy'];banks[variant]=bank
            late[variant]=values[:,-4:].mean(axis=1)
            for metric,value in [('energy',energies[variant]),('accuracy',late[variant].mean())]:
                window='whole' if metric=='energy' else 'late'
                expected=score[(score.variant==variant)&(score.metric==metric)&(score.window==window)].value.item()
                np.testing.assert_allclose(value,expected,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(banks['RESET'].index,banks['RETAIN'].index)
        for metric,value,se in [('energy',energies['RETAIN']-energies['RESET'],jackknife_se(deletes['RETAIN']-deletes['RESET'])),
                               ('accuracy',float((late['RETAIN']-late['RESET']).mean()),float((late['RETAIN']-late['RESET']).std(ddof=1)/8))]:
            row=contrast[(contrast.metric==metric)&(contrast.window==('whole' if metric=='energy' else 'late'))].iloc[0]
            np.testing.assert_allclose([value,se],[row.difference,row.mcse],atol=1e-12,rtol=0)
    return tables,inputs


def plot(sources: list[Path],output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new precision-retention figure directory')
    tables,inputs=load_sources(sources);curves,scores,contrasts=[tables[k] for k in ('curves','scores','contrasts')]
    identities=curves[['subject','point','capacity']].drop_duplicates().sort_values(['subject','capacity'],ascending=[True,False])
    contract=dict(claim='Precision retention improves autonomous learning in the 208 reentry case; effects are small in the other development cases.',
        archetype='quantitative grid',backend='python',width_mm=183,dpi=300,formats=['PNG'],
        overview='All fixed points; paired Energy and late128 accuracy differences',
        subject_panels='Each row a fixed point, RESET then RETAIN; every 64 path curves, means and observed curve',
        statistics='One paired whole-path jackknife MCSE; 64 matched seeds per fixed point; four development subjects; two points for208',
        data_selection='All points and all paths in supplied completed runs; no outcome-based exclusions',
        interpretation='Fixed parameters, own-choice feedback, development only; no behavior likelihood or oral claim',
        provenance='107/306: previous POOL coordinates applied to SEARCH.207: joint SEARCH fit.208: original plus behavior-grid point.',
        preflight_exceptions={'EXPORT-VECTOR':'Repository PNG-only rule','EXPORT-RASTER':'Repository PNG-only rule','RASTER-DPI':'300 dpi discussion draft'})
    output.mkdir(parents=True);(output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    for name,frame in tables.items():frame.to_csv(output/f'{name}_source.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False,
        'svg.fonttype':'none','pdf.fonttype':42})
    colors={'RESET':'#6A7E8A','RETAIN':'#AA7655'}
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,90/25.4))
    fig.subplots_adjust(left=.15,right=.96,bottom=.24,top=.80,wspace=.52)
    for i,(metric,window,title) in enumerate([('energy','whole','a  Whole-trajectory Energy'),('accuracy','late','b  Last-128-trial accuracy')]):
        ax=axes[i]
        for y,row in enumerate(identities.itertuples()):
            value=contrasts[(contrasts.subject==row.subject)&(contrasts.point==row.point)&(contrasts.metric==metric)&(contrasts.window==window)].iloc[0]
            ax.errorbar(value.difference,y,xerr=value.mcse,fmt='o',color='#744B32',ms=3.5,capsize=2,lw=.8)
        ax.axvline(0,color='.65',lw=.7)
        ax.set_yticks(range(len(identities)),[f'{r.subject}: K={r.capacity}' for r in identities.itertuples()])
        ax.set(ylim=(len(identities)-.4,-.6),title=title,
               xlabel='RETAIN − RESET (lower is better)' if metric=='energy' else 'RETAIN − RESET (higher is better)')
    fig.suptitle('Retaining learned rule precision: fixed-parameter development checks',fontsize=9,y=.95)
    fig.text(.5,.10,'64 paired autonomous paths per point; bars: 1 paired MCSE. Two 208 points belong to the same participant.',ha='center',fontsize=6)
    fig.text(.5,.045,'Parameter sources differ across participants; each contrast changes only precision retention.',ha='center',fontsize=6)
    fig.savefig(output/'overview.png',dpi=300);plt.close(fig)
    images=['overview.png']
    for subject,points in identities.groupby('subject',sort=True):
        rows=len(points);fig,axes=plt.subplots(rows,2,squeeze=False,figsize=(183/25.4,(65*rows+40)/25.4))
        fig.subplots_adjust(left=.09,right=.98,bottom=.20 if rows==1 else .14,top=.78 if rows==1 else .86,wspace=.28,hspace=.42)
        for i,point in enumerate(points.itertuples()):
            for col,variant in enumerate(('RESET','RETAIN')):
                ax=axes[i,col];data=curves[(curves.subject==subject)&(curves.point==point.point)&(curves.variant==variant)]
                for _,path in data.groupby('seed'):ax.plot(path.trial,path.accuracy,color=colors[variant],alpha=.13,lw=.45)
                means=data.groupby('trial').agg(mean=('accuracy','mean'),observed=('observed','first'))
                ax.plot(means.index,means['mean'],color=colors[variant],lw=1.8,label='Autonomous mean')
                ax.plot(means.index,means.observed,color='#202020',ls='--',lw=1.2,label='Participant')
                energy=scores[(scores.subject==subject)&(scores.point==point.point)&(scores.variant==variant)&(scores.metric=='energy')].value.item()
                ax.set(xlim=(1,int(data.trial.max())),ylim=(0,1.02),xlabel='Trial',ylabel='32-trial block accuracy',
                    title=f'{chr(97+2*i+col)}  K={point.capacity}, {variant}; Energy={energy:.3f}')
        axes[0,0].legend(loc='lower right')
        fig.suptitle(f'Subject {subject}: every original and retained-precision path',fontsize=9,y=.96)
        fig.text(.5,.09 if rows==1 else .07,'64 paths per panel; thin lines: all realizations; thick lines: means; dashed line: observed behavior.',ha='center',fontsize=6)
        fig.text(.5,.04 if rows==1 else .035,'Same fixed parameters and paired seeds; each autonomous path generates its own choices and feedback.',ha='center',fontsize=6)
        name=f'subject_{subject}_paths.png';fig.savefig(output/name,dpi=300);plt.close(fig);images.append(name)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>'
        '<h1>规则重新进入时保留精度：完整自主路径与开发复核</h1>'
        '<p>每个参数点比较同样的64个路径种子，全部路径保留。RESET在再次进入时重新初始化精度，RETAIN恢复最近在场精度。'
        '概览误差棒为一个配对模拟标准误。208的两个参数点属于同一被试，四人均为开发样本。</p>'
        '<p>107与306使用此前POOL参数下的SEARCH，207使用近期SEARCH联合拟合参数，208使用原SEARCH参数与行为网格的容量2参数。'
        '每项差值均在同一参数点内计算。行为似然检验和新机制口述检验分别推进。</p>'
        ''.join(f'<img src="{name}" alt="精度保持自主路径对照">' for name in images)+'</html>')
    for name,sha in inputs.items():
        if digest(Path(name))!=sha:raise ValueError('Figure input changed')
    (output/'manifest.json').write_text(json.dumps(dict(status='complete',input_sha256=inputs,
        source_sha256={str(Path(__file__).resolve()):digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir()}),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.sources,args.output)
