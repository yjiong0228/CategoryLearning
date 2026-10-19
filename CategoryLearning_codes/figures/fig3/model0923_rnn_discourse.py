"""Plot every frozen107 contrast report with signed actual-context effects.

Contract: show where model-specific feature effects support or oppose the chosen
response, without inferring a feature's role from sentence position. Quantitative
grid: three models, common color scale, all 33 contrast reports split by the
original training/later boundary. Python only; 183 mm wide, PNG at 300 dpi for
development review. This is one participant, one preselected H filter per arm,
and one behavior-selected RNN. No significance tests or population uncertainty.
Unmentioned coordinates are gray; rounded displayed zeros retain full source
precision. H uses original choice-conditioned weights. All source rows and the
prior-weighted alternative remain in the supplied analysis directory.

Use --source COMPLETED_DISCOURSE --output NEW_DIRECTORY.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_target_state_recovery import digest, read, verify, write

FEATURES = ('neck', 'head', 'leg', 'tail')


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new discourse figure directory')
    done = read(source/'completion.json')
    if done['status']!='complete' or done['subject']!=107 or done['parameters_changed'] or done['oral_used_for_fit']:
        raise ValueError('Require completed unchanged107 discourse comparison')
    inputs = {str(Path(__file__).resolve()):digest(Path(__file__)),str((source/'completion.json').resolve()):digest(source/'completion.json')}
    for name,sha in done['artifacts_sha256'].items():
        path=source/name
        if digest(path)!=sha:raise ValueError('Analysis source table changed')
        inputs[str(path.resolve())]=sha
    coverage=pd.read_csv(source/'coverage.csv',keep_default_na=False)
    texts=coverage[coverage.group=='all_named_or_collective'].copy()
    selected=texts[texts.has_contrast].sort_values('state_trial')
    if len(texts)!=192 or len(selected)!=33 or selected.groupby('phase').size().to_dict()!={'later':15,'training':18}:
        raise ValueError('Original contrast-report coverage changed')
    features=pd.read_csv(source/'paired_feature_trials.csv')
    features=features[(features.group=='all_named_or_collective')&(features.h_weighting=='choice_conditioned')]
    points=[]
    for row in selected.itertuples():
        for feature in FEATURES:
            matched=features[(features.state_trial==row.state_trial)&(features.feature==feature)]
            values={}
            if len(matched):
                if len(matched)!=2 or set(matched.h_arm)!={'SEARCH','SELECT_SEARCH'}:
                    raise ValueError('Need the same two H references and RNN value')
                np.testing.assert_array_equal(matched.actual_effect_rnn, np.repeat(matched.actual_effect_rnn.iloc[0],2))
                values={'RNN':float(matched.actual_effect_rnn.iloc[0]),
                        'MIX':float(matched.loc[matched.h_arm=='SEARCH','actual_effect_h'].iloc[0]),
                        'SELECT':float(matched.loc[matched.h_arm=='SELECT_SEARCH','actual_effect_h'].iloc[0])}
            for model in ('MIX','SELECT','RNN'):
                points.append(dict(state_trial=row.state_trial,iSession=row.iSession,iTrial=row.iTrial,
                    phase=row.phase,text=row.text,feature=feature,model=model,reported=bool(values),
                    actual_effect=values.get(model,np.nan)))
    points=pd.DataFrame(points)
    # Resolve a system CJK font by its family name; no machine-specific path is saved in source.
    cjk=font_manager.FontProperties(fname=font_manager.findfont('Noto Sans CJK JP',fallback_to_default=False))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.linewidth':.5,'svg.fonttype':'none','pdf.fonttype':42})
    output.mkdir(parents=True)
    points.to_csv(output/'plot_source.csv',index=False)
    cmap=plt.get_cmap('PuOr').copy();cmap.set_bad('#eeeeee')
    limits=Normalize(vmin=-1,vmax=1)
    for phase, phase_label in [('training','训练段'),('later','后段')]:
        frame=selected[selected.phase==phase];order=frame.state_trial.tolist();n=len(order)
        fig=plt.figure(figsize=(183/25.4,150/25.4))
        grid=fig.add_gridspec(1,3,left=.525,right=.982,bottom=.19,top=.87,wspace=.15)
        for column,model in enumerate(('MIX','SELECT','RNN')):
            ax=fig.add_subplot(grid[0,column])
            values=points[(points.phase==phase)&(points.model==model)].pivot(index='state_trial',columns='feature',values='actual_effect').reindex(index=order,columns=FEATURES).to_numpy()
            image=ax.imshow(values,cmap=cmap,norm=limits,aspect='auto',interpolation='nearest')
            ax.set_xticks(range(4),['脖','头','腿','尾'],fontproperties=cjk,fontsize=7)
            ax.tick_params(axis='both',length=0,pad=3)
            ax.set_yticks(range(n),frame.apply(lambda r:f"{int(r.state_trial)+1:03d}  {r.text}",axis=1) if column==0 else ['']*n,
                          fontproperties=cjk,fontsize=5.7)
            ax.set_title(('a  H · MIX','b  H · SELECT','c  RNN')[column],fontsize=8,fontweight='bold',pad=8)
            ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,n,1),minor=True)
            ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',length=0)
            for spine in ax.spines.values():spine.set_visible(False)
            for r in range(n):
                for c in range(4):
                    value=values[r,c]
                    if np.isfinite(value):
                        label='0.00' if abs(value)<.005 else f'{value:.2f}'
                        ax.text(c,r,label,ha='center',va='center',fontsize=5.2,
                                color='white' if abs(value)>.6 else '#222222')
        fig.text(.025,.965,'转折句中的即时特征作用',fontproperties=cjk,fontsize=11,weight='bold')
        fig.text(.025,.925,f'107 · {phase_label}全部 {n} 条转折句 · 原参数与文本分组固定',fontproperties=cjk,fontsize=7)
        color_axis=fig.add_axes([.57,.115,.36,.018])
        colorbar=fig.colorbar(image,cax=color_axis,orientation='horizontal',ticks=[-1,-.5,0,.5,1])
        colorbar.ax.tick_params(labelsize=6,length=2)
        fig.text(.025,.115,'正值：当前特征支持所选反应；负值：降低其概率。',fontproperties=cjk,fontsize=6.3)
        fig.text(.025,.074,'每格 = 保留当前特征值的选择概率 − 单独反射该特征后的概率。',fontproperties=cjk,fontsize=6.3)
        fig.text(.025,.044,'灰格为未提及部位；显示值四舍五入至两位小数，原值见源表。',fontproperties=cjk,fontsize=6.3)
        fig.text(.025,.018,'H：每方案一次预选滤波，使用选择条件权重；RNN：行为选定参数。开发诊断。',fontproperties=cjk,fontsize=6.3)
        fig.savefig(output/(phase+'.png'),dpi=300,facecolor='white')
        plt.close(fig)
    verify(inputs)
    write(output/'validation.json',dict(status='complete',inputs_sha256=inputs,source_rows=len(points),
        original_trials=192,nonempty_reports=190,contrast_reports=33,training_contrasts=18,later_contrasts=15,
        exclusions='159 non-contrast trial rows are outside this text-structure figure; all remain in analysis coverage',
        full_report_coverage_preserved=True,backend='python',png_only=True,dimensions_mm=[183,150],dpi=300,
        measures='Signed actual-context effect; common [-1,1] scale; rounded labels only',
        scope='One development participant; one preselected H filter per arm; no inferential error bars or tests',
        outputs_sha256={p.name:digest(p) for p in output.iterdir()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();plot(args.source,args.output)
