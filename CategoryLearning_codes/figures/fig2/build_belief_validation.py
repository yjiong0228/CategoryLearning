"""Fig2: choice fit and current-report validation of the latest fitted cohort.

Uses saved states only; no cognitive fitting, simulation, or carried oral reports.
Run from the repository root with a NEW --output directory. PNG only.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..fig1.build_learning_story import heading
from ..fig3.bottleneck_analysis import ROOT, sha256
from ..fig3.build_nine_subject_figures import style, TASK_COLORS, COLORS
from src.Bayesian_state.hypothesis_space.observation_model import ContinuousPartition
from src.Bayesian_state.evaluation.oral.scoring import OralAlignmentScoringMixin

DEFAULT_SOURCE='results/model_0826/fig34_twelve_subjects_20260921_v1'
ORDER=[102,104,118,122,307,314,315,328,206,215,221,222]


def compatible(belief: np.ndarray, oral: np.ndarray) -> np.ndarray:
    """Expected relative report likelihood, not rule-identification accuracy.

    Oral weights use a uniform hypothesis prior, so dividing by their maximum
    recovers relative report likelihoods, retaining tied/partial descriptions.
    Caller supplies only valid current reports.
    """
    q,o=np.asarray(belief,float),np.asarray(oral,float)
    if q.ndim!=2 or q.shape!=o.shape or not np.isfinite(q).all() or not np.isfinite(o).all():
        raise ValueError('Need aligned, finite two-dimensional distributions')
    if (q<0).any() or (o<0).any():raise ValueError('Negative probability')
    np.testing.assert_allclose(q.sum(1),1,atol=1e-7)
    np.testing.assert_allclose(o.sum(1),1,atol=1e-7)
    return np.sum(q*(o/o.max(1,keepdims=True)),axis=1)


def project(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Aggregate a batch over the core encoder's category-equivalence groups."""
    groups=np.asarray(groups,int)
    if values.shape[1]!=len(groups) or (groups<0).any():raise ValueError('Invalid groups')
    mapping=np.eye(groups.max()+1)[groups]
    return values@mapping


def report_pairs(frame: pd.DataFrame, q: np.ndarray, oral: np.ndarray,
                 valid: np.ndarray, groups: dict[int,np.ndarray]) -> pd.DataFrame:
    """Adjacent reports about the SAME category; invalid reports break its chain."""
    rows=[];previous={};last_session=None
    projections={c:project(q,g) for c,g in groups.items()}
    oral_projections={c:project(np.nan_to_num(oral,nan=0.),g) for c,g in groups.items()}
    for i,row in enumerate(frame.itertuples()):
        session=int(row.iSession);choice=int(row.choice)
        if session!=last_session:previous={};last_session=session
        if not valid[i]:previous.pop(choice,None);continue
        j=previous.get(choice)
        if j is not None and 1<=i-j<=32:
            gap=i-j
            rows.append({'subject':int(row.iSub),'task':int(row.task),'trial':i+1,
                         'previous_trial':j+1,'session':session,'choice':choice,'gap':gap,
                         'oral_tv':.5*np.abs(oral_projections[choice][i]-oral_projections[choice][j]).sum(),
                         'belief_tv_per_trial':.5*np.abs(projections[choice][i]-projections[choice][j]).sum()/gap})
        previous[choice]=i
    return pd.DataFrame(rows,columns=['subject','task','trial','previous_trial','session','choice','gap','oral_tv','belief_tv_per_trial'])


def matched_change(pairs: pd.DataFrame, threshold: float=.5) -> dict:
    """Within-person exact-gap matching; weights do not inflate participant n."""
    stable=[];changed=[];weights=[]
    for _,g in pairs.groupby('gap'):
        a=g[g.oral_tv.le(.1)].belief_tv_per_trial
        b=g[g.oral_tv.ge(threshold)].belief_tv_per_trial
        if len(a) and len(b):
            stable.append(a.mean());changed.append(b.mean());weights.append(min(len(a),len(b)))
    if not weights:return {'stable_change':np.nan,'report_change':np.nan,'change_difference':np.nan,'matched_weight':0}
    a=float(np.average(stable,weights=weights));b=float(np.average(changed,weights=weights))
    return {'stable_change':a,'report_change':b,'change_difference':b-a,'matched_weight':sum(weights)}


def analyze(source: Path, output: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    provenance_path=source/'base_analysis/manifest.json'
    provenance=json.loads(provenance_path.read_text())
    if provenance['config']['oral_sigma']!=.05:raise ValueError('Oral encoder scale changed')
    raw_path=ROOT/'data/exp123/processed/Task2_processed.csv'
    if sha256(raw_path)!=provenance['input_sha256'][str(raw_path.relative_to(ROOT))]:
        raise ValueError('Current data differs from the fitted analysis source')
    d=pd.read_csv(source/'base_analysis/trials.csv')
    people=pd.read_csv(source/'base_analysis/subjects.csv').set_index('subject').loc[ORDER]
    if len(d)!=7936 or set(d.iSub.unique())!=set(ORDER):raise ValueError('Unexpected fitted cohort')
    partitions={n:ContinuousPartition(4,n,similarity_n_samples=1) for n in (2,4)}
    equivalence={n:{c:OralAlignmentScoringMixin._oral_equivalence_groups(p,c)[0]
                    for c in range(1,n+1)} for n,p in partitions.items()}
    inputs=[provenance_path,raw_path,source/'base_analysis/trials.csv',source/'base_analysis/subjects.csv',Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]/'fig1/LEARNING_STORY_DESIGN.md']
    summaries=[];trial_tables=[];seeds=[];pairs_all=[];sensitivity=[];calibration=[]
    for sid in ORDER:
        g=d[d.iSub.eq(sid)].reset_index(drop=True);info=people.loc[sid]
        path=source/f'base_analysis/S{sid}_rule_distributions.npz';inputs.append(path)
        with np.load(path,allow_pickle=False) as z:
            q=z['belief'];o=z['instantaneous_oral'];oral_valid=z['oral_valid'].astype(bool)
        np.testing.assert_array_equal(g.trial,np.arange(1,len(g)+1))
        np.testing.assert_array_equal(oral_valid,g.oral_valid)
        selected=[];predictions=[];mask=None
        for variant,repeats in [('selected',8),('alternative',4)]:
            variant_seeds=[]
            for repeat in range(repeats):
                p=source/f'states/S{sid}/{variant}_{repeat:02}.npz';inputs.append(p)
                with np.load(p,allow_pickle=False) as z:
                    qq=z['marginal_prior'];pred=z['pred_category_probs'];seed=int(z['seed'])
                    current=z['valid_trial_mask'].astype(bool)&z['score_trial_mask'].astype(bool)
                    np.testing.assert_array_equal(z['observed_choice'],g.choice)
                    np.testing.assert_array_equal(z['observed_feedback'],g.feedback)
                    np.testing.assert_array_equal(z['true_category_index'],g.category.to_numpy()-1)
                if mask is None:mask=current
                np.testing.assert_array_equal(mask,current)
                np.testing.assert_allclose(qq.sum(1),1,atol=1e-7)
                np.testing.assert_allclose(pred.sum(1),1,atol=1e-7)
                v=mask&oral_valid;static=np.broadcast_to(qq[mask].mean(0),qq[v].shape)
                cc=compatible(qq[v],o[v]);ss=compatible(static,o[v])
                seeds.append({'subject':sid,'task':int(info.task),'variant':variant,'repeat':repeat,'seed':seed,
                              'compatibility':cc.mean(),'dynamic_gain':(cc-ss).mean()})
                variant_seeds.append(seed)
                if variant=='selected':selected.append(qq);predictions.append(pred)
            if len(set(variant_seeds))!=repeats:raise ValueError('Repeated seed')
        np.testing.assert_allclose(q,np.mean(selected,axis=0),atol=1e-7)
        np.testing.assert_array_equal(mask,g.scored)
        v=mask&oral_valid;uniform=np.ones_like(q[v])/q.shape[1]
        static=np.broadcast_to(q[mask].mean(0),q[v].shape)
        cc=compatible(q[v],o[v]);uu=compatible(uniform,o[v]);ss=compatible(static,o[v])
        table=g[['iSub','condition','task','iSession','trial','choice','category','feedback','text','oral_center','correct','oral_valid','oral_target_top','belief']].copy()
        table['scored']=mask
        for key,values in [('compatibility',cc),('uniform_compatibility',uu),('static_compatibility',ss),('dynamic_gain',cc-ss)]:
            table[key]=np.nan;table.loc[v,key]=values
        pred=np.mean(predictions,axis=0)
        pcorrect=pred[np.arange(len(g)),g.category.to_numpy(int)-1]
        obs=g.choice.eq(g.category).to_numpy(float)
        table['model_correct']=pcorrect;table['label_correct']=obs
        table['belief_seed_low']=np.min(selected,axis=0)[:,int(info.target)]
        table['belief_seed_high']=np.max(selected,axis=0)[:,int(info.target)]
        # Same bins for everyone. Points are descriptive participant/bin means.
        bins=np.minimum((pcorrect*5).astype(int),4)
        for b in range(5):
            m=mask&(bins==b)
            if m.any():calibration.append({'subject':sid,'task':int(info.task),'bin':b,'n':int(m.sum()),
                                          'predicted':pcorrect[m].mean(),'observed':obs[m].mean()})
        pairs=report_pairs(g,q,o,v,equivalence[2 if info.condition==1 else 4]);pairs_all.append(pairs)
        row={'subject':sid,'task':int(info.task),'n_trials':len(g),'n_scored':int(mask.sum()),
             'n_reports':int(v.sum()),'compatibility':cc.mean(),'uniform_compatibility':uu.mean(),
             'static_compatibility':ss.mean(),'dynamic_gain':(cc-ss).mean(),
             'choice_accuracy':obs[mask].mean(),'predicted_accuracy':pcorrect[mask].mean(),
             'n_report_pairs':len(pairs),**matched_change(pairs)}
        summaries.append(row);trial_tables.append(table)
        for threshold in (.25,.5,.75):
            sensitivity.append({'subject':sid,'task':int(info.task),'oral_change_threshold':threshold,**matched_change(pairs,threshold)})
    subjects=pd.DataFrame(summaries);trials=pd.concat(trial_tables,ignore_index=True);seed_table=pd.DataFrame(seeds)
    tables={'subjects':subjects,'trials':trials,'seeds':seed_table,'report_pairs':pd.concat(pairs_all,ignore_index=True),
            'change_sensitivity':pd.DataFrame(sensitivity),'calibration':pd.DataFrame(calibration)}
    output.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():table.to_csv(output/f'{name}.csv',index=False)
    manifest={'n_participants':12,'n_trials':len(d),'n_scored':int(subjects.n_scored.sum()),
              'n_reports':int(subjects.n_reports.sum()),'order':ORDER,'selected_repeats':8,'alternative_repeats':4,
              'oral_sigma':.05,'prediction_timing':'pre-choice','report_timing':'after choice, before feedback',
              'fit_scope':'full-sequence choices; oral not fitted; hypothesis catalog informed by oral',
              'uncertainty':'range across saved PF seeds; not confidence intervals',
              'comparison':'uniform-rule and within-person time-average Q; not fitted alternative models',
              'dpi':450,'width_mm':183,'height_mm':235,'python':platform.python_version(),
              'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,
              'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in inputs}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text(
        '# Fig2 当前口述检验草稿\n\n'
        f'12人，{len(d):,}个原始试次；共同mask后{int(subjects.n_scored.sum()):,}个试次、'
        f'{int(subjects.n_reports.sum()):,}条有效当前口述。\n\n'
        '- 主图：Figure2_belief_validation.png。\n'
        '- 全12人完整轨迹：FigureS2_all_participants.png。\n'
        '- PF重复、近优参数和阈值敏感性：FigureS2_validation_sensitivity.png。\n\n'
        '## 当前结果\n\n'
        f'- {int(subjects.compatibility.gt(subjects.uniform_compatibility).sum())}/12人口述兼容度高于均匀规则参照。\n'
        f'- {int(subjects.dynamic_gain.gt(0).sum())}/12人的时变信念比本人的整段平均信念兼容度高；微小正值不等于稳健优势。\n'
        f'- {int(subjects.change_difference.gt(0).sum())}/12人在口述变化期间有更大的模型信念变化（相同间隔匹配）。\n\n'
        'S122、S314的口述对应弱，保留在主图；S215、S328的目标信念有明显PF波动。\n'
        '不能据此宣称优于替代认知模型或证明顿悟机制。所有拟合仍带有原始停止诊断未解决的限制。\n\n'
        '## 数据与读图\n\n'
        'subjects.csv：所有个体汇总；trials.csv：当前报告、指标、行为与目标信念；seeds.csv：8+4重放的汇总；'
        'report_pairs.csv：同类别、同session、间隔≤32的口述对；change_sensitivity.csv：0.25/0.5/0.75阈值；'
        'calibration.csv：逐被试概率分箱。manifest.json保存输入hash、版本与时序。\n\n'
        '兼容度 C=Σ Q(h) O(h)/max O。它是相对报告似然的期望，不是规则识别正确率。'
        '静态参照=该被试所有有效试次Q的平均，用整段信息，仅作为时变信息的诊断。'
        '正式指标排除每人第1试次；轨迹图展示全部保存状态。'
        'Q位于选择前，报告位于选择后、反馈前；同一编码器使用报告实际所属的选择类别。'
        '口述没有进入参数拟合，但规则目录曾参考口述，且参数用整段选择估计。\n\n'
        '详细设计与统计定义见代码目录 fig1/LEARNING_STORY_DESIGN.md 和 fig2/VALIDATION_STORY.md。\n',encoding='utf-8')
    print(subjects[['subject','n_reports','compatibility','uniform_compatibility','dynamic_gain','change_difference']].to_string(index=False))
    return subjects,trials,seed_table


def framework(fig) -> None:
    heading(fig,.075,.915,'a','Reconstruct beliefs from choices; test them against verbal reports')
    ax=fig.add_axes([.075,.753,.90,.148]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    labels=[('Consider','A limited set of rules','capacity'),('Evaluate','Retain and update evidence','memory'),
            ('Search / revise','Use feedback and related rules','search + transfer'),('Choose','Convert belief into action','readout')]
    for j,(name,desc,module) in enumerate(labels):
        x=j*.255
        ax.plot([x,x+.225],[.72,.72],lw=2,color='#557A83')
        ax.text(x,.79,name,fontsize=7.5,weight='bold')
        ax.text(x,.55,desc,fontsize=6)
        ax.text(x,.38,module,fontsize=5.8,color='.45')
        if j<3:ax.annotate('',xy=(x+.248,.79),xytext=(x+.223,.79),arrowprops={'arrowstyle':'->','lw':.8,'color':'.45'})
    ax.text(0,.13,'Choices + feedback → fitted belief trajectories',fontsize=6.5,color=COLORS['belief'],weight='bold')
    ax.text(.53,.13,'Current verbal reports → external comparison',fontsize=6.5,color='.3',weight='bold')
    ax.text(0,-.03,'Each report describes the chosen category, after the choice and before feedback.',fontsize=6,color='.4')


def render(source: Path, output: Path, people: pd.DataFrame, trials: pd.DataFrame, seeds: pd.DataFrame) -> None:
    style();fig=plt.figure(figsize=(183/25.4,235/25.4))
    fig.text(.075,.978,'Fig. 2 | Testing reconstructed beliefs against current verbal reports',fontsize=10,weight='bold',va='top')
    fig.text(.075,.952,'12 fitted participants  ·  4 per task  ·  choice-only fitting  ·  current reports, without carrying them forward',fontsize=6.6,color='.4')
    framework(fig)
    heading(fig,.075,.709,'b','Do predicted choices fit?')
    heading(fig,.405,.709,'c','Do beliefs match reports?')
    heading(fig,.735,.709,'d','Does their timing help?')
    ax=fig.add_axes([.085,.525,.225,.16]);cal=pd.read_csv(output/'calibration.csv')
    ax.plot([0,1],[0,1],color='.7',lw=.7,ls='--')
    for task in (1,2,3):
        z=cal[cal.task.eq(task)]
        ax.scatter(z.predicted,z.observed,s=9+z.n/55,facecolors='none',edgecolors=TASK_COLORS[task],lw=.6,alpha=.6)
    pooled=cal.groupby('bin').apply(lambda z:pd.Series({'x':np.average(z.predicted,weights=z.n),
                                                     'y':np.average(z.observed,weights=z.n)}),include_groups=False)
    ax.plot(pooled.x,pooled.y,'o-',color='.2',lw=1,ms=3)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Model P(correct)',ylabel='Observed accuracy')
    ax.text(.01,1.06,'Full-sequence fit',transform=ax.transAxes,fontsize=6,color='.4')
    for x,key,reference in [(.405,'compatibility','uniform_compatibility'),(.735,'dynamic_gain',None)]:
        ax=fig.add_axes([x,.510,.225,.177])
        for i,sid in enumerate(ORDER):
            s=people.set_index('subject').loc[sid];runs=seeds[(seeds.subject==sid)&seeds.variant.eq('selected')]
            c=TASK_COLORS[int(s.task)]
            if reference:
                ax.plot([s[reference],s[key]],[i,i],color='.75',lw=.8)
                ax.scatter(s[reference],i,s=13,facecolors='white',edgecolors='.5',lw=.7,zorder=3)
            ax.plot([runs[key].min(),runs[key].max()],[i,i],color=c,lw=1,zorder=2)
            ax.scatter(s[key],i,s=15,color=c,lw=0,zorder=4)
            if not reference:
                alt=seeds[(seeds.subject==sid)&seeds.variant.eq('alternative')][key].mean()
                ax.scatter(alt,i,s=13,marker='D',edgecolors='.35',facecolors='white',lw=.6,zorder=3)
        ax.axvline(0,color='.75',lw=.7,zorder=0)
        for y in [3.5,7.5]:ax.axhline(y,color='.9',lw=.5)
        ax.set(ylim=(11.8,-.8),yticks=np.arange(12),yticklabels=[f'S{s}' for s in ORDER] if reference else [])
        ax.tick_params(axis='y',length=0,pad=3)
        ax.set_xlabel('Report compatibility (0–1)' if reference else 'Gain over static belief',labelpad=4,fontsize=6.5)
        if reference:ax.set(xlim=(-.02,.57),xticks=[0,.25,.5])
        else:ax.set(xlim=(-.075,.235),xticks=[0,.1,.2])
    fig.text(.075,.467,'b  Small circles: participant/bin; black: pooled.  c  Filled: model; open: uniform rules.  d  Diamond: one near-optimal parameter point.',fontsize=5.6,color='.4')
    fig.text(.075,.451,'c–d  Lines: range over 8 particle-filter runs. Compatibility is report support across all rules, not exact rule-identification accuracy.',fontsize=5.6,color='.4')
    heading(fig,.075,.414,'e','Do report changes and performance gains follow the inferred belief?')
    example_specs=[(215,650,460,768),(328,593,400,704)]
    for j,(sid,event,start,end) in enumerate(example_specs):
        x=.075+j*.475;g=trials[trials.iSub.eq(sid)].copy();task=int(g.task.iloc[0])
        top=fig.add_axes([x,.318,.415,.058]);bottom=fig.add_axes([x,.226,.415,.075],sharex=top)
        top.set_title(f'S{sid} · Task {task} · behavior-defined rise after trial {event}',loc='left',fontsize=6.5,pad=5)
        top.plot(g.trial,g.label_correct.rolling(32,min_periods=32).mean(),color='.25',lw=1,label='Observed')
        top.plot(g.trial,g.model_correct.rolling(32,min_periods=32).mean(),color=COLORS['predicted'],ls='--',lw=.9,label='Model')
        top.set(ylim=(0,1.04),yticks=[0,1]);top.tick_params(axis='x',labelbottom=False,bottom=False)
        bottom.fill_between(g.trial,g.belief_seed_low,g.belief_seed_high,color=COLORS['belief'],alpha=.15,lw=0)
        bottom.plot(g.trial,g.belief,color=COLORS['belief'],lw=1,label='Target-rule belief')
        valid=g.oral_valid.to_numpy(bool)&g.scored.to_numpy(bool);target=g.oral_target_top.to_numpy(bool)
        bottom.vlines(g.trial[valid],-.12,-.08,color='.8',lw=.55)
        bottom.vlines(g.trial[valid&target],-.13,-.065,color='.2',lw=.65)
        bottom.set(xlim=(start,end),ylim=(-.16,1.04),yticks=[0,.5,1],xlabel='Trial (window shown)',xticks=[start,event,end])
        for a in [top,bottom]:a.axvline(event+.5,color='.6',ls=':',lw=.7)
        if j==0:top.set_ylabel('Accuracy',fontsize=6);bottom.set_ylabel('Target belief',fontsize=6)
    fig.text(.075,.184,'Top: trailing 32-trial accuracy (solid observed; dashed model). Bottom: pre-choice belief, with PF range; report ticks below.',fontsize=5.7,color='.4')
    fig.text(.075,.170,'Dark ticks: the target is among the best-compatible rules in that current report. Gray ticks: other valid reports. Ties are retained.',fontsize=5.7,color='.4')
    heading(fig,.075,.137,'f','Across participants: more belief change when the reported rule changes?')
    ax=fig.add_axes([.095,.052,.855,.065]);ax.axhline(0,color='.65',lw=.7)
    for i,sid in enumerate(ORDER):
        row=people.set_index('subject').loc[sid]
        ax.scatter(i,row.change_difference,color=TASK_COLORS[int(row.task)],s=18,lw=0)
    for x in [3.5,7.5]:ax.axvline(x,color='.9',lw=.5)
    ax.set(xticks=range(12),xticklabels=[str(s) for s in ORDER],xlim=(-.5,11.5))
    ax.set_ylabel('Extra change\nper trial',fontsize=6,labelpad=3)
    fig.text(.075,.020,'f  Same-category reports; matched gaps ≤32 trials. Positive values: more model change during report-change intervals. Descriptive; n = 12.',fontsize=5.6,color='.4')
    fig.savefig(output/'Figure2_belief_validation.png',dpi=450,facecolor='white');plt.close(fig)
    atlas(output,people,trials)
    sensitivity_figure(output,people,seeds)


def atlas(output: Path, people: pd.DataFrame, trials: pd.DataFrame) -> None:
    """Full records for every fitted participant, including weak oral agreement."""
    fig=plt.figure(figsize=(183/25.4,240/25.4))
    fig.text(.075,.978,'Fig. S2 | Every fitted participant: behavior, target belief and verbal reports',fontsize=9,weight='bold',va='top')
    fig.text(.075,.95,'No case excluded. Target belief is not a measure of agreement with the full content of a report.',fontsize=6.5,color='.4')
    gs=fig.add_gridspec(4,3,left=.075,right=.975,bottom=.075,top=.90,wspace=.25,hspace=.62)
    for col in range(3):
        fig.text(.075+.315*col,.921,f'Task {col+1}',fontsize=8,weight='bold',color=TASK_COLORS[col+1])
        for row in range(4):
            sid=ORDER[col*4+row];g=trials[trials.iSub.eq(sid)];sub=gs[row,col].subgridspec(2,1,hspace=.12)
            a=fig.add_subplot(sub[0]);b=fig.add_subplot(sub[1],sharex=a)
            a.plot(g.trial,g.label_correct.rolling(32,min_periods=32).mean(),color='.25',lw=.85)
            a.plot(g.trial,g.model_correct.rolling(32,min_periods=32).mean(),color=COLORS['predicted'],lw=.75,ls='--')
            a.set_title(f'S{sid} · reports n = {int(people.set_index("subject").loc[sid].n_reports)}',loc='left',fontsize=6.5,pad=3)
            a.set(ylim=(0,1.04),yticks=[0,1]);a.tick_params(axis='x',labelbottom=False,bottom=False)
            b.fill_between(g.trial,g.belief_seed_low,g.belief_seed_high,color=COLORS['belief'],alpha=.15,lw=0)
            b.plot(g.trial,g.belief,color=COLORS['belief'],lw=.8)
            valid=g.oral_valid.to_numpy(bool)&g.scored.to_numpy(bool)
            dark=valid&g.oral_target_top.to_numpy(bool)
            b.vlines(g.trial[valid],-.13,-.09,color='.8',lw=.4);b.vlines(g.trial[dark],-.14,-.07,color='.2',lw=.5)
            b.set(xlim=(1,len(g)),ylim=(-.18,1.04),yticks=[0,1],xticks=[1,len(g)//2,len(g)])
            if col==0:a.set_ylabel('Accuracy',fontsize=6);b.set_ylabel('Belief',fontsize=6)
            if row==3:b.set_xlabel('Recorded trial')
    fig.text(.075,.035,'Accuracy: observed (solid), model (dashed), trailing 32. Belief: mean and range over 8 PF runs, before choice.',fontsize=6,color='.4')
    fig.text(.075,.019,'Current-report ticks: dark = target among tied best rules; gray = other valid reports. These are partial category reports, not unique full-rule labels.',fontsize=5.8,color='.4')
    fig.savefig(output/'FigureS2_all_participants.png',dpi=450,facecolor='white');plt.close(fig)


def sensitivity_figure(output: Path, people: pd.DataFrame, seeds: pd.DataFrame) -> None:
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,96/25.4))
    fig.subplots_adjust(left=.075,right=.975,bottom=.22,top=.75,wspace=.4)
    fig.text(.075,.955,'Fig. S2 | Numerical and definition sensitivity',fontsize=9,weight='bold')
    fig.text(.075,.87,'Same fitted cohort; no additional fits or particle runs. Near-optimal points are not model ablations.',fontsize=6.5,color='.4')
    for k,key in enumerate(['compatibility','dynamic_gain']):
        ax=axes[k]
        for i,sid in enumerate(ORDER):
            task=int(people.set_index('subject').loc[sid].task);color=TASK_COLORS[task]
            for variant,offset,marker in [('selected',-.11,'o'),('alternative',.11,'D')]:
                z=seeds[(seeds.subject==sid)&seeds.variant.eq(variant)]
                ax.plot([z[key].min(),z[key].max()],[i+offset]*2,color=color,lw=.8)
                ax.scatter(z[key].mean(),i+offset,s=12,marker=marker,facecolors=color if variant=='selected' else 'white',edgecolors=color,lw=.6,zorder=3)
        ax.axvline(0,color='.7',lw=.6)
        ax.set(yticks=range(12),yticklabels=[str(s) for s in ORDER],ylim=(11.8,-.8))
        ax.set_title('a  Report compatibility' if k==0 else 'b  Gain over static belief',loc='left',fontsize=7)
    check=pd.read_csv(output/'change_sensitivity.csv');ax=axes[2]
    for sid,g in check.groupby('subject'):
        ax.plot(g.oral_change_threshold,g.change_difference,'o-',color=TASK_COLORS[int(g.task.iloc[0])],ms=2.8,lw=.65,alpha=.7)
    ax.axhline(0,color='.7',lw=.6);ax.set(xticks=[.25,.5,.75],xlabel='Oral-change threshold (TV)',ylabel='Extra belief change per trial')
    ax.set_title('c  Report-change definition',loc='left',fontsize=7)
    fig.text(.075,.095,'a–b  Circle: selected parameter point (8 runs); diamond: one near-optimal point (4 runs). Lines: min–max PF range.',fontsize=6,color='.4')
    fig.text(.075,.044,'c  Stable reports: TV ≤0.1. Same-category, same-session intervals; exact-gap matched within participant. No confidence intervals or p values.',fontsize=5.8,color='.4')
    fig.savefig(output/'FigureS2_validation_sensitivity.png',dpi=450,facecolor='white');plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(DEFAULT_SOURCE))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    people,trials,seeds=analyze(source,output);render(source,output,people,trials,seeds)


if __name__=='__main__':main()
