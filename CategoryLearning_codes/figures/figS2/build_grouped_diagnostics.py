"""Five question-led supplementary figures from a completed S129 PMH fit.

Read-only extraction; no optimization, PF rerun or new scientific model.
All outputs and numerical plot sources go into a fresh directory.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BLUE, INK, ORANGE, PALE = '#487DA8', '#333333', '#C57D47', '#DDE7EE'


def build(output: Path, model_dir: Path, case_sources: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    sources = output/'source_data'; sources.mkdir()
    inputs = []
    def csv(path, name=None):
        inputs.append(path); shutil.copy2(path,sources/(name or path.name))
        return pd.read_csv(path)
    def js(path, name=None):
        inputs.append(path);shutil.copy2(path,sources/(name or path.name))
        return json.loads(path.read_text())
    trial=csv(case_sources/'trial_source.csv')
    oral=csv(case_sources/'oral_distribution.csv').to_numpy()
    overlap=csv(case_sources/'distribution_overlap.csv')
    case=js(case_sources/'manifest.json','case_manifest.json')
    assert case['subject']==129 and case['n_trials']==256
    trace=csv(model_dir/'search_diagnostics/fine/best_error_trajectory.csv')
    internal=csv(model_dir/'internal_trajectories/internal_cognitive_trial_summary.csv')
    js(model_dir/'internal_trajectories/analysis_manifest.json','internal_manifest.json')
    auto_manifest=js(model_dir/'autonomous_trajectories/analysis_manifest.json','autonomous_manifest.json')
    auto_summary=csv(model_dir/'autonomous_trajectories/autonomous_trajectory_summary.csv')
    residual=csv(model_dir/'evaluation/behavior_ppc/sequential_residual_trial_data.csv')
    final_path=model_dir/'optimization/subject_129/final_rescore.jsonl'
    inputs.append(final_path);shutil.copy2(final_path,sources/final_path.name)
    final=[json.loads(line) for line in final_path.read_text().splitlines()]
    rawpath=model_dir/'simulation/cache/subject_129_raw_runs.gz';inputs.append(rawpath)
    runs=[]
    with gzip.open(rawpath,'rb') as f:
        while True:
            try:runs.append(pickle.load(f))
            except EOFError:break
    assert len(runs)==16
    for r in runs:
        assert r['subject_id']==129 and r['selection_prediction_mode']=='prior_t'
        np.testing.assert_array_equal(r['metrics_by_mode']['prior_t']['observed_choice_index'],trial.observed_choice-1)
    probs=np.array([r['metrics_by_mode']['prior_t']['pred_category_probs'] for r in runs])
    beliefs=np.array([r['state_log']['marginal_prior'] for r in runs])
    active=np.array([r['state_log']['marginal_active_probability'] for r in runs])
    assert beliefs.shape==active.shape==(16,256,29)
    np.testing.assert_allclose(beliefs.sum(axis=2),1,atol=1e-8)
    np.testing.assert_allclose(active.sum(axis=2),5,atol=1e-8)
    assert np.isfinite(probs).all() and np.isfinite(beliefs).all()
    np.testing.assert_allclose(probs.sum(axis=2),1,atol=1e-8)
    prior=beliefs.mean(axis=0);presence=active.mean(axis=0)
    pchoice2=probs.mean(axis=0)[:,1]
    score=residual.score_trial.to_numpy(dtype=bool)
    assert score.sum()==255 and not score[0] and score[1:].all()
    np.testing.assert_array_equal(trial.trial,residual.trial)
    np.testing.assert_array_equal(trial.observed_accuracy,residual.observed_correct)
    np.testing.assert_allclose(trial.model_correct_probability,residual.correct_probability,atol=1e-8)
    valid=np.isfinite(oral).all(axis=1)
    np.testing.assert_allclose(oral[valid].sum(axis=1),1,atol=1e-8)
    np.testing.assert_allclose(np.minimum(prior[valid],oral[valid]).sum(axis=1),overlap.overlap[valid],atol=1e-8)
    auto_path=model_dir/'autonomous_trajectories/autonomous_trajectory_arrays.npz';inputs.append(auto_path)
    auto=np.load(auto_path)
    assert auto['feedback'].shape==(500,256)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,
        'ytick.labelsize':6.5,'legend.fontsize':6.5,'legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'svg.fonttype':'none','pdf.fonttype':42})
    outputs=[]
    def page(number, heading, question, height=158):
        fig=plt.figure(figsize=(183/25.4,height/25.4))
        fig.text(.09,.962,f'S2.{number} | {heading}',fontsize=11,weight='bold')
        fig.text(.09,.918,question,fontsize=7.5,color=INK)
        return fig
    def panel(fig,rect,label,title):
        ax=fig.add_axes(rect);ax.set_title(f'{label}  {title}',loc='left',pad=9,weight='bold');return ax
    def timeaxis(ax):
        ax.set_xlim(1,256);ax.set_xticks([1,128,256]);ax.set_xlabel('Trial')
    def save(fig,name,footnote):
        fig.text(.09,.028,footnote,fontsize=6.5,color=INK)
        fig.savefig(output/name,dpi=450,facecolor='white');plt.close(fig);outputs.append(name)
    def roll(v,w=16):
        v=pd.Series(np.asarray(v,dtype=float)).copy();v.iloc[0]=np.nan
        return v.rolling(w,min_periods=w).mean()

    # 1. Alternative parameter explanations are central, not just convergence.
    fig=page(1,'Parameter estimation','Do different searches select a clearly distinct parameter explanation?',180)
    ax=panel(fig,[.10,.62,.35,.22],'a','Fine-search paths')
    for i,(rid,g) in enumerate(trace.groupby('restart_id')):
        ax.plot(g.step_in_restart,g.best_error,lw=1,label=f'Start {rid+1}',color=plt.cm.Blues(.42+.16*i))
    ax.set(xlabel='Coordinate update',ylabel='Mean choice NLL');ax.legend(ncol=2)
    ax=panel(fig,[.60,.62,.35,.22],'b','Independent final rescoring')
    vals=np.array([x['aggregated_error'] for x in final]);delta=(vals-vals.min())*score.sum()
    ax.bar(range(1,5),delta,color=PALE,width=.6)
    ax.scatter([vals.argmin()+1],[0],color=BLUE,s=20,zorder=4,clip_on=False)
    ax.set(xticks=range(1,5),xlabel='Candidate rank in fine search',ylabel='Δ total NLL from best',ylim=(0,1.12))
    ax.text(.98,.94,'All four candidates: ΔNLL < 1',ha='right',va='top',transform=ax.transAxes,fontsize=6.5)
    rows=[]
    labels=['Capacity M','Persistent execution χ','Memory γ','Correct-event probability','Error-event probability','Global-search baseline','Failure → search gain','Failure → global gain','Initial precision β₀','Support update η₊','Refutation update η₋']
    for x in final:
        flat={}
        for k,v in x['hyperparams'].items():flat.update(v if isinstance(v,dict) else {k:v})
        def end(s):return next(v for k,v in flat.items() if k.endswith(s))
        rows.append([end('.capacity'),int(end('.persistent_execution.enabled')),end('.gamma'),end('.event_after_correct'),end('.event_after_error'),end('.global_search'),end('.accumulator_logit_gain'),end('.global_search_failure_gain'),end('.beta_init'),end('.increase_rate'),end('.decrease_rate')])
    table_data=pd.DataFrame(np.array(rows).T,index=labels,columns=[f'C{i+1}' for i in range(4)])
    table_data.to_csv(sources/'candidate_parameters.csv')
    pd.DataFrame({'candidate':range(1,5),'mean_nll':vals,'delta_total_nll':delta}).to_csv(sources/'final_scores.csv',index=False)
    ax=panel(fig,[.10,.125,.85,.37],'c','Parameter alternatives retained for final scoring');ax.axis('off')
    cells=[[label]+[f'{v:g}' if float(v).is_integer() else f'{v:.3g}' for v in table_data.loc[label]] for label in labels]
    tab=ax.table(cellText=cells,colLabels=['Parameter','C1','C2','C3','C4 · selected'],colWidths=[.42,.145,.145,.145,.145],cellLoc='center',bbox=[0,0,1,1])
    tab.auto_set_font_size(False);tab.set_fontsize(7)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#DFE3E6');cell.set_linewidth(.4)
        if r==0:cell.set_facecolor('#E8EDF1');cell.set_text_props(weight='bold')
        elif c==4:cell.set_facecolor('#EDF4F8')
        if c==0:cell.set_text_props(ha='left')
    fig.text(.10,.080,'Selected M = 5 and γ = 0.97 reach the candidate-grid upper bounds.',fontsize=7,color=ORANGE)
    save(fig,'S2_1_parameter_estimation.png','S129 · 255 scored trials · final scoring: 128 particles × 16 seeds · close scores do not establish uniqueness.')

    # 2. Diagnose deviations, rather than redraw the main accuracy trajectory.
    fig=page(2,'Behavioral fit details','Where does observed-history prediction remain inaccurate?',158)
    ax=panel(fig,[.10,.58,.35,.25],'a','Choice-probability calibration')
    cal=pd.DataFrame({'p':pchoice2[score],'y':(trial.observed_choice.to_numpy()[score]==2).astype(float)})
    cal['bin']=np.minimum((cal.p*5).astype(int),4)
    cal=cal.groupby('bin').agg(predicted=('p','mean'),observed=('y','mean'),n=('y','size')).reset_index();cal.to_csv(sources/'calibration.csv',index=False)
    assert cal.n.sum()==255
    ax.plot([0,1],[0,1],':',color='.7',lw=.8);ax.plot(cal.predicted,cal.observed,'o-',color=BLUE,lw=1,ms=4)
    for r in cal.itertuples():ax.annotate(f'n={r.n}',(r.predicted,r.observed),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6)
    ax.set(xlabel='Predicted P(choice = 2)',ylabel='Observed fraction',xlim=(-.03,1.03),ylim=(-.03,1.10),xticks=[0,.5,1],yticks=[0,.5,1])
    err=trial.observed_accuracy.to_numpy()-trial.model_correct_probability.to_numpy()
    block=pd.DataFrame({'trial':trial.trial,'residual':err,'score':score});block['block']=(block.trial-1)//32+1
    block=block[block.score].groupby('block').agg(mean_residual=('residual','mean'),n=('residual','size')).reset_index();block.to_csv(sources/'block_residual.csv',index=False)
    ax=panel(fig,[.60,.58,.35,.25],'b','Bias in consecutive blocks')
    ax.axhline(0,color='.65',lw=.7);ax.bar(block.block,block.mean_residual,color=[ORANGE if v<0 else BLUE for v in block.mean_residual])
    ax.set(xlabel='32-trial block',ylabel='Observed − predicted',xticks=range(1,9),ylim=(-.25,.25))
    ax=panel(fig,[.10,.15,.35,.25],'c','When prediction is biased')
    curve=roll(err);ax.axhline(0,color='.65',lw=.7);ax.plot(trial.trial,curve,color=ORANGE,lw=1)
    ax.set(ylabel='Residual (16-trial mean)',ylim=(-.35,.35));timeaxis(ax)
    ac=[]
    for lag in range(1,11):
        mask=score[:-lag]&score[lag:]
        ac.append(np.corrcoef(err[:-lag][mask],err[lag:][mask])[0,1])
    ax=panel(fig,[.60,.15,.35,.25],'d','Raw residual autocorrelation')
    ax.axhline(0,color='.65',lw=.7);ax.vlines(range(1,11),0,ac,color=BLUE,lw=1);ax.scatter(range(1,11),ac,color=BLUE,s=14)
    ax.set(xlabel='Lag (trials)',ylabel='Correlation',xticks=[1,5,10],ylim=(-.3,.3))
    pd.DataFrame({'trial':trial.trial,'raw_residual':err,'rolling16':curve,'score_trial':score}).to_csv(sources/'residual_source.csv',index=False)
    pd.DataFrame({'lag':range(1,11),'correlation':ac}).to_csv(sources/'residual_autocorrelation.csv',index=False)
    save(fig,'S2_2_behavioral_fit.png','S129 · full-sequence fit · descriptive calibration and residuals; overlapping windows are not independent samples.')

    # 3. Separate online state meanings from inference diagnostics.
    fig=page(3,'Internal states and numerical support','What does a rule probability mean, and how stable is the estimate?',158)
    ax=panel(fig,[.10,.58,.35,.25],'a','Present versus believed: target H0')
    ax.plot(trial.trial,presence[:,0],color=ORANGE,lw=1,label='Present in workspace')
    ax.plot(trial.trial,prior[:,0],color=BLUE,lw=1,label='Rule belief')
    ax.set(ylim=(0,1.03),ylabel='Probability');timeaxis(ax);ax.legend(loc='upper left')
    ax=panel(fig,[.60,.58,.35,.25],'b','H0 belief across PF seeds')
    lo,hi=np.quantile(beliefs[:,:,0],[.1,.9],axis=0)
    ax.fill_between(trial.trial,lo,hi,color=BLUE,alpha=.22,label='10–90% seed spread')
    ax.plot(trial.trial,prior[:,0],color=BLUE,lw=1,label='Mean of 16 seeds')
    ax.set(ylim=(0,1.03),ylabel='Pre-choice belief');timeaxis(ax);ax.legend(loc='upper left')
    ax=panel(fig,[.10,.15,.35,.25],'c','Particle-weight concentration')
    ax.plot(internal.trial,internal.mean_pre_choice_ess,color=BLUE,lw=.8,label='Before choice')
    ax.plot(internal.trial,internal.mean_post_choice_ess,color=ORANGE,lw=.8,label='After choice')
    ax.axhline(64,color='.6',ls=':',lw=.8,label='Resampling threshold')
    ax.set(ylabel='Mean ESS (128 particles)',ylim=(0,132));timeaxis(ax);ax.legend(loc='lower left',fontsize=6)
    ax=panel(fig,[.60,.15,.35,.25],'d','Support for complete past paths')
    ax.plot(internal.trial,internal.unique_ancestors,color=ORANGE,lw=1,label='Unique ancestors')
    ax.plot(internal.trial,internal.effective_ancestors,color=BLUE,lw=1,label='Effective ancestors')
    ax.set(ylabel='Count across 16 seeds');timeaxis(ax);ax.legend(loc='upper left',fontsize=6)
    pd.DataFrame({'trial':trial.trial,'target_present':presence[:,0],'target_belief':prior[:,0],'seed_q10':lo,'seed_q90':hi}).to_csv(sources/'state_comparison.csv',index=False)
    pd.DataFrame(beliefs[:,:,0].T,columns=[f'PF_seed_{i+1}' for i in range(16)]).to_csv(sources/'target_belief_by_seed.csv',index=False)
    save(fig,'S2_3_state_reliability.png','Fixed parameters · seed spread is numerical variation, not parameter uncertainty · early terminal ancestry is limited.')

    # 4. Existing oral diagnostics, explicitly not a completed robustness test.
    fig=page(4,'Oral evidence: diagnostic checks','Does correspondence extend beyond the late concentration on the target rule?',153)
    ax=panel(fig,[.10,.59,.85,.24],'a','Full-space correspondence over learning')
    ax.plot(trial.trial,overlap.overlap,color='.75',lw=.6,label='Trialwise')
    ax.plot(trial.trial,overlap.overlap.rolling(32,min_periods=32).mean(),color=BLUE,lw=1.2,label='32-trial mean')
    ax.axvline(85.5,color='.8',ls=':',lw=.7);ax.axvline(170.5,color='.8',ls=':',lw=.7)
    ax.set(ylabel='Overlap (1 − TV)',ylim=(0,1.03));timeaxis(ax);ax.legend(loc='upper left',ncol=2)
    stage=np.where(trial.trial<=85,'Early',np.where(trial.trial<=170,'Middle','Late'))
    def entropy(p):return -(np.where(p>0,p*np.log(np.clip(p,1e-12,1)),0)).sum(axis=1)/np.log(29)
    diagnostics=pd.DataFrame({'trial':trial.trial,'stage':stage,'overlap':overlap.overlap,'model_entropy':entropy(prior),'oral_entropy':entropy(oral),'valid':valid})
    diagnostics.to_csv(sources/'oral_diagnostics.csv',index=False)
    phases=diagnostics[diagnostics.valid].groupby('stage',sort=False).agg(overlap=('overlap','mean'),model_entropy=('model_entropy','mean'),oral_entropy=('oral_entropy','mean'),n=('trial','size')).reindex(['Early','Middle','Late'])
    phases.to_csv(sources/'oral_phase_summary.csv')
    ax=panel(fig,[.10,.16,.35,.24],'b','Correspondence by fixed third')
    ax.bar(range(3),phases.overlap,color=[PALE,PALE,BLUE],width=.55)
    for i,r in enumerate(phases.itertuples()):ax.text(i,r.overlap+.035,f'{r.overlap:.2f}',ha='center',fontsize=7)
    ax.set(xticks=range(3),xticklabels=['1–85','86–170','171–256'],xlabel='Trial range',ylabel='Mean overlap',ylim=(0,1))
    ax=panel(fig,[.60,.16,.35,.24],'c','How concentrated are the distributions?')
    ax.plot(range(3),phases.model_entropy,'o-',color=BLUE,lw=1,label='Model belief')
    ax.plot(range(3),phases.oral_entropy,'s-',color=INK,lw=1,label='Oral distribution')
    ax.set(xticks=range(3),xticklabels=['Early','Middle','Late'],ylabel='Normalized entropy',ylim=(0,1));ax.legend(loc='upper right')
    save(fig,'S2_4_oral_diagnostics.png','Oral σ = 0.05 · fixed temporal thirds · encoding-sensitivity and temporal-null tests remain pending.')

    # 5. Autonomous learning: distributional disagreement is shown directly.
    fig=page(5,'Autonomous learning','How does the model learn when it generates its own choices and feedback?',158)
    ax=panel(fig,[.10,.59,.85,.24],'a','Complete autonomous trajectories')
    x=auto['rolling_trial'];curves=auto['rolling_accuracy'];band={'trial':x}
    for key,alpha,label in [('central_90_indices',.12,'Central 90% of trajectories'),('central_50_indices',.28,'Central 50% of trajectories')]:
        subset=curves[auto[key]];low=subset.min(axis=0);high=subset.max(axis=0)
        ax.fill_between(x,low,high,color=BLUE,alpha=alpha,lw=0,label=label)
        band[key+'_low']=low;band[key+'_high']=high
    medoid=auto_manifest['whole_curve_region']['overall_medoid_rollout_index']
    ax.plot(x,curves[medoid],color=BLUE,lw=1,label='Model medoid');ax.plot(x,auto['observed_rolling_accuracy'],color=INK,lw=1,label='Observed')
    ax.set(ylim=(0,1.03),ylabel='Accuracy (16-trial mean)');timeaxis(ax);ax.legend(loc='lower right',ncol=2,fontsize=6)
    band['medoid']=curves[medoid];band['observed']=auto['observed_rolling_accuracy'];pd.DataFrame(band).to_csv(sources/'autonomous_band_source.csv',index=False)
    ax=panel(fig,[.10,.16,.35,.24],'b','When sustained mastery begins')
    onset=auto_summary.sustained_mastery_onset.to_numpy();reached=np.sort(onset[np.isfinite(onset)])
    cdf=np.array([(reached<=t).sum()/500 for t in range(1,257)])
    ax.step(range(1,257),cdf,where='post',color=BLUE,lw=1)
    ax.axvline(auto_manifest['mastery']['observed_onset'],color=INK,ls='--',lw=.8)
    ax.text(.03,.94,'Model median: 49\nObserved: 194\nNot reached: 1.8%',transform=ax.transAxes,va='top',fontsize=6.5)
    ax.set(ylabel='Fraction of all 500 rollouts',ylim=(0,1.03));timeaxis(ax)
    ax=panel(fig,[.60,.16,.35,.24],'c','Performance over the last 64 trials')
    final_acc=auto_summary.final_block_accuracy.to_numpy();observed_final=trial.observed_accuracy.iloc[-64:].mean()
    counts,edges=np.histogram(final_acc,bins=np.linspace(0,1,17));ax.stairs(counts/500,edges,fill=True,color=BLUE,alpha=.5)
    ax.axvline(observed_final,color=INK,ls='--',lw=1,label=f'Observed: {observed_final:.3f}')
    ax.set(xlabel='Final-block accuracy',ylabel='Fraction of rollouts',xlim=(0,1));ax.legend(loc='upper left')
    pd.DataFrame({'trial':range(1,257),'mastery_cdf':cdf}).to_csv(sources/'mastery_cdf.csv',index=False)
    pd.DataFrame({'left':edges[:-1],'right':edges[1:],'count':counts,'fraction':counts/500}).to_csv(sources/'final_accuracy_histogram.csv',index=False)
    assert counts.sum()==500
    save(fig,'S2_5_autonomous_learning.png','500 rollouts · fixed fitted parameters and 256-trial schedule · uncertainty reflects process and choice variation only.')
    auto.close()
    shutil.copy2(__file__,output/Path(__file__).name)
    (output/'manifest.json').write_text(json.dumps({'subject':129,'condition':1,'n_trials':256,'figures':outputs,
        'status':'five_part_single_subject_review','dpi':450,'width_mm':183,
        'new_fits_or_simulations':False,'pending':['parameter_boundary_expansion','oral_encoding_sensitivity','temporal_null_comparison','held_out_validation','recovery_and_ablation'],
        'source_sha256':{str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'figure_sha256':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in outputs}},indent=2)+'\n')
    print('\n'.join(str(output/name) for name in outputs))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--case-sources',type=Path,required=True)
    a=p.parse_args();build(a.output,a.model_dir,a.case_sources)
