"""Supplementary Fig2: S129 fit checks from saved outputs, without new fits.

Ten panels distinguish conditional prediction, online states, terminal ancestry,
and autonomous learning. Existing output directories are never overwritten.
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
BLUE, BLACK, ORANGE = '#487DA8', '#333333', '#C57D47'


def build(output: Path, model_dir: Path, case_sources: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    sources = output / 'source_data'
    sources.mkdir()
    inputs = []

    def read_csv(relative, name=None):
        path = model_dir / relative
        inputs.append(path)
        shutil.copy2(path, sources / (name or path.name))
        return pd.read_csv(path)

    case = json.loads((case_sources/'manifest.json').read_text())
    assert case['subject'] == 129 and case['n_trials'] == 256
    for name in ['trial_source.csv','belief_source.csv','oral_distribution.csv','distribution_overlap.csv','manifest.json']:
        shutil.copy2(case_sources/name, sources/('case_'+name))
        inputs.append(case_sources/name)
    trial = pd.read_csv(case_sources/'trial_source.csv')
    overlap = pd.read_csv(case_sources/'distribution_overlap.csv')
    trace = read_csv('search_diagnostics/fine/best_error_trajectory.csv')
    residual = read_csv('evaluation/behavior_ppc/sequential_residual_trial_data.csv')
    internal = read_csv('internal_trajectories/internal_cognitive_trial_summary.csv')
    auto_summary = read_csv('autonomous_trajectories/autonomous_trajectory_summary.csv')
    read_csv('evaluation/behavior_ppc/behavior_ppc_subject_summary.csv')
    read_csv('evaluation/basic/accuracy_band_summary.csv')
    final_path = model_dir/'optimization/subject_129/final_rescore.jsonl'
    final = [json.loads(line) for line in final_path.read_text().splitlines()]
    inputs.append(final_path)
    shutil.copy2(final_path,sources/final_path.name)
    assert int(residual.score_trial.sum()) == 255
    values = np.array([row['aggregated_error'] for row in final])
    final_table = pd.DataFrame({'search_rank':np.arange(1,len(final)+1),
        'mean_choice_nll':values,'delta_total_nll':(values-values.min())*255})
    final_table.to_csv(sources/'final_candidate_comparison.csv',index=False)
    selected_path = model_dir/'optimization/subject_129/best_hyperparams.json'
    selected = json.loads(selected_path.read_text())
    inputs.append(selected_path)
    params = selected['selected']['best_params']
    pd.DataFrame({'parameter':list(params),'value':list(params.values())}).to_csv(sources/'selected_parameters.csv',index=False)
    for folder in ['internal_trajectories','autonomous_trajectories']:
        path = model_dir/folder/'analysis_manifest.json'
        inputs.append(path);shutil.copy2(path,sources/(folder+'_manifest.json'))
    auto_manifest = json.loads((model_dir/'autonomous_trajectories/analysis_manifest.json').read_text())
    npz_path = model_dir/'autonomous_trajectories/autonomous_trajectory_arrays.npz'
    inputs.append(npz_path)
    auto = np.load(npz_path)
    stream = model_dir/'simulation/cache/subject_129_raw_runs.gz'
    inputs.append(stream)
    active = []
    with gzip.open(stream,'rb') as handle:
        while True:
            try:
                run = pickle.load(handle)
                assert run['subject_id'] == 129
                active.append(np.asarray(run['state_log']['marginal_active_probability']))
            except EOFError:
                break
    assert len(active) == 16
    active = np.mean(active,axis=0)
    assert active.shape == (256,29) and np.isfinite(active).all()
    np.testing.assert_allclose(active.sum(axis=1),5,atol=1e-7)
    pd.DataFrame(active,columns=[f'H{i}' for i in range(29)]).to_csv(sources/'online_active_probability.csv',index=False)
    np.testing.assert_array_equal(trial.trial,residual.trial)
    np.testing.assert_array_equal(trial.observed_accuracy,residual.observed_correct)
    np.testing.assert_allclose(trial.model_correct_probability,residual.correct_probability,atol=1e-8)
    observed = trial.observed_accuracy.to_numpy()
    predicted = trial.model_correct_probability.to_numpy()
    def rolling(x, window=16):
        # Keep the pipeline convention: trial 1 is initialization; first window ends at 17.
        y = pd.Series(np.asarray(x,dtype=float)).copy();y.iloc[0] = np.nan
        return y.rolling(window,min_periods=window).mean()
    behavior = pd.DataFrame({'trial':trial.trial,'observed_rolling16':rolling(observed),
        'predicted_rolling16':rolling(predicted),'raw_residual':observed-predicted,
        'residual_rolling16':rolling(observed-predicted)})
    behavior.to_csv(sources/'behavior_plot_source.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':6.5,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':6,
        'ytick.labelsize':6,'legend.fontsize':5.8,'legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'svg.fonttype':'none','pdf.fonttype':42})
    fig, axes = plt.subplots(5,2,figsize=(183/25.4,250/25.4))
    fig.subplots_adjust(left=.09,right=.96,top=.925,bottom=.075,hspace=.85,wspace=.40)
    fig.text(.09,.975,'Supplementary Fig. 2 | Individual model checks',weight='bold',fontsize=10)
    fig.text(.09,.951,'S129 · Task 1 · 256 trials · full PMH fit · fixed-parameter diagnostics',fontsize=7,color=BLACK)
    def title(ax,letter,text):
        ax.set_title(f'{letter}  {text}',loc='left',pad=7,fontweight='bold')
    def trials(ax):
        ax.set_xlim(1,256);ax.set_xticks([1,128,256]);ax.set_xlabel('Trial')
    ax=axes[0,0];title(ax,'a','Fine-search convergence')
    for i,(rid,group) in enumerate(trace.groupby('restart_id')):
        ax.plot(group.step_in_restart,group.best_error,lw=.9,label=f'Start {int(rid)+1}',color=plt.cm.Blues(.40+.17*i))
    ax.set(xlabel='Coordinate update',ylabel='Mean choice NLL')
    ax.legend(ncol=2,loc='upper right')
    ax=axes[0,1];title(ax,'b','Independent final rescoring')
    ax.bar(np.arange(1,5),final_table.delta_total_nll,color=[BLUE if i==values.argmin() else '#CCD4D9' for i in range(4)],width=.60)
    ax.scatter([values.argmin()+1],[0],color=BLUE,s=16,zorder=3,clip_on=False)
    ax.set(xticks=[1,2,3,4],xlabel='Candidate rank in fine search',ylabel='Δ total NLL from best')
    ax.text(.97,.95,f'Best NLL = {values.min():.3f}/trial\n128 particles × 16 seeds',transform=ax.transAxes,ha='right',va='top',fontsize=6,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
    ax=axes[1,0];title(ax,'c','Observed-history prediction')
    ax.plot(trial.trial,behavior.observed_rolling16,color=BLACK,lw=1,label='Observed')
    ax.plot(trial.trial,behavior.predicted_rolling16,color=BLUE,lw=1,label='Model')
    ax.axhline(.5,color='.75',ls=':',lw=.6);ax.set(ylim=(0,1.03),ylabel='Accuracy (16-trial mean)');trials(ax)
    ax.legend(loc='lower right',ncol=2)
    ax=axes[1,1];title(ax,'d','Conditional prediction residual')
    ax.axhline(0,color='.65',lw=.6)
    ax.plot(trial.trial,behavior.residual_rolling16,color=ORANGE,lw=1)
    ax.fill_between(trial.trial,0,behavior.residual_rolling16,color=ORANGE,alpha=.15)
    ax.set(ylabel='Observed − predicted',ylim=(-.35,.35));trials(ax)
    ax=axes[2,0];title(ax,'e','Rules present in the workspace')
    im=ax.imshow(active.T,aspect='auto',origin='upper',interpolation='nearest',cmap='Blues',vmin=0,vmax=1,extent=[.5,256.5,28.5,-.5])
    ax.set_yticks([0,9,19,28],labels=['H0','H9','H19','H28']);trials(ax)
    cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.035,ticks=[0,1]);cb.ax.tick_params(labelsize=5,length=2)
    ax=axes[2,1];title(ax,'f','Online search probability')
    ax.plot(internal.trial,internal.online_swap_probability,color=BLUE,lw=.65)
    ax.set(ylim=(0,.6),ylabel='Search-event probability');trials(ax)
    ax=axes[3,0];title(ax,'g','Full-space oral correspondence')
    ax.plot(overlap.trial,overlap.overlap,color='.75',lw=.4,label='Trialwise')
    ax.plot(overlap.trial,overlap.overlap.rolling(32,min_periods=32).mean(),color=BLUE,lw=1,label='32-trial mean')
    ax.set(ylim=(0,1.03),ylabel='Overlap (1 − TV)');trials(ax);ax.legend(loc='upper left')
    ax=axes[3,1];title(ax,'h','Terminal ancestry support')
    ax.plot(internal.trial,internal.unique_ancestors,color=ORANGE,lw=.9,label='Unique')
    ax.plot(internal.trial,internal.effective_ancestors,color=BLUE,lw=.9,label='Effective')
    ax.set(ylabel='Ancestor count');trials(ax);ax.legend(loc='upper left')
    ax.text(.04,.56,'Limited early-history support',fontsize=5.7,color=BLACK,transform=ax.transAxes)
    ax=axes[4,0];title(ax,'i','Autonomous learning trajectories')
    x=auto['rolling_trial'];curves=auto['rolling_accuracy']
    bands={'trial':x}
    for key,alpha in [('central_90_indices',.12),('central_50_indices',.27)]:
        chosen=curves[auto[key]];lo=chosen.min(axis=0);hi=chosen.max(axis=0)
        ax.fill_between(x,lo,hi,color=BLUE,alpha=alpha,lw=0)
        bands[key+'_low']=lo;bands[key+'_high']=hi
    mid=auto_manifest['whole_curve_region']['overall_medoid_rollout_index']
    ax.plot(x,curves[mid],color=BLUE,lw=.85,label='Medoid')
    ax.plot(x,auto['observed_rolling_accuracy'],color=BLACK,lw=.85,label='Observed')
    ax.set(ylim=(0,1.03),ylabel='Accuracy (16-trial mean)');trials(ax);ax.legend(loc='lower right',ncol=2)
    bands['medoid']=curves[mid];bands['observed']=auto['observed_rolling_accuracy']
    pd.DataFrame(bands).to_csv(sources/'autonomous_band_plot_source.csv',index=False)
    ax=axes[4,1];title(ax,'j','Sustained-mastery onset')
    onset=auto_summary.sustained_mastery_onset.to_numpy()
    reached=np.sort(onset[np.isfinite(onset)])
    cdf=np.array([(reached<=t).sum()/len(onset) for t in range(1,257)])
    ax.step(np.arange(1,257),cdf,where='post',color=BLUE,lw=1)
    observed_onset=auto_manifest['mastery']['observed_onset']
    ax.axvline(observed_onset,color=BLACK,ls='--',lw=.8)
    ax.text(.03,.95,f'Model median: {np.median(reached):.0f}\nObserved: {observed_onset:.0f}\nNot reached: {np.mean(~np.isfinite(onset)):.1%}',transform=ax.transAxes,va='top',fontsize=6)
    ax.set(ylim=(0,1.03),ylabel='Fraction of all 500 rollouts');trials(ax)
    pd.DataFrame({'trial':np.arange(1,257),'cumulative_fraction':cdf}).to_csv(sources/'mastery_cdf.csv',index=False)
    fig.text(.09,.032,'Descriptive, in-sample diagnostics. Autonomous rollouts include process and choice variation; parameters are fixed.',fontsize=6,color=BLACK)
    fig.text(.09,.019,'H0: target rule. Full-space overlap is supplementary to the target-based alignment in main Fig2b.',fontsize=6,color=BLACK)
    fig.savefig(output/'FigureS2_draft.png',dpi=450,facecolor='white')
    plt.close(fig)
    manifest={'subject':129,'condition':1,'trial_count':256,'size_mm':[183,250],'dpi':450,
        'status':'single_subject_diagnostic_draft','autonomous_rollouts':500,
        'source_sha256':{str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'figure_sha256':hashlib.sha256((output/'FigureS2_draft.png').read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(__file__,output/Path(__file__).name)
    print(output/'FigureS2_draft.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--case-sources',type=Path,required=True)
    a=p.parse_args();build(a.output,a.model_dir,a.case_sources)
