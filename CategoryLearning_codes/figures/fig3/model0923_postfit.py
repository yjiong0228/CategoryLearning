"""PNG-only diagnostic figures for the twelve-subject Model 0923 postfit audit.

Contract: distinguish observed-history prediction, autonomous behavior variation,
and fixed-parameter latent-state uncertainty. All trials and catalogue rules
are retained in source tables. No group hypothesis tests are shown.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.Bayesian_state.evaluation.autonomous_trajectories import rolling_binary_ensemble

BLUE, GRAY, ORANGE = '#3578A7', '#555555', '#D58A37'


def style() -> None:
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})


def save(fig, path: Path) -> None:
    fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def subject_figures(subject: int, condition: int, arrays, states: dict,
                    rollouts: np.ndarray, capacity: int, window: int, output: Path) -> None:
    style()
    trial=np.arange(1,len(arrays.choices)+1)
    probabilities=states['probabilities']
    observed=(arrays.feedback==1).astype(float)
    correct=probabilities[np.arange(len(trial)),arrays.categories-1]
    selected=probabilities[np.arange(len(trial)),arrays.choices-1]
    x, observed_curve=rolling_binary_ensemble(observed,window_size=window)
    _, predicted_curve=rolling_binary_ensemble(correct,window_size=window)
    _, simulation_curves=rolling_binary_ensemble(rollouts==1,window_size=window)
    fig,axes=plt.subplots(2,2,figsize=(7.2047244094,5.0393700787),layout='constrained')
    a,b,c,d=axes.flat
    a.plot(x,observed_curve,color=GRAY,label='Observed')
    a.plot(x,predicted_curve,color=BLUE,label='Conditional prediction')
    a.set(ylabel='Rolling exact accuracy',ylim=(0,1),title='a  Prediction along observed history')
    a.legend(fontsize=7)
    low,med,high=np.quantile(simulation_curves,[.1,.5,.9],axis=0)
    b.fill_between(x,low,high,color=BLUE,alpha=.18,label='Autonomous 10–90%')
    b.plot(x,med,color=BLUE,label='Autonomous median')
    b.plot(x,observed_curve,color=GRAY,label='Observed')
    b.set(ylabel='Rolling exact accuracy',ylim=(0,1),title='b  Fixed-parameter autonomous behavior')
    b.legend(fontsize=7)
    c.plot(trial[1:],-np.log(np.clip(selected[1:],1e-12,1)),color=BLUE,lw=.7)
    c.set(ylabel='Choice NLL per scored trial',title='c  Where prediction fails')
    residual=observed-correct
    _, residual_curve=rolling_binary_ensemble(residual,window_size=window)
    d.axhline(0,color=GRAY,lw=.6)
    d.plot(x,residual_curve,color=ORANGE)
    d.set(ylabel='Rolling observed − predicted',title='d  Conditional accuracy residual')
    for ax in axes.flat:ax.set(xlabel='Valid trial',xlim=(1,len(trial)))
    fig.suptitle(f'S{subject} · condition {condition} · in-sample behavior checks',fontsize=10)
    save(fig,output/'behavior.png')
    pd.DataFrame({'valid_trial':x,'observed_rolling_exact_accuracy':observed_curve,
                  'conditional_rolling_correct_probability':predicted_curve,
                  'autonomous_q10':low,'autonomous_median':med,'autonomous_q90':high,
                  'rolling_accuracy_residual':residual_curve}).to_csv(output/'behavior_plot_source.csv',index=False)

    fig,axes=plt.subplots(6,1,figsize=(7.2047244094,8.2677165354),layout='constrained',sharex=True,
                          gridspec_kw={'height_ratios':[1.25,1.25,1.25,.8,.8,.8]})
    for ax,name,title in zip(axes[:3],('active','belief','choice_weights'),
                            ('a  Candidate inclusion probability','b  Rule belief','c  Contribution to choice')):
        values=states['filtered_'+name]
        image=ax.imshow(values.T,origin='lower',aspect='auto',extent=[.5,len(trial)+.5,-.5,values.shape[1]-.5],
                        cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
        ax.set(ylabel='Rule index',title=title)
        fig.colorbar(image,ax=ax,label='Probability / weight',pad=.01)
    a=axes[3]
    for name,label,color in [('candidate_count','Candidates',BLUE),('effective_participation','Effective participation',ORANGE)]:
        mean,q10,q50,q90=states['filtered_'+name].T
        a.fill_between(trial,q10,q90,color=color,alpha=.15)
        a.plot(trial,mean,color=color,label=label,lw=.8)
    a.axhline(capacity,color=GRAY,ls='--',lw=.6,label='Capacity upper bound')
    a.set(ylabel='Number',ylim=(.8,capacity+.5),title='d  Within-path counts: mean and 10–90% range')
    a.legend(ncol=3,fontsize=7)
    mean,q10,q50,q90=states['filtered_choice_weighted_beta'].T
    axes[4].fill_between(trial,q10,q90,color=BLUE,alpha=.18)
    axes[4].plot(trial,mean,color=BLUE,lw=.8)
    axes[4].set(ylabel='Choice-weighted β',ylim=(0,25.5),title='e  Prediction precision: mean and 10–90% range')
    if condition==3:
        for k in range(3):axes[5].plot(trial,states['pairing_post_feedback'][:,k],label=f'Pairing {k+1}',lw=.8)
        axes[5].set(ylabel='Pairing belief',ylim=(0,1),title='f  Pairing belief after feedback')
        axes[5].legend(ncol=3,fontsize=7)
    else:
        for name,color,label in [('search',BLUE,'Search'),('global_search',ORANGE,'Global given search')]:
            axes[5].plot(trial,states['predictive_'+name],color=color,label=label,lw=.8)
        axes[5].set(ylabel='Policy probability',ylim=(0,1),title='f  Search policy before choice')
        axes[5].legend(ncol=2,fontsize=7)
    axes[-1].set(xlabel='Valid trial',xlim=(1,len(trial)))
    fig.suptitle(f'S{subject} · pre-feedback state conditioned on current choice\nFixed selected parameters; latent ranges are pointwise',fontsize=9)
    save(fig,output/'representations.png')


def cohort_figures(summary: pd.DataFrame, parameters: pd.DataFrame, output: Path) -> None:
    style()
    fig,axes=plt.subplots(1,2,figsize=(7.2047244094,3.2283464567),layout='constrained')
    ordered=summary.sort_values(['condition','subject'])
    index=np.arange(len(ordered))
    axes[0].plot(index,ordered['selected_fit_nll'],'o',color=GRAY,label='Original final rescore')
    axes[0].plot(index,ordered['mean_choice_nll'],'s',color=BLUE,label='Fresh PF seeds')
    axes[0].set(ylabel='Mean choice NLL',title='a  Fixed-parameter score reproducibility')
    axes[0].legend(fontsize=7)
    axes[1].bar(index,ordered['nll_gain_over_uniform'],color=BLUE)
    axes[1].set(ylabel='NLL improvement over uniform',title='b  Fit improvement within each task')
    for ax in axes:
        ax.set_xticks(index,ordered['subject'].astype(str),rotation=60)
        ax.set_xlabel('Subject')
    save(fig,output/'cohort_behavior.png')
    estimate=parameters.pivot(index='parameter',columns='subject',values='estimate')
    scaled=parameters.assign(position=(parameters.estimate-parameters.grid_min)/(parameters.grid_max-parameters.grid_min)).pivot(index='parameter',columns='subject',values='position')
    fig,ax=plt.subplots(figsize=(7.2047244094,4.7244094488),layout='constrained')
    image=ax.imshow(scaled.values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
    ax.set_xticks(np.arange(len(scaled.columns)),scaled.columns.astype(str))
    ax.set_yticks(np.arange(len(scaled.index)),scaled.index)
    for j,subject in enumerate(scaled.columns):
        for i,name in enumerate(scaled.index):
            row=parameters[(parameters.subject==subject)&(parameters.parameter==name)].iloc[0]
            text=f'{estimate.loc[name,subject]:g}'+('†' if row.inactive_at_selected_point else '')
            ax.text(j,i,text,ha='center',va='center',fontsize=7,color='white' if scaled.iloc[i,j]>.65 else GRAY)
    ax.set(title='Selected parameters · † inactive under the selected parameter combination',xlabel='Subject')
    fig.colorbar(image,ax=ax,label='Position in finite search support')
    save(fig,output/'parameter_audit.png')


def main() -> None:
    import argparse
    from src.Bayesian_state.utils.paths import ROOT_DIR
    from src.Bayesian_state.workflows.analysis.analyze_model_0923_postfit import run
    from src.Bayesian_state.utils.parallel import single_threaded_processes
    parser=argparse.ArgumentParser(description='Export the Model 0923 first-stage analysis and PNG figures')
    parser.add_argument('--config',type=Path,default=Path('configs/exp123/specific_models/model_0923_selected12_postfit.yaml'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke and args.output_dir is None:parser.error('--smoke requires a new --output-dir')
    config=args.config if args.config.is_absolute() else ROOT_DIR/args.config
    output=args.output_dir if args.output_dir is None or args.output_dir.is_absolute() else ROOT_DIR/args.output_dir
    with single_threaded_processes():
        result=run(config,output,args.smoke,subject_figures,cohort_figures)
    print({'status':result['status'],'seconds':result['elapsed_seconds']})


if __name__=='__main__':main()
