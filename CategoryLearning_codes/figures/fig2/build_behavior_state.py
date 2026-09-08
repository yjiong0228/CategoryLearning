"""Fig2b/c: descriptive S101 preview from existing online PF exports only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'results/model_0826/subject_101_full_pattern_check'
STATE = SOURCE / 'primary/model_evaluation/internal_cognitive_trajectories_v1'
BLUE = '#487DA8'


def build(output: Path, oral_sigma: float = 0.05) -> None:
    files = [STATE / name for name in ['internal_cognitive_trial_summary.csv',
             'internal_cognitive_belief_source.csv', 'hypothesis_catalog.csv', 'analysis_manifest.json']]
    files += [ROOT / 'data/exp123/processed/Task2_processed.csv', SOURCE / 'README.md', SOURCE / 'primary_config.yaml']
    trial = pd.read_csv(files[0])
    belief = pd.read_csv(files[1])
    catalog = pd.read_csv(files[2])
    raw = pd.read_csv(files[4]).query('iSub == 101').copy()
    # Preserve recorded session/block/trial order; verify the model export row by row.
    assert len(raw) == len(trial) == 320 and raw.condition.eq(1).all()
    assert not raw.duplicated(['iSession', 'iBlock', 'iTrial']).any()
    assert np.array_equal(trial.trial, np.arange(1, 321))
    assert np.array_equal(raw.choice, trial.observed_choice)
    assert np.array_equal(raw.feedback, trial.observed_feedback)
    assert np.array_equal(raw.choice.eq(raw.category), raw.feedback.eq(1))
    p = trial.online_correct_probability.to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    prior = belief.pivot(index='hypothesis', columns='trial', values='online_prior').reindex(index=range(29), columns=range(1,321)).to_numpy()
    assert np.isfinite(prior).all() and (prior >= 0).all()
    np.testing.assert_allclose(prior.sum(axis=0), 1, atol=1e-8)
    correct = raw.feedback.eq(1).to_numpy(dtype=float)
    # Binary task: convert probability of the true category to P(choice=2).
    p2 = np.where(raw.category.to_numpy() == 2, p, 1-p)
    choice2 = raw.choice.to_numpy() == 2
    observed_p = np.where(choice2, p2, 1-p2)
    nll = float(-np.log(np.clip(observed_p, 1e-12, 1)).mean())
    known = raw.text.fillna('').str.strip().ne('').to_numpy()
    mentions = raw[[f'feature{i}_use' for i in range(1,5)]].to_numpy()
    assert np.isin(mentions, [0,1]).all()
    from src.Bayesian_state.evaluation.oral.scoring import OralAlignmentScoringMixin
    oral_encoder = OralAlignmentScoringMixin()
    oral_result = oral_encoder.compute_oral_mass_probabilities(raw, subjects=[101], oral_center_sigma=oral_sigma)[101]
    from src.Bayesian_state.hypothesis_space import ContinuousPartition
    partition = ContinuousPartition(n_dims=4, n_cats=2)
    ideal_reports = {}
    for category in [1, 2]:
        center = oral_encoder._category_prototypes(partition, 0, category-1)[0]
        q, diagnostics = oral_encoder._center_oral_distribution(center, category, partition, center_sigma=oral_sigma, return_diagnostics=True)
        ideal_reports[category] = {'distribution':q, 'diagnostics':diagnostics}
    ideal_q, _ = oral_encoder._category_state_distribution(ideal_reports, partition, 'center', center_sigma=oral_sigma)
    oral_target = oral_result['oral_mass'][:, 0]
    data = pd.DataFrame({'trial':trial.trial, 'observed_accuracy':correct,
                         'model_correct_probability':p, 'p_choice2':p2, 'choice2':choice2,
                         'report_available':known})
    data['oral_target_mass'] = oral_target
    data['model_target_prior'] = prior[0]
    for i in range(4): data[f'feature{i+1}_use'] = mentions[:,i]
    for name in ['observed_accuracy', 'model_correct_probability']:
        data[name+'_rolling32'] = data[name].rolling(32, min_periods=32).mean()
    blocks = data.assign(block=(data.trial-1)//40+1).groupby('block').agg(
        observed=('observed_accuracy','mean'), predicted=('model_correct_probability','mean'), n=('trial','size')).reset_index()
    data['calibration_bin'] = np.minimum((p2*5).astype(int),4)
    calibration = data.groupby('calibration_bin').agg(predicted=('p_choice2','mean'),
                       observed=('choice2','mean'), n=('trial','size')).reset_index()
    assert blocks.n.sum() == calibration.n.sum() == 320
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output/'trial_source.csv', index=False)
    blocks.to_csv(output/'block_source.csv', index=False)
    calibration.to_csv(output/'calibration_source.csv', index=False)
    belief[['trial','hypothesis','online_prior']].to_csv(output/'belief_source.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,
        'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':0.6,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig = plt.figure(figsize=(183/25.4, 202/25.4))
    fig.text(.06,.973,'b   Learning, rule beliefs and verbal reports',fontsize=10,weight='bold')
    fig.text(.06,.951,'Task 1 · S101 · 320 trials | Preliminary parameter-transfer check',fontsize=7,color='#555555')
    left, width = .17,.72
    ax = fig.add_axes([left,.756,width,.16])
    ax.plot(data.trial,data.observed_accuracy_rolling32,color='#333333',lw=1.15,label='Observed')
    ax.plot(data.trial,data.model_correct_probability_rolling32,color=BLUE,lw=1.25,label='Model: online prediction')
    ax.axhline(.5,color='.7',lw=.6,ls='--'); ax.set(ylim=(0,1.03),ylabel='Accuracy\n(32-trial mean)',xlim=(.5,320.5),yticks=[0,.5,1])
    ax.legend(loc='lower right',fontsize=6.5,ncol=2)
    ax.tick_params(labelbottom=False)
    ax = fig.add_axes([left,.565,width,.149])
    cmap = LinearSegmentedColormap.from_list('belief_blue',['#FFFFFF',BLUE,'#174568'])
    im = ax.imshow(prior,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1,extent=[.5,320.5,28.5,-.5])
    groups = [(0,0,'Target: F1'),(1,3,'Other thresholds'),(4,9,'Feature order'),(10,15,'Feature sum'),(16,18,'Paired sums'),(19,24,'Similarity'),(25,28,'Center bands')]
    ax.set_yticks([(a+b)/2 for a,b,_ in groups],labels=[s for _,_,s in groups])
    for _,b,_ in groups[:-1]: ax.axhline(b+.5,color='.8',lw=.4)
    ax.tick_params(axis='y',length=0);ax.tick_params(labelbottom=False)
    ax.set_title('Online rule belief · all 29 rules (one row per rule)',loc='left',pad=6)
    cb = fig.colorbar(im,cax=fig.add_axes([.907,.565,.012,.149]),ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=6,length=2)
    ax = fig.add_axes([left,.437,width,.087])
    ax.plot(data.trial, data.model_target_prior.rolling(32, min_periods=32).mean(), color=BLUE, lw=1.1, label='Model')
    ax.plot(data.trial, data.oral_target_mass.rolling(32, min_periods=32).mean(), color='#333333', lw=1.1, label='Oral')
    # Encoder reference, not an empirical or universal upper bound.
    ax.axhline(ideal_q[0], color='.6', ls=':', lw=.7)
    ax.set(xlim=(.5,320.5),ylim=(0,1.03),yticks=[0,.5,1],ylabel='Target mass')
    ax.set_title(f'Target-based alignment · full space · oral sigma = {oral_sigma:g}',loc='left',pad=4)
    ax.legend(loc='lower right',ncol=2,fontsize=6)
    ax.tick_params(labelbottom=False)
    ax = fig.add_axes([left,.349,width,.074])
    for i in range(4):
        x = data.trial.to_numpy()[(mentions[:,i]==1)&known]
        ax.vlines(x,i-.25,i+.25,color=BLUE,lw=.65)
    for x in data.trial[~known]: ax.axvspan(x-.5,x+.5,color='.9',lw=0)
    ax.set(ylim=(3.6,-.6),yticks=range(4),yticklabels=['F1','F2','F3','F4'],xlim=(.5,320.5),xlabel='Trial',ylabel='Reported\nfeatures')
    ax.tick_params(axis='y',length=0)
    for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    for a in fig.axes[:]:
        if a.get_xlim()[1] > 300:
            a.set_xticks([1,80,160,240,320])
    fig.text(.06,.289,'c   Behavioral checks for the available case',fontsize=10,weight='bold')
    ax = fig.add_axes([.17,.105,.29,.14])
    ax.plot([0,1],[0,1],ls='--',color='.7',lw=.7,zorder=0)
    ax.scatter(blocks.predicted,blocks.observed,s=24,color=BLUE)
    for row in blocks.itertuples():
        ax.annotate(str(row.block),(row.predicted,row.observed),xytext={3:(-9,4),5:(-5,9),6:(3,-10)}.get(row.block,(4,3)),textcoords='offset points',fontsize=6)
    ax.set(xlim=(.35,1),ylim=(.35,1),xticks=[.5,.75,1],yticks=[.5,.75,1],xlabel='Predicted accuracy',ylabel='Observed accuracy')
    ax.set_title('Eight consecutive 40-trial blocks',loc='left',pad=7)
    ax = fig.add_axes([.60,.105,.29,.14])
    ax.plot([0,1],[0,1],ls='--',color='.7',lw=.7,zorder=0)
    ax.plot(calibration.predicted,calibration.observed,'o-',color=BLUE,lw=1,ms=4)
    for row in calibration.itertuples():
        ax.annotate(f'n={row.n}',(row.predicted,row.observed),xytext=(0,7),ha='center',textcoords='offset points',fontsize=6)
    ax.set(xlim=(-.03,1.03),ylim=(-.03,1.12),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted P(choice = 2)',ylabel='Observed fraction choice = 2')
    ax.set_title('Choice calibration · five fixed bins',loc='left',pad=7)
    fig.text(.17,.035,f'Choice NLL = {nll:.3f} nats/trial   |   Uniform baseline = {np.log(2):.3f}   |   n = 1 participant',fontsize=7)
    fig.text(.17,.016,'Existing parameters; observed-history conditioning. No held-out evaluation or population inference.',fontsize=6.5,color='#555555')
    fig.savefig(output/'fig2bc_behavior_state_draft.png',dpi=450,facecolor='white')
    plt.close(fig)
    oral_matrix = oral_result['oral_mass'].T
    valid = np.isfinite(oral_matrix).all(axis=0)
    np.testing.assert_allclose(oral_matrix[:, valid].sum(axis=0), 1, atol=1e-8)
    overlap = np.minimum(prior, oral_matrix).sum(axis=0)
    pd.DataFrame(oral_matrix.T, columns=[f'H{i}' for i in range(29)]).to_csv(output/'oral_distribution.csv', index=False)
    pd.DataFrame({'trial':data.trial,'overlap':overlap}).to_csv(output/'distribution_overlap.csv',index=False)
    fig, axes = plt.subplots(3, 1, figsize=(183/25.4, 160/25.4), sharex=True,
                             gridspec_kw={'height_ratios':[2,2,1]})
    fig.subplots_adjust(left=.18,right=.89,top=.90,bottom=.10,hspace=.28)
    fig.suptitle(f'Full-space alignment · S101 · oral sigma = {oral_sigma:g}',fontsize=10)
    for ax, matrix, title in zip(axes[:2], [prior,oral_matrix], ['Model: online rule belief','Oral: latest report by category']):
        full_im = ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1,extent=[.5,320.5,28.5,-.5])
        ax.set_title(title,loc='left')
        ax.set_yticks([(a+b)/2 for a,b,_ in groups],labels=[label for _,_,label in groups])
        ax.tick_params(axis='y',length=0)
        for _, end, _ in groups[:-1]: ax.axhline(end+.5,color='.8',lw=.4)
    fig.colorbar(full_im,cax=fig.add_axes([.915,.38,.012,.50]),ticks=[0,.5,1])
    axes[2].plot(data.trial,overlap,color='.75',lw=.5,label='Trialwise')
    axes[2].plot(data.trial,pd.Series(overlap).rolling(32,min_periods=32).mean(),color=BLUE,lw=1.2,label='32-trial mean')
    axes[2].set(ylim=(0,1),ylabel='Distribution overlap',xlabel='Trial',xticks=[1,80,160,240,320])
    axes[2].legend(loc='lower right',ncol=2,fontsize=6)
    fig.text(.18,.028,'Overlap = sum of shared probability mass across all 29 rules; 1 = identical distributions.',fontsize=6.5)
    fig.savefig(output/'full_space_alignment_draft.png',dpi=450,facecolor='white')
    plt.close(fig)
    script = Path(__file__)
    shutil.copy2(script,output/script.name)
    manifest = {'subject':101,'condition':1,'n_subjects':1,'n_trials':320,'reports_available':int(known.sum()),
                'choice_nll':nll,'uniform_nll':float(np.log(2)),'observed_accuracy':float(correct.mean()),
                'predicted_accuracy':float(p.mean()),'n_rules':29,'oral_center_sigma':oral_sigma, 'ideal_report_target_mass':float(ideal_q[0]),'oral_state_mode':'latest_by_category','oral_alignment_space':'full','prediction_timing':'pre-current-choice; past observed history',
                'fit_scope':'0826 engine with previously selected 0818 parameters; no held-out evaluation',
                'pf_repeats':16,'particles_per_repeat':128,'rolling_window':32,'source_sha256':
                {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files+[script]}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--oral-sigma', type=float, default=0.05)
    args = parser.parse_args()
    build(args.output, args.oral_sigma)
