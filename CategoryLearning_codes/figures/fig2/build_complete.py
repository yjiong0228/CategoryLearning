"""Whole Fig2 layout: existing evidence plus explicitly empty result panels."""
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
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from CategoryLearning_codes.figures.fig2.build_behavior_state import build as build_case

ROOT=Path(__file__).resolve().parents[3]

def build(output, case_sources=None, task3_case_sources=None):
    output.mkdir(parents=True,exist_ok=False)
    if case_sources is None:
        build_case(output/'case_sources',oral_sigma=.05)
    else:
        shutil.copytree(case_sources, output/'case_sources')
    case = json.loads((output/'case_sources/manifest.json').read_text())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':6.5,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(183/25.4,240/25.4))
    from CategoryLearning_codes.figures.fig2.framework_wide import draw_framework, illustrative_episode
    framework=Path(__file__).with_name('framework_wide.py')
    draw_framework(fig,[.075,.665,.895,.309])
    standalone=plt.figure(figsize=(163.785/25.4,74.16/25.4))
    draw_framework(standalone,[0,0,1,1])
    standalone.savefig(output/'fig2a_framework_wide.png',dpi=450,facecolor='white')
    plt.close(standalone)
    shutil.copy2(framework,output/framework.name)
    (output/'schematic_episode.json').write_text(json.dumps(illustrative_episode(),indent=2)+'\n')
    # The framework and quantitative panels use native matplotlib artists.
    def blank(rect,title):
        x,y,w,h=rect
        fig.text(x,y+h+.014,title,fontsize=8,weight='bold',va='bottom')
        fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor='white',edgecolor='#BDC5CA',lw=.8))
    blank((.405,.272,.245,.344),'c   Task 2 · Individual example')
    panels = [('b', .075, output/'case_sources', 1)]
    if task3_case_sources is None:
        blank((.725,.272,.245,.344),'d   Task 3 · Individual example')
    else:
        shutil.copytree(task3_case_sources, output/'task3_case_sources')
        panels.append(('d', .695, output/'task3_case_sources', 3))
    panel_metadata = {}
    readout_notes = []
    for letter, x, source_dir, task in panels:
        metadata = json.loads((source_dir/'manifest.json').read_text())
        expected_condition = 1 if task == 1 else 2
        assert metadata.get('condition', 1) == expected_condition
        assert metadata['oral_center_sigma'] == .05
        subject, n_trials = metadata['subject'], metadata['n_trials']
        d = pd.read_csv(source_dir/'trial_source.csv')
        b = pd.read_csv(source_dir/'belief_source.csv').pivot(index='hypothesis', columns='trial', values='online_prior').to_numpy()
        o = pd.read_csv(source_dir/'oral_distribution.csv').to_numpy().T
        n_rules = b.shape[0]
        target_hypothesis = metadata.get('target_hypothesis', 0 if task == 1 else 42)
        assert b.shape == o.shape == (n_rules, n_trials)
        assert len(d) == n_trials and 0 <= target_hypothesis < n_rules
        fitted = metadata.get('case_label') == 'Individually fitted PMH'
        readout = 'persistent execution' if metadata.get('persistent_execution', False) else 'mixture readout'
        readout_notes.append(f'S{subject}: {readout}')
        panel_metadata[letter] = {'subject':subject, 'task':task, 'condition':expected_condition,
            'n_trials':n_trials, 'n_rules':n_rules, 'target_hypothesis':target_hypothesis,
            'source':str(source_dir.relative_to(output)/'manifest.json')}
        fig.text(x,.638,f'{letter}   Task {task} · S{subject}',fontsize=8,weight='bold')
        fig.text(x,.622,f'{"Fitted PMH" if fitted else "Existing case"} · {n_trials} trials',fontsize=6,color='.4')
        left,width=x+.015,.24
        ax=fig.add_axes([left,.552,width,.061])
        ax.plot(d.trial,d.observed_accuracy_rolling32,color='.2',lw=.9,label='Observed')
        ax.plot(d.trial,d.model_correct_probability_rolling32,color='#487DA8',lw=.9,label='Model')
        ax.set(ylim=(0,1),ylabel='Accuracy',yticks=[0,.5,1],xlim=(1,n_trials))
        ax.tick_params(labelbottom=False);ax.legend(loc='lower right',fontsize=5,ncol=2,handlelength=1,columnspacing=.6)
        cmap=LinearSegmentedColormap.from_list('rule_blue',['white','#487DA8','#174568'])
        for bottom,matrix,title in [(.452,b,'Model: pre-choice belief'),(.352,o,'Oral: latest by category')]:
            ax=fig.add_axes([left,bottom,width,.071])
            im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest',extent=[.5,n_trials+.5,n_rules-.5,-.5])
            ax.set_title(title,loc='left',pad=3)
            ticks = [0,9,19,28] if n_rules == 29 else [0,42,77,115]
            ax.set_yticks(ticks,labels=[f'H{i}' for i in ticks]);ax.tick_params(labelbottom=False,length=2)
        cb=fig.colorbar(im,cax=fig.add_axes([x+.265,.352,.008,.171]),ticks=[0,.5,1]);cb.ax.tick_params(labelsize=5,length=2)
        ax=fig.add_axes([left,.272,width,.055])
        target = pd.DataFrame({'trial':d.trial,'model_target_mass':b[target_hypothesis],'oral_target_mass':o[target_hypothesis]})
        for column, color, label in [('model_target_mass','#487DA8','Model'),('oral_target_mass','.2','Oral')]:
            target[column+'_rolling32'] = target[column].rolling(32,min_periods=32).mean()
            ax.plot(d.trial,target[column+'_rolling32'],color=color,lw=.9,label=label)
        target.to_csv(source_dir/'target_alignment.csv',index=False)
        ax.legend(loc='upper left',fontsize=5,ncol=2,handlelength=1,columnspacing=.6,frameon=False)
        ax.set(ylim=(0,1),ylabel=f'Target H{target_hypothesis}',xlabel='Trial',xticks=[1,n_trials//2,n_trials],yticks=[0,.5,1],xlim=(1,n_trials))
    for x,title in [(.075,'e   Group behavioral prediction'),(.405,'f   Group oral–model alignment'),(.725,'g   Key model ablations')]:
        blank((x,.065,.245,.147),title)
    fig.text(.075,.035,'32-trial trailing means · oral sigma = 0.05 · target alignment = full-space probability mass.',fontsize=6,color='.4')
    note = '; '.join(readout_notes) + '. Full-sequence fits; no held-out evaluation.' if task3_case_sources else (
        f"S{case['subject']}: full-sequence PMH fit · mixture readout · no held-out evaluation." if case.get('case_label') == 'Individually fitted PMH' else
        'Task 1 uses transferred parameters; no held-out evaluation.')
    fig.text(.075,.023,note,fontsize=6,color='.4')
    fig.savefig(output/'Figure2_draft.png',dpi=450,facecolor='white')
    from matplotlib.transforms import Bbox
    case_box = Bbox.from_extents(.025,.240,.38,.655).transformed(fig.transFigure).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output/'fig2b_individual_detail.png',dpi=900,facecolor='white',bbox_inches=case_box)
    if task3_case_sources is not None:
        # Exclude neighboring placeholder artists from the enlarged case crop.
        neighbors = list(fig.artists) + [text for text in fig.texts if text.get_position()[0] < .69]
        for artist in neighbors:
            artist.set_visible(False)
        detail_box = Bbox.from_extents(.645,.240,1,.655).transformed(fig.transFigure).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output/'fig2d_individual_detail.png',dpi=900,facecolor='white',bbox_inches=detail_box)
    plt.close(fig)
    shutil.copy2(__file__,output/'build_complete.py')
    manifest={'status':'layout_draft_with_empty_panels','size_mm':[183,240],'dpi':450,'oral_sigma':.05,
              'populated':['a']+list(panel_metadata),'empty':[p for p in 'bcdefg' if p not in panel_metadata],'panels':panel_metadata,'alignment_panel':'target_based_full_space','case_source':'case_sources/manifest.json','subject':case['subject'],'n_trials':case['n_trials'],
              'framework_sha256':hashlib.sha256(framework.read_bytes()).hexdigest(),
              'figure_sha256':hashlib.sha256((output/'Figure2_draft.png').read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output/'Figure2_draft.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-sources',type=Path,help='Previously exported fitted case; omit for historical S101.')
    parser.add_argument('--task3-case-sources',type=Path,help='Exported condition-2 fitted case for panel d.')
    args=parser.parse_args()
    build(args.output,args.case_sources,args.task3_case_sources)
