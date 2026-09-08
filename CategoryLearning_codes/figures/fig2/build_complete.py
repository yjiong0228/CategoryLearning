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

def build(output):
    output.mkdir(parents=True,exist_ok=False)
    build_case(output/'case_sources',oral_sigma=.05)
    d=pd.read_csv(output/'case_sources/trial_source.csv')
    b=pd.read_csv(output/'case_sources/belief_source.csv').pivot(index='hypothesis',columns='trial',values='online_prior').to_numpy()
    o=pd.read_csv(output/'case_sources/oral_distribution.csv').to_numpy().T
    overlap=pd.read_csv(output/'case_sources/distribution_overlap.csv')
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
    fig.text(.075,.638,'b   Task 1 · S101',fontsize=8,weight='bold')
    fig.text(.075,.622,'Existing case · 320 trials',fontsize=6,color='.4')
    blank((.405,.272,.245,.344),'c   Task 2 · Individual example')
    blank((.725,.272,.245,.344),'d   Task 3 · Individual example')
    left,width=.09,.24
    ax=fig.add_axes([left,.552,width,.061])
    ax.plot(d.trial,d.observed_accuracy_rolling32,color='.2',lw=.9,label='Observed')
    ax.plot(d.trial,d.model_correct_probability_rolling32,color='#487DA8',lw=.9,label='Model')
    ax.set(ylim=(0,1),ylabel='Accuracy',yticks=[0,.5,1],xlim=(1,320))
    ax.tick_params(labelbottom=False);ax.legend(loc='lower right',fontsize=5,ncol=2,handlelength=1,columnspacing=.6)
    cmap=LinearSegmentedColormap.from_list('rule_blue',['white','#487DA8','#174568'])
    for bottom,matrix,title in [(.452,b,'Model rule belief'),(.352,o,'Oral rule distribution')]:
        ax=fig.add_axes([left,bottom,width,.071])
        im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest',extent=[.5,320.5,28.5,-.5])
        ax.set_title(title,loc='left',pad=3)
        ax.set_yticks([0,9,19,28],labels=['H0','H9','H19','H28']);ax.tick_params(labelbottom=False,length=2)
    cb=fig.colorbar(im,cax=fig.add_axes([.34,.352,.008,.171]),ticks=[0,.5,1]);cb.ax.tick_params(labelsize=5,length=2)
    ax=fig.add_axes([left,.272,width,.055])
    ax.plot(d.trial,overlap.overlap.rolling(32,min_periods=32).mean(),color='#487DA8',lw=.9)
    ax.set(ylim=(0,1),ylabel='Overlap',xlabel='Trial',xticks=[1,160,320],yticks=[0,.5,1],xlim=(1,320))
    for x,title in [(.075,'e   Group behavioral prediction'),(.405,'f   Group oral–model alignment'),(.725,'g   Key model ablations')]:
        blank((x,.065,.245,.147),title)
    fig.text(.075,.025,'Draft layout · oral sigma = 0.05 · Task 1 uses transferred parameters; no held-out evaluation.',fontsize=6,color='.4')
    fig.savefig(output/'Figure2_draft.png',dpi=450,facecolor='white')
    plt.close(fig)
    shutil.copy2(__file__,output/'build_complete.py')
    manifest={'status':'layout_draft_with_empty_panels','size_mm':[183,240],'dpi':450,'oral_sigma':.05,
              'populated':['a','b'],'empty':['c','d','e','f','g'],'case_source':'case_sources/manifest.json',
              'framework_sha256':hashlib.sha256(framework.read_bytes()).hexdigest(),
              'figure_sha256':hashlib.sha256((output/'Figure2_draft.png').read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output/'Figure2_draft.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
