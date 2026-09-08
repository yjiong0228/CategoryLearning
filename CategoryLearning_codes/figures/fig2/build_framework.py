"""Render a compact Model 0826 panel; schematic shapes are not fitted results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/categorylearning-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK='#303438'; MUTED='#69737A'; BLUE='#487DA8'; PALE='#F0F4F7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'mathtext.fontset':'dejavusans'})


def build(output):
    fig=plt.figure(figsize=(120/25.4,90/25.4))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,120),ylim=(0,90));ax.axis('off')
    def text(x,y,s,size=7,color=INK,ha='center',**kw):
        return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',**kw)
    def arrow(points,color=INK,dashed=False):
        if len(points)>2:
            ax.plot(*zip(*points[:-1]),color=color,lw=.8,ls=(0,(3,2)) if dashed else '-',solid_capstyle='round')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=7,
                                    linewidth=.8,color=color,linestyle=(0,(3,2)) if dashed else '-',shrinkA=0,shrinkB=0))
    def box(x,y,w,h,fc='white',ec='#C6CDD2',lw=.7):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.3',fc=fc,ec=ec,lw=lw))
    text(3,85,'a',10,ha='left',weight='bold')
    text(9,85,'Finite rule search and inference',8,ha='left')
    # A restrained shaded field denotes the unobserved cognitive calculations.
    box(28,15,73,64,fc=PALE,ec='none')
    text(31,76.5,'Prediction and belief update',6,color=MUTED,ha='left')
    text(12,75,'Stimulus',7)
    segments=[((5,66),(15,66)),((5,66),(2,70)),((15,66),(18,72)),((18,72),(22,69)),
              ((6,66),(4,60)),((6,66),(8,60)),((14,66),(12,60)),((14,66),(16,60))]
    for a,b in segments:ax.plot([a[0],b[0]],[a[1],b[1]],color=INK,lw=1)
    text(12,56,r'$\mathbf{x}_t$',8)
    arrow([(23,66),(31,66)])
    text(42,71.5,'Perception',7)
    text(42,65.5,r'$\widetilde{\mathbf{x}}_t=\mathrm{clip}(\mathbf{x}_t+\varepsilon_t)$',6.5)
    text(42,59.5,'Subject-specific noise',5.8,color=MUTED)
    arrow([(56,66),(68,66)])
    text(83,72,'Choice prediction',7)
    text(83,65,r'$p_t(c)$',10)
    text(82,58,'Belief mixture  /  persistent rule',5.6,color=BLUE)
    arrow([(99,66),(107,66)])
    text(112,74,'Choice',7)
    text(112,66,r'$y_t$',10)
    # Three equal cards show an illustrative M=3 workspace, with no fake posterior bars.
    box(31,32,41,23,fc='white',ec=BLUE,lw=.9)
    text(51.5,51,'Limited workspace',7,weight='bold')
    text(51.5,47,r'$M$ active rules: $A_t$',6.5,color=MUTED)
    for i in range(3):
        x=35+i*11
        ax.add_patch(Rectangle((x,36),8,8,fc='#F8FAFC',ec='#C4D0D9',lw=.65))
        # Abstract rule partitions; no specific hypothesis is declared correct.
        if i==0:ax.plot([x+4,x+4],[36.7,43.3],color=BLUE,lw=.85)
        elif i==1:ax.plot([x+.7,x+7.3],[36.7,43.3],color=BLUE,lw=.85)
        else:ax.plot([x+.7,x+7.3],[43.3,36.7],color=BLUE,lw=.85)
    text(51.5,28,'Rule beliefs '+r'$\pi_t^-$'+'  ·  precision '+r'$\beta_t(h)$',6.1,color=BLUE)
    arrow([(72,44),(82,44),(82,54)],BLUE)
    # Choice and feedback act after prediction. Both inform cognitive updates.
    text(112,29.5,'Task\nfeedback',6.5)
    text(112,23,r'$r_t$',10)
    arrow([(112,61),(112,33)])
    box(75,16,25,19,fc='white',ec='#AAB9C4')
    text(87.5,30,'Fading belief update',6.1)
    text(87.5,24,r'$\pi_t^+\propto(\pi_t^-)^{\gamma}L_t$',7.5)
    text(87.5,19,'Update rule precision',6.1)
    arrow([(107,23),(101,23)])
    # The observed choice is retained when evaluating feedback support for each rule.
    arrow([(109,61),(103,61),(103,38),(95,38),(95,35)],color=MUTED)
    arrow([(76,35),(68,35)],BLUE,True)
    text(75,38.5,r'$t+1$',5.7,color=BLUE)
    box(3,18,23,24,fc='white',ec=BLUE)
    text(14.5,37,'Rule search',7,weight='bold')
    text(14.5,31,'Search event '+r'$E$',6.3)
    text(14.5,25,'Local / global '+r'$g$',6.3)
    text(14.5,20,'Failure history '+r'$F$',5.8,color=MUTED)
    arrow([(26,36),(30,36)],BLUE,True)
    text(17,46,'Replace candidates',5.8,color=BLUE)
    arrow([(112,18),(112,11),(14.5,11),(14.5,17)],color=MUTED)
    text(64,11,'Feedback controls next-trial search',5.8,color=MUTED,
         bbox={'facecolor':'white','edgecolor':'none','pad':1})
    text(4,5,'PF: choice-weighted inference over latent paths',6,color=MUTED,ha='left')
    text(117,5,'Dashed: next trial',5.7,color=BLUE,ha='right')
    fig.savefig(output/'fig2a_framework_draft.png',dpi=450,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    build(args.output)
    shutil.copy2(__file__,args.output/'build_framework.py')
    sources=[Path('manuscript/model_0826.tex'),Path('manuscript/figures/model_0826_framework.png'),
             Path('CategoryLearning_paper/references/Weiss et al_2021_Interacting with volatile environments stabilizes hidden-state inference and.pdf')]
    manifest={'status':'schematic_draft_not_results','size_mm':[120,90],'dpi':450,'backend':'matplotlib',
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(args.output/'fig2a_framework_draft.png')

if __name__=='__main__':main()
