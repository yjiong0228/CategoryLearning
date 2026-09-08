"""Mechanism-first illustration, with an explicitly synthetic one-trial example.

Numerical bars follow the manuscript update equations; graph positions only
illustrate local versus global search and are not a similarity embedding.
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse, Polygon

BLUE='#487DA8'
INK='#303438'
GRAY='#9AA7AF'
PALE='#EDF3F7'
ERROR='#B57957'


def illustrative_episode():
    """Analytic binary example; these values are not fitted subject parameters."""
    prior=np.array([.60,.25,.15])
    beta=np.full(3,8.)
    # Signed distance d(category 1) - d(category 2) at F1=.25,F3=.75.
    signed_distance=np.array([.25,-.25,0.])
    p2=1/(1+np.exp(-beta*signed_distance))
    support=1-p2  # observed choice=2, feedback=0
    gamma=.8
    faded=prior**gamma
    faded/=faded.sum()
    posterior=faded * support
    posterior/=posterior.sum()
    kappa=2*(support-.5)
    eta_plus,eta_minus=.25,.25
    after=np.where(kappa>=0,beta+eta_plus*kappa*(25-beta),beta-eta_minus*beta*(-kappa))
    assert np.isclose(prior.sum(),1) and np.isclose(posterior.sum(),1)
    assert faded[0]<prior[0] and faded[-1]>prior[-1]
    assert np.array_equal(np.argsort(faded),np.argsort(prior))
    assert np.isclose(faded.sum(),1)
    assert posterior[0]<prior[0] and posterior[1]>prior[1]
    assert after[0]<beta[0] and after[1]>beta[1]
    return {'status':'illustrative_not_fitted','stimulus_F1_F3':[.25,.75],
            'prior':prior.tolist(),'p_choice2_by_rule':p2.tolist(),
            'mixture_p_choice2':float(prior@p2),'persistent_h1_p_choice2':float(p2[0]),
            'choice':2,'feedback':0,'gamma':gamma,'faded_prior':faded.tolist(),'feedback_support':support.tolist(),'posterior':posterior.tolist(),
            'beta_before':beta.tolist(),'beta_after':after.tolist(),
            'eta_plus':eta_plus,'eta_minus':eta_minus,'capacity':3,
            'possible_replacement':'h3 -> h4; h1 retained even if it is the execution rule'}


def draw_framework(fig, rect):
    ep=illustrative_episode()
    ax=fig.add_axes(rect);ax.set(xlim=(-.5,164.5),ylim=(-.5,75));ax.axis('off')
    def txt(x,y,s,size=6.2,ha='center',color=INK,**kw):
        ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,**kw)
    def arrow(start,end,color=GRAY,lw=.9,rad=0,dashed=False):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=7,
                                    lw=lw,color=color,connectionstyle=f'arc3,rad={rad}',
                                    linestyle='--' if dashed else '-',shrinkA=1,shrinkB=1))
    def rule(x,y,index,w=8,point=False,new=False):
        # Same category shading across all rules: category 1 blue, category 2 white.
        ax.add_patch(Rectangle((x,y),w,w,fc='white',ec=BLUE if new else '#BBC8D1',lw=1.1 if new else .65))
        if index==1:
            ax.add_patch(Rectangle((x,y),w,w/2,fc=PALE,ec='none'))
            ax.plot([x,x+w],[y+w/2]*2,color=BLUE,lw=.8)
        elif index==2:
            ax.add_patch(Rectangle((x,y),w/2,w,fc=PALE,ec='none'))
            ax.plot([x+w/2]*2,[y,y+w],color=BLUE,lw=.8)
        elif index==3:
            ax.add_patch(Polygon([(x,y),(x+w,y),(x,y+w)],fc=PALE,ec='none'))
            ax.plot([x,x+w],[y+w,y],color=BLUE,lw=.8)
        else:
            ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+w)],fc=PALE,ec='none'))
            ax.plot([x,x+w],[y,y+w],color=BLUE,lw=.8)
        if point:ax.plot(x+.25*w,y+.75*w,'o',ms=2,color=INK,zorder=6)
    def probbar(x,y,p2,w=14,h=2.5):
        ax.add_patch(Rectangle((x,y),w*(1-p2),h,fc=BLUE,ec='none'))
        ax.add_patch(Rectangle((x+w*(1-p2),y),w*p2,h,fc='#CCD4D9',ec='none'))
    txt(0,72,'a   Finite rule search',8,ha='left',weight='bold')
    txt(164,72,'Illustrative trial',6,ha='right',color=GRAY)
    # Current trial: a common perceived stimulus is evaluated by every active rule.
    txt(0,64,'Predict',7,ha='left',weight='bold')
    txt(12,58,'Stimulus',6.5)
    ax.plot([4,4,21],[39,55,55],color='#D4DBDF',lw=.6)
    for j,value in enumerate([.25,.5,.75,.5]):
        ax.plot([7+j*4]*2,[40,53],color='#E4E8EB',lw=2.5)
        ax.plot([7+j*4]*2,[40,40+13*value],color='#71848F',lw=2.5)
        txt(7+j*4,37.5,f'F{j+1}',5)
    # Perceptual jitter around stimulus values is schematic, and not oral sigma.
    ax.plot([7,7],[42.2,44.5],color=INK,lw=.7)
    txt(12,33.5,r'$\mathbf{x}_t\rightarrow\widetilde{\mathbf{x}}_t$',7)
    arrow((23,47),(30,47),color=INK)
    ax.add_patch(FancyBboxPatch((31,33),48,28,boxstyle='round,pad=0,rounding_size=1.2',fc='#F7F9FA',ec='#CFD9DF',lw=.6))
    txt(55,64,r'Workspace $A_t$ · $M=3$',7,weight='bold')
    txt(57,58.5,'Belief',5.5,color=GRAY)
    txt(73,58.5,'Prediction',5.5,color=GRAY)
    ys=[52,43,34]
    for i,y in enumerate(ys):
        rule(33,y,i+1,7,point=True)
        txt(43,y+3.5,rf'$h_{i+1}$',7)
        ax.add_patch(Rectangle((48,y+2.5),15,2,fc='#E2E8EC',ec='none'))
        ax.add_patch(Rectangle((48,y+2.5),15*ep['prior'][i],2,fc=BLUE,ec='none'))
        probbar(67,y+2,ep['p_choice2_by_rule'][i],9,3)
    # Alternative readouts: a weighted fan-in versus a single execution path.
    txt(106,64,'Readout',7,weight='bold')
    for base,mode,p2 in [(52,'Mixture',ep['mixture_p_choice2']),(38,'Persistent',ep['persistent_h1_p_choice2'])]:
        txt(108,base+8,mode,6.3)
        for i in range(3):
            y=base+3-i*3
            ax.add_patch(Circle((89,y),.7,fc=BLUE if mode=='Mixture' or i==0 else '#DCE2E6',ec='none'))
            ax.plot([89.8,99],[y,base],color=BLUE if mode=='Mixture' or i==0 else '#DEE4E8',
                    lw=(.4+2.5*ep['prior'][i]) if mode=='Mixture' else 1.8 if i==0 else .4,zorder=1)
        ax.add_patch(Circle((100,base),1,fc=BLUE,ec='none'))
        arrow((101,base),(105,base))
        probbar(106,base-1.5,p2,14,3)
    arrow((80,47),(84,47),color=INK)
    txt(106,33.5,'1',5.5,ha='left',color=BLUE);txt(120,33.5,'2',5.5,ha='right',color=GRAY)
    txt(140,59,'Choice',6.5)
    ax.add_patch(Circle((140,49),3,fc='#D9E0E5',ec='#74858E',lw=.7));txt(140,49,'2',8,weight='bold')
    # Both modes can yield the shown response; no claim that both are run simultaneously.
    arrow((121,52),(135,49),color=GRAY)
    arrow((121,38),(135,48),color=GRAY,rad=.2)
    txt(158,59,'Feedback',6.5)
    arrow((144,49),(151,49),color=INK)
    txt(158,49,r'$\times$',16,color=ERROR)
    txt(158,41,'Incorrect',5.8,color=ERROR)
    ax.plot([157,144,144],[37,32,22],color=ERROR,lw=.9)
    arrow((144,22),(140,18),color=ERROR)
    # Factorize the same update to expose memory's effect. The intermediate
    # normalized power is a visualization, not an additional stored model state.
    ax.add_patch(FancyBboxPatch((86,7),37,20,boxstyle='round,pad=0,rounding_size=1',fc='#F0F4F7',ec='#CBD8E0',lw=.6))
    txt(105,29,'Memory',7,weight='bold')
    txt(134,29,'Update',7,weight='bold')
    for x,values,title in [(89,ep['prior'],'Prior'),(110,ep['faded_prior'],'Faded'),(129,ep['posterior'],'Updated')]:
        txt(x+4,24,title,5.5)
        for i,value in enumerate(values):
            ax.add_patch(Rectangle((x+i*3,10),2.2,15*value,fc=BLUE,ec='none'))
            txt(x+i*3+1,8.3,rf'$h_{i+1}$',5)
        ax.plot([x-.5,x+8.5],[10,10],color=GRAY,lw=.5)
    # Before-memory outlines make the contraction visible at this small scale.
    for i,value in enumerate(ep['prior']):
        ax.add_patch(Rectangle((110+i*3,10),2.2,15*value,fc='none',ec=GRAY,lw=.6,ls=':'))
    arrow((99,17),(108,17),color=BLUE)
    txt(104,20.5,r'$\gamma=0.8$',5.8,color=BLUE)
    arrow((120,17),(128,17),color=ERROR)
    txt(124,20.5,r'$\times L_t$',6,color=ERROR)
    # A supported rule sharpens and a contradicted rule flattens its readout.
    txt(151,24,'Precision',6)
    dx=np.linspace(-.5,.5,80)
    for index,y in [(1,16.5),(0,7)]:
        ax.plot([143,160],[y+2.5]*2,color='#E1E6E9',lw=.5)
        for beta,ls,col in [(8,'--',GRAY),(ep['beta_after'][index],'-',BLUE)]:
            ax.plot(143+(dx+.5)*17,y+5/(1+np.exp(-beta*dx)),ls=ls,color=col,lw=.9)
        txt(162,y+2.5,rf'$h_{index+1}$',5.5)
    txt(151,3,'Boundary distance',5.2,color=GRAY)
    # Search-space diagram: nodes are rules, not fitted geometric coordinates.
    txt(63,29,'Search',7,weight='bold')
    positions=np.array([[55,11],[60,19],[68,14],[73,22],[79,12],[88,19],[95,11],[91,25],[53,25]],dtype=float)
    positions[:,0]=45+(positions[:,0]-50)*.70
    for i,j in [(0,1),(1,2),(2,3),(2,4),(3,5),(4,5),(5,6),(5,7),(1,8)]:
        ax.plot(positions[[i,j],0],positions[[i,j],1],color='#E0E6E9',lw=.7,zorder=0)
    # Local kernel contributions are centered on all active candidates, with
    # differing intensity after feedback. Global proposals include distant nodes.
    for idx,alpha in [(1,.12),(2,.28),(4,.16)]:
        x,y=positions[idx];ax.add_patch(Circle((x,y),3.2,fc=BLUE,alpha=alpha,ec='none',zorder=0))
    for idx,(x,y) in enumerate(positions):
        ax.add_patch(Circle((x,y),1.2,fc=BLUE if idx in [1,2,4] else 'white',ec=BLUE if idx in [1,2,4] else GRAY,lw=.7))
    txt(52,19,r'$h_1$',5,color='white');txt(57.6,14,r'$h_2$',5,color='white');txt(65.3,12,r'$h_3$',5,color='white')
    # Two alternative proposal routes (rather than forcing search after every error).
    arrow((59,16),(60.4,21),color=BLUE,lw=.8)
    arrow((79.3,25),(75.1,25),color=GRAY,rad=-.15,dashed=True)
    txt(56.2,25,'Local',5.5,color=BLUE);txt(75.1,29,'Global',5.5,color=GRAY)
    ax.add_patch(Circle((61.1,22),2.3,fill=False,ec=BLUE,lw=.8))
    txt(61.1,22,r'$h_4$',5,color=BLUE)
    txt(62.5,4.5,r'$1-E$ retain  ·  $E$ search',6)
    ax.plot([138,138,83,83],[9,5,5,18],color=GRAY,lw=.7)
    arrow((83,18),(79,18),color=GRAY)
    # One possible replacement outcome: retained h1/h2 and new h4, no made-up
    # post-transport belief values. Persistent h1 is protected in this example.
    txt(0,29,r'$t+1$ · Possible replacement',7,ha='left',weight='bold')
    ax.add_patch(FancyBboxPatch((1,9),39,16,boxstyle='round,pad=0,rounding_size=1.1',fc='#F7F9FA',ec='#CFD9DF',lw=.7))
    for x,index in [(4,1),(16,2),(28,4)]:
        rule(x,14,index,8,new=index==4)
        txt(x+4,11.5,rf'$h_{index}$',6.3,color=BLUE if index==4 else INK)
    arrow((45,18),(42,18),color=BLUE)
    txt(21,4.5,r'$\{h_1,h_2,h_3\}\rightarrow\{h_1,h_2,h_4\}$',7)
    txt(0,.9,'PF averages possible latent paths; diagram shows one illustrative episode.',5.6,ha='left',color=GRAY)
    return ax
