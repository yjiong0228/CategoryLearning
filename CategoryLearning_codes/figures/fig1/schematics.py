"""Matplotlib redraws of the stimulus, task contrasts and psychological trial flow."""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Ellipse, FancyBboxPatch, Polygon, Rectangle

FEATURE_COLORS = {"head": "#A6A443", "neck": "#BF925A", "tail": "#8D82B5", "leg": "#589AAF"}
FEATURE_LABELS = {"head": "Head", "neck": "Neck", "tail": "Tail", "leg": "Legs"}


def _direction(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array([np.cos(angle), np.sin(angle)])


def draw_animal(ax, box=(0., 0., 1., 1.), ranges=False, labels=False, linewidth=1.25):
    """Fixed angles, fixed body and continuous variable lengths; illustrative specimen.

    Normalized 0–1 maps to .25–1.25 body lengths as described in the experimental
    methods. Head range is shown at the illustrated neck length, so the two ranges
    are not mistaken for a single jointly varying feature.
    """
    left, bottom, width, height = box
    aspect = (ax.get_position().width * ax.figure.get_figwidth()
              / (ax.get_position().height * ax.figure.get_figheight()))
    scale = min(width / 6.3, height / (5.0 * aspect))
    origin = np.array([left + width*.43, bottom + height*.47])
    transform = lambda v: origin + np.asarray(v)*np.array([scale, scale*aspect])
    body_a, body_b = np.array([-.9,.0]), np.array([.9,.0])
    length = 1.35
    neck_tip = body_b + length*_direction(52)
    parts = [("tail",body_a,150), ("neck",body_b,52), ("head",neck_tip,-28),
             ("leg",body_a,-72), ("leg",body_a,-108),
             ("leg",body_b,-72), ("leg",body_b,-108)]
    body = np.array([transform(body_a),transform(body_b)])
    ax.plot(body[:,0],body[:,1],color="#222222",lw=linewidth,solid_capstyle="round",zorder=3)
    for name, anchor, degrees in parts:
        vec = _direction(degrees)
        if ranges:
            lengths = np.linspace(.45,2.25,31)
            points = np.array([transform(anchor + value*vec) for value in lengths])
            segments = np.stack([points[:-1],points[1:]],axis=1)
            colors = [(v,v,v,1) for v in np.linspace(.3,.88,30)]
            ax.add_collection(LineCollection(segments,colors=colors,linewidths=1.2,linestyles="dotted",zorder=1))
            for endpoint, shade in [(lengths[0],".35"),(lengths[-1],".86")]:
                p=transform(anchor + endpoint*vec);ax.scatter(*p,s=6,c=shade,linewidths=0,zorder=2)
        line=np.array([transform(anchor),transform(anchor+length*vec)])
        ax.plot(line[:,0],line[:,1],color="#252525",lw=linewidth,solid_capstyle="round",zorder=3)
    if labels:
        positions={"tail":(-2.55,1.40),"neck":(1.35,2.0),"head":(3.05,1.05),"leg":(-.1,-2.08)}
        for name, point in positions.items():
            p=transform(point);ax.text(*p,FEATURE_LABELS[name],ha="center",va="center",fontsize=6,
                                      color=FEATURE_COLORS[name],fontstyle="italic")


def stimulus_panel(ax, labels: bool = True) -> None:
    """Illustrative line-drawing stimulus; not an actual presented trial."""
    segments = [((.22,.55),(.66,.55),"#303030"), ((.22,.55),(.06,.8),"#777777"),
                ((.66,.55),(.78,.86),"#777777"), ((.78,.86),(.97,.70),"#777777"),
                ((.25,.55),(.19,.18),"#777777"), ((.25,.55),(.35,.18),"#777777"),
                ((.61,.55),(.53,.18),"#777777"), ((.61,.55),(.72,.18),"#777777")]
    for start, end, color in segments:
        ax.plot([start[0],end[0]],[start[1],end[1]],color=color,lw=1.4)
    if labels:
        for x,y,t in [(.02,.93,"Tail"),(.58,.96,"Neck"),(.88,.91,"Head"),(.35,.02,"Legs")]:
            ax.text(x,y,t,fontsize=6,ha="center")
    ax.set(xlim=(-.07,1.05),ylim=(-.08,1.15));ax.axis("off")


def task_panel(ax, task: dict) -> None:
    ax.axis("off"); ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.5,.98,f"Task {task['task']}",ha="center",va="top",color=task["color"],weight="bold",fontsize=7)
    ax.text(.5,.81,"2 categories" if task["task"]==1 else "4 categories",ha="center",fontsize=6.5)
    ax.text(.5,.65,"Feature 1",ha="center",fontsize=6)
    for x in [.25,.75]:
        ax.plot([.5,x],[.59,.43],color="#555555",lw=.7)
    if task["task"] == 1:
        for x,t in [(.25,"C1"),(.75,"C2")]: ax.text(x,.34,t,ha="center",fontsize=6.5)
    else:
        for x,t in [(.25,"Feature 2"),(.75,"Feature 3")]:
            ax.text(x,.38,t,ha="center",fontsize=5.7)
            for dx in [-.12,.12]: ax.plot([x,x+dx],[.32,.20],color="#555555",lw=.7)
        for x,t in zip([.13,.37,.63,.87],["C1","C2","C3","C4"]):ax.text(x,.12,t,ha="center",fontsize=6)
    text = "Feedback: 0 / 1" if task["task"] != 2 else "Feedback: 0 / 0.5 / 1"
    ax.text(.5,-.03,text,ha="center",fontsize=6)


def _microphone(ax,x,y,scale):
    aspect = (ax.get_position().width * ax.figure.get_figwidth()
              / (ax.get_position().height * ax.figure.get_figheight()))
    sx = scale / aspect
    ax.add_patch(FancyBboxPatch((x-sx*.16,y),sx*.32,scale*.52,
                              boxstyle="round,pad=0,rounding_size=.006",fc="#444451",ec="none",zorder=20))
    ax.add_patch(Arc((x,y+scale*.20),sx*.62,scale*.62,theta1=180,theta2=360,color="#444451",lw=.9,zorder=20))
    ax.plot([x,x],[y-scale*.11,y-scale*.32],color="#444451",lw=.9,zorder=20)
    ax.plot([x-sx*.20,x+sx*.20],[y-scale*.32]*2,color="#444451",lw=.9,zorder=20)


def procedure_panel(ax):
    """Overlapping display sequence: fixation, stimulus/choice, ready, report, feedback."""
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    width,height=.18,.69
    titles=["Fixation","Categorize","Prepare report","Speak","Feedback"]
    for i,title in enumerate(titles):
        x=.01+i*.197;y=.26-i*.055
        ax.add_patch(FancyBboxPatch((x+.006,y-.009),width,height,boxstyle="round,pad=.005,rounding_size=.018",fc=".90",ec="none",zorder=i*2))
        ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=.005,rounding_size=.018",fc="#C3C3C3",ec="white",lw=1,zorder=i*2+1))
        ax.text(x+width/2,y+height+.035,title,ha="center",fontsize=5.7,zorder=20)
        if i==0:ax.text(x+width/2,y+height*.53,"+",ha="center",va="center",fontsize=14,weight="bold",zorder=20)
        if i==1:
            draw_animal(ax,box=(x+.012,y+.20,width-.035,height-.22),linewidth=.8)
            for dx,label in [(.045,"F"),(.116,"J")]:
                ax.text(x+dx,y+.085,label,ha="center",va="center",fontsize=6,
                        bbox={"boxstyle":"circle,pad=.18","fc":".90","ec":".3","lw":.55},zorder=20)
        if i==2:ax.text(x+width/2,y+height*.5,"Ready…",ha="center",va="center",fontsize=7,weight="bold",zorder=20)
        if i==3:_microphone(ax,x+.078,y+.31,.34)
        if i==4:ax.text(x+width/2,y+height*.5,"1",ha="center",va="center",fontsize=13,weight="bold",zorder=20)
    ax.annotate("",xy=(.985,.002),xytext=(.025,.15),arrowprops={"arrowstyle":"->","lw":.8,"color":".5"})
    ax.text(.49,-.06,"Time  ·  verbal report before feedback",ha="center",fontsize=5.7,color=".4")
    ax.text(.015,-.19,"Two-category example; screen sequence shown schematically, not to scale.",fontsize=5.5,color=".45")
