"""Plot the joint-reason diagnostic from verified original model states.

Core conclusion: a report-wide choice effect can coexist with almost no joint
potential effect for every mentioned feature. Panel a uses explicitly synthetic
examples; b–d show all35 complete later reports of development subject107.
Archetype: quantitative grid. Python/matplotlib,183×135mm,300dpi PNG only.
Source tables retain both phases, weights and semantic readings. One original
512-particle filter per policy; no across-filter or population uncertainty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import digest,load
from src.Bayesian_state.evaluation.model_0923_joint_reason_contexts import full_context_effects

COLORS={'MIX':'#3876A6','SELECT':'#C37A38'}


def run(results: Path,validation: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new joint-reason figure directory')
    inputs={}
    def bind(path):
        path=Path(path).resolve();sha=digest(path);inputs[str(path)]=sha;return path
    verified=load(bind(validation/'completion.json'))
    assert verified['status']=='complete' and verified['complete_reports_per_cell']==75
    context_verified=load(bind(validation/'contexts_validation.json'))
    assert context_verified['status']=='complete'
    assert digest(bind(validation/'all_context_summary.csv'))==context_verified['summary_sha256']
    summary=pd.read_csv(validation/'all_context_summary.csv')
    selected=summary[(summary.weighting=='choice_conditioned')&(summary.phase=='later')&(summary.context_scope=='all_features')]
    assert len(selected)==4
    tables={};curve_tables={}
    for policy in ('MIX','SELECT'):
        for reading in ('midpoint','relative'):
            source=results/f'joint_reason_contexts_s107_{policy.lower()}_{reading}_20261020_v1'
            done=load(bind(source/'completion.json'));assert done['status']=='complete'
            for name in ('summary.csv','threshold_curves.csv','report_trials.csv'):
                assert digest(bind(source/name))==done['artifacts_sha256'][name]
                tables[policy+'_'+reading+'_'+name]=source/name
            coverage=results/f'joint_reason_profile_s107_{policy.lower()}_{reading}_20261020_v1'/'coverage.csv'
            bind(coverage);tables[policy+'_'+reading+'_coverage.csv']=coverage
            counts=pd.read_csv(coverage)
            assert len(counts)==192 and counts.query("phase=='later' and profile_available").shape[0]==35
            curve_tables[policy,reading]=pd.read_csv(source/'threshold_curves.csv').query(
                "weighting=='choice_conditioned' and phase=='later' and kind=='potential_relevance' and context_scope=='all_features'")
        np.testing.assert_array_equal(curve_tables[policy,'midpoint'].mass_above,curve_tables[policy,'relative'].mass_above)
    mask=((np.arange(4)[:,None]>>np.arange(2))&1).astype(bool)
    examples={'One or the other':(np.array([[.9,.1,.9,.1],[.9,.9,.1,.1]]),[True,True]),
              'Both in each state':(np.array([[.9,.5,.5,.1]]),[True,True]),
              'Redundant OR':(np.array([[.9,.9,.9,.1]]),[True,True]),
              'OR; one reported':(np.array([[.9,.9,.9,.1]]),[True,False]),
              'Unused extra feature':(np.array([[.9,.1,.9,.1]]),[True,True])}
    synthetic=[]
    for name,(q,reported) in examples.items():
        e=full_context_effects(q,mask,np.array(reported))
        synthetic.append(dict(example=name,mean_feature_effect=float(e['mean_positive_effect'].mean()),
                              within_state_minimum=float(e['minimum_positive_effect'].mean())))
    synthetic=pd.DataFrame(synthetic)
    output.mkdir(parents=True)
    synthetic.to_csv(output/'synthetic_examples.csv',index=False)
    shutil.copyfile(validation/'all_context_summary.csv',output/'all_summary.csv')
    for name,source in tables.items():shutil.copyfile(source,output/name)
    contract=dict(core_conclusion='Whole-report reflection effects can be carried by a subset of the reported features',
        archetype='quantitative_grid',backend='python',size_mm=[183,135],export=['PNG'],dpi=300,
        observed_scope='Subject107;35complete later reports;one original512-particle filter per policy',
        synthetic_scope='Five explicitly labelled analytic probe tables in panel a',
        statistics='Conditional particle-weighted effects averaged equally across the same reports; no inferential intervals',
        data_selection='Panels b–d show all35complete later reports with choice-conditioned weights; source tables retain40training reports and prior weights',
        integrity='Original data and reports preserved; thresholds show effect scales, no selected acceptance threshold')
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    # 183 × 135 mm; explicit inch values keep the static size audit unambiguous.
    fig,axes=plt.subplots(2,2,figsize=(7.20472440945,5.31496062992))
    fig.subplots_adjust(left=.19,right=.975,bottom=.23,top=.865,wspace=.94,hspace=.80)
    fig.suptitle('Report-wide effects and joint feature relevance',fontsize=10,y=.975)
    for ax,label in zip(axes.flat,'abcd'):
        ax.text(-.18,1.10,label,transform=ax.transAxes,weight='bold',fontsize=9)
    ax=axes[0,0];y=np.arange(5)
    ax.scatter(synthetic.mean_feature_effect,y-.08,c='#8C969D',s=22,marker='o',label='Mean over features')
    ax.scatter(synthetic.within_state_minimum,y+.08,c='#242C32',s=22,marker='D',label='Minimum within state')
    ax.set_yticks(y,synthetic.example);ax.invert_yaxis();ax.set_xlim(-.04,.86)
    ax.set_xticks([0,.4,.8]);ax.set_xlabel('Potential positive effect')
    ax.set_title('Synthetic examples\nGrey: mean; black: within-state minimum',fontsize=7,pad=11)
    ax=axes[0,1]
    metrics=['report_reflection_effect','mean_positive_effect_mean','minimum_positive_effect_mean']
    labels=['All reported\nfeatures reflected','Mean reported\nfeature effect','Within-state\nminimum effect']
    for policy,offset,marker in [('MIX',-.08,'o'),('SELECT',.08,'s')]:
        row=selected[(selected.policy==policy)&(selected.reading=='midpoint')].iloc[0]
        ax.scatter([row[k] for k in metrics],np.arange(3)+offset,color=COLORS[policy],marker=marker,s=22,label=policy)
    ax.set_yticks(range(3),labels);ax.set_ylim(2.45,-.3);ax.set_xlim(-.04,.75);ax.set_xticks([0,.25,.5,.75])
    ax.set_xlabel('Mean change in choice probability');ax.set_title('107: 35 complete later reports',fontsize=8,pad=11)
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,fontsize=7,loc='center',bbox_to_anchor=(.79,.925),ncol=2,columnspacing=1.,handletextpad=.4)
    ax.text(.10,2.03,'MIX: 1.6×10⁻⁷\nSELECT: 0',fontsize=6.7,va='center')
    ax=axes[1,0]
    for policy,marker in [('MIX','o'),('SELECT','s')]:
        table=curve_tables[policy,'midpoint'];positive=table[table.threshold>0]
        ax.semilogx(positive.threshold,positive.mass_above,color=COLORS[policy],marker=marker,markersize=2.5,lw=1.1)
    ax.set_ylim(-.01,.28);ax.set_xlim(1e-10,1.);ax.set_xticks([1e-10,1e-6,1e-2,1.])
    ax.set_xlabel('Minimum-effect threshold ε');ax.set_ylabel('State mass above ε')
    ax.set_title('Keep the effect scale visible',fontsize=8,pad=11)
    ax.text(.03,.90,'At ε=0: MIX 0.251; SELECT 0',transform=ax.transAxes,fontsize=6.4)
    ax=axes[1,1]
    for policy,offset,marker in [('MIX',-.05,'o'),('SELECT',.05,'s')]:
        rows=selected[selected.policy==policy].set_index('reading').loc[['midpoint','relative']]
        ax.scatter(np.arange(2)+offset,rows.numeric_report_truth_mass,color=COLORS[policy],marker=marker,s=24)
    ax.set_xticks([0,1],['Midpoint','Relative']);ax.set_xlim(-.35,1.35);ax.set_ylim(0,1.)
    ax.set_ylabel('Mass satisfying report');ax.set_title('Perceived content can agree',fontsize=8,pad=11)
    fig.text(.04,.035,'b–d: development subject 107; one original filter per policy; 512 particles.\n'
        'All 35 complete later reports retained. Common reflection contexts cover all four perceived features.\n'
        'Only reported features enter the minimum. Numeric truth retains both readings. These are diagnostic effects.',
        fontsize=6.6,linespacing=1.5)
    filename=output/'Figure3_s107_joint_reason_profile.png';fig.savefig(filename,dpi=300);plt.close(fig)
    for path,sha in inputs.items():assert digest(Path(path))==sha,path
    qa=dict(status='rendered_pending_visual_review',inputs_sha256=inputs,source_sha256=digest(Path(__file__)),
        all_observed_source_tables_copied_byte_exact=all(digest(source)==digest(output/name) for name,source in tables.items()),
        semantic_reading_relevance_curves_identical=True,scientific_validation=str(validation),
        output_sha256=digest(filename),warnings={'PNG_only':'Repository standing export instruction overrides the static vector/TIFF export requirement',
        'DPI':'300dpi research review figure; final submission format remains unselected',
        'log_axis':'Zero threshold retained in source data and stated in panel c;all positive thresholds plotted'})
    (output/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','validation','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.results,args.validation,args.output)
