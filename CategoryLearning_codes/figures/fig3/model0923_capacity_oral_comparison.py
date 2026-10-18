"""Show choice likelihood, conditioned oral evidence and all autonomous paths.

Run --choice-scores SCORES --oral-comparison PAIRED_ABSOLUTE_COMPARISON
--relative-oral-comparison PAIRED_RELATIVE_COMPARISON --output NEW_DIRECTORY.
The two points are the original and behavior-selected capacity candidates.
All other parameters must agree. Retain the complete six-cell choice table,
every selected-point autonomous path, and all oral windows in source CSVs.
Only PNG is exported, at 183 x 160 mm and 300 dpi for research discussion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model0923_capacity_memory_scores import digest, read, load_tables

CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'


def load_pair(directory: Path) -> tuple[dict, pd.DataFrame, dict]:
    manifest = read(directory/'manifest.json')
    if manifest['status'] != 'complete' or manifest['arm'] != 'SEARCH':
        raise ValueError('Need a completed same-SEARCH parameter comparison')
    inputs = {str((directory/'manifest.json').resolve()):digest(directory/'manifest.json')}
    for name, expected in manifest['input_sha256'].items():
        if digest(Path(name)) != expected: raise ValueError('Comparison input changed')
        inputs[name] = expected
    for name, expected in manifest['artifacts'].items():
        if digest(directory/name) != expected: raise ValueError('Comparison artifact changed')
        inputs[str((directory/name).resolve())] = expected
    before, after = [manifest[key]['candidate']['hyperparams'] for key in ('original_parameters','revised_parameters')]
    if set(before) != set(after) or {name for name in before if before[name] != after[name]} != {CAPACITY}:
        raise ValueError('This figure requires a capacity-only parameter change')
    return manifest, pd.read_csv(directory/'scores.csv'), inputs


def path_curves(probe: Path, candidate: dict, observed: np.ndarray, width: int) -> tuple[pd.DataFrame, dict]:
    blocks, seeds, inputs = [], [], {}
    for file in sorted((probe/'jobs').glob('*.json')):
        job = read(file)
        if job['candidate']['id'] != candidate['id']: continue
        if job['candidate']['hyperparams'] != candidate['hyperparams']:
            raise ValueError('Autonomous and oral parameter coordinates differ')
        target = Path(job['target']).with_suffix('.npz'); record = target.with_suffix('.json')
        if digest(target) != read(record)['predictions_sha256']:
            raise ValueError('Autonomous path bank changed')
        inputs.update({str(f.resolve()):digest(f) for f in (file,record,target)})
        with np.load(target) as z:
            np.testing.assert_array_equal(z['seeds'],job['seeds'])
            correct = z['choices'] == z['categories'][None,:]
            if correct.shape[1] != len(observed): raise ValueError('Autonomous horizon differs')
            values = np.stack([correct[:,start:start+width].mean(axis=1)
                               for start in range(0,len(observed),width)],axis=1)
            blocks.extend(values); seeds.extend(z['seeds'].tolist())
    if not seeds or len(set(seeds)) != len(seeds): raise ValueError('Missing or duplicate autonomous path seeds')
    rows = []
    for seed, values in sorted(zip(seeds,blocks)):
        for block, value in enumerate(values):
            start = block*width; stop = min(start+width,len(observed))
            rows.append(dict(point=candidate['id'],capacity=candidate['hyperparams'][CAPACITY],seed=seed,
                block=block+1,start_trial=start+1,end_trial=stop,trial_center=(start+1+stop)/2,
                accuracy=float(value),observed=float(observed[start:stop].mean())))
    return pd.DataFrame(rows), inputs


def plot(choice: Path, oral: Path, relative: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new capacity/oral figure directory')
    tables, context = load_tables(choice,False)
    original, absolute_scores, inputs = load_pair(oral)
    alternative, relative_scores, extra = load_pair(relative); inputs.update(extra)
    for name in ('subject','arm'):
        if original[name] != alternative[name]: raise ValueError('Reading comparisons differ in identity')
    for name in ('original_parameters','revised_parameters'):
        if original[name]['candidate'] != alternative[name]['candidate']:
            raise ValueError('Reading comparisons use different parameter points')
    candidates = [original[name]['candidate'] for name in ('original_parameters','revised_parameters')]
    if (context['subject'] != 208 or context['choice_trials'] != 576
            or context['prefix_trials'] != 384
            or [r['hyperparams'][CAPACITY] for r in candidates] != [4,2]
            or any(r['hyperparams']['engine.modules.memory_mod.kwargs.gamma'] != 0 for r in candidates)):
        raise ValueError('This focused figure requires the 208 K4/K2, gamma-zero comparison')
    old_id, new_id = [r['id'] for r in candidates]
    if old_id != context['anchor'] or original['subject'] != context['subject']:
        raise ValueError('Original oral parameter differs from the choice-grid anchor')
    full = tables['scores'].query("phase == 'full'").set_index('point')
    if full.loc[new_id,'choice_nll'] != full.choice_nll.min():
        raise ValueError('Revised point must be selected by the full behavior score')
    score = full.loc[new_id]; n = context['choice_trials']
    log_ratio = float(-score.delta_choice_nll*n)
    if abs(log_ratio) > 700: raise ValueError('Likelihood ratio is outside display range')
    ratio = float(np.exp(log_ratio))
    chosen = tables['trials'][tables['trials'].point.eq(new_id)].sort_values('trial')
    observed = tables['trials'][tables['trials'].point.eq(old_id)].sort_values('trial').correct.to_numpy()
    probe = Path(read(choice/'protocol.json')['probe'])
    width = int(read(probe/'protocol.json')['settings']['block_width'])
    paths = []
    for candidate in candidates:
        frame, extra = path_curves(probe,candidate,observed,width); paths.append(frame); inputs.update(extra)
        if frame.seed.nunique() != full.loc[candidate['id'],'paths']:
            raise ValueError('Autonomous path count differs from the scored bank')
    np.testing.assert_array_equal(paths[0].seed.unique(),paths[1].seed.unique())
    path_table = pd.concat(paths,ignore_index=True)
    curves = pd.read_csv(probe/'curves.csv')
    for point, frame in path_table.groupby('point'):
        expected = curves[curves.point.eq(point)].sort_values('trial')
        # Reconstruct the exact realized-choice curve used by the Energy score.
        np.testing.assert_array_equal(frame.groupby('block').end_trial.first(),expected.trial)
        np.testing.assert_allclose(frame.groupby('block').accuracy.mean(),expected['mean'],atol=1e-12,rtol=0)
    oral_table = pd.concat([absolute_scores.assign(reading='Absolute'),relative_scores.assign(reading='Relative')],ignore_index=True)
    keys = ['window','domain','subset']
    # Catalogue support can change with the reading; the complete-report
    # denominator is fixed and is the only subset displayed in panel b.
    pd.testing.assert_frame_equal(absolute_scores[absolute_scores.subset.eq('complete')].set_index(keys)[['reports']],
                                  relative_scores[relative_scores.subset.eq('complete')].set_index(keys)[['reports']])
    inputs.update(context['inputs_sha256'])
    contract = dict(claim='Near-equal sequence likelihood and better conditioned oral alignment can coexist with worse autonomous learning.',
        archetype='quantitative grid',backend='python',width_mm=183,height_mm=160,dpi=300,formats=['PNG'],
        panels=['every-trial choice loss difference','five oral windows and both readings','all original autonomous paths','all revised autonomous paths'],
        parameter_points=[r['id'] for r in candidates],changed_coordinate=CAPACITY,
        data_selection='Original versus full-choice-NLL minimum, fixed before oral comparison; all other grid scores retained in source table.',
        paths_per_point=int(full.loc[old_id,'paths']),path_display='All sampled-choice block curves; bold means; shared observed curve; no filtering by success.',
        uncertainty='Oral differences: 1 paired whole-filter jackknife MCSE; autonomous thin curves are realizations, not confidence intervals.',
        boundaries='Retrospective development; original training boundary shown; oral excluded from fitting; likelihood ratio is not a parameter posterior.',
        source_preflight_exceptions={'EXPORT-VECTOR':'Repository PNG-only rule.','EXPORT-RASTER':'PNG-only rule.','RASTER-DPI':'300 dpi discussion figure.'},
        log_likelihood_ratio=log_ratio,log_likelihood_ratio_mcse=float(score.delta_choice_nll_mc_se*n),relative_sequence_likelihood=ratio)
    output.mkdir(parents=True)
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    tables['scores'].to_csv(output/'choice_grid_source.csv',index=False)
    tables['trials'].to_csv(output/'choice_trial_source.csv',index=False)
    oral_table.to_csv(output/'oral_source.csv',index=False);path_table.to_csv(output/'autonomous_source.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes = plt.subplots(2,2,figsize=(183/25.4,160/25.4))
    fig.subplots_adjust(left=.10,right=.97,bottom=.17,top=.85,wspace=.43,hspace=.52)
    ax = axes[0,0]
    ax.plot(chosen.trial,chosen.cumulative_delta_choice_nll,color='#A57248',lw=1.)
    ax.axhline(0,color='.55',lw=.6);ax.axvline(context['prefix_trials'],color='.6',ls=':',lw=.7)
    ax.set(xlabel='Trial',ylabel='Cumulative NLL difference (K2 − K4)',title='a  Sequence likelihood: a small net gain',xlim=(1,n))
    ax.text(.03,.96,f'Full Δ NLL = {-log_ratio:.3f}\nRelative likelihood = {ratio:.2f}',
        transform=ax.transAxes,ha='left',va='top',fontsize=6)
    ax = axes[0,1]
    windows = ['first64','training','test','last64','whole']
    for reading,color,marker,offset in [('Absolute','#444444','o',-.10),('Relative','#999999','s',.10)]:
        rows = oral_table[(oral_table.reading==reading)&(oral_table.domain=='structural')&(oral_table.subset=='complete')].set_index('window').loc[windows]
        ax.errorbar(rows.support_difference,np.arange(len(windows))+offset,xerr=rows.support_difference_mcse,
            fmt=marker,color=color,ms=3,elinewidth=.8,capsize=1.5,label=reading+' reading')
    ax.axvline(0,color='.6',lw=.6);ax.set_yticks(range(len(windows)),['First 64','First 384','Last 192','Last 64','All 576'])
    ax.set(ylim=(4.6,-.6),xlabel='Δ explicit-content support (K2 − K4)',title='b  Behavior-conditioned oral support')
    ax.legend(loc='lower right',fontsize=5.8)
    for ax,candidate,label,color in zip(axes[1],candidates,['c','d'],['#39758B','#A57248']):
        frame = path_table[path_table.point.eq(candidate['id'])]
        for _,part in frame.groupby('seed'):ax.plot(part.trial_center,part.accuracy,color=color,alpha=.12,lw=.4)
        means = frame.groupby('block').agg(trial=('trial_center','first'),accuracy=('accuracy','mean'),observed=('observed','first'))
        ax.plot(means.trial,means.accuracy,color=color,lw=1.8,label='Autonomous mean')
        ax.plot(means.trial,means.observed,color='#202020',lw=1.2,ls='--',label='Participant')
        k = int(candidate['hyperparams'][CAPACITY]);energy = full.loc[candidate['id'],'energy']
        ax.set(xlabel='Trial',ylabel='Block accuracy',ylim=(0,1.02),xlim=(1,n),
            title=f'{label}  K={k}: all {frame.seed.nunique()} paths; Energy={energy:.3f}')
        ax.axvline(context['prefix_trials'],color='.65',ls=':',lw=.7)
    axes[1,0].legend(loc='lower right')
    fig.suptitle(f'Subject {context["subject"]}: similar likelihood, different autonomous learning',fontsize=9,y=.97)
    fig.text(.5,.925,'Only active-rule capacity changes (K4 → K2); belief retention γ = 0.',ha='center',fontsize=7)
    fig.text(.5,.10,f'{context["filter_seeds"]} filter seeds × {context["particles"]} particles; oral bars: 1 paired MCSE. '
        f'All {len(observed)} trials and all complete reports retained.',ha='center',fontsize=6)
    fig.text(.5,.065,f'Autonomous paths use their own choices and feedback; {width}-trial blocks. Dotted line: original time split.',ha='center',fontsize=6)
    fig.text(.5,.03,'Retrospective development comparison. Oral reports do not select parameters; MCSE excludes parameter uncertainty.',ha='center',fontsize=6)
    fig.savefig(output/'capacity_oral_comparison.png',dpi=300);plt.close(fig)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>'
        '<h1>208：相近序列似然、较高口述支持与较差自主学习同时出现</h1>'
        f'<p>两组只改变容量K。K2相对K4的完整序列对数似然差为{log_ratio:.6f}，相对似然约为{ratio:.3f}。'
        '口述为固定参数后的外部检验；该似然比未加入参数先验。a保留576题累计差值；b分别展示两种固定读法的显式判断支持。'
        'c、d各保留64条完整自主路径，细线是模拟作答的分组正确率，粗线是均值，黑虚线为被试。Energy越低越好。</p>'
        '<p>口述误差棒为八个共同滤波种子的一个配对模拟标准误；全部完整报告使用相同分母。'
        '自主路径具有各自的作答与反馈历史。这些结果分别检验行为序列概率、给定行为后的表征解释和自主学习过程。</p>'
        '<img src="capacity_oral_comparison.png" alt="容量参数在行为、口述与自主学习上的不同结果">'
        '<p>完整六格选择评分、所有逐题选择差值、全部口述窗口和128条自主路径均保存在同目录源表中。</p></html>')
    for name,sha in inputs.items():
        if digest(Path(name)) != sha: raise ValueError('Figure input changed')
    (output/'manifest.json').write_text(json.dumps(dict(status='complete',inputs_sha256=inputs,
        source_sha256={str(Path(__file__).resolve()):digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir()}),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('choice-scores','oral-comparison','relative-oral-comparison','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();plot(args.choice_scores,args.oral_comparison,args.relative_oral_comparison,args.output)
