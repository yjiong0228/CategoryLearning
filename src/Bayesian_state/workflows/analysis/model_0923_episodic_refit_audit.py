"""Independent artifact checks for completed admission refits and figures.

Check every recorded job, probability and rule-process array, task feedback,
parameter point, source hash and numeric-thread count. Add a new QA file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from .model_0923_h_validation import read, write, digest, pooled_nll, REPO
from ...metrics.task import category_learning_feedback


def audit(fit: Path, evaluation: Path, summary: Path, figures: Path) -> dict:
    frozen = read(fit/'nominees.json'); protocol = read(fit/'protocol.json'); settings = frozen['settings']
    spaces = read(fit/'parameter_spaces.json')
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need scientific frozen parameters')
    if read(fit/'completion.json')['nomination_sha256'] != digest(fit/'nominees.json'): raise ValueError('Nominees changed')
    if read(evaluation/'protocol.json')['nomination_sha256'] != digest(fit/'nominees.json'): raise ValueError('Evaluation parameters changed')
    if read(evaluation/'completion.json')['status'] != 'complete': raise ValueError('Evaluation incomplete')
    if protocol['data_manifest_sha256'] != digest(Path(frozen['data'])/'preparation_manifest.json'): raise ValueError('Prepared data manifest changed')
    for name,sha in frozen['source_sha256'].items():
        if digest(REPO/name) != sha or digest(fit/'source'/name) != sha: raise ValueError('Study source changed')
    counts, cpu, corrections = Counter(), Counter(), Counter(); maximum = 1.
    nominees = {(r['subject'],r['model']):r for r in frozen['nominees']}
    for root in (fit, evaluation):
        for file in sorted((root/'jobs').glob('*.json')):
            job = read(file); target = Path(job['target']); row = read(target.with_suffix('.json'))
            meta = read(Path(job['folder'])/'prepared.json')
            if row['candidate'] != job['candidate']: raise ValueError('Job parameter metadata changed')
            point = job['candidate']['hyperparams']; space = spaces[job['model']]
            if set(point) != set(space) or any(point[k] not in values for k,values in space.items()):
                raise ValueError('Parameter point lies outside the registered search space')
            if job['source_sha256'] != frozen['source_sha256']: raise ValueError('Job used different implementation')
            if 'Traceback' in file.with_suffix('.log').read_text(): raise ValueError('Failed worker in completed experiment')
            if any(pool['num_threads'] != 1 for pool in row['numeric_pools']): raise ValueError('Nested numeric threads')
            if digest(target.with_suffix('.npz')) != row['predictions_sha256']: raise ValueError('Predictions changed')
            for name,sha in meta['frozen_cognitive_source_sha256'].items():
                if digest(Path(job['runtime'])/name) != sha: raise ValueError('Archived cognitive core changed')
            phase = job['phase']; counts['jobs'] += 1; cpu[root.name] += row['cpu_seconds']
            with np.load(target.with_suffix('.npz')) as arrays:
                np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
                n = meta['training_trials'] if phase == 'training' else meta['trials']
                shape = len(job['seeds']), n
                for field in ('correct_probability','observed_probability','probabilities'):
                    if field not in arrays: continue
                    p = arrays[field]
                    if p.shape[:2] != shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1+1e-12):
                        raise ValueError('Invalid probability shape or range')
                if phase == 'training':
                    counts['training_filters'] += len(job['seeds'])
                    np.testing.assert_allclose(pooled_nll(arrays['observed_probability']),row['mean_nll'],atol=1e-14,rtol=0)
                else:
                    point = nominees[meta['subject'],job['model']]['candidate']
                    if point['hyperparams'] != job['candidate']['hyperparams']: raise ValueError('Frozen parameters changed')
                    p = arrays['probabilities']; np.testing.assert_allclose(p.sum(axis=-1),1.,atol=1e-12,rtol=0)
                    categories = arrays['categories']
                    indices = np.broadcast_to(categories[None,:,None]-1,(*shape,1))
                    np.testing.assert_array_equal(np.take_along_axis(p,indices,axis=-1)[...,0],arrays['correct_probability'])
                    if phase == 'evaluate':
                        counts['evaluation_filters'] += len(job['seeds'])
                        indices = np.broadcast_to(arrays['choices'][None,:,None]-1,(*shape,1))
                        np.testing.assert_array_equal(np.take_along_axis(p,indices,axis=-1)[...,0],arrays['observed_probability'])
                        np.testing.assert_allclose(pooled_nll(arrays['observed_probability']),row['mean_nll'],atol=1e-14,rtol=0)
                    elif phase == 'autonomous':
                        counts['paths'] += len(job['seeds'])
                        for field in ('choices','feedback','present','contribution','beta','proposed',
                                      'search','rejected','target_exited','target_capacity_dropped','active_count'):
                            if arrays[field].shape != shape or not np.isfinite(arrays[field]).all():
                                raise ValueError(f'Invalid autonomous diagnostic array: {field}')
                        expected_feedback = np.array([[category_learning_feedback(int(choice),int(category),condition=meta['condition'])
                            for choice,category in zip(path,categories)] for path in arrays['choices']])
                        np.testing.assert_array_equal(expected_feedback,arrays['feedback'])
                        if not np.isin(arrays['present'],[0,1]).all(): raise ValueError('Invalid target presence')
                        beta = arrays['beta'][arrays['present'].astype(bool)]
                        if not np.isfinite(beta).all() or np.any((beta <= 0)|(beta > job['arm_settings']['upper'])):
                            raise ValueError('Invalid active-rule precision')
                        if np.any(arrays['contribution'][arrays['present']==0] != 0): raise ValueError('Absent rule contributes to choice')
                        if np.any((arrays['contribution'] < 0)|(arrays['contribution'] > 1+1e-12)): raise ValueError('Invalid choice contribution')
                        capacity = point['hyperparams']['engine.modules.hypo_transitions_mod.kwargs.max_capacity']
                        if np.any((arrays['active_count'] < 1)|(arrays['active_count'] > capacity)): raise ValueError('Capacity violated')
                        if digest(Path(job['audit'])) != job['audit_sha256']: raise ValueError('Geometry audit changed')
                    else: raise ValueError('Unexpected experiment phase')
            fixes = read(target.with_suffix('.experiment.json'))
            for field in ('feedback_corrections','snapshot_corrections','recall_corrections'):
                for fix in fixes[field]:
                    if not 1 < fix['maximum'] <= 1+8*np.finfo(float).eps: raise ValueError('Correction exceeds roundoff guard')
                    maximum = max(maximum,fix['maximum']); corrections[fix['phase']] += 1
    expected_cells = len(settings['subjects'])*len(settings['arms'])
    if counts['paths'] != expected_cells*settings['autonomous_paths']: raise ValueError('Missing paths')
    if counts['evaluation_filters'] != expected_cells*settings['budgets']['evaluate']['seeds']: raise ValueError('Missing evaluation filters')
    manifest = read(summary/'manifest.json')
    for name,sha in manifest['inputs_sha256'].items():
        if digest(Path(name)) != sha: raise ValueError('Summary source changed')
    for name,sha in manifest['metric_sources_sha256'].items():
        if digest(summary/'metric_sources'/name) != sha: raise ValueError('Archived score implementation changed')
    for name,sha in manifest['summary_sha256'].items():
        if digest(summary/name) != sha or digest(figures/name) != sha: raise ValueError('Summary table changed')
    scores = pd.read_csv(summary/'scores.csv'); paths = pd.read_csv(summary/'paths.csv')
    np.testing.assert_allclose(scores.choice_nll, scores.correct_loss+scores.error_loss, atol=1e-12, rtol=0)
    pairwise = pd.read_csv(summary/'cohort_pairwise.csv')
    expected_pairs = 3*len(settings['arms'])*(len(settings['arms'])-1)//2
    if len(pairwise) != expected_pairs or pairwise.duplicated(['phase','reference','model']).any():
        raise ValueError('Incomplete pairwise cohort comparisons')
    for row in pairwise.itertuples():
        phase = scores[scores.phase==row.phase]
        for metric in ('choice_nll','energy','crps'):
            expected_difference = phase.loc[phase.model==row.model,metric].mean()-phase.loc[phase.model==row.reference,metric].mean()
            np.testing.assert_allclose(getattr(row,'delta_'+metric),expected_difference,atol=1e-12,rtol=0)
    for row in scores[scores.phase=='whole'].itertuples():
        cell = paths[(paths.subject==row.subject)&(paths.model==row.model)]
        if len(cell) != row.paths: raise ValueError('Summary path count differs from saved path table')
        np.testing.assert_allclose(
            [row.late_accuracy,row.late_reaches_observed,row.target_ever,row.target_proposed,row.max_active_count],
            [cell.late_accuracy.mean(),(cell.late_accuracy>=row.observed_late).mean(),cell.target_ever.mean(),
             (cell.target_proposals>0).mean(),cell.max_active_count.max()],
            atol=1e-12, rtol=0)
    if digest(figures/'source_manifest.json') != digest(summary/'manifest.json'): raise ValueError('Figure provenance changed')
    if digest(figures/'figure_source.py') != digest(REPO/'CategoryLearning_codes/figures/fig3/model0923_episodic_refit.py'):
        raise ValueError('Plot source changed')
    qa = read(figures/'qa_render.json'); sizes = {}
    if len(qa['images']) != len(settings['subjects'])+4 or qa['paths'] != counts['paths']: raise ValueError('Incomplete figures')
    for name,sha in qa['images'].items():
        if digest(figures/name) != sha: raise ValueError('Figure changed')
        with Image.open(figures/name) as im:
            if im.width != 2161 or abs(im.info['dpi'][0]-300) > .1: raise ValueError('Unexpected raster export')
            sizes[name] = dict(width=im.width,height=im.height,dpi=im.info['dpi'])
    result = dict(status='pass',counts=dict(counts),cpu_seconds=dict(cpu),corrections=dict(corrections),
        maximum_corrected_probability=maximum,images=sizes,
        limitations='Bounded search and development participants; Monte Carlo uncertainty at fixed parameters.')
    write(figures/'qa_complete.json',result)
    shutil.copy2(__file__,figures/'audit_source.py')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit','evaluation','summary','figures'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.fit,args.evaluation,args.summary,args.figures)
    print(dict(status=result['status'],counts=result['counts'],cpu_hours=sum(result['cpu_seconds'].values())/3600))
