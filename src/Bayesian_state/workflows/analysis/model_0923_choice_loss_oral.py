"""Locate choice-loss changes by observed outcome and fixed oral coverage.

Run --scores COMPLETED_CAPACITY_MEMORY_SCORES --oral-audit AUDIT --output NEW.
Retain every grid point and trial. Reuse saved filter probabilities and audited
report identities. Oral strata describe existing behavior-score differences;
they do not fit or select parameters. Coverage is specific to the actual chosen
category. Content roles and possible rule descriptions remain separate questions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_0923_decision_basis_alignment as alignment
from ...evaluation.model_0923_h_validation import nll, jackknife_scores, jackknife_se

COVERAGE = ('supported_for_choice', 'unsupported_for_choice', 'incomplete')


def loss_strata(p: np.ndarray, reference: np.ndarray, correct: np.ndarray,
                coverage: np.ndarray, cut: int) -> list[dict]:
    """Attribute pooled choice loss to fixed subsets with paired filter errors."""
    p, reference = np.asarray(p, float), np.asarray(reference, float)
    correct, coverage = np.asarray(correct), np.asarray(coverage)
    if (p.ndim != 2 or p.shape != reference.shape or len(p) < 2 or not 0 < cut < p.shape[1]
            or correct.shape != (p.shape[1],) or coverage.shape != correct.shape
            or not np.isin(correct, [0,1]).all() or not np.isin(coverage, COVERAGE).all()
            or any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) for x in (p, reference))):
        raise ValueError('Need valid paired filters, binary outcomes and fixed report coverage')
    ordinal = np.arange(p.shape[1]); rows = []
    windows = dict(prefix=ordinal < cut, later=ordinal >= cut, full=np.ones(len(ordinal), bool))
    for phase, window in windows.items():
        for outcome in ('all', 'correct', 'error'):
            for scope in ('all', *COVERAGE):
                mask = window.copy()
                if outcome != 'all': mask &= correct == (outcome == 'correct')
                if scope != 'all': mask &= coverage == scope
                count = int(mask.sum()); total = int(window.sum())
                if count:
                    delta = nll(p[:, mask])-nll(reference[:, mask])
                    error = jackknife_se(jackknife_scores(p[:, mask])-jackknife_scores(reference[:, mask]))
                else:
                    delta = error = np.nan
                rows.append(dict(phase=phase, outcome=outcome, coverage=scope, trials=count, window_trials=total,
                    mean_delta_nll=delta, mean_delta_nll_mcse=error,
                    total_delta_nll=delta*count if count else 0., total_delta_nll_mcse=error*count if count else 0.,
                    window_mean_contribution=delta*count/total if count else 0.,
                    window_mean_contribution_mcse=error*count/total if count else 0.))
    return rows


def run(source: Path, audit: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new choice-loss diagnostic directory')
    read, digest = alignment.read, alignment.digest
    done, protocol, oral_manifest = (read(p) for p in
        (source/'completion.json', source/'protocol.json', audit/'manifest.json'))
    if done['status'] != 'complete' or done['smoke'] or oral_manifest['status'] != 'complete':
        raise ValueError('Require complete scientific scores and oral audit')
    inputs = {str(p.resolve()):digest(p) for p in
              (source/'completion.json', source/'protocol.json', audit/'manifest.json')}
    for folder, hashes in [(source, done['artifacts_sha256']), (audit, oral_manifest['artifacts'])]:
        for name, sha in hashes.items():
            if digest(folder/name) != sha: raise ValueError('Score or oral audit changed')
            inputs[str((folder/name).resolve())] = sha
    for hashes in (protocol['input_sha256'], oral_manifest['input_sha256']):
        for name, sha in hashes.items():
            file = alignment.ROOT/name
            if digest(file) != sha: raise ValueError('Original behavioral or oral input changed')
            inputs[str(file.resolve())] = sha
    score_rows = pd.read_csv(source/'scores.csv'); trial_scores = pd.read_csv(source/'trial_scores.csv')
    anchor = score_rows.loc[score_rows.anchor, 'point'].unique()
    if len(anchor) != 1: raise ValueError('Need one original parameter point')
    oral, cut = alignment.align_identities(pd.read_csv(audit/'trials.csv'), protocol['plans'][0]['job'], done['trials'])
    if cut != done['training_trials']: raise ValueError('Original time boundary changed')
    folder = Path(protocol['plans'][0]['job']['folder']); stimuli, categories = [], []
    for name in ('training.npz', 'sealed_test.npz'):
        with np.load(folder/name) as z:
            stimuli.append(z['stimulus']); categories.append(z['categories'])
    stimulus, category = np.concatenate(stimuli), np.concatenate(categories)
    if stimulus.shape != (len(oral),4) or category.shape != (len(oral),):
        raise ValueError('Prepared stimulus sequence differs from oral identities')
    coverage = np.where(oral.scope.ne('complete'), 'incomplete',
        np.where(oral.structural_support_any_component.gt(0), 'supported_for_choice', 'unsupported_for_choice'))
    banks = {}
    for row in done['records']:
        file = Path(row['target']); record = row['record']
        if digest(file) != record['predictions_sha256']: raise ValueError('Saved filter bank changed')
        with np.load(file) as z:
            np.testing.assert_array_equal(z['seeds'], protocol['seeds'])
            np.testing.assert_array_equal(z['choices'], oral.choice.to_numpy())
            np.testing.assert_array_equal(z['categories'], category)
            banks[row['candidate']['id']] = dict(p=z['observed_probability'], correct=z['choices']==z['categories'])
        inputs[str(file)] = digest(file)
    if len(banks) != done['points'] or set(banks) != set(score_rows.point):
        raise ValueError('Incomplete parameter bank')
    reference = banks[anchor[0]]; summaries, trials = [], []
    for point, bank in sorted(banks.items()):
        np.testing.assert_array_equal(bank['correct'], reference['correct'])
        cell = score_rows[score_rows.point.eq(point)]
        identity = {name: cell.iloc[0][name] for name in ('subject','model','point','anchor','memory','capacity')}
        rows = loss_strata(bank['p'], reference['p'], bank['correct'], coverage, cut)
        for row in rows:
            if row['outcome'] == row['coverage'] == 'all':
                expected = cell[cell.phase.eq(row['phase'])].iloc[0]
                np.testing.assert_allclose([row['mean_delta_nll'],row['mean_delta_nll_mcse']],
                    [expected.delta_choice_nll,expected.delta_choice_nll_mc_se], atol=1e-12, rtol=0)
            summaries.append(dict(identity, **row))
        frame = trial_scores[trial_scores.point.eq(point)].sort_values('trial').reset_index(drop=True).copy()
        np.testing.assert_array_equal(frame.trial, np.arange(1,len(oral)+1))
        np.testing.assert_array_equal(frame.correct, bank['correct'])
        np.testing.assert_allclose(frame.observed_probability, bank['p'].mean(axis=0), atol=1e-12, rtol=0)
        loss = -np.log(np.maximum(bank['p'].mean(axis=0), np.finfo(float).tiny))
        baseline_loss = -np.log(np.maximum(reference['p'].mean(axis=0), np.finfo(float).tiny))
        np.testing.assert_allclose(frame.delta_choice_nll, loss-baseline_loss, atol=1e-12, rtol=0)
        for name in ('iSession','iTrial','choice','text','tokens','query_id','scope',
                     'structural_support_any_component','functional_support_any_component'):
            frame[name] = oral[name].to_numpy()
        for dimension in range(4):
            name = f'feature{dimension+1}'
            frame[name] = stimulus[:,dimension]
            frame[name+'_name'] = oral[name+'_name'].to_numpy()
        frame['category'] = category
        frame['coverage'] = coverage
        trials.append(frame)
    summary, trials = pd.DataFrame(summaries), pd.concat(trials, ignore_index=True)
    queries = trials.groupby(['point','phase','correct','scope','query_id'], dropna=False).agg(
        trials=('trial','size'),total_delta_nll=('delta_choice_nll','sum'),example=('text','first')).reset_index()
    output.mkdir(parents=True)
    for name, frame in [('strata',summary),('trials',trials),('queries',queries)]: frame.to_csv(output/(name+'.csv'),index=False)
    display = summary[summary.coverage.eq('all') & summary.outcome.ne('all')]
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<style>body{max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>参数变化的选择损失与口述覆盖</h1><p>所有差值为该参数点减原参数。'
        '按实际作答正误及固定口述覆盖分层，全部候选与试次保留。覆盖表示当前目录在实际所选类别下'
        '是否至少有一个分量支持完整报告；内容角色、其他类别下的可表达性和被试真正使用的规则仍需分别检查。</p>'
        '<p>分层用于解释已经计算的行为分数，口述未参与参数选择。每个窗口的正确与错误损失贡献之和等于总差值。'
        '模拟标准误逐次删除完整滤波种子，未包括参数与被试不确定性。query汇总为描述性定位。</p>'
        +display.to_html(index=False,float_format=lambda x:f'{x:.6f}')+
        '<p><a href="strata.csv">完整分层</a> · <a href="trials.csv">全部试次</a> · <a href="queries.csv">口述判断组</a></p>')
    for name, sha in inputs.items():
        if digest(Path(name)) != sha: raise ValueError('Diagnostic input changed')
    sources = [Path(__file__), Path(alignment.__file__),
               alignment.ROOT/'src/Bayesian_state/evaluation/model_0923_h_validation.py']
    manifest = dict(status='complete', subject=done['subject'], source=str(source.resolve()), oral_audit=str(audit.resolve()),
        data_use='development diagnosis; all grid points retained; oral does not select parameters',
        input_sha256=inputs, source_sha256={str(p.relative_to(alignment.ROOT)):digest(p) for p in sources},
        artifacts={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--oral-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.scores,args.oral_audit,args.output)
