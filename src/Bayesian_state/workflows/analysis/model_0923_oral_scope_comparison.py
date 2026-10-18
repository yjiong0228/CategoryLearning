"""Separate added report coverage from changes on originally complete reports.

Use --original ALIGNMENT --revised ALIGNMENT --output NEW_DIRECTORY. Both inputs
must use the same behavioral states. The report preserves every trial identity
and checks unchanged scores on the original complete-report subset before
summarizing original, newly complete and revised complete reports separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_decision_basis_alignment import digest, read
from .model_0923_association_diagnostic import write


def compare_frames(original: pd.DataFrame, revised: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject changed identities, removed complete reports or altered old scores."""
    original = original.rename(columns={'arm_name':'arm'})
    revised = revised.rename(columns={'arm_name':'arm'})
    keys = ['iSub','arm','ordinal_trial']
    old, new = (f.sort_values(keys).reset_index(drop=True) for f in (original,revised))
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError('Duplicate trial identity.')
    identity = keys+['iSession','iTrial','choice','text','training_trials']
    pd.testing.assert_frame_equal(old[identity],new[identity])
    complete = old.scope.eq('complete')
    if not new.loc[complete,'scope'].eq('complete').all():
        raise ValueError('Originally complete reports must remain complete.')
    pd.testing.assert_frame_equal(old.loc[complete,['tokens','scope']],new.loc[complete,['tokens','scope']])
    scores = [p+d+'_'+v for p in ('','prior_') for d in ('structural','functional')
              for v in ('support','conflict','undetermined')]
    np.testing.assert_allclose(old.loc[complete,scores],new.loc[complete,scores],rtol=0,atol=1e-12)
    probability = 'choice_probability' if 'choice_probability' in old else 'mean_choice_probability'
    np.testing.assert_array_equal(old[probability],new[probability])
    new = new.copy(); new['original_scope'] = old.scope
    new['newly_complete'] = new.scope.eq('complete')&~complete
    changed = new.loc[new.newly_complete].copy()
    rows = []
    for (subject,arm), group in new.groupby(['iSub','arm'],sort=True):
        ordinal = group.ordinal_trial; cut = group.training_trials.unique()
        if len(cut)!=1: raise ValueError('Training boundary differs within participant.')
        windows = dict(whole=ordinal>0,training=ordinal<=cut[0],test=ordinal>cut[0],
                       first64=ordinal<=64,last64=ordinal>ordinal.max()-64)
        scopes = dict(original_complete=group.original_scope.eq('complete'),
                      newly_complete=group.newly_complete,revised_complete=group.scope.eq('complete'))
        for window, mask in windows.items():
            for scope, included in scopes.items():
                part = group.loc[mask&included]
                for domain in ('structural','functional'):
                    rows.append(dict(subject=int(subject),arm=arm,window=window,scope=scope,domain=domain,
                        reports=len(part),**{v:float(part[domain+'_'+v].mean())
                                           for v in ('support','conflict','undetermined')}))
    return pd.DataFrame(rows), changed


def load(directory: Path) -> tuple[pd.DataFrame, dict, dict]:
    association = (directory/'completion.json').exists()
    file = directory/('completion.json' if association else 'manifest.json')
    manifest = read(file)
    if manifest['status']!='complete' or manifest['smoke']:
        raise ValueError('Need a complete scientific alignment.')
    inputs = {str(file.resolve()):digest(file)}
    for name, sha in manifest['artifacts'].items():
        if digest(directory/name)!=sha: raise ValueError('Alignment artifact changed.')
        inputs[str((directory/name).resolve())] = sha
    if association:
        protocol = read(directory/'protocol.json')
        if digest(directory/'protocol.json')!=manifest['protocol_sha256']:
            raise ValueError('Association protocol changed.')
        inputs[str((directory/'protocol.json').resolve())] = digest(directory/'protocol.json')
        state = dict(kind='association',nomination=protocol['behavior_nomination_sha256'],
                     arrays={f.name:str(f.resolve()) for f in (directory/'states').glob('*.npz')})
        for name in state['arrays'].values():
            if digest(Path(name))!=manifest['input_sha256'][name]: raise ValueError('Association state changed.')
            inputs[name] = digest(Path(name))
        table = 'trial_scores.csv'
    else:
        state = dict(kind='particle',arrays=manifest['state_arrays'],protocol=manifest['source_protocol_sha256'])
        for name,sha in state['arrays'].items():
            if digest(Path(name))!=sha: raise ValueError('Particle state changed.')
            inputs[name] = sha
        table = 'trials.csv'
    return pd.read_csv(directory/table), state, inputs


def run(original: Path, revised: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new scope-comparison output directory.')
    old, left, inputs = load(original); new, right, other = load(revised); inputs.update(other)
    if left['kind']!=right['kind']: raise ValueError('Behavioral model families differ.')
    if left['kind']=='particle':
        if left!=right: raise ValueError('Particle state inputs differ.')
    else:
        if left['nomination']!=right['nomination'] or set(left['arrays'])!=set(right['arrays']):
            raise ValueError('Behavioral nomination groups differ.')
        for name in left['arrays']:
            with np.load(left['arrays'][name]) as a,np.load(right['arrays'][name]) as b:
                if set(a.files)!=set(b.files): raise ValueError('Behavioral state fields differ.')
                for key in a.files: np.testing.assert_array_equal(a[key],b[key])
    summary, changed = compare_frames(old,new)
    output.mkdir(parents=True)
    summary.to_csv(output/'scope_scores.csv',index=False)
    changed.to_csv(output/'newly_complete_reports.csv',index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<h1>口述编码补充后的共同报告与新增报告</h1><p>行为状态完全相同。原完整报告的编码和得分保持一致，'
        '新增完整报告单列后，再汇总新完整报告范围。缺失均值表示该范围没有报告。此表为描述性评分，未估计额外抽样误差。</p>'
        +summary.loc[summary.window.eq('test')&summary.domain.eq('structural')].to_html(index=False))
    write(output/'manifest.json',dict(status='complete',original=str(original.resolve()),revised=str(revised.resolve()),
        state_arrays_identical=True,original_complete_scores_unchanged=True,
        unique_added_reports=len(changed[['iSub','iSession','iTrial']].drop_duplicates()),
        added_by_subject=changed[['iSub','iSession','iTrial']].drop_duplicates().groupby('iSub').size().to_dict(),
        inputs_sha256=inputs,source_sha256={str(Path(__file__).resolve()):digest(Path(__file__))},
        artifacts={f.name:digest(f) for f in output.iterdir()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','revised','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.original,args.revised,args.output)
