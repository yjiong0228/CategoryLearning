"""Compare behavior-refitted RETAIN and ONLINE against identical oral reports.

Use --original RETAIN_ALIGNMENT --revised ONLINE_ALIGNMENT --subject ID
--arm POOL_OR_SEARCH --output NEW_DIRECTORY. Completed nominations, actual
ONLINE evaluation sidecars, exact state replays and shared remaining search
supports are required. Scores and feedback strata use the existing joint-mass
calculation; differences include the belief change and re-estimation together.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_0923_oral_parameter_comparison as comparison
from . import model_0923_oral_feedback_strata as strata
from . import model_0923_online_evidence_refit as refit
from . import model_0923_online_evidence_refit_states as states
from ...evaluation.oral.decision_basis_scoring import DOMAINS, LABELS
from ...evaluation.oral.information_controls import jackknife_error


def validate_identities(before: dict, after: dict, *, executed_sources: dict | None = None) -> None:
    """Permit the declared belief change and behavioral re-estimation together."""
    if (before['mechanism'].get('retain_learned_precision') is not True
            or after['mechanism'].get('retain_learned_precision') is not True
            or before['mechanism'].get('online_belief', False) is not False
            or after['mechanism'].get('online_belief') is not True
            or before.get('structural_intervention') != 'retained_rule_precision'
            or after.get('structural_intervention') != 'retained_rule_precision'
            or before.get('parameter_estimation_scope') != 'prefix'
            or after.get('parameter_estimation_scope') != 'prefix'
            or before.get('parameter_candidate_policy') not in
                ('common_search_policy_behavior_refit','equal_budget_behavior_refit')
            or after.get('parameter_candidate_policy') != 'projected_common_search_policy_behavior_refit'):
        raise ValueError('Require RETAIN then ONLINE, both nominated by their behavior-prefix fits')
    matched = deepcopy(after)
    del matched['mechanism']['online_belief']
    # The ONLINE entry replaces one wrapper name in source_sha256. Its worker
    # still binds that imported wrapper through retention_reference_inputs.
    # Accept a missing declaration only with this separately checked hash.
    for name, sha in (executed_sources or {}).items():
        if name in before['model_sources'] and name not in matched['model_sources']:
            matched['model_sources'][name] = sha
    comparison.validate_comparison(before, matched, 'parameters')


def validate_nomination(identity: dict, manifest: dict, frozen: dict,
                        protocol: dict, done: dict, nomination_sha: str,
                        subject: int, arm: str) -> None:
    """Reject state points that were not frozen before later choices were scored."""
    if (frozen.get('status') != 'frozen_before_test' or frozen.get('smoke') is not False
            or protocol.get('smoke') is not False or done.get('smoke') is not False
            or done.get('status') != 'complete' or done.get('test_choices_used') is not False
            or frozen['settings'] != protocol['settings']
            or frozen['source_sha256'] != protocol['source_sha256']
            or done.get('nomination_sha256') != nomination_sha
            or manifest.get('refit_nomination_sha256') != nomination_sha):
        raise ValueError('Require completed behavior-prefix nominations bound to the oral replay')
    selected = [row for row in frozen['nominees'] if (row['subject'],row['model']) == (subject,arm)]
    if (len(selected) != 1 or any(identity['candidate'][key] != selected[0]['candidate'][key]
                                  for key in ('id','hyperparams'))):
        raise ValueError('Oral state point differs from the behavior-selected nominee')
    if any(identity['model_sources'].get(name) != sha for name,sha in frozen['source_sha256'].items()):
        raise ValueError('State replay changed its nominated fitting implementation')


def bind_refits(before: dict, after: dict, old_manifest: dict, new_manifest: dict,
                subject: int, arm: str) -> dict:
    read, digest = comparison.alignment.read, comparison.alignment.digest
    old_fit, new_fit = Path(old_manifest['retained_fit']), Path(new_manifest['online_fit'])
    inputs = {}
    def bind(path):
        path = Path(path).resolve();inputs[str(path)] = digest(path);return path
    banks = []
    for fit, identity, manifest, sidecar in [(old_fit,before,old_manifest,'retention_protocol.json'),
                                            (new_fit,after,new_manifest,'online_refit_protocol.json')]:
        frozen, protocol, done, intervention = [read(bind(fit/name)) for name in
            ('nominees.json','protocol.json','completion.json',sidecar)]
        validate_nomination(identity,manifest,frozen,protocol,done,digest(fit/'nominees.json'),subject,arm)
        for name, sha in frozen['source_sha256'].items():
            if digest(bind(refit.ROOT/name)) != sha:
                raise ValueError('Fitted source changed before the comparison')
        banks.append((frozen,intervention))
    old, _ = banks[0]; new, intervention = banks[1]
    settings = new['settings']
    if (refit.ROOT/settings['retained_reference_fit']).resolve() != old_fit.resolve():
        raise ValueError('ONLINE fit refers to a different retained-precision fit')
    retention = refit.helpers()
    references = refit.reference_inputs(retention,retention.helpers(),settings)
    if (intervention['input_sha256'] != references
            or intervention.get('belief_intervention') != refit.INTERVENTION
            or intervention.get('fixed_coordinates') != settings['fixed_coordinates']
            or new_manifest.get('fixed_coordinates') != settings['fixed_coordinates']
            or new_manifest.get('belief_intervention') != refit.INTERVENTION):
        raise ValueError('Changed projected search policy or ONLINE intervention provenance')
    for key,value in settings['fixed_coordinates'].items():
        if after['candidate']['hyperparams'].get(key) != value:
            raise ValueError('Frozen inactive coordinate differs from the fitted point')
    inputs.update(references)
    # Reuse the executed-evaluation validator, including ONLINE and precision
    # sidecars. A plan naming the intervention alone does not establish execution.
    source = Path(new_manifest['source'])
    state_protocol = read(bind(source/'protocol.json'))
    original_evaluation = refit.ROOT/state_protocol['settings']['source']
    _, metadata = states.source_jobs(state_protocol['settings'],original_evaluation,False)
    if (metadata['refit_nomination_sha256'] != new_manifest['refit_nomination_sha256']
            or Path(metadata['online_fit']).resolve() != new_fit.resolve()):
        raise ValueError('Executed evaluation and oral states refer to different ONLINE fits')
    inputs.update(metadata['refit_replay_inputs'])
    return inputs


def feedback_comparison(old: dict, new: dict, subject: int, arm: str) -> tuple[pd.DataFrame, dict]:
    """Keep each feedback level and preserve common complete-report masks."""
    np.testing.assert_array_equal(old['seeds'],new['seeds'])
    for field in ('feedback','categories'):
        np.testing.assert_array_equal(old[field],new[field])
    pd.testing.assert_frame_equal(old['trials'][comparison.IDENTITY],new['trials'][comparison.IDENTITY])
    rows, deleted = [], {}
    for key,a in old['estimates'].items():
        b = new['estimates'][key]
        np.testing.assert_array_equal(a['mask'],b['mask'])
        error = jackknife_error(b['deleted']-a['deleted'])
        for label,bank in [('retained',old),('online',new)]:
            deleted[label+'_'+key[0]+'_'+key[1]] = bank['estimates'][key]['deleted']
        for m,metric in enumerate(strata.METRICS):
            for d,domain in enumerate(DOMAINS):
                row = dict(subject=subject,arm=arm,window=key[0],feedback_group=key[1],metric=metric,
                    domain=domain,complete_reports=a['reports'],trials=a['trials'],filter_seeds=len(old['seeds']))
                for k,label in enumerate(LABELS):
                    row.update({f'original_{label}':a['point'][m,d,k],f'revised_{label}':b['point'][m,d,k],
                        f'{label}_difference':b['point'][m,d,k]-a['point'][m,d,k],f'{label}_difference_mcse':error[m,d,k]})
                rows.append(row)
    return pd.DataFrame(rows), deleted


def run(original: Path, revised: Path, output: Path, subject: int, arm: str) -> None:
    if output.exists():
        raise FileExistsError('Use a new ONLINE refit oral-comparison directory')
    old, before, inputs = comparison.load_group(original,subject,arm)
    new, after, more = comparison.load_group(revised,subject,arm);inputs.update(more)
    old_manifest = comparison.alignment.read(original/'manifest.json')
    new_manifest = comparison.alignment.read(revised/'manifest.json')
    refit_inputs = bind_refits(before,after,old_manifest,new_manifest,subject,arm)
    inputs.update(refit_inputs)
    executed_sources = {str(Path(path).relative_to(refit.ROOT)):sha for path,sha in refit_inputs.items()
                        if Path(path).is_relative_to(refit.ROOT) and Path(path).suffix == '.py'}
    validate_identities(before,after,executed_sources=executed_sources)
    scores, trials = comparison.paired_scores(old,new)
    old_bank, more = strata.load_bank(original,subject,arm,old);inputs.update(more)
    new_bank, more = strata.load_bank(revised,subject,arm,new);inputs.update(more)
    feedback, deleted = feedback_comparison(old_bank,new_bank,subject,arm)
    trials['feedback'] = old_bank['feedback'];trials['category'] = old_bank['categories']
    for path, sha in inputs.items():
        if comparison.alignment.digest(Path(path)) != sha:
            raise ValueError('Input changed during the ONLINE refit comparison')
    output.mkdir(parents=True)
    for name, table in [('scores',scores),('trials',trials),('feedback_scores',feedback)]:
        table.to_csv(output/(name+'.csv'),index=False)
    with (output/'feedback_jackknife.npz').open('xb') as stream:
        np.savez_compressed(stream,**deleted,seeds=old_bank['seeds'])
    note = ('两方案均保留规则已学精度，差异包括H在线证据形成信念及行为参数重新估计。'
        '参数仅用原前段选择确定，口述仅用于冻结状态后的表征评价。ONLINE的亲和度转移坐标固定，'
        '其余支持范围和阶段预算上限沿用RETAIN；共同起点在固定坐标上投影，实际计算量单独核对。'
        '两方案使用相同报告、固定读法和完整滤波种子。误差为整次滤波配对删除所得的模拟标准误。'
        'training与test表示开发被试的原时间分段，后段继续作为时间外推诊断。')
    display = scores[scores.domain.eq('structural') & scores.subset.eq('complete')]
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>ONLINE重估后的口述比较</title><style>body{max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>ONLINE重估参数后与RETAIN的口述比较</h1><p>'+note+'</p>'+display.to_html(index=False,float_format=lambda x:f'{x:.6f}')
        +'<p><a href="scores.csv">全部整句评分</a> · <a href="trials.csv">完整试次</a> · '
        '<a href="feedback_scores.csv">反馈分层与去掉本题选择加权的对照</a></p></html>')
    paths = [Path(__file__),Path(comparison.__file__),Path(strata.__file__),Path(refit.__file__),Path(states.__file__),
             Path(comparison.alignment.__file__),refit.ROOT/'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py']
    (output/'source').mkdir()
    for path in paths:
        shutil.copy2(path,output/'source'/path.name)
    result = dict(status='complete',subject=subject,arm=arm,original=str(original.resolve()),revised=str(revised.resolve()),
        contrast_kind='online_evidence_refit',seed_relation='paired',original_parameters=before,revised_parameters=after,
        bound_reference_sources=executed_sources,
        original_scores_reproduced=True,revised_scores_reproduced=True,input_sha256=inputs,
        source_sha256={str(p.relative_to(refit.ROOT)):comparison.alignment.digest(p) for p in paths},
        artifacts={p.name:comparison.alignment.digest(p) for p in output.iterdir() if p.is_file()},
        data_use='Behavior-prefix estimation; combined ONLINE belief and re-estimation effect; oral evaluation only; development participants.',
        uncertainty='paired_delete_one_complete_filter_seed_mcse')
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('original','revised','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--subject',type=int,required=True)
    parser.add_argument('--arm',choices=('POOL','SEARCH'),required=True)
    args = parser.parse_args();run(args.original,args.revised,args.output,args.subject,args.arm)
