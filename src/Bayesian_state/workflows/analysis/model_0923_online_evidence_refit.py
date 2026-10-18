"""Refit ONLINE at the retained-precision reference search budgets.

Use --phase search --config YAML --output NEW_DIRECTORY [--smoke], then
--phase evaluate --fit FIT --output NEW_DIRECTORY. Only original training choices
select parameters. The inactive affinity-transfer coordinate is fixed explicitly;
all other supports, seed namespaces and stage limits match the completed RETAIN
reference. POOL and SEARCH receive common projected starts. Oral is excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
RETENTION='src/Bayesian_state/workflows/analysis/model_0923_reentry_precision_refit.py'
ONLINE='src/Bayesian_state/model/modules/hypothesis_transition/online_evidence_belief.py'
TRANSFER='engine.modules.hypo_transitions_mod.kwargs.transfer_scale'
INTERVENTION='online_retention_evidence'
EXTRA={'belief_intervention','fixed_coordinates','retained_reference_fit'}


def helpers():
    spec=importlib.util.spec_from_file_location('online_evidence_refit_retention',ROOT/RETENTION)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def validate_settings(settings: dict, reference: dict) -> None:
    """Close one known inactive channel and preserve the remaining search policy."""
    if (settings.get('belief_intervention')!=INTERVENTION
            or set(settings.get('fixed_coordinates',{}))!={TRANSFER}
            or not settings.get('retained_reference_fit')
            or not 1<=settings.get('parallel_budget',0)<=128):
        raise ValueError('Declare ONLINE, its inactive transfer coordinate and a bounded worker budget')
    changed=EXTRA|{'scope','parallel_budget'}
    if ({k:v for k,v in settings.items() if k not in changed}
            !={k:v for k,v in reference.items() if k not in changed}):
        raise ValueError('Retained reference supports, trial scope, stage limits or seeds changed')
    if TRANSFER in settings['initial_ray_coordinates']:
        raise ValueError('An inactive coordinate cannot define a starting ray')


def fixed_space(space: dict, fixed: dict) -> dict:
    result=deepcopy(space)
    for key,value in fixed.items():
        if key not in result or value not in result[key]:raise ValueError('Fixed coordinate outside original support')
        result[key]=[value]
    return result


def projected_warm_starts(frozen: dict, fixed: dict) -> dict:
    """Project only the inactive coordinate; regenerate candidate IDs downstream."""
    projected=deepcopy(frozen)
    for row in projected['nominees']:
        row['candidate']['hyperparams'].update(fixed)
    return projected


def reference_inputs(retention, recalled, settings: dict) -> dict:
    v=recalled.base.validation();source=ROOT/settings['retained_reference_fit']
    protocol=v.read(source/'protocol.json');frozen=v.read(source/'nominees.json');done=v.read(source/'completion.json')
    validate_settings(settings,protocol['settings'])
    if (done['status']!='complete' or done['smoke'] or frozen['smoke'] or protocol['smoke']
            or frozen['status']!='frozen_before_test' or frozen['settings']!=protocol['settings']
            or done['nomination_sha256']!=v.digest(source/'nominees.json')
            or frozen['source_sha256']!=protocol['source_sha256']):
        raise ValueError('Require a complete frozen retained-precision reference fit')
    clean={k:value for k,value in settings.items() if k not in EXTRA}
    inputs=retention.reference_inputs(recalled,clean)
    for name,sha in protocol['source_sha256'].items():
        if v.digest(ROOT/name)!=sha:raise ValueError('Retained reference implementation changed')
        inputs[str((ROOT/name).resolve())]=sha
    for name in ('protocol.json','nominees.json','completion.json','retention_protocol.json'):
        inputs[str((source/name).resolve())]=v.digest(source/name)
    for arm in settings['arms']:fixed_space(recalled.base.model_space(settings,arm),settings['fixed_coordinates'])
    return inputs


def worker(file: Path) -> None:
    retention=helpers();recalled=retention.helpers();retention.helpers=lambda:recalled
    original_validation=recalled.base.validation
    original_auto=recalled.refit.autonomous
    v=original_validation();job=v.read(file)
    if job.get('belief_intervention')!=INTERVENTION or job.get('online_belief') is not True:
        raise ValueError('Refit worker requires the explicit ONLINE intervention')
    for key,value in job['fixed_coordinates'].items():
        if job['candidate']['hyperparams'][key]!=value:raise ValueError('Inactive fitted coordinate changed')

    def with_online(function,own_job,workflow):
        module=recalled.base.load('src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief',ONLINE)
        cls=sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
        with module.online_evidence_belief(cls,enabled=True) as events:function(own_job,workflow)
        target=Path(own_job['target']);record=v.read(target.with_suffix('.json'))
        count=len(own_job['seeds'])*(1 if own_job['phase']=='autonomous' else own_job['particles'])
        if Counter(e['trial'] for e in events)!={t:count for t in range(1,record['trials'])}:
            raise ValueError('ONLINE refit update timing differs from the validated mechanism')
        v.write(target.with_suffix('.online.json'),dict(enabled=True,belief_intervention=INTERVENTION,
            fixed_coordinates=own_job['fixed_coordinates'],reconstructions=len(events),
            nonentry_reconstructions=sum(e['newcomer_count']==0 for e in events),
            exact_transition_timing=True,evidence_matches_retained_state=True))

    def validation():
        module=original_validation();score=module.score
        module.score=lambda own_job,workflow:with_online(score,own_job,workflow)
        return module

    recalled.base.validation=validation
    recalled.refit.autonomous=lambda own_job,workflow:with_online(original_auto,own_job,workflow)
    retention.worker(file)


def configure(retention, recalled, settings: dict, inputs: dict) -> None:
    recalled.configure()
    old_hashes,old_execute,old_archive=recalled.base.source_hashes,recalled.execute,recalled.base.archive
    old_space,old_starts=recalled.base.model_space,recalled.starting_banks

    def hashes(v):
        result=old_hashes(v)
        result.update({name:v.digest(ROOT/name) for name in (ENTRY,RETENTION,ONLINE,retention.MECHANISM)})
        return result

    def execute(jobs,output,budget):
        tagged=[dict(deepcopy(job),structural_intervention=retention.INTERVENTION,retain_precision=True,
            retention_reference_inputs=inputs,belief_intervention=INTERVENTION,online_belief=True,
            fixed_coordinates=settings['fixed_coordinates']) for job in jobs]
        old_execute(tagged,output,budget)

    def archive(output,config,source_hashes):
        old_archive(output,config,source_hashes)
        recalled.base.validation().write(output/'online_refit_protocol.json',dict(
            belief_intervention=INTERVENTION,structural_intervention=retention.INTERVENTION,
            input_sha256=inputs,fixed_coordinates=settings['fixed_coordinates'],
            parameter_selection='Original training choices only; oral excluded',
            comparison='ONLINE versus completed retained-precision reference; same remaining supports and stage limits',
            initialization='Project the original shared warm starts on the inactive coordinate; deduplicate and complete the declared common bank',
            computation='Actual unique candidate and filter counts are audited separately',
            parameter_interpretation='The transfer coordinate is fixed and carries no individual psychological interpretation'))

    recalled.ENTRY=ENTRY
    recalled.base.source_hashes,recalled.base.execute,recalled.base.archive=hashes,execute,archive
    recalled.base.model_space=lambda own,arm:fixed_space(old_space(own,arm),own['fixed_coordinates'])
    recalled.starting_banks=lambda own,frozen,smoke:old_starts(own,projected_warm_starts(frozen,own['fixed_coordinates']),smoke)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('search','evaluate'))
    for name in ('config','fit','output','worker'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker);return
    retention=helpers();recalled=retention.helpers()
    if args.phase=='search' and args.config and args.output:
        settings=yaml.safe_load(args.config.read_text())
        inputs=reference_inputs(retention,recalled,settings);configure(retention,recalled,settings,inputs)
        recalled.search(args.config,args.output,args.smoke)
    elif args.phase=='evaluate' and args.fit and args.output and not args.smoke:
        settings=recalled.base.validation().read(args.fit/'nominees.json')['settings']
        inputs=reference_inputs(retention,recalled,settings);configure(retention,recalled,settings,inputs)
        recalled.base.evaluate(args.fit,args.output)
    else:parser.error('Use search with --config or evaluate with --fit, and a new --output')


if __name__=='__main__':main()
