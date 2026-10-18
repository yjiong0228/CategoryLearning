"""Read frozen ONLINE refit states without changing the fitted mechanism.

Use --phase replay --config YAML --output NEW [--smoke], then --phase align
--source REPLAY --oral-audit FIXED_CODING --output NEW. Replays bind completed
training-only nominees to the actual evaluated jobs, their ONLINE sidecars,
fixed inactive coordinate and original seeds. The existing state observer and
ONLINE context perform the computation. Smoke preserves the original particle
count and the first seed of the first declared arm, for at most sixteen trials.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
STATES = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_states.py'
REFIT = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_refit.py'
USE = 'development_online_evidence_refit'


def helpers():
    spec = importlib.util.spec_from_file_location('online_refit_state_base', ROOT / STATES)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked_execution(planned: dict, actual: dict, references: dict, fixed: dict,
                      precision: dict, online: dict) -> None:
    """A saved evaluation plan is insufficient without the executed interventions."""
    expected = dict(planned, structural_intervention='retained_rule_precision', retain_precision=True,
        retention_reference_inputs=references, belief_intervention='online_retention_evidence',
        online_belief=True, fixed_coordinates=fixed)
    if (actual != expected or actual['phase'] != 'evaluate' or actual['model'] not in ('POOL','SEARCH')
            or any(actual['candidate']['hyperparams'].get(k) != v for k,v in fixed.items())
            or precision.get('retained_precision') is not True
            or precision.get('structural_intervention') != 'retained_rule_precision'
            or online.get('enabled') is not True
            or online.get('belief_intervention') != 'online_retention_evidence'
            or online.get('fixed_coordinates') != fixed
            or online.get('exact_transition_timing') is not True
            or online.get('evidence_matches_retained_state') is not True):
        raise ValueError('ONLINE refit evaluation changed its actual intervention or inactive coordinate')


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    from src.Bayesian_state.utils.seeding import stable_seed
    from . import model_0923_online_evidence_refit as refit
    replay = helpers().helpers()
    read, digest = replay.read, replay.digest
    completed, evaluation = read(source/'completion.json'), read(source/'protocol.json')
    fit = Path(evaluation['fit'])
    required = ('protocol.json','nominees.json','completion.json','online_refit_protocol.json')
    protocol, frozen, done, intervention = (read(fit/name) for name in required)
    nomination = digest(fit/'nominees.json')
    if (completed['status'] != 'complete' or done['status'] != 'complete'
            or done['test_choices_used'] is not False or done['smoke'] or protocol['smoke'] or frozen['smoke']
            or frozen['status'] != 'frozen_before_test'
            or any(r['nomination_sha256'] != nomination for r in (done,evaluation,completed))
            or frozen['settings'] != protocol['settings']
            or frozen['source_sha256'] != protocol['source_sha256']
            or evaluation['source_sha256'] != frozen['source_sha256']):
        raise ValueError('Require a completed training-only ONLINE fit and its frozen evaluation')
    original = frozen['settings']
    if (original.get('belief_intervention') != refit.INTERVENTION
            or intervention.get('belief_intervention') != refit.INTERVENTION
            or not settings['subjects'] or len(set(settings['subjects'])) != len(settings['subjects'])
            or not set(settings['subjects']) <= set(original['subjects'])
            or not settings['arms'] or len(set(settings['arms'])) != len(settings['arms'])
            or not set(settings['arms']) <= {'POOL','SEARCH'}
            or not set(settings['arms']) <= set(original['arms'])):
        raise ValueError('Require unique declared ONLINE development subjects and proposal arms')
    retention = refit.helpers()
    for name in (REFIT,refit.ONLINE,refit.RETENTION,retention.MECHANISM):
        if frozen['source_sha256'].get(name) != digest(ROOT/name):
            raise ValueError('Missing executed ONLINE or precision-retention source')
    references = refit.reference_inputs(retention, retention.helpers(), original)
    if (intervention['input_sha256'] != references
            or intervention['fixed_coordinates'] != original['fixed_coordinates']):
        raise ValueError('ONLINE refit reference or fixed coordinate changed')
    bindings = dict(references)
    for name, sha in frozen['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Fitted implementation changed')
        bindings[str((ROOT/name).resolve())] = sha
    for file in [*(fit/name for name in required),source/'protocol.json',source/'completion.json',ROOT/STATES,ROOT/ENTRY]:
        bindings[str(file.resolve())] = digest(file)
    budget = original['budgets']['evaluate']
    if settings['seeds_per_arm'] != budget['seeds']:
        raise ValueError('Preserve every original evaluation filter seed')
    nominees = {(r['subject'],r['model']):r['candidate'] for r in frozen['nominees']}
    plans = {job['name']:job for job in evaluation['jobs']}
    if len(nominees) != len(frozen['nominees']) or len(plans) != len(evaluation['jobs']) or completed['jobs'] != len(plans):
        raise ValueError('Duplicate or incomplete ONLINE nomination or evaluation')
    files, selected = [], []
    for subject in settings['subjects'][:1] if smoke else settings['subjects']:
        for arm in settings['arms'][:1] if smoke else settings['arms']:
            point = nominees[subject,arm]
            folder = (Path(frozen['data'])/f'subject_{subject}').resolve()
            if read(folder/'prepared.json')['subject'] != subject:
                raise ValueError('Wrong prepared participant')
            seeds = [stable_seed(dict(protocol=original['protocol'],base=original['seed'],
                stage='evaluate',subject=subject,repeat=i)) for i in range(budget['seeds'])]
            particles = {int(k):v for k,v in original['particle_overrides'].items()}.get(subject,budget['particles'])
            group = sorted((source/'jobs').glob(f'evaluate_S{subject}_{arm}_*_C*.json'))
            if len(group) != len(seeds):
                raise ValueError('Incomplete original evaluation bank')
            for i,(file,seed) in enumerate(zip(group,seeds)):
                name = f'evaluate_S{subject}_{arm}_{point["id"]}_C{i:03d}'
                planned = plans.get(name)
                if (planned is None or file.stem != name or planned['phase'] != 'evaluate' or planned['smoke']
                        or planned['model'] != arm or planned['candidate'] != point or planned['seeds'] != [seed]
                        or planned['particles'] != particles or Path(planned['folder']).resolve() != folder
                        or planned['arm_settings'] != original['arms'][arm]
                        or planned['mechanism'] != original['mechanism']
                        or planned['source_sha256'] != frozen['source_sha256']):
                    raise ValueError('Saved evaluation differs from the behavior-selected nominee')
                actual = read(file)
                target = Path(planned['target'])
                precision_file, online_file = target.with_suffix('.precision.json'),target.with_suffix('.online.json')
                precision, online = read(precision_file), read(online_file)
                checked_execution(planned,actual,references,original['fixed_coordinates'],precision,online)
                record = read(target.with_suffix('.json'))
                if online['reconstructions'] != particles*(record['trials']-1):
                    raise ValueError('ONLINE transition count differs from the completed filter')
                for path in (file,precision_file,online_file):
                    bindings[str(path.resolve())] = digest(path)
            files.extend(group[:1] if smoke else group)
            selected.append(dict(subject=subject,arm=arm,parameter_id=point['id']))
    metadata = dict(parameter_estimation_scope='prefix',evaluation_use=USE,
        parameter_candidate_policy='projected_common_search_policy_behavior_refit',
        structural_intervention='retained_rule_precision',belief_intervention=refit.INTERVENTION,
        selected_parameters=selected,online_fit=str(fit.resolve()),refit_nomination_sha256=nomination,
        fixed_coordinates=original['fixed_coordinates'],refit_replay_inputs=bindings,
        report_arm_policy='Each recorded arm names its original POOL or SEARCH proposal; both use ONLINE')
    return files, metadata


def worker(file: Path) -> None:
    online = helpers()
    replay = online.helpers()
    job = replay.read(file)
    if (job.get('belief_intervention') != 'online_retention_evidence' or job.get('online_belief') is not True
            or job['model'] not in ('POOL','SEARCH')
            or job.get('reported_arm') != job['model'] or job.get('parameter_estimation_scope') != 'prefix'
            or job.get('evaluation_use') != USE
            or any(job['candidate']['hyperparams'].get(k) != v for k,v in job['fixed_coordinates'].items())):
        raise ValueError('Replay must preserve the frozen ONLINE mechanism, proposal arm and selection scope')
    online.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    replay = helpers().helpers()
    original_prepare = replay.prepare
    replay.source_jobs, replay.ENTRY = source_jobs, ENTRY

    def prepare(own_config, own_output, own_smoke):
        protocol,jobs = original_prepare(own_config,own_output,own_smoke)
        for job in jobs:
            job['reported_arm'] = job['model']
        # Preserve the unchanged ONLINE wrapper alongside the generic observer.
        target = own_output/'source'/STATES
        target.parent.mkdir(parents=True,exist_ok=True)
        replay.shutil.copy2(ROOT/STATES,target)
        return protocol,jobs

    replay.prepare = prepare
    replay.run(config,output,smoke)


def estimation_context(protocol: dict) -> tuple[dict,str]:
    """Describe the actual projected search policy when presenting oral scores."""
    if (protocol.get('parameter_estimation_scope') != 'prefix' or protocol.get('evaluation_use') != USE
            or protocol.get('belief_intervention') != 'online_retention_evidence'
            or protocol.get('structural_intervention') != 'retained_rule_precision'
            or protocol.get('parameter_candidate_policy') != 'projected_common_search_policy_behavior_refit'
            or not protocol.get('online_fit') or not protocol.get('refit_nomination_sha256')
            or not protocol.get('fixed_coordinates')):
        raise ValueError('ONLINE oral evaluation needs its complete behavior-selection provenance')
    keys = ('parameter_estimation_scope','evaluation_use','belief_intervention','structural_intervention',
            'parameter_candidate_policy','online_fit','refit_nomination_sha256','fixed_coordinates','selected_parameters')
    metadata = {key:protocol[key] for key in keys}
    metadata['window_definition'] = 'original_chronological_boundary'
    note = ('<p>本次参数由ONLINE机制下的原前段选择数据估计。关闭的亲和度转移坐标固定，其他13个可变坐标'
            '沿用原支持范围、阶段预算上限和种子约定。共同热启动仅在固定坐标上投影，去重后补足原定数量；'
            '实际候选数与计算量单独核对。口述用于冻结参数后的表征检验。后段属于开发中的时间外推诊断，'
            '与原方案的差值包含信念机制与重新估参的共同作用。</p>')
    return metadata,note


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from . import model_0923_decision_basis_alignment as alignment
    protocol = alignment.read(source/'protocol.json')
    if protocol['source_sha256'].get(ENTRY) != alignment.digest(ROOT/ENTRY):
        raise ValueError('ONLINE refit readout implementation changed after replay')
    original = alignment.estimation_context
    alignment.estimation_context = estimation_context
    try:
        alignment.run(source,oral_audit,output,reference_model='POOL')
    finally:
        alignment.estimation_context = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('replay','align'),default='replay')
    for name in ('config','source','oral-audit','output','worker'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'replay' and args.config and args.output:
        run(args.config,args.output,args.smoke)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output and not args.smoke:
        align(args.source,args.oral_audit,args.output)
    else:
        parser.error('Use replay with --config, or align with --source and --oral-audit; supply a new --output')
