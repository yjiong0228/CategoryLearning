"""Replay behavior-selected SEPARATE nominees for external oral validation.

Use --phase replay --config YAML --output NEW [--smoke], then --phase align
--source REPLAY --oral-audit AUDIT --output NEW. Both original proposal arms,
all independent evaluation seeds and the fitted particle count are preserved.
The existing SEPARATE observer reproduces each saved choice prediction exactly.
Smoke reads the first seed of each arm for at most sixteen trials. Explicit
eight-trial evaluation fixtures are accepted only with --smoke and remain
ineligible for scientific oral alignment.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
STATES = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_states.py'
REFIT = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_refit.py'
SUMMARY = 'src/Bayesian_state/evaluation/model_0923_separate_belief_refit.py'
INTERVENTION = 'separate_gamma_belief_evidence'
USE = 'development_separate_belief_refit'


def helper(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    return helper('separate_refit_state_base', STATES)


def checked_execution(planned: dict, actual: dict, references: dict, fixed: dict,
                      record: dict, precision: dict, belief: dict) -> None:
    from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import check_execution
    expected = dict(planned, structural_intervention='retained_rule_precision', retain_precision=True,
        retention_reference_inputs=references, belief_intervention=INTERVENTION,
        online_belief=False, probe_mechanism='SEPARATE', fixed_coordinates=fixed)
    if (actual != expected or actual['phase'] != 'evaluate' or actual['model'] not in ('POOL', 'SEARCH')
            or len(actual['seeds']) != 1
            or any(record[k] != actual[k] for k in ('candidate', 'model', 'particles', 'seeds'))):
        raise ValueError('SEPARATE refit evaluation differs from the frozen plan')
    check_execution(actual, record, belief, precision)


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    from src.Bayesian_state.utils.seeding import stable_seed
    refit = helper('separate_refit_state_fit', REFIT)
    replay = helpers().helpers().helpers(); read, digest = replay.read, replay.digest
    completed, evaluation = read(source / 'completion.json'), read(source / 'protocol.json')
    fit = Path(evaluation['fit'])
    required = ('protocol.json', 'nominees.json', 'completion.json', 'separate_refit_protocol.json')
    protocol, frozen, done, intervention = (read(fit / name) for name in required)
    nomination = digest(fit / 'nominees.json'); fixture = evaluation.get('smoke', False)
    if (fixture and not smoke or any(r['smoke'] is not fixture for r in (protocol, frozen, done))
            or completed['status'] != 'complete' or done['status'] != 'complete'
            or done['test_choices_used'] is not False or frozen['status'] != 'frozen_before_test'
            or any(r['nomination_sha256'] != nomination for r in (done, evaluation, completed))
            or frozen['settings'] != protocol['settings']
            or frozen['source_sha256'] != protocol['source_sha256']
            or evaluation['source_sha256'] != frozen['source_sha256']):
        raise ValueError('Require a completed behavior-only SEPARATE fit; fixtures require smoke')
    original = frozen['settings']
    if (settings['sample_role'] != 'development' or not 1 <= settings['workers'] <= 128
            or original.get('belief_intervention') != INTERVENTION
            or intervention.get('belief_intervention') != INTERVENTION
            or not settings['subjects'] or len(set(settings['subjects'])) != len(settings['subjects'])
            or not set(settings['subjects']) <= set(original['subjects'])
            or settings['arms'] != ['POOL', 'SEARCH'] or set(original['arms']) != set(settings['arms'])):
        raise ValueError('Declare both SEPARATE proposal arms and unique development subjects')
    for name in (REFIT, refit.PROBE, refit.BELIEF):
        if frozen['source_sha256'].get(name) != digest(ROOT / name):
            raise ValueError('Missing executed SEPARATE refit or belief source')
    online = refit.load_helper('separate_state_reference_online', refit.ONLINE)
    retention = online.helpers(); references = refit.reference_inputs(online, retention, retention.helpers(), original)
    if (intervention['input_sha256'] != references
            or intervention['fixed_coordinates'] != original['fixed_coordinates']):
        raise ValueError('SEPARATE refit reference or fixed coordinate changed')
    bindings = dict(references)
    for name, sha in frozen['source_sha256'].items():
        if digest(ROOT / name) != sha: raise ValueError('Fitted implementation changed')
        bindings[str((ROOT / name).resolve())] = sha
    for path in [*(fit / name for name in required), source / 'protocol.json', source / 'completion.json',
                 *(ROOT / name for name in (ENTRY, STATES, REFIT, SUMMARY, replay.REENTRY, replay.PROPOSAL,
                                            helpers().BELIEF, helpers().OBSERVER))]:
        bindings[str(path.resolve())] = digest(path)
    budget = original['budgets']['evaluate']
    if fixture:
        if evaluation.get('validation_budget') != {'particles': 4, 'seeds': 1, 'trials': 8}:
            raise ValueError('Require the explicit short evaluation fixture budget')
        budget = evaluation['validation_budget']
    if settings['seeds_per_arm'] != budget['seeds']:
        raise ValueError('Preserve every declared evaluation filter seed')
    nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
    plans = {j['name']: j for j in evaluation['jobs']}
    if (len(nominees) != len(frozen['nominees']) or len(plans) != len(evaluation['jobs'])
            or completed['jobs'] != len(plans)):
        raise ValueError('Duplicate or incomplete nomination/evaluation')
    files, selected = [], []
    for subject in settings['subjects'][:1] if smoke else settings['subjects']:
        for arm in settings['arms']:
            point = nominees[subject, arm]
            data = evaluation['validation_data'] if fixture else frozen['data']
            folder = (Path(data) / f'subject_{subject}').resolve()
            if read(folder / 'prepared.json')['subject'] != subject: raise ValueError('Wrong participant')
            seeds = [stable_seed(dict(protocol=original['protocol'], base=original['seed'],
                stage='evaluate', subject=subject, repeat=i)) for i in range(budget['seeds'])]
            particles = (budget['particles'] if fixture else
                {int(k): v for k, v in original['particle_overrides'].items()}.get(subject, budget['particles']))
            group = sorted((source / 'jobs').glob(f'evaluate_S{subject}_{arm}_*_C*.json'))
            if len(group) != len(seeds): raise ValueError('Incomplete independent filter bank')
            for i, (file, seed) in enumerate(zip(group, seeds)):
                name = f'evaluate_S{subject}_{arm}_{point["id"]}_C{i:03d}'; planned = plans.get(name)
                if (planned is None or file.stem != name or planned['phase'] != 'evaluate'
                        or planned['smoke'] is not fixture or planned['model'] != arm
                        or planned['candidate'] != point or planned['seeds'] != [seed]
                        or planned['particles'] != particles or Path(planned['folder']).resolve() != folder
                        or planned['arm_settings'] != original['arms'][arm] or planned['mechanism'] != original['mechanism']
                        or planned['source_sha256'] != frozen['source_sha256']):
                    raise ValueError('Evaluation differs from the behavior-selected nominee')
                actual = read(file); target = Path(planned['target'])
                record, precision, belief = (read(target.with_suffix(s)) for s in
                                            ('.json', '.precision.json', '.belief.json'))
                checked_execution(planned, actual, references, original['fixed_coordinates'], record, precision, belief)
                if fixture and record['trials'] != 8: raise ValueError('Fixture is not the declared short sequence')
                for path in (file, folder / 'prepared.json', target.with_suffix('.json'),
                             target.with_suffix('.precision.json'), target.with_suffix('.belief.json')):
                    bindings[str(path.resolve())] = digest(path)
            files.extend(group[:1] if smoke else group)
            selected.append(dict(subject=subject, arm=arm, parameter_id=point['id']))
    return files, dict(parameter_estimation_scope='prefix', evaluation_use=USE,
        parameter_candidate_policy='projected_common_search_policy_behavior_refit',
        structural_intervention='retained_rule_precision', belief_intervention=INTERVENTION,
        selected_parameters=selected, separate_fit=str(fit.resolve()), refit_nomination_sha256=nomination,
        fixed_coordinates=original['fixed_coordinates'], refit_replay_inputs=bindings,
        report_arm_policy='POOL/SEARCH identify the original proposal; both execute SEPARATE',
        scientific_scope=not fixture)


def worker(file: Path) -> None:
    """Reuse the tested SEPARATE state worker with refit labels and POOL imports."""
    separate = helpers(); observer = separate.helpers(); replay = observer.helpers(); job = replay.read(file)
    if (job.get('evaluation_use') != USE or job.get('parameter_estimation_scope') != 'prefix'
            or job['model'] not in ('POOL', 'SEARCH') or job['reported_arm'] != 'SEPARATE_' + job['model']
            or job.get('belief_intervention') != INTERVENTION or job.get('online_belief') is not False):
        raise ValueError('Refitted replay lost its mechanism, arm or selection identity')
    original_load, original_write = replay.load, replay.write

    def load(name, relative):
        module = original_load(name, relative)
        if relative == replay.REFIT:
            base_load = module.base.load

            def with_shared_normalizer(child_name, child_relative):
                proposal_name = 'src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal'
                if child_relative == replay.REENTRY and proposal_name not in sys.modules:
                    base_load(proposal_name, replay.PROPOSAL)
                return base_load(child_name, child_relative)
            module.base.load = with_shared_normalizer
        return module

    def write(path, value):
        if path == Path(job['target']).with_suffix('.json'):
            if value['arm'] != job['reported_arm'] or value['proposal_arm'] != job['model']:
                raise ValueError('Unexpected SEPARATE readout label')
            value = dict(value, arm=job['model'])
        original_write(path, value)

    replay.load, replay.write = load, write
    observer.helpers = lambda: replay
    separate.helpers = lambda: observer
    separate.CELLS = ('SEPARATE_POOL', 'SEPARATE_SEARCH')
    separate.SCOPE, separate.USE = 'prefix', USE
    separate.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    replay = helpers().helpers().helpers(); original_prepare = replay.prepare
    replay.source_jobs, replay.ENTRY = source_jobs, ENTRY

    def prepare(own_config, own_output, own_smoke):
        protocol, jobs = original_prepare(own_config, own_output, own_smoke)
        for job in jobs:
            job['reported_arm'] = 'SEPARATE_' + job['model']
            if own_smoke:
                job['trial_limit'] = min(16, replay.read(Path(job['reference']).with_suffix('.json'))['trials'])
        for relative in (STATES, helpers().OBSERVER):
            target = own_output / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True); replay.shutil.copy2(ROOT / relative, target)
        return protocol, jobs

    replay.prepare = prepare; replay.run(config, output, smoke)


def estimation_context(protocol: dict) -> tuple[dict, str]:
    if (protocol.get('parameter_estimation_scope') != 'prefix' or protocol.get('evaluation_use') != USE
            or protocol.get('belief_intervention') != INTERVENTION
            or protocol.get('structural_intervention') != 'retained_rule_precision'
            or protocol.get('parameter_candidate_policy') != 'projected_common_search_policy_behavior_refit'
            or not protocol.get('separate_fit') or not protocol.get('refit_nomination_sha256')
            or not protocol.get('fixed_coordinates') or protocol.get('scientific_scope') is not True):
        raise ValueError('Scientific oral evaluation needs the full behavior-selection provenance')
    keys = ('parameter_estimation_scope', 'evaluation_use', 'belief_intervention', 'structural_intervention',
            'parameter_candidate_policy', 'separate_fit', 'refit_nomination_sha256', 'fixed_coordinates', 'selected_parameters')
    return dict({k: protocol[k] for k in keys}, window_definition='original_chronological_boundary'), (
        '<p>参数由SEPARATE机制下的原前段选择数据估计。γ保持规则信念证据，δ管理候选证据，'
        '关闭的亲和度转移坐标固定。POOL与SEARCH沿用原支持范围、共同起点、阶段预算和种子约定。'
        '全部状态先复现保存的行为预测，口述随后检验本题选择后、反馈前的局部分类依据。'
        '后段属于开发中的时间外推诊断；与原机制重估结果的差值包含机制与参数的共同作用。</p>')


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from . import model_0923_decision_basis_alignment as alignment
    protocol, done = alignment.read(source / 'protocol.json'), alignment.read(source / 'completion.json')
    if (done['status'] != 'complete' or done['smoke'] or done['jobs'] != done['exact_behavior_replays']
            or protocol['source_sha256'].get(ENTRY) != alignment.digest(ROOT / ENTRY)):
        raise ValueError('Need complete exact scientific state replays')
    original = alignment.estimation_context; alignment.estimation_context = estimation_context
    try: alignment.run(source, oral_audit, output, reference_model='POOL')
    finally: alignment.estimation_context = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('replay', 'align'), default='replay')
    for name in ('config', 'source', 'oral-audit', 'output', 'worker'): parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.phase == 'replay' and args.config and args.output: run(args.config, args.output, args.smoke)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output and not args.smoke:
        align(args.source, args.oral_audit, args.output)
    else: parser.error('Replay needs --config; align needs --source and --oral-audit; supply --output')
