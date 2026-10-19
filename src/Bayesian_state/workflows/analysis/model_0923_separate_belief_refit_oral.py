"""Compare refitted ONLINE and SEPARATE against the same oral reports.

Use --original ONLINE_ALIGNMENT --revised SEPARATE_ALIGNMENT --subject ID
--arm POOL_OR_SEARCH --output NEW. Both behavior-prefix nominations, original
execution tags, exact state replays, parameter supports and search policies are
checked before comparison. Whole-report and feedback-stratified scores reuse
the existing joint-mass calculations and paired complete-filter deletions.
Differences combine the evidence-memory change with behavioral re-estimation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from . import model_0923_online_evidence_refit_oral as old_adapter
from . import model_0923_online_evidence_refit_states as online_states
from . import model_0923_separate_belief_refit as refit
from . import model_0923_separate_belief_refit_states as states

comparison, strata = old_adapter.comparison, old_adapter.strata


def check_separate_state(job: dict, record: dict, protocol: dict, manifest: dict) -> None:
    """Require executed SEPARATE state evidence, including completed feedback."""
    gamma = job['candidate']['hyperparams']['engine.modules.memory_mod.kwargs.gamma']
    if (any(row.get('belief_intervention') != refit.INTERVENTION for row in (job, record, protocol, manifest))
            or any(row.get('evaluation_use') != states.USE for row in (job, record, protocol, manifest))
            or job.get('probe_mechanism') != 'SEPARATE' or job.get('online_belief') is not False
            or record.get('online_belief') is not False or record.get('separate_belief') is not True
            or job.get('reported_arm') != 'SEPARATE_' + job['model']
            or record.get('arm') != job['model'] or record.get('proposal_arm') != job['model']
            or record.get('belief_gamma') != gamma or record.get('belief_matches_saved_state') is not True
            or record.get('belief_updates') != record['particles'] * (record['trials'] - 1)
            or record.get('exact_behavior_replay') is not True
            or record.get('timing') != 'after_choice_before_current_feedback'
            or record.get('data_use') != 'behavior_only; no_oral_input'):
        raise ValueError('Missing or changed executed SEPARATE state provenance')


def load_separate_group(directory: Path, subject: int, arm: str) -> tuple[tuple, dict, dict]:
    scored, identity, inputs = comparison.load_group(directory, subject, arm)
    read = comparison.alignment.read
    manifest = read(directory / 'manifest.json'); source = Path(manifest['source'])
    protocol = read(source / 'protocol.json'); states.estimation_context(protocol)
    count = 0
    for file in sorted((source / 'states').glob('*.json')):
        record = read(file)
        if (record['subject'], record['arm']) != (subject, arm): continue
        job = read(source / 'jobs' / file.name)
        check_separate_state(job, record, protocol, manifest); count += 1
    if count != protocol['settings']['seeds_per_arm']:
        raise ValueError('Incomplete SEPARATE state group')
    identity['mechanism']['separate_belief'] = True
    return scored, identity, inputs


def validate_identities(before: dict, after: dict, *, executed_sources: dict | None = None) -> None:
    """Permit this one mechanism change and independently selected parameters."""
    for identity in (before, after):
        if (identity['mechanism'].get('retain_learned_precision') is not True
                or identity.get('structural_intervention') != 'retained_rule_precision'
                or identity.get('parameter_estimation_scope') != 'prefix'
                or identity.get('parameter_candidate_policy') != 'projected_common_search_policy_behavior_refit'):
            raise ValueError('Both arms require behavior-prefix refits under the common search policy')
    if (before['mechanism'].get('online_belief') is not True
            or before['mechanism'].get('separate_belief', False) is not False
            or after['mechanism'].get('online_belief', False) is not False
            or after['mechanism'].get('separate_belief') is not True):
        raise ValueError('Require ONLINE then SEPARATE evidence memory')
    old, new = deepcopy(before), deepcopy(after)
    old['mechanism'].pop('online_belief'); new['mechanism'].pop('separate_belief')
    old['mechanism'].pop('separate_belief', None); new['mechanism'].pop('online_belief', None)
    # The new entry binds its old implementation through reference_inputs.
    for name, sha in (executed_sources or {}).items():
        if name in old['model_sources'] and name not in new['model_sources']:
            new['model_sources'][name] = sha
    comparison.validate_comparison(old, new, 'parameters')


def bind_refits(before: dict, after: dict, old_manifest: dict, new_manifest: dict,
                subject: int, arm: str) -> dict:
    read, digest = comparison.alignment.read, comparison.alignment.digest
    old_fit, new_fit = Path(old_manifest['online_fit']), Path(new_manifest['separate_fit'])
    inputs, banks = {}, []

    def bind(path):
        path = Path(path).resolve(); inputs[str(path)] = digest(path); return path

    for fit, identity, manifest, sidecar in (
            (old_fit, before, old_manifest, 'online_refit_protocol.json'),
            (new_fit, after, new_manifest, 'separate_refit_protocol.json')):
        frozen, protocol, done, intervention = [read(bind(fit / name)) for name in
                                               ('nominees.json', 'protocol.json', 'completion.json', sidecar)]
        old_adapter.validate_nomination(identity, manifest, frozen, protocol, done,
                                        digest(fit / 'nominees.json'), subject, arm)
        for name, sha in frozen['source_sha256'].items():
            if digest(bind(refit.ROOT / name)) != sha: raise ValueError('Fitted source changed')
        banks.append((frozen, intervention))
    old, _ = banks[0]; new, intervention = banks[1]; settings = new['settings']
    refit.validate_settings(settings, old['settings'])
    if (refit.ROOT / settings['online_reference_fit']).resolve() != old_fit.resolve():
        raise ValueError('SEPARATE fit refers to a different ONLINE fit')
    online = refit.load_helper('separate_oral_online_reference', refit.ONLINE)
    retained = online.helpers()
    references = refit.reference_inputs(online, retained, retained.helpers(), settings)
    if (intervention['input_sha256'] != references
            or intervention.get('belief_intervention') != refit.INTERVENTION
            or intervention.get('fixed_coordinates') != settings['fixed_coordinates']
            or new_manifest.get('fixed_coordinates') != settings['fixed_coordinates']
            or new_manifest.get('belief_intervention') != refit.INTERVENTION):
        raise ValueError('SEPARATE belief or common search policy changed')
    inputs.update(references)
    for identity in (before, after):
        if any(identity['candidate']['hyperparams'].get(k) != v for k, v in settings['fixed_coordinates'].items()):
            raise ValueError('Inactive coordinate differs from the common fitted value')
    for manifest, fit, adapter, key in ((old_manifest, old_fit, online_states, 'online_fit'),
                                        (new_manifest, new_fit, states, 'separate_fit')):
        state_protocol = read(bind(Path(manifest['source']) / 'protocol.json'))
        evaluation = refit.ROOT / state_protocol['settings']['source']
        _, metadata = adapter.source_jobs(state_protocol['settings'], evaluation, False)
        if (metadata['refit_nomination_sha256'] != manifest['refit_nomination_sha256']
                or Path(metadata[key]).resolve() != fit.resolve()):
            raise ValueError('Executed filters and oral states refer to different nominees')
        inputs.update(metadata['refit_replay_inputs'])
    return inputs


def feedback_comparison(old: dict, new: dict, subject: int, arm: str):
    table, deleted = old_adapter.feedback_comparison(old, new, subject, arm)
    # Preserve the numerical calculation while recording the actual mechanisms.
    renamed = {}
    for key, value in deleted.items():
        prefix, suffix = key.split('_', 1)
        renamed[{'retained': 'online', 'online': 'separate'}[prefix] + '_' + suffix] = value
    return table, renamed


def run(original: Path, revised: Path, output: Path, subject: int, arm: str) -> None:
    if output.exists(): raise FileExistsError('Use a new SEPARATE refit oral comparison directory')
    old, before, inputs = comparison.load_group(original, subject, arm)
    new, after, more = load_separate_group(revised, subject, arm); inputs.update(more)
    old_manifest = comparison.alignment.read(original / 'manifest.json')
    new_manifest = comparison.alignment.read(revised / 'manifest.json')
    refit_inputs = bind_refits(before, after, old_manifest, new_manifest, subject, arm); inputs.update(refit_inputs)
    executed_sources = {str(Path(p).relative_to(refit.ROOT)): sha for p, sha in refit_inputs.items()
                        if Path(p).is_relative_to(refit.ROOT) and Path(p).suffix == '.py'}
    validate_identities(before, after, executed_sources=executed_sources)
    scores, trials = comparison.paired_scores(old, new)
    old_bank, more = strata.load_bank(original, subject, arm, old); inputs.update(more)
    new_bank, more = strata.load_bank(revised, subject, arm, new); inputs.update(more)
    feedback, deleted = feedback_comparison(old_bank, new_bank, subject, arm)
    trials['feedback'], trials['category'] = old_bank['feedback'], old_bank['categories']
    for path, sha in inputs.items():
        if comparison.alignment.digest(Path(path)) != sha: raise ValueError('Oral comparison input changed')
    output.mkdir(parents=True)
    for name, table in [('scores', scores), ('trials', trials), ('feedback_scores', feedback)]:
        table.to_csv(output / (name + '.csv'), index=False)
    with (output / 'feedback_jackknife.npz').open('xb') as stream:
        np.savez_compressed(stream, **deleted, seeds=old_bank['seeds'])
    note = ('ONLINE按δ保持形成规则信念的证据；SEPARATE按γ保持信念证据，δ继续管理候选。'
        '两方案均保留规则已学精度，个体参数各自由同一前段行为目标和共同搜索政策估计。'
        '本页差值为SEPARATE减ONLINE，包含机制变化与重新估参的共同作用。'
        '口述仅用于表征评价，两个方案使用相同报告、读法、时间分段及完整滤波种子。'
        '模拟标准误由配对删除完整滤波重复得到，描述固定参数下的计算波动。'
        '表格保留结构域与功能域的支持、冲突和未确定质量；反馈分层另检验本题选择加权的影响。')
    display = scores[scores.subset.eq('complete')]
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>SEPARATE重估后的口述比较</title><style>body{max-width:1400px;margin:30px auto;font:16px/1.6 sans-serif}'
        'td,th{padding:5px}</style><h1>SEPARATE与ONLINE重估后的口述比较</h1><p>' + note + '</p>'
        + display.to_html(index=False, float_format=lambda x: f'{x:.6f}')
        + '<p><a href="scores.csv">全部整句评分</a> · <a href="trials.csv">完整试次</a> · '
        '<a href="feedback_scores.csv">反馈分层与本题选择加权</a></p></html>')
    paths = [Path(module.__file__) for module in (old_adapter, comparison, strata, refit, states, online_states,
                                                 comparison.alignment)]
    paths += [Path(__file__), refit.ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py']
    (output / 'source').mkdir()
    for path in paths: shutil.copy2(path, output / 'source' / path.name)
    result = dict(status='complete', subject=subject, arm=arm, original=str(original.resolve()), revised=str(revised.resolve()),
        contrast_kind='separate_belief_refit', seed_relation='paired', original_parameters=before, revised_parameters=after,
        bound_reference_sources=executed_sources, original_scores_reproduced=True, revised_scores_reproduced=True,
        input_sha256=inputs, source_sha256={str(p.relative_to(refit.ROOT)): comparison.alignment.digest(p) for p in paths},
        artifacts={p.name: comparison.alignment.digest(p) for p in output.iterdir() if p.is_file()},
        data_use='Behavior-prefix estimation; combined evidence-memory and re-estimation effect; oral external validation; development.',
        uncertainty='paired_delete_one_complete_filter_seed_mcse')
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('original', 'revised', 'output'): parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--arm', choices=('POOL', 'SEARCH'), required=True)
    args = parser.parse_args(); run(args.original, args.revised, args.output, args.subject, args.arm)
