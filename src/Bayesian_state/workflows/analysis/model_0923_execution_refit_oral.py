"""Compare behavior-refitted MIX and SELECT on identical frozen oral reports.

Use --original MIX_ALIGNMENT --revised SELECT_ALIGNMENT --subject ID --arm
POOL_OR_SEARCH --output NEW. Full scientific state adapters bind the actual
nominees, task boundaries, complete filter banks and execution identities.
Scores and paired complete-filter MCSE reuse the existing joint-mass readout.
The contrast includes execution-policy change and behavior-only re-estimation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import model_0923_oral_parameter_comparison as comparison
from . import model_0923_oral_feedback_strata as strata
from . import model_0923_online_evidence_refit_oral as common

ROOT = Path(__file__).resolve().parents[4]
ADAPTERS = {
    'development_separate_belief_refit': ('model_0923_separate_belief_refit_states', 'separate_fit', 'MIX'),
    'development_dominant_execution_refit': ('model_0923_dominant_execution_refit_states', 'execution_fit', 'SELECT'),
    'development_cohort_belief_refit': ('model_0923_cohort_belief_states', 'cohort_fit', 'MIX'),
    'development_cohort_dominant_execution_refit': ('model_0923_cohort_execution_states', 'execution_fit', 'SELECT'),
}
read, digest = comparison.alignment.read, comparison.alignment.digest
POLICY = 'existing_dominant_rule_only'


def check_state(job: dict, record: dict, policy: str, arm: str) -> None:
    """Require executed SEPARATE evidence and the declared response policy."""
    label = arm if policy == 'MIX' else 'SELECT_' + arm
    gamma = job['candidate']['hyperparams']['engine.modules.memory_mod.kwargs.gamma']
    if (policy not in ('MIX', 'SELECT') or job.get('model') != arm or record.get('arm') != label
            or job.get('retain_precision') is not True or record.get('separate_belief') is not True
            or record.get('online_belief') is not False
            or job.get('belief_intervention') != 'separate_gamma_belief_evidence'
            or record.get('belief_intervention') != 'separate_gamma_belief_evidence'
            or record.get('belief_matches_saved_state') is not True or record.get('belief_gamma') != gamma
            or record.get('belief_updates') != record['particles'] * (record['trials'] - 1)
            or record.get('particle_transitions') != record['particles'] * record['trials']
            or record.get('timing') != 'after_choice_before_current_feedback'
            or record.get('data_use') != 'behavior_only; no_oral_input'
            or record.get('exact_behavior_replay') is not True
            or any(record.get(k) != job.get(k) for k in ('candidate', 'seeds', 'particles', 'evaluation_use'))):
        raise ValueError('Invalid completed SEPARATE state identity')
    if policy == 'SELECT':
        if (any(x.get('execution_policy') != 'SELECT' or x.get('rule_execution_policy') != POLICY for x in (job, record))
                or record.get('execution_updates') != record['particles'] * record['trials']):
            raise ValueError('Missing executed SELECT policy')
    elif any(x.get('execution_policy') is not None or x.get('rule_execution_policy') is not None for x in (job, record)):
        raise ValueError('MIX reference already changes the response policy')


def bind_state_fit(directory: Path, identity: dict, subject: int, arm: str, policy: str) -> tuple[Path, dict, dict]:
    """Reuse the original scientific replay adapter to revalidate its full bank."""
    manifest = read(directory / 'manifest.json')
    declaration = ADAPTERS.get(manifest.get('evaluation_use'))
    if declaration is None or declaration[2] != policy:
        raise ValueError('Unsupported scientific MIX/SELECT replay scope')
    name, fit_key, _ = declaration
    adapter = importlib.import_module('src.Bayesian_state.workflows.analysis.' + name)
    source = Path(manifest['source'])
    protocol = read(source / 'protocol.json')
    context, _ = adapter.estimation_context(protocol)
    if any(manifest.get(key) != value for key, value in context.items()):
        raise ValueError('Alignment changed its parameter-estimation context')
    files, metadata = adapter.source_jobs(protocol['settings'], ROOT / protocol['settings']['source'], False)
    if any(metadata.get(key) != value for key, value in context.items() if key in metadata):
        raise ValueError('State replay and executed evaluation context differ')
    inputs = dict(metadata['refit_replay_inputs'])
    expected = set()
    for file in files:
        job = read(file)
        prepared = read(Path(job['folder']) / 'prepared.json')
        if (prepared['subject'], job['model']) == (subject, arm):
            expected.add(str(file.resolve()))
    actual = set()
    label = arm if policy == 'MIX' else 'SELECT_' + arm
    for file in sorted((source / 'states').glob('*.json')):
        record = read(file)
        if (record['subject'], record['arm']) != (subject, label):
            continue
        job_file = source / 'jobs' / file.name
        job = read(job_file)
        check_state(job, record, policy, arm)
        actual.add(str(Path(job['original_job']).resolve()))
        for path in (file, job_file):
            inputs[str(path.resolve())] = digest(path)
    if len(actual) != protocol['settings']['seeds_per_arm'] or actual != expected:
        raise ValueError('Oral state bank differs from the complete frozen filter bank')
    fit = Path(manifest[fit_key])
    frozen, search, done = [read(fit / name) for name in ('nominees.json', 'protocol.json', 'completion.json')]
    common.validate_nomination(identity, manifest, frozen, search, done, digest(fit / 'nominees.json'), subject, arm)
    for path in [source / 'protocol.json', Path(adapter.__file__),
                 *[fit / name for name in ('nominees.json', 'protocol.json', 'completion.json')]]:
        inputs[str(path.resolve())] = digest(path)
    return fit, frozen, inputs


def common_arm(scored: tuple, arm: str) -> tuple:
    """Align only presentation labels after each original score is reproduced."""
    trials, rows, estimates = scored
    return trials.assign(arm=arm), [dict(row, arm=arm) for row in rows], estimates


def match_identities(before: dict, after: dict, references: dict) -> None:
    old, new = deepcopy(before), deepcopy(after)
    for identity in (old, new):
        if (identity.get('parameter_estimation_scope') != 'prefix'
                or identity.get('parameter_candidate_policy') != 'projected_common_search_policy_behavior_refit'
                or identity['mechanism'].get('retain_learned_precision') is not True
                or identity.get('structural_intervention') != 'retained_rule_precision'):
            raise ValueError('Require common-policy behavior-prefix nominees with retained precision')
    # Some original wrapper hashes are recorded in the validated reference
    # bindings of the new fit. A missing hash is accepted only via that proof.
    for name, sha in old['model_sources'].items():
        if name not in new['model_sources'] and references.get(str((ROOT / name).resolve())) == sha:
            new['model_sources'][name] = sha
    comparison.validate_comparison(old, new, 'parameters')


def run(original: Path, revised: Path, output: Path, subject: int, arm: str) -> None:
    if output.exists():
        raise FileExistsError('Use a new execution-refit oral comparison directory')
    old, before, inputs = comparison.load_group(original, subject, arm)
    new, after, more = comparison.load_group(revised, subject, 'SELECT_' + arm)
    inputs.update(more)
    old_fit, old_frozen, more = bind_state_fit(original, before, subject, arm, 'MIX'); inputs.update(more)
    _, new_frozen, more = bind_state_fit(revised, after, subject, arm, 'SELECT'); inputs.update(more)
    if (ROOT / new_frozen['settings']['execution_reference_fit']).resolve() != old_fit.resolve():
        raise ValueError('SELECT was fitted against another MIX reference')
    match_identities(before, after, inputs)
    scores, trials = comparison.paired_scores(common_arm(old, arm), common_arm(new, arm))
    old_bank, more = strata.load_bank(original, subject, arm, old); inputs.update(more)
    new_bank, more = strata.load_bank(revised, subject, 'SELECT_' + arm, new); inputs.update(more)
    old_bank['trials'] = old_bank['trials'].assign(arm=arm)
    new_bank['trials'] = new_bank['trials'].assign(arm=arm)
    feedback, deleted = common.feedback_comparison(old_bank, new_bank, subject, arm)
    deleted = {('mix_' + k.removeprefix('retained_') if k.startswith('retained_') else
                'select_' + k.removeprefix('online_')): value for k, value in deleted.items()}
    trials['feedback'], trials['category'] = old_bank['feedback'], old_bank['categories']
    sources = [Path(module.__file__) for module in (comparison, strata, common, comparison.alignment)]
    sources += [Path(__file__), ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py']
    inputs.update({str(path.resolve()): digest(path) for path in sources})
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Execution comparison input changed')
    output.mkdir(parents=True)
    for name, table in [('scores', scores), ('trials', trials), ('feedback_scores', feedback)]:
        table.to_csv(output / (name + '.csv'), index=False)
    np.savez_compressed(output / 'feedback_jackknife.npz', **deleted, seeds=old_bank['seeds'])
    note = ('本页比较重新估参后的MIX与SELECT，差值为SELECT减MIX。两者使用SEPARATE信念和已学精度保持，'
            '个体参数各自由原训练选择确定。相同报告、固定读法及完整滤波种子用于外部表征检验。'
            'original列为MIX，revised列为SELECT；误差为配对删除完整滤波所得的模拟标准误。'
            '后段属于开发中的时间外推诊断。整句支持检验单条局部规则分量，联合理由读出另行检验。')
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>MIX与SELECT重估后的口述比较</title><h1>MIX与SELECT重估后的口述比较</h1><p>' + note + '</p>'
        + scores[scores.subset.eq('complete')].to_html(index=False, float_format=lambda x: f'{x:.6f}')
        + '<p><a href="scores.csv">全部分数</a> · <a href="trials.csv">逐题结果</a> · '
        '<a href="feedback_scores.csv">反馈分层</a></p></html>')
    (output / 'source').mkdir()
    for path in sources:
        shutil.copy2(path, output / 'source' / path.name)
    result = dict(status='complete', subject=subject, arm=arm, original=str(original.resolve()),
                  revised=str(revised.resolve()), contrast='SELECT minus MIX after behavior-prefix re-estimation',
                  scientific_scope='development_external_oral_evaluation', seed_relation='paired',
                  original_parameters=before, revised_parameters=after, input_sha256=inputs,
                  original_scores_reproduced=True, revised_scores_reproduced=True,
                  artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()},
                  uncertainty='paired_delete_one_complete_filter_seed_mcse')
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'revised', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--arm', choices=('POOL', 'SEARCH'), required=True)
    args = parser.parse_args()
    run(args.original, args.revised, args.output, args.subject, args.arm)
