"""Check nonzero maintenance values after a zero-only confirmation shortlist.

Use --study COMPLETED_FIT --config YAML --output NEW [--smoke]. All nuisance
parameters stay at the frozen MIX training estimate. A fresh paired seed bank
scores MIX and every registered STICKY kappa on the original training prefix.
This is a conditional parameter diagnostic; the primary fitted estimates and
later-choice evaluation keep their own provenance. No oral reports are read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import model_1023_execution_study as study
from . import model_1023_execution_validation as validation
from .model_1022_belief_summary import arrays, nll, jackknife_se
from ...optimization.search.adaptive_proposals import candidate
from ...utils.seeding import stable_seed

read, write, digest = study.read, study.write, study.digest


def run(fit: Path, config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new conditional-parameter destination')
    policy = yaml.safe_load(config.read_text())
    if (not 1 <= policy['parallel_budget'] <= 128 or policy['particles'] < 1 or policy['seeds'] < 2
            or policy['selection_basis'] != 'all_sticky_confirmation_candidates_have_zero_kappa'):
        raise ValueError('Invalid declared diagnostic budget or participant-selection rule')
    done, protocol = read(fit / 'completion.json'), read(fit / 'protocol.json')
    frozen = read(fit / 'nominees.json')
    if (done['status'] != 'complete' or done['smoke'] != smoke
            or frozen['status'] != 'frozen_before_test' or frozen['later_choices_used']
            or frozen.get('oral_used', False)):
        raise ValueError('Require frozen training-only estimates of matching scope')
    if (done['nominees_sha256'] != digest(fit / 'nominees.json')
            or protocol['source_sha256'] != study.sources(study.driver.io)):
        raise ValueError('Frozen estimates or numerical implementation changed')
    if (policy['data_use'] != 'original_training_prefix_only'
            or policy['nuisance_parameter_source'] != 'frozen_MIX_training_estimate'
            or policy['kappa_values'] != protocol['policy']['kappa_values']):
        raise ValueError('Require the declared conditional grid and training boundary')
    subjects = policy['subjects'][:1] if smoke else policy['subjects']
    nominees = {row['subject']: row for row in frozen['nominees'] if row['arm'] == 'MIX'}
    if not set(subjects) <= set(nominees):
        raise ValueError('Required baseline training estimate missing')
    bindings = {str(fit / name): digest(fit / name) for name in ('nominees.json', 'protocol.json')}
    if not smoke:
        registration = Path(policy['registration'])
        registered = read(registration)
        if (registered['configuration_sha256'] != digest(config)
                or registered['fit_protocol_sha256'] != digest(fit / 'protocol.json')
                or registered['nominees_existed'] or registered['confirmation_complete_existed']):
            raise ValueError('Require the diagnostic policy registered before final confirmation')
        bindings[str(registration)] = digest(registration)
        audit = Path(policy['selection_audit'])
        predeclared = read(audit / 'completion.json')
        if registered['selection_audit_sha256'] != digest(audit / 'completion.json'):
            raise ValueError('Pre-confirmation selection audit differs from registration')
        coverage = pd.read_csv(audit / 'candidate_coverage.csv')
        if digest(audit / 'candidate_coverage.csv') != predeclared['candidate_coverage_sha256']:
            raise ValueError('Pre-confirmation coverage audit changed')
        selected = sorted(int(sid) for sid, rows in coverage.loc[coverage.model.eq('STICKY')].groupby('subject')
                          if set(rows.kappa) == {0.})
        if selected != sorted(subjects):
            raise ValueError('Diagnostic participants differ from the declared shortlist rule')
        confirmation = read(fit / 'confirmation_completion.json')
        for group in confirmation['groups']:
            if group['subject'] in subjects and group['model'] == 'STICKY':
                if any(row['hyperparams'][study.KAPPA] != 0 for row in group['rows']):
                    raise ValueError('Actual confirmation differs from its zero-only plan')
        bindings[str(audit / 'completion.json')] = digest(audit / 'completion.json')
        bindings[str(fit / 'confirmation_completion.json')] = digest(fit / 'confirmation_completion.json')
    used_seeds = {seed for file in (fit / 'jobs').glob('*.json') for seed in read(file).get('seeds', [])}
    count, particles = (2, 2) if smoke else (policy['seeds'], policy['particles'])
    own_seeds = {sid: [stable_seed(dict(base=policy['seed'], role='execution_conditional_kappa',
                        subject=sid, smoke=smoke, repeat=i)) for i in range(count)] for sid in subjects}
    flat_seeds = [seed for values in own_seeds.values() for seed in values]
    if len(set(flat_seeds)) != len(flat_seeds) or set(flat_seeds) & used_seeds:
        raise ValueError('Diagnostic and original seed banks overlap')
    jobs, settings = [], []
    for sid in subjects:
        nominee = nominees[sid]
        originals = sorted((fit / 'jobs').glob(f'confirmation_S{sid}_MIX_{nominee["candidate"]["id"]}_*.json'))
        if not originals:
            raise ValueError('Original confirmation job missing')
        original = read(originals[0])
        bindings[str(originals[0])] = digest(originals[0])
        if original['phase'] != 'training' or original['execution_mode'] != 'MIX':
            raise ValueError('Require an original training-only baseline job')
        meta = read(Path(original['folder']) / 'prepared.json')
        bindings[str(Path(original['folder']) / 'prepared.json')] = digest(Path(original['folder']) / 'prepared.json')
        if meta['subject'] != sid:
            raise ValueError('Original prepared participant differs')
        point = nominee['candidate']['hyperparams']
        settings.append(dict(subject=sid, nuisance_point=point, training_trials=meta['training_trials']))
        for mode, kappa in [('MIX', 0.)] + [('STICKY', k) for k in policy['kappa_values']]:
            label = 'MIX_reference' if mode == 'MIX' else f'STICKY_K{kappa:g}'
            own = candidate(dict(point, **{study.KAPPA: kappa}), 'conditional_parameter_diagnostic')
            for index, seed in enumerate(own_seeds[sid]):
                job = deepcopy(original)
                # Score writers replace a filename suffix; decimal points in
                # kappa labels would remove the repeat identifier and collide.
                name = f'conditional_S{sid}_{label}_C{index:03d}'.replace('.', 'p')
                job.update(name=name, diagnostic_subject=sid, diagnostic_label=label, execution_mode=mode,
                    candidate=own, seeds=[seed], particles=particles, smoke=smoke,
                    target=str((output / 'scores' / name).resolve()))
                jobs.append(job)
    targets = [Path(job['target']).with_suffix('.npz') for job in jobs]
    if len(set(targets)) != len(targets):
        raise ValueError('Conditional score files must be unique across points and repeats')
    output.mkdir(parents=True)
    source = str(Path(__file__).resolve().relative_to(study.ROOT))
    study.base.archive(output, config, dict(validation.sources(), **study.sources(study.driver.io),
                                          **{source: digest(Path(__file__))}))
    write(output / 'protocol.json', dict(policy=policy, smoke=smoke, settings=settings, jobs=jobs,
        inputs_sha256=bindings, fit=str(fit.resolve()), source_sha256=digest(Path(__file__)),
        interpretation='conditional scores at fixed nuisance parameters; original training data only'))
    workers = 1 if smoke else min(policy['parallel_budget'], len(os.sched_getaffinity(0)), len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        validation.NumericalQueue(pool).execute(jobs, output, workers)
    rows = []
    for own in settings:
        sid = own['subject']
        banks = {}
        for label in sorted({job['diagnostic_label'] for job in jobs if job['diagnostic_subject'] == sid}):
            group = [job for job in jobs if job['diagnostic_subject'] == sid and job['diagnostic_label'] == label]
            parts = [arrays(Path(job['target'])) for job in group]
            np.testing.assert_array_equal(np.concatenate([part['seeds'] for part in parts]), own_seeds[sid])
            probability = np.concatenate([part['observed_probability'] for part in parts])
            if probability.shape != (count, min(8, own['training_trials']) if smoke else own['training_trials']):
                raise ValueError('Conditional scores crossed the declared training boundary')
            banks[label] = probability
        reference = banks['MIX_reference']
        for label, probability in banks.items():
            deleted = np.array([nll(np.delete(probability, i, axis=0))-nll(np.delete(reference, i, axis=0))
                                for i in range(count)])
            rows.append(dict(subject=sid, label=label, kappa=0. if label == 'MIX_reference' else float(label.split('_K')[1]),
                particles=particles, seeds=count, trials=probability.shape[1], training_nll=nll(probability),
                difference_from_MIX=nll(probability)-nll(reference), difference_mcse=jackknife_se(deleted),
                nuisance_point_id=nominees[sid]['candidate']['id']))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'scores.csv', index=False)
    (output / 'report.md').write_text('\n'.join(['# 执行维持参数的条件检验', '',
        '被试按高精度确认候选全为零维持参数的事前记录选定。其他参数固定在混合模型的训练段估计值。',
        '所有已登记的维持参数使用新的配对种子，在原训练段计算选择似然。差值为当前点减混合模型，负值表示更好。',
        '该曲线以固定其他参数为条件；主模型的联合估计与后段评价见原比较。MCSE描述有限滤波重复的数值波动。',
        '零维持STICKY与MIX的理论边缘行为相同，其评分差异反映数值近似误差。',
        '本目录为小规模软件验证。' if smoke else '本目录为开发样本的补充参数检验。', '']))
    write(output / 'completion.json', dict(status='complete', smoke=smoke, subjects=subjects, jobs=len(jobs),
        source_sha256=digest(Path(__file__)), protocol_sha256=digest(output / 'protocol.json'),
        scores_sha256=digest(output / 'scores.csv'), report_sha256=digest(output / 'report.md')))
    print(table.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'config', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.study, args.config, args.output, args.smoke)
