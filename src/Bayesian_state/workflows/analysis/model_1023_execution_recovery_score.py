"""Open synthetic truths after fitting, then score parameters and latent states.

Use --phase infer --prepared CASES --fit FROZEN_FITS --output NEW, followed by
--phase summarize --inference COMPLETED_INFERENCE --output NEW. Inference
replays two saved filters per fitted arm and evaluates the generating point
at the same independent seeds. The first two generating-point filters also
capture states, separating state observability from parameter estimation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import model_1023_execution_recovery as recovery
from . import model_1023_execution_validation as validation
from . import model_1023_execution_capture as capture
from .model_1022_belief_summary import nll, jackknife_se
from ...optimization.search.adaptive_proposals import candidate

study = recovery.study
read, write, digest = recovery.read, recovery.write, recovery.digest


def checked_arrays(file: Path, hash_key: str = 'predictions_sha256') -> dict:
    meta = read(file.with_suffix('.json'))
    if digest(file.with_suffix('.npz')) != meta[hash_key]:
        raise ValueError('Saved numerical arrays changed')
    with np.load(file.with_suffix('.npz')) as z:
        return {k: z[k].copy() for k in z.files}


def infer(prepared: Path, fit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new post-freeze inference destination')
    done, preparation = read(fit / 'completion.json'), read(prepared / 'completion.json')
    if done['status'] != 'complete' or preparation['status'] != 'complete':
        raise ValueError('All recovery estimates must freeze before opening truths')
    if (preparation['cases_sha256'] != digest(prepared / 'cases.json')
            or preparation['truths_sha256'] != digest(prepared / 'truths.json')):
        raise ValueError('Recovery definitions or generating truths changed')
    for identity, sha in done['nominees_sha256'].items():
        if digest(fit / identity / 'nominees.json') != sha:
            raise ValueError('Frozen estimates changed before truth scoring')
    truths = {r['case_id']: r for r in read(prepared / 'truths.json')['rows']}
    definitions = read(prepared / 'cases.json')['cases']
    definitions = [r for r in definitions if r['case_id'] in done['nominees_sha256']]
    output.mkdir(parents=True)
    jobs, cases = [], []
    for definition in definitions:
        file = Path(definition['file'])
        if digest(file) != definition['sha256']:
            raise ValueError('Case changed before truth scoring')
        case = read(file); identity = case['case_id']; own = fit / identity
        folder = Path(case['settings']['data']) / f'subject_{case["subject"]}'
        for name, sha in case['arrays_sha256'].items():
            if digest(folder / name) != sha:
                raise ValueError('Synthetic observations changed before truth scoring')
        truth = truths[identity]
        if digest(Path(truth['states'])) != truth['states_sha256']:
            raise ValueError('Latent generating path changed')
        frozen = read(own / 'nominees.json')
        if frozen['truths_opened'] or frozen['later_choices_used'] or frozen['status'] != 'frozen_before_test':
            raise ValueError('Require training-only frozen synthetic estimates')
        stage = 'confirmation' if done['smoke'] else 'evaluate'
        for row in frozen['nominees']:
            source_jobs = sorted((own / 'jobs').glob(
                f'{stage}_S{case["subject"]}_{row["arm"]}_{row["candidate"]["id"]}_*.json'))
            expected = 2 if done['smoke'] else case['settings']['budgets']['evaluate']['seeds']
            if len(source_jobs) != expected:
                raise ValueError('Incomplete original evaluation seed bank')
            for index, source_job in enumerate(source_jobs):
                original = read(source_job)
                if len(original['seeds']) != 1 or original['source_sha256'] != study.sources(study.driver.io):
                    raise ValueError('Original numerical implementation or seed chunk changed')
                if index < 2:
                    job = deepcopy(original)
                    name = identity + '_fitted_' + row['arm'] + f'_C{index:03d}'
                    reference = Path(original['target']).with_suffix('.npz')
                    job.update(name=name, recovery_case=identity, inference_role='fitted_' + row['arm'],
                        target=str((output / 'scores' / name).resolve()),
                        state_capture_target=str((output / 'states' / name).resolve()),
                        target_rules=case['target_rules'], capture_oral=False, capture_sources=capture.sources(),
                        original_score=str(reference.resolve()), original_score_sha256=digest(reference))
                    jobs.append(job)
                if row['arm'] == truth['mode']:
                    oracle = deepcopy(original)
                    name = identity + f'_oracle_C{index:03d}'
                    oracle.update(name=name, recovery_case=identity, inference_role='oracle',
                        candidate=candidate(truth['point'], 'generating_point_after_all_fits_freeze'),
                        target=str((output / 'scores' / name).resolve()))
                    if index < 2:
                        oracle.update(state_capture_target=str((output / 'states' / name).resolve()),
                            target_rules=case['target_rules'], capture_oral=False, capture_sources=capture.sources())
                    jobs.append(oracle)
                    equivalent = (truth['mode'] == 'MIX' or truth['mode'] == 'STICKY' and
                                  truth['point'][study.KAPPA] == 0)
                    if equivalent:
                        control = deepcopy(original)
                        control_mode = 'STICKY' if truth['mode'] == 'MIX' else 'MIX'
                        name = identity + f'_oracle_equivalent_C{index:03d}'
                        point = dict(truth['point'], **{study.KAPPA: 0.})
                        control.update(name=name, recovery_case=identity, inference_role='oracle_equivalent',
                            execution_mode=control_mode,
                            candidate=candidate(point, 'analytically_equivalent_boundary_control'),
                            target=str((output / 'scores' / name).resolve()))
                        jobs.append(control)
        cases.append(dict(definition=definition, truth=truth, nominees=frozen['nominees'],
            fit=str(own.resolve()), fit_nominees_sha256=digest(own / 'nominees.json')))
    write(output / 'protocol.json', dict(prepared=str(prepared.resolve()), fit=str(fit.resolve()),
        smoke=done['smoke'], cases=cases, jobs=jobs, source_sha256=digest(Path(__file__)),
        numerical_sources=recovery.sources(), truths_opened_after_all_fits_freeze=True,
        truth_sha256=digest(prepared / 'truths.json'), fit_completion_sha256=digest(fit / 'completion.json')))
    workers = 1 if done['smoke'] else min(128, len(os.sched_getaffinity(0)), len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        validation.NumericalQueue(pool).execute(jobs, output, workers)
    write(output / 'completion.json', dict(status='complete', smoke=done['smoke'], cases=len(cases), jobs=len(jobs),
        protocol_sha256=digest(output / 'protocol.json'), source_sha256=digest(Path(__file__))))


def binary_scores(probability: np.ndarray, target: np.ndarray) -> dict:
    """Brier, log score and rank AUC, with prevalence and degeneracy exposed."""
    p, y = np.asarray(probability, float), np.asarray(target, bool)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < -1e-12) | (p > 1 + 1e-12)):
        raise ValueError('Invalid binary state probability')
    p = np.clip(p, 0., 1.)
    positive = int(y.sum()); negative = len(y) - positive
    auc = ((rankdata(p)[y].sum() - positive*(positive+1)/2) / (positive*negative)
           if positive and negative else np.nan)
    selected = np.where(y, p, 1.-p)
    return dict(trials=len(y), positives=positive, prevalence=float(y.mean()),
        brier=float(np.square(p-y).mean()), prevalence_brier=float(y.mean()*(1-y.mean())),
        log_loss=float(-np.log(np.maximum(selected, 1e-12)).mean()), auc=float(auc),
        threshold_accuracy=float(((p >= .5) == y).mean()),
        false_positive_rate=float((p[~y] >= .5).mean()) if negative else np.nan,
        confident_false_positive_rate=float((p[~y] >= .8).mean()) if negative else np.nan,
        false_negative_rate=float((p[y] < .5).mean()) if positive else np.nan,
        near_zero_true_probability=float((selected <= 1e-12).mean()))


def pool_states(parts: list[dict]) -> dict:
    """Pool predictive states and current-choice joint masses across filters."""
    for part in parts[1:]:
        for key in ('choices', 'categories', 'target_rules'):
            np.testing.assert_array_equal(part[key], parts[0][key])
    if len({int(part['seeds'][0]) for part in parts}) != len(parts):
        raise ValueError('Duplicate state capture seeds')
    n = len(parts[0]['choices'])
    observation = np.stack([p['prediction'][np.arange(n), p['choices']-1] for p in parts])
    result = {k: parts[0][k] for k in ('choices', 'categories', 'target_rules')}
    for key in parts[0]:
        if key in result or key == 'seeds':
            continue
        values = np.stack([part[key] for part in parts])
        if key.startswith('filtered_') or key == 'conditional_decision_sources':
            weights = observation.reshape(observation.shape + (1,)*(values.ndim-2))
            result[key] = (values * weights).sum(axis=0) / weights.sum(axis=0)
        else:
            result[key] = values.mean(axis=0)
    return result


def truth_states(raw: dict, n: int, rules: int, targets: np.ndarray) -> dict:
    ids = raw['rule'][:n]
    t, slot = np.nonzero(ids >= 0)
    if np.any(ids[t, slot] >= rules):
        raise ValueError('Truth rule lies outside the captured catalogue')
    active = np.zeros((n, rules)); active[t, ids[t, slot]] = 1.
    if not np.array_equal(active.sum(axis=1), (ids >= 0).sum(axis=1)):
        raise ValueError('Duplicate active rule identity in truth')
    result = dict(active=active)
    for name in ('belief', 'contribution'):
        values = np.zeros_like(active)
        values[t, ids[t, slot]] = raw[name][:n][t, slot]
        np.testing.assert_allclose(values.sum(axis=1), 1., atol=1e-12)
        result['execution' if name == 'contribution' else name] = values
    result['present'] = np.any(active[:, targets] > 0, axis=1)
    result['target_belief'] = result['belief'][:, targets].sum(axis=1)
    result['high_belief'] = result['present'] & (result['target_belief'] >= .5)
    result['executed'] = raw['executed'][:n]
    result['target_execution'] = result['execution'][:, targets].sum(axis=1)
    result['available_other_execution'] = result['present'] * (1.-result['target_execution'])
    result['high_belief_other_execution'] = result['high_belief'] * (1.-result['target_execution'])
    return result


def state_metrics(state: dict, raw_truth: dict, true_mode: str) -> tuple[list, list]:
    n, rules = state['predictive_belief'].shape
    truth = truth_states(raw_truth, n, rules, state['target_rules'])
    rows, onsets = [], []
    for timing in ('predictive', 'filtered'):
        comparisons = [('present', 'target_present'), ('high_belief', 'target_high_belief')]
        if true_mode != 'MIX':
            comparisons.extend((name, name) for name in ('target_execution', 'available_other_execution',
                                                         'high_belief_other_execution'))
        for name, field in comparisons:
            target = truth[name].astype(bool)
            probability = state[timing + '_' + field]
            rows.append(dict(timing=timing, state=name, **binary_scores(probability, target)))
            true_indices, detected = np.flatnonzero(target), np.flatnonzero(probability >= .5)
            onsets.append(dict(timing=timing, state=name,
                true_first_trial=int(true_indices[0]+1) if len(true_indices) else None,
                first_detected_trial=int(detected[0]+1) if len(detected) else None,
                true_event_observed=bool(len(true_indices)), detection_observed=bool(len(detected)),
                definition='first marginal probability >= 0.5; detection, not a posterior first-passage time'))
        for name in ('belief', 'execution'):
            rows.append(dict(timing=timing, state=name+'_vector', trials=n,
                rmse=float(np.sqrt(np.square(state[timing+'_'+name]-truth[name]).mean())),
                mean_total_variation=float(np.abs(state[timing+'_'+name]-truth[name]).sum(axis=1).mean()/2)))
        if true_mode != 'MIX':
            execution = state[timing + '_execution']
            identity = truth['executed'].astype(int)
            if np.any(identity < 0):
                raise ValueError('Discrete execution truth requires explicit identities')
            selected = execution[np.arange(n), identity]
            rows.append(dict(timing=timing, state='exact_execution_identity', trials=n,
                accuracy=float((execution.argmax(axis=1) == identity).mean()),
                log_loss=float(-np.log(np.maximum(selected, 1e-12)).mean()),
                near_zero_true_probability=float((selected <= 1e-12).mean())))
    if true_mode != 'MIX':
        wrong = state['choices'] != state['categories']
        labels = np.where(~truth['present'], 0, np.where(truth['target_execution'] > .5, 2, 1))
        p = state['conditional_decision_sources'][wrong]
        y = labels[wrong]
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12)
        if len(y):
            rows.append(dict(timing='filtered', state='error_source', trials=len(y),
                accuracy=float((p.argmax(axis=1) == y).mean()),
                brier=float(np.square(p-np.eye(3)[y]).sum(axis=1).mean()),
                log_loss=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
                near_zero_true_probability=float((p[np.arange(len(y)), y] <= 1e-12).mean()),
                absent_errors=int((y == 0).sum()), available_other_errors=int((y == 1).sum()),
                target_execution_errors=int((y == 2).sum())))
    return rows, onsets


def summarize(inference: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery summary destination')
    done, protocol = read(inference / 'completion.json'), read(inference / 'protocol.json')
    if done['status'] != 'complete' or done['protocol_sha256'] != digest(inference / 'protocol.json'):
        raise ValueError('Post-freeze inference incomplete or changed')
    models, parameters, states, onsets, point_scores, null_checks, equivalence_checks = [], [], [], [], [], [], []
    for record in protocol['cases']:
        definition = Path(record['definition']['file'])
        if digest(definition) != record['definition']['sha256']:
            raise ValueError('Recovery case definition changed during reporting')
        case = read(definition)
        identity, truth = case['case_id'], record['truth']
        if (digest(Path(truth['states'])) != truth['states_sha256']
                or digest(Path(record['fit']) / 'nominees.json') != record['fit_nominees_sha256']):
            raise ValueError('Truth or parameter estimates changed during reporting')
        jobs = [j for j in protocol['jobs'] if j['recovery_case'] == identity]
        oracle_jobs = [j for j in jobs if j['inference_role'] == 'oracle']
        oracle_parts = [checked_arrays(Path(j['target'])) for j in oracle_jobs]
        oracle_probability = np.concatenate([p['observed_probability'] for p in oracle_parts])
        n = oracle_probability.shape[1]; cut = min(case['training_trials'], n)
        control_jobs = [j for j in jobs if j['inference_role'] == 'oracle_equivalent']
        if control_jobs:
            parts = [checked_arrays(Path(j['target'])) for j in control_jobs]
            control = np.concatenate([p['observed_probability'] for p in parts])
            np.testing.assert_array_equal(np.concatenate([p['seeds'] for p in parts]),
                                          np.concatenate([p['seeds'] for p in oracle_parts]))
            deleted = np.array([nll(np.delete(control, i, axis=0)) -
                nll(np.delete(oracle_probability, i, axis=0)) for i in range(len(control))])
            equivalence_checks.append(dict(case_id=identity, generating_model=truth['mode'],
                equivalent_model=control_jobs[0]['execution_mode'], particles=control_jobs[0]['particles'],
                seeds=len(control), control_minus_generating_nll=nll(control)-nll(oracle_probability),
                difference_mcse=jackknife_se(deleted), theoretical_difference=0.))
        stage = 'confirmation' if done['smoke'] else 'evaluate'
        true_class = ('MIX_EQUIVALENT' if truth['mode'] == 'MIX' or
                       truth['mode'] == 'STICKY' and truth['point'][study.KAPPA] == 0 else truth['mode'])
        for nominee in record['nominees']:
            mode = nominee['arm']
            score = checked_arrays(Path(record['fit']) / stage / f'subject_{case["subject"]}' /
                                   mode / nominee['candidate']['id'])
            p = score['observed_probability']
            k = sum(len(v) > 1 for v in case['spaces'][mode].values())
            models.append(dict(case_id=identity, subject=case['subject'], generating_model=truth['mode'],
                generating_kappa=truth['point'][study.KAPPA], behavioral_equivalence_class=true_class,
                fitted_model=mode, fitted_kappa=nominee['candidate']['hyperparams'][study.KAPPA],
                nominal_parameters=k, training_trials=cut, test_trials=n-cut,
                training_nll=nll(p[:, :cut]), test_nll=nll(p[:, cut:]) if n > cut else np.nan,
                training_bic=2*cut*nll(p[:, :cut])+k*np.log(cut)))
            if mode == truth['mode']:
                np.testing.assert_array_equal(score['seeds'], np.concatenate([p['seeds'] for p in oracle_parts]))
                difference = nll(p[:, :cut]) - nll(oracle_probability[:, :cut])
                deleted = np.array([nll(np.delete(p[:, :cut], i, axis=0)) -
                    nll(np.delete(oracle_probability[:, :cut], i, axis=0)) for i in range(len(p))])
                point_scores.append(dict(case_id=identity, model=mode, estimate_minus_truth_train_nll=difference,
                    difference_mcse=jackknife_se(deleted),
                    truth_train_nll=nll(oracle_probability[:, :cut]), estimate_train_nll=nll(p[:, :cut])))
                for coordinate, levels in case['spaces'][mode].items():
                    if len(levels) < 2:
                        continue
                    value, estimated = truth['point'][coordinate], nominee['candidate']['hyperparams'][coordinate]
                    inactive = (coordinate.endswith('search_controller.failure_decay') and
                           truth['point'][study.driver.H+'search_controller.error_gain'] == 0)
                    parameters.append(dict(case_id=identity, subject=case['subject'], model=mode,
                        parameter=coordinate, truth=value, estimate=estimated, known_zero_effect=inactive,
                        error=estimated-value, range_scaled_error=(estimated-value)/(max(levels)-min(levels)),
                        grid_steps_error=levels.index(estimated)-levels.index(value),
                        estimate_on_boundary=estimated in (min(levels), max(levels))))
        with np.load(truth['states']) as z:
            raw_truth = {k: z[k] for k in z.files}
        for role in sorted({j['inference_role'] for j in jobs}):
            captured = [checked_arrays(Path(j['state_capture_target']), 'array_sha256') for j in jobs
                        if j['inference_role'] == role and j.get('state_capture_target')]
            if not captured:
                continue
            pooled = pool_states(captured)
            metrics, first = state_metrics(pooled, raw_truth, truth['mode'])
            common = dict(case_id=identity, subject=case['subject'], generating_model=truth['mode'],
                generating_kappa=truth['point'][study.KAPPA], inference=role, filter_seeds=len(captured))
            states.extend(dict(common, **row) for row in metrics)
            onsets.extend(dict(common, **row) for row in first)
            if role == 'fitted_STICKY' and (truth['mode'] != 'STICKY' or truth['point'][study.KAPPA] == 0):
                # A high state probability in a model with persistence is not
                # evidence that persistence generated the observations.
                lag = pooled['filtered_high_belief_other_execution']
                null_checks.append(dict(common, trials=len(lag),
                    fitted_high_belief_other_mean=float(lag.mean()),
                    confident_lag_trials=int((lag >= .8).sum()),
                    confident_lag_fraction=float((lag >= .8).mean()),
                    null_definition='generated without a stay bonus'))
    output.mkdir(parents=True)
    for name, rows in [('model_scores', models), ('parameter_recovery', parameters), ('state_recovery', states),
                       ('onset_detection', onsets), ('estimate_vs_generating_point', point_scores),
                       ('equivalence_checks', equivalence_checks)]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)
    table = pd.DataFrame(models)
    criterion = 'training_nll' if done['smoke'] else 'test_nll'
    winners = table.loc[table.groupby('case_id')[criterion].idxmin()].copy()
    winners['selection_criterion'] = criterion
    winners.to_csv(output / 'model_selections.csv', index=False)
    null_table = pd.DataFrame(null_checks)
    if len(null_table):
        null_table = null_table.merge(winners[['case_id', 'fitted_model', 'fitted_kappa']],
                                     on='case_id', validate='one_to_one')
    null_table.to_csv(output / 'null_execution_attribution.csv', index=False)
    notes = ['# 执行机制恢复检验', '',
        '所有个体参数在打开生成参数前冻结。生成点与估计点使用相同的独立滤波种子评价。',
        '模型选择主要依据模拟序列后段的逐题选择似然；训练BIC保留为描述性补充。',
        'MIX与维持参数为0的STICKY具有相同的边缘行为分布。只有一条活跃规则的试次中，执行维持参数没有作用。',
        '预测状态使用过去选择；过滤状态另加入本题选择，二者均在本题反馈之前。',
        'oracle使用生成参数，fitted使用重新估计的参数，用于评价参数估计对状态推断的影响。',
        '两者均包含有限粒子误差；零维持边界的等价模型使用相同参数与种子检查数值偏差。',
        'MIX的execution列表示规则贡献权重。离散执行身份和三类实际错误来源仅对离散生成模型评分。',
        '阈值首次达到0.5的时点表示检测时点；真实事件未发生与未检出分别记录。',
        '未加入维持偏置的生成模型作为对照，另列STICKY给出的高置信度执行滞后归因及模型选择结果。',
        '状态对数评分使用$10^{-12}$的概率下限，并同时报告接近零的真状态概率比例。',
        '每个生成点只有两次重复，恢复比例用于初步覆盖检查。误差表需结合事件次数、基准误差与边界估计阅读。', '']
    if done['smoke']:
        notes.insert(2, '本目录为软件小规模验证，结果不用于科学结论。')
    (output / 'report.md').write_text('\n'.join(notes))
    write(output / 'completion.json', dict(status='complete', smoke=done['smoke'], cases=len(protocol['cases']),
        source_sha256=digest(Path(__file__)), inference_protocol_sha256=digest(inference / 'protocol.json'),
        artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    print(winners[['case_id', 'generating_model', 'generating_kappa', 'fitted_model', criterion]].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('infer', 'summarize'), required=True)
    for name in ('prepared', 'fit', 'inference', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.phase == 'infer' and args.prepared and args.fit and args.output:
        infer(args.prepared, args.fit, args.output)
    elif args.phase == 'summarize' and args.inference and args.output:
        summarize(args.inference, args.output)
    else:
        parser.error('Infer needs --prepared --fit --output; summarize needs --inference --output')
