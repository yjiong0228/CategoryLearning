"""Read original filters to compare internal states across frozen parameters.

Use --evaluation COMPLETE_PAIRED_EVALUATION --audit-summary HUMAN_AUDIT_SUMMARY
--output NEW. No oral content or new simulation enters this analysis. Seed-level
joint moments are pooled before conditioning on the current choice. Target-rule
precision conditions additionally on target presence. All three pre-audit points
per person remain visible with their independent behavior score differences.
Whole-filter jackknife errors quantify numerical resolution of state contrasts.
The finite candidate set does not define a parameter confidence region.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import jackknife_se
from .oral.decision_basis_scoring import pool_joint
from ..workflows.analysis.model_0923_decision_basis_alignment import validate_basis_record
from ..workflows.analysis.model_1023_estimation_reliability import read, write, digest
from ..optimization.search.compact_joint import behavior_key


def joint_state_bank(arrays: list[dict]) -> dict:
    """Convert within-filter conditional moments back to joint moments."""
    first = arrays[0]
    for own in arrays:
        for key in ('choices', 'component_counts'):
            np.testing.assert_array_equal(own[key], first[key])
    probability = np.stack([a['choice_probability'] for a in arrays])
    bank = dict(probability=probability,
        prior=np.stack([a['prior_component_mass'].sum(-1) for a in arrays]),
        basis=np.stack([a['choice_component_joint_mass'].sum(-1) for a in arrays]),
        search=np.stack([a['predictive_search'] for a in arrays]),
        admission=np.stack([a['predictive_admission'] for a in arrays]))
    for name, key in [('active', 'filtered_active'), ('belief', 'filtered_belief'), ('beta', 'filtered_active_beta')]:
        values = np.stack([a[key] for a in arrays])
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid nonnegative captured state moment')
        if name == 'active' and np.any(values > 1+1e-12):
            raise ValueError('Rule membership probability exceeds one')
        bank[name] = values*probability[:, :, None]
    np.testing.assert_allclose(bank['prior'].sum(-1), 1, atol=1e-12, rtol=0)
    for key in ('basis', 'belief'):
        np.testing.assert_allclose(bank[key].sum(-1), probability, atol=1e-12, rtol=0)
    for key in ('prior', 'basis', 'probability'):
        if not np.isfinite(bank[key]).all() or np.any(bank[key] < 0):
            raise ValueError('Invalid captured joint or predictive mass')
    if np.any(bank['beta'][bank['active'] == 0] != 0):
        raise ValueError('Absent rules carry a nonzero precision moment')
    return bank


def pooled_states(bank: dict, target: int) -> dict:
    """Pool rule distributions and conditional precision at fixed parameters."""
    probability = bank['probability']
    result = {name: pool_joint(bank[name], probability) for name in ('basis', 'active', 'belief', 'beta')}
    if not 0 <= target < result['basis'].shape[1]:
        raise ValueError('Target rule outside the captured catalogue')
    present = result['active'][:, target]
    if np.any(result['basis'] > result['active']+1e-12):
        raise ValueError('Choice-conditioned rule contribution exceeds membership')
    precision = np.divide(result['beta'][:, target], present,
                          out=np.full_like(present, np.nan), where=present > 0)
    result['trials'] = dict(target_present=present, target_contribution=result['basis'][:, target],
        target_predictive_contribution=bank['prior'].mean(0)[:, target],
        target_precision_given_presence=precision, expected_active_rules=result['active'].sum(-1),
        predictive_search=bank['search'].mean(0), predictive_admission=bank['admission'].mean(0))
    for name in ('basis', 'belief'):
        np.testing.assert_allclose(result[name].sum(-1), 1, atol=1e-12, rtol=0)
    return result


def compare_states(a: dict, b: dict) -> dict:
    """Separate rule-use distribution change from target-specific change."""
    aa, bb = a['trials'], b['trials']
    if a['basis'].shape != b['basis'].shape:
        raise ValueError('State comparisons require the same trial/rule identities')
    # Total variation is 0 for identical distributions and 1 for disjoint ones.
    result = dict(basis_total_variation=.5*np.abs(a['basis']-b['basis']).sum(-1),
        belief_total_variation=.5*np.abs(a['belief']-b['belief']).sum(-1),
        target_presence_delta=aa['target_present']-bb['target_present'],
        target_contribution_delta=aa['target_contribution']-bb['target_contribution'],
        target_precision_delta=aa['target_precision_given_presence']-bb['target_precision_given_presence'])
    for name in ('target_presence', 'target_contribution', 'target_precision'):
        result[name+'_absolute_change'] = np.abs(result[name+'_delta'])
    return result


def read_point(case: dict, bind) -> tuple[dict, dict, dict]:
    source = Path(case['evaluation'])
    protocol = read(bind(source/'protocol.json'))
    frozen = read(bind(Path(protocol['fit'])/'nominees.json', protocol['nomination_sha256']))
    point = frozen['nominee']['candidate']
    if behavior_key(point['hyperparams']) != behavior_key(case['candidate_hyperparams']):
        raise ValueError('Capture parameters differ from the frozen sensitivity point')
    behavior, = read(bind(source/'conditional/evaluation.json'))['rows']
    autonomous, = read(bind(source/'autonomous/evaluation.json'))['rows']
    if behavior['particles'] != 1024 or len(behavior['seeds']) != 8:
        raise ValueError('Require eight full-precision original filters')
    ordered = []
    for file in sorted((source/'state_capture').glob('*.json')):
        record = read(bind(file))
        if (record['candidate'] != point or record['particles'] != 1024 or len(record['seeds']) != 1
                or record['subject'] != case['subject'] or record['condition'] != case['condition']
                or record['trials'] != behavior['trials']):
            raise ValueError('Captured filter identity or budget changed')
        validate_basis_record(bind(file.with_suffix('.npz'), record['array_sha256']), record)
        bind(record['original_job'], record['original_job_sha256'])
        bind(record['score_file'], record['score_sha256'])
        with np.load(file.with_suffix('.npz')) as saved:
            needed = ('prior_component_mass', 'choice_component_joint_mass', 'choice_probability',
                      'filtered_active', 'filtered_belief', 'filtered_active_beta', 'predictive_search',
                      'predictive_admission', 'choices', 'component_counts')
            arrays = {k: saved[k].copy() for k in needed}
        ordered.append((record['seeds'][0], arrays, record['catalogue_signature']))
    ordered.sort(key=lambda item: item[0])
    seeds = np.array([r[0] for r in ordered])
    np.testing.assert_array_equal(seeds, sorted(case['reference_seeds']['conditional']))
    if len(ordered) != 8 or any(r[2] != ordered[0][2] for r in ordered):
        raise ValueError('Original filters disagree on seed count or catalogue')
    bank = joint_state_bank([r[1] for r in ordered])
    meta = dict(seeds=seeds, choices=ordered[0][1]['choices'], catalogue=ordered[0][2],
                target_rule=autonomous['target_rule'], training_trials=behavior['training_trials'])
    return bank, pooled_states(bank, meta['target_rule']), meta


def summarize(evaluation: Path, audit_summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new state-sensitivity directory')
    done, audit_done = read(evaluation/'completion.json'), read(audit_summary/'completion.json')
    if (done['status'] != 'complete' or done['smoke'] or audit_done['status'] != 'complete'
            or audit_done['smoke'] or audit_done['phase'] != 'human'):
        raise ValueError('Require completed scientific human fits and paired evaluations')
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('State-sensitivity input changed: '+str(path))
        inputs[str(path)] = value
        return path

    for file in (evaluation/'completion.json', audit_summary/'completion.json', Path(__file__),
                 Path(inspect.getfile(pool_joint)), Path(inspect.getfile(jackknife_se)),
                 Path(inspect.getfile(validate_basis_record)), Path(inspect.getfile(behavior_key))):
        bind(file)
    protocol = read(bind(evaluation/'protocol.json'))
    if Path(protocol['study']).resolve() != Path(audit_done['study']).resolve():
        raise ValueError('Behavior audit and state evaluation use different parameter fits')
    plan = read(bind(evaluation/'cases.json', done['cases_sha256']))
    audit = pd.read_csv(bind(audit_summary/'audit_candidates.csv', audit_done['artifacts_sha256']['audit_candidates.csv']))
    cases = plan['cases']
    if len(cases) != len(audit) or {(c['subject'], c['candidate']) for c in cases} != set(zip(audit.subject, audit.candidate)):
        raise ValueError('State sensitivity differs from the pre-audit candidate set')
    records, trial_differences, scores, files = [], [], [], []
    output.mkdir(parents=True)
    for subject in sorted(audit.subject.unique()):
        own = [c for c in cases if c['subject'] == subject]
        selected, = [c for c in own if 'selected' in c['roles']]
        banks = {c['candidate']: read_point(c, bind) for c in own}
        reference_bank, reference_state, identity = banks[selected['candidate']]
        length, cut = len(identity['choices']), identity['training_trials']
        windows = [('training', slice(0, cut)), ('test', slice(cut, length)), ('whole', slice(0, length))]
        for case in own:
            bank, state, meta = banks[case['candidate']]
            for key in ('seeds', 'choices'):
                np.testing.assert_array_equal(meta[key], identity[key])
            if any(meta[k] != identity[k] for k in ('catalogue', 'target_rule', 'training_trials')):
                raise ValueError('Candidates use different state coordinates or training boundaries')
            audited, = audit[(audit.subject == subject) & (audit.candidate == case['candidate'])].itertuples()
            tag = dict(subject=subject, cohort=case['cohort'], condition=case['condition'],
                candidate=case['candidate'], selected='selected' in case['roles'], reference='reference' in case['roles'],
                target_rule=meta['target_rule'], training_trials=cut, audit_delta_nll=audited.delta_to_selected,
                audit_delta_mc_se=audited.delta_mc_se, no_clear_loss_at_mc_resolution=audited.no_clear_loss_at_mc_resolution)
            records.extend(dict(tag, trial=t+1, phase='training' if t < cut else 'test',
                                **{k: v[t] for k, v in state['trials'].items()}) for t in range(length))
            file = output/(case['case_id']+'_rule_states.npz')
            np.savez_compressed(file, **{k: state[k] for k in ('basis', 'belief', 'active')},
                                choices=meta['choices'], seeds=meta['seeds'])
            files.append(file.name)
            contrast = compare_states(state, reference_state)
            trial_differences.extend(dict(tag, trial=t+1, **{k: v[t] for k, v in contrast.items()}) for t in range(length))
            replicates = []
            for seed in range(len(meta['seeds'])):
                a = pooled_states({k: np.delete(v, seed, axis=0) for k, v in bank.items()}, meta['target_rule'])
                b = pooled_states({k: np.delete(v, seed, axis=0) for k, v in reference_bank.items()}, meta['target_rule'])
                replicates.append(compare_states(a, b))
            for window, index in windows:
                row = dict(tag, window=window, trials=len(meta['choices'][index]), filters=len(meta['seeds']))
                for key, value in contrast.items():
                    valid = np.isfinite(value[index])
                    # Fix the scored trial set across jackknife replicates.
                    available = all(np.isfinite(r[key][index][valid]).all() for r in replicates)
                    row[key] = value[index][valid].mean() if valid.any() else np.nan
                    row[key+'_trials'] = int(valid.sum())
                    row[key+'_mc_se'] = (jackknife_se(np.array([r[key][index][valid].mean() for r in replicates]))
                                        if valid.any() and available else np.nan)
                scores.append(row)
    for name, rows in [('trials', records), ('trial_contrasts', trial_differences), ('contrasts', scores)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'), index=False)
    write(output/'completion.json', dict(status='complete', evaluation=str(evaluation.resolve()),
        audit_summary=str(audit_summary.resolve()), subjects=sorted(map(int, audit.subject.unique())),
        candidates=len(cases), additional_filters=0, oral_data_used=False, inputs_sha256=inputs,
        timing='Current-choice-conditioned state before current feedback; predictive search and contribution labelled separately',
        pooling='Pool seed-level joint moments before conditioning; precision additionally conditions on rule presence',
        scope='Finite prefix-selected candidate sensitivity; individual-state inference at fixed parameters',
        numerical_uncertainty='Paired delete-one-filter jackknife; undefined precision comparisons retained explicitly',
        artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.suffix in ('.csv', '.npz')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'audit-summary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.evaluation, args.audit_summary, args.output)
