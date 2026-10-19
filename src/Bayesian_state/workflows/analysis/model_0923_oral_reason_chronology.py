"""Describe frozen oral wording beside completed behavior-only state estimates.

Use --config YAML --output NEW. Exact text templates describe observed wording;
they do not define exhaustive oral events or estimate report probabilities.
Keep every report, original session/trial identity and fixed 64-trial blocks.
Compare the reference rule's marginal presence and response contribution using
all completed filter repeats. These are marginals, not cross-time transitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...evaluation.model_0923_h_validation import digest, load, jackknife_se
from .model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[4]


def label_templates(texts: pd.Series, templates: dict[str, int]) -> pd.Series:
    """Match exact frozen wording without changing qualifiers or unknown reports."""
    if not templates or any(not isinstance(k, str) or type(v) is not int for k, v in templates.items()):
        raise ValueError('Declare literal text and its explicit reference branch label')
    return texts.map(templates).astype('Int64')


def pooled_series(joint: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-trial joint masses before division; delete complete filters."""
    joint, probability = np.asarray(joint, float), np.asarray(probability, float)
    if (joint.shape != probability.shape or joint.ndim != 2 or len(joint) < 2
            or not np.isfinite(joint).all() or not np.isfinite(probability).all()
            or np.any(probability <= 0) or np.any(probability > 1)
            or np.any(joint < 0) or np.any(joint > probability + 1e-12)):
        raise ValueError('Need aligned valid joint masses and multiple full filters')
    numerator, denominator = joint.sum(axis=0), probability.sum(axis=0)
    return numerator / denominator, (numerator - joint) / (denominator - probability)


def run(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral chronology directory')
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'subject', 'data', 'oral_audit', 'rule_audit', 'states', 'templates', 'block_width'}
            or settings['block_width'] != 64 or not settings['states']):
        raise ValueError('Declare fixed wording, completed state sources and 64-trial blocks')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError('Chronology input changed')
        inputs[str(path)] = sha
        return path

    def read(path):
        return load(bind(path))

    bind(config); bind(__file__)
    subject = settings['subject']; folder = ROOT / settings['data'] / f'subject_{subject}'
    meta = read(folder / 'prepared.json'); rule_audit = read(ROOT / settings['rule_audit'])
    if meta['subject'] != subject or rule_audit['subject'] != subject:
        raise ValueError('Behavior or reference rule belongs to another participant')
    reference_rule = rule_audit['target']; observed = []
    for name in ('training.npz', 'sealed_test.npz'):
        file = bind(folder / name)
        if digest(file) != meta['array_sha256'][name]:
            raise ValueError('Prepared behavioral arrays changed')
        with np.load(file) as z:
            observed.append({k: z[k] for k in z.files})
    behavior = {k: np.concatenate([r[k] for r in observed]) for k in observed[0]}
    archived = pd.read_csv(bind(meta['paths']['learning_data']), keep_default_na=False)
    archived = archived[archived.iSub == subject].sort_values(['iSession', 'iTrial']).reset_index(drop=True)
    oral = ROOT / settings['oral_audit']; oral_meta = read(oral / 'manifest.json')
    if oral_meta['status'] != 'complete' or digest(bind(oral / 'trials.csv')) != oral_meta['artifacts']['trials.csv']:
        raise ValueError('Original oral audit changed')
    for path, sha in oral_meta['input_sha256'].items():
        if digest(bind(ROOT / path)) != sha:
            raise ValueError('Original oral source changed')
    trials = pd.read_csv(oral / 'trials.csv', keep_default_na=False)
    trials = trials[trials.iSub == subject].sort_values('ordinal_trial').reset_index(drop=True)
    for name in ('iSession', 'iTrial', 'choice', 'text', 'condition', *(f'feature{i}_name' for i in range(1, 5))):
        np.testing.assert_array_equal(trials[name], archived[name])
    for name, column in [('choices', 'choice'), ('categories', 'category'), ('feedback', 'feedback')]:
        np.testing.assert_array_equal(behavior[name], archived[column])
    np.testing.assert_array_equal(behavior['stimulus'], archived[[f'feature{i}' for i in range(1, 5)]])
    np.testing.assert_array_equal(trials.ordinal_trial, np.arange(1, meta['trials'] + 1))
    trials = trials[['iSub', 'iSession', 'iTrial', 'ordinal_trial', 'choice', 'text', 'scope']].copy()
    trials['iBlock'] = archived.iBlock.to_numpy()
    trials['category'] = behavior['categories']; trials['correct'] = behavior['choices'] == behavior['categories']
    trials['template_branch'] = label_templates(trials.text, settings['templates'])
    trials['template_matched'] = trials.template_branch.notna()
    trials['template_choice_agrees'] = trials.template_branch.eq(trials.choice)
    trials['nonempty'] = trials.text.str.len().gt(0)
    trials['contains_longest_word'] = trials.text.str.contains('最长', regex=False)
    trials['phase'] = np.where(trials.ordinal_trial <= meta['training_trials'], 'training', 'later')
    trials['block64'] = (trials.ordinal_trial - 1) // 64 + 1
    state_curves, state_summary, state_bindings = [], [], []
    for label, relative in settings['states'].items():
        source = ROOT / relative; protocol = read(source / 'protocol.json'); done = read(source / 'completion.json')
        if done['status'] != 'complete' or done['smoke'] or protocol['smoke'] or done['exact_behavior_replays'] != done['jobs']:
            raise ValueError('Need all completed scientific state replays')
        for path, sha in protocol['refit_replay_inputs'].items():
            if digest(bind(path)) != sha:
                raise ValueError('State replay provenance changed')
        for path, sha in protocol['source_sha256'].items():
            if digest(bind(ROOT / path)) != sha:
                raise ValueError('State collector source changed')
        records = [(file, read(file)) for file in sorted((source / 'states').glob('*.json'))]
        selected = [(f, r) for f, r in records if r['subject'] == subject]
        expected = {(r['arm'], r['seed']) for r in done['records'] if r['subject'] == subject}
        if {(r['arm'], r['seeds'][0]) for _, r in selected} != expected or len(selected) != len(expected):
            raise ValueError('State filter bank differs from its completed manifest')
        for arm in sorted({r['arm'] for _, r in selected}):
            group = sorted([(f, r) for f, r in selected if r['arm'] == arm], key=lambda item: item[1]['seeds'][0])
            if len(group) != protocol['settings']['seeds_per_arm']:
                raise ValueError('Incomplete filter bank')
            probabilities, active_joint, response_joint = [], [], []
            for file, record in group:
                job = read(source / 'jobs' / file.name)
                if (Path(job['folder']).resolve() != folder.resolve() or not record['exact_behavior_replay']
                        or record['timing'] != 'after_choice_before_current_feedback'
                        or record['data_use'] != 'behavior_only; no_oral_input'
                        or record['trials'] != len(trials) or record['training_trials'] != meta['training_trials']):
                    raise ValueError('State identity, timing or behavioral boundary differs')
                array = bind(file.with_suffix('.npz'))
                if digest(array) != record['array_sha256']:
                    raise ValueError('State arrays changed')
                with np.load(array) as z:
                    np.testing.assert_array_equal(z['choices'], behavior['choices'])
                    probability = z['choice_probability']; active = z['filtered_active'][:, reference_rule]
                    packed = z['active_packed'][:, :, reference_rule // 8]
                    indicator = ((packed >> (reference_rule % 8)) & 1).astype(float)
                    np.testing.assert_allclose((indicator * z['conditional_particle_weights']).sum(axis=1), active, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(z['choice_component_joint_mass'].sum(axis=(1, 2)), probability, atol=1e-12, rtol=0)
                    probabilities.append(probability); active_joint.append(probability * active)
                    response_joint.append(z['choice_component_joint_mass'][:, reference_rule].sum(axis=-1))
                state_bindings.append(dict(model=label, arm=arm, seed=record['seeds'][0], candidate=record['candidate']['id']))
            for metric, joint in [('reference_presence', active_joint), ('reference_response_share', response_joint)]:
                values, deleted = pooled_series(joint, probabilities)
                for i, value in enumerate(values):
                    state_curves.append(dict(model=label, arm=arm, ordinal_trial=i+1, metric=metric, value=value))
                for block, table in trials.groupby('block64'):
                    ix = table.index.to_numpy()
                    state_summary.append(dict(model=label, arm=arm, block64=int(block), metric=metric,
                        value=values[ix].mean(), mc_se=jackknife_se(deleted[:, ix].mean(axis=1)), trials=len(ix)))
    blocks = trials.groupby('block64').agg(first_trial=('ordinal_trial', 'min'), last_trial=('ordinal_trial', 'max'),
        trials=('ordinal_trial', 'size'), nonempty=('nonempty', 'sum'), template_reports=('template_matched', 'sum'),
        longest_word_reports=('contains_longest_word', 'sum'), correct_rate=('correct', 'mean')).reset_index()
    repeated = []
    for text, table in trials.groupby('text', sort=False):
        counts = table.choice.value_counts()
        repeated.append(dict(text=text, reports=len(table), first_trial=table.ordinal_trial.min(), last_trial=table.ordinal_trial.max(),
            correct=int(table.correct.sum()), incorrect=int((~table.correct).sum()),
            modal_choice_share=counts.iloc[0]/len(table), choices=json.dumps({int(k):int(v) for k,v in counts.items()})))
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Bound input changed during chronology analysis')
    output.mkdir(parents=True)
    for name, table in [('trials', trials), ('blocks', blocks), ('repeated_reports', pd.DataFrame(repeated)),
                        ('state_trials', pd.DataFrame(state_curves)), ('state_blocks', pd.DataFrame(state_summary)),
                        ('state_sources', pd.DataFrame(state_bindings))]:
        table.to_csv(output / (name + '.csv'), index=False)
    write(output / 'completion.json', dict(status='complete', subject=subject, reference_rule=reference_rule,
        reports=len(trials), inputs_sha256=inputs, artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')},
        template_scope='Exact observed wording; no exhaustive semantics, state-transition timing or oral likelihood inferred',
        state_scope='Behavior-only marginals after current choice and before current feedback; eight full-filter repeats per arm',
        interpretation='Development chronology; reporting changes and internal learning remain distinct hypotheses'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.config, args.output)
