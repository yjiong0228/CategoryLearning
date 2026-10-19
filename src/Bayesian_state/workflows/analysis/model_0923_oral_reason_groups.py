"""Separate oral coding, catalogue coverage and inferred use for repeated reasons.

Use --config YAML --output NEW_DIRECTORY. Group exact wording together with the
actual choice, phase and parsing scope; identical wording can have different
catalogue coverage for different choices. Keep every report, including missing
and unresolved text. Inspect a declared rule's marginal presence, belief and
response share from all frozen filters. These descriptive means do not define
oral likelihoods or pathwise transition probabilities.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...evaluation.model_0923_h_validation import digest, load
from ...hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from ...hypothesis_space.rule_programs import compile_rule_program
from .model_0923_h_validation import write
from .model_0923_cohort_execution_oral_followthrough import verify_report

ROOT = Path(__file__).resolve().parents[4]
GROUP_KEYS = ['reading', 'model', 'arm', 'phase', 'text', 'choice', 'scope']


def summarize_groups(trials: pd.DataFrame) -> pd.DataFrame:
    """Keep choice and coding scope when describing repeated literal reports."""
    measures = ['functional_support', 'prior_functional_support', 'functional_conflict',
                'functional_undetermined', 'functional_availability_any_component',
                'diagnostic_rule_presence', 'diagnostic_rule_belief', 'diagnostic_rule_response_share']
    rows = []
    for key, group in trials.groupby(GROUP_KEYS, dropna=False, sort=False):
        coverage = group.functional_support_any_component.dropna().unique()
        if len(coverage) > 1:
            raise ValueError('One wording/choice group has inconsistent catalogue coverage')
        row = dict(zip(GROUP_KEYS, key), reports=len(group), correct=int(group.correct.sum()),
                   first_trial=int(group.ordinal_trial.min()), last_trial=int(group.ordinal_trial.max()),
                   catalogue_support_rules=float(coverage[0]) if len(coverage) else np.nan)
        for name in measures:
            row[name] = group[name].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def run(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral reason group directory')
    settings = yaml.safe_load(config.read_text())
    if set(settings) != {'subject', 'data', 'rule', 'alignments'} or settings['subject'] != 207:
        raise ValueError('Declare the 207 inputs and diagnostic rule explicitly')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); value = digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Bound reason-group input changed')
        inputs[str(path)] = value
        return path

    bind(config); bind(__file__)
    for function in (build_relational_0923_space, compile_rule_program, verify_report, digest, write):
        bind(inspect.getfile(function))
    folder = ROOT / settings['data'] / 'subject_207'
    prepared = load(bind(folder / 'prepared.json')); behavior = []
    for name in ('training.npz', 'sealed_test.npz'):
        path = bind(folder / name)
        if digest(path) != prepared['array_sha256'][name]:
            raise ValueError('Prepared behavior changed')
        with np.load(path) as z:
            behavior.append({k: z[k] for k in ('choices', 'categories', 'stimulus')})
    data = {k: np.concatenate([b[k] for b in behavior]) for k in behavior[0]}
    space = build_relational_0923_space(); rule = space.hypotheses[settings['rule']]
    if rule.family != 'dimension_max':
        raise ValueError('This diagnostic inspects the existing maximum-dimension rule')
    # Equal low values with a unique high feature give unambiguous category probes.
    probes = np.full((4, 4), .1); np.fill_diagonal(probes, .9)
    labels = compile_rule_program(rule).evaluate(probes) + 1
    geometry_labels = np.array([[any(np.all(part.A @ x <= part.b) for part in region.components)
                                 for region in rule.categories] for x in probes])
    np.testing.assert_array_equal(geometry_labels.sum(1), 1)
    np.testing.assert_array_equal(geometry_labels.argmax(1) + 1, labels)
    expected_signature = json.loads(json.dumps(space.signature))
    tables = []; cache = {}
    for item in settings['alignments']:
        source = ROOT / item['path']; manifest = verify_report(source)
        bind(source / 'manifest.json')
        for name in manifest['artifacts']:
            bind(source / name)
        for name in manifest['source_sha256']:
            bind(ROOT / name)
        if manifest['smoke'] or not manifest['full_replay_complete'] or manifest['subjects'] != [207]:
            raise ValueError('Require complete frozen 207 alignments')
        if (digest(bind(Path(manifest['source']) / 'protocol.json')) != manifest['source_protocol_sha256']
                or digest(bind(Path(manifest['oral_audit']) / 'manifest.json')) != manifest['oral_manifest_sha256']):
            raise ValueError('Original state or oral protocol changed')
        table = pd.read_csv(source / 'trials.csv', keep_default_na=False)
        for arm in manifest['arms']:
            selected = table[table.arm.eq(arm)].sort_values('state_trial').reset_index(drop=True).copy()
            np.testing.assert_array_equal(selected.state_trial, np.arange(prepared['trials']))
            np.testing.assert_array_equal(selected.choice, data['choices'])
            if not selected.training_trials.eq(prepared['training_trials']).all():
                raise ValueError('Original training boundary changed')
            np.testing.assert_array_equal(selected.phase,
                np.where(np.arange(prepared['trials']) < prepared['training_trials'], 'training', 'test'))
            states = Path(manifest['source']); key = (str(states), arm)
            if key not in cache:
                done = load(bind(states / 'completion.json')); protocol = load(bind(states / 'protocol.json'))
                if done['status'] != 'complete' or done['smoke'] or done['exact_behavior_replays'] != done['jobs']:
                    raise ValueError('Require all exact scientific state replays')
                for hashes in (protocol['source_sha256'], protocol['refit_replay_inputs']):
                    for name, expected in hashes.items():
                        if digest(bind(ROOT / name)) != expected:
                            raise ValueError('Original state provenance changed')
                records = [(f, load(f)) for f in sorted((states / 'states').glob('*.json'))]
                records = [(f, r) for f, r in records if r['arm'] == arm and r['subject'] == 207]
                if len(records) != protocol['settings']['seeds_per_arm'] or len({r['seeds'][0] for _,r in records}) != len(records):
                    raise ValueError('Incomplete or duplicate filter bank')
                probabilities = []; masses = []
                for file, record in records:
                    bind(file); array = bind(file.with_suffix('.npz'))
                    if (digest(array) != record['array_sha256'] or record['catalogue_signature'] != expected_signature
                            or manifest['state_arrays'][str(array)] != record['array_sha256']
                            or record['timing'] != 'after_choice_before_current_feedback'
                            or record['data_use'] != 'behavior_only; no_oral_input'):
                        raise ValueError('State identity or timing differs')
                    with np.load(array) as z:
                        np.testing.assert_array_equal(z['choices'], data['choices'])
                        p = z['choice_probability']; r = rule.index
                        probabilities.append(p)
                        masses.append(np.stack([p*z['filtered_active'][:, r], p*z['filtered_belief'][:, r],
                                                z['choice_component_joint_mass'][:, r].sum(-1)], axis=1))
                probabilities = np.asarray(probabilities); masses = np.asarray(masses)
                if (not np.isfinite(masses).all() or not np.isfinite(probabilities).all()
                        or np.any(masses < 0) or np.any(probabilities <= 0) or np.any(probabilities > 1)
                        or np.any(masses > probabilities[..., None] + 1e-12)):
                    raise ValueError('Invalid rule-state joint masses')
                cache[key] = masses.sum(0) / probabilities.sum(0)[:, None]
            for column in ('functional_support_any_component', 'functional_support', 'prior_functional_support',
                           'functional_conflict', 'functional_undetermined', 'functional_availability_any_component'):
                selected[column] = pd.to_numeric(selected[column], errors='raise' if selected[column].ne('').all() else 'coerce')
            selected['reading'] = item['reading']; selected['model'] = item['model']
            selected['correct'] = data['choices'] == data['categories']
            for i, name in enumerate(('presence', 'belief', 'response_share')):
                selected['diagnostic_rule_'+name] = cache[key][:, i]
            tables.append(selected)
    trials = pd.concat(tables, ignore_index=True); summary = summarize_groups(trials)
    if summary.reports.sum() != len(trials):
        raise ValueError('Reason grouping lost reports')
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError('Original input changed during reason grouping')
    output.mkdir(parents=True)
    trials.to_csv(output/'trials.csv', index=False); summary.to_csv(output/'groups.csv', index=False)
    write(output/'catalogue_probe.json', dict(rule=rule.index, family=rule.family, signature=space.signature,
        feature_order=trials.iloc[0][[f'feature{i}_name' for i in range(1,5)]].tolist(),
        unique_maximum_probes=probes.tolist(), categories=labels.tolist(),
        scope='Existing fixed category labels; no relabeling, permutations or new response model'))
    write(output/'completion.json', dict(status='complete', subject=207, trial_rows=len(trials), groups=len(summary),
        inputs_sha256=inputs, artifacts_sha256={p.name:digest(p) for p in output.iterdir()},
        scope='Descriptive exact wording and choice groups; unresolved reports kept; all completed filter repeats',
        original_coding_changed=False, parameters_changed=False, oral_used_for_fit=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.config, args.output)
