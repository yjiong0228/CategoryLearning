"""Evaluate frozen107 text groups against an original full decision probe bank.

Use --annotations COMPLETED_TEXT_AUDIT --sweep ORIGINAL_FULL_SWEEP --output NEW.
All text rows and original parsing status remain visible. Named/collectively
mentioned features and literal names before/after contrast words are separate
diagnostic groups. Empty groups stay in coverage but have no effect score.
No text group assigns a positive or negative causal role. This entry point has
been validated for the frozen107 coordinate order and one original filter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from ...evaluation.model_0923_h_validation import digest, load
from ...evaluation.oral import discourse_effects, joint_relevance
from .model_0923_h_validation import write

FEATURES = ('neck', 'tail', 'head', 'leg')
MEASURES = ('actual_effect', 'positive_context_effect',
            'negative_context_effect', 'absolute_context_effect')


def run(annotations: Path, sweep: Path, output: Path) -> None:
    """Preserve report coverage while evaluating declared feature groups."""
    if output.exists():
        raise FileExistsError('Use a new discourse-effects output directory')
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError('Discourse diagnostic input changed')
        inputs[str(path)] = sha
        return path

    def read(path):
        return load(bind(path))

    for file in (__file__, discourse_effects.__file__, joint_relevance.__file__):
        bind(file)
    audit = read(annotations / 'completion.json')
    freeze = read(annotations / 'text_annotation_freeze.json')
    original = read(sweep / 'completion.json')
    if (audit['status'] != 'complete' or freeze['status'] != 'complete'
            or original['status'] != 'complete' or original['short_diagnostic']
            or original['parameters_changed'] or original['oral_used']
            or len(original['seeds']) != 1 or freeze['model_states_read']
            or tuple(freeze['feature_order']) != FEATURES
            or audit['subject'] != 107 or original['subject'] != 107):
        raise ValueError('Require frozen107 text and one full original-state sweep')
    text_file = bind(annotations / 'text_annotations.csv')
    if (digest(text_file) != freeze['text_sha256']
            or digest(text_file) != audit['text_first_annotation_sha256']
            or digest(text_file) != audit['artifacts_sha256']['text_annotations.csv']):
        raise ValueError('Frozen text annotations changed')
    for source in (freeze, original):
        for path, sha in source['inputs_sha256'].items():
            if digest(bind(path)) != sha:
                raise ValueError('Original text or decision-state source changed')
    file = bind(sweep / 'particle_effects.npz')
    if digest(file) != original['outputs_sha256']['particle_effects.npz']:
        raise ValueError('Original reflection probe bank changed')
    with np.load(file) as values:
        data = {key: values[key] for key in values.files}
    frame = pd.read_csv(text_file, keep_default_na=False).sort_values('state_trial').reset_index(drop=True)
    n, cut, particles = (original[key] for key in ('trials', 'training_trials', 'particles'))
    if (n != 192 or cut != 128 or len(frame) != n or frame.subject.nunique() != 1
            or frame.subject.iloc[0] != 107 or frame.state_trial.tolist() != list(range(n))
            or frame.duplicated(['iSession', 'iTrial']).any()):
        raise ValueError('Text and decision-state trial identities differ')
    np.testing.assert_array_equal(data['choices'], frame.choice)
    np.testing.assert_array_equal(data['seeds'], original['seeds'])
    if (data['joint_reflection_probability'].shape != (n, particles, 16)
            or data['reflection_masks'].shape != (16, 4)):
        raise ValueError('Unexpected full probe bank shape')
    for key in ('literal_mask', 'named_or_collective_mask', 'encoded_mask', 'before_mask', 'after_mask'):
        values = frame[key].to_numpy()
        if (not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values > 15).any()):
            raise ValueError('Text masks must index the same four features')
    if (frame.has_contrast.dtype != bool or frame.nonempty.dtype != bool
            or not np.array_equal(frame.phase, np.where(np.arange(n) < cut, 'training', 'later'))):
        raise ValueError('Text contrast flags or chronological boundaries differ')
    for key in ('prior_weights', 'conditional_weights'):
        values = data[key]
        if values.shape != (n, particles) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid original particle weights')
        np.testing.assert_allclose(values.sum(1), 1., atol=1e-12, rtol=0)

    groups, features, coverage = [], [], []
    for t, row in frame.iterrows():
        definitions = dict(
            all_named_or_collective=int(row.named_or_collective_mask),
            literal_contrast_before=int(row.before_mask) if row.has_contrast else 0,
            literal_contrast_after=int(row.after_mask) if row.has_contrast else 0,
            original_complete_encoded=int(row.encoded_mask) if row.original_scope == 'complete' else 0,
        )
        active = {}
        for name, mask in definitions.items():
            available = bool(mask and row.nonempty)
            coverage.append(dict(subject=107,arm=original['arm'],state_trial=t,iSession=int(row.iSession),
                iTrial=int(row.iTrial),phase=row.phase,text=row.text,original_scope=row.original_scope,
                has_contrast=bool(row.has_contrast),group=name,feature_mask=mask,available=available,
                note='declared_named_features' if available else 'no_named_features_in_this_group'))
            if available:
                active[name] = ((mask >> np.arange(4)) & 1).astype(bool)
        if not active:
            continue
        for weighting, key in (('prior', 'prior_weights'), ('choice_conditioned', 'conditional_weights')):
            result = discourse_effects.group_effect_profiles(
                data['joint_reflection_probability'][t], data['reflection_masks'], active, data[key][t])
            for name, value in result.items():
                identity = dict(subject=107,arm=original['arm'],state_trial=t,iSession=int(row.iSession),
                    iTrial=int(row.iTrial),phase=row.phase,choice=int(row.choice),text=row.text,
                    original_scope=row.original_scope,group=name,feature_mask=definitions[name],weighting=weighting)
                groups.append(dict(identity, features=len(value['feature_indices']), **value['means']))
                for column, index in enumerate(value['feature_indices']):
                    features.append(dict(identity, feature=FEATURES[index],
                        **{measure: float(value['feature_means'][measure][column]) for measure in MEASURES}))
    table = pd.DataFrame(groups)
    feature_table = pd.DataFrame(features)
    if table.empty:
        raise ValueError('No nonempty named-feature groups are available')
    group_keys = ['subject', 'arm', 'phase', 'group', 'weighting']
    measures = list(next(iter(result.values()))['means'])
    output.mkdir(parents=True)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    table.to_csv(output / 'group_trials.csv', index=False)
    feature_table.to_csv(output / 'feature_trials.csv', index=False)
    for suffix, keys in (('', group_keys), ('_by_mask', group_keys + ['feature_mask'])):
        aggregation = dict(reports=('state_trial', 'size'))
        aggregation.update({name: (name, 'mean') for name in measures})
        table.groupby(keys).agg(**aggregation).reset_index().to_csv(output / f'summary{suffix}.csv', index=False)
    feature_table.groupby(group_keys + ['feature']).agg(
        reports=('state_trial', 'size'), **{name: (name, 'mean') for name in MEASURES}
    ).reset_index().to_csv(output / 'feature_summary.csv', index=False)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Discourse diagnostic input changed during evaluation')
    write(output / 'completion.json', dict(status='complete',subject=107,arm=original['arm'],
        trials=n,training_trials=cut,particles=particles,seeds=original['seeds'],
        inputs_sha256=inputs,artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')},
        annotation_source=str(annotations.resolve()),sweep_source=str(sweep.resolve()),
        positive_direction='Actual feature value raises observed-choice probability relative to its reflection',
        groups='Named feature sets; contrast groups include literal names in their declared text spans',
        context_scope='Common all-feature reflection contexts; minima within particles before integration',
        role_assignment='Text position does not assign supporting/opposing causal roles',
        scope='One original full filter per supplied mechanism; descriptive development diagnostic',
        original_coding_changed=False,oral_used_for_fit=False,
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotations', 'sweep', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.annotations, args.sweep, args.output)
