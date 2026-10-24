"""Validate frozen execution decision bases with fixed external oral coding.

Use --capture COMPLETE_STATES --config YAML --output NEW [--smoke]. The
shared constraint encoder and midpoint/relative sensitivity procedures are
unchanged. Every complete report remains in the primary denominator, including
reports with no matching catalogue rule. Oral observations never fit behavior.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import model_0923_decision_basis_alignment as alignment
from . import model_0923_oral_constraint_audit as audit
from . import model_0923_oral_reference_sensitivity as sensitivity
from . import model_1023_execution_capture as capture_tools
from .model_1023_execution_study import ROOT, read, write, digest


def export_basis(source: Path, output: Path) -> dict:
    """Create a provenance-bound view for the shared oral scoring interface."""
    if output.exists():
        raise FileExistsError('Use a new oral-basis adapter destination')
    completed, protocol = read(source / 'completion.json'), read(source / 'protocol.json')
    if (completed['status'] != 'complete' or not completed['exact_behavior_replay']
            or completed['protocol_sha256'] != digest(source / 'protocol.json')):
        raise ValueError('Require complete exact behavioral replay with passive oral bases')
    jobs = protocol['jobs']
    subjects = sorted({read(Path(j['folder']) / 'prepared.json')['subject'] for j in jobs})
    arms = ['MIX', 'MAP', 'STICKY']
    expected = 1 if completed['smoke'] else 2
    output.mkdir(parents=True)
    (output / 'states').mkdir(); (output / 'jobs').mkdir()
    bindings, counts = {}, {}
    for job in jobs:
        target = Path(job['state_capture_target'])
        record = read(target.with_suffix('.json'))
        if (not record['capture_oral'] or not record['exact_saved_filter_replay']
                or not {'catalogue_version', 'catalogue_signature'} <= set(record)
                or record['capture_sources'] != capture_tools.sources()
                or record['array_sha256'] != digest(target.with_suffix('.npz'))
                or record['original_job_sha256'] != digest(Path(record['original_job']))):
            raise ValueError('Oral capture lost its numerical identity')
        reference = Path(job['original_score'])
        if digest(reference) != job['original_score_sha256']:
            raise ValueError('Original behavioral probabilities changed')
        with np.load(target.with_suffix('.npz')) as saved, np.load(reference) as original:
            basis = {key.removeprefix('oral_'): saved[key].copy() for key in saved.files if key.startswith('oral_')}
            basis['seeds'] = saved['seeds'].copy()
            index, = np.flatnonzero(original['seeds'] == basis['seeds'][0])
            probability = basis['prediction'][np.arange(len(basis['choices'])), basis['choices'] - 1]
            np.testing.assert_array_equal(probability, original['observed_probability'][index])
            np.testing.assert_allclose(basis['choice_probability'], probability, atol=1e-14, rtol=0)
            if 'probabilities' in original:
                np.testing.assert_array_equal(basis['prediction'], original['probabilities'][index])
                np.testing.assert_array_equal(basis['choices'], original['choices'])
            elif not completed['smoke']:
                raise ValueError('Scientific oral replay requires the full saved probability vectors')
        destination = output / 'states' / target.name
        with destination.with_suffix('.npz').open('xb') as stream:
            np.savez_compressed(stream, **basis)
        # This was one additional replay of a saved filter. Its observer adds
        # no random draws, and the exact saved predictions were checked above.
        converted = dict(record, arm=record['model'], exact_behavior_replay=True,
            exact_original_filter_capture=False, additional_filters=1,
            timing='after_choice_before_current_feedback',
            array_sha256=digest(destination.with_suffix('.npz')),
            source_capture=str(target.with_suffix('.json').resolve()),
            source_capture_sha256=digest(target.with_suffix('.json')))
        write(destination.with_suffix('.json'), converted)
        write(output / 'jobs' / (target.name + '.json'), job)
        key = record['subject'], record['model']
        counts.setdefault(key, []).append(record['seeds'][0])
        bindings[str(target.with_suffix('.json').resolve())] = digest(target.with_suffix('.json'))
        bindings[str(target.with_suffix('.npz').resolve())] = record['array_sha256']
    if (set(counts) != {(sid, mode) for sid in subjects for mode in arms}
            or any(len(seeds) != expected or len(set(seeds)) != expected for seeds in counts.values())):
        raise ValueError('Missing or duplicate oral capture filters')
    own = dict(settings=dict(subjects=subjects, arms=arms, seeds_per_arm=expected), smoke=completed['smoke'],
        capture=str(source.resolve()), input_sha256=bindings,
        parameter_estimation_scope='chronological_prefix_behavior_only',
        evaluation_use='developmental_execution_mechanism_comparison',
        source_sha256=digest(Path(__file__)), additional_replay_filters=len(jobs),
        oral_observations_used_for_inference=False)
    write(output / 'protocol.json', own)
    write(output / 'completion.json', dict(status='complete', protocol_sha256=digest(output / 'protocol.json')))
    return own


def audit_denominators(scored: Path, coding: Path) -> dict:
    """Independently reconstruct complete-report counts and catalogue coverage."""
    trials, summary = [pd.read_csv(scored / name) for name in ('trials.csv', 'summary.csv')]
    original, matches = [pd.read_csv(coding / name) for name in ('trials.csv', 'component_matches.csv')]
    checked = 0
    for (sid, mode), frame in trials.groupby(['iSub', 'arm']):
        frame = frame.sort_values('state_trial').reset_index(drop=True)
        original_case = original.loc[original.iSub.eq(sid)].sort_values(['iSession', 'iTrial']).iloc[:len(frame)]
        for key in ('iSession', 'iTrial', 'scope', 'query_id'):
            np.testing.assert_array_equal(frame[key].fillna('').to_numpy(), original_case[key].fillna('').to_numpy())
        ordinal = np.arange(len(frame)); cut = int(frame.training_trials.iloc[0])
        windows = dict(whole=np.ones(len(frame), bool), training=ordinal < cut, test=ordinal >= cut,
            first64=ordinal < 64, last64=ordinal >= max(0, len(frame)-64))
        for domain in ('functional', 'structural'):
            coverage = matches.loc[matches[domain].eq('support')].groupby('query_id').size()
            covered = frame.query_id.map(coverage).fillna(0).to_numpy() > 0
            mass = frame[[domain+'_'+label for label in ('support', 'conflict', 'undetermined')]].to_numpy()
            complete = frame.scope.eq('complete').to_numpy()
            if not np.isfinite(mass[complete]).all():
                raise ValueError('Complete oral report lost its probability decomposition')
            np.testing.assert_allclose(mass[complete].sum(axis=1), 1., atol=1e-10)
            for name, window in windows.items():
                for subset in ('complete', 'catalogue_supported'):
                    mask = complete & window & (covered if subset == 'catalogue_supported' else True)
                    row, = summary.loc[summary.subject.eq(sid) & summary.arm.eq(mode) &
                        summary.domain.eq(domain) & summary.window.eq(name) & summary.subset.eq(subset)].to_dict('records')
                    if row['reports'] != mask.sum() or row['full_reports'] != (complete & window).sum():
                        raise ValueError('Oral denominator changed')
                    if mask.any():
                        np.testing.assert_allclose([row[k] for k in ('support', 'conflict', 'undetermined')],
                                                   mass[mask].mean(axis=0), atol=1e-10)
                    checked += 1
    return dict(status='complete', checked_groups=checked,
                checks='trial identity, complete denominators, fixed catalogue coverage, normalized mass and means')


def run(source: Path, config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new external-validation destination')
    policy = yaml.safe_load(config.read_text())
    if policy['oral_readings'] != ['midpoint', 'relative'] or policy['reference_model'] != 'MIX':
        raise ValueError('Require the registered readings and model reference')
    if read(source / 'completion.json')['smoke'] != smoke:
        raise ValueError('Capture and oral-validation scopes differ')
    output.mkdir(parents=True)
    exported = export_basis(source, output / 'basis')
    oral_policy = yaml.safe_load((ROOT / policy['oral_base_config']).read_text())
    oral_policy['subjects'] = exported['settings']['subjects']
    workers = 1 if smoke else min(policy['parallel_budget'], len(os.sched_getaffinity(0)))
    oral_policy['workers'] = workers
    own_config = output / 'oral_config.yaml'
    own_config.write_text(yaml.safe_dump(oral_policy, allow_unicode=True, sort_keys=False))
    audit.run(own_config, output / 'coding_midpoint', workers)
    sensitivity.run(output / 'coding_midpoint', output / 'coding_relative', workers)
    validation = {}
    for reading in policy['oral_readings']:
        coding, scored = output / ('coding_'+reading), output / ('alignment_'+reading)
        alignment.run(output / 'basis', coding, scored, reference_model=policy['reference_model'])
        validation[reading] = audit_denominators(scored, coding)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, policy=policy,
        source_sha256=digest(Path(__file__)), capture_protocol_sha256=digest(source / 'protocol.json'),
        oral_used_for_parameter_estimation=False, oral_used_for_state_inference=False,
        meaning='external content compatibility; not probability of generating an oral report',
        validation=validation, denominator='all complete reports, with separate fixed catalogue coverage',
        filter_seeds_per_arm=exported['settings']['seeds_per_arm'],
        source_manifests={reading: digest(output / ('alignment_'+reading) / 'manifest.json')
                          for reading in policy['oral_readings']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'config', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.capture, args.config, args.output, args.smoke)
