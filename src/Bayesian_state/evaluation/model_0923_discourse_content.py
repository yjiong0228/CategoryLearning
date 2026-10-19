"""Align frozen107 content with original active candidates and decision bases.

Use --encoding TEXT_AUDIT --annotations ORIGINAL107_TEXT --capture CAPTURE
(repeatable) --output NEW_DIRECTORY [--workers 4]. Whole content and individual
predicates are separate targets. Candidate bounds concern the observed category;
exact response-use mass retains the saved particle/rule/component joint state.
All trial identities, original/new/fragment scopes and both fixed readings stay
visible. This is a descriptive development diagnostic, not an oral likelihood.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..hypothesis_space.spaces import continuous
from ..workflows.analysis import model_0923_oral_constraint_audit as audit
from ..workflows.analysis.model_0923_target_state_recovery import digest, read, verify, write
from .model_0923_rnn_discourse import verify_trial_identity
from .oral import candidate_cues, content_support, constraint_alignment

FEATURES = ('neck', 'tail', 'head', 'leg')


def content_targets(report: dict) -> list[dict]:
    """Choose complete conjunctions and explicit predicates before scoring."""
    origin = ('original_complete' if report['original']['status'] == 'coded' else
              'new_complete' if report['content_complete'] else 'parsed_fragment')
    readings = [('midpoint', report['content']['tokens'])]
    if report['relative_reason'] == 'eligible':
        readings.append(('relative', report['relative_tokens']))
    targets = []
    for reading, tokens in readings:
        if report['content_complete']:
            targets.append(dict(origin=origin, reading=reading, target='whole_content',
                                predicate=-1, tokens=list(tokens)))
        for index, token in enumerate(tokens):
            targets.append(dict(origin=origin, reading=reading, target='predicate',
                                predicate=index, tokens=[token]))
    return targets


def run(encoding: Path, annotations: Path, captures: list[Path], output: Path, requested_workers: int) -> None:
    if output.exists(): raise FileExistsError('Use a new content-alignment directory')
    if requested_workers < 1 or not captures: raise ValueError('Need positive workers and original captures')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha: raise ValueError('Input changed')
        inputs[str(path)] = sha
        return path

    def record(path): return read(bind(path))

    for file in (__file__, content_support.__file__, constraint_alignment.__file__,
                 candidate_cues.__file__, inspect.getfile(verify_trial_identity),
                 audit.__file__, continuous.__file__): bind(file)
    coded = record(encoding/'manifest.json')
    if (coded['status'] != 'complete' or coded['sample_role'] != 'development'
            or coded['model_states_read'] or coded['responses_read'] or coded['stimulus_values_read']):
        raise ValueError('Require independently frozen development text coding')
    verify(coded['input_sha256']); inputs.update(coded['input_sha256'])
    for name, sha in coded['source_sha256'].items():
        if digest(bind(audit.ROOT/name)) != sha: raise ValueError('Frozen encoder source changed')
    for name, sha in coded['artifacts'].items():
        if digest(bind(encoding/name)) != sha: raise ValueError('Frozen encoding changed')
    reports = [json.loads(line) for line in (encoding/'reports.jsonl').read_text().splitlines()]
    reports = sorted((r for r in reports if r['iSub'] == 107), key=lambda r: r['ordinal_trial'])
    freeze = record(annotations/'text_annotation_freeze.json')
    if freeze['status'] != 'complete' or freeze['model_states_read'] or tuple(freeze['feature_order']) != FEATURES:
        raise ValueError('Require original model-independent107 annotations')
    verify(freeze['inputs_sha256']); inputs.update(freeze['inputs_sha256'])
    text_file = bind(annotations/'text_annotations.csv')
    if digest(text_file) != freeze['text_sha256']: raise ValueError('Original annotations changed')
    frame = pd.read_csv(text_file, keep_default_na=False).sort_values('state_trial').reset_index(drop=True)
    if len(reports) != len(frame) or len(reports) != 192: raise ValueError('Incomplete107 report coverage')
    targets, queries, coverage = [], {}, []
    for t, (report, row) in enumerate(zip(reports, frame.to_dict('records'))):
        if (report['ordinal_trial'] != t+1 or report['iSession'] != row['iSession']
                or report['iTrial'] != row['iTrial'] or report['text'] != row['text']
                or tuple(report[f'feature{k}_name'] for k in range(1, 5)) != FEATURES
                or report['role_status'] != 'unassigned'):
            raise ValueError('Text identity, feature order or role assignment differs')
        identity = dict(subject=107, state_trial=t, iSession=row['iSession'], iTrial=row['iTrial'],
            phase=row['phase'], choice=row['choice'], text=row['text'],
            original_status=report['original']['status'], content_complete=report['content_complete'],
            has_contrast=bool(report['contrast_markers']))
        selected = content_targets(report)
        coverage.append(dict(identity, target_rows=len(selected),
                             unresolved=json.dumps(report['content']['unparsed'], ensure_ascii=False)))
        for target in selected:
            payload = dict(condition=1, choice=int(row['choice']), tokens=target['tokens'])
            query = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
            queries[query] = dict(payload, query_id=query, tolerance=1e-9)
            targets.append(dict(identity, **target, query_id=query))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    workers = min(requested_workers, 128, len(os.sched_getaffinity(0)), len(queries))
    print(f'{len(targets)} content targets; {len(queries)} queries; {workers} workers', flush=True)
    if workers == 1:
        matches = list(map(audit.audit_query, queries.values()))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            matches = list(pool.map(audit.audit_query, queries.values(), chunksize=1))
    space = continuous.build_continuous_hypothesis_space(4, 2)
    max_components = max(len(c.components) for r in space for c in r.categories)
    query_tables, component_rows = {}, []
    for summary, rows in matches:
        tables = {domain: np.full((len(space), max_components), -1, int) for domain in ('structural', 'functional')}
        for row in rows:
            for domain in tables: tables[domain][row['rule'], row['component']] = content_support.LABELS.index(row[domain])
        query_tables[summary['query_id']] = tables
        component_rows.extend(rows)
    scores, arms, max_error = [], [], 0.
    for capture in captures:
        done = record(capture/'completion.json')
        if done['status'] != 'complete' or done['parameters_changed'] or done['oral_used']:
            raise ValueError('Require unchanged report-independent captures')
        verify(done['inputs_sha256']); inputs.update(done['inputs_sha256'])
        jobs = list((capture/'jobs').glob('*.json'))
        if len(jobs) != 1: raise ValueError('Require one original preselected filter per capture')
        job = record(jobs[0]); target = Path(job['target']); state = record(target.with_suffix('.json'))
        proof = record(target.with_suffix('.particle_basis.json'))
        engine = record(Path(job['folder'])/'engine.json')
        if (proof != done['worker'] or proof['schema_version'] != 1
                or state['subject'] != 107 or state['condition'] != 1
                or state['trials'] != 192 or state['training_trials'] != 128 or len(state['seeds']) != 1
                or not state['exact_behavior_replay'] or state['arm'] in arms
                or state['catalogue_signature'] != json.loads(json.dumps(space.signature))
                or engine['provenance']['label_mapping']['mode'] != 'fixed_task_labels'):
            raise ValueError('Original scientific identity or catalogue differs')
        file = bind(target.with_suffix('.npz'))
        if digest(file) != state['array_sha256'] or digest(file) != proof['array_sha256']:
            raise ValueError('Original captured arrays changed')
        with np.load(file) as z: data = {k:z[k] for k in z.files}
        verify_trial_identity(frame, data['choices'], state['training_trials'])
        arms.append(state['arm'])
        for row in targets:
            t, choice = row['state_trial'], int(row['choice'])
            counts = np.array([len(r.categories[choice-1].components) for r in space])
            np.testing.assert_array_equal(counts, data['component_counts'][:, choice-1])
            for domain, labels in query_tables[row['query_id']].items():
                values = content_support.content_support_profile(
                    data['particle_basis_rule_ids'][t], labels, counts,
                    data['particle_basis_category_probabilities'][t], data['particle_basis_active_belief'][t],
                    data['particle_basis_choice_weights'][t], data['particle_basis_component_weights'][t],
                    data['particle_basis_prior_weights'][t], choice)
                error = abs(values['choice_probability']-data['choice_probability'][t])
                max_error = max(max_error, error)
                if error > 1e-12: raise ValueError('Original choice probability failed reconstruction')
                for weighting, measures in values['scores'].items():
                    mass = data['prior_component_mass'][t] if weighting == 'prior' else (
                        data['choice_component_joint_mass'][t]/data['choice_probability'][t])
                    for number, label in enumerate(content_support.LABELS):
                        expected = float(mass[labels == number].sum())
                        np.testing.assert_allclose(measures[label+'_decision_mass'], expected, atol=1e-12, rtol=0)
                    scores.append(dict(row, tokens=json.dumps(row['tokens']), arm=state['arm'],
                                       domain=domain, weighting=weighting, **measures))
        print(f'{state["arm"]}: all192 trials and {len(targets)} targets checked', flush=True)
    result = pd.DataFrame(scores); keys = ['arm', 'domain', 'weighting', 'phase', 'origin', 'reading', 'target']
    measures = [name for name in result if name.endswith(('_lower', '_upper', '_decision_mass'))]
    summary = result.groupby(keys).agg(reports=('state_trial', 'nunique'), targets=('state_trial', 'size'),
        **{name:(name, 'mean') for name in measures}).reset_index()
    # Per-predicate means give longer reports more rows. Save equal-report
    # means separately so report length cannot silently determine comparisons.
    report_means = result.groupby(keys+['state_trial'])[measures].mean().reset_index()
    equal_report = report_means.groupby(keys).agg(reports=('state_trial', 'size'),
        **{name:(name, 'mean') for name in measures}).reset_index()
    output.mkdir(parents=True)
    target_table = pd.DataFrame(targets)
    target_table['tokens'] = target_table.tokens.map(json.dumps)
    for name, table in [('coverage', pd.DataFrame(coverage)), ('targets', target_table),
        ('component_matches', pd.DataFrame(component_rows)), ('scores', result), ('summary', summary),
        ('equal_report_summary', equal_report)]: table.to_csv(output/(name+'.csv'), index=False)
    verify(inputs)
    write(output/'completion.json', dict(status='complete', subject=107, trials=192, training_trials=128,
        arms=arms, targets=len(targets), queries=len(queries), score_rows=len(result), workers=workers,
        inputs_sha256=inputs, artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')},
        maximum_choice_reconstruction_error=max_error, original_decision_masses_reconstructed=True,
        parameters_changed=False, oral_used_for_fit=False, original_coding_changed=False,
        role_assignment='unassigned', candidate_scope='observed-response category component bounds',
        bounds='minimum/maximum over unspecified local components, followed by original particle weighting',
        decision_scope='exact saved particle/rule/component joint; current response conditions once',
        scope='Development107, one preselected filter per arm; original/new/fragment content scopes separate'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('encoding', 'annotations', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True, action='append')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.encoding, args.annotations, args.capture, args.output, args.workers)
