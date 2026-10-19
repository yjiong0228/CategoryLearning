"""Cross frozen RETAIN/ONLINE parameter points with the existing belief workers.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Four cells cross the two
behavioral nominees with two belief mechanisms; a fifth evaluates the declared
near-tied ONLINE challenger. Reuse completed nominee jobs and compatible ONLINE
probe paths. All full cells inherit the original evaluation seeds, particles,
path count and task boundary. No parameter selection or oral scoring occurs.
The smoke executes one original seed per role/cell, 8 filter trials and 16 own-
choice trials, on one worker, and verifies available saved prefixes exactly.
Outputs: protocol.json, jobs/, banks.json, scores.csv, contrasts.csv and
completion.json. Original numerical workers and scientific results are retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
ONLINE = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_refit.py'
SEMANTIC_FIELDS = ('phase', 'folder', 'model', 'mechanism', 'runtime', 'particles',
                   'arm_settings', 'structural_intervention', 'retain_precision')


def helpers():
    spec = importlib.util.spec_from_file_location('parameter_cross_online', ROOT / ONLINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compatible_probe(template: dict, probe: dict, candidate: dict) -> bool:
    """Reuse an old ONLINE path only at identical cognitive inputs and seeds."""
    return (all(template.get(k) == probe.get(k) for k in SEMANTIC_FIELDS)
            and template['seeds'] == probe['seeds'] and not probe['smoke']
            and probe.get('online_belief') is True
            and probe['candidate']['id'] == candidate['id']
            and probe['candidate']['hyperparams'] == candidate['hyperparams'])


def worker(file: Path) -> None:
    job = json.loads(file.read_text())
    mechanism = job['cross_mechanism']
    if mechanism not in ('RETAIN', 'ONLINE') or bool(job.get('online_belief')) != (mechanism == 'ONLINE'):
        raise ValueError('Cross cell and actual belief mechanism disagree')
    online = helpers()
    (online.worker if mechanism == 'ONLINE' else online.helpers().worker)(file)


def check_chunk(job: dict, mechanism: str, v, bind) -> tuple[dict, dict]:
    target = Path(job['target'])
    row = v.read(bind(target.with_suffix('.json')))
    precision = v.read(bind(target.with_suffix('.precision.json')))
    bind(target.with_suffix('.experiment.json'))
    automatic = job['phase'] == 'autonomous'
    copies = len(job['seeds']) * (1 if automatic else job['particles'])
    if (row['candidate'] != job['candidate'] or row['model'] != job['model']
            or not job['retain_precision'] or not precision['retained_precision']
            or precision['transitions'] != copies * row['trials']
            or row.get('paths' if automatic else 'seeds') != (len(job['seeds']) if automatic else job['seeds'])
            or any(pool['num_threads'] != 1 for pool in row['numeric_pools'])):
        raise ValueError('Executed cross chunk differs from declared parameter point or budget')
    if mechanism == 'ONLINE':
        execution = v.read(bind(target.with_suffix('.online.json')))
        timing = execution.get('exact_transition_timing', execution.get('completed_history_timing_exact'))
        if (not job.get('online_belief') or not execution['enabled'] or timing is not True
                or not execution['evidence_matches_retained_state']
                or execution['reconstructions'] != copies * (row['trials'] - 1)):
            raise ValueError('Missing or changed ONLINE feedback-evidence updates')
    elif job.get('online_belief'):
        raise ValueError('RETAIN cell executes ONLINE')
    path = bind(target.with_suffix('.npz'))
    if v.digest(path) != row['predictions_sha256']:
        raise ValueError('Executed cross arrays changed')
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
    p = arrays['probabilities']
    if p.shape[:2] != (len(job['seeds']), row['trials']) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid probability shape or values')
    np.testing.assert_allclose(p.sum(-1), 1., rtol=0, atol=1e-12)
    if not automatic:
        observed = np.take_along_axis(p, (arrays['choices'] - 1)[None, :, None], axis=2)[..., 0]
        np.testing.assert_array_equal(observed, arrays['observed_probability'])
        np.testing.assert_allclose(-np.log(np.maximum(observed.mean(0), np.finfo(float).tiny)).mean(),
                                   row['mean_nll'], rtol=0, atol=1e-12)
    return row, arrays


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, object]:
    from src.Bayesian_state.evaluation import model_0923_online_refit as comparison
    from src.Bayesian_state.optimization.search.adaptive_proposals import point_id

    settings = yaml.safe_load(config.read_text())
    if settings['arm'] != 'SEARCH' or not 1 <= settings['parallel_budget'] <= 128:
        raise ValueError('This declared diagnostic uses SEARCH and the shared CPU budget')
    online = helpers(); recalled = online.helpers().helpers(); v = recalled.base.validation()
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); value = v.digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Previously bound input changed')
        inputs[str(path)] = value
        return path

    bind(config); bind(__file__)
    summaries, frozen, templates = {}, {}, {}
    subject, arm = settings['subject'], settings['arm']
    for label, key in [('RETAIN', 'retained_summary'), ('ONLINE', 'online_summary')]:
        manifest, fit, _, hashes = comparison.retained.load_summary(ROOT / settings[key])
        inputs.update(hashes)
        inputs.update(comparison.bind_execution(manifest, fit, label == 'ONLINE'))
        summaries[label] = manifest; frozen[label] = fit
        executed = []
        for file in sorted((Path(manifest['evaluation']) / 'jobs').glob('*.json')):
            job = v.read(bind(file))
            if job['model'] == arm and v.read(Path(job['folder']) / 'prepared.json')['subject'] == subject:
                executed.append(job)
        templates[label] = {role: [j for j in executed if j['phase'] == role]
                            for role in ('evaluate', 'autonomous')}
    original = frozen['ONLINE']['settings']
    online.validate_settings(original, frozen['RETAIN']['settings'])
    for role in ('evaluate', 'autonomous'):
        a, b = (templates[label][role] for label in ('RETAIN', 'ONLINE'))
        if len(a) != len(b) or not a:
            raise ValueError('Incomplete source evaluation bank')
        for left, right in zip(a, b):
            if any(left[k] != right[k] for k in SEMANTIC_FIELDS) or left['seeds'] != right['seeds']:
                raise ValueError('Cross mechanisms require identical task, seed and precision settings')
    points = {}
    for label in ('RETAIN', 'ONLINE'):
        rows = [r for r in frozen[label]['nominees'] if (r['subject'], r['model']) == (subject, arm)]
        if len(rows) != 1:
            raise ValueError('Missing unique behavioral nominee')
        points[label] = rows[0]['candidate']
    confirmation = ROOT / settings['confirmation']
    proof = v.read(bind(confirmation / 'validation.json'))
    if proof['status'] != 'complete' or proof['nomination_changed']:
        raise ValueError('Need completed independent confirmation with unchanged nomination')
    for path, sha in proof['input_sha256'].items():
        if v.digest(bind(path)) != sha:
            raise ValueError('Confirmation input changed')
    scores = pd.read_csv(bind(confirmation / 'scores.csv'))
    challenger = scores[scores.candidate == settings['challenger']]
    if len(challenger) != 1 or challenger.iloc[0].production_nominee:
        raise ValueError('Declare one distinct behavior-confirmed sensitivity point')
    candidate_path = Path(summaries['ONLINE']['fit']) / 'jobs' / f'selection_S{subject}_{arm}_{settings["challenger"]}_C000.json'
    points['NEAR'] = v.read(bind(candidate_path))['candidate']
    for point in points.values():
        if point_id(point['hyperparams']) != point['id']:
            raise ValueError('Parameter identity mismatch')
        if any(point['hyperparams'].get(k) != value for k, value in original['fixed_coordinates'].items()):
            raise ValueError('Cross parameter changes the declared inactive coordinate')
    probe_path = ROOT / settings['online_probe']
    probe = v.read(bind(probe_path / 'protocol.json')); done = v.read(bind(probe_path / 'completion.json'))
    if done['status'] != 'complete' or done['smoke'] or done['protocol_sha256'] != v.digest(probe_path / 'protocol.json'):
        raise ValueError('Require completed original ONLINE path probe')
    probe_banks = [b for b in probe['banks'] if b['model'] == arm and b['schedule'] == 'online'
                   and b['candidate'] == points['RETAIN']]
    if len(probe_banks) != 1:
        raise ValueError('Original nominee absent from ONLINE path probe')
    probe_jobs = probe_banks[0]['chunks']
    for job in probe_jobs:
        for name, sha in job['source_sha256'].items():
            if v.digest(bind(ROOT / name)) != sha:
                raise ValueError('Probe implementation changed')
        check_chunk(job, 'ONLINE', v, bind)
    cells = [('RETAIN', 'RETAIN'), ('ONLINE', 'RETAIN'), ('RETAIN', 'ONLINE'),
             ('ONLINE', 'ONLINE'), ('ONLINE', 'NEAR')]
    jobs, banks = [], []
    for mechanism, parameter in cells:
        cell = f'{mechanism}_{parameter}'; candidate = points[parameter]
        bank = dict(cell=cell, mechanism=mechanism, parameter=parameter, candidate=candidate, roles={})
        for role in ('evaluate', 'autonomous'):
            bank['roles'][role] = []
            for index, template in enumerate(templates[mechanism][role][:1] if smoke else templates[mechanism][role]):
                reference = template if mechanism == parameter else None
                if mechanism == 'ONLINE' and parameter == 'RETAIN' and role == 'autonomous':
                    matches = [j for j in probe_jobs if compatible_probe(template, j, candidate)]
                    if len(matches) > 1:
                        raise ValueError('Duplicated reusable probe seed chunk')
                    reference = matches[0] if matches else None
                reused = reference is not None and not smoke
                if reused:
                    job = deepcopy(reference); check_chunk(job, mechanism, v, bind)
                else:
                    job = deepcopy(template)
                    job.update(name=f'{cell}_{role}_C{index:03d}', candidate=deepcopy(candidate),
                               smoke=smoke, cross_mechanism=mechanism, cross_parameter=parameter,
                               target=str((output / 'chunks' / f'{cell}_{role}_C{index:03d}').resolve()))
                    job['source_sha256'][ENTRY] = v.digest(Path(__file__))
                    if smoke:
                        job['seeds'] = job['seeds'][:1]
                    jobs.append(job)
                bank['roles'][role].append(dict(job=job, reused=reused,
                    prefix_reference=reference['target'] if smoke and reference else None))
        banks.append(bank)
    for path, sha in inputs.items():
        if v.digest(Path(path)) != sha:
            raise ValueError('Bound source changed during planning')
    return dict(settings=original, cross_settings=settings, smoke=smoke, jobs=jobs, banks=banks,
                inputs_sha256=inputs, sources={ENTRY: v.digest(Path(__file__))},
                parameter_selection='Frozen training-choice points; no new selection; oral excluded',
                scope='Development SEARCH mechanism/parameter cross and near-tied point sensitivity'), recalled


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new mechanism/parameter cross directory')
    protocol, recalled = prepare(config, output, smoke)
    v = recalled.base.validation(); output.mkdir(parents=True)
    if smoke:
        from src.Bayesian_state.workflows.analysis.model_0923_estimation_scope import smoke_data
        subject = protocol['cross_settings']['subject']
        source = Path(protocol['jobs'][0]['folder']).parent
        fixture = smoke_data(source, output, subject, v) / f'subject_{subject}'
        for job in protocol['jobs']:
            if job['phase'] == 'evaluate':
                job['folder'] = str(fixture.resolve())
        for file in fixture.iterdir():
            protocol['inputs_sha256'][str(file.resolve())] = v.digest(file)
    v.write(output / 'protocol.json', protocol)
    recalled.ENTRY = ENTRY
    recalled.execute(protocol['jobs'], output, 1 if smoke else protocol['cross_settings']['parallel_budget'])
    inputs = dict(protocol['inputs_sha256']); checks = 0; rows = []

    def bind(path):
        path = Path(path).resolve(); value = v.digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Bound cross input changed')
        inputs[str(path)] = value
        return path

    for bank in protocol['banks']:
        row = {k: val for k, val in bank.items() if k != 'roles'}
        for role, items in bank['roles'].items():
            for item in items:
                record, arrays = check_chunk(item['job'], bank['mechanism'], v, bind)
                if item['prefix_reference']:
                    with np.load(bind(Path(item['prefix_reference']).with_suffix('.npz'))) as z:
                        for key, actual in arrays.items():
                            expected = z[key]
                            if key == 'seeds': expected = expected[:1]
                            elif key == 'categories' or (key == 'choices' and role == 'evaluate'):
                                expected = expected[:record['trials']]
                            else: expected = expected[:1, :record['trials']]
                            np.testing.assert_array_equal(actual, expected); checks += actual.size
            if not smoke:
                aggregate = deepcopy(items[0]['job'])
                target = output / 'banks' / f'{bank["cell"]}_{role}'
                target.parent.mkdir(exist_ok=True)
                aggregate.update(target=str(target.resolve()), seeds=[s for x in items for s in x['job']['seeds']])
                combined = v.combine_evaluation_chunks(aggregate, [x['job'] for x in items], role == 'autonomous')
                row[role] = dict(target=aggregate['target'], record=combined)
        rows.append(row)
    v.write(output / 'banks.json', dict(status='complete', smoke=smoke, rows=rows))
    if not smoke:
        summarize(output, protocol, rows)
    for path, sha in inputs.items():
        if v.digest(Path(path)) != sha:
            raise ValueError('Bound input changed during cross execution')
    entries = [item for bank in protocol['banks'] for items in bank['roles'].values() for item in items]
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, cells=len(rows),
        new_filters=sum(len(x['job']['seeds']) for x in entries if not x['reused'] and x['job']['phase'] == 'evaluate'),
        new_paths=sum(len(x['job']['seeds']) for x in entries if not x['reused'] and x['job']['phase'] == 'autonomous'),
        reused_filters=sum(len(x['job']['seeds']) for x in entries if x['reused'] and x['job']['phase'] == 'evaluate'),
        reused_paths=sum(len(x['job']['seeds']) for x in entries if x['reused'] and x['job']['phase'] == 'autonomous'),
        exact_prefix_elements=checks, nominees_changed=False, inputs_sha256=inputs,
        protocol_sha256=v.digest(output / 'protocol.json'), artifacts={p.name: v.digest(p) for p in output.glob('*.csv')}))


def summarize(output: Path, protocol: dict, rows: list[dict]) -> None:
    from src.Bayesian_state.evaluation.model_0923_reentry_refit import match_task, window_estimate
    from src.Bayesian_state.evaluation.model_0923_estimation_scope import paired_difference, validate_generation
    from src.Bayesian_state.evaluation.model_0923_online_refit import feedback_scores
    from src.Bayesian_state.metrics.task import category_learning_feedback

    banks, estimates, scores, paths = {}, {}, [], []
    for row in rows:
        pair = []
        for role in ('evaluate', 'autonomous'):
            with np.load(Path(row[role]['target']).with_suffix('.npz')) as z:
                pair.append({k: z[k].copy() for k in z.files})
        c, a = pair; meta = dict(row['evaluate']['record'], target_rule=row['autonomous']['record']['target_rule'])
        validate_generation(a, row['autonomous']['record'])
        banks[row['cell']] = c, a, meta
        if len(banks) > 1: match_task(next(iter(banks.values())), banks[row['cell']])
        cut, length = meta['training_trials'], len(c['choices'])
        late = max(0, length - 128)
        for i, seed in enumerate(a['seeds']):
            paths.append(dict(cell=row['cell'], seed=int(seed),
                late_accuracy=float(np.mean(a['choices'][i, late:] == a['categories'][late:])),
                target_ever=bool(a['present'][i].any()),
                target_late_present=float(a['present'][i, late:].mean()),
                target_late_contribution=float(a['contribution'][i, late:].mean()),
                target_exits=int(a['target_exited'][i].sum()),
                target_capacity_drops=int(a['target_capacity_dropped'][i].sum()),
                search_events=int(a['search'][i].sum())))
        for phase, first, last in [('training', 0, cut), ('test', cut, length), ('whole', 0, length)]:
            estimate = window_estimate(c, a, first, last, protocol['settings']['features']['block_width'])
            estimates[row['cell'], phase] = estimate
            scores.append(dict(cell=row['cell'], phase=phase, **estimate['metrics']))
    pairs = [('ONLINE_RETAIN', 'RETAIN_RETAIN'), ('ONLINE_ONLINE', 'RETAIN_ONLINE'),
             ('RETAIN_ONLINE', 'RETAIN_RETAIN'), ('ONLINE_ONLINE', 'ONLINE_RETAIN'),
             ('ONLINE_NEAR', 'ONLINE_ONLINE')]
    contrasts, feedback = [], []
    for new, old in pairs:
        for phase in ('training', 'test', 'whole'):
            contrasts.append(dict(new=new, reference=old, phase=phase,
                **paired_difference(estimates[new, phase], estimates[old, phase])))
        c, _, meta = banks[old]
        levels = np.array([category_learning_feedback(int(x), int(y), condition=meta['condition'])
                           for x, y in zip(c['choices'], c['categories'])])
        table, _ = feedback_scores(c, banks[new][0], levels, meta['training_trials'])
        feedback.append(table.assign(new=new, reference=old))
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    pd.DataFrame(contrasts).to_csv(output / 'contrasts.csv', index=False)
    pd.concat(feedback).to_csv(output / 'feedback_scores.csv', index=False)
    pd.DataFrame(paths).to_csv(output / 'paths.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'): parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Use --config and --output, or an internal --worker')
