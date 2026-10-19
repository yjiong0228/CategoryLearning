"""Compare completed ONLINE and SEPARATE behavioral refits.

Run --online-summary OLD --separate-summary NEW --output NEW_DIRECTORY after
both summaries exist. The SEPARATE summary must include its actual execution
audit. Shared score functions retain the original trial windows, seed pairing
and complete-filter/path Monte Carlo errors. Contrasts include both the belief
mechanism and behavioral re-estimation. Oral data do not enter this comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_0923_online_refit as online
from . import model_0923_reentry_refit as retained
from . import model_0923_separate_belief_refit as summary
from .model_0923_estimation_scope import paired_difference, validate_generation
from .model_0923_h_validation import checked_arrays, digest, load
from .model_0923_refit_budget import audit_exchange, budget_contrasts
from ..metrics.task import category_learning_feedback
from ..workflows.analysis import model_0923_separate_belief_refit as refit
from ..workflows.analysis.model_0923_h_validation import write


def audited_inputs(source: Path, manifest: dict) -> dict:
    """Rebind the additional SEPARATE audit to the exact completed summary."""
    extra_path, audit_path = source / 'separate_manifest.json', source / 'execution_audit.json'
    extra, audits = load(extra_path), load(audit_path)
    if (extra.get('status') != 'complete' or extra.get('belief_intervention') != refit.INTERVENTION
            or extra.get('generic_manifest_sha256') != digest(source / 'manifest.json')
            or extra.get('execution_audit_sha256') != digest(audit_path)
            or set(audits) != {'fit', 'evaluation'}):
        raise ValueError('SEPARATE summary lacks its bound execution audit')
    inputs = {str(path.resolve()): digest(path) for path in (extra_path, audit_path)}
    inputs.update(extra['inputs_sha256'])
    inputs.update({str((refit.ROOT / name).resolve()): sha for name, sha in extra['source_sha256'].items()})
    if extra['source_sha256'].get(str(Path(summary.__file__).relative_to(refit.ROOT))) != digest(Path(summary.__file__)):
        raise ValueError('Execution audit used a different summary implementation')
    for role, audit in audits.items():
        files = list((Path(manifest[role]) / 'jobs').glob('*.json'))
        if (audit.get('status') != 'complete' or audit.get('smoke') is not False
                or audit['jobs'] != len(files) or extra['jobs'][role] != len(files)
                or len(audit['rows']) != len(files)
                or any(str(file.resolve()) not in audit['inputs_sha256'] for file in files)):
            raise ValueError('Audit does not cover the scientific worker jobs')
        inputs.update(audit['inputs_sha256'])
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('SEPARATE execution audit input changed')
    return inputs


def bind_separate_execution(source: Path, manifest: dict, frozen: dict) -> dict:
    """Check actual workers against the frozen policy and belief sidecars."""
    inputs = audited_inputs(source, manifest)
    settings = frozen['settings']
    driver = refit.load_helper('separate_comparison_online', refit.ONLINE)
    retention = driver.helpers()
    references = refit.reference_inputs(driver, retention, retention.helpers(), settings)
    inputs.update(references)
    fit, evaluation = Path(manifest['fit']), Path(manifest['evaluation'])
    declaration_path = fit / 'separate_refit_protocol.json'
    declaration = load(declaration_path)
    if (declaration['input_sha256'] != references
            or declaration.get('belief_intervention') != refit.INTERVENTION
            or declaration.get('fixed_coordinates') != settings['fixed_coordinates']):
        raise ValueError('SEPARATE search declaration changed')
    inputs[str(declaration_path.resolve())] = digest(declaration_path)
    for name, sha in frozen['source_sha256'].items():
        path = refit.ROOT / name
        if digest(path) != sha:
            raise ValueError('Executed fitting source changed')
        inputs[str(path.resolve())] = sha
    protocol = load(evaluation / 'protocol.json')
    plans = {job['name']: job for job in protocol['jobs']}
    files = sorted((evaluation / 'jobs').glob('*.json'))
    if (len(plans) != len(protocol['jobs']) or {file.stem for file in files} != set(plans)
            or len(plans) != load(evaluation / 'completion.json')['jobs']):
        raise ValueError('Missing or duplicated evaluation jobs')
    for file in files:
        job = load(file); target = Path(job['target'])
        summary.check_execution(job, load(target.with_suffix('.json')),
                                load(target.with_suffix('.belief.json')),
                                load(target.with_suffix('.precision.json')))
        base = dict(job)
        for key in ('online_belief', 'belief_intervention', 'fixed_coordinates', 'probe_mechanism'):
            del base[key]
        inputs.update(retained.executed_retention_inputs(plans[file.stem], base, settings, references))
    return inputs


def load_bank(manifest: dict, frozen: dict, keys: set, settings: dict) -> dict:
    """Validate complete prediction banks for each frozen behavioral nominee."""
    evaluation = Path(manifest['evaluation'])
    nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
    if set(nominees) != keys or len(nominees) != len(frozen['nominees']):
        raise ValueError('Incomplete or duplicated nomination')
    roles, bank = {}, {}
    for role in ('conditional', 'autonomous'):
        rows = load(evaluation / role / 'evaluation.json')['rows']
        roles[role] = {(r['subject'], r['model']): r for r in rows}
        if set(roles[role]) != keys or len(rows) != len(keys):
            raise ValueError('Incomplete or duplicated evaluation cells')
    for cell in sorted(keys):
        c_row, a_row = (roles[role][cell] for role in ('conditional', 'autonomous'))
        if any(row['candidate'] != nominees[cell] for row in (c_row, a_row)):
            raise ValueError('Evaluation differs from the behavioral nominee')
        c, a = (checked_arrays(evaluation / role, row) for role, row in
                (('conditional', c_row), ('autonomous', a_row)))
        validate_generation(a, a_row)
        np.testing.assert_array_equal(c['categories'], a['categories'])
        if (not np.isfinite(c['probabilities']).all() or np.any(c['probabilities'] < 0)
                or len(set(c['seeds'].tolist())) != len(c['seeds'])
                or len(c['seeds']) != settings['budgets']['evaluate']['seeds']
                or len(a['seeds']) != settings['autonomous_paths']
                or c_row['particles'] != settings['budgets']['evaluate']['particles']
                or any(c_row[k] != a_row[k] for k in ('condition', 'training_trials'))):
            raise ValueError('Invalid prediction bank, evaluation budget or task boundary')
        np.testing.assert_allclose(c['probabilities'].sum(-1), 1., rtol=0, atol=1e-12)
        actual = np.take_along_axis(c['probabilities'], (c['choices'] - 1)[None, :, None], axis=2)[..., 0]
        np.testing.assert_array_equal(actual, c['observed_probability'])
        bank[cell] = c, a, dict(c_row, target_rule=a_row['target_rule'])
    return bank


def feedback_scores(old: dict, new: dict, feedback: np.ndarray, cut: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse pooled-filter feedback losses with the actual mechanism names."""
    tables = online.feedback_scores(old, new, feedback, cut)
    return tuple(table.rename(columns={name: ('online_' + name[9:] if name.startswith('retained_')
        else 'separate_' + name[7:]) for name in table if name.startswith(('retained_', 'online_'))})
        for table in tables)


def path_contrasts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Pair complete autonomous paths; trials within a path stay together."""
    paths = [frame.set_index('seed').sort_index() for frame in (old, new)]
    if any(not p.index.is_unique or len(p) < 2 for p in paths):
        raise ValueError('Need unique, replicated complete paths')
    np.testing.assert_array_equal(paths[0].index, paths[1].index)
    rows = []
    for metric in ('late_accuracy', 'target_ever', 'target_late_present', 'target_late_contribution',
                   'target_exits', 'target_capacity_drops', 'search_events', 'target_proposals'):
        before, after = (p[metric].to_numpy(float) for p in paths)
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError('Nonfinite autonomous path summary')
        delta = after - before
        rows.append(dict(metric=metric, online=before.mean(), separate=after.mean(), delta=delta.mean(),
                         paired_mc_se=delta.std(ddof=1) / np.sqrt(len(delta)), paths=len(delta)))
    return pd.DataFrame(rows)


def compare(online_summary: Path, separate_summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new SEPARATE refit comparison directory')
    old, old_frozen, old_tables, inputs = retained.load_summary(online_summary)
    new, new_frozen, new_tables, more = retained.load_summary(separate_summary); inputs.update(more)
    settings = new_frozen['settings']
    refit.validate_settings(settings, old_frozen['settings'])
    if (refit.ROOT / settings['online_reference_fit']).resolve() != Path(old['fit']).resolve():
        raise ValueError('SEPARATE fit names a different ONLINE reference')
    inputs.update(online.bind_execution(old, old_frozen, True))
    inputs.update(bind_separate_execution(separate_summary, new, new_frozen))
    subjects, arms = settings['subjects'], list(settings['arms'])
    keys = {(subject, arm) for subject in subjects for arm in arms}
    audits = []
    for manifest in (old, new):
        audit, more = audit_exchange(Path(manifest['fit']), subjects)
        audits.append(audit); inputs.update(more)
    budgets = budget_contrasts(old_tables['budgets'], new_tables['budgets'], *audits)
    budgets = budgets.rename(columns={key: ('online_' + key[6:] if key.startswith('reset_')
        else 'separate_' + key[7:]) for key in budgets if key.startswith(('reset_', 'retain_'))})
    banks = [load_bank(manifest, frozen, keys, settings) for manifest, frozen in
             ((old, old_frozen), (new, new_frozen))]
    contrasts, feedback_rows, trial_rows, path_rows = [], [], [], []
    for cell in sorted(keys):
        subject, arm = cell; labels = dict(subject=subject, model=arm)
        retained.match_task(banks[0][cell], banks[1][cell])
        c, _, row = banks[0][cell]; cut, length = row['training_trials'], len(c['choices'])
        if not 0 < cut < length:
            raise ValueError('Need nonempty original training and later windows')
        for phase, start, stop in [('training', 0, cut), ('test', cut, length), ('whole', 0, length)]:
            estimates = []
            for bank, saved in zip(banks, (old_tables, new_tables)):
                conditional, autonomous, _ = bank[cell]
                estimate = retained.window_estimate(conditional, autonomous, start, stop,
                                                    settings['features']['block_width'])
                scores = saved['scores']
                selected = scores[(scores.subject == subject) & (scores.model == arm) & (scores.phase == phase)]
                if len(selected) != 1:
                    raise ValueError('Missing or duplicated saved score')
                for metric, value in estimate['metrics'].items():
                    np.testing.assert_allclose(value, selected.iloc[0][metric], rtol=0, atol=1e-12)
                estimates.append(estimate)
            contrasts.append(dict(**labels, phase=phase, trials=stop-start,
                                  **paired_difference(estimates[1], estimates[0])))
        feedback = np.array([category_learning_feedback(int(choice), int(category), condition=row['condition'])
                             for choice, category in zip(c['choices'], c['categories'])])
        f, t = feedback_scores(banks[0][cell][0], banks[1][cell][0], feedback, cut)
        feedback_rows.append(f.assign(**labels)); trial_rows.append(t.assign(**labels))
        paths = [saved['paths'].loc[(saved['paths'].subject == subject) & (saved['paths'].model == arm)]
                 for saved in (old_tables, new_tables)]
        path_rows.append(path_contrasts(*paths).assign(**labels))
    tables = {name: pd.concat([saved[name].assign(variant=label) for saved, label in
                              ((old_tables, 'ONLINE'), (new_tables, 'SEPARATE'))])
              for name in ('scores', 'curves', 'parameters', 'paths')}
    tables.update(contrasts=pd.DataFrame(contrasts), feedback_scores=pd.concat(feedback_rows),
        trials=pd.concat(trial_rows), path_contrasts=pd.concat(path_rows), budgets=budgets,
        exchange_candidates=pd.concat([a.assign(variant=v) for a, v in zip(audits, ('ONLINE', 'SEPARATE'))]))
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Comparison input changed during scoring')
    output.mkdir(parents=True)
    for name, table in tables.items(): table.to_csv(output / (name + '.csv'), index=False)
    sources = [Path(__file__), Path(online.__file__), Path(retained.__file__), Path(summary.__file__),
               Path(refit.__file__), *[Path(__file__).with_name(name) for name in
                 ('model_0923_episodic_refit.py', 'model_0923_estimation_scope.py',
                  'model_0923_h_generation_uncertainty.py', 'model_0923_h_validation.py',
                  'model_0923_refit_budget.py')]]
    (output / 'source').mkdir()
    for path in sources: shutil.copy2(path, output / 'source' / path.name)
    write(output / 'manifest.json', dict(status='complete', online_summary=str(online_summary.resolve()),
        separate_summary=str(separate_summary.resolve()), subjects=subjects, arms=arms, inputs_sha256=inputs,
        source_sha256={str(p.relative_to(refit.ROOT)): digest(p) for p in sources},
        tables_sha256={name + '.csv': digest(output / (name + '.csv')) for name in tables},
        contrast='SEPARATE minus ONLINE; belief mechanism and behavioral re-estimation jointly change',
        actual_search_compute_equal=bool(budgets.actual_compute_equal.all()),
        budget_matching='Same supports and stage limits; audited exchange deduplication changes actual work',
        data_use='Behavior-prefix estimation; later windows are development diagnostics; oral excluded',
        uncertainty='Paired complete-filter/path simulation errors conditional on frozen parameters'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('online-summary', 'separate-summary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); compare(args.online_summary, args.separate_summary, args.output)
