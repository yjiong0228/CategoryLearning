"""Compare each proposal arm before and after retained-precision refitting.

Use --reset-summary ORIGINAL --retained-summary RETAINED --output NEW_DIRECTORY
after summarizing both complete fits with model_0923_episodic_refit. The retained
fit must name the original RESET fit and preserve its candidate supports,
initial points, seeds and search limits. Actual exchange work can differ when
the same union rule removes different numbers of repeated candidates; audit
and report that difference explicitly. Each contrast includes both
the mechanism change and re-estimation of behavioral parameters. All subjects
are development subjects; oral reports do not enter these scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .model_0923_episodic_refit import loss_parts
from .model_0923_estimation_scope import paired_difference, validate_generation
from .model_0923_h_generation_uncertainty import path_jackknife
from .model_0923_refit_budget import audit_exchange, budget_contrasts
from .model_0923_h_validation import (
    block_paths, checked_arrays, digest, distribution_scores, jackknife_scores,
    jackknife_se, load, nll,
)
from ..workflows.analysis import model_0923_reentry_precision_refit as refit
from ..workflows.analysis.model_0923_h_validation import write


def load_summary(source: Path) -> tuple[dict, dict, dict, dict]:
    manifest = load(source/'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Need a completed refit summary')
    inputs = dict(manifest['inputs_sha256'])
    inputs[str((source/'manifest.json').resolve())] = digest(source/'manifest.json')
    for name, expected in manifest['summary_sha256'].items():
        inputs[str((source/name).resolve())] = expected
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('Refit summary input or table changed')
    tables = {name: pd.read_csv(source/(name+'.csv')) for name in
              ('scores', 'budgets', 'parameters', 'paths', 'curves')}
    fit, evaluation = Path(manifest['fit']), Path(manifest['evaluation'])
    frozen, done = load(fit/'nominees.json'), load(fit/'completion.json')
    protocol = load(evaluation/'protocol.json')
    evaluation_done = load(evaluation/'completion.json')
    if (done['status'] != 'complete' or done['smoke'] or done['test_choices_used'] is not False
            or evaluation_done['status'] != 'complete' or Path(protocol['fit']).resolve() != fit.resolve()
            or frozen['smoke'] or frozen['status'] != 'frozen_before_test'
            or any(item['nomination_sha256'] != digest(fit/'nominees.json')
                   for item in (done, protocol, evaluation_done))
            or protocol['source_sha256'] != frozen['source_sha256']):
        raise ValueError('Need completed behavior-prefix fits with frozen evaluation parameters')
    inputs[str((fit/'completion.json').resolve())] = digest(fit/'completion.json')
    return manifest, frozen, tables, inputs


def window_estimate(conditional: dict, autonomous: dict, first: int, last: int, width: int) -> dict:
    """Reuse sequence and distribution scores; delete complete random replicates."""
    human = conditional['choices'] == conditional['categories']
    generated = autonomous['choices'] == autonomous['categories'][None, :]
    p = conditional['observed_probability'][:, first:last]
    samples = block_paths(generated, first, last, width)
    observed = block_paths(human[None, :], first, last, width)[0]
    return dict(filter_seeds=conditional['seeds'], path_seeds=autonomous['seeds'],
        metrics=dict(choice_nll=nll(p), **distribution_scores(samples, observed),
                     **loss_parts(p, human[first:last])),
        jackknife=dict(choice_nll=jackknife_scores(p), **path_jackknife(samples, observed)))


def match_task(old: tuple, new: tuple) -> None:
    """A paired comparison needs identical task trials, boundaries and seed banks."""
    old_c, old_a, old_row = old
    new_c, new_a, new_row = new
    for field in ('condition', 'training_trials', 'particles', 'target_rule'):
        if old_row[field] != new_row[field]:
            raise ValueError('Refit comparison changed task or evaluation budget: '+field)
    for field in ('seeds', 'choices', 'categories'):
        np.testing.assert_array_equal(old_c[field], new_c[field])
    for field in ('seeds', 'categories'):
        np.testing.assert_array_equal(old_a[field], new_a[field])


def compare(reset_summary: Path, retained_summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new refitted-precision comparison directory')
    old, old_frozen, old_tables, inputs = load_summary(reset_summary)
    new, new_frozen, new_tables, more = load_summary(retained_summary)
    inputs.update(more)
    settings = new_frozen['settings']
    retained_fit = Path(new['fit'])
    if (refit.ROOT/settings['reset_reference_fit']).resolve() != Path(old['fit']).resolve():
        raise ValueError('Retained fit names a different RESET reference')
    references = refit.reference_inputs(refit.helpers(), settings)
    retention_file = retained_fit/'retention_protocol.json'
    retention = load(retention_file)
    if (retention['input_sha256'] != references
            or retention['structural_intervention'] != refit.INTERVENTION
            or any(new_frozen['source_sha256'].get(name) != digest(refit.ROOT/name)
                   for name in (refit.ENTRY, refit.MECHANISM))):
        raise ValueError('Retained-precision refit provenance changed')
    inputs.update(references)
    inputs[str(retention_file.resolve())] = digest(retention_file)
    for name, expected in new_frozen['source_sha256'].items():
        if digest(refit.ROOT/name) != expected:
            raise ValueError('Retained fitting implementation changed')
        inputs[str((refit.ROOT/name).resolve())] = expected
    subjects, arms = settings['subjects'], list(settings['arms'])
    keys = [(subject, arm) for subject in subjects for arm in arms]
    exchange_audits = []
    for fit in (Path(old['fit']), Path(new['fit'])):
        audit, more = audit_exchange(fit, subjects)
        exchange_audits.append(audit)
        inputs.update(more)
    budget_tables = [t['budgets'].loc[t['budgets'].subject.isin(subjects)
                                    & t['budgets'].model.isin(arms)] for t in (old_tables,new_tables)]
    actual_budgets = budget_contrasts(*budget_tables, *exchange_audits)
    scores, curves, parameters, path_differences, contrasts = [], [], [], [], []
    banks = {}
    for label, manifest, frozen, tables in [('RESET', old, old_frozen, old_tables),
                                           ('RETAIN', new, new_frozen, new_tables)]:
        evaluation = Path(manifest['evaluation'])
        nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
        roles = {role: {(r['subject'], r['model']): r for r in
                       load(evaluation/role/'evaluation.json')['rows']}
                 for role in ('conditional', 'autonomous')}
        for subject, arm in keys:
            cell = (subject, arm)
            c_row, a_row = (roles[role][cell] for role in ('conditional', 'autonomous'))
            if any(row['candidate'] != nominees[cell] for row in (c_row, a_row)):
                raise ValueError('Evaluation differs from the nominated behavioral parameters')
            c, a = (checked_arrays(evaluation/role, row) for role, row in
                    [('conditional', c_row), ('autonomous', a_row)])
            validate_generation(a, a_row)
            if (c_row['condition'] != a_row['condition']
                    or c_row['training_trials'] != a_row['training_trials']):
                raise ValueError('Conditional and autonomous task boundaries disagree')
            np.testing.assert_array_equal(c['categories'], a['categories'])
            if (not np.isfinite(c['probabilities']).all() or np.any(c['probabilities'] < 0)
                    or len(set(c['seeds'].tolist())) != len(c['seeds'])):
                raise ValueError('Invalid conditional probabilities or repeated filter seeds')
            np.testing.assert_allclose(c['probabilities'].sum(axis=-1), 1., atol=1e-12, rtol=0)
            actual = np.take_along_axis(c['probabilities'], (c['choices']-1)[None, :, None], axis=2)[..., 0]
            np.testing.assert_array_equal(actual, c['observed_probability'])
            if (len(a['seeds']) != settings['autonomous_paths']
                    or len(c['seeds']) != settings['budgets']['evaluate']['seeds']):
                raise ValueError('Missing evaluation paths or filter repeats')
            banks[label, cell] = c, a, dict(c_row, target_rule=a_row['target_rule'])
        mask = lambda table: table.subject.isin(subjects) & table.model.isin(arms)
        scores.append(tables['scores'].loc[mask(tables['scores'])].assign(variant=label))
        curves.append(tables['curves'].loc[mask(tables['curves'])].assign(variant=label))
        parameters.append(tables['parameters'].loc[mask(tables['parameters'])].assign(variant=label))
    for subject, arm in keys:
        cell = (subject, arm)
        match_task(banks['RESET', cell], banks['RETAIN', cell])
        c, _, row = banks['RESET', cell]
        cut, n = row['training_trials'], len(c['choices'])
        if not 0 < cut < n:
            raise ValueError('Need the original nonempty prefix and later window')
        for phase, first, last in [('training', 0, cut), ('test', cut, n), ('whole', 0, n)]:
            estimates = []
            for label, tables in [('RESET', old_tables), ('RETAIN', new_tables)]:
                c, a, _ = banks[label, cell]
                estimate = window_estimate(c, a, first, last, settings['features']['block_width'])
                table = tables['scores']
                saved = table[(table.subject == subject) & (table.model == arm) & (table.phase == phase)]
                if len(saved) != 1:
                    raise ValueError('Missing or duplicated score window')
                for metric, value in estimate['metrics'].items():
                    np.testing.assert_allclose(value, saved.iloc[0][metric], atol=1e-12, rtol=0)
                estimates.append(estimate)
            contrasts.append(dict(subject=subject, model=arm, phase=phase, trials=last-first,
                **paired_difference(estimates[1], estimates[0])))
        paths = []
        for tables in (old_tables, new_tables):
            table = tables['paths']
            paths.append(table[(table.subject == subject) & (table.model == arm)].set_index('seed').sort_index())
        np.testing.assert_array_equal(paths[0].index, paths[1].index)
        for metric in ('late_accuracy', 'target_ever', 'target_late_present', 'target_late_contribution',
                       'target_exits', 'target_capacity_drops', 'search_events', 'target_proposals'):
            before, after = (p[metric].to_numpy(float) for p in paths)
            d = after-before
            path_differences.append(dict(subject=subject, model=arm, metric=metric,
                reset=before.mean(), retain=after.mean(), delta=d.mean(),
                paired_mc_se=d.std(ddof=1)/np.sqrt(len(d)), paths=len(d)))
    # Actual retained workers add tags after the evaluation plan is saved.
    evaluation = Path(new['evaluation'])
    plans = {job['name']: job for job in load(evaluation/'protocol.json')['jobs']}
    files = sorted((evaluation/'jobs').glob('*.json'))
    if ({file.stem for file in files} != set(plans)
            or len(plans) != load(evaluation/'completion.json')['jobs']):
        raise ValueError('Missing retained evaluation jobs')
    for file in files:
        job = load(file)
        expected = dict(plans[file.stem], structural_intervention=refit.INTERVENTION,
                        retain_precision=True, retention_reference_inputs=references)
        sidecar = Path(job['target']).with_suffix('.precision.json')
        execution = load(sidecar)
        if (job != expected
                or execution.get('retained_precision') is not True
                or execution.get('structural_intervention') != refit.INTERVENTION):
            raise ValueError('Missing executed precision-retention intervention')
        inputs.update({str(file.resolve()): digest(file), str(sidecar.resolve()): digest(sidecar)})
    tables = dict(scores=pd.concat(scores), curves=pd.concat(curves), parameters=pd.concat(parameters),
                  contrasts=pd.DataFrame(contrasts), path_contrasts=pd.DataFrame(path_differences),
                  budgets=actual_budgets, exchange_candidates=pd.concat([
                      a.assign(variant=v) for a,v in zip(exchange_audits, ('RESET','RETAIN'))]))
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('Comparison input changed during scoring')
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output/(name+'.csv'), index=False)
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ('model_0923_episodic_refit.py', 'model_0923_estimation_scope.py',
         'model_0923_h_generation_uncertainty.py', 'model_0923_h_validation.py', 'model_0923_refit_budget.py')]]
    (output/'source').mkdir()
    for file in sources:
        shutil.copy2(file, output/'source'/file.name)
    write(output/'manifest.json', dict(status='complete', subjects=subjects, arms=arms,
        reset_summary=str(reset_summary.resolve()), retained_summary=str(retained_summary.resolve()),
        inputs_sha256=inputs, sources_sha256={p.name: digest(p) for p in sources},
        tables_sha256={name+'.csv': digest(output/(name+'.csv')) for name in tables},
        contrast='RETAIN refit minus original RESET; mechanism and re-estimation jointly change',
        budget_matching='common settings and limits; only verified exchange-union deduplication may change actual work',
        actual_search_compute_equal=bool(actual_budgets.actual_compute_equal.all()),
        data_use='behavior-prefix estimation; later windows are development diagnostics; oral excluded',
        uncertainty='paired complete-filter/path Monte Carlo standard errors at frozen nominated parameters'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reset-summary', 'retained-summary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.reset_summary, args.retained_summary, args.output)
