"""Compare cohort MIX and SELECT refits under the same original search policy.

Use --original MIX_SUMMARY --revised SELECT_SUMMARY --subject ID --output NEW.
Both completed summaries retain SEPARATE evidence and rule precision. Supports,
actual initial banks, seed namespaces and stage limits must match; adaptive
exchange work is reported separately. Reuse the existing cohort scoring and
whole-filter/path Monte Carlo errors. Oral data never enter these scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import model_0923_cohort_belief_comparison as cohort
from . import model_0923_reentry_refit as retained
from . import model_0923_separate_refit_comparison as shared
from .model_0923_h_validation import digest, load
from .model_0923_refit_budget import audit_exchange, budget_contrasts
from ..workflows.analysis import model_0923_cohort_execution_states as states
from ..workflows.analysis.model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[3]
EXECUTION_ADDITIONS = {
    'src/Bayesian_state/workflows/analysis/model_0923_cohort_execution_refit.py',
    'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_probe.py',
    'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_refit.py',
    'src/Bayesian_state/model/modules/hypothesis_transition/dominant_rule_execution.py',
}


def check_settings(original: dict, revised: dict, subject: int, reference_fit: Path) -> None:
    """Allow execution identity and available workers to change, preserving search."""
    allowed = {'execution_policy', 'execution_reference_fit', 'scope', 'parallel_budget'}
    if ({k: v for k, v in original.items() if k not in allowed} !=
            {k: v for k, v in revised.items() if k not in allowed}):
        raise ValueError('Comparison changed the original cohort search policy')
    if (any(s['subjects'] != [subject] or s['belief_mechanism'] != 'SEPARATE' for s in (original, revised))
            or original['scope'] != 'common_policy_development_belief_refit'
            or original.get('execution_policy') is not None
            or revised['scope'] != states.USE or revised['execution_policy'] != 'SELECT'
            or (ROOT / revised['execution_reference_fit']).resolve() != reference_fit.resolve()):
        raise ValueError('Require the matching participant and original SEPARATE MIX reference')


def presentation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the shared numerical scorer while naming the actual two policies."""
    rename = {}
    for key in frame:
        for prefix, label in [('online_', 'mix_'), ('separate_', 'select_'),
                              ('reset_', 'mix_'), ('retain_', 'select_')]:
            if key.startswith(prefix):
                rename[key] = label + key[len(prefix):]
    columns = [rename.get(key, key) for key in frame]
    if len(set(columns)) != len(columns):
        raise ValueError('Comparison column labels collide')
    return frame.rename(columns=rename)


def compare(original: Path, revised: Path, subject: int, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new completed-comparison directory')
    directories = [original, revised]
    records = [retained.load_summary(path) for path in directories]
    inputs = {}

    def merge(group):
        for path, value in group.items():
            path = str((ROOT / path).resolve())
            if path in inputs and inputs[path] != value:
                raise ValueError('Conflicting input bindings: ' + path)
            inputs[path] = value

    for directory, record, module in zip(directories, records, [cohort, states.comparison_helpers()]):
        merge(record[3]); merge(module.bind_summary(directory, record[0], 'SEPARATE'))
    frozen = [record[1] for record in records]
    fits = [Path(record[0]['fit']) for record in records]
    settings = [record['settings'] for record in frozen]
    check_settings(*settings, subject, fits[0])
    protocols, spaces = [], []
    for fit, point in zip(fits, frozen):
        protocol = load(fit / 'protocol.json'); space = load(fit / 'parameter_spaces.json')
        if (protocol['smoke'] or point['smoke'] or protocol['settings'] != point['settings']
                or protocol['source_sha256'] != point['source_sha256']):
            raise ValueError('Incomplete or inconsistent scientific reference')
        protocols.append(protocol); spaces.append(space)
        merge({str(fit / name): digest(fit / name) for name in ('protocol.json', 'parameter_spaces.json')})
    if spaces[0] != spaces[1] or protocols[0]['initial_banks'] != protocols[1]['initial_banks']:
        raise ValueError('Parameter supports or actual initial banks changed')
    old_source, new_source = [point['source_sha256'] for point in frozen]
    if (any(new_source.get(k) != v for k, v in old_source.items())
            or set(new_source) - set(old_source) != EXECUTION_ADDITIONS):
        raise ValueError('The candidate changes more than its declared execution implementation')
    keys = {(subject, arm) for arm in settings[0]['arms']}
    banks = [shared.load_bank(record[0], record[1], keys, settings[0]) for record in records]
    scorer = states.helper('src.Bayesian_state.evaluation._cohort_select_comparison', states.COMPARISON)
    scorer.MECHANISMS = ('MIX', 'SELECT')
    tables = scorer.score_banks(banks, [record[2] for record in records], settings[0])
    exchanges = []
    for fit in fits:
        frame, bound = audit_exchange(fit, [subject]); exchanges.append(frame); merge(bound)
    tables['budgets'] = budget_contrasts(records[0][2]['budgets'], records[1][2]['budgets'], *exchanges)
    tables['exchange_candidates'] = pd.concat([frame.assign(variant=label)
        for frame, label in zip(exchanges, ('MIX', 'SELECT'))])
    tables = {name: presentation_columns(frame) for name, frame in tables.items()}
    for path in (Path(__file__), Path(cohort.__file__), Path(retained.__file__), Path(shared.__file__), Path(states.__file__),
                 ROOT / 'src/Bayesian_state/evaluation/model_0923_estimation_scope.py',
                 ROOT / 'src/Bayesian_state/evaluation/model_0923_refit_budget.py'):
        merge({str(path): digest(path)})
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError('Comparison source changed: ' + path)
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    write(output / 'completion.json', dict(status='complete', subject=subject, arms=settings[0]['arms'],
        original=str(original.resolve()), revised=str(revised.resolve()), inputs_sha256=inputs,
        comparison='SELECT minus MIX; SEPARATE evidence in both; parameters selected from the same original prefix choices',
        policy='Same supports, actual initial banks, seeds and stage limits; adaptive exchange work reported separately',
        scientific_scope='Development comparison at the original chronological boundary; oral excluded',
        uncertainty='Paired whole-filter/path Monte Carlo errors; parameter and population uncertainty evaluated separately',
        actual_search_compute_equal=bool(tables['budgets'].actual_compute_equal.all()),
        artifacts={name + '.csv': digest(output / (name + '.csv')) for name in tables}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'revised', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--subject', type=int, required=True)
    args = parser.parse_args()
    compare(args.original, args.revised, args.subject, args.output)
