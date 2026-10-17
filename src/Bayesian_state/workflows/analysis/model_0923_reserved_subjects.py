"""Reserve participant identities before opening their scientific outcomes.

Inspect Model 0923 configuration subject fields and result path names, then
select the first two unlisted identifiers in each condition. Only iSub and
condition are loaded from the behavioral table. Absence from documented use
does not establish that no researcher has previously seen these participants.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import sys

import pandas as pd
import yaml

from .model_0923_decision_basis_replay import ROOT, digest, write

FIELDS = {'subjects', 'subject_ids', 'initial_subjects', 'plus_subjects', 'development_subjects',
          'new_validation_subjects', 'audit_subjects', 'recovery_subjects', 'replay_subjects', 'meg_subjects'}
PATTERN = re.compile(r'(?:^|[/_])(?:subject_|S)(\d{3})(?=[/_.]|$)')


def subject_id(value) -> int | None:
    if type(value) is int or isinstance(value, str) and value.isdigit():
        number = int(value)
        if 100 <= number <= 999:
            return number
    return None


def declared_subjects(value, prefix: str = '') -> list[tuple[int, str]]:
    """Extract explicit participant fields without using scores or parameters."""
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = f'{prefix}.{key}' if prefix else str(key)
            if key in FIELDS and isinstance(item, list):
                for entry in item:
                    subject = entry[0] if isinstance(entry, list) and entry else entry
                    number = subject_id(subject)
                    if number is not None:
                        rows.append((number, location))
                    elif isinstance(entry, dict):
                        rows.extend(declared_subjects(entry, location))
            elif key in {'subject', 'subject_id'} and subject_id(item) is not None:
                rows.append((subject_id(item), location))
            elif key == 'subject_overrides' and isinstance(item, dict):
                rows.extend((subject_id(subject), location) for subject in item if subject_id(subject) is not None)
            else:
                rows.extend(declared_subjects(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(declared_subjects(item, f'{prefix}[{index}]'))
    return rows


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new participant-reservation audit directory')
    use, hashes = [], {}
    for file in sorted((ROOT / 'configs').rglob('*0923*.yaml')):
        relative = str(file.relative_to(ROOT)); hashes[relative] = digest(file)
        use.extend(dict(subject=subject, kind='configuration', source=relative, field=field)
                   for subject, field in declared_subjects(yaml.safe_load(file.read_text())))
    # Path-name inspection covers existing runs whose participant list may no
    # longer occur in current configs. Numeric result contents remain unopened.
    witnesses = {}
    for directory, dirs, files in os.walk(ROOT / 'results/model_0923'):
        dirs[:] = [name for name in dirs if name not in {'source', 'sources', 'resources_cache', '__pycache__'}]
        relative = str(Path(directory).relative_to(ROOT))
        for name in dirs + files:
            path = relative + '/' + name
            for match in PATTERN.finditer(path):
                subject = int(match.group(1))
                # One concrete path per run and participant is enough to
                # document prior use; repeated numerical chunks add no evidence.
                key = (subject, Path(path).parts[2])
                witnesses.setdefault(key, dict(subject=subject, kind='result_path', source=path, field='path_name'))
    use.extend(witnesses.values())
    registry = pd.DataFrame(use).drop_duplicates().sort_values(['subject', 'kind', 'source', 'field'])
    data_file = ROOT / 'data/exp123/processed/Task2_processed.csv'
    identities = pd.read_csv(data_file, usecols=['iSub', 'condition']).drop_duplicates()
    if identities.iSub.duplicated().any() or set(identities.condition) != {1, 2, 3}:
        raise ValueError('Expected one declared condition per participant')
    available = identities[~identities.iSub.isin(registry.subject)].sort_values(['condition', 'iSub'])
    selected = available.groupby('condition', sort=True).head(2)
    if len(selected) != 6 or not selected.groupby('condition').size().eq(2).all():
        raise ValueError('Need two currently unlisted identifiers per condition')
    output.mkdir(parents=True)
    registry.to_csv(output / 'documented_use.csv', index=False)
    identities.sort_values(['condition', 'iSub']).to_csv(output / 'identities.csv', index=False)
    available.to_csv(output / 'unlisted_identities.csv', index=False)
    selected.to_csv(output / 'reserved_identities.csv', index=False)
    write(output / 'manifest.json', dict(status='complete',
        selected=selected.rename(columns={'iSub': 'subject'}).to_dict('records'),
        selection='first_two_identifiers_per_condition_absent_from_current_0923_configs_and_result_path_names',
        inspected_behavior_columns=['iSub', 'condition'], scientific_outcomes_opened=False,
        python=sys.version, versions={name: importlib.metadata.version(name) for name in ('pandas', 'PyYAML')},
        configuration_sha256=hashes, behavioral_table_sha256=digest(data_file),
        source_sha256=digest(Path(__file__)),
        artifacts={p.name: digest(p) for p in output.iterdir()},
        scope='reserved mechanism-evaluation candidates; full evaluation protocol remains to be frozen',
        limitation='documented model-specific use inventory; prior human exposure and cross-model use not established'))
    print(selected.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.output)
