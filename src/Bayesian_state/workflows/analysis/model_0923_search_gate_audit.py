"""Flag exactly closed search-parameter channels in completed POOL/SEARCH fits.

This read-only audit uses frozen parameter nominations and their source hashes.
It does not score behavior, read oral reports, or change parameter selection.
An unflagged coordinate still needs joint uncertainty and recovery assessment.
Run with repeated --fit directories and a new --output directory.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
H = 'engine.modules.hypo_transitions_mod.kwargs.'
C = H + 'search_controller.'
REQUIRED_SOURCES = tuple('src/Bayesian_state/model/modules/hypothesis_transition/' + name for name in (
    'mechanistic_rule_search.py', 'episodic_admission.py', 'episodic_proposal.py'))


def closed_channels(parameters: dict[str, float]) -> dict[str, str]:
    """Identify exact boundary conditions; numerical proximity is insufficient."""
    names = ('global_search', 'global_search_failure_gain', 'error_gain', 'failure_decay')
    values = {name: float(parameters[C + name]) for name in names}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Search parameters must be finite')
    if (not 0 <= values['global_search'] <= 1
            or not 0 <= values['global_search_failure_gain'] <= 1
            or not 0 <= values['failure_decay'] < 1 or values['error_gain'] < 0):
        raise ValueError('Search parameters outside the implemented range')
    result = {}
    if values['global_search'] == 1:
        result[H + 'local_structure_weight'] = 'global_search=1 closes the local proposal route'
        result[C + 'global_search_failure_gain'] = 'global_search=1 fixes the global fraction at 1'
    if values['error_gain'] == 0 and (
            values['global_search'] == 1 or values['global_search_failure_gain'] == 0):
        result[C + 'failure_decay'] = 'failure affects neither search occurrence nor global fraction'
    return result


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def nomination_rows(fit: Path) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Require completed scientific nominations and the audited H implementation."""
    file = fit / 'nominees.json'
    frozen = json.loads(file.read_text())
    completion_file = fit / 'completion.json'
    completion = json.loads(completion_file.read_text())
    if (completion.get('status') != 'complete' or frozen.get('smoke') is not False
            or completion.get('smoke') is not False
            or completion.get('nomination_sha256') != digest(file)
            or frozen.get('status') not in ('frozen_before_test', 'frozen_for_retrospective_diagnostics')):
        raise ValueError('Require complete scientific nominations with matching hashes')
    settings = frozen['settings']
    mechanism = settings['mechanism']
    if (list(settings['arms']) != ['POOL', 'SEARCH']
            or mechanism.get('kind') != 'E' or mechanism.get('capacity') != 'evidence_mean'
            or mechanism.get('search') != 'accumulate'):
        raise ValueError('Require the audited POOL/SEARCH mechanism')
    hashes = frozen['source_sha256']
    if not set(REQUIRED_SOURCES) <= set(hashes):
        raise ValueError('Missing audited H source provenance')
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Executing source differs from frozen fit: {name}')
    inputs = {str(file): digest(file), str(completion_file): digest(completion_file)}
    inputs.update({str(ROOT / name): expected for name, expected in hashes.items()})
    parameters, summaries = [], []
    if not frozen['nominees']:
        raise ValueError('No frozen nominees')
    seen = set()
    for row in frozen['nominees']:
        point = row['candidate']
        identity = dict(fit=str(fit), subject=row['subject'], model=row['model'],
                        objective=row.get('objective', 'prefix'), parameter_id=point['id'])
        key = (row['subject'], row['model'], identity['objective'])
        if key in seen or row['model'] not in settings['arms'] or identity['objective'] not in ('prefix', 'full'):
            raise ValueError('Invalid or duplicate nominee identity')
        seen.add(key)
        p = point['hyperparams']
        inactive = closed_channels(p)
        if not set(inactive) <= set(p):
            raise ValueError('Missing a flagged parameter coordinate')
        summaries.append(dict(identity, coordinates=len(p), closed_search_coordinates=len(inactive),
                              closed_parameters=';'.join(sorted(inactive))))
        for name, value in p.items():
            parameters.append(dict(identity, parameter=name, value=value,
                search_gate_status='closed' if name in inactive else 'not_flagged',
                reason=inactive.get(name, 'Joint identifiability is not assessed by this audit')))
    return parameters, summaries, inputs


def run(fits: list[Path], output: Path) -> None:
    """Write parameter flags, fit summaries, and reproducible source provenance."""
    if output.exists():
        raise FileExistsError('Use a new audit output directory')
    if not fits or len({path.resolve() for path in fits}) != len(fits):
        raise ValueError('Need distinct completed fit directories')
    parameters, nominees, inputs = [], [], {}
    for fit in fits:
        rows, summary, hashes = nomination_rows(fit)
        parameters.extend(rows); nominees.extend(summary); inputs.update(hashes)
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in (('parameters.csv', parameters), ('nominees.csv', nominees)):
        path = output / name
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        artifacts[name] = digest(path)
    source = Path(__file__).resolve()
    (output / 'audit_source.py').write_bytes(source.read_bytes())
    artifacts['audit_source.py'] = digest(output / 'audit_source.py')
    manifest = dict(status='complete', protocol='model0923_search_gate_audit_v1',
        nominees=len(nominees), parameter_rows=len(parameters),
        flagged_nominees=sum(row['closed_search_coordinates'] > 0 for row in nominees),
        input_sha256=inputs, artifact_sha256=artifacts, python=sys.version,
        source_sha256={str(source.relative_to(ROOT)): digest(source)},
        scope='Exact search-channel closure conditional on the other selected coordinates; '
              'not a full identifiability test, confidence interval, or parameter recovery',
        behavior_selection_changed=False, oral_read=False, raw_data_read=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f"{len(nominees)} nominees; {manifest['flagged_nominees']} with exactly closed search channels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.fit, args.output)


if __name__ == '__main__':
    main()
