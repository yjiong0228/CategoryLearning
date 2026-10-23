"""Cross two frozen parameter points with outcome-dependent belief retention.

Run --config YAML --output NEW [--smoke]. SUPPORT, SLOT and OUTCOME reuse the
shared numerical workers, likelihood and autonomous simulator. OUTCOME adds
outcome-gated rule forgetting to SLOT. Each mechanism receives both unchanged
parameter points, matched seeds and equal simulation budgets. This is a
development diagnosis, not independent refitting or held-out model validation.
The optional five-mechanism protocol adds ENTRY (error-dependent admission)
and ADAPT (both outcome retention and error-dependent admission). This gives
the two-by-two belief-mechanism comparison plus the SUPPORT reference.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[4]
PILOT = 'src/Bayesian_state/workflows/analysis/model_1022_belief_comparison.py'
spec = importlib.util.spec_from_file_location('belief_outcome_probe_pilot', ROOT / PILOT)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
read, write, digest = pilot.read, pilot.write, pilot.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
ADAPTER = 'src/Bayesian_state/model/modules/hypothesis_transition/belief_outcome_retention.py'
ENTRY_ADAPTER = 'src/Bayesian_state/model/modules/hypothesis_transition/belief_error_entry.py'
MECHANISMS = ('SUPPORT', 'SLOT', 'OUTCOME', 'ENTRY', 'ADAPT')


def sources(v) -> dict:
    return dict(pilot.sources(v), **{name: digest(ROOT / name) for name in (ENTRY, ADAPTER, ENTRY_ADAPTER)})


def worker(file: Path) -> None:
    job = read(file)
    variant = job['belief_variant']
    expected = 'SUPPORT' if variant == 'SUPPORT' else 'SLOT'
    if (variant not in MECHANISMS or job['comparison_arm'] != expected
            or job['source_sha256'] != sources(pilot.driver.io)):
        raise ValueError('Outcome probe mechanism or scientific source differs')
    original_load = pilot.driver.base.load

    def load(name, path):
        module = original_load(name, path)
        if variant in ('OUTCOME', 'ADAPT') and path == pilot.ADAPTER:
            outcome = original_load('src.Bayesian_state.model.modules.hypothesis_transition.belief_outcome_retention',
                                    ADAPTER)
            original = module.compact_belief

            @contextmanager
            def install(cls):
                with original(cls) as audit, outcome.belief_outcome_retention(cls) as retention:
                    audit['outcome_retention'] = retention
                    yield audit

            module.compact_belief = install
        if variant in ('ENTRY', 'ADAPT') and path == pilot.SLOT:
            admission = original_load('src.Bayesian_state.model.modules.hypothesis_transition.belief_error_entry',
                                      ENTRY_ADAPTER)
            original_slot = module.belief_slot_transfer

            @contextmanager
            def install_entry(cls):
                with original_slot(cls) as audit, admission.belief_error_entry(cls) as entries:
                    audit['error_entry'] = entries
                    yield audit

            module.belief_slot_transfer = install_entry
        return module

    pilot.driver.base.load = load
    try:
        pilot.worker(file)
    finally:
        pilot.driver.base.load = original_load


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    """Keep the visible cell and parameter identity while routing its modules."""
    mapped = deepcopy(jobs)
    for job in mapped:
        variant = job['model'].split('_AT_')[0]
        if variant not in MECHANISMS:
            raise ValueError('Unknown outcome-probe mechanism')
        job.update(diagnosis_cell=job['model'], belief_variant=variant,
                   model='SUPPORT' if variant == 'SUPPORT' else 'SLOT')
    original_entry = pilot.ENTRY
    try:
        pilot.ENTRY = ENTRY
        pilot.execute(mapped, output, budget)
    finally:
        pilot.ENTRY = original_entry


def run(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.workflows.analysis.model_1022_compact_evaluate import smoke_fixture
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new outcome-probe directory')
    policy = yaml.safe_load(config.read_text())
    if (policy['scope'] != 'fixed_parameter_outcome_retention_development_diagnosis'
            or policy['mechanisms'] not in (list(MECHANISMS[:3]), list(MECHANISMS))
            or len(policy['points']) != 2
            or not 1 <= policy['parallel_budget'] <= 128):
        raise ValueError('Require the registered six-cell or ten-cell development diagnosis')
    nominees, templates, inputs, settings = [], {}, {}, None
    for index, (label, point) in enumerate(policy['points'].items()):
        folder = ROOT / point['study']
        protocol, frozen = read(folder / 'protocol.json'), read(folder / 'nominees.json')
        if (read(folder / 'completion.json')['status'] != 'complete' or frozen['smoke']
                or frozen['status'] != 'frozen_before_test' or frozen['later_choices_used']
                or read(folder / 'summary/completion.json')['nominee_sha256'] != digest(folder / 'nominees.json')):
            raise ValueError('Probe points require intact frozen training nominations')
        for name in ('protocol.json', 'nominees.json', 'completion.json', 'summary/completion.json',
                     'autonomous_jobs.json'):
            inputs[str((folder / name).resolve())] = digest(folder / name)
        if settings is None:
            settings = deepcopy(protocol['settings'])
        elif settings['data'] != protocol['settings']['data']:
            raise ValueError('Parameter sources must use the same prepared data')
        for sid in policy['subjects']:
            selected, = [r for r in frozen['nominees'] if r['subject'] == sid and r['arm'] == point['arm']]
            template = next(j for j in read(folder / 'autonomous_jobs.json')['jobs']
                            if j['subject'] == sid and j['model'] == point['arm'])
            for mechanism in policy['mechanisms']:
                cell = 'SUPPORT' if index == 0 and mechanism == 'SUPPORT' else f'{mechanism}_AT_{label}'
                nominees.append(dict(subject=sid, arm=cell, mechanism=mechanism, parameter_source=point,
                                     candidate=deepcopy(selected['candidate'])))
                templates[sid, cell] = deepcopy(template)
    settings.update({k: deepcopy(policy[k]) for k in ('protocol', 'seed', 'subjects', 'parallel_budget',
                                                     'budgets', 'autonomous_paths', 'autonomous_chunk')})
    settings['arms'] = {row['arm']: deepcopy(templates[row['subject'], row['arm']]['arm_settings'])
                        for row in nominees}
    output.mkdir(parents=True)
    if smoke:
        if len(settings['subjects']) != 1:
            raise ValueError('Software validation uses one subject')
        settings['parallel_budget'] = 1
        settings['data'] = str(smoke_fixture(ROOT / settings['data'], output / 'software_fixture',
                                           settings['subjects'][0]).resolve())
    pilot.driver.base.source_hashes, pilot.driver.base.execute = sources, execute
    pilot.driver.base.archive(output, config, sources(pilot.driver.io))
    write(output / 'protocol.json', dict(settings=settings, policy=policy, smoke=smoke,
        source_sha256=sources(pilot.driver.io), inputs_sha256=inputs,
        python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        development_later_results_informed_hypothesis=True, parameters_refitted=False,
        decision='All fixed-parameter cells declared before simulation; no candidate selection'))
    write(output / 'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
                                        later_choices_used=False, oral_used=False))
    pilot.driver.base.stage(settings, ROOT / settings['data'], output,
        {(r['subject'], r['arm']): [r['candidate']] for r in nominees}, 'evaluate', smoke)
    jobs = []
    for row in nominees:
        sid, cell = row['subject'], row['arm']
        seeds = [stable_seed(dict(base=policy['seed'], subject=sid, stage='autonomous', repeat=i))
                 for i in range(4 if smoke else policy['autonomous_paths'])]
        for i in range(0, len(seeds), policy['autonomous_chunk']):
            name = f'autonomous_S{sid}_{cell}_{i:03d}'
            job = deepcopy(templates[sid, cell])
            job.update(name=name, model=cell, candidate=row['candidate'],
                       folder=str((ROOT / settings['data'] / f'subject_{sid}').resolve()),
                       seeds=seeds[i:i+policy['autonomous_chunk']], smoke=smoke,
                       source_sha256=sources(pilot.driver.io), target=str((output / 'autonomous' / name).resolve()))
            jobs.append(job)
    write(output / 'autonomous_jobs.json', dict(jobs=jobs))
    execute(jobs, output, settings['parallel_budget'])
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Frozen parameter source changed during the probe')
    if smoke:
        write(output / 'completion.json', dict(status='complete', smoke=True, cases=len(nominees),
            scope='Eight evaluation trials and sixteen autonomous trials; software only'))
    else:
        pilot.summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Use --config and --output, or an internal --worker')
