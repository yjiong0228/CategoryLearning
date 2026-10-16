"""Joint precision/H refits with matched candidate and particle budgets.

Four families separate additive learning-rate range, log coordinates, and the
precision ceiling. All use FREE H retention. Warm starts and beta-rate rays
precede a joint refinement of all fourteen coordinates. Selection uses only
training pre-choice probabilities. New destinations retain all source hashes,
filter repetitions and path arrays. All nine subjects are development data.

ADD10 stores a normalized learning rate in [0,1]; its effective additive rate
is ten times that coordinate. The existing experimental adapter implements
this exact parameterization. Default model configuration is unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]
VALIDATION = 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
PRECISION = 'src/Bayesian_state/model/modules/precision_interventions.py'
FEEDBACK = 'src/Bayesian_state/model/modules/hypothesis_transition/feedback_guided.py'
RATE = 'engine.modules.beta_mod.kwargs.learning_rate'
INIT = 'engine.modules.beta_mod.kwargs.beta_init'
DECAY = 'engine.modules.hypo_transitions_mod.kwargs.intervention.decay'
MEMORY = 'engine.modules.memory_mod.kwargs.gamma'


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validation():
    return load('precision_refit_validation', VALIDATION)


def settings_for_scoring(settings: dict) -> dict:
    return dict(settings, models={arm: settings['mechanism'] for arm in settings['arms']})


def model_space(settings: dict, arm: str) -> dict:
    v = validation()
    space, _ = v.model_space(settings_for_scoring(settings), arm)
    space[RATE] = settings['arms'][arm]['rates']
    return space


def project(point: dict, old_model: str, arm: str, settings: dict) -> dict:
    """Map warm starts explicitly; log-rate matching is local at beta_init."""
    point = deepcopy(point)
    if old_model == 'SHARED':
        point[DECAY] = point[MEMORY]
    if arm == 'ADD10':
        point[RATE] /= 10.
    elif arm.startswith('LOG'):
        # d beta ~= eta_log * beta^2 * gradient for a small log step.
        value = point[RATE] / point[INIT] ** 2
        rates = settings['arms'][arm]['rates']
        point[RATE] = min(rates, key=lambda r: (abs(r-value), r))
    space = model_space(settings, arm)
    if set(point) != set(space) or any(point[k] not in values for k, values in space.items()):
        raise ValueError('Warm start lies outside declared support')
    return point


def starting_banks(settings: dict, frozen: dict, smoke: bool) -> tuple[dict, dict]:
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, initial_points
    from src.Bayesian_state.utils.seeding import stable_seed
    v = validation(); banks, required = {}, {}
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    count = 5 if smoke else settings['initial_candidates']
    for subject in subjects:
        prior = {r['model']: r['candidate']['hyperparams'] for r in frozen['nominees'] if r['subject'] == subject}
        common = model_space(settings, 'ADD1')
        anchor = project(prior['FREE'], 'FREE', 'ADD1', settings)
        starts = initial_points(common, anchor, count+16,
            stable_seed(dict(base=settings['seed'], subject=subject, role='shared_initial')),
            cover_key=v.H_PATH+'max_capacity')
        for arm in settings['arms']:
            rows = {}
            for old in ('FREE', 'CUMULATIVE', 'SHARED'):
                row = candidate(project(prior[old], old, arm, settings), 'prior_training_nominee', source_model=old)
                rows.setdefault(row['id'], row)
                if old == 'FREE': required[subject, arm] = row
            anchor = required[subject, arm]['hyperparams']
            # Give each learning-rate level a test at the same old FREE point.
            for rate in settings['arms'][arm]['rates']:
                row = candidate(dict(anchor, **{RATE: rate}), 'precision_rate_ray')
                rows.setdefault(row['id'], row)
                if len(rows) >= count: break
            for i, point in enumerate(starts):
                if len(rows) >= count: break
                point = project(point, 'FREE', arm, settings)
                point[RATE] = settings['arms'][arm]['rates'][i % len(settings['arms'][arm]['rates'])]
                row = candidate(point, 'matched_common_coordinate_start')
                rows.setdefault(row['id'], row)
            if len(rows) != count: raise ValueError('Unequal initial budget')
            banks[subject, arm] = list(rows.values())
    return banks, required


def source_hashes(v) -> dict:
    hashes = v.source_hashes()
    names = [str(Path(__file__).resolve().relative_to(REPO)), PRECISION, FEEDBACK,
        'src/Bayesian_state/workflows/analysis/model_0923_h_training_audit.py',
        'src/Bayesian_state/workflows/analysis/model_0923_mechanism_interventions.py',
        'src/Bayesian_state/workflows/analysis/model_0923_cohort_path_diagnosis.py',
        'src/Bayesian_state/optimization/search/adaptive_proposals.py']
    for name in names: hashes[name] = v.digest(REPO/name)
    return hashes


def worker(file: Path) -> None:
    v = validation(); job = v.read(file)
    workflow = v.intervention_workflow(); workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', FEEDBACK)
    precision = load('src.Bayesian_state.model.modules.precision_interventions', PRECISION)
    from src.Bayesian_state.model.modules.beta import BetaModule
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    arm = job['arm_settings']
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        corrections = stack.enter_context(feedback.retention_roundoff_guard(cls))
        stack.enter_context(precision.precision_intervention(BetaModule, mode=arm['precision'], upper=arm['upper']))
        (v.autonomous if job['phase'] == 'autonomous' else v.score)(job, workflow)
        v.write(Path(job['target']).with_suffix('.experiment.json'), dict(arm_settings=arm,
            configured_rate=job['candidate']['hyperparams'][RATE],
            effective_rate=job['candidate']['hyperparams'][RATE]*arm['rate_scale'], corrections=corrections))


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = validation(); directory = output/'jobs'; directory.mkdir(exist_ok=True)
    def launch(job):
        file = directory/(job['name']+'.json'); v.write(file, job)
        launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='precision_refit_') as cache:
            env['NUMBA_CACHE_DIR'] = cache
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                    '--worker', str(file)], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')
    workers = parallel_job_count(budget, len(jobs))
    print(f'{len(jobs)} jobs; {workers} workers; numeric threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, j) for j in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 32 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete', flush=True)


def stage(settings: dict, data: Path, output: Path, banks: dict, name: str, smoke: bool) -> dict:
    from src.Bayesian_state.workflows.analysis.model_0923_h_training_audit import combine
    v = validation(); jobs = v.score_jobs(data, output, settings_for_scoring(settings), banks, name, smoke)
    groups, chunks = {}, []
    for job in jobs:
        subject = v.read(Path(job['folder'])/'prepared.json')['subject']
        if name in ('selection', 'confirmation') and not smoke:
            job['particles'] = settings['particle_overrides'].get(subject, job['particles'])
        job.update(arm_settings=settings['arms'][job['model']], source_sha256=source_hashes(v))
        groups[job['name']] = []
        width = len(job['seeds']) if name in ('screen', 'refine') else 1
        for i in range(0, len(job['seeds']), width):
            chunk = deepcopy(job); target = Path(job['target']); suffix = f'_C{i:03d}'
            chunk.update(name=job['name']+suffix, seeds=job['seeds'][i:i+width],
                         target=str(target.parent/'chunks'/(target.name+suffix)))
            groups[job['name']].append(chunk); chunks.append(chunk)
    lengths = {j['folder']: v.read(Path(j['folder'])/'prepared.json')['training_trials'] for j in chunks}
    chunks.sort(key=lambda j: -lengths[j['folder']]*j['particles']*len(j['seeds']))
    started = time.monotonic(); execute(chunks, output, 1 if smoke else settings['parallel_budget'])
    scored = {key: [] for key in banks}
    for job in jobs:
        subject = v.read(Path(job['folder'])/'prepared.json')['subject']
        scored[subject, job['model']].append(combine(job, groups[job['name']]))
    for rows in scored.values(): rows.sort(key=lambda r: (r['mean_nll'], r['id']))
    v.write(output/(name+'_completion.json'), dict(status='complete', stage=name, points=len(jobs),
        seed_jobs=len(chunks), seconds=time.monotonic()-started,
        groups=[dict(subject=k[0], model=k[1], rows=rows) for k, rows in scored.items()]))
    return scored


def shortlist(rows: list[dict], required: dict, count: int) -> list[dict]:
    selected = {required['id']: required}
    for row in rows:
        if len(selected) == count: break
        selected.setdefault(row['id'], row)
    if len(selected) != count: raise ValueError('Insufficient finalists')
    return list(selected.values())


def archive(output: Path, config: Path, hashes: dict) -> None:
    shutil.copy2(config, output/'config.yaml')
    for name in hashes:
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target)


def search(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.optimization.search.adaptive_proposals import propose_round, select_elites
    from src.Bayesian_state.utils.seeding import stable_seed
    v = validation(); settings = yaml.safe_load(config.read_text())
    data = REPO/settings['data']; frozen_file = REPO/settings['warm_start_fit']/'nominees.json'
    frozen = v.read(frozen_file)
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need scientific warm starts')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO/name) != sha: raise ValueError('Warm-start source changed')
    if output.exists(): raise FileExistsError('Use a new precision-refit directory')
    banks, required = starting_banks(settings, frozen, smoke)
    seen = {key: {r['id'] for r in rows} for key, rows in banks.items()}
    output.mkdir(parents=True); hashes = source_hashes(v); archive(output, config, hashes)
    v.write(output/'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        warm_start_nomination_sha256=v.digest(frozen_file), data_manifest_sha256=v.digest(data/'preparation_manifest.json'),
        data_boundary='training only; all subjects are development participants',
        initial_banks=[dict(subject=k[0], model=k[1], rows=rows) for k, rows in banks.items()],
        python=sys.version, versions={k: importlib.metadata.version(k) for k in ('numpy','scipy','pandas','PyYAML')}))
    rows = stage(settings, data, output, banks, 'screen', smoke)
    for key in banks:
        elites = select_elites(rows[key], 1 if smoke else settings['elites'], exclude_key=v.H_PATH+'max_capacity')
        proposals = propose_round(elites, model_space(settings, key[1]), seen[key],
            1 if smoke else settings['proposals_per_elite'],
            stable_seed(dict(base=settings['seed'], subject=key[0], role='joint_refine')), .5, .25)
        expected = 1 if smoke else settings['elites']*settings['proposals_per_elite']
        if len(proposals) != expected: raise ValueError('Unequal joint refinement budget')
        banks[key] = proposals + elites; seen[key].update(r['id'] for r in proposals)
    rows = stage(settings, data, output, banks, 'refine', smoke)
    # Carry the old FREE point through both high-budget stages to protect the
    # reference from noisy early elimination. All families receive this rule.
    for name, count in [('selection', settings['selection_candidates']), ('confirmation', settings['confirmation_candidates'])]:
        banks = {k: shortlist(r, required[k], 2 if smoke else count) for k, r in rows.items()}
        rows = stage(settings, data, output, banks, name, smoke)
    nominees = [dict(subject=k[0], model=k[1], candidate=r[0], finalists=r,
                     unique_candidates=len(seen[k]), reference_id=required[k]['id']) for k, r in rows.items()]
    v.write(output/'nominees.json', dict(status='frozen_before_test', settings=settings, smoke=smoke,
        source_sha256=hashes, data=str(data.resolve()), nominees=nominees))
    v.write(output/'completion.json', dict(status='complete', test_choices_used=False, smoke=smoke,
        nomination_sha256=v.digest(output/'nominees.json')))
    print('All precision nominees frozen before evaluation', flush=True)


def evaluate(fit: Path, output: Path) -> None:
    from src.Bayesian_state.utils.seeding import stable_seed
    v = validation(); frozen = v.read(fit/'nominees.json'); settings = frozen['settings']
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need scientific frozen fits')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO/name) != sha: raise ValueError('Fitted implementation changed')
    if output.exists(): raise FileExistsError('Use a new precision evaluation directory')
    banks = {(r['subject'],r['model']):[r['candidate']] for r in frozen['nominees']}
    groups, chunks = {}, []
    for role in ('conditional','autonomous'):
        jobs = v.score_jobs(Path(frozen['data']), output/role, settings_for_scoring(settings), banks, 'evaluate', False)
        groups[role] = []
        for job in jobs:
            subject = v.read(Path(job['folder'])/'prepared.json')['subject']
            overrides = {int(s): p for s,p in settings['particle_overrides'].items()}
            job.update(arm_settings=settings['arms'][job['model']], source_sha256=frozen['source_sha256'],
                       particles=overrides.get(subject,job['particles']))
            if role == 'autonomous':
                job['phase'] = 'autonomous'; job['name'] = 'AUTO_'+job['name']
                job['seeds'] = [stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],
                    role='autonomous',subject=subject,path=i)) for i in range(settings['autonomous_paths'])]
            width = settings['autonomous_chunk'] if role == 'autonomous' else 1
            own = []
            for i in range(0,len(job['seeds']),width):
                part = deepcopy(job); target = Path(job['target']); suffix = f'_C{i:03d}'
                part.update(name=job['name']+suffix,seeds=job['seeds'][i:i+width],
                            target=str(target.parent/'chunks'/(target.name+suffix)))
                own.append(part); chunks.append(part)
            groups[role].append((job,own))
    lengths = {j['folder']:v.read(Path(j['folder'])/'prepared.json')['trials'] for j in chunks}
    chunks.sort(key=lambda j:-lengths[j['folder']]*len(j['seeds'])*(1 if j['phase']=='autonomous' else j['particles']))
    output.mkdir(parents=True)
    v.write(output/'protocol.json',dict(fit=str(fit.resolve()),nomination_sha256=v.digest(fit/'nominees.json'),
        source_sha256=frozen['source_sha256'],data_use='development; all parameters frozen before test opening',jobs=chunks))
    execute(chunks, output, settings['parallel_budget'])
    for role, jobs in groups.items():
        rows = [v.combine_evaluation_chunks(job, parts, role=='autonomous') for job,parts in jobs]
        v.write(output/role/'evaluation.json',dict(status='complete',rows=rows))
    v.write(output/'completion.json',dict(status='complete',jobs=len(chunks),
        nomination_sha256=v.digest(fit/'nominees.json')))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path); p.add_argument('--phase',choices=('search','evaluate'))
    for name in ('config','fit','output'): p.add_argument('--'+name,type=Path)
    p.add_argument('--smoke',action='store_true'); a = p.parse_args()
    if a.worker: worker(a.worker)
    elif a.phase=='search' and a.config and a.output: search(a.config,a.output,a.smoke)
    elif a.phase=='evaluate' and a.fit and a.output and not a.smoke: evaluate(a.fit,a.output)
    else: p.error('Use --phase search --config YAML --output NEW [--smoke], or --phase evaluate --fit FIT --output NEW')


if __name__ == '__main__': main()
