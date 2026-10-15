"""Fit and evaluate explicit H candidates in isolated archived workers.

The existing H validation workflow owns parameter proposals, training scores,
chronological boundaries and prediction assembly. This adapter installs the
explicitly named models' set sampler and enforces fixed parameter values. A YAML
``fixed_size_models`` list identifies those models; each must use J with
refresh='search' and candidate_sampler='single_proposal_fixed_cardinality'.
The sampler marker makes unadapted workers reject that configuration. This
adapter consumes the marker after installing the named sampler. All other
models use the original sampler.

Optional ``fixed_parameters`` maps model names to supported parameter values.
Those coordinates have singleton search support, remain in every saved point,
and are checked again in each worker. This supports refitting an exit-free
mechanism with exit_rate=0 while fitting its other parameters normally.

Examples, after freezing a candidate protocol:
  python -m src.Bayesian_state.workflows.analysis.model_0923_h_candidate_validation \
    --phase search --config CONFIG --data PREPARED_DATA --output NEW_FIT
  python -m src.Bayesian_state.workflows.analysis.model_0923_h_candidate_validation \
    --phase audit --config AUDIT_CONFIG --fit NEW_FIT --output NEW_AUDIT
  python -m src.Bayesian_state.workflows.analysis.model_0923_h_candidate_validation \
    --phase evaluate --fit NEW_AUDIT --output NEW_PREDICTIONS

Use --phase autonomous for complete generated paths. Preparation remains in
the original validation entry point. Every destination must be new. Smoke
searches use one participant, eight training trials and one worker. The adapter
does not select candidate mechanisms or open reserved participants by itself.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[4]
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
HELPER = 'src/Bayesian_state/model/modules/hypothesis_transition/single_proposal_competition.py'
PROPOSALS = 'src/Bayesian_state/optimization/search/adaptive_proposals.py'
SAMPLER = 'single_proposal_fixed_cardinality'


def validation():
    spec = importlib.util.spec_from_file_location('candidate_validation_base', REPO / BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_models(settings: dict) -> set[str]:
    """Validate the declared sampler intervention without changing H settings."""
    names = settings.get('fixed_size_models', [])
    if (not names and not settings.get('fixed_parameters')) or len(names) != len(set(names)) or not set(names) <= set(settings['models']):
        raise ValueError('Need distinct fixed-size model names from the model protocol')
    for name in names:
        mechanism = settings['models'][name]
        if mechanism.get('kind') != 'J' or mechanism.get('refresh') != 'search':
            raise ValueError('Fixed-cardinality sampling requires J with refresh=search')
        if mechanism.get('candidate_sampler') != SAMPLER:
            raise ValueError('Fixed-cardinality model needs its explicit sampler marker')
    if any('candidate_sampler' in mechanism for name, mechanism in settings['models'].items() if name not in names):
        raise ValueError('Sampler marker belongs only to declared fixed-size models')
    return set(names)


def constrained_space(base_space, settings: dict, model: str) -> tuple[dict, dict]:
    """Keep fixed coordinates explicit and validate them against original support."""
    fixed = settings.get('fixed_parameters', {})
    if not isinstance(fixed, dict) or not set(fixed) <= set(settings['models']):
        raise ValueError('Fixed parameters must name declared models')
    space, anchor = deepcopy(base_space(settings, model))
    values = fixed.get(model, {})
    if not isinstance(values, dict) or not set(values) <= set(space):
        raise ValueError('Fixed parameters must name active search coordinates')
    for key, value in values.items():
        if isinstance(value, bool) or value not in space[key]:
            raise ValueError('Fixed parameter value is outside declared support')
        space[key] = [value]; anchor[key] = value
    return space, anchor


def core_job(job: dict) -> dict:
    """Consume the verified workflow marker before calling the shared model."""
    result = deepcopy(job)
    marker = result['mechanism'].pop('candidate_sampler', None)
    expected = SAMPLER if job['candidate_sampler'] == SAMPLER else None
    if marker != expected:
        raise ValueError('Declared and installed candidate samplers differ')
    for key, value in job.get('fixed_parameters', {}).items():
        if job['candidate']['hyperparams'].get(key) != value:
            raise ValueError('Candidate violates the frozen parameter constraint')
    return result


def worker(file: Path) -> None:
    v = validation(); job = v.read(file)
    for name, sha in job['source_sha256'].items():
        if v.digest(REPO / name) != sha:
            raise ValueError('Candidate validation source changed')
    sampler = job['candidate_sampler']
    if sampler not in ('original', 'single_proposal_fixed_cardinality'):
        raise ValueError('Unknown candidate sampler')
    original_workflow = v.intervention_workflow

    def workflow():
        w = original_workflow()
        if sampler == 'single_proposal_fixed_cardinality':
            fixed_models(dict(models={job['model']: job['mechanism']}, fixed_size_models=[job['model']]))
            original_install = w.install_transition

            def install():
                original_install()
                helper = w.load_source('src.Bayesian_state.model.modules.hypothesis_transition.single_proposal_competition', HELPER)
                helper.install_worker_intervention(sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'])

            w.install_transition = install
        return w

    v.intervention_workflow = workflow
    original_score, original_autonomous = v.score, v.autonomous
    v.score = lambda job, w: original_score(core_job(job), w)
    v.autonomous = lambda job, w: original_autonomous(core_job(job), w)
    v.worker(file)
    target = Path(job['target']); checks = target.parent / 'sampler_checks'
    checks.mkdir(exist_ok=True)
    v.write(checks / (target.name + '.json'), dict(model=job['model'], sampler=sampler,
        prediction_sha256=v.digest(target.with_suffix('.npz')), source_sha256=job['source_sha256']))


def configure(v, settings: dict) -> None:
    """Route existing search/evaluation operations through the recorded worker."""
    names = fixed_models(settings)
    original_hashes = v.source_hashes
    if settings.get('fixed_parameters'):
        original_space = v.model_space
        for model in settings['models']:
            constrained_space(original_space, settings, model)
        v.model_space = lambda config, model: constrained_space(original_space, config, model)

    def hashes():
        result = original_hashes()
        for file in (REPO / HELPER, REPO / PROPOSALS, Path(__file__).resolve()):
            result[str(file.relative_to(REPO))] = v.digest(file)
        return result

    def execute(jobs: list[dict], output: Path, budget: int) -> None:
        from src.Bayesian_state.utils.parallel import parallel_job_count
        workers = parallel_job_count(budget, len(jobs))
        directory = output / 'jobs'; directory.mkdir(exist_ok=True)

        def launch(original):
            job = deepcopy(original)
            job['candidate_sampler'] = 'single_proposal_fixed_cardinality' if job['model'] in names else 'original'
            job['fixed_parameters'] = deepcopy(settings.get('fixed_parameters', {}).get(job['model'], {}))
            core_job(job)
            file = directory / (job['name'] + '.json'); v.write(file, job)
            launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                        'sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            with tempfile.TemporaryDirectory(prefix='h_candidate_validation_') as cache, file.with_suffix('.log').open('x') as log:
                env['NUMBA_CACHE_DIR'] = cache
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                    '--worker', str(file.resolve())], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Candidate worker failed: {file.with_suffix(".log")}')

        print(f'Running {len(jobs)} candidate jobs with {workers} workers, numeric threads=1', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(launch, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 16 == 0 or index == len(jobs):
                    print(f'{index}/{len(jobs)} complete', flush=True)

    v.source_hashes = hashes
    v.execute = execute


def main() -> None:
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['search', 'audit', 'evaluate', 'autonomous'])
    for name in ('config', 'data', 'fit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker); return
    if args.output is None or args.phase is None:
        parser.error('--phase and --output are required')
    if args.phase == 'search' and (args.config is None or args.data is None):
        parser.error('search requires --config and --data')
    if args.phase != 'search' and args.fit is None:
        parser.error('audit/evaluation require --fit')
    if args.phase == 'audit' and args.config is None:
        parser.error('audit requires --config')
    v = validation()
    settings = yaml.safe_load(args.config.read_text()) if args.phase == 'search' else v.read(args.fit / 'nominees.json')['settings']
    configure(v, settings)
    if args.phase == 'search':
        v.search(args.config, args.data, args.output, args.smoke)
    elif args.phase == 'audit':
        from src.Bayesian_state.workflows.analysis import model_0923_h_training_audit as audit
        audit.execute = v.execute; audit.score_jobs = v.score_jobs; audit.source_hashes = v.source_hashes
        audit.run(args.fit, args.config, args.output, args.smoke)
    else:
        v.evaluate(args.fit, args.output, args.phase == 'autonomous')


if __name__ == '__main__':
    main()
