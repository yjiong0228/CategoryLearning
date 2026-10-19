"""Summarize a completed SEPARATE refit with the existing behavioral scores.

Run --fit FIT --evaluation EVALUATION --output NEW_DIRECTORY. This adapter
checks actual belief/precision execution for every training and evaluation
chunk, excludes belief sidecars from computational budgets, and identifies the
fixed affinity-transfer coordinate. The original NLL, Energy, path summaries
and Monte Carlo errors are reused unchanged. The generic manifest is retained;
separate_manifest.json binds this adapter and the additional execution audit.
Oral validation is a subsequent, separate analysis of the frozen nominees.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import model_0923_episodic_refit as base
from .model_0923_h_validation import load, digest, inactive_parameter_channels
from ..workflows.analysis.model_0923_h_validation import write
from ..workflows.analysis.model_0923_online_parameter_cross import check_chunk

ROOT = Path(__file__).resolve().parents[3]
INTERVENTION = 'separate_gamma_belief_evidence'
PREFIX = 'engine.modules.hypo_transitions_mod.kwargs.'
TRANSFER = PREFIX + 'transfer_scale'
GAMMA = 'engine.modules.memory_mod.kwargs.gamma'
SIDECARS = ('.experiment.json', '.precision.json', '.online.json', '.belief.json')


class _TrainingResults:
    """Give the frozen budget reader only its supported result files."""

    def __init__(self, stage: Path):
        self.stage, self.name = stage, stage.name

    def glob(self, pattern: str):
        return (file for file in self.stage.glob(pattern)
                if not file.name.endswith('.belief.json'))


def training_chunk_budgets(stage: Path) -> list[dict]:
    return base.training_chunk_budgets(_TrainingResults(stage))


def refit_inactive_channels(frozen: dict) -> pd.DataFrame:
    """Record exact closures; these flags do not establish identifiability."""
    settings = frozen['settings']
    if settings.get('belief_intervention') != INTERVENTION:
        raise ValueError('Require SEPARATE belief evidence')
    if (settings['mechanism'].get('capacity') != 'evidence_mean'
            or settings['mechanism'].get('kind') != 'E'):
        raise ValueError('Require the executed common evidence admission policy')
    rows = inactive_parameter_channels(frozen)
    added = [dict(subject=n['subject'], model=n['model'], parameter=TRANSFER,
                  value=n['candidate']['hyperparams'][TRANSFER],
                  reason='SEPARATE按γ保持的反馈证据重建规则信念；亲和度转移通道关闭，固定值用于复现')
             for n in frozen['nominees']]
    added_frame = pd.DataFrame(added)
    return added_frame if rows.empty else pd.concat([rows, added_frame], ignore_index=True)


def check_execution(job: dict, result: dict, belief: dict, precision: dict) -> None:
    """Validate the intervention identity and completed-feedback update count."""
    point = job['candidate']['hyperparams']
    copies = len(job['seeds']) * (1 if job['phase'] == 'autonomous' else job['particles'])
    if (job.get('belief_intervention') != INTERVENTION
            or job.get('probe_mechanism') != 'SEPARATE' or job.get('online_belief') is not False
            or job.get('fixed_coordinates') != {TRANSFER: .01} or point.get(TRANSFER) != .01
            or job.get('structural_intervention') != 'retained_rule_precision'
            or job.get('retain_precision') is not True
            or precision.get('structural_intervention') != 'retained_rule_precision'
            or precision.get('retained_precision') is not True
            or precision.get('transitions') != copies * result['trials']
            or belief.get('enabled') is not True or belief.get('belief_intervention') != INTERVENTION
            or belief.get('gamma') != point[GAMMA]
            or belief.get('delta') != [point[PREFIX + 'intervention.decay']]
            or belief.get('completed_feedback_timing_exact') is not True
            or belief.get('belief_matches_saved_state') is not True
            or belief.get('updates') != copies * (result['trials'] - 1)):
        raise ValueError('SEPARATE execution, memory rates or feedback timing differ')


def check_training_arrays(job: dict, result: dict, arrays: dict) -> None:
    """Training stores observed/correct probabilities, without category arrays."""
    if (any(result[k] != job[k] for k in ('candidate', 'model', 'seeds', 'particles'))
            or any(pool['num_threads'] != 1 for pool in result['numeric_pools'])):
        raise ValueError('Training parameters, seeds or computational budget changed')
    np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
    shape = len(job['seeds']), result['trials']
    for key in ('observed_probability', 'correct_probability'):
        values = arrays[key]
        if (values.shape != shape or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()):
            raise ValueError('Invalid saved training probabilities')
    np.testing.assert_allclose(base.nll(arrays['observed_probability']), result['mean_nll'],
                               rtol=0, atol=1e-12)


def audit_jobs(source: Path, *, smoke: bool = False) -> dict:
    """Audit declared workers, prediction arrays, tags and source bindings."""
    files = sorted((source / 'jobs').glob('*.json'))
    if not files:
        raise ValueError('No executed SEPARATE jobs')
    inputs, targets, rows = {}, set(), []

    def bind(path):
        path = Path(path).resolve()
        if str(path) not in inputs:
            inputs[str(path)] = digest(path)
        return path

    validation = SimpleNamespace(read=load, digest=digest)
    for file in files:
        job = load(bind(file)); target = Path(job['target']).resolve()
        if target in targets or job.get('smoke') is not smoke:
            raise ValueError('Repeated job target or changed scientific scope')
        targets.add(target)
        if job['phase'] not in ('training', 'evaluate', 'autonomous'):
            raise ValueError('Unknown job role')
        for name, sha in job['source_sha256'].items():
            path = bind(ROOT / name)
            if inputs[str(path)] != sha:
                raise ValueError('Executed source changed')
        if job['phase'] == 'training':
            result = load(bind(target.with_suffix('.json')))
            path = bind(target.with_suffix('.npz'))
            if digest(path) != result['predictions_sha256']:
                raise ValueError('Training arrays changed')
            bind(target.with_suffix('.experiment.json'))
            with np.load(path) as arrays:
                check_training_arrays(job, result, arrays)
        else:
            # Evaluation stores complete category predictions and path states.
            result, _ = check_chunk(job, 'SEPARATE', validation, bind)
        belief = load(bind(target.with_suffix('.belief.json')))
        precision = load(bind(target.with_suffix('.precision.json')))
        check_execution(job, result, belief, precision)
        expected_role = 'autonomous_from_start' if job['phase'] == 'autonomous' else job['phase']
        if result['role'] != expected_role:
            raise ValueError('Result role differs from declared job')
        copies = len(job['seeds']) * (1 if job['phase'] == 'autonomous' else job['particles'])
        rows.append(dict(phase=job['phase'], model=result['model'], subject=result['subject'],
                         state_updates=copies * result['trials'], runs=len(job['seeds'])))
    actual = {f.with_suffix('').resolve() for f in source.glob('**/chunks/*.json')
              if not f.name.endswith(SIDECARS)}
    if actual != targets:
        raise ValueError('Executed chunk coverage differs from declared jobs')
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Execution audit input changed')
    return dict(status='complete', smoke=smoke, jobs=len(files), rows=rows, inputs_sha256=inputs)


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new SEPARATE summary directory')
    frozen = load(fit / 'nominees.json')
    done = load(fit / 'completion.json')
    if (frozen['smoke'] or done['status'] != 'complete' or done['test_choices_used'] is not False
            or done['nomination_sha256'] != digest(fit / 'nominees.json')
            or frozen['settings'].get('belief_intervention') != INTERVENTION):
        raise ValueError('Require a completed, behavior-only scientific SEPARATE fit')
    if load(evaluation / 'completion.json')['status'] != 'complete':
        raise ValueError('Evaluation incomplete')
    audits = {name: audit_jobs(path) for name, path in (('fit', fit), ('evaluation', evaluation))}
    if (any(row['phase'] != 'training' for row in audits['fit']['rows'])
            or set(row['phase'] for row in audits['evaluation']['rows']) != {'evaluate', 'autonomous'}
            or audits['evaluation']['jobs'] != load(evaluation / 'completion.json')['jobs']):
        raise ValueError('Training/evaluation roles or completed job counts differ')
    extra_inputs = {str(path.resolve()): digest(path) for path in
                    (fit / 'completion.json', fit / 'separate_refit_protocol.json')}
    # Isolate adapter bindings from the generic module used by other analyses.
    spec = importlib.util.spec_from_file_location(
        'src.Bayesian_state.evaluation._separate_refit_summary', Path(base.__file__))
    summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)
    summary.training_chunk_budgets = training_chunk_budgets
    summary.refit_inactive_channels = refit_inactive_channels
    summary.summarize(fit, evaluation, output, reference_model='POOL')
    write(output / 'execution_audit.json', audits)
    shutil.copy2(__file__, output / 'separate_analysis_source.py')
    sources = [Path(__file__), Path(base.__file__), ROOT /
               'src/Bayesian_state/workflows/analysis/model_0923_online_parameter_cross.py']
    write(output / 'separate_manifest.json', dict(status='complete', belief_intervention=INTERVENTION,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        inputs_sha256=extra_inputs,
        generic_manifest_sha256=digest(output / 'manifest.json'),
        execution_audit_sha256=digest(output / 'execution_audit.json'),
        jobs={name: audit['jobs'] for name, audit in audits.items()},
        parameters='gamma: belief evidence; delta: admission/exit and recalled search; transfer: fixed inactive',
        interpretation='Behavior-only prefix refit; subsequent choices and autonomous generation evaluated separately'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); summarize(args.fit, args.evaluation, args.output)
