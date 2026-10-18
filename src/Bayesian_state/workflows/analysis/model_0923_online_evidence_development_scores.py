"""Score existing three-condition ONLINE development parameters without refitting.

Run --config YAML --output NEW_DIRECTORY [--smoke]. Completed autonomous
BASE/ONLINE banks supply identical parameters and runtime, preserving the
original parameter provenance. Fresh paired filter seeds evaluate choices.
Each subject exports the existing online-score schema for exact state replay.
Smoke runs eight trials, two four-particle filters per variant and an independent
RETAIN-only control per subject, using one process at a time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
SCORE = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_scores.py'
SPACE = 'src/Bayesian_state/workflows/analysis/model_0923_estimation_scope.py'


def helpers():
    spec = importlib.util.spec_from_file_location('online_development_score_worker', ROOT / SCORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked_banks(banks: list[dict], subjects: list[int]) -> dict[int, dict]:
    """Keep the previously completed participants and exact intervention pairs."""
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError('Require unique existing development subjects')
    selected = {}
    for subject in subjects:
        own = [b for b in banks if b['subject'] == subject]
        if len(own) != 2 or {b['schedule'] for b in own} != {'off', 'online'}:
            raise ValueError('Subject requires both completed development banks')
        left, right = (next(b for b in own if b['schedule'] == s) for s in ('off', 'online'))
        for key in ('candidate', 'model', 'parameter_provenance'):
            if left[key] != right[key]:
                raise ValueError('Development variants have different parameter provenance')
        if left['model'] != 'SEARCH' or not right['chunks']:
            raise ValueError('Require completed SEARCH development jobs')
        first = right['chunks'][0]
        for bank in own:
            if not bank['chunks']:
                raise ValueError('Missing development chunks')
            for job in bank['chunks']:
                if (job['candidate'] != first['candidate'] or job['candidate'] != bank['candidate']
                        or job['model'] != 'SEARCH' or job.get('retain_precision') is not True
                        or job['phase'] != 'autonomous' or job['smoke']):
                    raise ValueError('Changed completed development mechanism or candidate')
                for key in ('folder', 'runtime', 'mechanism', 'arm_settings'):
                    if job[key] != first[key]:
                        raise ValueError('Development task or cognitive implementation differs')
        selected[subject] = right
    return selected


def worker(file: Path) -> None:
    score = helpers()
    v = score.helpers().helpers().base.validation()
    if v.read(file).get('baseline_control'):
        score.helpers().worker(file)
    else:
        score.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    from .model_0923_capacity_memory_scores import choice_contrasts
    from .model_0923_estimation_scope import smoke_data
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new development choice destination')
    settings = yaml.safe_load(config.read_text())
    if (type(settings['filter_seeds']) is not int or not 2 <= settings['filter_seeds'] <= 8
            or type(settings['particles']) is not int or settings['particles'] < 2
            or not 1 <= settings['parallel_budget'] <= 128):
        raise ValueError('Invalid declared evaluation budget')
    score = helpers()
    retained = score.helpers()
    recalled = retained.helpers()
    v = recalled.base.validation()
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = v.digest(path)
        return path

    bind(config)
    source = ROOT / settings['source']
    original = v.read(bind(source / 'protocol.json'))
    done = v.read(bind(source / 'completion.json'))
    if (original['smoke'] or done['smoke'] or done['status'] != 'complete' or done['nominees_changed']
            or done['protocol_sha256'] != v.digest(source / 'protocol.json')):
        raise ValueError('Require the completed frozen-parameter development study')
    banks = checked_banks(original['banks'], settings['subjects'])
    # The proven choice worker supplies orchestration dependencies. Its S208
    # parameters and data are excluded from these three-subject score jobs.
    worker_reference = ROOT / settings['worker_reference']
    worker_protocol = v.read(bind(worker_reference / 'protocol.json'))
    worker_done = v.read(bind(worker_reference / 'completion.json'))
    if worker_done['status'] != 'complete' or worker_done['smoke']:
        raise ValueError('Choice worker reference must be a completed full screen')
    hashes = dict(original['source_sha256'])
    for name, sha in worker_protocol['source_sha256'].items():
        if name in hashes and hashes[name] != sha:
            raise ValueError('Development and choice worker implementations differ')
        hashes[name] = sha
    hashes.update({name:v.digest(ROOT / name) for name in (ENTRY, SCORE, SPACE)})
    for name, sha in dict(original['input_sha256'], **hashes).items():
        if v.digest(bind(ROOT / name)) != sha:
            raise ValueError('Changed original development input')
    # Reuse the retained-precision worker's input binding field with the actual
    # development provenance; this comparison performs no RESET refit.
    references = {str(p.resolve()):v.digest(p) for p in (source / 'protocol.json', source / 'completion.json')}
    output.mkdir(parents=True)
    particles, count = (4, 2) if smoke else (settings['particles'], settings['filter_seeds'])
    jobs, subjects = [], []
    for subject, bank in banks.items():
        original_job = bank['chunks'][0]
        actual = v.read(bind(source / 'jobs' / (original_job['name']+'.json')))
        if actual != original_job:
            raise ValueError('Planned development job differs from executed job')
        for suffix in ('.json', '.npz', '.online.json', '.precision.json', '.precision.npz', '.experiment.json'):
            bind(Path(actual['target']).with_suffix(suffix))
        record = v.read(Path(actual['target']).with_suffix('.json'))
        if record['predictions_sha256'] != v.digest(Path(actual['target']).with_suffix('.npz')):
            raise ValueError('Changed completed development prediction')
        folder = Path(actual['folder'])
        for name in ('prepared.json', 'engine.json', 'training.npz', 'sealed_test.npz'):
            bind(folder / name)
        child = output / f'subject_{subject}'
        child.mkdir()
        if smoke:
            folder = smoke_data(folder.parent, child, subject, v) / f'subject_{subject}'
        meta = v.read(folder / 'prepared.json')
        if meta['subject'] != subject:
            raise ValueError('Wrong prepared participant')
        seeds = [stable_seed(dict(base=settings['seed'], subject=subject,
                                  role='online_development_choice', repeat=i)) for i in range(count)]
        plans = []
        for enabled in (False, True):
            label = 'online' if enabled else 'baseline'
            name = f'{label}_S{subject}_SEARCH'
            keep = ('model','mechanism','candidate','runtime','arm_settings')
            job = dict({key:deepcopy(actual[key]) for key in keep}, name=name, phase='evaluate',
                folder=str(folder.resolve()), source_sha256=hashes, smoke=smoke, seeds=seeds, particles=particles,
                online_belief=enabled, retain_precision=True, structural_intervention=retained.INTERVENTION,
                retention_reference_inputs=references, target=str((child / 'conditional' / name).resolve()))
            chunks = [dict(job, name=name+f'_C{i:03d}', seeds=[seed],
                target=str((child / 'conditional' / 'chunks' / (name+f'_C{i:03d}')).resolve())) for i,seed in enumerate(seeds)]
            jobs.extend(chunks)
            plans.append(dict(model='SEARCH', variant=label, job=job, chunks=chunks))
        control = None
        if smoke:
            control = dict(deepcopy(plans[0]['chunks'][0]), baseline_control=True,
                name=f'control_S{subject}', target=str((child / 'control').resolve()))
            jobs.append(control)
        subjects.append(dict(subject=subject, folder=str(child.resolve()), plans=plans, control=control,
            parameter_provenance=bank['parameter_provenance'], training_trials=meta['training_trials']))
    protocol = dict(settings=original['settings'], score_settings=settings, smoke=smoke, subjects=subjects,
        particles=particles, filter_seeds=count, input_sha256=inputs, source_sha256=hashes,
        selection='Original development parameters frozen; no new parameter selection',
        data_use='Behavior choices only; original chronological boundaries; no oral or reserved participants')
    v.write(output / 'protocol.json', protocol)
    recalled.ENTRY = ENTRY
    recalled.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    for subject in subjects:
        child = Path(subject['folder'])
        (child / 'jobs').mkdir()
        arrays, records = {}, []
        for plan in subject['plans']:
            for chunk in plan['chunks']:
                shutil.copy2(output / 'jobs' / (chunk['name']+'.json'), child / 'jobs' / (chunk['name']+'.json'))
            record = v.combine_evaluation_chunks(plan['job'], plan['chunks'], False)
            with np.load(Path(plan['job']['target']).with_suffix('.npz')) as z:
                arrays[plan['variant']] = {key:z[key].copy() for key in z.files}
            records.append(dict(model='SEARCH',variant=plan['variant'],record=record,target=plan['job']['target']+'.npz'))
        a,b = arrays['online'], arrays['baseline']
        for key in ('choices','categories','seeds'):
            np.testing.assert_array_equal(a[key],b[key])
        for saved in (a,b):
            p = saved['probabilities']
            if not np.isfinite(p).all() or np.any(p < 0):
                raise ValueError('Invalid choice probabilities')
            np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
            np.testing.assert_array_equal(saved['observed_probability'],p[:,np.arange(p.shape[1]),saved['choices']-1])
        if smoke:
            with np.load(Path(subject['control']['target']).with_suffix('.npz')) as z:
                for key in z.files:
                    expected = b[key] if key in ('choices','categories') else b[key][:1]
                    np.testing.assert_array_equal(z[key],expected)
        cut = subject['training_trials']
        scores = [dict(model='SEARCH',**r) for r in choice_contrasts(a['observed_probability'],b['observed_probability'],cut)]
        p,q = a['observed_probability'].mean(0),b['observed_probability'].mean(0)
        loss = -np.log(np.maximum(p,np.finfo(float).tiny))+np.log(np.maximum(q,np.finfo(float).tiny))
        data = Path(subject['plans'][0]['job']['folder'])
        feedback = []
        for name in ('training.npz','sealed_test.npz'):
            with np.load(data / name) as z:
                feedback.extend(z['feedback'].tolist())
        if len(feedback) != len(p):
            raise ValueError('Feedback and prediction lengths differ')
        trials = [dict(model='SEARCH',trial=i+1,phase='prefix' if i<cut else 'later',
            correct=int(feedback[i]==1),feedback=feedback[i],online_probability=p[i],baseline_probability=q[i],
            delta_choice_nll=loss[i]) for i in range(len(p))]
        pd.DataFrame(scores).to_csv(child / 'scores.csv',index=False)
        pd.DataFrame(trials).to_csv(child / 'trial_scores.csv',index=False)
        child_protocol = dict(protocol, score_settings=dict(settings,subject=subject['subject']), plans=subject['plans'],
            parameter_provenance=subject['parameter_provenance'])
        v.write(child / 'protocol.json',child_protocol)
        v.write(child / 'completion.json',dict(status='complete',smoke=smoke,records=records,
            jobs=count*2,particles=particles,filter_seeds=count,trials=len(p),training_trials=cut,
            exact_independent_retained_control=smoke,nominees_changed=False,
            protocol_sha256=v.digest(child / 'protocol.json'),artifacts={p.name:v.digest(p) for p in child.glob('*.csv')}))
    for name,sha in inputs.items():
        if v.digest(Path(name)) != sha:
            raise ValueError('Bound input changed during evaluation')
    v.write(output / 'completion.json',dict(status='complete',smoke=smoke,jobs=len(jobs),
        subjects=[dict(subject=s['subject'],completion_sha256=v.digest(Path(s['folder'])/'completion.json')) for s in subjects],
        nominees_changed=False,protocol_sha256=v.digest(output / 'protocol.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config,args.output,args.smoke)
    else:
        parser.error('Use --config and --output, or an internal --worker')
