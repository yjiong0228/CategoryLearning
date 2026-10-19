"""Fit the existing TinyRNN on the H-study training prefixes, then freeze it.

Use --config YAML --output NEW_DIRECTORY [--smoke] with a CPU PyTorch runtime.
The original RNN training algorithm, hidden width, epochs and learning rate are
reused. Four declared initializations compete using final frozen-weight training
choice NLL. Test choices and oral reports do not select weights or initialization.
Condition 1 reads the first two response logits; other tasks retain all four.
After nomination, export full chronological probabilities and fixed-state 16-way
feature-reflection probes. The reflections measure potential feature effects;
they are not probabilities of verbal reports. No oral data are loaded here.

Smoke uses the first participant, one initialization, eight training trials and
three epochs. Every result lives in a new directory. The original RNN source is
archived and verified. The shared H implementation is unchanged.
After an assertion-only interruption, --reuse_fits OLD_OUTPUT
and --archived_driver EXACT_OLD_SOURCE verify and copy completed fits into a new
output. Training calculations, settings, inputs and saved predictions must match.
Any existing nominations must retain exactly the same seed, NLL and weights.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import yaml

from .model_0923_target_state_recovery import ROOT, digest, read, verify, write

ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
RNN = 'src/RNN_new/RNN_no_permutation.py'
IO = 'src/Bayesian_state/workflows/analysis/model_0923_target_state_recovery.py'


def rnn_module(path: Path, responses: int):
    """Reuse TinyRNN; restrict its readout to the task's valid response IDs."""
    if responses not in (2, 4):
        raise ValueError('This study has two or four response coordinates')
    import torch
    spec = importlib.util.spec_from_file_location('_frozen_rnn_comparator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEVICE = torch.device('cpu')
    original = module.TinyRNN

    class TaskRNN(original):
        def forward(self, x, c, f, h):
            logits, state, rho = super().forward(x, c, f, h)
            return logits[..., :responses], state, rho

    module.TinyRNN = TaskRNN
    return module


def validate_arrays(arrays: dict, responses: int) -> None:
    x, c, f = (arrays[k] for k in ('stimulus', 'choices', 'feedback'))
    if (x.ndim != 2 or x.shape[1] != 4 or not len(x) or c.shape != (len(x),)
            or f.shape != c.shape or not np.issubdtype(c.dtype, np.integer)
            or not all(np.isfinite(v).all() for v in (x, c, f))
            or np.any((x < 0) | (x > 1)) or np.any((c < 1) | (c > responses))
            or not np.isin(f, [0., .5, 1.]).all()):
        raise ValueError('Require chronological normalized stimuli, response IDs and experimental feedback')


def replay(module, model, arrays: dict, *, reflections: bool = False) -> dict:
    """Read each current stimulus from the state formed by previous history."""
    import torch
    model.eval()
    masks = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    h = torch.zeros(1, model.W_H.in_features)
    probabilities, states, rhos, probes = [], [], [], []
    with torch.no_grad():
        for t, stimulus in enumerate(arrays['stimulus']):
            previous = np.zeros(4) if t == 0 else module.one_hot(arrays['choices'][t-1]-1, 4)
            feedback = 0. if t == 0 else arrays['feedback'][t-1]
            x = torch.tensor(stimulus, dtype=torch.float32).view(1, 4)
            c = torch.tensor(previous, dtype=torch.float32).view(1, 4)
            f = torch.tensor([[feedback]], dtype=torch.float32)
            logits, h, rho = model(x, c, f, h)
            probability = torch.softmax(logits, dim=-1)[0].numpy()
            probabilities.append(probability)
            states.append(h[0].numpy().copy())
            rhos.append(rho[0].numpy().copy())
            if reflections:
                values = np.where(masks, 1.-stimulus, stimulus)
                changed = torch.tanh(rho + model.W_X(torch.tensor(values, dtype=torch.float32)))
                response_logits = model.readout(changed)[:, :len(probability)]
                probes.append(torch.softmax(response_logits, dim=-1).numpy())
    result = dict(probabilities=np.asarray(probabilities), state=np.asarray(states), rho=np.asarray(rhos))
    p = result['probabilities']
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Nonfinite RNN choice probabilities')
    # Bound summation roundoff by the number of float32 response coordinates.
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=p.shape[1]*np.finfo(p.dtype).eps, rtol=0)
    if reflections:
        result.update(reflection_probabilities=np.asarray(probes), reflection_masks=masks)
        # Batched and single-vector readouts accumulate float32 terms differently.
        np.testing.assert_allclose(result['reflection_probabilities'][:, 0], p,
                                   atol=model.W_H.in_features*np.finfo(p.dtype).eps, rtol=0)
    return result


def nll(probabilities: np.ndarray, choices: np.ndarray) -> float:
    p = probabilities.astype(float)[np.arange(len(choices)), choices-1]
    return float(-np.log(np.clip(p, 1e-12, 1.)).mean())


def worker(file: Path) -> None:
    import torch
    from threadpoolctl import threadpool_info
    job = read(file)
    verify(job['bindings'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    module = rnn_module(Path(job['rnn_source']), job['responses'])
    folder = Path(job['folder'])
    with np.load(folder/'training.npz') as z:
        arrays = {k:z[k].copy() for k in ('stimulus', 'choices', 'feedback', 'categories')}
    if job['smoke']:
        arrays = {k:v[:8] for k,v in arrays.items()}
    validate_arrays(arrays, job['responses'])
    start, cpu = time.monotonic(), time.process_time()
    target = Path(job['target'])
    metadata = {}
    if job['phase'] == 'fit':
        model, loss, _ = module.train(arrays['stimulus'], arrays['choices'], arrays['feedback'],
            H=job['hidden_width'], epochs=job['epochs'], lr=job['learning_rate'])
        captured = replay(module, model, arrays)
        with target.with_suffix('.pt').open('xb') as stream:
            torch.save(model.state_dict(), stream)
        metadata.update(checkpoint_sha256=digest(target.with_suffix('.pt')),
            training_nll=nll(captured['probabilities'], arrays['choices']), training_trials=len(arrays['choices']))
        captured['optimizer_loss'] = np.asarray(loss)
    elif job['phase'] == 'evaluate':
        model = module.TinyRNN(job['hidden_width'])
        model.load_state_dict(torch.load(job['checkpoint'], map_location='cpu', weights_only=True))
        cut = len(arrays['choices'])
        if not job['smoke']:
            with np.load(folder/'sealed_test.npz') as z:
                arrays = {k:np.concatenate([v,z[k]]) for k,v in arrays.items()}
        validate_arrays(arrays, job['responses'])
        captured = replay(module, model, arrays, reflections=True)
        reference_prediction = module.get_predictions(model, arrays['stimulus'], arrays['choices'], arrays['feedback'])
        reference_rho = module.compute_rho_traj(model, arrays['stimulus'], arrays['choices'], arrays['feedback'])
        np.testing.assert_array_equal(captured['probabilities'].argmax(axis=1), reference_prediction)
        np.testing.assert_array_equal(captured['rho'], reference_rho)
        with np.load(job['training_reference']) as reference:
            np.testing.assert_array_equal(captured['probabilities'][:cut], reference['probabilities'])
        metadata.update(training_nll=nll(captured['probabilities'][:cut], arrays['choices'][:cut]),
            test_nll=nll(captured['probabilities'][cut:], arrays['choices'][cut:]) if len(arrays['choices']) > cut else None,
            full_nll=nll(captured['probabilities'], arrays['choices']), training_trials=cut,
            test_trials=len(arrays['choices'])-cut, exact_original_predictions=True, exact_original_rho=True,
            exact_training_replay=True, reflection_roundoff_max=float(np.max(np.abs(
                captured['reflection_probabilities'][:, 0]-captured['probabilities']))))
    else:
        raise ValueError('Unknown RNN comparison phase')
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **captured, **arrays)
    verify(job['bindings'])
    write(target.with_suffix('.json'), dict(status='complete', subject=job['subject'], condition=job['condition'],
        seed=job['seed'], phase=job['phase'], smoke=job['smoke'], hidden_width=job['hidden_width'],
        responses=job['responses'], epochs=job['epochs'], learning_rate=job['learning_rate'],
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        numeric_pools=threadpool_info(), seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
        versions={name:importlib.metadata.version(name) for name in ('torch','numpy','pandas','scikit-learn')},
        predictions_sha256=digest(target.with_suffix('.npz')), **metadata))


def reusable_fits(source: Path, archived_driver: Path, settings: dict, bindings: dict) -> dict:
    """Reuse completed fits after an observational assertion-only correction.

    Verify the old driver against its original hash, preserve all training and
    prediction calculations by AST comparison, and verify every old input.
    """
    protocol = read(source/'protocol.json')
    if protocol['settings'] != json.loads(json.dumps(settings)) or protocol['smoke']:
        raise ValueError('Reuse is limited to the unchanged unfinished training protocol')
    nomination = source/'nominees.json'
    if nomination.exists():
        frozen = read(nomination)
        if (frozen.get('status') != 'frozen_before_test' or frozen.get('oral_used') is not False
                or frozen.get('test_used_for_selection') is not False):
            raise ValueError('Original nominations must remain frozen before test')
        bindings[str(nomination.resolve())] = digest(nomination)
    old_sources = protocol['inputs_sha256']
    entry_path = str((ROOT/ENTRY).resolve())
    if digest(archived_driver) != old_sources[entry_path]:
        raise ValueError('Need the exact original driver archived before its correction')
    before, after = [ast.parse(p.read_text()) for p in (archived_driver, ROOT/ENTRY)]
    def function(tree, name):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ('rnn_module','validate_arrays','worker','nll'):
        if ast.dump(function(before,name)) != ast.dump(function(after,name)):
            raise ValueError('Original fitting or scoring calculation changed')
    class RemoveAssertions(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='assert_allclose':
                return None
            return self.generic_visit(node)
    if ast.dump(RemoveAssertions().visit(function(before,'replay'))) != ast.dump(RemoveAssertions().visit(function(after,'replay'))):
        raise ValueError('Original replay calculation changed beyond its assertions')
    verify({p:sha for p,sha in old_sources.items() if p!=entry_path})
    bindings.update({p:sha for p,sha in old_sources.items() if p!=entry_path})
    for p in (archived_driver,source/'protocol.json'):
        bindings[str(p.resolve())] = digest(p)
    result = {}
    for job in protocol['jobs']:
        target = Path(job['target'])
        if not target.with_suffix('.json').exists():
            continue
        record = read(target.with_suffix('.json'))
        if (record['status']!='complete' or record['predictions_sha256']!=digest(target.with_suffix('.npz'))
                or record['checkpoint_sha256']!=digest(target.with_suffix('.pt'))):
            raise ValueError('Completed original fit changed')
        for suffix in ('.json','.npz','.pt'):
            file=target.with_suffix(suffix); bindings[str(file.resolve())]=digest(file)
        result[job['name']] = job
    return result


def run(config: Path, output: Path, smoke: bool, reuse: Path | None = None,
        archived_driver: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError('Use a new RNN comparison directory')
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'subjects','data','expected_boundaries','hidden_width','epochs','learning_rate','seeds','parallel_budget'}
            or settings['subjects'] != [107,207,208,306] or settings['hidden_width'] != 16
            or settings['epochs'] != 150 or settings['learning_rate'] != .001
            or len(settings['seeds']) != 4 or len(set(settings['seeds'])) != 4
            or not 1 <= settings['parallel_budget'] <= 16):
        raise ValueError('Require the declared four-subject, four-initialization original-training pilot')
    bindings = {str(p.resolve()):digest(p) for p in (config, ROOT/ENTRY, ROOT/RNN, ROOT/IO)}
    if bool(reuse) != bool(archived_driver) or smoke and reuse:
        raise ValueError('Reuse needs the original driver and a full comparison')
    reused = reusable_fits(reuse,archived_driver,settings,bindings) if reuse else {}
    output.mkdir(parents=True)
    for name in ('jobs','fits','evaluation','source'):
        (output/name).mkdir()
    frozen_source = output/'source/RNN_no_permutation.py'
    shutil.copyfile(ROOT/RNN, frozen_source)
    bindings[str(frozen_source.resolve())] = digest(frozen_source)
    jobs = []
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    seeds = settings['seeds'][:1] if smoke else settings['seeds']
    for subject in subjects:
        folder = ROOT/settings['data']/f'subject_{subject}'
        meta = read(folder/'prepared.json')
        expected = settings['expected_boundaries'][subject]
        if [meta[k] for k in ('condition','training_trials','test_trials')] != expected or meta['subject'] != subject:
            raise ValueError('Original task or chronological training boundary changed')
        for name in ('prepared.json','training.npz','sealed_test.npz'):
            path = folder/name
            bindings[str(path.resolve())] = digest(path)
            if name != 'prepared.json' and digest(path) != meta['array_sha256'][name]:
                raise ValueError('Prepared behavior changed')
        for seed in seeds:
            name = f'fit_S{subject}_seed{seed}'
            jobs.append(dict(name=name, phase='fit', subject=subject, condition=meta['condition'],
                responses=2 if meta['condition']==1 else 4, seed=seed, hidden_width=settings['hidden_width'],
                epochs=3 if smoke else settings['epochs'], learning_rate=settings['learning_rate'],
                folder=str(folder.resolve()), rnn_source=str(frozen_source.resolve()), smoke=smoke,
                target=str((output/'fits'/name).resolve())))
    workers = min(1 if smoke else settings['parallel_budget'], len(jobs), len(os.sched_getaffinity(0)))
    for job in jobs:
        job['bindings'] = dict(bindings)
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, workers=workers, jobs=jobs,
        inputs_sha256=bindings, selection='Minimum frozen-weight training choice NLL; seed breaks exact ties',
        training='Original TinyRNN.train, one-step detached hidden state, per-trial Adam update, final epoch',
        scope='Development predictive comparator; distinct model capacity and fitting budget recorded separately',
        oral_used=False, test_used_for_selection=False, python=sys.version,
        reused_fit_jobs=sorted(reused), original_fit_source=str(reuse.resolve()) if reuse else None))

    def launch(job):
        file = output/'jobs'/(job['name']+'.json'); write(file, job)
        if job['name'] in reused:
            original = reused[job['name']]
            for key in ('phase','subject','condition','responses','seed','hidden_width','epochs','learning_rate','folder','smoke'):
                if job[key]!=original[key]:raise ValueError('Reused fitting settings changed')
            for suffix in ('.json','.npz','.pt'):
                with Path(job['target']).with_suffix(suffix).open('xb') as stream:
                    stream.write(Path(original['target']).with_suffix(suffix).read_bytes())
            return read(Path(job['target']).with_suffix('.json'))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory(prefix='rnn_comparator_') as cache:
            env.update(MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable,'-m',ENTRY[:-3].replace('/','.'),'--worker',str(file.resolve())],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"RNN worker failed: {file.with_suffix('.log')}")
        record = read(Path(job['target']).with_suffix('.json'))
        if record['predictions_sha256'] != digest(Path(job['target']).with_suffix('.npz')):
            raise ValueError('RNN probability artifact changed')
        return record

    print(f'{len(jobs)} RNN training jobs; {workers} workers; one numeric thread', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(launch,j) for j in jobs]):
            records.append(future.result()); print(f'{len(records)}/{len(jobs)} fits complete',flush=True)
    nominees, evaluations = [], []
    for subject in subjects:
        candidates = sorted([r for r in records if r['subject']==subject],key=lambda r:(r['training_nll'],r['seed']))
        winner = candidates[0]
        original = next(j for j in jobs if j['subject']==subject and j['seed']==winner['seed'])
        checkpoint = Path(original['target']).with_suffix('.pt')
        if digest(checkpoint) != winner['checkpoint_sha256']:
            raise ValueError('Selected RNN weights changed')
        nominees.append(dict(subject=subject, seed=winner['seed'], training_nll=winner['training_nll'],
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), candidates=candidates))
        name = f'evaluate_S{subject}'
        evaluations.append(dict(original, name=name, phase='evaluate', checkpoint=str(checkpoint),
            training_reference=str(Path(original['target']).with_suffix('.npz')),
            target=str((output/'evaluation'/name).resolve()), bindings=dict(bindings, **{
                str(checkpoint):digest(checkpoint),
                str(Path(original['target']).with_suffix('.npz')):digest(Path(original['target']).with_suffix('.npz'))})))
    if reuse and (reuse/'nominees.json').exists():
        old_nominees = read(reuse/'nominees.json')['nominees']
        fields = ('subject','seed','training_nll','checkpoint_sha256')
        if ([tuple(n[k] for k in fields) for n in old_nominees]
                != [tuple(n[k] for k in fields) for n in nominees]):
            raise ValueError('Originally frozen RNN parameters changed')
    write(output/'nominees.json',dict(status='frozen_before_test',smoke=smoke,nominees=nominees,
        oral_used=False,test_used_for_selection=False))
    nomination = output/'nominees.json'
    evaluated = []
    with ThreadPoolExecutor(max_workers=min(workers,len(evaluations))) as pool:
        for job in evaluations:
            job['bindings'][str(nomination.resolve())] = digest(nomination)
        for future in as_completed([pool.submit(launch,j) for j in evaluations]):
            evaluated.append(future.result())
    pd.DataFrame([dict(subject=r['subject'],seed=r['seed'],training_nll=r['training_nll']) for r in records]).to_csv(output/'training_scores.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ('subject','condition','seed','training_trials','test_trials','training_nll','test_nll','full_nll')}
                  for r in evaluated]).to_csv(output/'choice_scores.csv',index=False)
    verify(bindings)
    write(output/'completion.json',dict(status='complete',smoke=smoke,fit_jobs=len(records),evaluation_jobs=len(evaluated),
        inputs_sha256=bindings,nomination_sha256=digest(nomination),protocol_sha256=digest(output/'protocol.json'),
        evaluations=evaluated,oral_used=False,test_used_for_selection=False,reused_fit_jobs=len(reused),
        artifacts_sha256={name:digest(output/name) for name in ('training_scores.csv','choice_scores.csv')}))
    print('Frozen RNN predictions and local reflection probes complete',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker','reuse_fits','archived_driver'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config,args.output,args.smoke,args.reuse_fits,args.archived_driver)
    else:
        parser.error('Provide --config and a new --output, or --worker')
