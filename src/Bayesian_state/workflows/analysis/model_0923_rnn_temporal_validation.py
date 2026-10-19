"""Choose original RNN checkpoints by forward validation inside training data.

Use --config YAML --output NEW_DIRECTORY [--smoke] in a CPU PyTorch runtime.
Two chronological folds remain inside each original H training prefix. Original
TinyRNN.train supplies its own checkpoints; its update algorithm is unchanged.
Choose the epoch count by pooled validation choice NLL averaged over the four
declared initializations. Refit four initializations on the full training prefix,
choose the seed by training NLL, freeze it, then run the existing full evaluator.
All subjects share this policy. Oral reports and outer outcomes select no weights.
This development comparison refines the training procedure after the original
development results were seen; reserved participants remain unopened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from . import model_0923_rnn_comparator as original
from .model_0923_target_state_recovery import ROOT, digest, read, verify, write

ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def fold_bounds(length: int, fractions: list[float]) -> list[tuple[int, int]]:
    """Return expanding-prefix fit ends and disjoint forward-validation ends."""
    values = np.asarray(fractions, float)
    if (not isinstance(length, int) or length < 4 or values.shape != (3,)
            or not np.isfinite(values).all() or not 0 < values[0] < values[1] < values[2] == 1):
        raise ValueError('Require three increasing fractions ending at one')
    cuts = np.floor(length*values).astype(int)
    if cuts[0] < 1 or not np.all(np.diff(cuts) > 0): raise ValueError('Empty training or validation fold')
    return [(int(cuts[0]), int(cuts[1])), (int(cuts[1]), int(cuts[2]))]


def choose_epochs(scores: pd.DataFrame, subjects: list[int], seeds: list[int],
                  epochs: list[int]) -> tuple[pd.DataFrame, dict[int, int]]:
    """Select using only complete inner-fold scores; shorter exact ties win."""
    keys = ['subject', 'seed', 'fold', 'epochs']
    expected = {(s, seed, fold, ep) for s in subjects for seed in seeds for fold in (0, 1) for ep in epochs}
    if (scores.duplicated(keys).any() or set(map(tuple, scores[keys].to_numpy())) != expected
            or not np.isfinite(scores[['validation_nll', 'validation_trials']]).all().all()
            or (scores.validation_nll < 0).any() or (scores.validation_trials <= 0).any()):
        raise ValueError('Need every declared subject, seed, fold and checkpoint')
    sizes = scores.groupby(['subject', 'fold']).validation_trials.nunique()
    if not sizes.eq(1).all(): raise ValueError('Validation trial counts differ across checkpoints or seeds')
    table = scores[keys+['validation_nll', 'validation_trials']].copy()
    table['loss'] = table.validation_nll*table.validation_trials
    summary = table.groupby(['subject', 'epochs']).agg(
        loss=('loss', 'sum'), scored_trial_instances=('validation_trials', 'sum')).reset_index()
    summary['validation_trials_per_seed'] = summary.scored_trial_instances/len(seeds)
    summary['initializations'] = len(seeds)
    summary['validation_nll'] = summary.loss/summary.scored_trial_instances
    selected = {int(s): int(g.sort_values(['validation_nll', 'epochs']).iloc[0].epochs)
                for s, g in summary.groupby('subject')}
    return summary, selected


def inner_worker(file: Path) -> None:
    import torch
    from threadpoolctl import threadpool_info
    job = read(file); verify(job['bindings'])
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(job['seed']); np.random.seed(job['seed'])
    module = original.rnn_module(Path(job['rnn_source']), job['responses'])
    with np.load(Path(job['folder'])/'training.npz') as z:
        arrays = {k:z[k][:job['validation_end']].copy() for k in ('stimulus', 'choices', 'feedback', 'categories')}
    original.validate_arrays(arrays, job['responses'])
    start, cpu = time.monotonic(), time.process_time(); cut = job['fit_end']
    model, loss, snapshots = module.train(arrays['stimulus'][:cut], arrays['choices'][:cut], arrays['feedback'][:cut],
        H=job['hidden_width'], epochs=job['epochs'], lr=job['learning_rate'])
    if sorted(ep+1 for ep in snapshots) != job['checkpoint_epochs']:
        raise ValueError('Original trainer checkpoint schedule differs')
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value.detach().numpy(), snapshots[job['epochs']-1].state_dict()[name].detach().numpy())
    probabilities, records, weights = [], [], {}
    for epoch in job['checkpoint_epochs']:
        checkpoint = snapshots[epoch-1]
        captured = original.replay(module, checkpoint, arrays)
        p = captured['probabilities']; probabilities.append(p)
        weights[epoch] = checkpoint.state_dict()
        records.append(dict(subject=job['subject'], seed=job['seed'], fold=job['fold'], epochs=epoch,
            fit_trials=cut, validation_trials=len(p)-cut,
            training_nll=original.nll(p[:cut], arrays['choices'][:cut]),
            validation_nll=original.nll(p[cut:], arrays['choices'][cut:])))
    target = Path(job['target'])
    with target.with_suffix('.pt').open('xb') as stream: torch.save(weights, stream)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays, probabilities=np.stack(probabilities),
                            checkpoint_epochs=np.array(job['checkpoint_epochs']), optimizer_loss=np.asarray(loss))
    verify(job['bindings'])
    write(target.with_suffix('.json'), dict(status='complete', phase='inner_validation',
        subject=job['subject'], seed=job['seed'], fold=job['fold'], smoke=job['smoke'], records=records,
        torch_version=torch.__version__, numeric_pools=threadpool_info(), torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(), seconds=time.monotonic()-start,
        cpu_seconds=time.process_time()-cpu, training_source='original TinyRNN.train and returned snapshots',
        predictions_sha256=digest(target.with_suffix('.npz')), checkpoint_sha256=digest(target.with_suffix('.pt'))))


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists(): raise FileExistsError('Use a new temporal-validation directory')
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'base_config','fold_fractions','checkpoint_epochs','parallel_budget'}
            or settings['fold_fractions'] != [.5,.75,1.] or settings['checkpoint_epochs'] != [1,51,101,150]
            or not 1 <= settings['parallel_budget'] <= 16):
        raise ValueError('Require the declared bounded temporal-validation policy')
    base_path = ROOT/settings['base_config']; base = yaml.safe_load(base_path.read_text())
    if (base['subjects'] != [107,207,208,306] or base['hidden_width'] != 16 or base['epochs'] != 150
            or base['learning_rate'] != .001 or len(base['seeds']) != 4 or len(set(base['seeds'])) != 4):
        raise ValueError('Original four-subject RNN settings changed')
    bindings = {str(p.resolve()):digest(p) for p in (config, base_path, ROOT/ENTRY,
                ROOT/original.ENTRY, ROOT/original.RNN, ROOT/original.IO)}
    output.mkdir(parents=True)
    for name in ('jobs','inner','fits','evaluation','source'): (output/name).mkdir()
    frozen_source = output/'source/RNN_no_permutation.py'
    shutil.copyfile(ROOT/original.RNN, frozen_source)
    bindings[str(frozen_source.resolve())] = digest(frozen_source)
    subjects = base['subjects'][:1] if smoke else base['subjects']
    seeds = base['seeds'][:1] if smoke else base['seeds']
    epochs = [1,2,3] if smoke else settings['checkpoint_epochs']
    metadata, jobs = {}, []
    for subject in subjects:
        folder = ROOT/base['data']/f'subject_{subject}'; meta = read(folder/'prepared.json')
        if (meta['subject'] != subject or [meta[k] for k in ('condition','training_trials','test_trials')]
                != base['expected_boundaries'][subject]): raise ValueError('Original task boundary changed')
        metadata[subject] = (folder, meta)
        for name in ('prepared.json','training.npz','sealed_test.npz'):
            file = folder/name; bindings[str(file.resolve())] = digest(file)
            if name != 'prepared.json' and digest(file) != meta['array_sha256'][name]: raise ValueError('Original behavior changed')
        for fold, (fit_end, validation_end) in enumerate(fold_bounds(8 if smoke else meta['training_trials'], settings['fold_fractions'])):
            for seed in seeds:
                name = f'inner_S{subject}_fold{fold}_seed{seed}'
                jobs.append(dict(name=name, phase='inner_validation', subject=subject, seed=seed, fold=fold,
                    condition=meta['condition'], responses=2 if meta['condition']==1 else 4,
                    hidden_width=base['hidden_width'], learning_rate=base['learning_rate'], epochs=max(epochs),
                    checkpoint_epochs=epochs, fit_end=fit_end, validation_end=validation_end,
                    folder=str(folder.resolve()), rnn_source=str(frozen_source.resolve()), smoke=smoke,
                    target=str((output/'inner'/name).resolve())))
    workers = min(1 if smoke else settings['parallel_budget'], len(jobs), len(os.sched_getaffinity(0)))
    for job in jobs: job['bindings'] = dict(bindings)
    write(output/'protocol.json', dict(settings=settings, original_settings=base, smoke=smoke, workers=workers,
        jobs=jobs, inputs_sha256=bindings, oral_used=False, test_used_for_selection=False,
        epoch_selection='Pooled forward-validation NLL over both folds and all declared seeds; shorter exact ties',
        seed_selection='After epoch selection, refit all seeds on full original training prefix; minimum training NLL',
        optimization='Original TinyRNN.train, returned checkpoints and original final evaluator',
        data_use='Development extension after original development comparisons; reserved subjects unopened'))

    def launch(job):
        file = output/'jobs'/(job['name']+'.json'); write(file, job)
        entry = ENTRY if job['phase']=='inner_validation' else original.ENTRY
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory(prefix='rnn_temporal_') as cache:
            env['MPLCONFIGDIR'] = cache
            with file.with_suffix('.log').open('x') as stream:
                process = subprocess.run([sys.executable,'-m',entry[:-3].replace('/','.'),'--worker',str(file.resolve())],
                                         cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode: raise RuntimeError(f'RNN worker failed: {file.with_suffix(".log")}')
        record = read(Path(job['target']).with_suffix('.json'))
        if digest(Path(job['target']).with_suffix('.npz')) != record['predictions_sha256']: raise ValueError('Prediction changed')
        return record

    def batch(stage):
        completed = []
        with ThreadPoolExecutor(max_workers=min(workers,len(stage))) as pool:
            for future in as_completed([pool.submit(launch,job) for job in stage]):
                completed.append(future.result()); print(f'{stage[0]["phase"]}: {len(completed)}/{len(stage)} complete',flush=True)
        return completed

    print(f'{len(jobs)} inner fits; {len(epochs)} original checkpoints each; {workers} workers; one numeric thread',flush=True)
    inner = batch(jobs)
    score_table = pd.DataFrame([row for result in inner for row in result['records']])
    summary, selected = choose_epochs(score_table, subjects, seeds, epochs)
    score_table.to_csv(output/'inner_scores.csv',index=False); summary.to_csv(output/'epoch_scores.csv',index=False)
    write(output/'epoch_selection.json', dict(status='frozen_before_outer_refit', selected_epochs=selected,
        inner_scores_sha256=digest(output/'inner_scores.csv'), epoch_scores_sha256=digest(output/'epoch_scores.csv'),
        oral_used=False, outer_test_used=False))
    selection = output/'epoch_selection.json'; bindings[str(selection.resolve())] = digest(selection)
    full_jobs = []
    for subject in subjects:
        folder, meta = metadata[subject]
        for seed in seeds:
            name = f'fit_S{subject}_seed{seed}'
            full_jobs.append(dict(name=name, phase='fit', subject=subject, condition=meta['condition'],
                responses=2 if meta['condition']==1 else 4, seed=seed, hidden_width=base['hidden_width'],
                epochs=selected[subject], learning_rate=base['learning_rate'], folder=str(folder.resolve()),
                rnn_source=str(frozen_source.resolve()), smoke=smoke, target=str((output/'fits'/name).resolve()), bindings=dict(bindings)))
    fitted = batch(full_jobs); nominees, evaluation_jobs = [], []
    for subject in subjects:
        candidates = sorted((r for r in fitted if r['subject']==subject), key=lambda r:(r['training_nll'],r['seed']))
        winner = candidates[0]; job = next(j for j in full_jobs if j['subject']==subject and j['seed']==winner['seed'])
        checkpoint = Path(job['target']).with_suffix('.pt'); reference = Path(job['target']).with_suffix('.npz')
        if digest(checkpoint) != winner['checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
        nominees.append(dict(subject=subject,seed=winner['seed'],epochs=winner['epochs'],training_nll=winner['training_nll'],
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),candidates=candidates))
        name = f'evaluate_S{subject}'
        evaluation_jobs.append(dict(job, name=name,phase='evaluate',checkpoint=str(checkpoint),
            training_reference=str(reference), target=str((output/'evaluation'/name).resolve()),
            bindings=dict(bindings, **{str(checkpoint):digest(checkpoint),str(reference):digest(reference)})))
    write(output/'nominees.json', dict(status='frozen_before_test',smoke=smoke,nominees=nominees,
                                    oral_used=False,test_used_for_selection=False))
    for job in evaluation_jobs: job['bindings'][str((output/'nominees.json').resolve())] = digest(output/'nominees.json')
    evaluated = batch(evaluation_jobs)
    pd.DataFrame([{k:r[k] for k in ('subject','seed','epochs','training_nll')} for r in fitted]).to_csv(output/'training_scores.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ('subject','condition','seed','epochs','training_trials','test_trials','training_nll','test_nll','full_nll')}
                  for r in evaluated]).to_csv(output/'choice_scores.csv',index=False)
    verify(bindings)
    write(output/'completion.json', dict(status='complete',smoke=smoke,inner_fit_jobs=len(inner),fit_jobs=len(fitted),
        evaluation_jobs=len(evaluated),evaluations=evaluated,selected_epochs=selected,inputs_sha256=bindings,
        nomination_sha256=digest(output/'nominees.json'),protocol_sha256=digest(output/'protocol.json'),
        oral_used=False,test_used_for_selection=False,artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')}))
    print('Temporal validation, full-prefix refit and frozen outer predictions complete',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'): parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true'); args = parser.parse_args()
    if args.worker: inner_worker(args.worker)
    elif args.config and args.output: run(args.config,args.output,args.smoke)
    else: parser.error('Provide --config and a new --output, or --worker')
