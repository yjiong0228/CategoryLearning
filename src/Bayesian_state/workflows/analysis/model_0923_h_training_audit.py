"""Rescore all H finalists on training data before opening held-out choices.

This optional numerical audit leaves the original fit immutable. It uses the
same pre-choice objective and archived cognitive worker, with more particles.
Complete independent filter seeds are the sole process-parallel units. The
derived nominees retain the original search protocol and all audit candidates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import yaml

from .model_0923_h_validation import REPO,digest,read,write,source_hashes,score_jobs,execute,pooled_nll


def combine(job: dict, chunks: list[dict]) -> dict:
    """Combine seed arrays before scoring, retaining every scored trial."""
    arrays=[];records=[]
    for chunk in chunks:
        target=Path(chunk['target']);row=read(target.with_suffix('.json'))
        if digest(target.with_suffix('.npz'))!=row['predictions_sha256']:raise ValueError('Audit chunk changed')
        with np.load(target.with_suffix('.npz')) as z:arrays.append({k:z[k] for k in z.files})
        records.append(row)
    result={key:np.concatenate([a[key] for a in arrays],axis=0) for key in arrays[0]}
    np.testing.assert_array_equal(result['seeds'],job['seeds'])
    target=Path(job['target'])
    with target.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**result)
    row=deepcopy(records[0]);row.update(seeds=job['seeds'],mean_nll=pooled_nll(result['observed_probability']),
        seconds=sum(r['seconds'] for r in records),cpu_seconds=sum(r['cpu_seconds'] for r in records),
        chunk_files=[c['target'] for c in chunks],predictions_sha256=digest(target.with_suffix('.npz')),
        role='training_only_high_particle_audit')
    write(target.with_suffix('.json'),row)
    return dict(job['candidate'],mean_nll=row['mean_nll'],score_file=str(target.with_suffix('.json')))


def run(fit: Path, config: Path, output: Path, smoke: bool) -> None:
    if output.exists():raise FileExistsError('Use a new training audit destination')
    frozen=read(fit/'nominees.json');audit=yaml.safe_load(config.read_text())
    if frozen['status']!='frozen_before_test' or frozen['smoke']:raise ValueError('Need complete scientific training candidates')
    for name,sha in frozen['source_sha256'].items():
        if digest(REPO/name)!=sha:raise ValueError('Training implementation changed')
    output.mkdir(parents=True);settings=deepcopy(frozen['settings'])
    settings['budgets']['audit']=dict(particles=audit['particles'],seeds=audit['seeds'])
    selected=frozen['nominees'][:1] if smoke else frozen['nominees']
    banks={(r['subject'],r['model']):r['finalists'][:2 if smoke else audit['finalists']] for r in selected}
    if any(len(v)!=(2 if smoke else audit['finalists']) for v in banks.values()):raise ValueError('Unequal audit candidate count')
    hashes=source_hashes();hashes[str(Path(__file__).resolve().relative_to(REPO))]=digest(Path(__file__))
    write(output/'training_audit_protocol.json',dict(parent_fit=str(fit.resolve()),parent_nomination_sha256=digest(fit/'nominees.json'),
        audit=audit,smoke=smoke,source_sha256=hashes,data_boundary='training.npz only; sealed_test.npz unopened',
        reason='64-particle finalist differences were comparable to filter-seed Monte Carlo variation'))
    jobs=score_jobs(Path(frozen['data']),output,settings,banks,'audit',smoke)
    groups={};chunks=[]
    for job in jobs:
        job['source_sha256']=hashes;groups[job['name']]=[]
        for index,seed in enumerate(job['seeds']):
            chunk=deepcopy(job);target=Path(job['target'])
            chunk.update(name=job['name']+f'_C{index:03d}',seeds=[seed],
                target=str(target.parent/'chunks'/(target.name+f'_C{index:03d}')))
            chunks.append(chunk);groups[job['name']].append(chunk)
    # Put longer sequences first so short tasks can fill the last wave.
    lengths={j['folder']:read(Path(j['folder'])/'prepared.json')['training_trials'] for j in jobs}
    chunks.sort(key=lambda j:-lengths[j['folder']])
    start=time.monotonic();execute(chunks,output,1 if smoke else audit['parallel_budget'])
    scored={key:[] for key in banks}
    for job in jobs:
        row=combine(job,groups[job['name']]);subject=read(Path(job['folder'])/'prepared.json')['subject']
        scored[(subject,job['model'])].append(row)
    nominees=[]
    for original in selected:
        rows=sorted(scored[(original['subject'],original['model'])],key=lambda r:(r['mean_nll'],r['id']))
        nominees.append(dict(original,candidate=rows[0],finalists=rows,previous_candidate_id=original['candidate']['id'],
            scientific_status='bounded_training_selection_after_uniform_high_particle_audit'))
    write(output/'nominees.json',dict(status='frozen_before_test',settings=settings,smoke=smoke,data=frozen['data'],
        source_sha256=hashes,nominees=nominees,parent_fit=str(fit.resolve()),
        parent_nomination_sha256=digest(fit/'nominees.json'),training_audit=audit,elapsed_seconds=time.monotonic()-start))
    print('Training-only audit complete; derived nominees frozen before test evaluation',flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();run(args.fit,args.config,args.output,args.smoke)


if __name__=='__main__':main()
