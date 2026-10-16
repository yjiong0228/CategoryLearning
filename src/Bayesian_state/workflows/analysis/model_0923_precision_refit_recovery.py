"""Run precision refits with a strictly bounded particle-snapshot repair.

--reuse-screen may import complete initial-screen jobs from the identical
original protocol. Original jobs used a strict probability check: successful
completion proves the new load guard would be an identity for every restored
snapshot. All numerical arrays are copied unchanged and individually hashed.
Missing/failed jobs run anew; subsequent stages use the guard for all families.
Both source versions and every reuse decision remain auditable. New output
directories are mandatory. Cognitive updates and selection budgets are unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('precision_refit_base',
    REPO/'src/Bayesian_state/workflows/analysis/model_0923_precision_refit.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

GUARD = 'src/Bayesian_state/model/modules/pairing_snapshot_roundoff.py'
ENTRY = 'src/Bayesian_state/workflows/analysis/model_0923_precision_refit_recovery.py'


def worker(file: Path) -> None:
    # Load the experiment helpers before the new guard; archived runtimes do
    # not contain these opt-in modules. The memory class itself stays archived.
    base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', base.FEEDBACK)
    guard = base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', GUARD)
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    with guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule) as corrections:
        base.worker(file)
    v=base.validation(); job=v.read(file)
    v.write(Path(job['target']).with_suffix('.snapshot.json'),dict(corrections=corrections))


def reuse_job(job: dict, parent: Path, output: Path) -> bool:
    """Import a complete matching run; strict old validation proves identity."""
    v=base.validation(); old_file=parent/'jobs'/(job['name']+'.json')
    if not old_file.exists(): return False
    old=v.read(old_file)
    fields=('phase','folder','model','mechanism','candidate','runtime','seeds','particles','smoke','arm_settings')
    if any(old[k]!=job[k] for k in fields): raise ValueError('Reuse job differs from the declared scientific protocol')
    # All original source files must still match; only this wrapper and guard
    # are added. Changing the candidate, algorithm, budget or seeds forbids reuse.
    for name,sha in old['source_sha256'].items():
        if job['source_sha256'].get(name)!=sha: raise ValueError('Original screen implementation changed')
    src=Path(old['target']); dst=Path(job['target'])
    if not all(src.with_suffix(ext).exists() for ext in ('.json','.npz','.experiment.json')): return False
    if 'Traceback' in old_file.with_suffix('.log').read_text(): return False
    row=v.read(src.with_suffix('.json'))
    if row['predictions_sha256']!=v.digest(src.with_suffix('.npz')): raise ValueError('Reusable predictions changed')
    dst.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('.json','.npz','.experiment.json'): shutil.copy2(src.with_suffix(ext),dst.with_suffix(ext))
    if src.with_suffix('.snapshot.json').exists():
        if old['source_sha256'].get(GUARD)!=job['source_sha256'][GUARD]:
            raise ValueError('A repaired job requires the identical snapshot guard')
        shutil.copy2(src.with_suffix('.snapshot.json'),dst.with_suffix('.snapshot.json'))
    else:
        v.write(dst.with_suffix('.snapshot.json'),dict(corrections=[],
            identity_reason='Every original restoration passed the strict [0,1] check; guard changes only overflowing evidence.'))
    v.write(output/'jobs'/(job['name']+'.json'),dict(job,reused_from=str(old_file.resolve())))
    directory=output/'reused_jobs';directory.mkdir(exist_ok=True)
    v.write(directory/(job['name']+'.json'),dict(source_job=str(old_file.resolve()),
        source_job_sha256=v.digest(old_file),source_predictions=str(src.with_suffix('.npz')),
        source_predictions_sha256=row['predictions_sha256'],original_source_sha256=old['source_sha256'],
        copied_files_sha256={ext:v.digest(dst.with_suffix(ext)) for ext in ('.json','.npz','.experiment.json')}))
    return True


def execute(jobs: list[dict], output: Path, budget: int, parents: list[Path] | None) -> None:
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v=base.validation();directory=output/'jobs';directory.mkdir(exist_ok=True)
    reused=[];pending=[]
    for job in jobs:
        if job['name'].startswith('screen_') and any(reuse_job(job,parent,output) for parent in (parents or [])):
            reused.append(job['name'])
        else:pending.append(job)
    if reused or (parents and jobs[0]['name'].startswith('screen_')):
        v.write(output/'screen_reuse.json',dict(parents=[str(parent.resolve()) for parent in parents],reused=reused,
            pending=[j['name'] for j in pending],parent_protocol_sha256={str(parent.resolve()):v.digest(parent/'protocol.json') for parent in parents},
            reason='Successful original jobs pass strict snapshot validation; repaired jobs require the identical guard.'))
    if not pending:return
    def launch(job):
        file=directory/(job['name']+'.json');v.write(file,job)
        launcher=('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='precision_snapshot_') as cache:
            env['NUMBA_CACHE_DIR']=cache
            with file.with_suffix('.log').open('x') as stream:
                result=subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(Path(__file__).resolve()),
                    '--worker',str(file)],cwd=base.REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')
    workers=parallel_job_count(budget,len(pending))
    print(f'{len(pending)} jobs; {len(reused)} unchanged jobs reused; {workers} workers; numeric threads=1',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(launch,j) for j in pending]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%32==0 or i==len(pending):print(f'{i}/{len(pending)} complete',flush=True)


def configure(parents: list[Path] | None = None) -> None:
    original=base.source_hashes
    def hashes(v):
        result=original(v)
        for name in (GUARD,ENTRY):result[name]=v.digest(base.REPO/name)
        return result
    base.source_hashes=hashes
    base.execute=lambda jobs,output,budget:execute(jobs,output,budget,parents)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path);p.add_argument('--phase',choices=('search','evaluate'))
    for name in ('config','fit','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--reuse-screen',type=Path,nargs='+')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.worker:worker(a.worker);return
    configure(a.reuse_screen)
    if a.phase=='search' and a.config and a.output:base.search(a.config,a.output,a.smoke)
    elif a.phase=='evaluate' and a.fit and a.output and not a.smoke and a.reuse_screen is None:base.evaluate(a.fit,a.output)
    else:p.error('Supply a search/evaluate phase and its inputs; --reuse-screen applies to search only')


if __name__=='__main__':main()
