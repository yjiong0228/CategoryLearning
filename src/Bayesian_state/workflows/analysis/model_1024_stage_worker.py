"""Frozen worker adapter for the bounded state-validation stage.

This entry dispatches existing shared generation, sampled execution capture
or zero-inertia marginalization. It adds source checks, no cognitive mechanism.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
MARGINAL='src/Bayesian_state/workflows/analysis/model_1024_marginal_execution.py'
SMOOTH='src/Bayesian_state/workflows/analysis/model_1024_smoothing.py'


def load(name: str,path: str):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


marginal=load('stage_marginal_worker',MARGINAL)
smooth=load('stage_sampled_worker',SMOOTH)
study=marginal.study
read,write,digest=study.read,study.write,study.digest


def sources() -> dict:
    return dict(marginal.sources(),**smooth.sources(),**{ENTRY:digest(ROOT/ENTRY)})


def worker(file: Path) -> None:
    job=read(file)
    if job['stage_sources']!=sources() or job.get('capture_oral'):
        raise ValueError('Stage worker sources changed or oral validation requested')
    if job['stage_kind'] in ('generation','bare'):
        study.worker(file)
    elif job['execution_mode']=='MIX':
        marginal.worker(file)
    elif job['execution_mode']=='STICKY':
        smooth.worker(file)
    else:raise ValueError('Unsupported stage numerical carrier')
    if job['stage_sources']!=sources():raise ValueError('Worker sources changed during run')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path,required=True)
    worker(parser.parse_args().worker)
