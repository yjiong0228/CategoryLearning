"""Passive state storage with preallocation and bounded temporary summaries.

Internal entry: --worker JOB, under the existing frozen behavioral runtime.
The original observer computes every particle value. This adapter replaces
list-plus-stack storage and calls the original state_means in trial blocks.
It leaves random draws, state transitions and observation weights untouched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
BASE='src/Bayesian_state/workflows/analysis/model_1024_individual_state_capture.py'


class PreallocatedHistory:
    """List-compatible append storage with one final array per recorded field."""

    def __init__(self, total: int, mapping: bool=True):
        self.total, self.mapping, self.length = total, mapping, 0
        self.data: dict[str,np.ndarray] = {}

    def __len__(self) -> int:
        return self.length

    def append(self, row) -> None:
        if self.length >= self.total:
            raise ValueError('History exceeds registered trial count')
        row=row if self.mapping else {'value':np.asarray(row)}
        if not self.data:
            self.data={key:np.empty((self.total,)+value.shape,dtype=value.dtype) for key,value in row.items()}
        if row.keys()!=self.data.keys():
            raise ValueError('Recorded fields changed')
        for key,value in row.items():
            if self.data[key].shape[1:]!=value.shape or self.data[key].dtype!=value.dtype:
                raise ValueError('Recorded shape or dtype changed')
            self.data[key][self.length]=value
        self.length+=1

    @property
    def values(self):
        if self.length!=self.total:
            raise ValueError('Incomplete history')
        return self.data if self.mapping else self.data['value']


def preallocated_collector(original):
    class Collector(original):
        def __init__(self, space, choices, capacity):
            super().__init__(space,choices,capacity)
            self.rows=PreallocatedHistory(len(self.choices))
            self.weights=PreallocatedHistory(len(self.choices),mapping=False)

        def arrays(self):
            return dict(self.rows.values,filtered_weights=self.weights.values,component_counts=self.counts)
    return Collector


def blocked_means(original, values: dict, weights: np.ndarray, rules: int, targets: list[int], block: int) -> dict:
    """Preserve within-trial reduction order while bounding temporary arrays."""
    if block < 1:
        raise ValueError('Positive trial-block size required')
    pieces=[]
    for start in range(0,len(weights),block):
        stop=min(len(weights),start+block)
        own={key:value if key=='component_counts' else value[start:stop] for key,value in values.items()}
        pieces.append(original(own,weights[start:stop],rules,targets))
    return {key:np.concatenate([part[key] for part in pieces],axis=0) for key in pieces[0]}


def worker(file: Path) -> None:
    spec=importlib.util.spec_from_file_location('precision_storage_capture',ROOT/BASE)
    capture=importlib.util.module_from_spec(spec);sys.modules[spec.name]=capture;spec.loader.exec_module(capture)
    job=capture.read(file)
    expected={ENTRY:capture.digest(ROOT/ENTRY)}
    if job.get('state_worker_entry')!=ENTRY or job.get('precision_storage_sources')!=expected:
        raise ValueError('Require the registered passive storage adapter')
    collector=capture.helper.IndividualStateCollector;means=capture.helper.state_means
    capture.helper.IndividualStateCollector=preallocated_collector(collector)
    capture.helper.state_means=lambda values,weights,rules,targets: blocked_means(
        means,values,weights,rules,targets,int(job['state_summary_block_trials']))
    try:
        capture.worker(file)
    finally:
        capture.helper.IndividualStateCollector=collector;capture.helper.state_means=means
    if expected!={ENTRY:capture.digest(ROOT/ENTRY)}:
        raise ValueError('Passive storage adapter changed during computation')
    path=Path(job['individual_target']).with_suffix('.storage.json')
    record=dict(status='complete',sources=expected,job_sha256=capture.digest(file),
        storage='preallocated arrays; original observer calculations; original state_means in trial blocks',
        block_trials=job['state_summary_block_trials'],scientific_capture_sources=capture.sources())
    with path.open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    worker(parser.parse_args().worker)
