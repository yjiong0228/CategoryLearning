"""Check retained distance-cache memory under a long stream of unique inputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tracemalloc

import numpy as np

from ...hypothesis_space.geometry import BoundaryGeometry
from ...hypothesis_space.spaces import build_continuous_hypothesis_space


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=20000)
    args = parser.parse_args()
    if args.queries < 10000:
        parser.error('Use at least 10000 unique queries to exceed the default entry limit')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    space = build_continuous_hypothesis_space(4,2)
    rows = []
    for name, kwargs in (('default',{}), ('small_byte_budget',{'distance_cache_max_bytes':4096})):
        geometry = BoundaryGeometry(space, **kwargs)
        geometry.category_distances(0,[[.1,.2,.3,.4]])  # Warm compiled projection first.
        geometry.clear_distance_cache()
        gc.collect()
        tracemalloc.start()
        point = np.array([[.1,.2,.3,.4]])
        snapshots = []
        for index in range(args.queries):
            point[0,0] = (index + .5) / args.queries
            geometry.category_distances(0,point)
            if index+1 in {512,4096,10000,args.queries}:
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                info = geometry.distance_cache_info()
                assert info['entries'] <= info['max_entries']
                assert info['payload_bytes'] <= info['max_bytes']
                snapshots.append({'queries':index+1,'python_traced_current_bytes':current,
                    'python_traced_peak_bytes':peak, **info})
        # A 100,000-row similarity batch must not be copied into the cache.
        before = geometry.distance_cache_info()
        large = np.zeros((100000,4))
        geometry.category_distances(0,large)
        after = geometry.distance_cache_info()
        assert (after['entries'],after['payload_bytes']) == (before['entries'],before['payload_bytes'])
        assert after['bypasses'] == before['bypasses'] + 1
        del large
        geometry.clear_distance_cache()
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        assert geometry.distance_cache_info()['entries'] == 0
        rows.append({'name':name,'snapshots':snapshots,'after_clear_current_bytes':current,
            'oversize_batch_bypassed':True})
        tracemalloc.stop()
    source = Path('src/Bayesian_state/hypothesis_space/geometry/distance_cache.py')
    report = {'queries_per_cache':args.queries,'results':rows,
        'cache_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'measurement':'tracemalloc current/peak allocations; not total process RSS; cache payload excludes Python bookkeeping'}
    (args.output_dir/'memory.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
