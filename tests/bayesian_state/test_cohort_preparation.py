"""Guard reuse policy and roster integrity without new scientific computation."""
from copy import deepcopy
import json

import pytest

from src.Bayesian_state.workflows.runs.prepare_model_0826_cohort import (
    check_receipts, policy_equal, split_subjects,
)
from src.Bayesian_state.optimization.adaptive_runtime import digest


def test_only_reporting_label_can_differ():
    old = {'analysis_id': 'old', 'base_seed': 42, 'search': {'max_cycles': 2}, 'precision': {'tolerance': .005}}
    new = deepcopy(old)
    new['analysis_id'] = 'new'
    assert policy_equal(old, new)
    new['precision']['tolerance'] = .006
    assert not policy_equal(old, new)
    new = deepcopy(old)
    new['base_seed'] += 1
    assert not policy_equal(old, new)
    assert old['analysis_id'] == 'old'


def test_reuse_inventory_rejects_duplicate_or_unknown_subjects():
    specs = [{'subject': s} for s in [102, 118, 122, 129]]
    assert split_subjects(specs, [122, 102, 118]) == [129]
    for reused in ([102, 102], [999]):
        with pytest.raises(ValueError):
            split_subjects(specs, reused)


def test_receipts_cover_exactly_unchanged_cache(tmp_path):
    cache = tmp_path/'cache/102/R2_seed1/p.npz'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'test cache identity')
    receipt = tmp_path/'batches/first/scores.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'cache_sha256': {str(cache.relative_to(tmp_path)): digest(cache)}}))
    assert check_receipts(tmp_path) == (1, 1)
    cache.write_bytes(b'altered')
    with pytest.raises(ValueError, match='checksum'):
        check_receipts(tmp_path)
    cache.unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        check_receipts(tmp_path)
