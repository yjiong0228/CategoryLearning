"""Numerical equivalence, ownership and bounded lifetime of distance shortcuts."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import pickle
import weakref

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.geometry.boundary import BoundaryGeometry
from src.Bayesian_state.hypothesis_space.geometry.distance_cache import ExactDistanceCache
from src.Bayesian_state.hypothesis_space.observation_model import ContinuousPartition
from src.Bayesian_state.hypothesis_space.spaces import build_continuous_hypothesis_space


def _geometry(**kwargs):
    return BoundaryGeometry(build_continuous_hypothesis_space(2, 2), **kwargs)


@pytest.mark.parametrize('n_cats', [2,4])
@pytest.mark.parametrize('mode', ['prototype','boundary'])
def test_zero_beta_matches_original_softmax_for_every_rule(n_cats, mode):
    partition = ContinuousPartition(4,n_cats)
    points = np.array([[0,0,0,0], [.5,.5,.5,.5], [1,1,1,1], [.1,.8,.3,.7]])
    geometry = partition.prototype_geometry if mode == 'prototype' else partition.boundary_geometry
    for hypothesis in range(partition.length):
        expected = geometry.category_probabilities(hypothesis, points, 0.)
        actual = partition.get_category_probabilities(hypothesis, [points], 0., distance_mode=mode)
        np.testing.assert_array_equal(actual, expected)
    # Both all-rule and mixed-beta likelihood normalization must stay identical.
    data = (points, [1,2,1,2], [1,0,.5,0])
    beta = np.zeros(partition.length)
    beta[:3] = [1.,5.,10.]
    fast = partition.calc_likelihood(range(partition.length), data, beta, distance_mode=mode)
    partition.zero_beta_fast_path = False
    original = partition.calc_likelihood(range(partition.length), data, beta, distance_mode=mode)
    np.testing.assert_array_equal(fast, original)


@pytest.mark.parametrize('mode', ['prototype','boundary'])
def test_zero_beta_really_skips_geometry_but_checks_inputs(monkeypatch, mode):
    partition = ContinuousPartition(2,2)
    geometry = partition.prototype_geometry if mode == 'prototype' else partition.boundary_geometry
    def forbidden(*args):
        raise AssertionError('Geometry called')
    monkeypatch.setattr(geometry, 'category_probabilities', forbidden)
    np.testing.assert_array_equal(partition.get_category_probabilities(0, [[.2,.3]], -0., mode), [[.5],[.5]])
    with pytest.raises(ValueError, match='Expected stimuli'):
        partition.get_category_probabilities(0, [np.ones((2,3))], 0., mode)
    with pytest.raises(IndexError):
        partition.get_category_probabilities(partition.length, [[[.2,.3]]], 0., mode)
    with pytest.raises(ValueError, match='Unsupported distance_mode'):
        partition.get_category_probabilities(0, [[[.2,.3]]], 0., 'invalid')
    for beta in (np.nextafter(0.,1.), 1.):
        with pytest.raises(AssertionError, match='Geometry called'):
            partition.get_category_probabilities(0, [[[.2,.3]]], beta, mode)
    for points in (np.array([[np.nan,.2]]), np.empty((0,2)), np.array([[1e300,.2]])):
        with pytest.raises(AssertionError, match='Geometry called'):
            partition.get_category_probabilities(0, [points], 0., mode)


def test_exact_cache_keys_include_rule_shape_and_actual_perceived_values():
    geometry = _geometry()
    points = np.array([[.2,.3],[.4,.5]])
    first = geometry.category_distances(0,points)
    np.testing.assert_array_equal(first, geometry.category_distances(0,np.asfortranarray(points)))
    assert geometry.distance_cache_info()['hits'] == 1
    points[0,0] = np.nextafter(points[0,0], 1.)
    geometry.category_distances(0,points)
    geometry.category_distances(0,points[:1])
    geometry.category_distances(1,points)
    assert geometry.distance_cache_info()['misses'] == 4
    reference = _geometry(distance_cache_max_entries=0)
    np.testing.assert_array_equal(geometry.category_distances(0,points), reference.category_distances(0,points))


def test_cached_arrays_cannot_be_corrupted_through_returned_buffers_or_headers():
    geometry = _geometry()
    points = np.array([[.2,.3]])
    actual = geometry.category_distances(0,points)
    expected = actual.copy()
    assert not actual.flags.writeable
    with pytest.raises(ValueError):
        actual[0,0] = 99.
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    with pytest.raises(ValueError):
        actual.base.setflags(write=True)
    actual.shape = (2,)
    actual.dtype = np.uint8
    np.testing.assert_array_equal(geometry.category_distances(0,points), expected)


@pytest.mark.parametrize('setting,value', [('projection_iterations',7), ('tolerance',1e-4),
                                         ('method','kkt_active_set_projection'), ('dykstra_backend','python')])
def test_solver_setting_changes_invalidate_cache(setting, value):
    geometry = _geometry()
    points = np.array([[.8,.2]])
    geometry.category_distances(0,points)
    if getattr(geometry,setting) == value:
        pytest.skip('Requested backend is already in use')
    setattr(geometry,setting,value)
    expected = geometry._category_distances_uncached(0,points)
    np.testing.assert_array_equal(geometry.category_distances(0,points), expected)
    info = geometry.distance_cache_info()
    assert info['invalidations'] == 1 and info['entries'] == 1


def test_space_identity_and_geometry_instance_isolate_entries():
    first = _geometry()
    other = _geometry()
    points = np.array([[.2,.3]])
    first.category_distances(0,points)
    assert other.distance_cache_info()['entries'] == 0
    first.space = build_continuous_hypothesis_space(2,2, center_band_tolerance=.2)
    np.testing.assert_array_equal(first.category_distances(0,points), first._category_distances_uncached(0,points))
    assert first.distance_cache_info()['invalidations'] == 1


def test_lru_eviction_and_byte_budget_do_not_change_distances():
    geometry = _geometry(distance_cache_max_entries=2, distance_cache_max_bytes=64)
    reference = _geometry(distance_cache_max_entries=0)
    points = [np.array([[x,.3]]) for x in (.1,.2,.4)]
    for point in (points[0],points[1],points[0],points[2]):
        np.testing.assert_array_equal(geometry.category_distances(0,point), reference.category_distances(0,point))
    info = geometry.distance_cache_info()
    assert info['entries'] == 2 and info['payload_bytes'] == 64 and info['hits'] == 1
    geometry.category_distances(0,points[1])
    assert geometry.distance_cache_info()['misses'] == 4
    # Oversize calls should not evict the useful single-trial entries.
    batch = np.repeat(points[0],3,axis=0)
    np.testing.assert_array_equal(geometry.category_distances(0,batch),reference.category_distances(0,batch))
    assert geometry.distance_cache_info()['entries'] == 2
    assert geometry.distance_cache_info()['bypasses'] == 1


def test_long_unique_stream_is_bounded_and_clear_releases_payload():
    geometry = _geometry(distance_cache_max_entries=7, distance_cache_max_bytes=160)
    for x in np.linspace(0,1,500):
        geometry.category_distances(0,[[x,.3]])
        info = geometry.distance_cache_info()
        assert info['entries'] <= 5 and info['payload_bytes'] <= 160
    geometry.clear_distance_cache()
    assert geometry.distance_cache_info()['entries'] == 0
    assert geometry.distance_cache_info()['payload_bytes'] == 0
    reference = weakref.ref(geometry)
    del geometry
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize('copy_method', [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_serialization_keeps_geometry_but_starts_with_empty_cache(copy_method):
    geometry = _geometry(distance_cache_max_entries=5, distance_cache_max_bytes=512)
    points = np.array([[.2,.3]])
    expected = geometry.category_distances(0,points).copy()
    copied = copy_method(geometry)
    assert copied.distance_cache_info()['entries'] == 0
    assert copied.distance_cache_info()['max_entries'] == 5
    np.testing.assert_array_equal(copied.category_distances(0,points), expected)
    geometry.clear_distance_cache()
    assert copied.distance_cache_info()['entries'] == 1


def test_shared_cache_threaded_reads_are_exact_and_accounted_once():
    geometry = _geometry(distance_cache_max_entries=3)
    points = np.array([[.2,.3]])
    expected = geometry._category_distances_uncached(0,points)
    with ThreadPoolExecutor(max_workers=4) as pool:
        arrays = list(pool.map(lambda _: geometry.category_distances(0,points), range(40)))
    for actual in arrays:
        np.testing.assert_array_equal(actual,expected)
    assert geometry.distance_cache_info()['entries'] == 1
    assert geometry.distance_cache_info()['payload_bytes'] == 32


@pytest.mark.parametrize('value', [-1, .5, True, np.nan])
def test_invalid_cache_limits_are_rejected(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        ExactDistanceCache(max_entries=value)
    with pytest.raises(ValueError, match='nonnegative integer'):
        ExactDistanceCache(max_bytes=value)


def test_zero_limits_disable_retention_without_changing_results():
    geometry = _geometry(distance_cache_max_bytes=0)
    points = np.array([[.2,.3]])
    for _ in range(2):
        np.testing.assert_array_equal(geometry.category_distances(0,points), geometry._category_distances_uncached(0,points))
    assert geometry.distance_cache_info()['entries'] == 0
    assert geometry.distance_cache_info()['bypasses'] == 2
