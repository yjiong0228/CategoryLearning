"""Run-log indexing must preserve Python sequence order, including cached slices."""

import pytest

from src.Bayesian_state.utils.streaming import StreamList


@pytest.mark.parametrize("buffer_size", [2, 16])
@pytest.mark.parametrize("index", [
    slice(None), slice(None, None, 2), slice(1, 6, 2),
    slice(None, None, -1), slice(4, 0, -2), slice(-5, -1),
    slice(10, 20), slice(4, 1), slice(-20, 20, 3),
])
def test_slices_match_list_with_and_without_cached_indices(tmp_path, buffer_size, index):
    values = list(range(6))
    stream = StreamList(str(tmp_path / "runs.gz"), 0, buffer_size=buffer_size)
    stream.extend(values)
    assert stream[index] == values[index]
    assert stream[1] == 1
    assert stream[-1] == 5
    assert stream[index] == values[index]
    assert list(stream) == values


def test_cached_slice_does_not_shift_later_records(tmp_path):
    stream = StreamList(str(tmp_path / "runs.gz"), 0)
    stream.extend([0, 1, 2, 3])
    assert stream[1] == 1
    assert stream[:3] == [0, 1, 2]
    assert stream[2] == 2


def test_index_and_slice_boundaries(tmp_path):
    stream = StreamList(str(tmp_path / "runs.gz"), 0)
    assert stream[::-1] == []
    stream.extend([10, 20])
    assert stream[-2] == 10
    for index in [2, -3]:
        with pytest.raises(IndexError):
            stream[index]
    with pytest.raises(ValueError):
        stream[::0]


def test_slicing_does_not_change_append_or_reopen(tmp_path):
    path = str(tmp_path / "runs.gz")
    stream = StreamList(path, 0, buffer_size=2)
    stream.extend([0, 1, 2])
    assert stream[::-1] == [2, 1, 0]
    stream.append(3)
    assert stream[1::2] == [1, 3]
    reopened = StreamList(path, 4)
    assert reopened[:] == [0, 1, 2, 3]
