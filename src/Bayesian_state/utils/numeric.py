"""
Basics
"""
import numpy as np
from scipy.spatial.distance import cdist


def softmax(mat: np.ndarray, beta: float = 1., axis=None) -> np.ndarray:
    """
    Softmax on d
    """
    d = len(mat.shape)
    m = mat - np.max(mat)
    ret = np.exp(m * beta)
    match axis:
        case None:
            return ret / np.sum(ret)
        case _:
            axis = axis % d
            return ret / np.sum(ret, axis=axis, keepdims=True)


def euc_dist(this: np.ndarray, other: np.ndarray):
    """
    Euclidean distance with shape adjustment
    """
    this_shape = this.shape
    other_shape = other.shape
    assert this_shape[-1] == other_shape[-1]
    mat = cdist(this.reshape(-1, this_shape[-1]),
                other.reshape(-1, other_shape[-1]))

    return mat.reshape(*this_shape[:-1], *other_shape[:-1])


def entropy(dist: np.ndarray):
    """
    Calculate the entropy
    Assert: np.sum(dist) == 1
    """
    dist = dist[dist != 0]
    return np.sum(-dist * np.log(dist))
