# Third party code
#
# The following code are copied or modified from:
# https://github.com/openai/evolution-strategies-starter.

import numpy as np


def compute_ranks(x):
    """Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in
    [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """Return ranks that is normliazed to [-0.5, 0.5] with the rewards as input.
    Args:
        x(np.array): an array of rewards.
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


def itergroups(items, group_size):
    """An iterator that iterates a list with batch data."""
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    """Compute the gradients for updating the parameters.
    Args:
        weights(np.array): the nomalized rewards computed by the function `compute_centered_ranks`.
        vecs(np.array): the noise added to the parameters.
        batch_size(int): the batch_size for speeding up the computation.
    Return:
    total(np.array): aggregated gradient. 
    """
    total = 0
    for batch_weights, batch_vecs in zip(
            itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(
            np.asarray(batch_weights, dtype=np.float32),
            np.asarray(batch_vecs, dtype=np.float32))
    return total


def unflatten(flat_array, array_shapes):
    i = 0
    arrays = []
    for shape in array_shapes:
        size = np.prod(shape, dtype=np.int)
        array = flat_array[i:(i + size)].reshape(shape)
        arrays.append(array)
        i += size
    assert len(flat_array) == i
    return arrays
