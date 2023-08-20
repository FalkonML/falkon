import numbers

import numpy as np

from .switches import decide_cuda
from .tictoc import TicToc
from .threading import PropagatingThread

__all__ = ("PropagatingThread", "TicToc", "decide_cuda", "check_random_generator")


def check_random_generator(seed):
    """Turn seed into a np.random.Generator instance

    Parameters
    ----------
    seed : None | int | instance of Generator
        If seed is None, return the Generator singleton used by np.random.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)
