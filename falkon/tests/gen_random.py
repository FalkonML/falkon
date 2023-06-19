import numpy as np
import scipy.sparse
import torch

from falkon.sparse.sparse_tensor import SparseTensor
from falkon.c_ext import copy_triang


def gen_random_multi(*sizes, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=tuple(sizes), dtype=dtype)
    if F:
        return np.asfortranarray(out)
    return out


def gen_random(a, b, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=(a, b), dtype=dtype)
    if F:
        return np.asfortranarray(out)
    return out


def gen_random_pd(t, dtype, F=False, seed=0):
    A = torch.from_numpy(gen_random(t, t, dtype, F, seed))
    copy_triang(A, upper=True)
    A *= 1
    A += 2
    A += torch.eye(t, dtype=A.dtype) * t * 4
    return A


def gen_sparse_matrix(a, b, dtype, density=0.1, seed=0, sparse_fromat='csr') -> SparseTensor:
    out = random_sparse(a, b, density=density, sparse_format=sparse_fromat, dtype=dtype, seed=seed)

    return SparseTensor.from_scipy(out)


def random_sparse(m, n, density=0.01, sparse_format='coo', dtype=None,
                  seed=None, data_rvs=None):
    # noinspection PyArgumentList
    dtype = np.dtype(dtype)
    mn = m * n
    tp = np.intc
    if mn > np.iinfo(tp).max:
        tp = np.int64

    # Number of non zero values
    k = int(density * m * n)

    random_state = np.random.RandomState(seed)

    if data_rvs is None:
        data_rvs = random_state.rand

    generator = np.random.default_rng(seed)
    ind = generator.choice(mn, size=k, replace=False)

    j = np.floor(ind * 1. / m).astype(tp, copy=False)
    i = (ind - j * m).astype(tp, copy=False)
    # noinspection PyArgumentList
    vals = data_rvs(k).astype(dtype, copy=False)
    return scipy.sparse.coo_matrix((vals, (i, j)), shape=(m, n)).asformat(sparse_format, copy=False)
