import numpy as np
import scipy.sparse

from falkon.sparse.sparse_tensor import SparseTensor
from falkon.la_helpers.cyblas import copy_triang


def gen_random(a, b, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=(a, b), dtype=dtype)
    if F:
        return np.asfortranarray(out)
    return out


def gen_random_pd(t, dtype, F=False, seed=0):
    A = gen_random(t, t, dtype, F, seed)
    copy_triang(A, upper=True)
    #A += A.T
    #A *= 2
    #A += 20
    A *= 1
    A += 2
    A.flat[::t + 1] += t*4
    return A


def gen_sparse_matrix(a, b, dtype, density=0.1, seed=0) -> SparseTensor:
    out = random_sparse(a, b, density=density, format='csr', dtype=dtype, seed=seed)

    return SparseTensor.from_scipy(out)


def random_sparse(m, n, density=0.01, format='coo', dtype=None,
                  seed=None, data_rvs=None):
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
    vals = data_rvs(k).astype(dtype, copy=False)
    return scipy.sparse.coo_matrix((vals, (i, j)), shape=(m, n)).asformat(format, copy=False)