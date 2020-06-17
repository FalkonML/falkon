import numpy as np
import scipy.sparse
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.cyblas import copy_triang
from numpy.random import PCG64
from scipy.spatial.distance import cdist

__all__ = ("gen_random", "gen_random_pd", "gen_sparse_matrix", "naive_gaussian_kernel",
           "naive_linear_kernel", "naive_exponential_kernel",
           "naive_polynomial_kernel", "mw_gaussian_kernel")


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


def naive_gaussian_kernel(X1, X2, sigma):
    pairwise_dists = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma ** 2))


def mw_gaussian_kernel(X1, X2, S):
    R = np.linalg.cholesky(S)
    sq1 = np.sum((X1 @ R) ** 2, 1).reshape((-1, 1))
    sq2 = np.sum((X2 @ R) ** 2, 1).reshape((-1, 1))

    out = -2 * (X1 @ S @ X2.T)
    out = (sq2.T + out) + sq1
    return np.exp(-0.5 * out)


def naive_linear_kernel(X1, X2, beta, sigma):
    return beta + (1 / sigma ** 2) * X1 @ X2.T


def naive_exponential_kernel(X1, X2, alpha):
    out = X1 @ X2.T
    return np.exp(out * alpha)


def naive_polynomial_kernel(X1, X2, alpha, beta, degree):
    out = X1 @ X2.T
    return np.power(out * alpha + beta, degree)
