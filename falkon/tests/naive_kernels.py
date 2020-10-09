import numpy as np
from scipy.spatial.distance import cdist

__all__ = ("naive_gaussian_kernel", "naive_sigmoid_kernel", "naive_laplacian_kernel",
           "naive_linear_kernel", "naive_polynomial_kernel")


def naive_gaussian_kernel(X1, X2, sigma):
    pairwise_dists = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma ** 2))


def naive_laplacian_kernel(X1, X2, sigma):
    pairwise_dists = cdist(X1, X2, 'euclidean')
    return np.exp(-pairwise_dists / sigma)


def naive_linear_kernel(X1, X2, beta, sigma):
    return beta + (1 / sigma ** 2) * X1 @ X2.T


def naive_sigmoid_kernel(X1, X2, alpha, beta):
    out = X1 @ X2.T
    return np.tanh(out * alpha + beta)


def naive_polynomial_kernel(X1, X2, alpha, beta, degree):
    out = X1 @ X2.T
    return np.power(out * alpha + beta, degree)
