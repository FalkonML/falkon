import math
import numpy as np
from scipy.spatial.distance import cdist

__all__ = ("naive_gaussian_kernel", "naive_sigmoid_kernel", "naive_laplacian_kernel",
           "naive_linear_kernel", "naive_polynomial_kernel", "naive_matern_kernel",)


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

def naive_matern_kernel(X1, X2, sigma, nu):
    pairwise_dists = cdist(X1 / sigma, X2 / sigma, 'euclidean')

    if nu == 0.5:
        K = np.exp(-pairwise_dists)
    elif nu == 1.5:
        K = pairwise_dists * math.sqrt(3)
        K = (1. + K) * np.exp(-K)
    elif nu == 2.5:
        K = pairwise_dists * math.sqrt(5)
        K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
    elif nu == np.inf:
        K = np.exp(-pairwise_dists ** 2 / 2.0)
    return K

