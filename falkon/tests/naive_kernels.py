import math

import numpy as np
import torch
from scipy.spatial.distance import cdist

__all__ = (
    "naive_gaussian_kernel",
    "naive_sigmoid_kernel",
    "naive_laplacian_kernel",
    "naive_linear_kernel",
    "naive_polynomial_kernel",
    "naive_matern_kernel",
    "naive_diff_gaussian_kernel",
    "naive_diff_sigmoid_kernel",
    "naive_diff_laplacian_kernel",
    "naive_diff_linear_kernel",
    "naive_diff_polynomial_kernel",
    "naive_diff_matern_kernel",
)


def naive_diff_gaussian_kernel(X1, X2, sigma):
    pairwise_dists = torch.cdist(X1 / sigma, X2 / sigma, p=2).square()
    return torch.exp(-0.5 * pairwise_dists)


def naive_diff_laplacian_kernel(X1, X2, sigma):
    # http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#laplacian
    pairwise_dists = torch.cdist(X1 / sigma, X2 / sigma, p=2)
    return torch.exp(-pairwise_dists)


def naive_diff_linear_kernel(X1, X2, beta, gamma):
    return naive_linear_kernel(X1, X2, beta, gamma)


def naive_diff_sigmoid_kernel(X1, X2, gamma, beta):
    out = X1 @ X2.T
    return torch.tanh(out * gamma + beta)


def naive_diff_polynomial_kernel(X1, X2, gamma, beta, degree):
    out = X1 @ X2.T
    return torch.pow(out * gamma + beta, degree)


def naive_diff_matern_kernel(X1, X2, sigma, nu):
    pairwise_dists = torch.cdist(X1 / sigma, X2 / sigma, p=2)

    if nu == 0.5:
        K = torch.exp(-pairwise_dists)
    elif nu == 1.5:
        K = pairwise_dists * math.sqrt(3)
        K = (1.0 + K) * torch.exp(-K)
    elif nu == 2.5:
        K = pairwise_dists * math.sqrt(5)
        K = (1.0 + K + K**2 / 3.0) * torch.exp(-K)
    elif nu == np.inf:
        K = torch.exp(-(pairwise_dists**2) / 2.0)
    return K


def naive_gaussian_kernel(X1, X2, sigma):
    pairwise_dists = cdist(X1, X2, "sqeuclidean")
    return np.exp(-pairwise_dists / (2 * sigma**2))


def naive_laplacian_kernel(X1, X2, sigma):
    pairwise_dists = cdist(X1, X2, "euclidean")
    return np.exp(-pairwise_dists / sigma)


def naive_linear_kernel(X1, X2, beta, gamma):
    return beta + gamma * X1 @ X2.T


def naive_sigmoid_kernel(X1, X2, alpha, beta):
    out = X1 @ X2.T
    return np.tanh(out * alpha + beta)


def naive_polynomial_kernel(X1, X2, alpha, beta, degree):
    out = X1 @ X2.T
    return np.power(out * alpha + beta, degree)


def naive_matern_kernel(X1, X2, sigma, nu):
    pairwise_dists = cdist(X1 / sigma, X2 / sigma, "euclidean")

    if nu == 0.5:
        K = np.exp(-pairwise_dists)
    elif nu == 1.5:
        K = pairwise_dists * math.sqrt(3)
        K = (1.0 + K) * np.exp(-K)
    elif nu == 2.5:
        K = pairwise_dists * math.sqrt(5)
        K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
    elif nu == np.inf:
        K = np.exp(-(pairwise_dists**2) / 2.0)
    return K
