import pytest
import numpy as np
import torch
from scipy.spatial.distance import cdist
if False:
    from falkon import FalkonOptions
    from falkon.kernels import GaussianKernel
    from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
    from falkon.tests.gen_random import gen_random


n = 10
d = 2


def naive_gaussian_kernel(X1, X2, sigma):
    pairwise_dists = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma ** 2))


# @pytest.fixture(scope="module")
def A() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, d, 'float64', False, seed=92))


def _test_gk_gradient(A):
    sigma = torch.tensor([5.0]).requires_grad_()
    dk = DiffGaussianKernel(sigma)
    nk = GaussianKernel(sigma.item())
    vec = torch.randn(A.shape[0], 1, dtype=A.dtype)

    grad_outputs = torch.zeros(A.shape[0], 1, dtype=A.dtype)
    grad_outputs[0] = 1

    dk_val = dk.mmv(A, A, vec)
    dk_gd = torch.autograd.grad(dk_val, sigma, grad_outputs=grad_outputs)[0]

    pairwise_dists = torch.from_numpy(cdist(A.numpy(), A.numpy(), 'sqeuclidean')).to(dtype=A.dtype)
    gamma = -1 / (2 * sigma ** 2)
    jacobian = (pairwise_dists / sigma ** 3) * torch.exp(pairwise_dists * gamma)
    jacobian = jacobian @ vec
    nk_gd = jacobian.T @ grad_outputs

    print(dk_gd, nk_gd)
    np.testing.assert_allclose(dk_gd.item(), nk_gd.item(), rtol=1e-5)


def _test_diag_gk_gradient(A):
    opt = FalkonOptions(keops_active="no")
    sigma = torch.tensor([5.0, 6.0]).requires_grad_()
    vec = torch.randn(A.shape[0], 1, dtype=A.dtype)
    dk = DiffGaussianKernel(sigma, opt=opt)
    nk = GaussianKernel(sigma, opt=opt)

    grad_outputs = torch.zeros(A.shape[0], 1, dtype=A.dtype)
    grad_outputs[0] = 1
    grad_outputs[1] = 1

    dk_val = dk.mmv(A, A, vec)
    dk_gd = torch.autograd.grad(dk_val, sigma, grad_outputs=grad_outputs)[0]
    print(dk_gd)

    full_dists = nk(A, A)
    for i in range(d):
        Ai = A[:, i].reshape(-1, 1)
        dim_dists = torch.from_numpy(cdist(Ai.numpy(), Ai.numpy(), 'sqeuclidean')).to(dtype=A.dtype) / (sigma[i]**3)
        i_grad = ((dim_dists * full_dists) @ vec).T @ grad_outputs
        print("Grad %d (computed)=%f" % (i, i_grad.item()))
        np.testing.assert_allclose(i_grad.item(), dk_gd[i].item(), rtol=1e-5)


def test_scalar_gaussian_gradient():
    x1 = torch.tensor([1.0, 2.0])
    x2 = torch.tensor([2.5, 0.5])
    sigma = torch.tensor(5.0).requires_grad_()

    gk = torch.exp(- torch.sum((x1 - x2)**2) / (2 * sigma ** 2) )
    grad_gk = torch.autograd.grad(gk, sigma)[0]

    delta = torch.sum((x1 - x2)**2)
    grad_nk = (delta / (sigma ** 3)) * torch.exp(-delta / (2 * sigma ** 2))

    # print("Gradient (autograd) = %f" % (grad_gk))
    # print("Gradient (calc) = %f" % (grad_nk))
    np.testing.assert_allclose(grad_nk.item(), grad_gk.item(), rtol=1e-5)


def _test_scalar_dgauss_grad():
    dt = torch.float32
    x1 = torch.tensor([1.0, 2.5], dtype=dt)
    x2 = torch.tensor([9.5, 0.5], dtype=dt)
    sigma = torch.tensor([5.0, 6.0], dtype=dt).requires_grad_()

    gk = torch.exp(-0.5 * torch.sum((x1/sigma - x2/sigma)**2))
    grad_gk = torch.autograd.grad(gk, sigma)[0]

    sigma_15 = sigma ** 1.5
    grad_nk = (x1/sigma_15 - x2/sigma_15)**2 * torch.exp(-0.5 * torch.sum((x1/sigma - x2/sigma)**2))

    print()
    print("Gradient (autograd) = %s" % (grad_gk))
    print("Gradient (calc) = %s" % (grad_nk))
    np.testing.assert_allclose(grad_nk.detach().numpy(), grad_gk.detach().numpy(), rtol=1e-5)
