import pytest
import torch

from falkon.kernels import Kernel
from falkon.tests.gen_random import gen_random

n = 20
m = 5
d = 3
t = 2


@pytest.fixture(scope="module")
def A() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, d, "float32", False, seed=92))


@pytest.fixture(scope="module")
def B() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, d, "float32", False, seed=93))


@pytest.fixture(scope="module")
def v() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, t, "float32", False, seed=94))


@pytest.fixture(scope="module")
def w() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, t, "float32", False, seed=95))


class BasicLinearKernel(Kernel):
    def __init__(self, lengthscale, options):
        super().__init__("basic_linear", options)
        if isinstance(lengthscale, float):
            self.lengthscale = torch.tensor(lengthscale)
        else:
            self.lengthscale = lengthscale

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool, **kwargs) -> torch.Tensor:
        # To support different devices/data types, you must make sure
        # the lengthscale is compatible with the data.
        lengthscale = self.lengthscale.to(device=X1.device, dtype=X1.dtype)

        scaled_X1 = X1 * lengthscale

        if diag:
            out.copy_(torch.sum(scaled_X1 * X2, dim=-1))
        else:
            # The dot-product row-by-row on `X1` and `X2` can be computed
            # on many rows at a time with matrix multiplication.
            out = torch.matmul(scaled_X1, X2.T, out=out)

        return out

    def compute_sparse(self, X1, X2, out, diag, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Sparse not implemented")


def basic_linear_kernel(X1, X2, lengthscale):
    return (X1 * lengthscale) @ X2.T


def test_mmv(A, B, v):
    lscale = 3.0
    k = BasicLinearKernel(lscale, None)

    out = k.mmv(A, B, v)
    torch.testing.assert_close(out, basic_linear_kernel(A, B, lscale) @ v)


def test_mmv_out(A, B, v):
    lscale = 3.0
    k = BasicLinearKernel(lscale, None)

    out = torch.empty(A.shape[0], v.shape[-1])
    k.mmv(A, B, v, out=out)
    torch.testing.assert_close(out, basic_linear_kernel(A, B, lscale) @ v)


def test_dmmv(A, B, v, w):
    lscale = 3.0
    k = BasicLinearKernel(lscale, None)

    out = k.dmmv(A, B, v, w)
    K = basic_linear_kernel(A, B, lscale)
    torch.testing.assert_close(out, K.T @ (K @ v + w))


def test_dmmv_out(A, B, v, w):
    lscale = 3.0
    k = BasicLinearKernel(lscale, None)

    out = torch.empty(B.shape[0], w.shape[-1])
    k.dmmv(A, B, v, w, out=out)
    K = basic_linear_kernel(A, B, lscale)
    torch.testing.assert_close(out, K.T @ (K @ v + w))


class BasicLinearKernelWithKwargs(Kernel):
    """
    Kwargs is going to be a binary mask, selecting only certain features from X1, X2
    """

    def __init__(self, lengthscale, options):
        super().__init__("basic_linear_kwargs", options)
        if isinstance(lengthscale, float):
            self.lengthscale = torch.tensor(lengthscale)
        else:
            self.lengthscale = lengthscale

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool, **kwargs) -> torch.Tensor:
        lengthscale = self.lengthscale.to(device=X1.device, dtype=X1.dtype)
        indices_x1 = kwargs["indices_m1"]
        indices_x2 = kwargs["indices_m2"]

        X1_ = X1 * indices_x1
        X2_ = X2 * indices_x2

        scaled_X1 = X1_ * lengthscale

        # The dot-product row-by-row on `X1` and `X2` can be computed
        # on many rows at a time with matrix multiplication.
        out = torch.matmul(scaled_X1, X2_.T, out=out)

        return out

    def compute_sparse(self, X1, X2, out, diag, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Sparse not implemented")


def basic_linear_kernel_with_kwargs(X1, X2, indices_x1, indices_x2, lengthscale):
    return ((X1 * indices_x1) * lengthscale) @ (X2 * indices_x2).T


def test_mmv_kwargs(A, B, v):
    lscale = 3.0
    indices_m1 = torch.bernoulli(torch.full_like(A, 0.5))
    indices_m2 = torch.bernoulli(torch.full_like(B, 0.5))
    k = BasicLinearKernelWithKwargs(lscale, None)
    out = k.mmv(A, B, v, kwargs_m1={"indices_m1": indices_m1}, kwargs_m2={"indices_m2": indices_m2})
    torch.testing.assert_close(out, basic_linear_kernel_with_kwargs(A, B, indices_m1, indices_m2, lscale) @ v)


def test_mmv_kwargs_out(A, B, v):
    lscale = 3.0
    indices_m1 = torch.bernoulli(torch.full_like(A, 0.5))
    indices_m2 = torch.bernoulli(torch.full_like(B, 0.5))
    k = BasicLinearKernelWithKwargs(lscale, None)

    out = torch.empty(A.shape[0], v.shape[-1])
    k.mmv(A, B, v, out=out, kwargs_m1={"indices_m1": indices_m1}, kwargs_m2={"indices_m2": indices_m2})
    torch.testing.assert_close(out, basic_linear_kernel_with_kwargs(A, B, indices_m1, indices_m2, lscale) @ v)


def test_dmmv_kwargs(A, B, v, w):
    lscale = 3.0
    indices_m1 = torch.bernoulli(torch.full_like(A, 0.5))
    indices_m2 = torch.bernoulli(torch.full_like(B, 0.5))
    k = BasicLinearKernelWithKwargs(lscale, None)

    out = k.dmmv(A, B, v, w, kwargs_m1={"indices_m1": indices_m1}, kwargs_m2={"indices_m2": indices_m2})
    K = basic_linear_kernel_with_kwargs(A, B, indices_m1, indices_m2, lscale)
    torch.testing.assert_close(out, K.T @ (K @ v + w))


def test_dmmv_kwargs_out(A, B, v, w):
    lscale = 3.0
    indices_m1 = torch.bernoulli(torch.full_like(A, 0.5))
    indices_m2 = torch.bernoulli(torch.full_like(B, 0.5))
    k = BasicLinearKernelWithKwargs(lscale, None)

    out = torch.empty(B.shape[0], w.shape[-1])
    k.dmmv(A, B, v, w, out=out, kwargs_m1={"indices_m1": indices_m1}, kwargs_m2={"indices_m2": indices_m2})
    K = basic_linear_kernel_with_kwargs(A, B, indices_m1, indices_m2, lscale)
    torch.testing.assert_close(out, K.T @ (K @ v + w))


if __name__ == "__main__":
    pytest.main()
