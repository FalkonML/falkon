#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:49:21 2017

@author: alessandro
"""
import functools
import warnings
from abc import ABC
from typing import Optional, Union

import torch

from falkon.options import FalkonOptions
from falkon.sparse.sparse_ops import sparse_matmul
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.kernels import Kernel, KeopsKernelMixin

__all__ = ("LinearKernel", "PolynomialKernel", "SigmoidKernel")

_float_sc_type = Union[torch.Tensor, float]


def extract_float(d: _float_sc_type) -> float:
    if isinstance(d, torch.Tensor):
        try:
            # tensor.item() works if tensor is a scalar, otherwise it throws
            # a value error.
            return d.item()
        except ValueError:
            raise ValueError("Item is not a scalar")
    else:
        try:
            return float(d)
        except TypeError:
            raise TypeError("Item must be a scalar or a tensor.")


class DotProdKernel(Kernel, ABC):
    """Base class for dot-product based kernels
    The classes inheriting from `DotProdKernel` are all kernels based on a dot product
    between the input data points (i.e. k(x, x') = x.T @ x' ).

    This class supports sparse data.

    Parameters
    ----------
    name : str
        Descriptive name of the specialized kernel
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.

    Notes
    ------
    Classes which inherit from this one should implement the `_transform` method to modify
    the output of the dot-product (e.g. scale it).
    """

    kernel_type = "dot-product"

    def __init__(self, name, opt: Optional[FalkonOptions] = None):
        super().__init__(name, self.kernel_type, opt)

    def _prepare(self, X1, X2):
        return None

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        return None

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        out.addmm_(X1, X2)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        return sparse_matmul(X1, X2, out)


class LinearKernel(DotProdKernel, KeopsKernelMixin):
    """Linear Kernel with optional scaling and translation parameters.

    The kernel implemented here is the covariance function in the original
    input space (i.e. `X @ X.T`) with optional parameters to translate
    and scale the kernel: `beta + 1/(sigma**2) * X @ X.T`

    Parameters
    -----------
    beta : float-like
        Additive constant for the kernel, default: 0.0
    sigma : float-like
        Multiplicative constant for the kernel. The kernel will
        be multiplied by the inverse of sigma squared. Default: 1.0
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.

    Examples
    --------
    >>> k = LinearKernel(beta=0.0, sigma=2.0)
    >>> X = torch.randn(100, 3)  # 100 samples in 3 dimensions
    >>> kernel_matrix = k(X, X)
    >>> torch.testing.assert_allclose(kernel_matrix, X @ X.T * (1/2**2))
    """

    def __init__(self,
                 beta: _float_sc_type = 0.0,
                 sigma: _float_sc_type = 1.0,
                 opt: Optional[FalkonOptions] = None):
        super().__init__("Linear", opt)

        self.beta = torch.tensor(extract_float(beta), dtype=torch.float64)
        self.sigma = torch.tensor(extract_float(sigma), dtype=torch.float64)
        if self.sigma == 0:
            self.gamma: torch.Tensor = torch.tensor(0.0, dtype=torch.float64)
        else:
            # noinspection PyTypeChecker
            self.gamma: torch.Tensor = 1 / self.sigma**2

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        formula = '(gamma * (X | Y) + beta) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'gamma = Pm(1)',
            'beta = Pm(1)'
        ]
        other_vars = [
            torch.tensor([self.gamma]).to(dtype=X1.dtype, device=X1.device),
            torch.tensor([self.beta]).to(dtype=X1.dtype, device=X1.device)
        ]
        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _finalize(self, A, d):
        gamma = self.gamma.to(A)
        beta = self.beta.to(A)
        A.mul_(gamma)
        A.add_(beta)
        return A

    def __str__(self):
        return f"LinearKernel(sigma={self.sigma})"

    def __repr__(self):
        return self.__str__()


class PolynomialKernel(DotProdKernel, KeopsKernelMixin):
    r"""Polynomial kernel with multiplicative and additive constants.

    Follows the formula

    .. math::

        (\alpha * X_1^\top X_2 + \beta)^{\mathrm{degree}}

    Where all operations apart from the matrix multiplication are taken element-wise.

    Parameters
    ----------
    alpha : float-like
        Multiplicative constant
    beta : float-like
        Additive constant
    degree : float-like
        Power of the polynomial kernel
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.
    """
    def __init__(self,
                 alpha: _float_sc_type,
                 beta: _float_sc_type,
                 degree: _float_sc_type,
                 opt: Optional[FalkonOptions] = None):
        super().__init__("Polynomial", opt)

        self.alpha = torch.tensor(extract_float(alpha), dtype=torch.float64)
        self.beta = torch.tensor(extract_float(beta), dtype=torch.float64)
        self.degree = torch.tensor(extract_float(degree), dtype=torch.float64)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'alpha = Pm(1)',
            'beta = Pm(1)'
        ]
        other_vars = [
            torch.tensor([self.alpha]).to(dtype=X1.dtype, device=X1.device),
            torch.tensor([self.beta]).to(dtype=X1.dtype, device=X1.device)
        ]

        is_int_pow = self.degree == self.degree.to(dtype=torch.int32)
        if is_int_pow:
            formula = f'Pow((alpha * (X | Y) + beta), {int(self.degree.item())}) * v'
        else:
            formula = 'Powf((alpha * (X | Y) + beta), degree) * v'
            aliases.append('degree = Pm(1)')
            other_vars.append(torch.tensor([self.degree]).to(dtype=X1.dtype, device=X1.device))

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _finalize(self, A, d):
        alpha = self.alpha.to(A)
        beta = self.beta.to(A)
        degree = self.degree.to(A)
        A.mul_(alpha)
        A.add_(beta)
        A.pow_(degree)
        return A

    def __str__(self):
        return f"PolynomialKernel(alpha={self.alpha}, beta={self.beta}, degree={self.degree})"

    def __repr__(self):
        return self.__str__()


class SigmoidKernel(DotProdKernel, KeopsKernelMixin):
    r"""Sigmoid (or hyperbolic tangent) kernel function, with additive and multiplicative constants.

    Follows the formula

    .. math::

        k(x, y) = \tanh(\alpha x^\top y + \beta)

    Parameters
    ----------
    alpha : float-like
        Multiplicative constant
    beta : float-like
        Multiplicative constant
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.


    """

    def __init__(self,
                 alpha: _float_sc_type,
                 beta: _float_sc_type,
                 opt: Optional[FalkonOptions] = None):
        super().__init__("Sigmoid", opt)
        self.alpha = torch.tensor(extract_float(alpha), dtype=torch.float64)
        self.beta = torch.tensor(extract_float(beta), dtype=torch.float64)

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            warnings.warn("KeOps implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation.")

        return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            warnings.warn("KeOps implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation.")
        return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _finalize(self, A, d):
        alpha = self.alpha.to(A)
        beta = self.beta.to(A)
        A.mul_(alpha)
        A.add_(beta)
        A.tanh_()
        return A

    def __str__(self):
        return f"SigmoidKernel(alpha={self.alpha}, beta={self.beta})"

    def __repr__(self):
        return self.__str__()