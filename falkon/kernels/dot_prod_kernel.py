#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:49:21 2017

@author: alessandro
"""
import functools
from abc import ABC

import torch

from falkon.sparse.sparse_ops import sparse_matmul
from falkon.sparse.sparse_tensor import SparseTensor
from . import Kernel, KeopsKernelMixin

__all__ = ("LinearKernel", "PolynomialKernel", "ExponentialKernel")


def extract_float(d):
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

    Arguments:
    ----------
     - name : str
        Descriptive name of the specialized kernel
     - opt : CompOpt or dict or None
        Options which will be passed to the kernel operations
     - **kwargs
         Additional options which will be passed to operations involving this kernel.

    Notes:
    ------
    Classes which inherit from this one should implement the `_transform` method to modify
    the output of the dot-product (e.g. scale it).
    """

    kernel_type = "dot-product"

    def __init__(self, name, opt=None, **kw):
        super().__init__(name, self.kernel_type, opt, **kw)

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

    Parameters:
    -----------
     - beta : float (optional, default 0.0)
        Additive constant for the kernel
     - sigma : float (optional, default 1.0)
        Multiplicative constant for the kernel. The kernel will
        be multiplied by the inverse of sigma squared.
     - opt : Union(Dict, CompOpt)
        Options dictionary.
     - kw : dict
        Additional options passed through keywords (see the `opt` argument
        for a description of available options).
    """
    def __init__(self, beta=0, sigma=1, opt=None, **kw):
        super().__init__("Linear", opt, **kw)

        self.beta = torch.tensor(extract_float(beta), dtype=torch.float64)
        self.sigma = torch.tensor(extract_float(sigma), dtype=torch.float64)
        if self.sigma == 0:
            self.gamma: torch.Tensor = torch.tensor(0.0, dtype=torch.float64)
        else:
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
            torch.tensor([self.gamma]).to(dtype=X1.dtype),
            torch.tensor([self.beta]).to(dtype=X1.dtype)
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

    def _prepare(self, X1, X2, **kwargs):
        # TODO: Most likely we just keep the self.beta.to(X1).
        if 'u' in kwargs:
            vec = kwargs['u']
        elif 'v' in kwargs:
            vec = kwargs['v']
        else:
            return self.beta.to(X1)

        # Here we need the vector!
        return self.beta.to(vec) * vec.sum(0)

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        return self.beta.to(dtype=X1.dtype, device=X1.device)

    def _finalize(self, A, d):
        gamma = self.gamma.to(A)
        A.mul_(gamma)
        A.add_(d.to(A))
        return A

    def __str__(self):
        return f"LinearKernel(sigma={self.sigma})"

    def __repr__(self):
        return self.__str__()


class PolynomialKernel(DotProdKernel, KeopsKernelMixin):
    def __init__(self, alpha, beta, degree, opt=None, **kw):
        super().__init__("Polynomial", opt, **kw)

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
            torch.tensor([self.alpha]).to(dtype=X1.dtype),
            torch.tensor([self.beta]).to(dtype=X1.dtype)
        ]

        is_int_pow = self.degree == self.degree.to(dtype=torch.int32)
        if is_int_pow:
            formula = f'Pow((alpha * (X | Y) + beta), {int(self.degree.item())}) * v'
        else:
            formula = f'Powf((alpha * (X | Y) + beta), degree) * v'
            aliases.append('degree = Pm(1)')
            other_vars.append(torch.tensor([self.degree]).to(dtype=X1.dtype))

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


class ExponentialKernel(DotProdKernel, KeopsKernelMixin):
    def __init__(self, alpha, opt=None, **kw):
        super().__init__("Exponential", opt, **kw)
        self.alpha = torch.tensor(extract_float(alpha), dtype=torch.float64)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        formula = f'Exp(alpha * (X | Y)) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'alpha = Pm(1)',
        ]
        other_vars = [torch.tensor([self.alpha]).to(dtype=X1.dtype)]

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
        A.mul_(alpha)
        A.exp_()
        return A

    def __str__(self):
        return f"ExponentialKernel(alpha={self.alpha})"

    def __repr__(self):
        return self.__str__()
