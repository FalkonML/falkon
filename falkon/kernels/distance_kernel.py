#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:49:21 2017

@author: alessandro
"""
import collections
import functools
import math
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from falkon.options import BaseOptions, FalkonOptions

from falkon.sparse import sparse_ops
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.kernels import Kernel, KeopsKernelMixin

DistKerContainer = collections.namedtuple('DistKerContainer', ['sq1', 'sq2'])


class L2DistanceKernel(Kernel, ABC):
    """Base class for L2-based kernels

    Such kernels are characterized by the squared norm of the difference between each input
    sample. This involves computing the squared norm in `_prepare`, and a simple matrix
    multiplication in `_apply`.
    In `_finalize` the squared norm and matrix multiplication are added together to form
    the kernel matrix.
    Subclasses should implement the `_transform` method which applies additional elementwise
    transformations to the kernel matrix. `_transform` is called after `_finalize`.

    This class supports sparse data.

    Parameters
    ----------
    name : str
        Descriptive name of the specialized kernel
    opt : CompOpt or dict or None
        Options which will be passed to the kernel operations

    Notes
    ------
    To efficiently compute kernels of the form k(x, x') = ||x - x'||^2 between two matrices of
    data-points we decompose the squared norm of the difference into 3 terms:
    ||X||^2 and -2*XX'^T and ||X'||^2
    The first and third term are calculated in the `_prepare` method while the second is
    calculated in the `_apply` method. Finally the three terms are combined in the `_finalize`
    method.
    """
    kernel_type = "l2distance"

    def __init__(self, name, opt: Optional[FalkonOptions] = None):
        super().__init__(name, self.kernel_type, opt)

    def _prepare(self, X1: torch.Tensor, X2: torch.Tensor) -> DistKerContainer:
        return DistKerContainer(
            sq1=torch.norm(X1, p=2, dim=1, keepdim=True).pow_(2),
            sq2=torch.norm(X2, p=2, dim=1, keepdim=True).pow_(2)
        )

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor) -> DistKerContainer:
        sq1 = torch.empty(X1.size(0), dtype=X1.dtype, device=X1.device)
        sparse_ops.sparse_square_norm(X1, sq1)
        sq2 = torch.empty(X2.size(0), dtype=X1.dtype, device=X1.device)
        sparse_ops.sparse_square_norm(X2, sq2)
        return DistKerContainer(
            sq1=sq1.reshape(-1, 1), sq2=sq2.reshape(-1, 1)
        )

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor) -> None:
        out.addmm_(X1, X2)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor) -> torch.Tensor:
        return sparse_ops.sparse_matmul(X1, X2, out)

    def _finalize(self, A: torch.Tensor, d: DistKerContainer) -> torch.Tensor:
        A.mul_(-2.0)
        A.add_(d.sq1.to(A))
        A.add_(d.sq2.to(A).t())
        A.clamp_min_(0)
        return self._transform(A)

    @abstractmethod
    def _transform(self, A: torch.Tensor):
        pass


class GaussianKernel(L2DistanceKernel, KeopsKernelMixin):
    """Class for computing the Gaussian kernel and related kernel-vector products

    The Gaussian kernel is one of the most common and effective kernel embeddings
    since it is infinite dimensional, and governed by a single parameter. The kernel length-scale
    determines the width of the Gaussian distribution which is placed on top of each point.
    A larger sigma corresponds to a wide Gaussian, so that the relative influence of far away
    points will be high for computing the kernel at a given datum.
    On the opposite side of the spectrum, a small sigma means that only nearby points will
    influence the kernel.

    Parameters
    -----------
    sigma
        The length-scale of the kernel.
        This can be a scalar, and then it corresponds to the standard deviation
        of Gaussian distribution from which the kernel is derived.
        It can also be a vector of size `d` or a matrix of size `d*d` where `d`
        is the dimensionality of the data which will be used with the kernel.

        In this case sigma will be the inverse square of the standard deviation,
        so for example converting from the vectorial sigma to the scalar sigma can
        be done by `vec_sigma = -1/(sigma**2)`
    opt
        Additional options to be forwarded to the matrix-vector multiplication
        routines.

    Examples
    --------
    Creating a Gaussian kernel with a single length-scale. Operations on this kernel will not
    use KeOps.

    >>> K = GaussianKernel(sigma=3.0, opt=FalkonOptions(no_keops=True))

    Creating a Gaussian kernel with a different length-scale per dimension

    >>> K = GaussianKernel(sigma=torch.tensor([1.0, 3.5, 7.0]))

    Creating a Gaussian kernel object with full covariance matrix (randomly chosen)

    >>> mat = torch.randn(3, 3, dtype=torch.float64)
    >>> sym_mat = mat @ mat.T
    >>> K = GaussianKernel(sigma=sym_mat)
    >>> K
    GaussianKernel(sigma=tensor([[ 2.0909,  0.0253, -0.2490],
            [ 0.0253,  0.3399, -0.5158],
            [-0.2490, -0.5158,  4.4922]], dtype=torch.float64))  #random


    Notes
    -----
    The Gaussian kernel with a single length-scale follows

    .. math::

        k(x, x') = \\exp{-\\dfrac{\\lVert x - x' \\rVert^2}{2\\sigma^2}}


    When the length-scales are specified as a matrix, the RBF kernel is determined by

    .. math::

        k(x, x') = \\exp{-\\dfrac{1}{2}x\\Sigma^{-1}x'}


    In both cases, the actual computation follows a different path, working on the expanded
    norm. The KeOps implementation is fully contained in the :meth:`_keops_mmv_impl` method,
    while our implementation uses methods :meth:`_prepare`, :meth:`_apply`, and :meth:`_transform`
    and is driven by the functions in :mod:`falkon.mmv_ops`.
    """
    kernel_name = "gaussian"

    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        super().__init__(self.kernel_name, opt)

        self.sigma, self.gaussian_type = self._get_sigma_kt(sigma)

        if self.gaussian_type == 'single':
            self.gamma = torch.tensor(
                -0.5 / (self.sigma.item() ** 2), dtype=torch.float64).item()
        else:
            self.gamma = torch.cholesky(self.sigma, upper=False)
            self.kernel_type = "l2-multi-distance"

    @staticmethod
    def _get_sigma_kt(sigma):
        if isinstance(sigma, torch.Tensor):
            try:
                # tensor.item() works if tensor is a scalar, otherwise it throws
                # a value error.
                sigma.item()
                return sigma, "single"
            except ValueError:
                pass

            if sigma.dim() == 1 or sigma.size(1) == 1:
                return torch.diagflat(sigma), "multi"

            if sigma.dim() != 2:
                raise TypeError("Sigma can be specified as a 1D or a 2D tensor. "
                                "Found %dD tensor" % (sigma.dim()))
            if sigma.size(0) != sigma.size(1):
                raise TypeError("Sigma passed as a 2D matrix must be square. "
                                "Found dimensions %s" % (sigma.size()))
            return sigma, "multi"
        else:
            try:
                sigma = float(sigma)
                return torch.tensor(sigma, dtype=torch.float64), "single"
            except TypeError:
                raise TypeError("Sigma must be a scalar or a tensor.")

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        if self.gaussian_type == 'single':
            formula = 'Exp(g * SqDist(x1, x2)) * v'
            aliases = [
                'x1 = Vi(%d)' % (X1.shape[1]),
                'x2 = Vj(%d)' % (X2.shape[1]),
                'v = Vj(%d)' % (v.shape[1]),
                'g = Pm(1)'
            ]
            other_vars = [torch.tensor([self.gamma]).to(dtype=X1.dtype)]
        else:
            dim = self.gamma.shape[0]
            formula = (
                'Exp( -IntInv(2) * SqDist('
                f'TensorDot(x1, g, Ind({dim}), Ind({dim}, {dim}), Ind(0), Ind(0)), '
                f'TensorDot(x2, g, Ind({dim}), Ind({dim}, {dim}), Ind(0), Ind(0)))) * v'
            )
            aliases = [
                'x1 = Vi(%d)' % (X1.shape[1]),
                'x2 = Vj(%d)' % (X2.shape[1]),
                'v = Vj(%d)' % (v.shape[1]),
                'g = Pm(%d)' % (dim ** 2)
            ]
            other_vars = [self.gamma.reshape(-1).to(dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _prepare(self, X1, X2):
        if self.gaussian_type == "single":
            return super()._prepare(X1, X2)
        else:
            return self.prepare_multisigma(X1, X2)

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        if self.gaussian_type != "single":
            raise NotImplementedError(
                "Sparse Gaussian kernel only implemented with scalar sigma.")
        return super()._prepare_sparse(X1, X2)

    def _apply(self, X1, X2, out):
        if self.gaussian_type == "single":
            return super()._apply(X1, X2, out)
        else:
            return self.apply_multisigma(X1, X2, out)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        if self.gaussian_type != "single":
            raise NotImplementedError(
                "Sparse Gaussian kernel only implemented with scalar sigma.")
        return super()._apply_sparse(X1, X2, out)

    def _transform(self, A) -> torch.Tensor:
        if self.gaussian_type == "single":
            return self.transform_singlesigma(A)
        else:
            return self.transform_multisigma(A)

    def prepare_multisigma(self, X1, X2):
        chol = self.gamma.to(X1)
        return DistKerContainer(
            sq1=torch.norm(X1 @ chol, p=2, dim=1, keepdim=True).pow_(2),
            sq2=torch.norm(X2 @ chol, p=2, dim=1, keepdim=True).pow_(2)
        )

    def apply_multisigma(self, X1, X2, out):
        sigma = self.sigma.to(X1)
        out.addmm_(X1 @ sigma, X2)

    # noinspection PyMethodMayBeStatic
    def transform_multisigma(self, A: torch.Tensor) -> torch.Tensor:
        A.mul_(-0.5)
        A.exp_()
        return A

    def transform_singlesigma(self, A: torch.Tensor) -> torch.Tensor:
        A.mul_(self.gamma)
        A.exp_()
        return A

    def __repr__(self):
        return f"GaussianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"Gaussian kernel<{self.sigma}>"


class LaplacianKernel(GaussianKernel):
    """Class for computing the Laplacian kernel, and related kernel-vector products.

    The Laplacian kernel is similar to the Gaussian kernel, but less sensitive to changes
    in the parameter `sigma`.

    Parameters
    ----------
    sigma
        The length-scale of the Laplacian kernel

    Notes
    -----
    The Laplacian kernel is determined by the following formula

    .. math::

        k(x, x') = \\exp{-\\frac{\\lVert x - x' \\rVert}{\\sigma}}

    """
    def __init__(self, sigma: float, opt: Optional[BaseOptions] = None):
        # With respect to the Gaussian kernel we need to change the value of gamma,
        # and from squared norm to norm. The latter change requires a different impl. of
        # the `_prepare` methods, and of the keops formula.
        self.kernel_name = "laplacian"
        super().__init__(sigma, opt)

        self.gaussian_type = 'single'
        self.gamma = torch.tensor(
            -1 / self.sigma.item(), dtype=torch.float64).item()

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        formula = 'Exp(g * Sqrt(SqDist(x1, x2))) * v'
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'g = Pm(1)'
        ]
        other_vars = [torch.tensor([self.gamma]).to(dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def _finalize(self, A: torch.Tensor, d: DistKerContainer) -> torch.Tensor:
        A.mul_(-2.0)
        A.add_(d.sq1.to(A))
        A.add_(d.sq2.to(A).t())
        A.clamp_min_(0)
        A.sqrt_()
        return self._transform(A)

    def __repr__(self):
        return f"LaplacianKernel(sigma={self.sigma})"