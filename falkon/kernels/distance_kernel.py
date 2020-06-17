#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:49:21 2017

@author: alessandro
"""
import collections
import functools
from abc import ABC, abstractmethod

import torch

from falkon.sparse.sparse_ops import sparse_matmul
from . import Kernel, KeopsKernelMixin
from ..sparse import sparse_ops
from ..sparse.sparse_tensor import SparseTensor

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
    To efficiently compute kernels of the form k(x, x') = ||x - x'||^2 between two matrices of
    data-points we decompose the squared norm of the difference into 3 terms:
    ||X||^2 and -2*XX'^T and ||X'||^2
    The first and third term are calculated in the `_prepare` method while the second is
    calculated in the `_apply` method. Finally the three terms are combined in the `_finalize`
    method.
    """
    kernel_type = "l2distance"

    def __init__(self, name, opt=None, **kw):
        super().__init__(name, self.kernel_type, opt, **kw)

    def _prepare(self, X1, X2):
        return DistKerContainer(
            sq1=torch.norm(X1, p=2, dim=1, keepdim=True).pow_(2),
            sq2=torch.norm(X2, p=2, dim=1, keepdim=True).pow_(2)
        )

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        sq1 = torch.empty(X1.size(0), dtype=X1.dtype, device=X1.device)
        sparse_ops.square_norm(X1, sq1)
        sq2 = torch.empty(X2.size(0), dtype=X1.dtype, device=X1.device)
        sparse_ops.square_norm(X2, sq2)
        return DistKerContainer(
            sq1=sq1.reshape(-1, 1), sq2=sq2.reshape(-1, 1)
        )

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor) -> None:
        out.addmm_(X1, X2)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        return sparse_matmul(X1, X2, out)

    def _finalize(self, A, d):
        A.mul_(-2.0)
        A.add_(d.sq1.to(A))
        A.add_(d.sq2.to(A).t())
        A.clamp_min_(0)
        return self._transform(A)

    @abstractmethod
    def _transform(self, A):
        pass


class GaussianKernel(L2DistanceKernel, KeopsKernelMixin):
    """Gaussian or RBF kernel class.
    Can be used to compute the kernel itself, or to compute kernel-vector
    products.

    Parameters:
    -----------
    sigma : Optional[float]
        The length-scale of the kernel.
        This can be a scalar, and then it corresponds to the standard deviation
        of Gaussian distribution from which the kernel is derived.
        It can also be a vector of size `d` or a matrix of size `d*d` where `d`
        is the dimensionality of the data which will be used with the kernel.

        In this case sigma will be the inverse square of the standard deviation,
        so for example converting from the vectorial sigma to the scalar sigma can
        be done by `vec_sigma = -1/(sigma**2)`
    opt : Union[dict, CompOpt, None]
        Additional options to be forwarded to the matrix-vector multiplication
        routines.
    """
    kernel_name = "gaussian"

    def __init__(self, sigma, opt=None, **kw):
        super().__init__(self.kernel_name, opt, **kw)

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

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
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

    def _transform(self, A):
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

    def transform_multisigma(self, A):
        A.mul_(-0.5)
        A.exp_()
        return A

    def transform_singlesigma(self, A):
        A.mul_(self.gamma)
        A.exp_()
        return A

    def __repr__(self):
        return f"GaussianKernel(sigma={self.sigma})"


class LaplacianKernel(GaussianKernel):
    def __init__(self, sigma, opt=None, **kw):
        self.kernel_name = "laplacian"
        super().__init__(sigma, opt, **kw)

        self.gaussian_type = 'single'
        # We only need to change gamma wrt to the gaussian kernel
        self.gamma = torch.tensor(
            -1 / self.sigma.item(), dtype=torch.float64).item()

    def __repr__(self):
        return f"LaplacianKernel(sigma={self.sigma})"
