from typing import Union

import torch

from falkon.utils.cyblas import mul_triang, copy_triang
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.helpers import (decide_cuda)
from . import precond as prec
from .utils import *
from ..utils import CompOpt, TicToc
from ..utils.tensor_helpers import create_same_stride, is_f_contig, create_fortran


class FalkonPreconditioner(prec.Preconditioner):
    """Approximated Cholesky Preconditioner for FALKON.

    The preconditioner is based on the :math:`K_{MM}` kernel between the
    inducing points. A two step approximation of the inverse matrix
    via two cholesky decompositions is performed.

    `T = chol(K_MM)    => T.T @ T = K_MM`
    `A = chol(1/M * (T @ T.T) + lambda)`
    So `T` and `A` are both upper triangular.

    Here we store T in the upper triangular part of the `fC` matrix,
    and A in the upper triangular part of the matrix.
    Whenever we need to use one or the other we need to reset the
    diagonal of `fC` since it is shared between the two matrices.

    Parameters:
    -----------
     - _lambda : float
        The regularization parameter for KRR. Must be greater than 0.
     - kernel : falkon.kernel.Kernel
        The kernel object. This is used to compute the M*M kernel
        between inducing points. This kernel is then overwritten by
        the preconditioner itself.
     - opt : Union[CompOpt, dict]
        Additional options to be used in computing the preconditioner.
        Relevant options are:
         - epsilon : the jitter to add to the kernel matrix to make
            it positive-definite and allow Cholesky decomposition.
            This can be either a float, or a dictionary mapping from
            torch datatypes (e.g. float32, float64) to an appropriate
            float. Typically float32 requires more jitter than float64.
         - cpu_preconditioner : a boolean value which overrides CPU/GPU
            settings and forces the function to compute the whole
            preconditioner on the CPU. If set to False, we fall back to
            the usual CPU/GPU settings (i.e. 'use_cpu' option and the
            availability of a GPU).
    """

    def __init__(self, _lambda, kernel, opt=None):
        super().__init__()
        self.params = CompOpt(opt)
        self.params.setdefault('pc_epsilon', {torch.float32: 1e-5, torch.float64: 1e-13})
        self.params.setdefault('cpu_preconditioner', False)
        self.params.setdefault('debug', False)

        self._use_cuda = decide_cuda(self.params) and not self.params.cpu_preconditioner

        self._lambda = _lambda
        self.kernel = kernel

        self.fC = None
        self.dT = None
        self.dA = None

    def init(self, X: Union[torch.Tensor, SparseTensor], opt=None):
        """Compute the preconditioner
        """
        params: CompOpt = self.params.copy()
        if opt is not None:
            params.update(opt)
            self._use_cuda = decide_cuda(params) and not params.cpu_preconditioner
        params = CompOpt(params)

        dtype = X.dtype

        if isinstance(params.pc_epsilon, dict):
            eps = params.pc_epsilon[dtype]
        else:
            eps = params.pc_epsilon

        M = X.size(0)

        with TicToc("Kernel", debug=params.debug):
            if isinstance(X, torch.Tensor):
                C = create_same_stride((M, M), X, dtype=dtype, device='cpu',
                                       pin_memory=self._use_cuda)
            else:  # If sparse tensor we need fortran for kernel calculation
                C = create_fortran((M, M), dtype=dtype, device='cpu', pin_memory=self._use_cuda)
            self.kernel(X, X, out=C, opt=params)
        self.fC = C.numpy()
        if not is_f_contig(C):
            self.fC = self.fC.T

        with TicToc("Cholesky 1", debug=params.debug):
            # Compute T: lower(fC) = T.T
            inplace_add_diag(self.fC, eps * M)
            self.fC = potrf_wrapper(self.fC, clean=False, upper=False,
                                    use_cuda=self._use_cuda, opt=params)
            # Save the diagonal which will be overwritten when computing A
            self.dT = C.diag()

        with TicToc("Copy triangular", debug=params.debug):
            # Copy lower(fC) to upper(fC):  upper(fC) = T.
            copy_triang(self.fC, upper=False)

        if self._use_cuda:
            with TicToc("LAUUM", debug=params.debug):
                # Product upper(fC) @ upper(fC).T : lower(fC) = T @ T.T
                self.fC = lauum_wrapper(self.fC, upper=True, use_cuda=self._use_cuda, opt=params)
        else:
            with TicToc("LAUUM", debug=params.debug):
                # Product lower(fC).T @ lower(fC) : lower(fC) = T @ T.T
                self.fC = lauum_wrapper(self.fC, upper=False, use_cuda=self._use_cuda, opt=params)

        with TicToc("Cholesky 2", debug=params.debug):
            # lower(fC) = 1/M * T@T.T
            self.fC = mul_triang(self.fC, upper=False, preserve_diag=False, multiplier=1 / M)
            # lower(fC) = 1/M * T@T.T + lambda * I
            inplace_add_diag(self.fC, self._lambda)
            # Cholesky on lower(fC) : lower(fC) = A.T
            self.fC = potrf_wrapper(self.fC, clean=False, upper=False,
                                    use_cuda=self._use_cuda, opt=params)
            self.dA = C.diag()

    @check_init("fC", "dT", "dA")
    def invA(self, v):
        inplace_set_diag(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=1)

    @check_init("fC", "dT", "dA")
    def invAt(self, v):
        inplace_set_diag(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=0)

    @check_init("fC", "dT", "dA")
    def invT(self, v):
        inplace_set_diag(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=0)

    @check_init("fC", "dT", "dA")
    def invTt(self, v):
        inplace_set_diag(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=1)

    @check_init("fC", "dT", "dA")
    def apply(self, v):
        return self.invT(self.invA(v))

    @check_init("fC", "dT", "dA")
    def apply_t(self, v):
        return self.invAt(self.invTt(v))

    def __str__(self):
        return f"FalkonPreconditioner(_lambda={self._lambda}, kernel={self.kernel})"
