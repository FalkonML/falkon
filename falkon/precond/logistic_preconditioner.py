from typing import Union

import numpy as np
import scipy.linalg.blas as sclb
import torch
from falkon.utils.cyblas import copy_triang, vec_mul_triang, mul_triang

from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.helpers import (choose_fn, decide_cuda)
from . import precond as prec
from .utils import *
from ..utils import CompOpt, TicToc
from ..utils.tensor_helpers import create_same_stride, is_f_contig, create_fortran


class LogisticPreconditioner(prec.Preconditioner):
    """Approximate Cholesky Preconditioner for Logistic-FALKON.

    The preconditioner is based on the K_MM kernel between the
    inducing points. A two step approximation of the inverse matrix
    via two cholesky decompositions is performed.

    T = chol(K_MM)    => T.T @ T = K_MM
    A = chol(1/M * (T @ (T.T @ W)) + lambda)
    So T and A are both upper triangular.
    W is a diagonal matrix of weights derived from the 2nd derivative of the loss function.

    Here we store T in the upper triangular part of the `fC` matrix,
    and A in the upper triangular part of the matrix.
    Whenever we need to use one or the other we need to reset the
    diagonal of `fC` since it is shared between the two matrices.
    W is of size `M` and is the only difference from the normal FALKON preconditioner.

    Parameters:
    -----------
     - _lambda : float
        The regularization parameter for KRR. Must be greater than 0.
     - kernel : falkon.kernel.Kernel
        The kernel object. This is used to compute the M*M kernel
        between inducing points. This kernel is then overwritten by
        the preconditioner itself.
     - loss : falkon.gsc_losses.Loss
        The loss-function used for defining kernel weights.
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

    def __init__(self, kernel, loss, opt=None):
        super().__init__()
        self.params = CompOpt(opt)
        self.params.setdefault('pc_epsilon', {torch.float32: 1e-5, torch.float64: 1e-15})
        self.params.setdefault('cpu_preconditioner', False)
        self.params.setdefault('debug', False)

        self._use_cuda = decide_cuda(self.params) and not self.params.cpu_preconditioner

        self.kernel = kernel
        self.loss = loss

        self.fC = None
        self.dT = None
        self.dA = None

    def _trmm(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_np = alpha.numpy()
        if not alpha_np.flags.f_contiguous:
            # This never happens since alpha is always 1D
            alpha_np = np.copy(alpha_np, order="F")

        trmm = choose_fn(self.fC.dtype, sclb.dtrmm, sclb.strmm, "TRMM")
        out = trmm(alpha=1.0, a=self.fC, b=alpha_np, side=0, lower=0, trans_a=1, diag=0, overwrite_b=1)
        return torch.from_numpy(out)

    def init(self,
             X: Union[torch.Tensor, SparseTensor],
             Y: torch.Tensor,
             alpha: torch.Tensor,
             _lambda: float,
             N: int,
             opt=None):
        params: CompOpt = self.params.copy()
        if opt is not None:
            params.update(opt)
            self._use_cuda = decide_cuda(params) and not params.cpu_preconditioner
        params = CompOpt(params)

        if Y.shape[1] != 1:
            raise ValueError("Logistic preconditioner can only deal with 1D outputs.")

        dtype = X.dtype
        M = X.size(0)

        if isinstance(params.pc_epsilon, dict):
            eps = params.pc_epsilon[dtype]
        else:
            eps = params.pc_epsilon

        if self.fC is None:
            # This is done only at the first iteration of the logistic-falkon algorithm
            # It sets the `T` variable from the paper (chol(kMM)) to the upper part of `self.fC`
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

            with TicToc("Add diag", debug=params.debug):
                # Compute T: lower(fC) = T.T
                inplace_add_diag(self.fC, eps * M)
            with TicToc("Cholesky 1", debug=params.debug):
                self.fC = potrf_wrapper(self.fC, clean=True, upper=False,
                                        use_cuda=self._use_cuda, opt=params)
                # Save the diagonal which will be overwritten when computing A
                self.dT = C.diag()
            with TicToc("Copy triangular", debug=params.debug):
                # Copy lower(fC) to upper(fC):  upper(fC) = T.
                copy_triang(self.fC, upper=False)
        else:
            if not self._use_cuda:
                # Copy non-necessary for cuda since LAUUM will do the copying
                with TicToc("Copy triangular", debug=params.debug):
                    # Copy upper(fC) to lower(fC): lower(fC) = T.T
                    copy_triang(self.fC, upper=True)  # does not copy the diagonal
            # Setting diagonal necessary for trmm
            inplace_set_diag(self.fC, self.dT)

        # Compute W
        with TicToc("TRMM", debug=params.debug):
            # T is on upper(fC). Compute T.T @ alpha
            alpha = self._trmm(alpha.clone())
        with TicToc("W (ddf)", debug=params.debug):
            W = self.loss.ddf(Y, alpha)
        with TicToc("W-Multiply", debug=params.debug):
            W.sqrt_()
            self.fC = vec_mul_triang(self.fC, W.numpy().reshape(-1), side=0, upper=False)

        if self._use_cuda:
            with TicToc("LAUUM", debug=params.debug):
                # Product upper(fC) @ upper(fC).T : lower(fC) = T @ T.T
                self.fC = lauum_wrapper(self.fC, upper=True, use_cuda=self._use_cuda, opt=params)
        else:
            with TicToc("LAUUM", debug=params.debug):
                # Product lower(fC).T @ lower(fC) : lower(fC) = T @ T.T
                self.fC = lauum_wrapper(self.fC, upper=False, use_cuda=self._use_cuda, opt=params)

        # NOTE: Here the multiplier is 1/N instead of the more common 1/M!
        mul_triang(self.fC, upper=False, preserve_diag=False, multiplier=1/N)

        with TicToc("Add diag", debug=params.debug):
            # lower(fC) = 1/N * T@T.T + lambda * I
            inplace_add_diag(self.fC, _lambda)

        with TicToc("Cholesky 2", debug=params.debug):
            # Cholesky on lower(fC) : lower(fC) = A.T
            self.fC = potrf_wrapper(self.fC, clean=False, upper=False,
                                    use_cuda=self._use_cuda, opt=params)
            self.dA = torch.from_numpy(self.fC).diag()

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
        return f"LogisticPreconditioner(kernel={self.kernel}, loss={self.loss})"
