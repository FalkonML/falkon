from typing import Union

import numpy as np
import scipy.linalg.blas as sclb
import torch

from falkon.options import FalkonOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.cyblas import copy_triang, vec_mul_triang, mul_triang
from falkon.utils.helpers import (choose_fn)
from . import preconditioner as prec
from .pc_utils import *
from falkon.utils import TicToc, decide_cuda
from falkon.utils.tensor_helpers import create_same_stride, is_f_contig, create_fortran


class LogisticPreconditioner(prec.Preconditioner):
    """Approximate Cholesky Preconditioner for Logistic-FALKON.

    The preconditioner is based on the K_MM kernel between the
    inducing points. A two step approximation of the inverse matrix
    via two cholesky decompositions is performed.

    ```
    T = chol(K_MM)    => T.T @ T = K_MM
    A = chol(1/M * (T @ (T.T @ W)) + lambda)
    ```

    So `T` and `A` are both upper triangular.
    `W` is a diagonal matrix of weights derived from the 2nd derivative of the loss function.

    Here we store `T` in the upper triangular part of the `fC` matrix,
    and `A` in the upper triangular part of the matrix.
    Whenever we need to use one or the other we need to reset the
    diagonal of `fC` since it is shared between the two matrices.
    `W` is of size `M` and is the only difference with respect to the normal FALKON preconditioner
    (:cls:`falkon.preconditioner.FalkonPreconditioner`).

    Parameters
    -----------
    kernel : falkon.kernel.Kernel
        The kernel object. This is used to compute the M*M kernel
        between inducing points. This kernel is then overwritten by
        the preconditioner itself.
    loss : falkon.gsc_losses.Loss
        The loss-function used for defining kernel weights.
    opt : FalkonOptions
        Additional options to be used in computing the preconditioner.
        Relevant options are:

        - pc_epsilon : the jitter to add to the kernel matrix to make
            it positive-definite and allow Cholesky decomposition.
            This can be either a float, or a dictionary mapping from
            torch datatypes (e.g. float32, float64) to an appropriate
            float. Typically float32 requires more jitter than float64.
        - cpu_preconditioner : a boolean value which overrides CPU/GPU
            settings and forces the function to compute the whole
            preconditioner on the CPU. If set to False, we fall back to
            the usual CPU/GPU settings (i.e. 'use_cpu' option and the
            availability of a GPU).

    See Also
    --------
    :class:`falkon.gsc_losses.LogisticLoss` :
        for an example of loss function used for kernel reweighting.
    :class:`falkon.models.LogisticFalkon` :
        for the logistic kernel estimator which uses this preconditioner.
    """

    def __init__(self, kernel, loss, opt: FalkonOptions):
        super().__init__()
        self.params = opt
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
        out = trmm(alpha=1.0, a=self.fC, b=alpha_np, side=0, lower=0, trans_a=1, diag=0,
                   overwrite_b=1)
        return torch.from_numpy(out)

    def init(self,
             X: Union[torch.Tensor, SparseTensor],
             Y: torch.Tensor,
             alpha: torch.Tensor,
             penalty: float,
             N: int) -> None:
        """Initialize the preconditioner matrix.

        This method must be called before the preconditioner becomes usable.

        Parameters
        ----------
        X : MxD tensor
            Matrix of Nystroem centers
        Y : Mx1 tensor
            Vector of targets corresponding to the Nystroem centers `X`
        alpha : Mx1 tensor
            Parameter vector (of the same dimension as `Y`) which gives the current
            solution to the optimization problem.
        penalty : float
            Regularization amount
        N : int
            Number of points in the full data-set.

        Notes
        -----
        If `debug=True` is present in the options, this method will print a lot of extra
        information pertaining timings of the various preconditioner operations. This can be
        useful to help understand how the preconditioner works.
        """
        if Y.shape[1] != 1:
            raise ValueError("Logistic preconditioner can only deal with 1D outputs.")

        dtype = X.dtype
        M = X.size(0)

        eps = self.params.pc_epsilon(dtype)

        if self.fC is None:
            # This is done only at the first iteration of the logistic-falkon algorithm
            # It sets the `T` variable from the paper (chol(kMM)) to the upper part of `self.fC`
            with TicToc("Kernel", debug=self.params.debug):
                if isinstance(X, torch.Tensor):
                    C = create_same_stride((M, M), X, dtype=dtype, device='cpu',
                                           pin_memory=self._use_cuda)
                else:  # If sparse tensor we need fortran for kernel calculation
                    C = create_fortran((M, M), dtype=dtype, device='cpu', pin_memory=self._use_cuda)
                self.kernel(X, X, out=C, opt=self.params)
            self.fC = C.numpy()
            if not is_f_contig(C):
                self.fC = self.fC.T

            with TicToc("Add diag", debug=self.params.debug):
                # Compute T: lower(fC) = T.T
                inplace_add_diag(self.fC, eps * M)
            with TicToc("Cholesky 1", debug=self.params.debug):
                self.fC = potrf_wrapper(self.fC, clean=True, upper=False,
                                        use_cuda=self._use_cuda, opt=self.params)
                # Save the diagonal which will be overwritten when computing A
                self.dT = C.diag()
            with TicToc("Copy triangular", debug=self.params.debug):
                # Copy lower(fC) to upper(fC):  upper(fC) = T.
                copy_triang(self.fC, upper=False)
        else:
            if not self._use_cuda:
                # Copy non-necessary for cuda since LAUUM will do the copying
                with TicToc("Copy triangular", debug=self.params.debug):
                    # Copy upper(fC) to lower(fC): lower(fC) = T.T
                    copy_triang(self.fC, upper=True)  # does not copy the diagonal
            # Setting diagonal necessary for trmm
            inplace_set_diag(self.fC, self.dT)

        # Compute W
        with TicToc("TRMM", debug=self.params.debug):
            # T is on upper(fC). Compute T.T @ alpha
            alpha = self._trmm(alpha.clone())
        with TicToc("W (ddf)", debug=self.params.debug):
            W = self.loss.ddf(Y, alpha)
        with TicToc("W-Multiply", debug=self.params.debug):
            W.sqrt_()
            self.fC = vec_mul_triang(self.fC, W.numpy().reshape(-1), side=0, upper=False)

        if self._use_cuda:
            with TicToc("LAUUM", debug=self.params.debug):
                # Product upper(fC) @ upper(fC).T : lower(fC) = T @ T.T
                self.fC = lauum_wrapper(self.fC, upper=True,
                                        use_cuda=self._use_cuda, opt=self.params)
        else:
            with TicToc("LAUUM", debug=self.params.debug):
                # Product lower(fC).T @ lower(fC) : lower(fC) = T @ T.T
                self.fC = lauum_wrapper(self.fC, upper=False,
                                        use_cuda=self._use_cuda, opt=self.params)

        # NOTE: Here the multiplier is 1/N instead of the more common 1/M!
        mul_triang(self.fC, upper=False, preserve_diag=False, multiplier=1 / N)

        with TicToc("Add diag", debug=self.params.debug):
            # lower(fC) = 1/N * T@T.T + lambda * I
            inplace_add_diag(self.fC, penalty)

        with TicToc("Cholesky 2", debug=self.params.debug):
            # Cholesky on lower(fC) : lower(fC) = A.T
            self.fC = potrf_wrapper(self.fC, clean=False, upper=False,
                                    use_cuda=self._use_cuda, opt=self.params)
            self.dA = torch.from_numpy(self.fC).diag()

    @check_init("fC", "dT", "dA")
    def invA(self, v):
        """Solve the system of equations :math:`Ax = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=1)

    @check_init("fC", "dT", "dA")
    def invAt(self, v):
        """Solve the system of equations :math:`A^\\top x = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=0)

    @check_init("fC", "dT", "dA")
    def invT(self, v):
        """Solve the system of equations :math:`Tx = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=0)

    @check_init("fC", "dT", "dA")
    def invTt(self, v):
        """Solve the system of equations :math:`T^\\top x = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=1)

    @check_init("fC", "dT", "dA")
    def apply(self, v):
        """Solve two systems of equations :math:`ATx = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        return self.invT(self.invA(v))

    @check_init("fC", "dT", "dA")
    def apply_t(self, v):
        """Solve two systems of equations :math:`A^\\top T^\\top x = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        return self.invAt(self.invTt(v))

    def __str__(self):
        return f"LogisticPreconditioner(kernel={self.kernel}, loss={self.loss})"
