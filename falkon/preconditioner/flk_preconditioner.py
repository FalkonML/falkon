from typing import Union, Optional

import torch

from falkon.la_helpers import mul_triang, copy_triang, trsm, vec_mul_triang
from falkon.options import FalkonOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils import TicToc, decide_cuda
from falkon.utils.helpers import check_same_device
from falkon.utils.tensor_helpers import create_same_stride, is_f_contig, create_fortran
from .pc_utils import *
from .preconditioner import Preconditioner


class FalkonPreconditioner(Preconditioner):
    r"""Approximated Cholesky Preconditioner for FALKON.

    The preconditioner is based on the :math:`K_{MM}` kernel between the
    inducing points. A two step approximation of the inverse matrix
    via two Cholesky decompositions is performed.

    Starting with :math:`K_{MM}` we obtain :math:`T = \mathrm{chol}(K_{MM})`.
    Then we can obtain :math:`A = \mathrm{chol}(\frac{1}{M} T T^\top + \lambda)` via another Cholesky
    decomposition. Both `T` and `A` are upper triangular: the first gets stored in the upper
    triangle of the :math:`K_{MM}` matrix (called `fC` in the code), while the second is stored
    in the lower triangle.

    Whenever we want to use one of the two triangles we must reset the matrix diagonal, since
    it is shared between the two matrices.

    Parameters
    -----------
    penalty : float
        The regularization parameter for KRR. Must be greater than 0.
    kernel : falkon.kernels.kernel.Kernel
        The kernel object. This is used to compute the M*M kernel
        between inducing points. The kernel matrix is then overwritten by
        the preconditioner itself.
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

    """

    def __init__(self, penalty: float, kernel, opt: FalkonOptions):
        super().__init__()
        self.params = opt
        self._use_cuda = decide_cuda(self.params) and not self.params.cpu_preconditioner

        self._lambda = penalty
        self.kernel = kernel
        # self.weight_vec = weight_vec

        self.fC: Optional[torch.Tensor] = None
        self.dT: Optional[torch.Tensor] = None
        self.dA: Optional[torch.Tensor] = None

    def init(self, X: Union[torch.Tensor, SparseTensor], weight_vec: Optional[torch.Tensor] = None):
        """Initialize the preconditioner matrix.

        This method must be called before the preconditioner can be used.

        Parameters
        ----------
        X : torch.Tensor
            The (M x D) matrix of Nystroem centers
        weight_vec
            An optional vector of size (M x 1) which is used for reweighted least-squares.
            This vector should contain the weights corresponding to the Nystrom centers.
        """
        if X.is_cuda and not self._use_cuda:
            raise RuntimeError("use_cuda is set to False, but data is CUDA tensor. "
                               "Check your options.")
        if weight_vec is not None and not check_same_device(X, weight_vec):
            raise ValueError(f"Weights and data are not on the same device "
                             f"({weight_vec.device}, {X.device})")
        if weight_vec is not None and weight_vec.shape[0] != X.shape[0]:
            raise ValueError(f"Weights and Nystrom centers should have the same first dimension. "
                             f"Found instead {weight_vec.shape[0]}, {X.shape[0]}.")
        dtype = X.dtype
        dev = X.device
        eps = self.params.pc_epsilon(X.dtype)
        M = X.size(0)

        with TicToc("Kernel", debug=self.params.debug):
            if isinstance(X, torch.Tensor):
                C = create_same_stride((M, M), X, dtype=dtype, device=dev,
                                       pin_memory=self._use_cuda)
            else:  # If sparse tensor we need fortran for kernel calculation
                C = create_fortran((M, M), dtype=dtype, device=dev, pin_memory=self._use_cuda)
            self.kernel(X, X, out=C, opt=self.params)
        if not is_f_contig(C):
            C = C.T

        with TicToc("Cholesky 1", debug=self.params.debug):
            # Compute T: lower(fC) = T.T
            inplace_add_diag_th(C, eps * M)
            C = potrf_wrapper(C, clean=False, upper=False,
                              use_cuda=self._use_cuda, opt=self.params)
            # Save the diagonal which will be overwritten when computing A
            self.dT = C.diag()

        with TicToc("Copy triangular", debug=self.params.debug):
            # Copy lower(fC) to upper(fC):  upper(fC) = T.
            copy_triang(C, upper=False)

        # Weighted least-squares uses slightly different preconditioner
        if weight_vec is not None:
            with TicToc("Weighting", debug=self.params.debug):
                weight_vec.sqrt_()
                # We switch on CUDA since LAUUM uses upper(C) or lower(C) depending on this.
                if self._use_cuda:
                    vec_mul_triang(C, weight_vec.numpy().reshape(-1), side=0, upper=True)
                else:
                    vec_mul_triang(C, weight_vec.numpy().reshape(-1), side=1, upper=False)

        if self._use_cuda:
            with TicToc("LAUUM(CUDA)", debug=self.params.debug):
                # Product upper(fC) @ upper(fC).T : upper(fC) = T @ T.T
                C = lauum_wrapper(C, upper=True, use_cuda=self._use_cuda, opt=self.params)
        else:
            with TicToc("LAUUM(CPU)", debug=self.params.debug):
                # Product lower(fC).T @ lower(fC) : upper(fC) = T @ T.T
                C = lauum_wrapper(C, upper=False, use_cuda=self._use_cuda, opt=self.params)

        with TicToc("Cholesky 2", debug=self.params.debug):
            # lower(fC) = 1/M * T@T.T
            mul_triang(C, upper=False, preserve_diag=False, multiplier=1 / M)
            # lower(fC) = 1/M * T@T.T + lambda * I
            inplace_add_diag_th(C, self._lambda)
            # Cholesky on lower(fC) : lower(fC) = A.T
            C = potrf_wrapper(C, clean=False, upper=False,
                              use_cuda=self._use_cuda, opt=self.params)
            self.dA = C.diag()

        self.fC = C

    def to(self, device):
        if self.fC is not None:
            self.fC = self.fC.to(device)
        if self.dT is not None:
            self.dT = self.dT.to(device)
        if self.dA is not None:
            self.dA = self.dA.to(device)
        return self

    @check_init("fC", "dT", "dA")
    def invA(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`Ax = v` for unknown vector :math:`x`.

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
        :func:`~falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag_th(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=1)

    @check_init("fC", "dT", "dA")
    def invAt(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`A^\top x = v` for unknown vector :math:`x`.

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
        inplace_set_diag_th(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=0)

    @check_init("fC", "dT", "dA")
    def invT(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`Tx = v` for unknown vector :math:`x`.

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
        inplace_set_diag_th(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=0)

    @check_init("fC", "dT", "dA")
    def invTt(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`T^\top x = v` for unknown vector :math:`x`.

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
        inplace_set_diag_th(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=1)

    @check_init("fC", "dT", "dA")
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve two systems of equations :math:`ATx = v` for unknown vector :math:`x`.

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
    def apply_t(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve two systems of equations :math:`A^\top T^\top x = v` for unknown vector :math:`x`.

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
        return f"FalkonPreconditioner(_lambda={self._lambda}, kernel={self.kernel})"
