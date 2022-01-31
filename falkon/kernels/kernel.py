import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Union

import torch
from falkon.sparse import SparseTensor

from falkon.mmv_ops.fmm import fmm
from falkon.mmv_ops.fmmv import fdmmv, fmmv
from falkon.utils.helpers import check_same_dtype, check_sparse, check_same_device
from falkon.options import FalkonOptions


class Kernel(torch.nn.Module, ABC):
    """Abstract kernel class. Kernels should inherit from this class, overriding appropriate methods.

    To extend Falkon with new kernels, you should read the documentation of this class
    carefully, and take a look at the existing implementation of :class:`~falkon.kernels.GaussianKernel`
    or :class:`~falkon.kernels.LinearKernel`.

    There are several abstract methods which should be implemented, depending on which kind of operations
    which are supported by the implementing kernel.

    The :meth:`compute` method should compute the kernel matrix, without concerns for differentiability,
    :meth:`compute_diff` instead should compute the kernel matrix in such a way that the output
    is differentiable with respect to the inputs, and to the kernel parameters. Finally the
    :meth:`compute_sparse` method is used to compute the kernel for sparse input matrices. It need
    not be differentiable.

    Kernels may have several parameters, for example the length-scale of the Gaussian kernel, the
    exponent of the polynomial kernel, etc. The kernel should be differentiable with respect to
    some such parameters (the afore mentioned length-scale for example), but not with respect to
    others (for example the nu parameter of Matern kernels). Each concrete kernel class must
    specify the differentiable parameters with the :meth:`diff_params` method, and other parameters
    with the :meth:`nondiff_params`.
    Additionally kernels which implemenet the :meth:`compute_diff` method should also implement
    the :meth:`detach` method which returns a new instance of the kernel, with its parameters
    detached from the computation graph.

    To provide a KeOps implementation, you will have to inherit also from the
    :class:`~falkon.kernels.keops_helpers.KeopsKernelMixin` class, and implement its abstract methods.
    In case a KeOps implementation is provided, you should make sure to override the
    :meth:`_decide_mmv_impl` and :meth:`_decide_dmmv_impl` so that the KeOps implementation is
    effectively used. Have a look at the :class:`~falkon.kernels.PolynomialKernel` class for
    an example of how to integrate KeOps in the kernel.

    Parameters
    ----------
    name
        A short name for the kernel (e.g. "Gaussian")
    kernel_type
        A short string describing the type of kernel. This may be used to create specialized
        functions in :mod:`falkon.mmv_ops` which optimize for a specific kernel type.
    opt
        Base set of options to be used for operations involving this kernel.
    """
    def __init__(self, name: str, kernel_type: str, opt: Optional[FalkonOptions]):
        super().__init__()
        self.name = name
        self.kernel_type = kernel_type
        if opt is None:
            opt = FalkonOptions()
        self.params: FalkonOptions = opt

    @staticmethod
    def _check_dmmv_dimensions(X1: torch.Tensor, X2: torch.Tensor, v: Optional[torch.Tensor],
                               w: Optional[torch.Tensor], out: Optional[torch.Tensor]):
        # Parameter validation
        if v is None and w is None:
            raise ValueError("One of v and w must be specified to run fdMMV.")

        if X1.dim() != 2:
            raise ValueError("Matrix X1 must be 2D.")
        if X2.dim() != 2:
            raise ValueError("Matrix X2 must be 2D.")
        if v is not None and v.dim() == 1:
            v = v.reshape((-1, 1))
        if v is not None and v.dim() != 2:
            raise ValueError(
                f"v must be a vector or a 2D matrix. Found {len(v.shape)}D.")
        if w is not None and w.dim() == 1:
            w = w.reshape((-1, 1))
        if w is not None and w.dim() != 2:
            raise ValueError(
                f"w must be a vector or a 2D matrix. Found {len(w.shape)}D.")

        # noinspection PyUnresolvedReferences
        T = v.size(1) if v is not None else w.size(1)
        M = X2.size(0)
        if out is not None and out.shape != (M, T):
            raise ValueError(
                f"Output dimension is incorrect. "
                f"Expected ({M}, {T}) found {out.shape}")
        if v is not None and v.shape != (X2.size(0), T):
            raise ValueError(
                f"Dimensions of matrix v are incorrect: "
                f"Expected ({M}, {T}) found {v.shape}")
        if w is not None and w.shape != (X1.size(0), T):
            raise ValueError(
                f"Dimensions of matrix w are incorrect: "
                f"Expected ({X1.size(0)}, {T}) found {w.shape}")

        if not check_same_dtype(X1, X2, v, w, out):
            raise TypeError("Data types of input matrices must be equal.")

        return X1, X2, v, w, out

    @staticmethod
    def _check_mmv_dimensions(X1: torch.Tensor, X2: torch.Tensor, v: torch.Tensor,
                              out: Optional[torch.Tensor]):
        # Parameter validation
        if X1.dim() != 2:
            raise ValueError("Matrix X1 must be 2D.")
        if X2.dim() != 2:
            raise ValueError("Matrix X2 must be 2D.")
        if v.dim() == 1:
            v = v.reshape((-1, 1))
        if v.dim() != 2:
            raise ValueError(
                f"v must be a vector or a 2D matrix. Found {len(v.shape)}D.")

        if out is not None and out.shape != (X1.size(0), v.size(1)):
            raise ValueError(
                f"Output dimension is incorrect. "
                f"Expected ({X1.size(0)}, {v.size(1)}) found {out.shape}")
        if v.shape != (X2.size(0), v.size(1)):
            raise ValueError(
                f"Dimensions of matrix v are incorrect: "
                f"Expected ({X2.size(0)}, {v.size(1)}) found {v.shape}")

        if not check_same_dtype(X1, X2, v, out):
            raise TypeError("Data types of input matrices must be equal.")

        return X1, X2, v, out

    @staticmethod
    def _check_mm_dimensions(X1: torch.Tensor, X2: torch.Tensor, diag: bool, out: Optional[torch.Tensor]):
        # Parameter validation
        if X1.dim() != 2:
            raise ValueError("Matrix X1 must be 2D.")
        if X2.dim() != 2:
            raise ValueError("Matrix X2 must be 2D.")
        N = X1.size(0)
        M = X2.size(0)
        if not diag:
            if out is not None and out.shape != (N, M):
                raise ValueError(
                    f"Output dimension is incorrect. "
                    f"Expected ({N}, {M}) found {out.shape}.")
        else:
            if N != M:
                raise ValueError(
                    f"Cannot compute the kernel diagonal "
                    f"between matrices with {N} and {M} samples.")
            if out is not None and out.reshape(-1).shape[0] != N:
                raise ValueError(
                    f"Output dimension is incorrect. "
                    f"Expected ({N}) found {out.shape}.")

        if not check_same_dtype(X1, X2, out):
            raise TypeError("Data types of input matrices must be equal.")

        return X1, X2, out

    @staticmethod
    def _check_device_properties(*args, fn_name: str, opt: FalkonOptions):
        if not check_same_device(*args):
            raise RuntimeError("All input arguments to %s must be on the same device" % (fn_name))

    def __call__(self,
                 X1: torch.Tensor,
                 X2: torch.Tensor,
                 diag: bool = False,
                 out: Optional[torch.Tensor] = None,
                 opt: Optional[FalkonOptions] = None):
        """Compute the kernel matrix between ``X1`` and ``X2``

        Parameters
        ----------
        X1 : torch.Tensor
            The first data-matrix for computing the kernel. Of shape (N x D):
            N samples in D dimensions.
        X2 : torch.Tensor
            The second data-matrix for computing the kernel. Of shape (M x D):
            M samples in D dimensions. Set ``X2 == X1`` to compute a symmetric kernel.
        diag : bool
            Whether to compute just the diagonal of the kernel matrix, or the whole matrix.
        out : torch.Tensor or None
            Optional tensor of shape (N x M) to hold the output. If not provided it will
            be created.
        opt : Optional[FalkonOptions]
            Options to be used for computing the operation. Useful are the memory size options
            and CUDA options.

        Returns
        -------
        out : torch.Tensor
            The kernel between ``X1`` and ``X2``.
        """
        X1, X2, out = self._check_mm_dimensions(X1, X2, diag, out)
        self._check_device_properties(X1, X2, out, fn_name="kernel", opt=opt)
        params = self.params
        if opt is not None:
            params = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        mm_impl = self._decide_mm_impl(X1, X2, diag, params)
        return mm_impl(self, params, out, diag, X1, X2)

    def _decide_mm_impl(self, X1: torch.Tensor, X2: torch.Tensor, diag: bool, opt: FalkonOptions):
        """Choose which `mm` function to use for this data.

        Note that `mm` functions compute the kernel itself so **KeOps may not be used**.

        Parameters
        ----------
        X1 : torch.Tensor
            First data matrix, of shape (N x D)
        X2 : torch.Tensor
            Second data matrix, of shape (M x D)
        diag : bool
            Whether to compute just the diagonal of the kernel matrix, or the whole matrix.
        opt : FalkonOptions
            Falkon options. Options may be specified to force GPU or CPU usage.

        Returns
        -------
        mm_fn
            A function which allows to perform the `mm` operation.

        Notes
        -----
        This function decides based on the inputs: if the inputs are sparse, it will choose
        the sparse implementations; if CUDA is detected, it will choose the CUDA implementation;
        otherwise it will simply choose the basic CPU implementation.
        """
        sparsity = check_sparse(X1, X2)
        if not all(sparsity) and any(sparsity):
            raise ValueError("Either all or none of 'X1', 'X2' must be sparse.")
        return fmm

    def mmv(self,
            X1: Union[torch.Tensor, SparseTensor],
            X2: Union[torch.Tensor, SparseTensor],
            v: torch.Tensor,
            out: Optional[torch.Tensor] = None, opt: Optional[FalkonOptions] = None):
        # noinspection PyShadowingNames
        """Compute matrix-vector multiplications where the matrix is the current kernel.

        Parameters
        ----------
        X1 : torch.Tensor
            The first data-matrix for computing the kernel. Of shape (N x D):
            N samples in D dimensions.
        X2 : torch.Tensor
            The second data-matrix for computing the kernel. Of shape (M x D):
            M samples in D dimensions. Set `X2 == X1` to compute a symmetric kernel.
        v : torch.Tensor
            A vector to compute the matrix-vector product. This may also be a matrix of shape
            (M x T), but if `T` is very large the operations will be much slower.
        out : torch.Tensor or None
            Optional tensor of shape (N x T) to hold the output. If not provided it will
            be created.
        opt : Optional[FalkonOptions]
            Options to be used for computing the operation. Useful are the memory size options
            and CUDA options.

        Returns
        -------
        out : torch.Tensor
            The (N x T) output.

        Examples
        --------
        >>> import falkon, torch
        >>> k = falkon.kernels.GaussianKernel(sigma=2)  # You can substitute the Gaussian kernel by any other.
        >>> X1 = torch.randn(100, 3)
        >>> X2 = torch.randn(150, 3)
        >>> v = torch.randn(150, 1)
        >>> out = k.mmv(X1, X2, v, out=None)
        >>> out.shape
        torch.Size([100, 1])
        """
        X1, X2, v, out = self._check_mmv_dimensions(X1, X2, v, out)

        params = self.params
        if opt is not None:
            params = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        mmv_impl = self._decide_mmv_impl(X1, X2, v, params)
        return mmv_impl(X1, X2, v, self, out, params)

    def _decide_mmv_impl(self,
                         X1: Union[torch.Tensor, SparseTensor],
                         X2: Union[torch.Tensor, SparseTensor],
                         v: torch.Tensor, opt: FalkonOptions):
        """Choose which `mmv` function to use for this data.

        Note that `mmv` functions compute the kernel-vector product

        Parameters
        ----------
        X1 : torch.Tensor
            First data matrix, of shape (N x D)
        X2 : torch.Tensor
            Second data matrix, of shape (M x D)
        v : torch.Tensor
            Vector for the matrix-vector multiplication (M x T)
        opt : FalkonOptions
            Falkon options. Options may be specified to force GPU or CPU usage.

        Returns
        -------
        mmv_fn
            A function which allows to perform the `mmv` operation.

        Notes
        -----
        This function decides based on the inputs: if the inputs are sparse, it will choose
        the sparse implementations; if CUDA is detected, it will choose the CUDA implementation;
        otherwise it will simply choose the basic CPU implementation.
        """
        sparsity = check_sparse(X1, X2)
        if not all(sparsity) and any(sparsity):
            raise ValueError("Either all or none of 'X1', 'X2' must be sparse.")
        return fmmv

    def dmmv(self,
             X1: Union[torch.Tensor, SparseTensor],
             X2: Union[torch.Tensor, SparseTensor],
             v: Optional[torch.Tensor],
             w: Optional[torch.Tensor], out: Optional[torch.Tensor] = None,
             opt: Optional[FalkonOptions] = None):
        # noinspection PyShadowingNames
        """Compute double matrix-vector multiplications where the matrix is the current kernel.

        The general form of `dmmv` operations is: `Kernel(X2, X1) @ (Kernel(X1, X2) @ v + w)`
        where if `v` is None, then we simply have `Kernel(X2, X1) @ w` and if `w` is None
        we remove the additive factor.
        **At least one of `w` and `v` must be provided**.

        Parameters
        ----------
        X1 : torch.Tensor
            The first data-matrix for computing the kernel. Of shape (N x D):
            N samples in D dimensions.
        X2 : torch.Tensor
            The second data-matrix for computing the kernel. Of shape (M x D):
            M samples in D dimensions. Set `X2 == X1` to compute a symmetric kernel.
        v : torch.Tensor or None
            A vector to compute the matrix-vector product. This may also be a matrix of shape
            (M x T), but if `T` is very large the operations will be much slower.
        w : torch.Tensor or None
            A vector to compute matrix-vector products. This may also be a matrix of shape
            (N x T) but if `T` is very large the operations will be much slower.
        out : torch.Tensor or None
            Optional tensor of shape (M x T) to hold the output. If not provided it will
            be created.
        opt : Optional[FalkonOptions]
            Options to be used for computing the operation. Useful are the memory size options
            and CUDA options.

        Returns
        -------
        out : torch.Tensor
            The (M x T) output.

        Examples
        --------
        >>> import falkon, torch
        >>> k = falkon.kernels.GaussianKernel(sigma=2)  # You can substitute the Gaussian kernel by any other.
        >>> X1 = torch.randn(100, 3)  # N is 100, D is 3
        >>> X2 = torch.randn(150, 3)  # M is 150
        >>> v = torch.randn(150, 1)
        >>> w = torch.randn(100, 1)
        >>> out = k.dmmv(X1, X2, v, w, out=None)
        >>> out.shape
        torch.Size([150, 1])
        """
        X1, X2, v, w, out = self._check_dmmv_dimensions(X1, X2, v, w, out)
        params = self.params
        if opt is not None:
            params = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        dmmv_impl = self._decide_dmmv_impl(X1, X2, v, w, params)
        sparsity = check_sparse(X1, X2)
        diff = False
        if not any(sparsity):
            diff = any([
                t.requires_grad for t in [X1, X2, v, w] + list(self.diff_params.values())
                if t is not None
            ])
        return dmmv_impl(X1, X2, v, w, self, out, diff, params)

    def _decide_dmmv_impl(self,
                          X1: Union[torch.Tensor, SparseTensor],
                          X2: Union[torch.Tensor, SparseTensor],
                          v: Optional[torch.Tensor],
                          w: Optional[torch.Tensor], opt: FalkonOptions):
        """Choose which `dmmv` function to use for this data.

        Note that `dmmv` functions compute double kernel-vector products (see :meth:`dmmv` for
        an explanation of what they are).

        Parameters
        ----------
        X1 : torch.Tensor
            First data matrix, of shape (N x D)
        X2 : torch.Tensor
            Second data matrix, of shape (M x D)
        v : torch.Tensor or None
            Vector for the matrix-vector multiplication (M x T)
        w : torch.Tensor or None
            Vector for the matrix-vector multiplicatoin (N x T)
        opt : FalkonOptions
            Falkon options. Options may be specified to force GPU or CPU usage.

        Returns
        -------
        dmmv_fn
            A function which allows to perform the `mmv` operation.

        Notes
        -----
        This function decides based on the inputs: if the inputs are sparse, it will choose
        the sparse implementations; if CUDA is detected, it will choose the CUDA implementation;
        otherwise it will simply choose the basic CPU implementation.
        """
        sparsity = check_sparse(X1, X2)
        if not all(sparsity) and any(sparsity):
            raise ValueError("Either all or none of 'X1', 'X2' must be sparse.")
        return fdmmv

    @abstractmethod
    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool):
        """
        Compute the kernel matrix of ``X1`` and ``X2`` - without regards for differentiability.

        The kernel matrix should be stored in ``out`` to ensure the correctness of allocatable
        memory computations.

        Parameters
        ----------
        X1 : torch.Tensor
            The left matrix for computing the kernel
        X2 : torch.Tensor
            The right matrix for computing the kernel
        out : torch.Tensor
            The output matrix into which implementing classes should store the kernel.

        Returns
        -------
        out : torch.Tensor
            The kernel matrix. Should use the same underlying storage as the parameter ``out``.
        """
        pass

    @abstractmethod
    def compute_diff(self, X1: torch.Tensor, X2: torch.Tensor, diag: bool):
        """
        Compute the kernel matrix of ``X1`` and ``X2``. The output should be differentiable with
        respect to `X1`, `X2`, and all kernel parameters returned by the :meth:`diff_params` method.

        Parameters
        ----------
        X1 : torch.Tensor
            The left matrix for computing the kernel
        X2 : torch.Tensor
            The right matrix for computing the kernel

        Returns
        -------
        out : torch.Tensor
            The constructed kernel matrix.
        """
        pass

    @abstractmethod
    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor,
                       diag: bool, **kwargs) -> torch.Tensor:
        """
        Compute the kernel matrix of ``X1`` and ``X2`` which are two sparse matrices, storing the output
        in the dense matrix ``out``.

        Parameters
        ----------
        X1 : SparseTensor
            The left matrix for computing the kernel
        X2 : SparseTensor
            The right matrix for computing the kernel
        out : torch.Tensor
            The output matrix into which implementing classes should store the kernel.
        kwargs
            Additional keyword arguments which some sparse implementations might require. Currently
            the keyword arguments passed by the :func:`falkon.mmv_ops.fmmv.sparse_mmv_run_thread`
            and :func:`falkon.mmv_ops.fmm.sparse_mm_run_thread` functions are:

            - X1_csr : the X1 matrix in CSR format
            - X2_csr : the X2 matrix in CSR format

        Returns
        -------
        out : torch.Tensor
            The kernel matrix. Should use the same underlying storage as the parameter `out`.
        """
        pass

    @property
    @abstractmethod
    def diff_params(self) -> Dict[str, torch.Tensor]:
        """
        A dictionary mapping parameter names to their values for all **differentiable** parameters
        of the kernel.

        Returns
        -------
        params :
            A dictionary mapping parameter names to their values
        """
        pass

    @property
    @abstractmethod
    def nondiff_params(self) -> Dict[str, Any]:
        """
        A dictionary mapping parameter names to their values for all **non-differentiable**
        parameters of the kernel.

        Returns
        -------
        params :
            A dictionary mapping parameter names to their values
        """
        pass

    @abstractmethod
    def detach(self) -> 'Kernel':
        """Detaches all differentiable parameters of the kernel from the computation graph.

        Returns
        -------
        k :
            A new instance of the kernel sharing the same parameters, but detached from the
            computation graph.
        """
        pass

    def extra_mem(self) -> Dict[str, float]:
        """Compute the amount of extra memory which will be needed when computing this kernel.

        Often kernel computation needs some extra memory allocations. To avoid using too large
        block-sizes which may lead to OOM errors, you should declare any such extra allocations
        for your kernel here.

        Indicate extra allocations as coefficients on the required dimensions. For example,
        if computing a kernel needs to re-allocate the data-matrix (which is of size n * d),
        the return dictionary will be: `{'nd': 1}`. Other possible coefficients are on `d`, `n`, `m`
        which are respectively the data-dimension, the number of data-points in the first data
        matrix and the number of data-points in the second matrix. Pairwise combinations of the
        three dimensions are possible (i.e. `nd`, `nm`, `md`).
        Make sure to specify the dictionary keys as is written here since they will not be
        recognized otherwise.

        Returns
        -------
        extra_allocs : dictionary
            A dictionary from strings indicating on which dimensions the extra-allocation is
            needed (allowed strings: `'n', 'm', 'd', 'nm', 'nd', 'md'`) to floating-point numbers
            indicating how many extra-allocations are needed.
        """
        return {}

    def __str__(self):
        return f"<{self.name} kernel>"
