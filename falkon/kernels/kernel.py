import dataclasses
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Any

import torch

import falkon
from falkon.mmv_ops.fmm_cpu import fmm_cpu_sparse, fmm_cpu
from falkon.mmv_ops.fmmv_cpu import fdmmv_cpu_sparse, fmmv_cpu_sparse, fmmv_cpu, fdmmv_cpu
from falkon.utils.helpers import check_same_dtype, check_sparse, check_same_device
from falkon.utils import decide_cuda
from falkon.options import FalkonOptions


class Kernel(ABC):
    """Abstract kernel class. Kernels should inherit from this class, overriding appropriate methods.

    To extend Falkon with new kernels, you should read the documentation of this class
    carefully. In particular, you will **need** to implement :meth:`_prepare`, :meth:`_apply` and
    :meth:`_finalize` methods.

    Other methods which should be optionally implemented are the sparse versions
    :meth:`_prepare_sparse` and :meth:`_apply_sparse` (note that there is no `_finalize_sparse`,
    since the `_finalize` takes as input a partial kernel matrix, and even with sparse data,
    kernel matrices are assumed to be dense. Therefore, even for sparse data, the :meth:`_finalize`
    method will be used.

    To provide a KeOps implementation, you will have to inherit also from the
    :class:`~falkon.kernels.keops_helpers.KeopsKernelMixin` class, and implement its abstract methods. In case
    a KeOps implementation is provided, you should make sure to override the
    :meth:`_decide_mmv_impl` and :meth:`_decide_dmmv_impl` so that the KeOps implementation is
    effectively used. Have a look at the :class:`falkon.kernels.PolynomialKernel` class for
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
        self.name = name
        self.kernel_type = kernel_type
        if opt is None:
            opt = FalkonOptions()
        self.params: FalkonOptions = opt

    @staticmethod
    def _check_dmmv_dimensions(X1, X2, v, w, out):
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
    def _check_mmv_dimensions(X1, X2, v, out):
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
    def _check_mm_dimensions(X1, X2, out):
        # Parameter validation
        if X1.dim() != 2:
            raise ValueError("Matrix X1 must be 2D.")
        if X2.dim() != 2:
            raise ValueError("Matrix X2 must be 2D.")
        N = X1.size(0)
        M = X2.size(0)
        if out is not None and out.shape != (N, M):
            raise ValueError(
                f"Output dimension is incorrect. "
                f"Expected ({N}, {M}) found {out.shape}")

        if not check_same_dtype(X1, X2, out):
            raise TypeError("Data types of input matrices must be equal.")

        return X1, X2, out

    @staticmethod
    def _check_device_properties(*args, fn_name: str, opt: FalkonOptions):
        if not check_same_device(*args):
            raise RuntimeError("All input arguments to %s must be on the same device" % (fn_name))

    def __call__(self, X1, X2, out=None, opt: Optional[FalkonOptions] = None):
        """Compute the kernel matrix between `X1` and `X2`

        Parameters
        ----------
        X1 : torch.Tensor
            The first data-matrix for computing the kernel. Of shape (N x D):
            N samples in D dimensions.
        X2 : torch.Tensor
            The second data-matrix for computing the kernel. Of shape (M x D):
            M samples in D dimensions. Set `X2 == X1` to compute a symmetric kernel.
        out : torch.Tensor or None
            Optional tensor of shape (N x M) to hold the output. If not provided it will
            be created.
        opt : Optional[FalkonOptions]
            Options to be used for computing the operation. Useful are the memory size options
            and CUDA options.

        Returns
        -------
        out : torch.Tensor
            The kernel between `X1` and `X2`.
        """
        X1, X2, out = self._check_mm_dimensions(X1, X2, out)
        self._check_device_properties(X1, X2, out, fn_name="kernel", opt=opt)
        params = self.params
        if opt is not None:
            params = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        mm_impl = self._decide_mm_impl(X1, X2, params)
        return mm_impl(X1, X2, self, out, params)

    def _decide_mm_impl(self, X1, X2, opt: FalkonOptions):
        """Choose which `mm` function to use for this data.

        Note that `mm` functions compute the kernel itself so **KeOps may not be used**.

        Parameters
        ----------
        X1 : torch.Tensor
            First data matrix, of shape (N x D)
        X2 : torch.Tensor
            Second data matrix, of shape (M x D)
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
        use_cuda = decide_cuda(opt)
        sparsity = check_sparse(X1, X2)
        if not all(sparsity) and any(sparsity):
            raise ValueError("Either all or none of 'X1', 'X2' must be sparse.")
        sparsity = all(sparsity)
        if (X1.device.type == 'cuda') and (not use_cuda):
            warnings.warn("kernel backend was chosen to be CPU, but GPU input tensors found. "
                          "Defaulting to use the GPU (note this may cause issues later). "
                          "To force usage of the CPU backend, please pass CPU tensors; "
                          "to avoid this warning if the GPU backend is "
                          "desired, check your options (i.e. set 'use_cpu=False').")
            use_cuda = True
        if use_cuda:
            from falkon.mmv_ops.fmm_cuda import fmm_cuda, fmm_cuda_sparse
            if sparsity:
                return fmm_cuda_sparse
            else:
                return fmm_cuda
        else:
            if sparsity:
                return fmm_cpu_sparse
            else:
                return fmm_cpu

    def mmv(self, X1, X2, v, out=None, opt: Optional[FalkonOptions] = None):
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
        >>> k = falkon.kernels.GaussianKernel(sigma=2)  # You can substitute the Gaussian kernel by any other.
        >>> X1 = torch.randn(100, 3)
        >>> X2 = torch.randn(150, 3)
        >>> v = torch.randn(150, 1)
        >>> out = k.mmv(X1, X2, v, out=None)
        >>> out.shape
        torch.Size([100, 1])
        """
        X1, X2, v, out = self._check_mmv_dimensions(X1, X2, v, out)
        self._check_device_properties(X1, X2, v, out, fn_name="mmv", opt=opt)

        params = self.params
        if opt is not None:
            params = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        mmv_impl = self._decide_mmv_impl(X1, X2, v, params)
        return mmv_impl(X1, X2, v, self, out, params)

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
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
        use_cuda = decide_cuda(opt)
        sparsity = check_sparse(X1, X2)
        if not all(sparsity) and any(sparsity):
            raise ValueError("Either all or none of 'X1', 'X2' must be sparse.")
        if (X1.device.type == 'cuda') and (not use_cuda):
            warnings.warn("kernel-vector product backend was chosen to be CPU, but GPU input "
                          "tensors found. Defaulting to use the GPU (note this may "
                          "cause issues later). To force usage of the CPU backend, "
                          "please pass CPU tensors; to avoid this warning if the GPU backend is "
                          "desired, check your options (i.e. set 'use_cpu=False').")
            use_cuda = True
        sparsity = all(sparsity)
        if use_cuda:
            from falkon.mmv_ops.fmmv_cuda import fmmv_cuda, fmmv_cuda_sparse
            if sparsity:
                return fmmv_cuda_sparse
            else:
                return fmmv_cuda
        else:
            if sparsity:
                return fmmv_cpu_sparse
            else:
                return fmmv_cpu

    def dmmv(self, X1, X2, v, w, out=None, opt: Optional[FalkonOptions] = None):
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
        self._check_device_properties(X1, X2, v, w, out, fn_name="dmmv", opt=opt)
        params = self.params
        if opt is not None:
            params = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        dmmv_impl = self._decide_dmmv_impl(X1, X2, v, w, params)
        return dmmv_impl(X1, X2, v, w, self, out, params)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
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
        use_cuda = decide_cuda(opt)
        sparsity = check_sparse(X1, X2)
        if not all(sparsity) and any(sparsity):
            raise ValueError("Either all or none of 'X1', 'X2' must be sparse.")
        if (X1.device.type == 'cuda') and (not use_cuda):
            warnings.warn("kernel-vector double product backend was chosen to be CPU, but GPU "
                          "input tensors found. Defaulting to use the GPU (note this may "
                          "cause issues later). To force usage of the CPU backend, "
                          "please pass CPU tensors; to avoid this warning if the GPU backend is "
                          "desired, check your options (i.e. set 'use_cpu=False').")
            use_cuda = True
        sparsity = all(sparsity)
        if use_cuda:
            from falkon.mmv_ops.fmmv_cuda import fdmmv_cuda, fdmmv_cuda_sparse
            if sparsity:
                return fdmmv_cuda_sparse
            else:
                return fdmmv_cuda
        else:
            if sparsity:
                return fdmmv_cpu_sparse
            else:
                return fdmmv_cpu

    @abstractmethod
    def _prepare(self, X1, X2) -> Any:
        """Pre-processing operations necessary to compute the kernel.

        This function will be called with two blocks of data which may be subsampled on the
        first dimension (i.e. X1 may be of size `n x D` where `n << N`). The function should
        not modify `X1` and `X2`. If necessary, it may return some data which is then made available
        to the :meth:`_finalize` method.

        For example, in the Gaussian kernel, this method is used to compute the squared norms
        of the datasets.

        Parameters
        ----------
        X1 : torch.Tensor
            (n x D) tensor. It is a block of the `X1` input matrix, possibly subsampled in the
            first dimension.
        X2 : torch.Tensor
            (m x D) tensor. It is a block of the `X2` input matrix, possibly subsampled in the
            first dimension.

        Returns
        -------
        Data which may be used for the :meth:`_finalize` method. If no such information is
        necessary, returns None.
        """
        pass

    @abstractmethod
    def _apply(self, X1, X2, out) -> None:
        """Main kernel operation, usually matrix multiplication.

        This function will be called with two blocks of data which may be subsampled on the
        first and second dimension (i.e. X1 may be of size `n x d` where `n << N` and `d << D`).
        The output shall be stored in the `out` argument, and not be returned.

        Parameters
        ----------
        X1 : torch.Tensor
            (n x d) tensor. It is a block of the `X1` input matrix, possibly subsampled in the
            first dimension.
        X2 : torch.Tensor
            (m x d) tensor. It is a block of the `X2` input matrix, possibly subsampled in the
            first dimension.
        out : torch.Tensor
            (n x m) tensor. A tensor in which the output of the operation shall be accumulated.
            This tensor is initialized to 0 before calling `_apply`, but in case of subsampling
            of the data along the second dimension, multiple calls will be needed to compute a
            single (n x m) output block. In such case, the first call to this method will have
            a zeroed tensor, while subsequent calls will simply reuse the same object.
        """
        pass

    @abstractmethod
    def _finalize(self, A, d):
        """Final actions to be performed on a partial kernel matrix.

        All elementwise operations on the kernel matrix should be performed in this method.
        Operations should be performed inplace by modifying the matrix `A`, to improve memory
        efficiency. If operations are not in-place, out-of-memory errors are possible when
        using the GPU.

        Parameters
        ----------
        A : torch.Tensor
            A (m x n) tile of the kernel matrix, as obtained by the :meth:`_apply` method.
        d
            Additional data, as computed by the :meth:`_prepare` method.

        Returns
        -------
        A
            The same tensor as the input, if operations are performed in-place. Otherwise
            another tensor of the same shape.
        """
        pass

    @abstractmethod
    def _prepare_sparse(self, X1, X2):
        """Data preprocessing for sparse tensors.

        This is an equivalent to the :meth:`_prepare` method for sparse tensors.

        Parameters
        ----------
        X1 : falkon.sparse.sparse_tensor.SparseTensor
            Sparse tensor of shape (n x D), with possibly n << N.
        X2 : falkon.sparse.sparse_tensor.SparseTensor
            Sparse tensor of shape (m x D), with possibly m << M.

        Returns
        -------
        Data derived from `X1` and `X2` which is needed by the :meth:`_finalize` method when
        finishing to compute a kernel tile.
        """
        raise NotImplementedError("_prepare_sparse not implemented for kernel %s" %
                                  (self.kernel_type))

    @abstractmethod
    def _apply_sparse(self, X1, X2, out) -> None:
        """Main kernel computation for sparse tensors.

        Unlike the :meth`_apply` method, the `X1` and `X2` tensors are only subsampled along
        the first dimension. Take note that the `out` tensor **is not sparse**.

        Parameters
        ----------
        X1 : falkon.sparse.sparse_tensor.SparseTensor
            Sparse tensor of shape (n x D), with possibly n << N.
        X2 : falkon.sparse.sparse_tensor.SparseTensor
            Sparse tensor of shape (m x D), with possibly m << M.
        out : torch.Tensor
            Tensor of shape (n x m) which should hold a tile of the kernel. The output of this
            function (typically a matrix multiplication) should be placed in this tensor.
        """
        raise NotImplementedError("_apply_sparse not implemented for kernel %s" %
                                  (self.kernel_type))

    def __str__(self):
        return f"<{self.name} kernel>"
