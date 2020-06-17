import math
from typing import Optional

import numpy as np
import torch

import falkon
#from falkon.mmv_ops.sparse import transpose_csr, narrow_sparse_row, copy_sparse_to_dense
from falkon.mmv_ops.utils import _setup_opt, _get_cpu_ram
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.helpers import sizeof_dtype

__all__ = ("fmm_cpu", "fmm_cpu_sparse")


def blockwise_fmm_cpu(
        X1: torch.Tensor,
        X2: torch.Tensor,
        kernel: 'falkon.kernels.Kernel',
        out: torch.Tensor,
        max_mem: float) -> torch.Tensor:
    """
    The data-types of X1, X2 and out must be the same

    Parameters:
    -----------
     - X1 : tensor[N, D]
     - X2 : tensor[M, D]
     - kernel : Kernel
     - out : tensor[N, M]
     - max_mem : float
        The maximal amount of memory we can use on top of the amount
        already used by 'X1', 'X2' and 'out' matrices. This is used
        to constrain the block-size
    """
    # Invariant checks
    if sizeof_dtype(X1.dtype) == 8:
        raise RuntimeError("Blockwise kernel is only useful when data "
                           "uses 32-bit floating point")

    N, D = X1.shape
    M = X2.shape[0]

    # Compute the amount of additional memory in 8-byte blocks:
    # - D * batch_size (copy of chunk of X)
    # - D * batch_size (copy of chunk of Y)
    # - batch_size * batch_size (chunk of distance matrix)
    # Hence x² + 2*D*x = M, where x=batch_size, M=max_mem
    max_mem /= 8
    tmp = 2*D
    batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * max_mem)) / 2
    batch_size = int(math.floor(batch_size))
    if batch_size < 1:
        raise MemoryError(("Available memory %.2fGB is insufficient "
                           "for blockwise fMM.") % (max_mem / 2**30))

    out_chunk = torch.empty((batch_size, batch_size), dtype=torch.float64, device='cpu')

    for i in range(0, N, batch_size):
        len_i = min(N - i, batch_size)
        X1_chunk = X1.narrow(0, i, len_i).to(dtype=torch.float64)

        for j in range(0, M, batch_size):
            len_j = min(M - j, batch_size)

            if X1 is X2 and j < i:
                # when X1 is X2 the distance matrix is symmetric so we only need
                # to compute half of it.
                out[i:i+len_i, j:j+len_j].copy_(out[j:j+len_j, i:i+len_i].T)
            else:
                cur_out_chunk = out_chunk.narrow(0, 0, len_i).narrow(1, 0, len_j)
                cur_out_chunk.fill_(0.0)
                X2_chunk = X2.narrow(0, j, len_j).to(dtype=torch.float64)

                ddd = kernel._prepare(X1_chunk, X2_chunk)
                kernel._apply(X1_chunk, X2_chunk.T, cur_out_chunk)
                kernel._finalize(cur_out_chunk, ddd)

                # Copy back cur_out_chunk -> out
                out[i:i+len_i, j:j+len_j].copy_(cur_out_chunk.to(dtype=out.dtype))
    return out


def blockwise_fmm_cpu_sparse(
        X1: SparseTensor,
        X2: SparseTensor,
        kernel: 'falkon.kernels.Kernel',
        out: torch.Tensor,
        max_mem: float) -> torch.Tensor:
    """
    The data-types of X1, X2 and out must be the same

    Parameters:
    -----------
     - X1 : tensor[N, D]
     - X2 : tensor[M, D]
     - kernel : Kernel
     - out : tensor[N, M]
     - max_mem : float
        The maximal amount of memory we can use on top of the amount
        already used by 'X1', 'X2' and 'out' matrices. This is used
        to constrain the block-size
    """
    # Invariant checks
    if sizeof_dtype(X1.dtype) == 8:
        raise RuntimeError("Blockwise kernel is only useful when data "
                           "uses 32-bit floating point")

    N, D = X1.size()
    M = X2.size(0)
    density = (X1.nnz() + X2.nnz()) / (X1.size(0) * X1.size(1) + X2.size(0) * X2.size(1))

    # Compute the amount of additional memory in 8-byte blocks:
    # - D * batch_size * density + batch_size (copy of chunk of X)
    # - D * batch_size * density + batch_size (copy of chunk of Y)
    # - batch_size * batch_size (chunk of distance matrix)
    # Hence x² + 2*D*x + 2*x = M, where x=batch_size, M=max_mem
    max_mem /= 8
    tmp = 2 * D * math.sqrt(density) - 2
    batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * max_mem)) / 2
    batch_size = int(math.floor(batch_size))
    if batch_size < 1:
        raise MemoryError(("Available memory %.2fGB is insufficient "
                           "for blockwise fMM.") % (max_mem * 8 / 2**30))

    out_chunk = torch.empty((batch_size, batch_size), dtype=torch.float64, device='cpu')

    for i in range(0, N, batch_size):
        len_i = min(N - i, batch_size)
        # Here we need to copy a chunk of the row-pointers and all of the data
        X1_chunk = X1.narrow_rows(i, len_i).to(dtype=torch.float64)

        for j in range(0, M, batch_size):
            len_j = min(M - j, batch_size)

            if X1 is X2 and j < i:
                # when X1 is X2 the distance matrix is symmetric so we only need
                # to compute half of it.
                out[i:i+len_i, j:j+len_j].copy_(out[j:j+len_j, i:i+len_i].T)
            else:
                cur_out_chunk = out_chunk[:len_i, :len_j]
                cur_out_chunk.fill_(0.0)
                X2_chunk = X2.narrow_rows(j, len_j).to(dtype=torch.float64)

                ddd = kernel._prepare_sparse(X1_chunk, X2_chunk)
                kernel._apply_sparse(X1_chunk, X2_chunk.transpose_csc(), cur_out_chunk)
                kernel._finalize(cur_out_chunk, ddd)
                # Copy back cur_out_chunk -> out
                out[i:i+len_i, j:j+len_j].copy_(cur_out_chunk.to(dtype=out.dtype))
    return out


def fmm_cpu(
        X1: torch.Tensor,
        X2: torch.Tensor,
        kernel: 'falkon.kernels.Kernel',
        out: Optional[torch.Tensor],
        opt) -> torch.Tensor:
    """Compute kernel value on matrices X1 and X2: `out = kernel(X1, X2)`

    Parameters:
    -----------
    X1 : [N, D] array
    X2 : [M, D] array
    kernel : Kernel
        Class representing the desired kernel function
    out : Optional([N, M] array)
        Array for storing the kernel output. If None, will be allocated within the function.
    opt : Union(Dict, CompOpt)
        Options dictionary. Supported options are
         - 'final_type', the data-type of the output array. If 'out' is not None and it's
            data-type clashes with the setting of 'final_type', the out matrix will not be
            modified.

    Returns:
    --------
    out : [N, M] array
        The kernel between X1 and X2.
    """
    opt = _setup_opt(opt, is_cpu=True)
    ntot, dtot = X1.size()
    mtot = X2.size(0)

    if out is None:
        out = torch.empty(ntot, mtot, dtype=X1.dtype)

    if sizeof_dtype(X1.dtype) < 8 and opt.no_single_kernel:
        avail_mem = _get_cpu_ram(opt, 0.9)
        if avail_mem <= 0:
            raise MemoryError("Memory insufficient for kernel evaluation.")

        blockwise_fmm_cpu(X1, X2, kernel, out, avail_mem)
    else:
        # Do the kernel computation on the spot
        out.fill_(0.0)
        ddd = kernel._prepare(X1, X2)
        kernel._apply(X1, X2.T, out)
        kernel._finalize(out, ddd)

    return out


def fmm_cpu_sparse(
        X1: SparseTensor,
        X2: SparseTensor,
        kernel: 'falkon.kernels.Kernel',
        out: Optional[torch.Tensor],
        opt) -> torch.Tensor:
    opt = _setup_opt(opt, is_cpu=True)
    ntot, dtot = X1.size()
    mtot = X2.size(0)

    if out is None:
        out = torch.empty(ntot, mtot, dtype=X1.dtype)

    if sizeof_dtype(X1.dtype) < 8 and opt.no_single_kernel:
        avail_mem = _get_cpu_ram(opt, 0.9)
        if avail_mem <= 0:
            raise MemoryError("Memory insufficient for kernel evaluation.")

        blockwise_fmm_cpu_sparse(X1, X2, kernel, out, avail_mem)
    else:
        # Do the kernel computation on the spot
        out.fill_(0.0)
        ddd = kernel._prepare_sparse(X1, X2)
        kernel._apply_sparse(X1, X2.transpose_csc(), out)
        kernel._finalize(out, ddd)

    return out
