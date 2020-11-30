import math
from typing import Optional

import torch

import falkon
from falkon.mmv_ops.utils import _setup_opt, _get_cpu_ram
from falkon.options import BaseOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.helpers import (
    sizeof_dtype, select_dim_over_nd, select_dim_over_nm_v2
)
from falkon.utils.tensor_helpers import create_same_stride


def fmmv_cpu_sparse(X1: SparseTensor,
                    X2: SparseTensor,
                    v: torch.Tensor,
                    kernel: 'falkon.kernels.Kernel',
                    out: Optional[torch.Tensor],
                    opt: BaseOptions):
    opt = _setup_opt(opt, is_cpu=True)

    dtype = X1.dtype
    ntot, dtot = X1.size()
    mtot, T = v.size()

    # Create output matrix
    if out is None:
        out = torch.empty(ntot, T, dtype=dtype)
    out.fill_(0.0)

    avail_mem = _get_cpu_ram(opt, 0.95) / sizeof_dtype(dtype)
    # Narrowing X1, X2: n + m
    # Prepare - not computable, depends on kernel
    # ker_chunk : n*m
    # finalize : 0 (if can be implemented in place, kernel-dependent)
    n, m = select_dim_over_nm_v2(max_n=ntot, max_m=mtot, coef_nm=1, coef_n=1, coef_m=1, rest=0,
                                 max_mem=avail_mem)

    ker_chunk = create_same_stride((n, m), out, dtype, device='cpu')
    for i in range(0, ntot, n):
        ic = min(n, ntot - i)
        cur_out = out[i:i + ic, :]
        X1_chunk = X1.narrow_rows(i, ic)
        for j in range(0, mtot, m):
            jc = min(m, mtot - j)
            X2_chunk = X2.narrow_rows(j, jc)
            cur_ker_chunk = ker_chunk[:ic, :jc]
            cur_ker_chunk.fill_(0.0)

            ddd = kernel._prepare_sparse(X1_chunk, X2_chunk)
            kernel._apply_sparse(X1_chunk, X2_chunk.transpose_csc(), cur_ker_chunk)
            kernel._finalize(cur_ker_chunk, ddd)

            # Multiply by the vector v
            cur_out.addmm_(cur_ker_chunk, v.narrow(0, j, jc))
    return out


def fmmv_cpu(X1, X2, v, kernel, out, opt):
    """Blockwise kernel-vector product

    This function computes ``kernel(X1, X2) @ v`` in a blockwise fashion, to avoid having the
    whole N*M kernel matrix in memory at once.
    Note that while the principle is that of matrix-vector product, `v` can have more than
    one column.

    Parameters
    -----------
    X1
        [N, D] array
    X2
        [M, D] array
    v
        [M, T] array
    kernel
        Class representing the desired kernel function
    out : torch.Tensor or None
        [N, T] array for storing the kernel-vector product output.
        If None, will be allocated within the function.
    opt
        Basic options dictionary, used for determining available memory.
    """
    opt = _setup_opt(opt, is_cpu=True)

    ntot, dtot = X1.size(0), X1.size(1)
    M, T = v.size()
    dtype = v.dtype

    # Create output matrix
    if out is None:
        out = torch.empty(ntot, T, dtype=dtype)

    avail_mem = _get_cpu_ram(opt, 0.95) / sizeof_dtype(dtype)
    # Only necessary memory allocation is that for the temporary kernel
    # `temp_out` of size n*M
    extra_mem = kernel.extra_mem()
    n, d = select_dim_over_nd(max_n=ntot, max_d=dtot, coef_nd=extra_mem.get('nd', 0),
                              coef_n=M + extra_mem.get('n', 0) + extra_mem.get('nm', 0) * M,
                              coef_d=extra_mem.get('d', 0) + extra_mem.get('md', 0) * M,
                              rest=extra_mem.get('m', 0), max_mem=avail_mem)

    # Run batched matrix multiplication
    for i in range(0, ntot, n):
        ic = min(n, ntot - i)

        ddd = kernel._prepare(X1.narrow(0, i, ic), X2)  # , v=v)
        temp_out = torch.zeros(ic, M, dtype=dtype)
        for k in range(0, dtot, d):
            kc = min(d, dtot - k)
            X1d = X1[i: i + ic, k: k + kc]
            X2d = X2[:, k: k + kc]
            kernel._apply(X1d, X2d.T, temp_out)

        # temp_out = fnc(X1*X2', X1, X2)
        kernel._finalize(temp_out, ddd)

        torch.mm(temp_out, v, out=out[i: i + ic, :])
    return out


def fdmmv_cpu(X1, X2, v, w, kernel, out, opt):
    """Calculate a double kernel-vector product.

    This function computes the following quantity: ``kernel(X1, X2).T @ (kernel(X1, X2) @ v + w)``
    Where one of `v` or `w` can be empty.
    All arrays passed to this function must be 2-dimensional, although
    the second dimension can be unitary.

    The expression is not computed directly. We separate the computation
    into smaller blocks so as to reduce the total memory consumption (the
    large N*M kernel matrix is never wholly stored in RAM.)

    Parameters
    -----------
    X1
        [N, D] array
    X2
        [M, D] array
    v : torch.Tensor or None
        [M, T] array. But note that at least one of v or w must be specified.
    w : torch.Tensor or None
        [N, T] array. But note that at least one of v or w must be specified.
    kernel
        Class representing the desired kernel function
    out : torch.Tensor or None
        [M, T] array for storing the kernel-vector product output.
        If None, will be allocated within the function.
    opt
        Basic options dictionary, used for determining available memory.
    """
    opt = _setup_opt(opt, is_cpu=True)

    # Parameter validation
    if v is None and w is None:
        raise ValueError("One of v and w must be specified to run fMMV.")
    T = v.shape[1] if v is not None else w.shape[1]
    ntot, dtot = X1.size()
    M = X2.size(0)
    dtype = X1.dtype

    # Create output matrix
    if out is None:
        out = torch.empty(M, T, dtype=dtype)
    out.fill_(0)

    avail_mem = _get_cpu_ram(opt, 0.95) / sizeof_dtype(dtype)
    # The only necessary temporary matrices are: `temp_out` of size n*M and
    # temp_w_block of size n*T
    extra_mem = kernel.extra_mem()
    n, d = select_dim_over_nd(max_n=ntot, max_d=dtot, coef_nd=extra_mem.get('nd', 0),
                              coef_n=M + T + extra_mem.get('n', 0) + extra_mem.get('nm', 0) * M,
                              coef_d=extra_mem.get('d', 0) + extra_mem.get('md', 0) * M,
                              rest=extra_mem.get('m', 0), max_mem=avail_mem)

    # Run Batched Matrix Computation
    for i in range(0, ntot, n):
        ic = min(n, ntot - i)

        ddd = kernel._prepare(X1[i: i + ic, :], X2)
        temp_out = torch.zeros(ic, M, dtype=dtype)
        for k in range(0, dtot, d):
            kc = min(d, dtot - k)
            X1d = X1[i: i + ic, k: k + kc]
            X2d = X2[:, k: k + kc]
            kernel._apply(X1d, X2d.T, temp_out)
        kernel._finalize(temp_out, ddd)  # fnc(X1*X2', X1, X2) [n x M]

        w_blk = torch.zeros(ic, T, dtype=dtype)  # n x T
        if w is not None:
            w_blk.copy_(w[i: i + ic, :])
        if v is not None:
            # w_blk + c_out * v => (n x T) + (n x M)*(M x T)
            w_blk.addmm_(temp_out, v)

        out.add_(torch.mm(temp_out.T, w_blk))
    return out


def fdmmv_cpu_sparse(X1: SparseTensor,
                     X2: SparseTensor,
                     v: Optional[torch.Tensor],
                     w: Optional[torch.Tensor],
                     kernel,
                     out: Optional[torch.Tensor] = None,
                     opt: Optional[BaseOptions] = None):
    opt = _setup_opt(opt, is_cpu=True)

    # Parameter validation
    if v is None and w is None:
        raise ValueError("One of v and w must be specified to run fMMV.")
    T = v.size(1) if v is not None else w.size(1)
    ntot, dtot = X1.size()
    M = X2.size(0)
    dtype = X1.dtype

    # Create output matrix
    if out is None:
        out = torch.empty(M, T, dtype=dtype)
    out.fill_(0)

    avail_mem = _get_cpu_ram(opt, 0.95) / sizeof_dtype(dtype)
    # Narrow X1 : n
    # ker_chunk : n*M
    # w_blk     : n*T
    n = avail_mem / (M * T + 1)
    n = int(math.floor(n))
    if n < 1:
        raise MemoryError(("Available memory %.2fGB is insufficient "
                           "for blockwise fdMMv.") % (avail_mem * sizeof_dtype(dtype) / 2**30))

    # Allocate fixed arrays
    ker_chunk = create_same_stride((n, M), out, dtype, device='cpu')
    w_blk = create_same_stride((n, T), out, dtype, device='cpu')
    # Run blocked fdmmv
    for i in range(0, ntot, n):
        ic = min(n, ntot - i)
        X1_chunk = X1.narrow_rows(i, ic)
        cur_ker_chunk = ker_chunk[:ic]
        cur_ker_chunk.fill_(0.0)
        ddd = kernel._prepare_sparse(X1_chunk, X2)
        kernel._apply_sparse(X1_chunk, X2.transpose_csc(), cur_ker_chunk)
        kernel._finalize(cur_ker_chunk, ddd)

        # Multiply by the vector v
        cur_w_blk = w_blk[:ic]  # n x T
        cur_w_blk.fill_(0.0)
        if w is not None:
            cur_w_blk.copy_(w[i: i + ic, :])
        if v is not None:
            # w_blk + c_out * v => (n x T) + (n x M)*(M x T)
            cur_w_blk.addmm_(cur_ker_chunk, v)
        out.addmm_(cur_ker_chunk.T, cur_w_blk)
    del ker_chunk, w_blk
    return out
