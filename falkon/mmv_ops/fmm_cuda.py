#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:09:33 2017

@author: alessandro
"""
from dataclasses import dataclass
from typing import Optional, Union

import torch

import falkon
from falkon.mmv_ops.utils import *
from falkon.options import BaseOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.cuda_helpers import copy_to_host_noorder
from falkon.utils.helpers import (
    select_dim_fMM, calc_gpu_block_sizes, sizeof_dtype,
    select_dim_over_m
)
from falkon.utils.tensor_helpers import create_same_stride, cast_tensor, create_fortran

__all__ = ("fmm_cuda", "fmm_cuda_sparse")


@dataclass(frozen=True)
class ArgsFmm():
    X1: Union[torch.Tensor, SparseTensor]
    X2: Union[torch.Tensor, SparseTensor]
    out: torch.Tensor
    kernel: 'falkon.kernels.Kernel'
    gpu_dtype: torch.dtype
    max_mem: float


def _sparse_fmm(proc_idx, queue, device_id):
    a: ArgsFmm = queue.get()
    X1: SparseTensor = a.X1
    X2: SparseTensor = a.X2
    out = a.out
    kernel, gpu_dtype = a.kernel, a.gpu_dtype
    max_mem = a.max_mem

    ntot, dtot = X1.shape
    mtot = X2.size(0)

    avail_mem = max_mem / sizeof_dtype(gpu_dtype)
    # Memory usage:
    # X1_chunk : ntot + 2 * D * ntot * density
    # X2_chunk : dtot + 2 * D * mtot * density (because is transposed)
    # sparse_out : ntot + 2 * ntot * mtot * density (assume density=1 here)
    # ker_gpu  : mtot * ntot
    n, m = select_dim_over_m(
        maxN=ntot, maxM=mtot, tot=avail_mem,
        coef_nm=3, coef_m=2*dtot*X2.density, coef_n=2+2*dtot*X1.density, rest=dtot)

    tc_device = torch.device('cuda:%d' % (int(device_id)))
    with torch.cuda.device(tc_device):
        # Initialize GPU buffers
        g_out = create_same_stride((n, m), out, gpu_dtype, tc_device)
        cpu_buf = None
        if X1.dtype != gpu_dtype:
            cpu_buf = create_same_stride((n, m), out, gpu_dtype, 'cpu', pin_memory=True)

        for j in range(0, mtot, m):
            jc = min(m, mtot - j)

            X2_chunk = X2.narrow_rows(j, jc).to(dtype=gpu_dtype)
            X2_chunk_d = SparseTensor.from_scipy(
                X2_chunk.transpose_csc().to_scipy().tocsr(copy=False)) \
                .index_to_int() \
                .to(device=tc_device)
            for i in range(0, ntot, n):
                ic = min(n, ntot - i)

                X1_chunk = X1.narrow_rows(i, ic).to(dtype=gpu_dtype)
                X1_chunk_d = X1_chunk.index_to_int().to(device=tc_device)
                cur_g_out = g_out.narrow(0, 0, ic).narrow(1, 0, jc)
                cur_g_out.fill_(0.0)

                ddd = kernel._prepare_sparse(X1_chunk, X2_chunk)
                cur_g_out = kernel._apply_sparse(X1_chunk_d, X2_chunk_d, cur_g_out)
                cur_g_out = kernel._finalize(cur_g_out, ddd)
                copy_to_host_noorder(ic, jc, cur_g_out, 0, 0, out, i, j, cpu_buf)
                del ddd, X1_chunk_d, X1_chunk
            del X2_chunk, X2_chunk_d
        del g_out
    return out


def _generic_fmm(proc_idx, queue, device_id):
    a: ArgsFmm = queue.get()
    X1: torch.Tensor = a.X1
    X2: torch.Tensor = a.X2
    out = a.out
    kernel, gpu_dtype = a.kernel, a.gpu_dtype
    max_mem = a.max_mem

    ntot, dtot = X1.shape
    mtot = X2.shape[0]

    # This function is slightly faster if we limit the sizes
    # of the processed blocks slightly. Especially when doing
    # a cold run since pinned-memory allocation is extremely slow.
    # We don't want to do it if we're memory constrained though.
    if max_mem > 4*2**30:
        max_mem /= 4
    avail_mem = max_mem / sizeof_dtype(gpu_dtype)
    # Memory usage:
    # - gOut    : n x m
    # - g_ssX1  : n x d
    # - g_sX2   : m x d
    # total : n*d + m*d + n*m
    n, d, m = select_dim_fMM(avail_mem, ntot, dtot, mtot)

    tc_device = torch.device('cuda:%d' % (int(device_id)))
    with torch.cuda.device(tc_device):
        # Initialize GPU buffers
        g_out = create_same_stride((n, m), out, gpu_dtype, tc_device)
        g_X1d = create_same_stride((n, d), X1, gpu_dtype, tc_device)
        g_X2d = create_same_stride((m, d), X2, gpu_dtype, tc_device)
        cpu_buf = None
        if X1.dtype != gpu_dtype:
            cpu_buf = create_same_stride((n, m), out, gpu_dtype, 'cpu', pin_memory=True)

        for j in range(0, mtot, m):
            jc = min(m, mtot - j)
            X2_chunk = cast_tensor(X2.narrow(0, j, jc), dtype=gpu_dtype, warn=False).pin_memory()

            for i in range(0, ntot, n):
                ic = min(n, ntot - i)

                if _gpu_tns_same_memory(X1, X2) and j < i:
                    out[i:i + ic, j:j + jc].copy_(out[j:j + jc, i:i + ic].T)
                else:
                    X1_chunk = cast_tensor(X1.narrow(0, i, ic), dtype=gpu_dtype, warn=False).pin_memory()

                    ddd = kernel._prepare(X1_chunk, X2_chunk)

                    cur_g_out = g_out.narrow(0, 0, ic).narrow(1, 0, jc)
                    cur_g_out.fill_(0.0)

                    for k in range(0, dtot, d):
                        kc = min(d, dtot - k)
                        # Move to GPU
                        cur_g_X1d = g_X1d.narrow(0, 0, ic).narrow(1, 0, kc)
                        cur_g_X1d.copy_(X1_chunk.narrow(1, k, kc))
                        cur_g_X2d = g_X2d.narrow(0, 0, jc).narrow(1, 0, kc)
                        cur_g_X2d.copy_(X2_chunk.narrow(1, k, kc))
                        # Apply
                        a.kernel._apply(cur_g_X1d, cur_g_X2d.T, cur_g_out)

                    a.kernel._finalize(cur_g_out, ddd)
                    copy_to_host_noorder(ic, jc, cur_g_out, 0, 0, out, i, j, cpu_buf)
                    del ddd
        del g_out, g_X1d, g_X2d
    return out


def fmm_cuda(X1: torch.Tensor,
             X2: torch.Tensor,
             kernel: 'falkon.kernels.Kernel',
             out: Optional[torch.Tensor] = None,
             opt: Optional[BaseOptions] = None) -> torch.Tensor:
    """
    performs fnc(X1*X2', X1, X2) in blocks on multiple GPUs
    """
    opt = _setup_opt(opt)
    _check_contiguity((X1, 'X1'), (X2, 'X2'), (out, 'out'))

    N = X1.size(0)
    M = X2.size(0)
    if out is None:
        out = create_same_stride((N, M), X1, X1.dtype, 'cpu', pin_memory=True)
    gpu_info = _get_gpu_info(opt, slack=0.9)
    block_sizes = calc_gpu_block_sizes(gpu_info, N)

    # If float32 we need to upcast to float64 to avoid numerical precision errors
    # in the kernel
    gpu_dtype = X1.dtype
    if sizeof_dtype(X1.dtype) < 8 and opt.no_single_kernel:
        gpu_dtype = torch.float64

    # Create the arguments passed to each subprocess
    args = []
    for i, g in enumerate(gpu_info):
        bwidth = block_sizes[i + 1] - block_sizes[i]
        if bwidth <= 0: continue
        args.append((ArgsFmm(X1=X1.narrow(0, block_sizes[i], bwidth),
                             X2=X2, out=out.narrow(0, block_sizes[i], bwidth),
                             kernel=kernel, gpu_dtype=gpu_dtype, max_mem=g.usable_ram), g.Id))
    _start_wait_processes(_generic_fmm, args)
    torch.cuda.empty_cache()
    return out


def fmm_cuda_sparse(X1: SparseTensor,
                    X2: SparseTensor,
                    kernel: 'falkon.kernels.Kernel',
                    out: Optional[torch.Tensor] = None,
                    opt: Optional[BaseOptions] = None) -> torch.Tensor:
    opt = _setup_opt(opt)
    _check_contiguity((out, 'out'))
    N = X1.size(0)
    M = X2.size(0)
    if out is None:
        out = create_fortran((N, M), X1.dtype, 'cpu', pin_memory=True)
    gpu_info = _get_gpu_info(opt, slack=0.9)
    block_sizes = calc_gpu_block_sizes(gpu_info, N)

    # If float32 we need to upcast to float64 to avoid numerical precision errors
    # in the kernel
    gpu_dtype = X1.dtype
    if sizeof_dtype(X1.dtype) < 8 and opt.no_single_kernel:
        gpu_dtype = torch.float64

    # Create the arguments passed to each subprocess
    args = []
    for i, g in enumerate(gpu_info):
        bwidth = block_sizes[i + 1] - block_sizes[i]
        if bwidth <= 0: continue
        args.append((ArgsFmm(
            X1=X1.narrow_rows(block_sizes[i], bwidth),
            X2=X2, out=out.narrow(0, block_sizes[i], bwidth),
            kernel=kernel, gpu_dtype=gpu_dtype, max_mem=g.usable_ram), g.Id))
    _start_wait_processes(_sparse_fmm, args)
    torch.cuda.empty_cache()
    return out
