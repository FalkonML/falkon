#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import torch
import torch.cuda as tcd

import falkon
from falkon.cuda.cudart_gpu import cuda_memcpy2d_async
from falkon.mmv_ops.utils import *
from falkon.options import BaseOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.cuda_helpers import copy_to_host_noorder, copy_to_host
from falkon.utils.stream_utils import sync_current_stream
from falkon.utils.helpers import (
    calc_gpu_block_sizes, sizeof_dtype,
    select_dim_over_nm, select_dim_over_nm_v2,
)
from falkon.utils.tensor_helpers import create_same_stride, create_fortran, is_f_contig

__all__ = ("fmm_cuda", "fmm_cuda_sparse")


@dataclass(frozen=True)
class ArgsFmm():
    X1: Union[torch.Tensor, SparseTensor]
    X2: Union[torch.Tensor, SparseTensor]
    out: torch.Tensor
    kernel: 'falkon.kernels.Kernel'
    gpu_dtype: torch.dtype
    max_mem: float
    num_streams: int = 1


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
    n, m = select_dim_over_nm_v2(max_n=ntot, max_m=mtot, coef_nm=3,
                                 coef_n=2 + 2 * dtot * X1.density, coef_m=2 * dtot * X2.density,
                                 rest=dtot, max_mem=avail_mem)

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


# noinspection PyUnboundLocalVariable
def _generic_fmm(proc_idx, queue, device_id):
    # Unpack the function arguments
    a: ArgsFmm = queue.get()
    X1: torch.Tensor = a.X1
    X2: torch.Tensor = a.X2
    cuda_inputs = X1.is_cuda
    out = a.out
    kernel, gpu_dtype = a.kernel, a.gpu_dtype
    max_mem = a.max_mem
    num_streams = a.num_streams

    # flags and local variables
    change_dtype = gpu_dtype != X1.dtype
    X1_equal_X2 = _gpu_tns_same_memory(X1, X2)
    use_gpu_bufs = change_dtype or not cuda_inputs
    stride = "F" if is_f_contig(out, strict=True) else "C"
    j_iter = 0
    dts = sizeof_dtype(gpu_dtype)
    tc_device = torch.device('cuda:%d' % (int(device_id)))
    avail_mem = max_mem / dts

    # Choose block sizes n, m such that we won't run out of GPU memory
    ntot, d = X1.shape
    mtot = X2.shape[0]
    extra_mem = kernel.extra_mem()
    if cuda_inputs and not change_dtype:
        # No allocation will be performed by us. Only in-kernel stuff.
        n, m = select_dim_over_nm(max_n=ntot, max_m=mtot, d=d,
                                  coef_nd=extra_mem.get('nd', 0),
                                  coef_md=extra_mem.get('md', 0),
                                  coef_nm=extra_mem.get('nm', 0),
                                  coef_n=extra_mem.get('n', 0),
                                  coef_m=extra_mem.get('m', 0),
                                  rest=extra_mem.get('d', 0),
                                  max_mem=avail_mem)
    else:
        n, m = select_dim_over_nm(max_n=ntot, max_m=mtot, d=d,
                                  coef_nd=num_streams * (extra_mem.get('nd', 0) + 1),
                                  coef_md=num_streams * (extra_mem.get('md', 0) + 1),
                                  coef_nm=num_streams * (extra_mem.get('nm', 0) + 1),
                                  coef_n=extra_mem.get('n', 0),
                                  coef_m=extra_mem.get('m', 0),
                                  rest=extra_mem.get('d', 0),
                                  max_mem=avail_mem)

    # Create streams
    streams = [tcd.Stream(device=tc_device) for _ in range(num_streams)]

    # Create buffers
    if use_gpu_bufs:
        gX1 = create_same_stride((n, d), X1, gpu_dtype, tc_device)
        gX2_list = [create_same_stride((m, d), X2, gpu_dtype, tc_device) for _ in range(num_streams)]
        gout_list = [create_same_stride((n, m), out, gpu_dtype, tc_device) for _ in range(num_streams)]
    if not cuda_inputs:
        cpu_buf_list = [create_same_stride((n, m), out, gpu_dtype, 'cpu', pin_memory=True) for _ in range(num_streams)]

    # Define helpers for the copy-back operations (from cpu_buf to output)
    copy_ops = [None] * num_streams

    def wrap_copy_op(stream_idx):
        if copy_ops[stream_idx] is not None:
            copy_ops[stream_idx]()
            copy_ops[stream_idx] = None

    def do_copy_op(output, buf, i_, ic_, j_, jc_):
        # This function will also do the type conversion
        output[i_:i_ + ic_, j_:j_ + jc_].copy_(buf[:ic_, :jc_])

    # Kernel computation begin
    with tcd.device(tc_device):
        for i in range(0, ntot, n):
            ic = min(n, ntot - i)

            with tcd.stream(streams[j_iter % len(streams)]):
                X1_chunk = X1.narrow(0, i, ic)
                if use_gpu_bufs:
                    cur_gX1 = gX1.narrow(0, 0, ic)
                    cur_gX1.copy_(X1_chunk, non_blocking=True)
                else:
                    cur_gX1 = X1_chunk

            for j in range(0, mtot, m):
                jc = min(m, mtot - j)
                # Choose the buffers for this inner iteration
                stream_id = j_iter % len(streams)
                stream = streams[stream_id]
                if use_gpu_bufs:
                    gX2 = gX2_list[stream_id]
                    gout = gout_list[stream_id]
                if not cuda_inputs:
                    cpu_buf = cpu_buf_list[stream_id]

                # Sync for buffers we must use now (e.g. 2 previous iters)
                with tcd.stream(stream):  # Inner-loop
                    stream.synchronize()
                    wrap_copy_op(stream_id)

                    if X1_equal_X2 and j < i:  # Shortcut for symmetric kernels
                        jc = min(m, mtot - j)
                        out[i:i + ic, j:j + jc].copy_(out[j:j + jc, i:i + ic].T, non_blocking=True)
                        j_iter += 1
                        continue

                    # Copy (CPU->GPU)
                    X2_chunk = X2.narrow(0, j, jc)
                    if use_gpu_bufs:
                        cur_gX2 = gX2.narrow(0, 0, jc)
                        cur_gX2.copy_(X2_chunk, non_blocking=True)
                    else:
                        cur_gX2 = X2_chunk

                    if use_gpu_bufs:
                        cur_gout = gout[:ic, :jc]
                    else:
                        cur_gout = out[i:i + ic, j:j + jc]
                    cur_gout.fill_(0.0)

                    # Compute
                    ddd = kernel._prepare(cur_gX1, cur_gX2)
                    kernel._apply(cur_gX1, cur_gX2.T, cur_gout)
                    cur_gout = kernel._finalize(cur_gout, ddd)

                    # Copy Back (GPU->CPU)
                    if not cuda_inputs:
                        # copy_ does not care about the contiguity of copies, as long as it's consistent
                        # however, in case of C-contiguous inputs it will create an intermediate array
                        # which is undesired. We use cuda_memcpy2d_async which works well with C-contiguous
                        # arrays.
                        if stride == "F":
                            copy_to_host(ic, jc, cur_gout, 0, 0, cpu_buf, 0, 0, s=stream)
                        else:
                            cuda_memcpy2d_async(
                                dst=cpu_buf.data_ptr(), dpitch=cpu_buf.stride(0) * dts,
                                src=cur_gout.data_ptr(), spitch=cur_gout.stride(0) * dts,
                                width=jc * dts, height=ic, stream=stream._as_parameter_)
                        copy_ops[stream_id] = partial(do_copy_op, out, cpu_buf, i, ic, j, jc)
                    elif change_dtype:
                        out.narrow(0, i, ic).narrow(1, j, jc).copy_(cur_gout, non_blocking=True)
                j_iter += 1

            for i in range(num_streams):
                streams[i].synchronize()
                wrap_copy_op(i)

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

    N = X1.shape[0]
    M = X2.shape[0]
    device = X1.device
    if out is None:
        out = create_same_stride((N, M), X1, X1.dtype, device=device,
                                 pin_memory=False)
    gpu_info = _get_gpu_info(opt, slack=0.9)
    block_sizes = calc_gpu_block_sizes(gpu_info, N)

    # If float32 we need to upcast to float64 to avoid numerical precision errors
    # in the kernel
    gpu_dtype = X1.dtype
    if sizeof_dtype(X1.dtype) < 8 and opt.no_single_kernel:
        gpu_dtype = torch.float64

    if device.type == 'cuda':
        sync_current_stream(device)
        single_gpu_info = [g for g in gpu_info if g.Id == device.index][0]
        args = ArgsFmm(X1=X1, X2=X2, out=out, kernel=kernel, gpu_dtype=gpu_dtype,
                       max_mem=single_gpu_info.usable_ram,
                       num_streams=opt.num_fmm_streams)
        _call_direct(_generic_fmm, (args, device.index))
    else:
        # Create the arguments passed to each subprocess
        args = []
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            args.append((ArgsFmm(X1=X1.narrow(0, block_sizes[i], bwidth),
                                 X2=X2, out=out.narrow(0, block_sizes[i], bwidth),
                                 kernel=kernel, gpu_dtype=gpu_dtype, max_mem=g.usable_ram,
                                 num_streams=opt.num_fmm_streams), g.Id))
        _start_wait_processes(_generic_fmm, args)
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
        if bwidth <= 0:
            continue
        args.append((ArgsFmm(
            X1=X1.narrow_rows(block_sizes[i], bwidth),
            X2=X2, out=out.narrow(0, block_sizes[i], bwidth),
            kernel=kernel, gpu_dtype=gpu_dtype, max_mem=g.usable_ram), g.Id))
    _start_wait_processes(_sparse_fmm, args)
    torch.cuda.empty_cache()
    return out
