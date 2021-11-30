import dataclasses
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.cuda as tcd

from falkon.mmv_ops.utils import (
    _get_gpu_info, _call_direct, _start_wait_processes, _dev_from_id, _is_incore
)
from falkon.options import BaseOptions
from falkon.utils.helpers import (
    sizeof_dtype, choose_fn, check_same_device, calc_gpu_block_sizes, select_dim_over_n
)
from falkon.kernels import Kernel
from falkon.utils.tensor_helpers import (
    extract_fortran, extract_same_stride, is_f_contig
)
from falkon.utils.device_copy import copy


@dataclass(frozen=True)
class KernelTrsmArgs:
    m1: torch.Tensor
    m2: torch.Tensor
    tri: torch.Tensor
    kernel: Kernel
    max_mem: float
    lower: bool
    transpose: bool


def inplace_trsm(A: torch.Tensor, v: torch.Tensor, alpha: float, lower: bool, transpose: bool) -> torch.Tensor:
    if not is_f_contig(v, strict=False):
        # noinspection PyArgumentList
        raise ValueError(f"vector must be f-contig. Found shape {v.shape}, stride {v.stride()}.")
    if not is_f_contig(A, strict=False):
        transpose = not transpose
    dev = A.device
    if dev.type == 'cuda':
        from falkon.cuda.initialization import cublas_handle
        from falkon.cuda.cublas_gpu import cublasDtrsm, cublasStrsm, cublas_stream
        cublas_hdl = cublas_handle(A.device.index)
        trsm_fn = choose_fn(A.dtype, cublasDtrsm, cublasStrsm, "TRSM")
        s = tcd.current_stream(dev)
        with cublas_stream(cublas_hdl, s._as_parameter_):
            uplo = 'L' if lower else 'U'
            trans = 'T' if transpose else 'N'
            trsm_fn(cublas_hdl, side='L', uplo=uplo, trans=trans, diag='N', m=v.shape[0], n=v.shape[1],
                    alpha=alpha, A=A.data_ptr(), lda=A.stride(1), B=v.data_ptr(), ldb=v.stride(1))
    else:
        from scipy.linalg import blas as sclb
        trsm_fn = choose_fn(A.dtype, sclb.dtrsm, sclb.strsm, "TRSM")
        trsm_fn(alpha, A.numpy(), v.numpy(), side=0, lower=lower, trans_a=transpose, overwrite_b=1)
    return v


def trsm_fro_block_sizes(n, m, d, avail_mem, incore) -> Tuple[int, int]:
    """
    Needed memory (out-of-core):
     - n*d, m*d matrices to hold the inputs to fmm
     - n*m to hold the fmm output (also for in-core)
     - m*m to hold the triangular matrix
    """
    coef_nd, coef_md, coef_m = 0, 0, 0
    if not incore:
        coef_nd += 1
        coef_md += 1
        coef_m += m

    blk_n = select_dim_over_n(max_n=n, m=m, d=d, max_mem=avail_mem,
                              coef_nm=1, coef_nd=coef_nd, coef_md=coef_md, coef_n=0,
                              coef_m=coef_m, coef_d=0, rest=0)
    mem_needed = blk_n * m
    if not incore:
        mem_needed += (m + blk_n) * d
        mem_needed += m**2
    return blk_n, mem_needed


def kernel_trsm_fro_runner(proc_idx, queue, device_id):
    a: KernelTrsmArgs = queue.get()
    tri, m1, m2, kernel, lower, transpose, max_mem = a.tri, a.m1, a.m2, a.kernel, a.lower, a.transpose, a.max_mem
    avail_mem = max_mem / sizeof_dtype(m1.dtype)
    # We reduce the used memory drastically, since we must leave some space for the actual kernel
    # computation.
    avail_mem /= 2.0
    n, d = m1.shape
    m = m2.shape[0]
    dev = _dev_from_id(device_id)
    incore = _is_incore(dev, m1.device)
    blk_n, mem_needed = trsm_fro_block_sizes(n=m1.shape[0], m=m2.shape[0], d=m1.shape[1], avail_mem=avail_mem, incore=incore)
    # We want to allow kernels in float32 precision,
    # since not too interested in the accuracy of the `knm` kernel
    kernel_opt = dataclasses.replace(kernel.params, no_single_kernel=False)

    # Initialize extra buffers
    out_scalar = torch.zeros(1, dtype=m1.dtype, device=dev)
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_out = extract_fortran(flat_gpu, size=(m, blk_n), offset=flat_offset)  # F-contig since in-core TRSM requires it so
    flat_offset += np.prod(dev_out.shape)
    dev_m1, dev_m2, dev_tri = m1, m2, tri
    if not incore:
        dev_m1 = extract_same_stride(flat_gpu, size=(blk_n, d), other=m1, offset=flat_offset)
        flat_offset += np.prod(dev_m1.shape)
        dev_m2 = extract_same_stride(flat_gpu, size=(m, d), other=m2, offset=flat_offset)
        flat_offset += np.prod(dev_m2.shape)
        dev_tri = extract_same_stride(flat_gpu, size=(m, m), other=tri, offset=flat_offset)
        flat_offset += np.prod(dev_tri.shape)

    with ExitStack() as stack:
        s1, s2 = None, None
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev)
            s2 = tcd.Stream(dev)  # Only used for copying tri to device
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))

        if not incore:
            copy(m2, dev_m2, s=s1)
            copy(tri, dev_tri, s=s2)

        for i in range(0, n, blk_n):
            leni = min(blk_n, n - i)
            if incore:
                c_dev_m1 = dev_m1[i: i + leni, :]
            else:
                c_dev_m1 = copy(m1[i: i + leni, :], dev_m1[:leni, :], s=s1)
            c_dev_out = dev_out[:, :leni]
            kernel(dev_m2, c_dev_m1, out=c_dev_out, opt=kernel_opt)
            if not incore and s2 is not None:
                s2.synchronize()
            inplace_trsm(dev_tri, c_dev_out, 1.0, lower=lower, transpose=transpose)
            out_scalar.add_(c_dev_out.sum())
    return out_scalar.item()


def kernel_trsm_fro(tri: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor, kernel: Kernel,
                    lower: bool, transpose: bool, opt: Optional[BaseOptions] = None) -> float:
    if opt is None:
        opt = BaseOptions()
    if not check_same_device(tri, m1, m2):
        raise ValueError("All input matrices must be on the same device.")
    data_dev = m1.device
    comp_dev_type = 'cpu' if opt.use_cpu or not torch.cuda.is_available() else 'cuda'

    if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
        args = KernelTrsmArgs(
            m1=m1, m2=m2, tri=tri, kernel=kernel, max_mem=opt.max_cpu_mem,
            lower=lower, transpose=transpose)
        return _call_direct(kernel_trsm_fro_runner, (args, -1))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = KernelTrsmArgs(
            m1=m1, m2=m2, tri=tri, kernel=kernel, max_mem=single_gpu_info.usable_memory,
            lower=lower, transpose=transpose)
        return _call_direct(kernel_trsm_fro_runner, (args, data_dev.index))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        args = []  # Arguments passed to each subprocess
        block_sizes = calc_gpu_block_sizes(gpu_info, m1.shape[0])
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            args.append((KernelTrsmArgs(
                m1=m1.narrow(0, block_sizes[i], bwidth), m2=m2, tri=tri, kernel=kernel,
                max_mem=g.usable_memory, lower=lower, transpose=transpose), g.Id))
        outputs = _start_wait_processes(kernel_trsm_fro_runner, args)
        return sum(outputs)
    else:
        raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
