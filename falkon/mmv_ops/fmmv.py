from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Sequence

import numpy as np
import torch
import torch.cuda as tcd
import torch.cuda.comm

import falkon
from falkon.mmv_ops.utils import *
from falkon.options import BaseOptions
from falkon.sparse import SparseTensor
from falkon.utils.device_copy import copy
from falkon.utils.helpers import (
    calc_gpu_block_sizes,
    sizeof_dtype,
    select_dim_over_nm_v2,
    select_dim_over_n,
)
from falkon.utils.tensor_helpers import (
    create_same_stride,
    extract_fortran,
)


@dataclass(frozen=True)
class ArgsFmmv:
    X1: Union[torch.Tensor, SparseTensor]
    X2: Union[torch.Tensor, SparseTensor]
    v: torch.Tensor
    out: torch.Tensor
    kernel: 'falkon.kernels.Kernel'
    max_mem: float
    w: torch.Tensor = None
    differentiable: bool = False


def _sparse_mmv_blk_sizes(n, d, m, t, avail_mem, extra_mem, incore: bool, m1_density: float,
                          m2_density: float):
    # Memory needs:
    # chunk of x1: n + 2*d*n*density
    # chunk of x2: d + 2*d*m*density (because is transposed)
    # chunk of kernel: n + 2*n*m*density (assume density=1)
    # dense matrices: kernel (n*m) + output (n*t) + vector (m*t)
    coef_nm = 3  # Both dense and sparse spspmm output must be allocated.
    coef_n, coef_m, coef_rest = 0, 0, 0
    if not incore:
        coef_n += 2 + 2 * d * m1_density + t
        coef_m += 2 * d * m2_density + t
        coef_rest = d
    blk_n, blk_m = select_dim_over_nm_v2(
        max_n=n, max_m=m, max_mem=avail_mem,
        coef_nm=coef_nm + extra_mem.get('nm', 0),
        coef_n=coef_n + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d,
        coef_m=coef_m + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d,
        rest=coef_rest + extra_mem.get('d', 0))
    # here mem_needed is only the dense blocks!
    mem_needed = blk_m * blk_n
    if not incore:
        mem_needed += (blk_n + blk_m) * t
    return blk_n, blk_m, mem_needed


def _dense_mmv_blk_sizes(n: int, d: int, m: int, t: int, avail_mem: float,
                         extra_mem: Dict[str, float], m1_ic: bool, m2_ic: bool, v_ic: bool,
                         out_ic: bool) -> Tuple[int, int, int]:
    coef_nm = 1  # for the kernel block
    coef_n = d if not m1_ic else 0  # For m1
    coef_m = d if not m2_ic else 0  # For m2
    coef_n = coef_n + t if not out_ic else coef_n  # For output vector
    coef_m = coef_m + t if not v_ic else coef_m  # For v
    blk_n, blk_m = select_dim_over_nm_v2(
        max_n=n, max_m=m, max_mem=avail_mem,
        coef_nm=coef_nm + extra_mem.get('nm', 0),
        coef_n=coef_n + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d,
        coef_m=coef_m + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d,
        rest=extra_mem.get('d', 0))
    mem_needed = blk_m * blk_n  # for the kernel block
    mem_needed += blk_n * coef_n  # for m1 and output vector
    mem_needed += blk_m * coef_m  # for m2 and v
    return blk_n, blk_m, mem_needed


def mmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, out = a.X1, a.X2, a.v, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    differentiable = a.differentiable
    dev = _dev_from_id(device_id)
    incore = _is_incore(dev, X1.device)
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)

    # Choose batch sizes
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    extra_mem = kernel.extra_mem()
    n, d = X1.shape
    m, t = v.shape
    if differentiable:
        diff_coef_nm = 4
        assert not is_sparse, "Sparse + differentiable mmvs are not supported"
        blk_n, blk_m = select_dim_over_nm_v2(
            max_n=n, max_m=m, max_mem=avail_mem,
            coef_nm=diff_coef_nm + extra_mem.get('nm', 0),
            coef_n=2 * (d + t + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d),
            coef_m=2 * (d + t + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d),
            rest=extra_mem.get('d', 0))
        return mmv_diff_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, dev)
    if is_sparse:
        blk_n, blk_m, mem_needed = _sparse_mmv_blk_sizes(
            n=n, m=m, d=d, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore,
            m1_density=X1.density, m2_density=X2.density)
        return sparse_mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev)
    else:
        m1_ic, m2_ic, v_ic, out_ic = (_is_incore(dev, X1.device), _is_incore(dev, X2.device),
                                      _is_incore(dev, v.device), _is_incore(dev, out.device))
        blk_n, blk_m, mem_needed = _dense_mmv_blk_sizes(
            n=n, m=m, d=d, t=t, avail_mem=avail_mem, extra_mem=extra_mem, m1_ic=m1_ic, m2_ic=m2_ic,
            v_ic=v_ic, out_ic=out_ic)
        return mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev)


def sparse_mmv_run_thread(m1: SparseTensor, m2: SparseTensor, v: torch.Tensor,
                          out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int,
                          blk_m: int, mem_needed: int, dev: torch.device):
    """Inner loop to compute (part of) a kernel-vector product for sparse input matrices.

    Parameters
    ----------
    m1
        Left input tensor for computing the kernel
    m2
        Right input tensor for computing the kernel
    v
        Dense vector to be multiplied by the kernel matrix
    out
        Dense output vector which should store the result of the kernel vector product on exit
        from this function.
    kernel
        Kernel object, used for computing the kernel. This must implement the
        :meth:`falkon.kernels.kernel.Kernel.compute_sparse` method.
    blk_n
        Block size for the first axis of `m1`
    blk_m
        Block size for the first ais of `m2`
    mem_needed
        Memory needed for pre-allocations
    dev
        Device on which to run the calculations

    Returns
    -------
    out : torch.Tensor
        The kernel matrix. Should use the same underlying storage as the parameter `out`.
    """
    incore = _is_incore(dev, m1.device)
    N, D = m1.shape
    M, T = v.shape

    """ Initialize extra buffers """
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    # ker_gpu must be fortran-ordered due to cusparse csr2dense function (TODO: only on CUDA)
    ker_gpu = extract_fortran(flat_gpu, size=(blk_n, blk_m), offset=flat_offset)
    flat_offset += np.prod(ker_gpu.shape)
    dev_v, dev_out = None, None
    if not incore:
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out,
                                             offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(blk_m, T), other=v, offset=flat_offset)

    with ExitStack() as stack:
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev)
            s2 = tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))

        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)

            c_m1 = m1.narrow_rows(i, leni)
            if incore:  # Note that CUDA-incore is not allowed to happen (so this is CPU->CPU)
                c_dev_out = out[i: i + leni]
                c_dev_m1 = c_m1
            else:  # CPU -> CUDA
                c_dev_out = dev_out[:leni]
                c_dev_m1 = m1.index_to_int().to(device=dev, non_blocking=True)
            c_dev_out.fill_(0.0)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)

                c_m2 = m2.narrow_rows(j, lenj)
                if incore:  # CPU -> CPU
                    c_dev_m2 = c_m2.transpose_csc()
                    c_dev_v = v[j: j + lenj]
                else:  # CPU -> CUDA
                    c_dev_m2 = SparseTensor.from_scipy(
                        c_m2.transpose_csc().to_scipy().tocsr(copy=False)) \
                        .index_to_int() \
                        .to(device=dev, non_blocking=True)
                    c_dev_v = copy(v[j: j + lenj], dev_v[:lenj], s=s2)
                c_dev_ker = ker_gpu[:leni, :lenj].fill_(0.0)

                c_dev_ker = kernel.compute_sparse(c_dev_m1, c_dev_m2, c_dev_ker,
                                                  X1_csr=c_m1, X2_csr=c_m2)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)

                # Copy output to host
                if not incore:
                    copy(c_dev_out, out[i: i + leni], s=s1)


def mmv_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: Optional[torch.Tensor],
                   out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int, blk_m: int,
                   mem_needed: int, dev: torch.device):
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    m1_ic, m2_ic, v_ic, out_ic = (_is_incore(dev, m1.device), _is_incore(dev, m2.device),
                                  _is_incore(dev, v.device), _is_incore(dev, out.device))
    incore = all((m1_ic, m2_ic, v_ic, out_ic))
    N, D = m1.shape
    M, T = v.shape

    # Initialize extra buffers
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_ker, flat_offset = _extract_flat(flat_gpu, size=(blk_n, blk_m), other=out,
                                         offset=flat_offset)
    if m1_ic:
        dev_m1 = None
    else:
        dev_m1, flat_offset = _extract_flat(flat_gpu, size=(blk_n, D), other=m1, offset=flat_offset)
    if m2_ic:
        dev_m2 = None
    else:
        dev_m2, flat_offset = _extract_flat(flat_gpu, size=(blk_m, D), other=m2, offset=flat_offset)
    if v_ic:
        dev_v = None
    else:
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(blk_m, T), other=v, offset=flat_offset)
    if out_ic:
        dev_out = None
    else:
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out,
                                             offset=flat_offset)

    with ExitStack() as stack:
        s1, s2 = None, None
        if dev.type == 'cuda':
            s1, s2 = tcd.current_stream(dev), tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            if m1_ic:
                c_dev_m1 = m1[i: i + leni, :]
            else:
                c_dev_m1 = copy(m1[i: i + leni, :], dev_m1[:leni, :], s=s1)
            if out_ic:
                c_dev_out = out[i: i + leni]
            else:
                c_dev_out = dev_out[:leni]
            c_dev_out.fill_(0.0)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                if m2_ic:
                    c_dev_m2 = m2[j: j + lenj, :]
                else:
                    c_dev_m2 = copy(m2[j: j + lenj, :], dev_m2[:lenj, :], s=s1)
                if v_ic:
                    c_dev_v = v[j: j + lenj, :]
                else:
                    c_dev_v = copy(v[j: j + lenj, :], dev_v[:lenj, :], s=s2)
                c_dev_ker = dev_ker[:leni, :lenj].fill_(0.0)

                c_dev_ker = kernel.compute(c_dev_m1, c_dev_m2, c_dev_ker)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)
                if not incore:
                    s1.synchronize()  # sync necessary to avoid s2 overwriting dev_v/dev_w
            # end iter over M
            if not out_ic:
                copy(c_dev_out, out[i: i + leni], s=s1)
        # end iter over N
    # exit context manager (device, stream)


def mmv_diff_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: Optional[torch.Tensor],
                        out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int, blk_m: int,
                        dev: torch.device):
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    incore = _is_incore(dev, m1.device)
    N, D = m1.shape
    M, T = v.shape

    inputs = [m1, m2, v] + list(kernel.diff_params.values())
    grads = []
    for ipt in inputs:
        if ipt.requires_grad:
            grads.append(torch.zeros_like(ipt))
        else:
            grads.append(None)
    inputs_need_grad, input_idxs = zip(
        *[(ipt, idx) for idx, ipt in enumerate(inputs) if ipt.requires_grad])

    with ExitStack() as stack:
        if dev.type == 'cuda':
            s1, s2 = tcd.current_stream(dev), tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            c_dev_m1 = m1[i: i + leni, :].to(dev, non_blocking=True, copy=False)
            c_dev_m1_g = None if grads[0] is None else grads[0][i: i + leni, :].to(dev, non_blocking=True, copy=False)
            c_dev_out = out[i: i + leni, :].to(dev, non_blocking=True, copy=False)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                c_dev_m2 = m2[j: j + lenj, :].to(dev, non_blocking=True, copy=False)
                c_dev_m2_g = None if grads[1] is None else grads[1][j: j + lenj, :].to(dev, non_blocking=True, copy=False)
                with ExitStack() as stack_s2:
                    if not incore:
                        stack_s2.enter_context(tcd.stream(s2))
                    c_dev_v = v[j: j + lenj, :].to(dev, non_blocking=True, copy=False)
                    c_dev_v_g = None if grads[2] is None else grads[2][j: j + lenj, :].to(dev, non_blocking=True, copy=False)
                c_dev_ker = kernel.compute_diff(c_dev_m1, c_dev_m2)
                if not incore:
                    s2.synchronize()
                c_dev_mmv = c_dev_ker @ c_dev_v
                c_inputs = [c_dev_m1, c_dev_m2, c_dev_v] + list(kernel.diff_params.values())
                c_dev_grads_old = [c_dev_m1_g, c_dev_m2_g, c_dev_v_g] + grads[3:]
                c_dev_grads = torch.autograd.grad(
                    c_dev_mmv, [c_inputs[idx] for idx in input_idxs], grad_outputs=c_dev_out)
                for c_grad, c_idx in zip(c_dev_grads, input_idxs):
                    c_dev_grads_old[c_idx].add_(c_grad)
                if grads[1] is not None:
                    grads[1][j: j + lenj, :] = c_dev_m2_g.to(grads[1].device, non_blocking=True, copy=False)
                if grads[2] is not None:
                    grads[2][j: j + lenj, :] = c_dev_v_g.to(grads[2].device, non_blocking=True, copy=False)
                if not incore:
                    s1.synchronize()  # sync necessary to avoid s2 overwriting dev_v/dev_w
            # end iter over M
            if grads[0] is not None:
                grads[0][i: i + leni, :] = c_dev_m1_g.to(grads[0].device, non_blocking=True, copy=False)
        # end iter over N
    # exit context manager (device, stream)
    return grads


def _sparse_dmmv_blk_sizes(n, d, m, t, avail_mem, extra_mem: dict, incore: bool,
                           dev_out_exists: bool, m1_density: float, m2_density: float):
    # Memory needs:
    # chunk of X1              : n + 2*d*n*density
    # full X2                  : d + 2*d*m*density (it's transposed)
    # sparse output (internal) : n + 2*n*m*density (density assumed = 1)
    # And the dense matrices: kernel(m*n), w(n*t), v(m*t), output(m*t)
    coef_nm = 3  # Both dense and sparse spspmm output must be allocated.
    coef_nd, coef_md, coef_nt, coef_mt = 0, 0, 0, 0
    coef_nt += 1  # for dev_w allocation
    if not incore:
        coef_nd += 2 * m1_density  # for x1-chunk
        coef_md += 2 * m2_density  # for x2
        coef_mt += 1  # for v
        if not dev_out_exists:
            coef_mt += 1  # for output
    blk_n = select_dim_over_n(
        max_n=n, m=m, d=d, max_mem=avail_mem,
        coef_nm=coef_nm + extra_mem.get('nm', 0),
        coef_nd=coef_nd + extra_mem.get('nd', 0),
        coef_md=coef_md + extra_mem.get('md', 0),
        coef_n=coef_nt * t + 2 + extra_mem.get('n', 0) + t * extra_mem.get('nt', 0),
        coef_m=coef_mt * t + extra_mem.get('m', 0) + t * extra_mem.get('mt', 0),
        coef_d=1 + extra_mem.get('d', 0), rest=0)

    mem_needed = blk_n * m
    mem_needed += blk_n * t  # dev_w
    if not incore:
        mem_needed += m * t
        if not dev_out_exists:
            mem_needed += m * t

    return blk_n, mem_needed


def _dense_dmmv_blk_sizes(n, d, m, t, avail_mem: float, extra_mem: dict,
                          m1_ic: bool, m2_ic: bool, v_ic: bool, out_ic: bool) -> Tuple[int, int]:
    coef_nd, coef_md, coef_mt = 0, 0, 0
    coef_nm = 1  # for kernel block
    coef_nt = 1  # for dev_w allocation
    if not m1_ic:
        coef_nd += 1  # x1
    if not m2_ic:
        coef_md += 1  # x2
    if not v_ic:
        coef_mt += 1  # v
    if not out_ic:
        coef_mt += 1  # output
    blk_n = select_dim_over_n(
        max_n=n, m=m, d=d, max_mem=avail_mem,
        coef_nm=coef_nm + extra_mem.get('nm', 0),
        coef_nd=coef_nd + extra_mem.get('nd', 0),
        coef_md=coef_md + extra_mem.get('md', 0),
        coef_n=coef_nt * t + extra_mem.get('n', 0) + t * extra_mem.get('nt', 0),
        coef_m=coef_mt * t + extra_mem.get('m', 0) + t * extra_mem.get('mt', 0),
        coef_d=extra_mem.get('d', 0), rest=0)
    mem_needed = blk_n * m
    mem_needed += blk_n * t * coef_nt
    mem_needed += blk_n * d * coef_nd
    mem_needed += m * d * coef_md
    mem_needed += m * t * coef_mt
    return blk_n, mem_needed


def dmmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, w, out = a.X1, a.X2, a.v, a.w, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    assert not a.differentiable, "D-MMV not implemented for differentiable outputs"
    dev = _dev_from_id(device_id)
    incore = _is_incore(dev, X1.device)
    dev_out_exists = out.device == dev  # out has already been allocated on the computation device
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)

    # Choose batch sizes
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    extra_mem = kernel.extra_mem()
    n, d = X1.shape
    m, t = v.shape

    if is_sparse:
        blk_n, mem_needed = _sparse_dmmv_blk_sizes(
            n=n, d=d, m=m, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore,
            dev_out_exists=dev_out_exists, m1_density=X1.density, m2_density=X2.density)
        sparse_dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev)
    else:
        m1_ic, m2_ic, v_ic, out_ic = (_is_incore(dev, X1.device), _is_incore(dev, X2.device),
                                      _is_incore(dev, v.device), _is_incore(dev, out.device))
        blk_n, mem_needed = _dense_dmmv_blk_sizes(
            n=n, d=d, m=m, t=t, avail_mem=avail_mem, extra_mem=extra_mem,
            m1_ic=m1_ic, m2_ic=m2_ic, v_ic=v_ic, out_ic=out_ic)
        dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev)


def sparse_dmmv_run_thread(m1: SparseTensor, m2: SparseTensor, v: torch.Tensor,
                           w: Optional[torch.Tensor], out: torch.Tensor,
                           kernel: 'falkon.kernels.Kernel',
                           blk_n: int, mem_needed: int, dev: torch.device):
    incore = _is_incore(dev, m1.device)
    dev_out_exists = out.device == dev  # out has already been allocated on the computation device
    N, D = m1.shape
    M, T = v.shape

    """ Initialize extra buffers """
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    # ker_gpu must be fortran-ordered due to cusparse csr2dense function (TODO: only on CUDA)
    ker_gpu = extract_fortran(flat_gpu, size=(blk_n, M), offset=flat_offset)
    flat_offset += np.prod(ker_gpu.shape)
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w,
                                       offset=flat_offset)
    dev_out, dev_v, dev_m2 = out, v, m2
    if not incore:
        if not dev_out_exists:
            dev_out, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=out,
                                                 offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=v, offset=flat_offset)
    dev_out.fill_(0.0)

    with ExitStack() as stack:
        s1 = None
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        if not incore:  # Note that CUDA-incore is not allowed to happen (CPU->CUDA)
            copy(v, dev_v, s=s1)
            dev_m2 = SparseTensor.from_scipy(
                m2.transpose_csc().to_scipy().tocsr(copy=False)) \
                .index_to_int() \
                .to(device=dev)
        else:
            dev_m2 = m2.transpose_csc()

        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)

            c_m1 = m1.narrow_rows(i, leni)
            if incore:  # Note that CUDA-incore is not allowed to happen (so this is CPU->CPU)
                c_dev_m1 = c_m1
            else:  # CPU -> CUDA
                c_dev_m1 = c_m1.index_to_int().to(device=dev, non_blocking=True)
            if w is None:
                c_dev_w = dev_w[:leni, :].fill_(0.0)
            else:
                c_dev_w = copy(w[i: i + leni, :], dev_w[:leni, :], s=s1)

            c_dev_ker = ker_gpu[:leni].fill_(0.0)
            c_dev_ker = kernel.compute_sparse(c_dev_m1, dev_m2, c_dev_ker,
                                              X1_csr=c_m1, X2_csr=m2)

            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
        if not incore and not dev_out_exists:
            copy(dev_out, out, s=s1)


def dmmv_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: torch.Tensor,
                    w: Optional[torch.Tensor], out: torch.Tensor,
                    kernel: 'falkon.kernels.Kernel', blk_n: int, mem_needed: int,
                    dev: torch.device):
    # k(x2, x1) @ (k(x1, x2) @ v + w)
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    m1_ic, m2_ic, v_ic, out_ic = (_is_incore(dev, m1.device), _is_incore(dev, m2.device),
                                  _is_incore(dev, v.device), _is_incore(dev, out.device))
    N, D = m1.shape
    M, T = v.shape

    # Initialize extra buffers
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_ker, flat_offset = _extract_flat(flat_gpu, size=(blk_n, M), other=out, offset=flat_offset)
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w,
                                       offset=flat_offset)
    if m1_ic:
        dev_m1 = None
    else:
        dev_m1, flat_offset = _extract_flat(flat_gpu, size=(blk_n, D), other=m1, offset=flat_offset)
    if m2_ic:
        dev_m2 = m2
    else:
        dev_m2, flat_offset = _extract_flat(flat_gpu, size=(M, D), other=m2, offset=flat_offset)
    if v_ic:
        dev_v = v
    else:
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=v, offset=flat_offset)
    if out_ic:
        dev_out = out
    else:
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=out, offset=flat_offset)

    with ExitStack() as stack:
        s1, s2 = None, None
        if dev.type == 'cuda':
            s1, s2 = tcd.current_stream(dev), tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        dev_out.fill_(0.0)
        if not m2_ic:
            copy(m2, dev_m2, s=s1)
        if not v_ic:
            copy(v, dev_v, s=s2)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            if m1_ic:
                c_dev_m1 = m1[i: i + leni, :]
            else:
                c_dev_m1 = copy(m1[i: i + leni, :], dev_m1[:leni, :], s=s1)
            if w is not None:
                c_dev_w = copy(w[i: i + leni, :], dev_w[:leni, :], s=s2)
            else:
                c_dev_w = dev_w[:leni, :].fill_(0.0)

            c_dev_ker = dev_ker[:leni, :].fill_(0.0)
            c_dev_ker = kernel.compute(c_dev_m1, dev_m2, c_dev_ker)
            if dev.type == 'cuda':
                s2.synchronize()
            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
            if dev.type == 'cuda':
                s1.synchronize()  # sync necessary to avoid s2 overwriting dev_v/dev_w

        if not out_ic:
            copy(dev_out, out, s=s1)


# noinspection PyMethodOverriding
class KernelMmvFnFull(torch.autograd.Function):
    @staticmethod
    def run_cpu_cpu(X1: Union[torch.Tensor, SparseTensor], X2: Union[torch.Tensor, SparseTensor],
                    v: torch.Tensor, out: torch.Tensor, kernel, options, diff):
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel, max_mem=options.max_cpu_mem,
                        differentiable=diff)
        return _call_direct(mmv_run_starter, (args, -1))

    @staticmethod
    def run_cpu_gpu(X1: Union[torch.Tensor, SparseTensor], X2: Union[torch.Tensor, SparseTensor],
                    v: torch.Tensor, out: torch.Tensor, kernel, options, diff):
        is_sparse = isinstance(X1, SparseTensor)
        gpu_info = _get_gpu_info(options, slack=0.9)
        args = []  # Arguments passed to each subprocess
        block_sizes = calc_gpu_block_sizes(gpu_info, X1.shape[0])
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            if is_sparse:
                X1_block = X1.narrow_rows(block_sizes[i], bwidth)
            else:
                X1_block = X1.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFmmv(
                X1=X1_block,
                X2=X2, v=v,
                out=out.narrow(0, block_sizes[i], bwidth),
                kernel=kernel, max_mem=g.usable_memory,
                differentiable=diff), g.Id))
        outputs = _start_wait_processes(mmv_run_starter, args)
        if not diff:
            return outputs
        if len(gpu_info) == 1:
            return outputs[0]
        # Need to rejoin the gradient with respect to X1
        fin_outputs = []
        for i in range(len(outputs[0])):
            if outputs[0][i] is None:
                fin_outputs.append(None)
            elif i == 0:
                fin_outputs.append(torch.cat([o[i] for o in outputs], dim=0))
            else:
                fin_outputs.append(sum(o[i] for o in outputs))
        return fin_outputs

    @staticmethod
    def run_gpu_gpu(X1: Union[torch.Tensor, SparseTensor], X2: Union[torch.Tensor, SparseTensor],
                    v: torch.Tensor, out: torch.Tensor, kernel, options, diff):
        if isinstance(X1, SparseTensor):
            raise NotImplementedError("In-core, sparse fmmv not implemented. "
                                      "Use the out-of-core version instead.")
        data_dev = X1.device
        gpu_info = _get_gpu_info(options, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel,
                        max_mem=single_gpu_info.usable_memory, differentiable=diff)
        return _call_direct(mmv_run_starter, (args, data_dev.index))

    @staticmethod
    def forward(ctx, kernel: 'falkon.kernels.Kernel', opt: Optional[BaseOptions],
                out: Optional[torch.Tensor], X1: Union[torch.Tensor, SparseTensor],
                X2: Union[torch.Tensor, SparseTensor], v: torch.Tensor,
                *kernel_params):
        is_sparse = isinstance(X1, SparseTensor)
        if is_sparse:
            differentiable = False
        else:
            _check_contiguity((X1, 'X1'), (X2, 'X2'), (v, 'v'), (out, 'out'))
            differentiable = any([t.requires_grad for t in [X1, X2, v] + [*kernel_params]])
        data_devs = (X1.device, X2.device, v.device)
        comp_dev_type = 'cpu' if opt.use_cpu or not torch.cuda.is_available() else 'cuda'
        N, D = X1.shape
        T = v.shape[-1]
        # Create output matrix
        out = create_output_mat(out, data_devs, is_sparse, shape=(N, T), dtype=v.dtype,
                                comp_dev_type=comp_dev_type, other_mat=X1)

        with torch.inference_mode():
            if not isinstance(X1, SparseTensor) and X1.requires_grad:
                X1d = X1.detach()
            else:
                X1d = X1
            if not isinstance(X2, SparseTensor) and X2.requires_grad:
                X2d = X2.detach()
            else:
                X2d = X2
            vd = v.detach()
            kerneld = kernel.detach()
            if comp_dev_type == 'cpu' and all([ddev.type == 'cpu' for ddev in data_devs]):
                KernelMmvFnFull.run_cpu_cpu(X1d, X2d, vd, out, kerneld, opt, False)
            elif comp_dev_type == 'cuda' and all([ddev.type == 'cuda' for ddev in data_devs]):
                KernelMmvFnFull.run_gpu_gpu(X1d, X2d, vd, out, kerneld, opt, False)
            elif comp_dev_type == 'cuda':
                KernelMmvFnFull.run_cpu_gpu(X1d, X2d, vd, out, kerneld, opt, False)
            else:
                raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")

        if not differentiable:
            ctx.mark_non_differentiable(out)
        else:
            ctx.save_for_backward(X1, X2, v, *kernel_params)
            ctx.kernel = kernel
            ctx.opt = opt
        return out

    @staticmethod
    def backward(ctx, outputs):
        X1, X2, v, *kernel_params = ctx.saved_tensors

        data_dev = X1.device
        comp_dev_type = 'cpu' if ctx.opt.use_cpu or not torch.cuda.is_available() else 'cuda'

        # We must rerun MM in differentiable mode this time.
        with torch.autograd.enable_grad():
            if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
                grads = KernelMmvFnFull.run_cpu_cpu(X1, X2, v, outputs, ctx.kernel, ctx.opt, True)
            elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
                grads = KernelMmvFnFull.run_gpu_gpu(X1, X2, v, outputs, ctx.kernel, ctx.opt, True)
            elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
                grads = KernelMmvFnFull.run_cpu_gpu(X1, X2, v, outputs, ctx.kernel, ctx.opt, True)
            else:
                raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
            return tuple([None, None, None] + grads)


def fmmv(X1: Union[torch.Tensor, SparseTensor],
         X2: Union[torch.Tensor, SparseTensor],
         v: torch.Tensor,
         kernel: 'falkon.kernels.Kernel',
         out: Optional[torch.Tensor] = None,
         opt: Optional[BaseOptions] = None):
    return KernelMmvFnFull.apply(kernel, opt, out, X1, X2, v, *kernel.diff_params.values())


def fdmmv(X1: Union[torch.Tensor, SparseTensor], X2: Union[torch.Tensor, SparseTensor],
          v: torch.Tensor, w: Optional[torch.Tensor],
          kernel: 'falkon.kernels.Kernel', out: Optional[torch.Tensor] = None,
          differentiable: bool = False,
          opt: Optional[BaseOptions] = None) -> torch.Tensor:
    r"""Double kernel-vector product

    Computes kernel :math:`K = k(X_1, X_2)` and then the double kernel-vector
    product :math:`K^{\top} (K v + w)`.

    Parameters
    ----------
    X1
        :math:`n \times d` input matrix.
    X2
        :math:`m \times d` input matrix.
    v
        :math:`m \times t` vector (to be multiplied by the kernel)
    w
        :math:`n \times t` vector to be added to the first k-v product
    kernel
        Kernel object responsible for computing kernel blocks
    out
        Optional output matrix of size :math:`m \times t`. If specified, the output will be
        stored in it, otherwise a new output matrix will be allocated
    differentiable
        Whether the inputs are intended to be differentiated with. Currently setting this
        to ``True`` results in a :code:`NotImplementedError`.
    opt
        Options to be used for this operation

    Returns
    -------
    out
        Output of the double kernel-vector product. Will use the same storage as the `out`
        parameter if it was specified
    """
    if differentiable:
        raise NotImplementedError(
            "Manual D-MMV is not implemented for differentiable inputs. "
            "Try using the KeOps version instead.")
    is_sparse = isinstance(X1, SparseTensor)
    if not is_sparse:
        _check_contiguity((X1, 'X1'), (X2, 'X2'), (v, 'v'), (out, 'out'))
    data_devs = [X1.device, X2.device, v.device]
    if w is not None:
        data_devs.append(w.device)
    comp_dev_type = 'cpu' if opt.use_cpu or not torch.cuda.is_available() else 'cuda'

    N, D = X1.shape[-2:]
    M, T = v.shape[-2:]
    # Create output matrix
    out = create_output_mat(out, data_devs, is_sparse, shape=(M, T), dtype=v.dtype,
                            comp_dev_type=comp_dev_type, other_mat=X1)

    if comp_dev_type == 'cpu' and all([ddev.type == 'cpu' for ddev in data_devs]):
        args = ArgsFmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel, max_mem=opt.max_cpu_mem)
        _call_direct(dmmv_run_starter, (args, -1))
    elif comp_dev_type == 'cuda' and all([ddev.type == 'cuda' for ddev in data_devs]):
        if is_sparse:
            raise NotImplementedError("In-core, sparse fdmmv not implemented. "
                                      "Use the out-of-core version instead.")
        gpu_info = _get_gpu_info(opt, slack=0.9)
        data_dev = data_devs[0]
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel,
                        max_mem=single_gpu_info.usable_memory)
        _call_direct(dmmv_run_starter, (args, data_dev.index))
    elif comp_dev_type == 'cuda':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        args = []  # Arguments passed to each subprocess
        wrlk = []  # Outputs for each subprocess
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            if len(gpu_info) == 1 and out.device.index == gpu_info[i].Id:
                cur_out_gpu = out
            else:
                cur_out_gpu = create_same_stride((M, T), out, out.dtype, f'cuda:{gpu_info[i].Id}')
            gpu_info[i].usable_memory -= M * T * sizeof_dtype(X1.dtype)
            wrlk.append(cur_out_gpu)
            if is_sparse:
                X1_block = X1.narrow_rows(block_sizes[i], bwidth)
            else:
                X1_block = X1.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFmmv(
                X1=X1_block,
                X2=X2, v=v,
                w=w.narrow(0, block_sizes[i], bwidth) if w is not None else None,
                out=cur_out_gpu,
                kernel=kernel, max_mem=g.usable_memory), g.Id))
        _start_wait_processes(dmmv_run_starter, args)
        if len(wrlk) > 1:  # Sum up all subprocess outputs and copy to `out` on host.
            # noinspection PyTypeChecker
            fastest_device: int = np.argmax([d.speed for d in gpu_info])
            copy(torch.cuda.comm.reduce_add(wrlk, destination=gpu_info[fastest_device].Id), out)
        else:
            if wrlk[0].data_ptr() != out.data_ptr():
                copy(wrlk[0], out)
    else:
        raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
    return out
