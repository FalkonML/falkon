from contextlib import ExitStack
from typing import Optional

import numpy as np
import torch
import torch.cuda as tcd
from falkon.kernels import GaussianKernel

from falkon.options import BaseOptions
from falkon.utils.tensor_helpers import (
    create_same_stride,
    extract_same_stride,
)
from falkon.utils.helpers import (
    calc_gpu_block_sizes,
    sizeof_dtype,
    select_dim_over_bnm,
)
from falkon.mmv_ops.utils import (
    _get_gpu_info,
    _call_direct,
    _start_wait_processes,
    _setup_opt,
    _check_contiguity,
    ensure_batch_dim,
)
from falkon.mmv_ops.fmmv_cuda import ArgsFmmv


def _extract_flat(flat_tn, size, other, offset):
    struct_tn = extract_same_stride(flat_tn, size=size, other=other, offset=offset)
    offset += np.prod(struct_tn.shape)
    return struct_tn, offset


def _is_incore(computation_device: torch.device, data_device: torch.device) -> bool:
    return computation_device.type == data_device.type


def mmv_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: torch.Tensor, vout: torch.Tensor,
                   kernel: GaussianKernel, b0: int, b1: int, b2: int, dev: torch.device):
    dt = m1.dtype
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    incore = _is_incore(dev, m1.device)
    B, N, D = m1.shape
    M = m2.shape[-2]
    T = v.shape[-1]
    b0, b1, b2 = min(b0, B), min(b1, N), min(b2, M)

    """ Initialize extra buffers """
    flat_offset = 0
    total_memory = b0 * b1 * b2
    if not incore:
        total_memory += (b0 * b1 * D) + (b0 * b2 * D) + (b0 * b2 * T) + (b0 * b1 * T)
    print(f"Before start: CUDA memory usage: {torch.cuda.max_memory_allocated(dev) / 2**20:.4f}MB")
    flat_dev_t = torch.empty(size=(total_memory,), dtype=dt, device=dev)
    print(f"Big alloc: CUDA memory usage: {torch.cuda.max_memory_allocated(dev) / 2**20:.4f}MB")
    dev_nm_temp, flat_offset = _extract_flat(flat_dev_t, size=(b0, b1, b2), other=m1,
                                             offset=flat_offset)
    if incore:
        dev_vout = vout
    else:
        dev_m1, flat_offset = _extract_flat(flat_dev_t, size=(b0, b1, D), other=m1,
                                            offset=flat_offset)
        dev_m2, flat_offset = _extract_flat(flat_dev_t, size=(b0, b2, D), other=m2,
                                            offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_dev_t, size=(b0, b2, T), other=v,
                                           offset=flat_offset)
        dev_vout, flat_offset = _extract_flat(flat_dev_t, size=(b0, b1, T), other=vout,
                                              offset=flat_offset)

    """ Run splitting along B, N, M """
    with ExitStack() as stack:
        if dev.type == 'cuda':
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(tcd.current_stream(dev)))
        for a in range(0, B, b0):
            lena = min(b0, B - a)
            for b in range(0, N, b1):
                lenb = min(b1, N - b)
                if incore:
                    c_dev_m1 = m1[a:a + lena, b:b + lenb, :]
                    c_dev_vout = dev_vout[a:a + lena, b:b + lenb]
                else:
                    # noinspection PyUnboundLocalVariable
                    c_dev_m1 = dev_m1[:lena, :lenb, :]
                    c_dev_m1.copy_(m1[a:a + lena, b:b + lenb, :], non_blocking=True)
                    c_dev_vout = dev_vout[:lena, :lenb]

                c_dev_vout.fill_(0.0)
                for c in range(0, M, b2):
                    lenc = min(b2, M - c)
                    c_dev_nm = dev_nm_temp[:lena, :lenb, :lenc]
                    if incore:
                        c_dev_m2 = m2[a:a + lena, c:c + lenc, :]
                        c_dev_v = v[a:a + lena, c:c + lenc, :]
                    else:
                        # noinspection PyUnboundLocalVariable
                        c_dev_m2 = dev_m2[:lena, :lenc, :]
                        c_dev_m2.copy_(m2[a:a + lena, c:c + lenc, :], non_blocking=True)
                        # noinspection PyUnboundLocalVariable
                        c_dev_v = dev_v[:lena, :lenc, :]
                        c_dev_v.copy_(v[a:a + lena, c:c + lenc, :], non_blocking=True)

                    # Compute kernel sub-matrix
                    kernel.compute(c_dev_m1, c_dev_m2, c_dev_nm)
                    # Multiply kernel sub-matrix by a vector: b*n*m @ b*n*t = b*n*t
                    c_dev_vout.baddbmm_(c_dev_nm, c_dev_v)
                # end iter over M
                if not incore:
                    c_host_vout = vout[a:a + lena, b:b + lenb]
                    c_host_vout.copy_(c_dev_vout, non_blocking=True)
            # end iter over N
        # end iter over B
    # exit context manager (device, stream)
    print(f"At end: CUDA memory usage: {torch.cuda.max_memory_allocated(dev) / 2**20:.4f}MB")


def mmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, out = a.X1, a.X2, a.v, a.out
    kernel: GaussianKernel = a.kernel
    max_mem = a.max_mem
    if device_id < 0:
        dev = torch.device('cpu')
    else:
        dev = torch.device('cuda:%d' % device_id)
    ooc_mul = 0 if _is_incore(dev, X1.device) else 1

    # Choose batch sizes
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    extra_mem = kernel.extra_mem()
    b, n, m = select_dim_over_bnm(
        max_b=X1.shape[0],
        max_n=X1.shape[-2],
        max_m=X2.shape[-2],
        d=X1.shape[-1],
        coef_bnd=1 * ooc_mul + extra_mem.get('nd', 0),
        coef_bmd=1 * ooc_mul + extra_mem.get('md', 0),
        coef_bnm=1 + extra_mem.get('nm', 0),
        coef_bn=v.shape[-1] * ooc_mul + extra_mem.get('bn', 0),
        coef_bm=v.shape[-1] * ooc_mul + extra_mem.get('bm', 0),
        rest=extra_mem.get('d', 0),
        max_mem=avail_mem
    )
    print(f"Start batch-mmv tracing. B={X1.shape[0]}, N={X1.shape[-2]}, D={X1.shape[-1]}, T={v.shape[-1]} -- b0 {b}, b1 {n}, b2 {m}")
    print(f"Available memory {avail_mem / 2**20:.4f}MB")


    # Run
    mmv_run_thread(X1, X2, v, out, kernel, b, n, m, dev)


def batch_fmmv_incore(X1: torch.Tensor, X2: torch.Tensor, v: torch.Tensor,
                      kernel, out: Optional[torch.Tensor] = None,
                      opt: Optional[BaseOptions] = None) -> torch.Tensor:
    """
    In-core batch kernel-vector multiplication.

    Batched kernel-vector multiplication may be faster than running multiple individual
    kernel-vector multiplications whenever each individual operation cannot saturate the available
    GPUs.
    Data matrices and the vector batch must be supplied in contiguous memory. This function will
    be more efficient with row-contiguous (C-order) data, since the data is split preferentially
    along the batch dimension.

    The dimensions of the input tensors follow a common naming convention: The data matrices
    must be of size `(b, n, d)` and `(b, m, d)` and the vector must be of size `(b, m, t)`.
    The output of this function will have size `(b, n, t)`.

    This function will work in-core, so if the data passed is on the CPU, all operations will run
    on the CPU. If the data passed is on the GPU, all operations will run on the GPU. All matrices
    must be on the same device.

    Parameters
    ----------
    X1 : (b, n, d) tensor
        Batched data matrix. CPU or CUDA.
    X2 : (b, m, d) tensor
        Batched data matrix. CPU or CUDA.
    v : (b, m, t) tensor
        Batched 'vector'. Commonly `t` is equal to 1, but it may also be greater than 1 so `v` is
        really a batched matrix. CPU or CUDA.
    kernel
        The falkon kernel used for calculating sub-kernel matrices. The kernel must have a
        `compute` function (currently only for the `GaussianKernel`).
    out : (b, n, t) tensor or None
        Optional tensor into which the output will be stored. If `None` is passed, the function will
        allocate the output tensor.
    opt
        Additional options passed to the function. Especially useful are the `max_cpu_mem` and
        `max_gpu_mem` keys.

    Returns
    -------
    out : (b, n, t) tensor
        The result of the kernel-vector multiplication. Will be on the same device as the input
        data.
    """
    comp_dev_type = 'cuda' if X1.device.type == 'cuda' else 'cpu'
    return _batch_fmmv(X1, X2, v, kernel, comp_dev_type, out, opt)


def batch_fmmv_ooc(X1: torch.Tensor, X2: torch.Tensor, v: torch.Tensor,
                   kernel, out: Optional[torch.Tensor] = None,
                   opt: Optional[BaseOptions] = None) -> torch.Tensor:
    """
    Out-of-core batch kernel-vector multiplication.

    Batched kernel-vector multiplication may be faster than running multiple individual
    kernel-vector multiplications whenever each individual operation cannot saturate the available
    GPUs.
    Data matrices and the vector batch must be supplied in contiguous memory. This function will
    be more efficient with row-contiguous (C-order) data, since the data is split preferentially
    along the batch dimension.

    The dimensions of the input tensors follow a common naming convention: The data matrices
    must be of size `(b, n, d)` and `(b, m, d)` and the vector must be of size `(b, m, t)`.
    The output of this function will have size `(b, n, t)`.

    This function will work out-of-core, so the input data must be on the CPU, and all computations
    will run on the GPU taking care of splitting the data so that it fits in memory. Passing
    GPU data to this function will result in an error. For the same reason, calling this function
    on a CPU-only installation of falkon will result in an error. In both these cases, the
    :func:`batch_fmmv_incore` function can be used to run batched kernel-vector multiplication.

    Parameters
    ----------
    X1 : (b, n, d) tensor
        Batched data matrix. CPU.
    X2 : (b, m, d) tensor
        Batched data matrix. CPU.
    v : (b, m, t) tensor
        Batched 'vector'. Commonly `t` is equal to 1, but it may also be greater than 1 so `v` is
        really a batched matrix. CPU.
    kernel
        The falkon kernel used for calculating sub-kernel matrices. The kernel must have a
        `compute` function (currently only for the `GaussianKernel`).
    out : (b, n, t) tensor or None
        Optional tensor into which the output will be stored. If `None` is passed, the function will
        allocate the output tensor.
    opt
        Additional options passed to the function. Especially useful are the `max_cpu_mem` and
        `max_gpu_mem` keys.

    Returns
    -------
    out : (b, n, t) tensor
        The result of the kernel-vector multiplication.
    """
    comp_dev_type = 'cpu' if X1.device.type == 'cuda' else 'cuda'
    return _batch_fmmv(X1, X2, v, kernel, comp_dev_type, out, opt)


def _batch_fmmv(X1: torch.Tensor,
                X2: torch.Tensor,
                v: torch.Tensor,
                kernel,
                comp_dev_type: str,
                out: Optional[torch.Tensor] = None,
                opt: Optional[BaseOptions] = None) -> torch.Tensor:
    """
    X1 : N x D
    X2 : M x D
    v  : M x T

    performs  fnc(X1*X2', X1, X2) * v   : N x T
    in blocks on multiple GPUs
    """
    opt = _setup_opt(opt)
    _check_contiguity((X1, 'X1'), (X2, 'X2'), (v, 'v'), (out, 'out'))
    X1, X2, v, out = ensure_batch_dim(X1, X2, v, out)
    data_dev = X1.device

    B, N, D = X1.shape
    T = v.shape[-1]
    # Create output matrix
    if out is None:
        out = create_same_stride((B, N, T), X1, v.dtype, device=data_dev,
                                 pin_memory=data_dev.type != 'cuda')
    out.fill_(0.0)

    if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel, max_mem=opt.max_cpu_mem)
        _call_direct(mmv_run_starter, (args, -1))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel,
                        max_mem=single_gpu_info.usable_ram)
        _call_direct(mmv_run_starter, (args, data_dev.index))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        args = []  # Arguments passed to each subprocess
        if B == 1:
            block_sizes = calc_gpu_block_sizes(gpu_info, N)
            for i, g in enumerate(gpu_info):
                bwidth = block_sizes[i + 1] - block_sizes[i]
                if bwidth <= 0:
                    continue
                args.append((ArgsFmmv(
                    X1=X1.narrow(1, block_sizes[i], bwidth),
                    X2=X2, v=v,
                    out=out.narrow(1, block_sizes[i], bwidth),
                    kernel=kernel, max_mem=g.usable_ram), g.Id))
        else:
            block_sizes = calc_gpu_block_sizes(gpu_info, B)
            for i, g in enumerate(gpu_info):
                bwidth = block_sizes[i + 1] - block_sizes[i]
                if bwidth <= 0:
                    continue
                args.append((ArgsFmmv(
                    X1=X1.narrow(0, block_sizes[i], bwidth),
                    X2=X2.narrow(0, block_sizes[i], bwidth),
                    v=v.narrow(0, block_sizes[i], bwidth),
                    out=out.narrow(0, block_sizes[i], bwidth),
                    kernel=kernel, max_mem=g.usable_ram), g.Id))
        _start_wait_processes(mmv_run_starter, args)
    else:
        raise RuntimeError("Requested CPU computations with CUDA data. "
                           "This should not happen, please file a bug.")
    return out
