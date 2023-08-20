from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.cuda as tcd
import torch.cuda.comm

import falkon
from falkon.mmv_ops.utils import (
    _dev_from_id,
    _is_incore,
    _extract_flat,
    _call_direct,
    _get_gpu_info,
    _start_wait_processes,
    _check_contiguity,
    create_output_mat,
)
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
    kernel: "falkon.kernels.Kernel"
    max_mem: float
    w: torch.Tensor = None
    differentiable: bool = False


def _init_two_streams(
    stack: ExitStack, dev: torch.device, tid: int
) -> Tuple[Optional[tcd.Stream], Optional[tcd.Stream]]:
    """
    Initialize two CUDA streams (if device is a GPU). If the thread ID is -1, we are initializing
    in the main thread so s1 will be the `current_stream`. Otherwise, it will not be a newly created
    stream. s2 is always a new stream.
    """
    s1, s2 = None, None
    if dev.type == "cuda":
        s1 = tcd.current_stream(dev) if tid == -1 else tcd.Stream(dev)
        s2 = tcd.Stream(dev)
        stack.enter_context(tcd.device(dev))
        stack.enter_context(tcd.stream(s1))
    return s1, s2


def _mmv_blk_sizes(
    n: int,
    d: int,
    m: int,
    t: int,
    avail_mem: float,
    m1_ic: bool,
    m2_ic: bool,
    v_ic: bool,
    out_ic: bool,
    m1_sparsity: float,
    m2_sparsity: float,
    dtype: Union[np.dtype, torch.dtype],
    kernel: "falkon.kernels.Kernel",
    is_differentiable: bool,
    is_sparse: bool,
) -> Tuple[int, int, int]:
    extra_mem = kernel.extra_mem(is_differentiable, is_sparse, dtype)
    normal_mem = defaultdict(int)
    normal_mem["nm"] = 1  # kernel block
    if is_sparse:
        m1_sparsity = m1_sparsity * 2  # to account for the storage complexity of CSR matrices
        m2_sparsity = m2_sparsity * 2  # to account for the storage complexity of CSR matrices
    multiplier = 1
    if is_differentiable:
        multiplier = 2  # for gradients
    if not m1_ic:
        normal_mem["n"] += d * m1_sparsity * multiplier
    if not m2_ic:
        normal_mem["m"] += d * m2_sparsity * multiplier
    if not out_ic:
        normal_mem["n"] += t * multiplier
    if not v_ic:
        normal_mem["m"] += t * multiplier

    blk_n, blk_m = select_dim_over_nm_v2(
        max_n=n,
        max_m=m,
        max_mem=avail_mem,
        coef_nm=normal_mem["nm"] + extra_mem.get("nm", 0),
        coef_n=normal_mem["n"] + extra_mem.get("n", 0) + extra_mem.get("nd", 0) * d,
        coef_m=normal_mem["m"] + extra_mem.get("m", 0) + extra_mem.get("md", 0) * d,
        rest=normal_mem["0"] + extra_mem.get("d", 0) * d + extra_mem.get("0", 0),
    )
    mem_needed = blk_m * blk_n
    if not out_ic:
        mem_needed += blk_n * t
    if not v_ic:
        mem_needed += blk_m * t
    if not is_sparse:
        if not m1_ic:
            mem_needed += blk_n * d
        if not m2_ic:
            mem_needed += blk_m * d
    return blk_n, blk_m, mem_needed


def mmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, out = a.X1, a.X2, a.v, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    differentiable = a.differentiable
    dev = _dev_from_id(device_id)
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)

    # Choose batch sizes
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    m1_ic, m2_ic, v_ic, out_ic = (
        _is_incore(dev, X1.device),
        _is_incore(dev, X2.device),
        _is_incore(dev, v.device),
        _is_incore(dev, out.device),
    )
    blk_n, blk_m, mem_needed = _mmv_blk_sizes(
        n=X1.size(-2),
        d=X1.size(-1),
        m=X2.size(-2),
        t=v.size(-1),
        avail_mem=avail_mem,
        m1_ic=m1_ic,
        m2_ic=m2_ic,
        v_ic=v_ic,
        out_ic=out_ic,
        m1_sparsity=X1.density if is_sparse else 1.0,
        m2_sparsity=X2.density if is_sparse else 1.0,
        dtype=X1.dtype,
        kernel=kernel,
        is_differentiable=differentiable,
        is_sparse=is_sparse,
    )
    if differentiable:
        assert not is_sparse, "Sparse + differentiable mmvs are not supported"
        return mmv_diff_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, dev, tid=proc_idx)
    if is_sparse:
        return sparse_mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev, tid=proc_idx)
    else:
        return mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev, tid=proc_idx)


def sparse_mmv_run_thread(
    m1: SparseTensor,
    m2: SparseTensor,
    v: torch.Tensor,
    out: torch.Tensor,
    kernel: "falkon.kernels.Kernel",
    blk_n: int,
    blk_m: int,
    mem_needed: int,
    dev: torch.device,
    tid: int,
):
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
    tid
        Thread ID or -1 if on main thread

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
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(blk_m, T), other=v, offset=flat_offset)

    with ExitStack() as stack, torch.inference_mode():
        s1, s2 = _init_two_streams(stack, dev, tid)  # enters stream 1
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)

            c_m1 = m1.narrow_rows(i, leni)
            if incore:  # Note that CUDA-incore is not allowed to happen (so this is CPU->CPU)
                c_dev_out = out[i : i + leni]
                c_dev_m1 = c_m1
            else:  # CPU -> CUDA
                c_dev_out = dev_out[:leni]
                c_dev_m1 = c_m1.index_to_int().to(device=dev, non_blocking=True)
            c_dev_out.fill_(0.0)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)

                c_m2 = m2.narrow_rows(j, lenj)
                if incore:  # CPU -> CPU
                    c_dev_m2 = c_m2.transpose_csc()
                    c_dev_v = v[j : j + lenj]
                else:  # CPU -> CUDA
                    c_dev_m2 = (
                        SparseTensor.from_scipy(c_m2.transpose_csc().to_scipy().tocsr(copy=False))
                        .index_to_int()
                        .to(device=dev, non_blocking=True)
                    )
                    with ExitStack() as stack2:
                        if dev.type == "cuda":
                            stack2.enter_context(tcd.stream(s2))
                        c_dev_v = copy(v[j : j + lenj], dev_v[:lenj], non_blocking=True)
                c_dev_ker = ker_gpu[:leni, :lenj].fill_(0.0)

                c_dev_ker = kernel.compute_sparse(c_dev_m1, c_dev_m2, c_dev_ker, diag=False, X1_csr=c_m1, X2_csr=c_m2)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)
                if not incore:
                    s1.synchronize()  # sync necessary to avoid s2 overwriting dev_v/dev_w
            # end iter over M
            if not incore:  # Copy output to host
                copy(c_dev_out, out[i : i + leni], non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()
        # end iter over N
    # exit context manager (device, stream)


def mmv_run_thread(
    m1: torch.Tensor,
    m2: torch.Tensor,
    v: Optional[torch.Tensor],
    out: torch.Tensor,
    kernel: "falkon.kernels.Kernel",
    blk_n: int,
    blk_m: int,
    mem_needed: int,
    dev: torch.device,
    tid: int,
):
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    m1_ic, m2_ic, v_ic, out_ic = (
        _is_incore(dev, m1.device),
        _is_incore(dev, m2.device),
        _is_incore(dev, v.device),
        _is_incore(dev, out.device),
    )
    incore = all((m1_ic, m2_ic, v_ic, out_ic))
    N, D = m1.shape
    M, T = v.shape

    # Initialize extra buffers
    # dev_ker = create_fortran((blk_n, blk_m), dtype=m1.dtype, device=dev)
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_ker, flat_offset = _extract_flat(flat_gpu, size=(blk_n, blk_m), other=out, offset=flat_offset)
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
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out, offset=flat_offset)

    with ExitStack() as stack, torch.inference_mode():
        s1, s2 = _init_two_streams(stack, dev, tid)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            # c_dev_m1 = m1[i: i + leni, :].to(dev, copy=False, non_blocking=True)
            # c_dev_out = out[i: i + leni, :].to(dev, copy=False, non_blocking=True)
            if m1_ic:
                c_dev_m1 = m1[i : i + leni, :]
            else:
                c_dev_m1 = copy(m1[i : i + leni, :], dev_m1[:leni, :], non_blocking=True)
            if out_ic:
                c_dev_out = out[i : i + leni]
            else:
                c_dev_out = dev_out[:leni]
            c_dev_out.fill_(0.0)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                if m2_ic:
                    c_dev_m2 = m2[j : j + lenj, :]
                else:
                    c_dev_m2 = copy(m2[j : j + lenj, :], dev_m2[:lenj, :], non_blocking=True)
                # c_dev_m2 = m2[j: j + lenj].to(dev, copy=False, non_blocking=True)
                if v_ic:
                    c_dev_v = v[j : j + lenj, :]
                else:
                    with ExitStack() as stack2:
                        if dev.type == "cuda":
                            stack2.enter_context(tcd.stream(s2))
                        c_dev_v = copy(v[j : j + lenj, :], dev_v[:lenj, :], non_blocking=True)
                # with ExitStack() as stack2:
                #     if dev.type == 'cuda':
                #         stack2.enter_context(tcd.stream(s2))
                #     c_dev_v = v[j: j + lenj, :].to(dev, copy=False, non_blocking=True)
                c_dev_ker = dev_ker[:leni, :lenj].fill_(0.0)

                c_dev_ker = kernel.compute(c_dev_m1, c_dev_m2, c_dev_ker, diag=False)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)
                if not incore:
                    s1.synchronize()  # sync necessary to avoid s2 overwriting dev_v/dev_w
            # end iter over M
            if not out_ic:
                copy(c_dev_out, out[i : i + leni], non_blocking=True)
            # if not out_ic:
            #     out[i: i + leni].copy_(c_dev_out, non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()
        # end iter over N
    # exit context manager (device, stream)


def mmv_diff_run_thread(
    m1: torch.Tensor,
    m2: torch.Tensor,
    v: Optional[torch.Tensor],
    out: torch.Tensor,
    kernel: "falkon.kernels.Kernel",
    blk_n: int,
    blk_m: int,
    dev: torch.device,
    tid: int,
):
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
    inputs_need_grad, input_idxs = zip(*[(ipt, idx) for idx, ipt in enumerate(inputs) if ipt.requires_grad])

    with ExitStack() as stack:
        s1, s2 = _init_two_streams(stack, dev, tid)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            c_dev_m1 = m1[i : i + leni, :].to(dev, non_blocking=True, copy=False)
            c_dev_m1_g = None if grads[0] is None else grads[0][i : i + leni, :].to(dev, non_blocking=True, copy=False)
            c_dev_out = out[i : i + leni, :].to(dev, non_blocking=True, copy=False)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                c_dev_m2 = m2[j : j + lenj, :].to(dev, non_blocking=True, copy=False)
                c_dev_m2_g = (
                    None if grads[1] is None else grads[1][j : j + lenj, :].to(dev, non_blocking=True, copy=False)
                )
                with ExitStack() as stack_s2:
                    if not incore:
                        stack_s2.enter_context(tcd.stream(s2))
                    c_dev_v = v[j : j + lenj, :].to(dev, non_blocking=True, copy=False)
                    c_dev_v_g = (
                        None if grads[2] is None else grads[2][j : j + lenj, :].to(dev, non_blocking=True, copy=False)
                    )
                c_dev_ker = kernel.compute_diff(c_dev_m1, c_dev_m2, diag=False)
                if not incore:
                    s2.synchronize()
                # main MMV operation on current block
                c_dev_mmv = c_dev_ker @ c_dev_v
                # Build inputs for torch.autograd.grad
                c_inputs = [c_dev_m1, c_dev_m2, c_dev_v] + list(kernel.diff_params.values())
                c_dev_grads = torch.autograd.grad(
                    c_dev_mmv, [c_inputs[idx] for idx in input_idxs], grad_outputs=c_dev_out
                )
                c_dev_grads_old = [c_dev_m1_g, c_dev_m2_g, c_dev_v_g] + grads[3:]
                for c_grad, c_idx in zip(c_dev_grads, input_idxs):
                    c_dev_grads_old[c_idx].add_(c_grad)
                # Move grads to host
                if grads[1] is not None:
                    grads[1][j : j + lenj, :].copy_(c_dev_m2_g, non_blocking=True)
                if grads[2] is not None:
                    grads[2][j : j + lenj, :].copy_(c_dev_v_g, non_blocking=True)
                if not incore:
                    s1.synchronize()  # sync necessary to avoid s2 overwriting dev_v/dev_w
            # end iter over M
            # Move grads to host
            if grads[0] is not None:
                grads[0][i : i + leni, :].copy_(c_dev_m1_g, non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()
        # end iter over N
    # exit context manager (device, stream)
    return grads


def _dmmv_blk_sizes(
    n: int,
    d: int,
    m: int,
    t: int,
    avail_mem: float,
    m1_ic: bool,
    m2_ic: bool,
    v_ic: bool,
    w_ic: bool,
    out_ic: bool,
    m1_sparsity: float,
    m2_sparsity: float,
    dtype: Union[np.dtype, torch.dtype],
    kernel: "falkon.kernels.Kernel",
    is_differentiable: bool,
    is_sparse: bool,
) -> Tuple[int, int]:
    extra_mem = kernel.extra_mem(is_differentiable, is_sparse, dtype)
    normal_mem = defaultdict(int)
    normal_mem["nm"] = 1  # kernel block
    normal_mem["nt"] = 1  # w block (TODO: This alloc should be removed if it's IC)
    if is_sparse:
        m1_sparsity = m1_sparsity * 2  # to account for the storage complexity of CSR matrices
        m2_sparsity = m2_sparsity * 2  # to account for the storage complexity of CSR matrices
    if not m1_ic:
        normal_mem["nd"] += m1_sparsity
    if not m2_ic:
        normal_mem["md"] += m2_sparsity
    if not out_ic:
        normal_mem["mt"] += 1
    if not v_ic:
        normal_mem["mt"] += 1

    blk_n = select_dim_over_n(
        max_n=n,
        m=m,
        d=d,
        max_mem=avail_mem,
        coef_nm=normal_mem["nm"] + extra_mem.get("nm", 0),
        coef_nd=normal_mem["nd"] + extra_mem.get("nd", 0),
        coef_md=normal_mem["md"] + extra_mem.get("md", 0),
        coef_n=(normal_mem["nt"] + extra_mem.get("nt", 0)) * t + extra_mem.get("n", 0),
        coef_m=(normal_mem["mt"] + extra_mem.get("mt", 0)) * t + extra_mem.get("m", 0),
        coef_d=extra_mem.get("d", 0),
        rest=extra_mem.get("0", 0),
    )
    mem_needed = blk_n * (m + t)  # for kernel block and w
    if not m1_ic and not is_sparse:
        mem_needed += blk_n * d  # m1
    if not m2_ic and not is_sparse:
        mem_needed += m * d  # m2
    if not v_ic:
        mem_needed += m * t
    if not out_ic:
        mem_needed += m * t
    return blk_n, mem_needed


def dmmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, w, out = a.X1, a.X2, a.v, a.w, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    assert not a.differentiable, "D-MMV not implemented for differentiable outputs"
    dev = _dev_from_id(device_id)
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)

    # Choose batch sizes
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    blk_n, mem_needed = _dmmv_blk_sizes(
        n=X1.size(-2),
        d=X1.size(-1),
        m=X2.size(-2),
        t=v.size(-1),
        avail_mem=avail_mem,
        m1_ic=_is_incore(dev, X1.device),
        m2_ic=_is_incore(dev, X2.device),
        v_ic=_is_incore(dev, v.device),
        w_ic=_is_incore(dev, w.device) if w is not None else False,
        out_ic=_is_incore(dev, out.device),
        m1_sparsity=X1.density if is_sparse else 1.0,
        m2_sparsity=X2.density if is_sparse else 1.0,
        dtype=X1.dtype,
        kernel=kernel,
        is_differentiable=False,
        is_sparse=is_sparse,
    )

    if is_sparse:
        sparse_dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev, tid=proc_idx)
    else:
        dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev, tid=proc_idx)


def sparse_dmmv_run_thread(
    m1: SparseTensor,
    m2: SparseTensor,
    v: torch.Tensor,
    w: Optional[torch.Tensor],
    out: torch.Tensor,
    kernel: "falkon.kernels.Kernel",
    blk_n: int,
    mem_needed: int,
    dev: torch.device,
    tid: int,
):
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
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w, offset=flat_offset)
    dev_out, dev_v, dev_m2 = out, v, m2
    if not incore:
        if not dev_out_exists:
            dev_out, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=v, offset=flat_offset)

    with ExitStack() as stack, torch.inference_mode():
        s1 = None
        if dev.type == "cuda":
            s1 = tcd.current_stream(dev) if tid == -1 else tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        dev_out.fill_(0.0)  # Needs to be inside inference mode
        if not incore:  # Note that CUDA-incore is not allowed to happen (CPU->CUDA)
            copy(v, dev_v, non_blocking=True)
            dev_m2 = (
                SparseTensor.from_scipy(m2.transpose_csc().to_scipy().tocsr(copy=False)).index_to_int().to(device=dev)
            )
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
                c_dev_w = copy(w[i : i + leni, :], dev_w[:leni, :], non_blocking=True)

            c_dev_ker = ker_gpu[:leni].fill_(0.0)
            c_dev_ker = kernel.compute_sparse(c_dev_m1, dev_m2, c_dev_ker, diag=False, X1_csr=c_m1, X2_csr=m2)

            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
        if not incore and not dev_out_exists:
            copy(dev_out, out, non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()


def dmmv_run_thread(
    m1: torch.Tensor,
    m2: torch.Tensor,
    v: torch.Tensor,
    w: Optional[torch.Tensor],
    out: torch.Tensor,
    kernel: "falkon.kernels.Kernel",
    blk_n: int,
    mem_needed: int,
    dev: torch.device,
    tid: int,
):
    # k(x2, x1) @ (k(x1, x2) @ v + w)
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    m1_ic, m2_ic, v_ic, out_ic = (
        _is_incore(dev, m1.device),
        _is_incore(dev, m2.device),
        _is_incore(dev, v.device),
        _is_incore(dev, out.device),
    )
    N, D = m1.shape
    M, T = v.shape

    # Initialize extra buffers
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_ker, flat_offset = _extract_flat(flat_gpu, size=(blk_n, M), other=out, offset=flat_offset)
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w, offset=flat_offset)
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

    with ExitStack() as stack, torch.inference_mode():
        s1, s2 = _init_two_streams(stack, dev, tid)
        dev_out.fill_(0.0)
        if not m2_ic:
            copy(m2, dev_m2, non_blocking=True)
        if not v_ic:
            with ExitStack() as stack2:
                if s2 is not None:
                    stack2.enter_context(tcd.stream(s2))
                copy(v, dev_v, non_blocking=True)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            if m1_ic:
                c_dev_m1 = m1[i : i + leni, :]
            else:
                c_dev_m1 = copy(m1[i : i + leni, :], dev_m1[:leni, :], non_blocking=True)
            if w is not None:
                with ExitStack() as stack2:
                    if s2 is not None:
                        stack2.enter_context(tcd.stream(s2))
                    c_dev_w = copy(w[i : i + leni, :], dev_w[:leni, :], non_blocking=True)
            else:
                c_dev_w = dev_w[:leni, :].fill_(0.0)

            c_dev_ker = dev_ker[:leni, :].fill_(0.0)
            c_dev_ker = kernel.compute(c_dev_m1, dev_m2, c_dev_ker, diag=False)
            if s2 is not None:
                s2.synchronize()
            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
            if s1 is not None:
                s1.synchronize()  # sync necessary to avoid s2 overwriting dev_v/dev_w

        if not out_ic:
            copy(dev_out, out, non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()


# noinspection PyMethodOverriding
class KernelMmvFnFull(torch.autograd.Function):
    @staticmethod
    def run_cpu_cpu(
        X1: Union[torch.Tensor, SparseTensor],
        X2: Union[torch.Tensor, SparseTensor],
        v: torch.Tensor,
        out: torch.Tensor,
        kernel,
        options,
        diff,
    ):
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel, max_mem=options.max_cpu_mem, differentiable=diff)
        return _call_direct(mmv_run_starter, (args, -1))

    @staticmethod
    def run_cpu_gpu(
        X1: Union[torch.Tensor, SparseTensor],
        X2: Union[torch.Tensor, SparseTensor],
        v: torch.Tensor,
        out: torch.Tensor,
        kernel,
        options,
        diff,
    ):
        is_sparse = isinstance(X1, SparseTensor)
        gpu_info = _get_gpu_info(options, slack=options.memory_slack)
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
            args.append(
                (
                    ArgsFmmv(
                        X1=X1_block,
                        X2=X2,
                        v=v,
                        out=out.narrow(0, block_sizes[i], bwidth),
                        kernel=kernel,
                        max_mem=g.usable_memory,
                        differentiable=diff,
                    ),
                    g.Id,
                )
            )
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
    def run_gpu_gpu(
        X1: Union[torch.Tensor, SparseTensor],
        X2: Union[torch.Tensor, SparseTensor],
        v: torch.Tensor,
        out: torch.Tensor,
        kernel,
        options,
        diff,
    ):
        if isinstance(X1, SparseTensor):
            raise NotImplementedError("In-core, sparse fmmv not implemented. Use the out-of-core version instead.")
        data_dev = X1.device
        gpu_info = _get_gpu_info(options, slack=options.memory_slack)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFmmv(
            X1=X1, X2=X2, v=v, out=out, kernel=kernel, max_mem=single_gpu_info.usable_memory, differentiable=diff
        )
        return _call_direct(mmv_run_starter, (args, data_dev.index))

    @staticmethod
    def forward(
        ctx,
        kernel: "falkon.kernels.Kernel",
        opt: Optional[BaseOptions],
        out: Optional[torch.Tensor],
        X1: Union[torch.Tensor, SparseTensor],
        X2: Union[torch.Tensor, SparseTensor],
        v: torch.Tensor,
        *kernel_params,
    ):
        is_sparse = isinstance(X1, SparseTensor)
        if is_sparse:
            differentiable = False
        else:
            _check_contiguity((X1, "X1"), (X2, "X2"), (v, "v"), (out, "out"))
            differentiable = any(t.requires_grad for t in [X1, X2, v] + [*kernel_params])
        data_devs = (X1.device, X2.device, v.device)
        comp_dev_type = "cpu" if opt.use_cpu or not torch.cuda.is_available() else "cuda"
        N, D = X1.shape
        T = v.shape[-1]
        # Create output matrix
        out = create_output_mat(
            out, data_devs, is_sparse, shape=(N, T), dtype=v.dtype, comp_dev_type=comp_dev_type, other_mat=X1
        )

        with torch.inference_mode():
            if comp_dev_type == "cpu" and all(ddev.type == "cpu" for ddev in data_devs):
                KernelMmvFnFull.run_cpu_cpu(X1, X2, v, out, kernel, opt, False)
            elif comp_dev_type == "cuda" and all(ddev.type == "cuda" for ddev in data_devs):
                KernelMmvFnFull.run_gpu_gpu(X1, X2, v, out, kernel, opt, False)
            elif comp_dev_type == "cuda":
                KernelMmvFnFull.run_cpu_gpu(X1, X2, v, out, kernel, opt, False)
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
        comp_dev_type = "cpu" if ctx.opt.use_cpu or not torch.cuda.is_available() else "cuda"

        # We must rerun MM in differentiable mode this time.
        with torch.autograd.enable_grad():
            if comp_dev_type == "cpu" and data_dev.type == "cpu":
                grads = KernelMmvFnFull.run_cpu_cpu(X1, X2, v, outputs, ctx.kernel, ctx.opt, True)
            elif comp_dev_type == "cuda" and data_dev.type == "cuda":
                grads = KernelMmvFnFull.run_gpu_gpu(X1, X2, v, outputs, ctx.kernel, ctx.opt, True)
            elif comp_dev_type == "cuda" and data_dev.type == "cpu":
                grads = KernelMmvFnFull.run_cpu_gpu(X1, X2, v, outputs, ctx.kernel, ctx.opt, True)
            else:
                raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
            return tuple([None, None, None] + grads)


def fmmv(
    X1: Union[torch.Tensor, SparseTensor],
    X2: Union[torch.Tensor, SparseTensor],
    v: torch.Tensor,
    kernel: "falkon.kernels.Kernel",
    out: Optional[torch.Tensor] = None,
    opt: Optional[BaseOptions] = None,
):
    if isinstance(kernel, falkon.kernels.DiffKernel):
        return KernelMmvFnFull.apply(kernel, opt, out, X1, X2, v, *kernel.diff_params.values())
    else:
        return KernelMmvFnFull.apply(kernel, opt, out, X1, X2, v)


def fdmmv(
    X1: Union[torch.Tensor, SparseTensor],
    X2: Union[torch.Tensor, SparseTensor],
    v: torch.Tensor,
    w: Optional[torch.Tensor],
    kernel: "falkon.kernels.Kernel",
    out: Optional[torch.Tensor] = None,
    differentiable: bool = False,
    opt: Optional[BaseOptions] = None,
) -> torch.Tensor:
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
            "Manual D-MMV is not implemented for differentiable inputs. Try using the KeOps version instead."
        )
    is_sparse = isinstance(X1, SparseTensor)
    if not is_sparse:
        _check_contiguity((X1, "X1"), (X2, "X2"), (v, "v"), (out, "out"))
    data_devs = [X1.device, X2.device, v.device]
    if w is not None:
        data_devs.append(w.device)
    comp_dev_type = "cpu" if opt.use_cpu or not torch.cuda.is_available() else "cuda"

    N, D = X1.shape[-2:]
    M, T = v.shape[-2:]

    with torch.inference_mode():
        # Create output matrix
        out = create_output_mat(
            out, data_devs, is_sparse, shape=(M, T), dtype=v.dtype, comp_dev_type=comp_dev_type, other_mat=X1
        )

        if comp_dev_type == "cpu" and all(ddev.type == "cpu" for ddev in data_devs):
            args = ArgsFmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel, max_mem=opt.max_cpu_mem)
            _call_direct(dmmv_run_starter, (args, -1))
        elif comp_dev_type == "cuda" and all(ddev.type == "cuda" for ddev in data_devs):
            if is_sparse:
                raise NotImplementedError("In-core, sparse fdmmv not implemented. Use the out-of-core version instead.")
            gpu_info = _get_gpu_info(opt, slack=opt.memory_slack)
            data_dev = data_devs[0]
            single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
            args = ArgsFmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel, max_mem=single_gpu_info.usable_memory)
            _call_direct(dmmv_run_starter, (args, data_dev.index))
        elif comp_dev_type == "cuda":
            gpu_info = _get_gpu_info(opt, slack=opt.memory_slack)
            args = []  # Arguments passed to each subprocess
            wrlk = []  # Outputs for each subprocess
            block_sizes = calc_gpu_block_sizes(gpu_info, N)
            for i, g in enumerate(gpu_info):
                bwidth = block_sizes[i + 1] - block_sizes[i]
                if bwidth <= 0:
                    continue
                if len(gpu_info) == 1 and out.device.index == g.Id:
                    cur_out_gpu = out
                else:
                    cur_out_gpu = create_same_stride((M, T), out, out.dtype, f"cuda:{g.Id}")
                    g.usable_memory -= M * T * sizeof_dtype(X1.dtype)
                wrlk.append(cur_out_gpu)
                if is_sparse:
                    X1_block = X1.narrow_rows(block_sizes[i], bwidth)
                else:
                    X1_block = X1.narrow(0, block_sizes[i], bwidth)
                args.append(
                    (
                        ArgsFmmv(
                            X1=X1_block,
                            X2=X2,
                            v=v,
                            w=w.narrow(0, block_sizes[i], bwidth) if w is not None else None,
                            out=cur_out_gpu,
                            kernel=kernel,
                            max_mem=g.usable_memory,
                        ),
                        g.Id,
                    )
                )
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
