from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.cuda as tcd
import torch.cuda.comm
import torch.multiprocessing

from falkon.kernels import Kernel, L2DistanceKernel
from falkon.mmv_ops.utils import (
    _setup_opt, _check_contiguity, _get_gpu_info,
    _start_wait_processes,
    _call_direct
)
from falkon.options import BaseOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.cuda_helpers import (
    copy_to_device_noorder, copy_to_host_noorder
)
from falkon.utils.stream_utils import sync_current_stream
from falkon.utils.helpers import (
    calc_gpu_block_sizes,
    sizeof_dtype,
    select_dim_over_nd,
    select_dim_over_nm_v2
)
from falkon.utils.tensor_helpers import (
    create_same_stride,
    extract_same_stride,
    create_fortran,
    create_C,
    extract_fortran,
)

__all__ = ("fmmv_cuda", "fdmmv_cuda", "fmmv_cuda_sparse", "fdmmv_cuda_sparse")


@dataclass(frozen=True)
class ArgsFmmv:
    X1: Union[torch.Tensor, SparseTensor]
    X2: Union[torch.Tensor, SparseTensor]
    v: torch.Tensor
    out: torch.Tensor
    kernel: Kernel
    max_mem: float


@dataclass(frozen=True)
class ArgsFdmmv:
    X1: Union[torch.Tensor, SparseTensor]
    X2: Union[torch.Tensor, SparseTensor]
    v: torch.Tensor
    w: torch.Tensor
    out: torch.Tensor
    kernel: Kernel
    max_mem: float


def _extract_flat(flat_tn, size, other, offset):
    struct_tn = extract_same_stride(flat_tn, size=size, other=other, offset=offset)
    offset += np.prod(struct_tn.shape)
    return struct_tn, offset


def sparse_fmmv(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()

    X1: SparseTensor = a.X1
    X2: SparseTensor = a.X2
    v, out = a.v, a.out
    kernel, max_mem = a.kernel, a.max_mem
    dtype = X1.dtype
    cuda_inputs = X1.is_cuda
    ntot, dtot = X1.shape
    mtot, T = v.size()

    avail_mem = max_mem / sizeof_dtype(dtype)
    # Memory needs:
    # X1_chunk : N + 2*D*N*density
    # X2_chunk : D + 2*D*M*density (because is transposed)
    # sparse_out : N + 2*N*M*(density) (assume density = 1)
    # ker_gpu  : M*N
    # mmv_gpu  : N*T
    # v_gpu    : M*T
    # Other: GPU buffer
    n, m = select_dim_over_nm_v2(max_n=ntot, max_m=mtot, coef_nm=3,
                                 coef_n=2 + 2 * dtot * X1.density + T,
                                 coef_m=2 * dtot * X2.density + T, rest=dtot, max_mem=avail_mem)

    ddev = torch.device('cuda:%d' % int(device_id))
    with tcd.device(ddev):
        # First collect necessary memory
        mem_needed = mtot * T + n * T + n * m
        # Create flat tensor
        flat_gpu_tn = torch.empty(size=(mem_needed,), dtype=dtype, device=ddev)
        # Extract the sub-tensors
        flat_offset = 0
        v_gpu = extract_same_stride(flat_gpu_tn, size=(mtot, T), other=v, offset=flat_offset)
        flat_offset += np.prod(v_gpu.shape)
        copy_to_device_noorder(mtot, T, v, 0, 0, v_gpu, 0, 0)
        mmv_gpu = extract_same_stride(flat_gpu_tn, size=(n, T), other=out, offset=flat_offset)
        flat_offset += np.prod(mmv_gpu.shape)
        # ker_gpu should be fortran-ordered due to cusparse csr2dense function
        ker_gpu = extract_fortran(flat_gpu_tn, size=(n, m), offset=flat_offset)
        flat_offset += np.prod(ker_gpu.shape)

        for i in range(0, ntot, n):
            ic = min(n, ntot - i)

            cur_mmv_gpu = mmv_gpu[:ic]  # n x T
            cur_mmv_gpu.fill_(0.0)

            X1_chunk = X1.narrow_rows(i, ic)
            X1_chunk_d = X1_chunk.index_to_int().to(device=ddev)
            for j in range(0, mtot, m):
                jc = min(m, mtot - j)

                X2_chunk = X2.narrow_rows(j, jc)
                # Prepare sparse on CPU
                ddd = kernel._prepare_sparse(X1_chunk, X2_chunk)

                # Transpose X2-chunk and convert it to CSR. This uses lots of RAM
                X2_chunk_d = SparseTensor.from_scipy(
                    X2_chunk.transpose_csc().to_scipy().tocsr(copy=False)) \
                    .index_to_int() \
                    .to(device=ddev)

                cur_ker_gpu = ker_gpu[:ic, :jc]
                cur_ker_gpu.fill_(0.0)
                # Run the matrix multiplication (kernel apply)
                kernel._apply_sparse(X1_chunk_d, X2_chunk_d, cur_ker_gpu)
                cur_ker_gpu = kernel._finalize(cur_ker_gpu, ddd)

                # Multiply by the vector v
                cur_mmv_gpu.addmm_(cur_ker_gpu, v_gpu.narrow(0, j, jc))
                del ddd, X2_chunk, X2_chunk_d

            # send result to CPU
            if not cuda_inputs:
                copy_to_host_noorder(ic, T, cur_mmv_gpu, 0, 0, out, i, 0)
            del X1_chunk, X1_chunk_d
    return out


def generic_fmmv(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()

    X1, X2, v, out = a.X1, a.X2, a.v, a.out
    kernel, max_mem = a.kernel, a.max_mem
    dtype = X1.dtype
    cuda_inputs = X1.is_cuda
    ntot, dtot = X1.size()
    M, T = v.size()

    # GPU Memory Usage:
    # ker_gpu  : n*M
    # v_gpu    : M*T
    # X1s_gpu  : n*d
    # X2s_gpu  : M*d
    # mmv_gpu  : n*T
    # ----------
    # total : n*d + n*(M+T) + d*M + M*T
    avail_mem = max_mem / sizeof_dtype(dtype)
    extra_mem = kernel.extra_mem()
    n, d = select_dim_over_nd(max_n=ntot, max_d=dtot,
                              coef_nd=1 + extra_mem.get('nd', 0),
                              coef_n=M + T + extra_mem.get('n', 0) + extra_mem.get('nm', 0) * M,
                              coef_d=M + extra_mem.get('d', 0) + extra_mem.get('md', 0) * M,
                              rest=M * T + extra_mem.get('m', 0),
                              max_mem=avail_mem)

    ddev = torch.device('cuda:%d' % int(device_id))
    s1 = tcd.current_stream(ddev)
    #print("Usage before start, chosen n=%d, d=%d - dev %s: %.5fMB" % (n, d, ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
    with tcd.device(ddev), tcd.stream(s1):
        # First collect necessary memory
        mem_needed = n * M
        if not cuda_inputs:
            mem_needed += M * T + n * d + M * d + n * T
        # Create flat tensor
        flat_gpu_tn = torch.empty(size=(mem_needed,), dtype=dtype, device=ddev)
        #print("After big Alloc %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
        # Extract the sub-tensors
        flat_offset = 0
        ker_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, M), other=X1, offset=flat_offset)
        if not cuda_inputs:
            X1s_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, d), other=X1, offset=flat_offset)
            X2s_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, d), other=X2, offset=flat_offset)
            mmv_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, T), other=out, offset=flat_offset)
            v_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, T), other=v, offset=flat_offset)
            copy_to_device_noorder(M, T, v, 0, 0, v_gpu, 0, 0, s=s1)
        else:
            v_gpu = v
        #print("After extractions %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))

        for i in range(0, ntot, n):
            ic = min(n, ntot - i)
            ddd = kernel._prepare(X1.narrow(0, i, ic), X2)
            c_g_ker = ker_gpu.narrow(0, 0, ic)
            c_g_ker.fill_(0.0)
            #print("After prepare %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
            for k in range(0, dtot, d):
                kc = min(d, dtot - k)
                if cuda_inputs:
                    c_g_X1s = X1[i:i + ic, k:k + kc]
                    c_g_X2s = X2[:, k:k + kc]
                else:
                    c_g_X1s = copy_to_device_noorder(ic, kc, X1, i, k, X1s_gpu, 0, 0, s=s1)
                    c_g_X2s = copy_to_device_noorder(M, kc, X2, 0, k, X2s_gpu, 0, 0, s=s1)
                kernel._apply(c_g_X1s, c_g_X2s.T, c_g_ker)
                #print("After apply %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
            kernel._finalize(c_g_ker, ddd)
            #print("After finalize %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
            # Multiply by the vector v
            if cuda_inputs:
                c_g_mmv = out[i:i + ic, :]
            else:
                c_g_mmv = mmv_gpu[:ic, :]
            torch.mm(c_g_ker, v_gpu, out=c_g_mmv)  # n x T
            #print("After mm %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
            # Copy back to host
            if not cuda_inputs:
                copy_to_host_noorder(ic, T, c_g_mmv, 0, 0, out, i, 0, s=s1)
            #print("After copy %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
        s1.synchronize()
    #print("returning %s: %.5fMB" % (ddev, torch.cuda.max_memory_allocated(ddev) / 2**20))
    return out


def sparse_fdmmv(proc_idx, queue, device_id):
    a: ArgsFdmmv = queue.get()
    X1: SparseTensor = a.X1
    X2: SparseTensor = a.X2
    v, w, out = a.v, a.w, a.out
    kernel, max_mem = a.kernel, a.max_mem
    dtype = X1.dtype
    N, D = X1.shape
    M = X2.size(0)
    if v is None:
        T = w.shape[1]
    else:
        T = v.shape[1]

    # Memory needs:
    # X1_chunk : ntot + 2 * D * ntot * density
    # X2       : dtot + 2 * D * M * density (because is transposed)
    # sparse_out : ntot + 2 * ntot * M * density (assume here density = 1)
    # ker_gpu  : M * ntot
    # w_gpu    : ntot * T
    # v_gpu    : M * T
    # out_gpu  : M * T
    avail_mem = max_mem / sizeof_dtype(dtype)
    den = 2 * D * X1.density + 2 + 3 * M + T
    sub = D + 2 * D * M * X2.density + M * T
    if v is not None:
        sub += M * T
    n = (avail_mem - sub) / den
    n = min(int(n), N)
    if n < 1:
        raise MemoryError("Not enough memory to run sparse dfmmv")

    ddev = torch.device('cuda:%d' % int(device_id))
    with tcd.device(ddev):
        # First collect necessary memory
        mem_needed = n * M + n * T
        if not out.is_cuda:
            mem_needed += M * T
        if v is not None:
            mem_needed += M * T
        # Create flat tensor
        flat_gpu_tn = torch.empty(size=(mem_needed,), dtype=dtype, device=ddev)
        # Extract the sub-tensors
        flat_offset = 0
        ker_gpu = extract_fortran(flat_gpu_tn, size=(n, M), offset=flat_offset)
        flat_offset += np.prod(ker_gpu.shape)
        w_gpu = extract_same_stride(flat_gpu_tn, size=(n, T), other=out, offset=flat_offset)
        flat_offset += np.prod(w_gpu.shape)
        if not out.is_cuda:
            out_gpu = extract_same_stride(flat_gpu_tn, size=(M, T), other=out, offset=flat_offset)
            flat_offset += np.prod(out_gpu.shape)
        else:
            out_gpu = out
        out_gpu.fill_(0.0)
        if v is not None:
            v_gpu = extract_same_stride(flat_gpu_tn, size=(M, T), other=v, offset=flat_offset)
            flat_offset += np.prod(v_gpu.shape)
            copy_to_device_noorder(M, T, v, 0, 0, v_gpu, 0, 0)
        # Sparse GPU data is allocated separately.
        X2_d = SparseTensor.from_scipy(
            X2.transpose_csc().to_scipy().tocsr(copy=False)) \
            .index_to_int() \
            .to(device=ddev)

        for i in range(0, N, n):
            ic = min(n, N - i)
            X1_chunk = X1.narrow_rows(i, ic)
            X1_chunk_d = X1_chunk.index_to_int().to(device=ddev)

            ker_chunk = ker_gpu[:ic]
            ker_chunk.fill_(0.0)

            # TODO: This is wasteful (X2 will be prepared many times over)
            ddd = kernel._prepare_sparse(X1_chunk, X2)
            ker_chunk = kernel._apply_sparse(X1_chunk_d, X2_d, ker_chunk)
            ker_chunk = kernel._finalize(ker_chunk, ddd)

            if w is not None:
                c_g_w = copy_to_device_noorder(ic, T, w, i, 0, w_gpu, 0, 0)
            else:
                c_g_w = w_gpu.narrow(0, 0, ic)
                c_g_w.fill_(0.0)

            if v is not None:
                c_g_w.addmm_(ker_chunk, v_gpu)
            out_gpu.addmm_(ker_chunk.T, c_g_w)
            del ddd, X1_chunk, X1_chunk_d

        if not out.is_cuda:
            copy_to_device_noorder(M, T, out_gpu, 0, 0, out, 0, 0)
    return out


def generic_fdmmv(proc_idx, queue, device_id):
    a: ArgsFdmmv = queue.get()
    X1, X2, v, w, out = a.X1, a.X2, a.v, a.w, a.out
    kernel, max_mem = a.kernel, a.max_mem
    dtype = X1.dtype
    cuda_inputs = X1.is_cuda
    N, D = X1.size()
    M = X2.shape[0]
    if v is None:
        T = w.shape[1]
    else:
        T = v.shape[1]

    # Memory usage:
    # v    : M x T
    # K    : n x M
    # X1d  : n x d
    # X2d  : M x d
    # Kv   : n x T
    # out2 : M x T
    # sq1  : n x 1
    # sq2  : M x 1
    # ------------
    # total : n*d + M*d + n*(M + T) + 2*M*T + M
    avail_mem = max_mem / sizeof_dtype(dtype)
    # FIXME: There seems to be a bug where if we let avail_mem like it is
    #        for 32-bit data-types some copy fails. In such case we need
    #        to free up some more memory and then everything runs fine.
    if sizeof_dtype(dtype) == 4:
        avail_mem /= 2
    rest_coef = 2 * M * T if v is not None else M * T
    extra_mem = kernel.extra_mem()
    n, d = select_dim_over_nd(max_n=N, max_d=D,
                              coef_nd=1 + extra_mem.get('nd', 0),
                              coef_n=M + T + 1 + extra_mem.get('n', 0) + extra_mem.get('nm', 0) * M,
                              coef_d=M + extra_mem.get('d', 0) + extra_mem.get('md', 0) * M,
                              rest=rest_coef + M + extra_mem.get('m', 0),
                              max_mem=avail_mem)
    ddev = torch.device('cuda:%d' % int(device_id))
    s1 = tcd.current_stream(ddev)
    with tcd.device(ddev), tcd.stream(s1):
        # First collect necessary memory
        mem_needed = n * M + n * T
        if not cuda_inputs:
            mem_needed += n * d + M * d + M * T
            if v is not None:
                mem_needed += M * T
        # Create flat tensor
        flat_gpu_tn = torch.empty(size=(mem_needed,), dtype=dtype, device=ddev)
        # Extract the sub-tensors
        flat_offset = 0
        ker_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, M), other=out, offset=flat_offset)
        w_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, T), other=out, offset=flat_offset)
        if not cuda_inputs:
            X1s_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, d), other=X1, offset=flat_offset)
            X2s_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, d), other=X2, offset=flat_offset)
            out_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, T), other=out, offset=flat_offset)
            if v is not None:
                v_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, T), other=v, offset=flat_offset)
                copy_to_device_noorder(M, T, v, 0, 0, v_gpu, 0, 0, s=s1)
        else:
            out_gpu = out
            if v is not None:
                v_gpu = v
        out_gpu.fill_(0.0)

        # Algorithm start
        for i in range(0, N, n):
            ic = min(n, N - i)
            ddd = kernel._prepare(X1.narrow(0, i, ic), X2)

            c_g_ker = ker_gpu.narrow(0, 0, ic)
            c_g_ker.fill_(0.0)
            for k in range(0, D, d):
                kc = min(d, D - k)
                if cuda_inputs:
                    c_g_X1s = X1[i:i + ic, k:k + kc]
                    c_g_X2s = X2[:, k:k + kc]
                else:
                    c_g_X1s = copy_to_device_noorder(ic, kc, X1, i, k, X1s_gpu, 0, 0, s=s1)
                    c_g_X2s = copy_to_device_noorder(M, kc, X2, 0, k, X2s_gpu, 0, 0, s=s1)
                kernel._apply(c_g_X1s, c_g_X2s.T, c_g_ker)
            kernel._finalize(c_g_ker, ddd)

            if w is not None:
                c_g_w = copy_to_device_noorder(ic, T, w, i, 0, w_gpu, 0, 0, s=s1)
            else:
                c_g_w = w_gpu.narrow(0, 0, ic)
                c_g_w.fill_(0.0)
            if v is not None:
                c_g_w.addmm_(c_g_ker, v_gpu)
            out_gpu.addmm_(c_g_ker.T, c_g_w)

        if not cuda_inputs:
            copy_to_device_noorder(M, T, out_gpu, 0, 0, out, 0, 0, s=s1)
        s1.synchronize()
    return out


def distk_fmmv(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, out = a.X1, a.X2, a.v, a.out
    kernel: L2DistanceKernel = a.kernel
    max_mem = a.max_mem

    N, D = X1.shape
    M = X2.shape[0]
    T = v.shape[1]
    dtype = X1.dtype
    cuda_inputs = X1.is_cuda

    # GPU memory usage:
    # X1s : n x D
    # X2s : m x D
    # vs  : m x T
    # nm  : n x m
    # out : n x T
    # -----------
    # total: n*m + n * (D + T) + m * (D + T) = R
    avail_mem = max_mem / sizeof_dtype(dtype)
    n, m = select_dim_over_nm_v2(max_n=N, max_m=M, coef_nm=1, coef_n=D + T, coef_m=D + T, rest=0,
                                 max_mem=avail_mem)

    ddev = torch.device('cuda:%d' % int(device_id))
    s1 = tcd.current_stream(ddev)
    with tcd.device(ddev), tcd.stream(s1):
        mem_needed = n * m
        if not cuda_inputs:
            mem_needed += n * T + n * D + m * D + m * T
        flat_gpu_tn = torch.empty(size=(mem_needed,), dtype=dtype, device=ddev)

        flat_offset = 0
        nm_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, m), other=X1, offset=flat_offset)
        if not cuda_inputs:
            out_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, T), other=out, offset=flat_offset)
            X1s_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, D), other=X1, offset=flat_offset)
            X2s_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(m, D), other=X2, offset=flat_offset)
            vs_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(m, T), other=v, offset=flat_offset)

        for i in range(0, N, n):
            nb = min(n, N - i)
            if cuda_inputs:
                cur_X1s_gpu = X1.narrow(0, i, nb)  # n x D
            else:
                cur_X1s_gpu = copy_to_device_noorder(nb, D, X1, i, 0, X1s_gpu, 0, 0, s=s1)
            sq1 = torch.norm(cur_X1s_gpu, p=2, dim=1, keepdim=True).pow_(2)
            if cuda_inputs:
                cur_out_gpu = out.narrow(0, i, nb)  # n x T
            else:
                cur_out_gpu = out_gpu.narrow(0, 0, nb)  # n x T
            cur_out_gpu.fill_(0.0)

            for j in range(0, M, m):
                mb = min(m, M - j)
                if cuda_inputs:
                    cur_X2s_gpu = X2.narrow(0, j, mb)  # m x D
                    cur_vs_gpu = v.narrow(0, j, mb)  # m x T
                else:
                    cur_X2s_gpu = copy_to_device_noorder(mb, D, X2, j, 0, X2s_gpu, 0, 0, s=s1)  # m x D
                    cur_vs_gpu = copy_to_device_noorder(mb, T, v, j, 0, vs_gpu, 0, 0, s=s1)  # m x T
                cur_nm_gpu = nm_gpu[:nb, :mb]  # n x m

                sq2 = torch.norm(cur_X2s_gpu, p=2, dim=1, keepdim=True).pow_(2)
                torch.mm(cur_X1s_gpu, cur_X2s_gpu.T, out=cur_nm_gpu)

                cur_nm_gpu.mul_(-2.0)
                cur_nm_gpu.add_(sq1)
                cur_nm_gpu.add_(sq2.T)
                cur_nm_gpu.clamp_min_(0)
                kernel._transform(cur_nm_gpu)

                # Multiply by the vector v
                cur_out_gpu.addmm_(cur_nm_gpu, cur_vs_gpu)  # n x T
            if not cuda_inputs:
                # send result to CPU
                copy_to_host_noorder(nb, T, out_gpu, 0, 0, out, i, 0, s=s1)
        s1.synchronize()
    return out


def distk_fdmmv(proc_idx, queue, device_id):
    a: ArgsFdmmv = queue.get()
    X1, X2, v, w, out = a.X1, a.X2, a.v, a.w, a.out
    kernel: L2DistanceKernel = a.kernel
    max_mem = a.max_mem
    N, D = X1.size()
    M = X2.size(0)
    T = v.shape[1] if v is not None else w.shape[1]
    dtype = X1.dtype
    cuda_inputs = X1.is_cuda

    # Memory usage:
    # v    : M x T
    # K    : n x M
    # X1ss : n x d
    # X2s  : M x d
    # Kv   : n x T
    # out  : M x T
    # sq1  : n x 1
    # sq2  : M x 1
    # ------------
    # total : n*d + M*d + n*(M + T + 1) + 2*M*T + M
    avail_mem = max_mem / sizeof_dtype(dtype)
    rest_coef = 2 * M * T if v is not None else M * T
    n, d = select_dim_over_nd(max_n=N, max_d=D, coef_nd=1, coef_n=M + T + 1, coef_d=M,
                              rest=rest_coef + M, max_mem=avail_mem)
    ddev = torch.device('cuda:%d' % int(device_id))
    s1 = tcd.current_stream(ddev)
    s2 = tcd.Stream(ddev)
    with tcd.device(ddev), tcd.stream(s1):
        # First collect necessary memory
        mem_needed = n * M + n * T + n + M
        if not cuda_inputs:
            mem_needed += n * d + M * d
            if v is not None:
                mem_needed += M * T
        if not out.is_cuda:
            mem_needed += M * T
        # Create flat tensor
        flat_gpu_tn = torch.empty(size=(mem_needed,), dtype=dtype, device=ddev)
        # Extract the sub-tensors
        flat_offset = 0
        if v is not None:
            if not cuda_inputs:
                v_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, T), other=v, offset=flat_offset)
                copy_to_device_noorder(M, T, v, 0, 0, v_gpu, 0, 0, s=s1)
            else:
                v_gpu = v
        K_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, M), other=X1, offset=flat_offset)
        Kv_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, T), other=X1, offset=flat_offset)
        if out.is_cuda:
            out_gpu = out
        else:
            out_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, T), other=out, offset=flat_offset)
        out_gpu.fill_(0.0)
        if not cuda_inputs:
            X1ss_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n, d), other=X1, offset=flat_offset)
            X2s_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M, d), other=X2, offset=flat_offset)
        sq1_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(n,), other=X1, offset=flat_offset)
        sq2_gpu, flat_offset = _extract_flat(flat_gpu_tn, size=(M,), other=X1, offset=flat_offset)

        for i in range(0, N, n):
            nb = min(N - i, n)

            cur_K_gpu = K_gpu[:nb]  # nb x M
            cur_K_gpu.fill_(0.0)

            for j in range(0, D, d):
                db = min(D - j, d)
                s1.synchronize()  # need that the add_(sq2_gpu.T) op is complete to avoid overwrite
                # Parallelize two matrix transfers
                with tcd.stream(s2):
                    if cuda_inputs:
                        cur_X2s_gpu = X2[:, j:j + db]
                    else:
                        cur_X2s_gpu = copy_to_device_noorder(M, db, X2, 0, j, X2s_gpu, 0, 0, s=s2)
                    torch.norm(cur_X2s_gpu, p=2, dim=1, keepdim=True, out=sq2_gpu).pow_(2)
                if cuda_inputs:
                    cur_X1ss_gpu = X1[i:i + nb, j:j + db]
                else:
                    cur_X1ss_gpu = copy_to_device_noorder(nb, db, X1, i, j, X1ss_gpu, 0, 0, s=s1)
                torch.norm(cur_X1ss_gpu, p=2, dim=1, keepdim=True, out=sq1_gpu).pow_(2)

                s1.wait_stream(s2)  # need that cur_X2s_gpu and sq2_gpu are available.
                cur_K_gpu.addmm_(mat1=cur_X1ss_gpu, mat2=cur_X2s_gpu.T, alpha=-2.0)
                cur_K_gpu.add_(sq1_gpu)
                cur_K_gpu.add_(sq2_gpu.T)
                cur_K_gpu.clamp_min_(0)

            cur_K_gpu = kernel._transform(cur_K_gpu)

            if w is not None:
                cur_Kv_gpu = copy_to_device_noorder(nb, T, w, i, 0, Kv_gpu, 0, 0, s=s1)  # n x T
                if v is not None:
                    cur_Kv_gpu.addmm_(cur_K_gpu, v_gpu)
            else:
                # v cannot be None if w is None
                cur_Kv_gpu = Kv_gpu.narrow(0, 0, nb)  # n x T
                torch.mm(cur_K_gpu, v_gpu, out=cur_Kv_gpu)  # n x T

            # Multiply transposed kernel with the Kv result.
            out_gpu.addmm_(cur_K_gpu.T, cur_Kv_gpu)  # M x T

        if not out.is_cuda:
            copy_to_host_noorder(M, T, out_gpu, 0, 0, out, 0, 0, s=s1)
        s1.synchronize()
    return out


def fmmv_cuda(X1: torch.Tensor,
              X2: torch.Tensor,
              v: torch.Tensor,
              kernel,
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
    device = X1.device

    N = X1.size(0)
    # Create output matrix
    if out is None:
        out = create_same_stride((N, v.size(1)), X1, v.dtype, device=device,
                                 pin_memory=device.type != 'cuda')
    out.fill_(0.0)

    if kernel.kernel_type == "l2distance" and kernel.name == "gaussian":
        target = distk_fmmv
    else:
        target = generic_fmmv

    gpu_info = _get_gpu_info(opt, slack=0.9)

    if device.type == 'cuda':
        single_gpu_info = [g for g in gpu_info if g.Id == device.index][0]
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel,
                        max_mem=single_gpu_info.usable_ram)
        _call_direct(target, (args, device.index))
    else:
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        # Create queues
        args = []  # Arguments passed to each subprocess
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            args.append((ArgsFmmv(
                X1=X1.narrow(0, block_sizes[i], bwidth),
                X2=X2, v=v,
                out=out.narrow(0, block_sizes[i], bwidth),
                kernel=kernel, max_mem=g.usable_ram), g.Id))

        _start_wait_processes(target, args)
    return out


def fmmv_cuda_sparse(X1: SparseTensor,
                     X2: SparseTensor,
                     v: torch.Tensor,
                     kernel,
                     out: Optional[torch.Tensor] = None,
                     opt: Optional[BaseOptions] = None) -> torch.Tensor:
    opt = _setup_opt(opt)
    _check_contiguity((v, 'v'), (out, 'out'))
    device = X1.device

    N = X1.size(0)
    # Create output matrix
    if out is None:
        out = create_fortran((N, v.shape[1]), X1.dtype, device, pin_memory=device.type != 'cuda')
    out.fill_(0.0)

    gpu_info = _get_gpu_info(opt, slack=0.9)

    if device.type == 'cuda':
        single_gpu_info = [g for g in gpu_info if g.Id == device.index][0]
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel,
                        max_mem=single_gpu_info.usable_ram)
        _call_direct(sparse_fmmv, (args, device.index))
    else:
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        # Create queues
        args = []  # Arguments passed to each subprocess
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            args.append((ArgsFmmv(
                X1=X1.narrow_rows(block_sizes[i], bwidth),
                X2=X2, v=v,
                out=out.narrow(0, block_sizes[i], bwidth),
                kernel=kernel, max_mem=g.usable_ram), g.Id))

        _start_wait_processes(sparse_fmmv, args)
    return out


def fdmmv_cuda(X1: torch.Tensor,
               X2: torch.Tensor,
               v: Optional[torch.Tensor],
               w: Optional[torch.Tensor],
               kernel,
               out: Optional[torch.Tensor] = None,
               opt: Optional[BaseOptions] = None) -> torch.Tensor:
    """
    X1 : N x D
    X2 : M x D
    v  : M x T
    w  : N x T

    performs fnc(X1*X2', X1, X2)' * ( fnc(X1*X2', X1, X2) * v  +  w )  : M x T
    in blocks on multiple GPUs

    Assume all inputs have the same data type
    """
    opt = _setup_opt(opt)
    _check_contiguity((X1, 'X1'), (X2, 'X2'), (v, 'v'), (w, 'w'), (out, 'out'))
    device = X1.device
    if v is None and w is None:
        raise ValueError("one of 'v' or 'w' must not be None.")

    T = v.shape[1] if v is not None else w.shape[1]
    M = X2.shape[0]
    N = X1.shape[0]

    if out is None:
        out = create_same_stride((M, T), X1, X1.dtype, device=device,
                                 pin_memory=device.type != 'cuda')

    gpu_info = _get_gpu_info(opt, slack=0.9)

    if kernel.kernel_type == "l2distance" and kernel.name == "gaussian":
        target = distk_fdmmv
    else:
        target = generic_fdmmv

    if device.type == 'cuda':
        single_gpu_info = [g for g in gpu_info if g.Id == device.index][0]
        args = ArgsFdmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel,
                         max_mem=single_gpu_info.usable_ram)
        _call_direct(target, (args, device.index))
    else:
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        wrlk = []  # outputs for each subprocess.
        args = []
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue

            cur_out_gpu = create_same_stride((M, T), X1, X1.dtype, f'cuda:{gpu_info[i].Id}')  # M x T
            wrlk.append(cur_out_gpu)

            cur_w = None
            if w is not None:
                cur_w = w.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFdmmv(X1=X1.narrow(0, block_sizes[i], bwidth),
                                   X2=X2, v=v, w=cur_w, out=cur_out_gpu,
                                   kernel=kernel, max_mem=g.usable_ram), g.Id))
        _start_wait_processes(target, args)
        if len(wrlk) > 1:
            # noinspection PyTypeChecker
            fastest_device: int = np.argmax([d.speed for d in gpu_info])
            out.copy_(
                torch.cuda.comm.reduce_add(
                    wrlk, destination=gpu_info[fastest_device].Id))
        else:
            out.copy_(wrlk[0])
    return out


def fdmmv_cuda_sparse(X1: SparseTensor,
                      X2: SparseTensor,
                      v: Optional[torch.Tensor],
                      w: Optional[torch.Tensor],
                      kernel,
                      out: Optional[torch.Tensor] = None,
                      opt: Optional[BaseOptions] = None) -> torch.Tensor:
    opt = _setup_opt(opt)
    _check_contiguity((v, 'v'), (w, 'w'), (out, 'out'))
    device = X1.device
    if v is None and w is None:
        raise ValueError("one of 'v' or 'w' must not be None.")
    T = v.shape[1] if v is not None else w.shape[1]
    M = X2.size(0)
    N = X1.size(0)
    # Create output matrix
    if out is None:
        out = create_C((M, T), X1.dtype, device, pin_memory=device.type != 'cuda')

    gpu_info = _get_gpu_info(opt, slack=0.95)

    if device.type == 'cuda':
        sync_current_stream(device)
        single_gpu_info = [g for g in gpu_info if g.Id == device.index][0]
        args = ArgsFdmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel,
                         max_mem=single_gpu_info.usable_ram)
        _call_direct(sparse_fdmmv, (args, device.index))
    else:
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        wrlk = []  # outputs for each subprocess.
        args = []
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            cur_out_gpu = create_C((M, T), X1.dtype, f'cuda:{gpu_info[i].Id}')  # M x T
            wrlk.append(cur_out_gpu)

            cur_w = None

            if w is not None:
                cur_w = w.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFdmmv(
                X1=X1.narrow_rows(block_sizes[i], bwidth),
                X2=X2, v=v, w=cur_w, out=cur_out_gpu,
                kernel=kernel, max_mem=g.usable_ram), g.Id))
        _start_wait_processes(sparse_fdmmv, args)
        if len(wrlk) > 1:
            # noinspection PyTypeChecker
            fastest_device: int = np.argmax([d.speed for d in gpu_info])
            out.copy_(
                torch.cuda.comm.reduce_add(
                    wrlk, destination=gpu_info[fastest_device].Id))
        else:
            out.copy_(wrlk[0])
    return out
