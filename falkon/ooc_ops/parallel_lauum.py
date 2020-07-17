import math
import threading
from dataclasses import dataclass
from typing import List

import scipy.linalg.lapack as scll
import torch
from falkon.utils.cyblas import zero_triang, copy_triang

from falkon.cuda.cublas_gpu import *
from falkon.cuda.cudart_gpu import cuda_memcpy2d_async
from falkon.utils.cuda_helpers import copy_to_device, copy_to_host
from falkon.utils.helpers import choose_fn, sizeof_dtype
from falkon.utils.tensor_helpers import create_fortran

__all__ = ("par_lauum_c_lower", "par_lauum_f_lower", "BlockAlloc")


@dataclass(frozen=True, eq=False, repr=True)
class BlockAlloc:
    start: int
    end: int
    length: int


def _rounddown(num, multiple):
    return int(math.floor(num / multiple) * multiple)


def _round_nb_size(size, multiple):
    if size > multiple:
        return _rounddown(size, multiple)
    else:
        return max(1, size)


def par_lauum_f_lower(A: torch.Tensor,
                      block_allocs: List[BlockAlloc],
                      my_rows: List[int],
                      barrier: threading.Barrier,
                      device_id: int,
                      cublas_handle,
                      independent_output: bool):
    N = A.shape[0]

    lauum_fn = choose_fn(A.dtype, scll.dlauum, scll.slauum, "Lapack LAUUM")
    trmm_fn = choose_fn(A.dtype, cublasDtrmm, cublasStrmm, "cuBlas TRMM")
    gemm_fn = choose_fn(A.dtype, cublasDgemm, cublasSgemm, "cuBlas GEMM")
    syrk_fn = choose_fn(A.dtype, cublasDsyrk, cublasSsyrk, "cuBlas SYRK")

    tc_device = torch.device('cuda:%d' % (device_id))
    s1 = torch.cuda.Stream(device=tc_device)
    s2 = torch.cuda.Stream(device=tc_device)
    cublasSetStream(cublas_handle, s1._as_parameter_)

    max_block_size = max(ba.length for ba in block_allocs)
    my_rows = sorted(my_rows)

    with torch.cuda.device(tc_device), torch.cuda.stream(s1):
        # Preallocate 2 columns
        whole_col_b = create_fortran((A.shape[0], max_block_size), A.dtype, tc_device)
        whole_col_r = create_fortran((A.shape[0], max_block_size), A.dtype, tc_device)
        temp_bb = create_fortran((max_block_size, max_block_size), A.dtype, 'cpu', pin_memory=True)

        for b in range(len(block_allocs)):
            bb = block_allocs[b]
            # Load col b.
            # Instead of loading the whole column only load the last rows
            # as necessary by inspecting the minimum value in my_rows which is >= b.
            try:
                min_row = min([r for r in my_rows if r >= b])
                b_start = block_allocs[min_row].start
                col_b = copy_to_device(N - b_start, bb.length, A, b_start, bb.start, whole_col_b, 0,
                                       0, s1)
            except ValueError:
                pass  # No column here
            if not independent_output:
                barrier.wait()

            for r in my_rows:
                if r < b:
                    continue
                if r == b:
                    # SYRK on g_b[bb.length:, :] with output replacing g_b[:bb.length, :]
                    # C = beta*C + alpha * op(A) @ op(A).T
                    if b_start + bb.length < N:
                        syrk_fn(cublas_handle,
                                uplo='L', trans='T',
                                n=bb.length, k=col_b.shape[0] - bb.length,
                                alpha=1.0, A=col_b[bb.length:, :].data_ptr(), lda=col_b.stride(1),
                                beta=0.0, C=col_b.data_ptr(), ldc=col_b.stride(1))
                    # CPU LAUUM on A[bb.start:bb.end, bb.start:bb.end]. This is a bit messy, should do cleanup.
                    Abb = A[bb.start:bb.end, bb.start:bb.end]  # L\U
                    if independent_output:
                        Abb_np = Abb.numpy().copy(order="F")
                        # Make symmetric: L\L
                        copy_triang(Abb_np, upper=False)
                        uu, info = lauum_fn(Abb_np, lower=1, overwrite_c=True)  # LAU\L
                        Abb.copy_(torch.from_numpy(uu.T))  # L\LAU
                    else:
                        uu, info = lauum_fn(Abb.numpy(), lower=1, overwrite_c=False)  # LAU\L
                        if b_start + bb.length < N:
                            zero_triang(uu, upper=True)
                        Abb.copy_(torch.from_numpy(uu))
                    if b_start + bb.length < N:
                        # It is IMPORTANT to do the copy on s1 and then sync it.
                        tbb = copy_to_host(bb.length, bb.length, col_b, 0, 0, temp_bb, 0, 0, s1)
                        s1.synchronize()
                        if independent_output:
                            Abb.add_(torch.triu(tbb.T))
                        else:
                            Abb.add_(tbb)
                else:  # r > b
                    br = block_allocs[r]

                    # Load column r. Since r > b this column will be shorter than column b
                    col_r = copy_to_device(N - br.start, br.length, A, br.start, br.start,
                                           whole_col_r, 0, 0, s1)
                    # Restrict column b to only the last 'r' rows
                    ccb = col_b[br.start - b_start:, :]

                    # TRMM on g_r[0:br.length, :] which is triangular (r*r)
                    #         and cur_g_b[0:br.length, :]
                    #         output is a r*b matrix and should be stored in a separate g_out block
                    # Could store output in the first rows of g_b
                    # C = alpha * op(A) @ B -- A triangular
                    trmm_fn(
                        handle=cublas_handle,
                        side='L', uplo='L', trans='T', diag='N',
                        m=br.length, n=bb.length,
                        alpha=1.0, A=col_r.data_ptr(), lda=col_r.stride(1),
                        B=ccb.data_ptr(), ldb=ccb.stride(1),
                        C=ccb.data_ptr(), ldc=ccb.stride(1))

                    # GEMM on g_r[br.length:, :].T and cur_g_b[bb.length:, :]
                    #         output  is the same r*b matrix as before, outputs need to be summed.
                    # C = alpha * op(A) @ op(B) + beta * C
                    if br.end < N:
                        gemm_fn(handle=cublas_handle,
                                transa='T', transb='N',
                                m=br.length, n=bb.length, k=col_r.shape[0] - br.length,
                                alpha=1.0, A=col_r[br.length:, :].data_ptr(), lda=col_r.stride(1),
                                B=ccb[br.length:, :].data_ptr(), ldb=ccb.stride(1),
                                beta=1.0, C=ccb.data_ptr(), ldc=ccb.stride(1))
                    # Copy back to A[r, b]
                    if independent_output:
                        _temp_cpu = copy_to_host(br.length, bb.length, ccb, 0, 0, temp_bb, 0, 0, s1)
                        s1.synchronize()
                        A[bb.start:bb.end, br.start:br.end].copy_(_temp_cpu.T)
                    else:
                        s1.synchronize()
                        copy_to_host(br.length, bb.length, ccb, 0, 0, A, br.start, bb.start, s2)
            s2.synchronize()


def par_lauum_c_lower(A: torch.Tensor,
                      block_allocs: List[BlockAlloc],
                      my_rows: List[int],
                      barrier: threading.Barrier,
                      device_id: int,
                      cublas_handle,
                      independent_output: bool):
    N = A.shape[0]
    dts = sizeof_dtype(A.dtype)

    lauum_fn = choose_fn(A.dtype, scll.dlauum, scll.slauum, "Lapack LAUUM")
    trmm_fn = choose_fn(A.dtype, cublasDtrmm, cublasStrmm, "cuBlas TRMM")
    gemm_fn = choose_fn(A.dtype, cublasDgemm, cublasSgemm, "cuBlas GEMM")
    syrk_fn = choose_fn(A.dtype, cublasDsyrk, cublasSsyrk, "cuBlas SYRK")

    tc_device = torch.device('cuda:%d' % (device_id))
    s1 = torch.cuda.Stream(device=tc_device)
    s2 = torch.cuda.Stream(device=tc_device)
    s1_cuda, s2_cuda = s1._as_parameter_, s2._as_parameter_
    cublasSetStream(cublas_handle, s1_cuda)

    max_block_size = max(ba.length for ba in block_allocs)
    my_rows = sorted(my_rows)

    with torch.cuda.device(tc_device), torch.cuda.stream(s1):
        # Preallocate 2 block-columns. The single block is a CPU buffer
        whole_col_b = create_fortran((A.shape[0] * max_block_size,), A.dtype, tc_device)
        whole_col_r = create_fortran((A.shape[0] * max_block_size,), A.dtype, tc_device)
        temp_bb = create_fortran((max_block_size, max_block_size), A.dtype, 'cpu',
                                 pin_memory=True).T
        for b in range(len(block_allocs)):
            bb = block_allocs[b]
            # Load col b.
            # Instead of loading the whole column only load the last rows
            # as necessary by inspecting the minimum value in my_rows which is >= b.
            try:
                min_row = min([r for r in my_rows if r >= b])
                b_start = block_allocs[min_row].start
                cuda_memcpy2d_async(
                    dst=whole_col_b.data_ptr(), dpitch=max_block_size * dts,
                    src=A[b_start, bb.start].data_ptr(), spitch=A.shape[1] * dts,
                    width=bb.length * dts, height=N - b_start, stream=s1_cuda)
            except ValueError:
                # all of `my_rows` are smaller than `b`.
                pass
            if not independent_output:
                barrier.wait()

            for r in my_rows:
                if r < b:
                    continue
                if r == b:
                    is_last_row = b_start + bb.length == N
                    # SYRK on g_b[bb.length:, :] with output replacing g_b[:bb.length, :]
                    # C = beta*C + alpha * op(A) @ op(A).T
                    if not is_last_row:
                        syrk_fn(cublas_handle, uplo='U', trans='N',
                                n=bb.length, k=N - b_start - bb.length,
                                alpha=1.0, A=whole_col_b[bb.length * max_block_size:].data_ptr(),
                                lda=max_block_size,
                                beta=0.0, C=whole_col_b.data_ptr(), ldc=max_block_size)
                    # Run LAUUM on CPU on Abb.T (transpose because LAPACK works in F-order)
                    # Result will be on upper(uu). So if we copy back to lower(A), we must copy
                    # back uu.T -- otherwise we should copy back uu directly.
                    Abb = A[bb.start:bb.end, bb.start:bb.end]
                    if independent_output:
                        Abb_np = Abb.T.numpy().copy(order="F")  # U\L
                        copy_triang(Abb_np, upper=True)  # L\L
                        uu, info = lauum_fn(Abb_np, lower=1, overwrite_c=True)  # LAU\L
                        Abb.copy_(torch.from_numpy(uu.T))  # L \ LAU
                    else:
                        uu, info = lauum_fn(Abb.T.numpy(), lower=0, overwrite_c=False)
                        # Zeroing must happen if the SYRK output is to be added: otherwise the
                        # non-processed part of Abb (i.e. upper(Abb) if not independent_output)
                        # will be multiplied by 2.
                        if not is_last_row:
                            zero_triang(uu, upper=False)
                        Abb.copy_(torch.from_numpy(uu.T))

                    if not is_last_row:
                        cuda_memcpy2d_async(
                            dst=temp_bb.data_ptr(), dpitch=max_block_size * dts,
                            src=whole_col_b.data_ptr(), spitch=max_block_size * dts,
                            width=bb.length * dts, height=bb.length, stream=s1_cuda)
                        s1.synchronize()  # TODO: Check if failure when this commented out.
                        if independent_output:
                            Abb.add_(torch.triu(temp_bb[:bb.length, :bb.length].T))
                        else:
                            Abb.add_(temp_bb[:bb.length, :bb.length])
                else:  # r > b
                    br = block_allocs[r]

                    # Load column r. Since r > b this column will be shorter than column b
                    cuda_memcpy2d_async(
                        dst=whole_col_r.data_ptr(), dpitch=max_block_size * dts,
                        src=A[br.start, br.start].data_ptr(), spitch=A.shape[1] * dts,
                        width=br.length * dts, height=N - br.start, stream=s1_cuda)
                    # s1.synchronize()
                    # Restrict column b to only the last 'r' rows
                    ccb = whole_col_b[(br.start - b_start) * max_block_size:]

                    # TRMM on g_r[0:br.length, :] which is triangular (r*r)
                    #         and cur_g_b[0:br.length, :]
                    #         output is a r*b matrix and should be stored in a separate g_out block
                    # Could store output in the first rows of g_b
                    # C = alpha * op(A) @ B -- A triangular
                    trmm_fn(
                        handle=cublas_handle,
                        side='R', uplo='U', trans='T', diag='N',
                        m=bb.length, n=br.length,
                        alpha=1.0, A=whole_col_r.data_ptr(), lda=max_block_size,
                        B=ccb.data_ptr(), ldb=max_block_size,
                        C=ccb.data_ptr(), ldc=max_block_size)

                    # GEMM on g_r[br.length:, :].T and cur_g_b[bb.length:, :]
                    #         output  is the same r*b matrix as before, outputs need to be summed.
                    # C = alpha * op(A) @ op(B) + beta * C
                    if br.end < N:
                        gemm_fn(handle=cublas_handle, transa='N', transb='T',
                                m=bb.length, n=br.length, k=N - br.start - br.length,
                                alpha=1.0,
                                A=ccb[br.length * max_block_size:].data_ptr(),
                                lda=max_block_size,
                                B=whole_col_r[br.length * max_block_size:].data_ptr(),
                                ldb=max_block_size,
                                beta=1.0, C=ccb.data_ptr(), ldc=max_block_size)

                    # Copy back to A[r, b]
                    if independent_output:
                        # Copy must be transposed, copy to temp_bb first.
                        cublasGetMatrixAsync(
                            rows=bb.length, cols=br.length, elem_size=dts,
                            A=ccb.data_ptr(), lda=max_block_size,
                            B=temp_bb.data_ptr(), ldb=max_block_size, stream=s1_cuda)
                        s1.synchronize()
                        A[bb.start:bb.end, br.start:br.end].copy_(temp_bb[:br.length, :bb.length].T)
                    else:
                        s1.synchronize()
                        cublasGetMatrixAsync(
                            rows=bb.length, cols=br.length, elem_size=dts,
                            A=ccb.data_ptr(), lda=max_block_size,
                            B=A[br.start, bb.start].data_ptr(), ldb=A.shape[0],
                            stream=s2_cuda)
            s2.synchronize()
