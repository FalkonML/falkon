import math
import threading
from dataclasses import dataclass
from typing import List

import scipy.linalg.lapack as scll
import torch

from falkon.cuda.cublas_gpu import *
from falkon.cuda.cudart_gpu import cuda_memcpy2d_async
from falkon.utils.cuda_helpers import copy_to_device, copy_to_host
from falkon.utils.helpers import choose_fn, sizeof_dtype
from falkon.utils.tensor_helpers import create_fortran
from falkon.la_helpers import zero_triang, copy_triang
from falkon.ooc_ops.cuda import cuda_lauum_lower
from falkon.la_helpers.cuda_la_helpers import cuda_transpose


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
    is_cuda = A.device.type == "cuda"

    trmm_fn = choose_fn(A.dtype, cublasDtrmm, cublasStrmm, "cuBlas TRMM")
    gemm_fn = choose_fn(A.dtype, cublasDgemm, cublasSgemm, "cuBlas GEMM")
    syrk_fn = choose_fn(A.dtype, cublasDsyrk, cublasSsyrk, "cuBlas SYRK")

    tc_device = torch.device('cuda:%d' % (device_id))
    s1 = torch.cuda.Stream(device=tc_device)
    s2 = torch.cuda.Stream(device=tc_device)
    s3 = torch.cuda.Stream(device=tc_device)
    cublasSetStream(cublas_handle, s1._as_parameter_)

    max_block_size = max(ba.length for ba in block_allocs)
    my_rows = sorted(my_rows)

    with torch.cuda.device(tc_device), torch.cuda.stream(s1):
        # Preallocate 2 columns
        if not is_cuda:
            whole_col_b = create_fortran((A.shape[0], max_block_size), A.dtype, tc_device)
            whole_col_r = create_fortran((A.shape[0], max_block_size), A.dtype, tc_device)
        syrk_out = create_fortran((max_block_size, max_block_size), A.dtype, tc_device)
        lauum_out = create_fortran((max_block_size, max_block_size), A.dtype, tc_device)
        temp_bb = create_fortran((max_block_size, max_block_size), A.dtype, 'cpu', pin_memory=True)

        for b in range(len(block_allocs)):
            bb = block_allocs[b]
            # Load col b.
            # Instead of loading the whole column only load the last rows
            # as necessary by inspecting the minimum value in my_rows which is >= b.
            try:
                min_row = min([r for r in my_rows if r >= b])
                b_start = block_allocs[min_row].start
                if is_cuda:
                    col_b: torch.Tensor = A[b_start:N, bb.start:bb.end]
                else:
                    col_b: torch.Tensor = copy_to_device(
                        N - b_start, bb.length, A, b_start, bb.start, whole_col_b, 0, 0, s1)
            except ValueError:
                pass  # No column here
            if not independent_output:
                barrier.wait()

            for r in my_rows:
                if r == b:
                    # SYRK on col_b[bb.length:, :] with output into syrk_out[:bb.length, :bb.length]
                    # C = beta*C + alpha * op(A) @ op(A).T
                    if b_start + bb.length < N:
                        cur_syrk_out = syrk_out[:bb.length, :bb.length]
                        syrk_fn(cublas_handle, uplo='L', trans='T',
                                n=bb.length, k=col_b.shape[0] - bb.length,
                                alpha=1.0, A=col_b[bb.length:, :].data_ptr(), lda=col_b.stride(1),
                                beta=0.0, C=cur_syrk_out.data_ptr(), ldc=syrk_out.stride(1))

                    with torch.cuda.stream(s3):
                        cur_lauum_out = lauum_out[:bb.length, :bb.length]
                        # Note that col_b[:bb.length, :bb.length] == Abb
                        if independent_output:
                            # In the independent output case we need to preserve tril(Abb) instead!
                            cur_lauum_out.copy_(col_b[:bb.length, :bb.length].T)
                        else:
                            # In normal case we need triu(Abb) to be preserved in the upper triangle of lauum_out
                            cur_lauum_out.copy_(col_b[:bb.length, :bb.length])

                        # LAUUM on col_b[:bb.length, :bb.length], into lauum_out[:bb.length, :bb.length]
                        cuda_lauum_lower(n=bb.length, A=col_b[:bb.length, :bb.length], lda=A.shape[0], B=cur_lauum_out, ldb=max_block_size)
                    s3.synchronize()

                    # Add outputs of SYRK and LAUUM (only if SYRK was performed)
                    if b_start + bb.length < N:
                        s1.synchronize()
                        cur_lauum_out.add_(cur_syrk_out)

                    # Copy lauum_out into the original matrix, while preserving the other side
                    # of the triangular matrix. This depends on the `independent_output` flag.
                    Abb = A[bb.start:bb.end, bb.start:bb.end]
                    if independent_output:
                        Abb.copy_(cur_lauum_out.T)
                    else:
                        Abb.copy_(cur_lauum_out)
                elif r > b:
                    br = block_allocs[r]

                    # Load column r. Since r > b this column will be shorter than column b
                    if is_cuda:
                        col_r = A[br.start:N, br.start:br.end]
                    else:
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
                        if is_cuda:
                            A[bb.start:bb.end, br.start:br.end].copy_(ccb[:br.length, :bb.length].T)
                        else:
                            _temp_cpu = copy_to_host(br.length, bb.length, ccb, 0, 0, temp_bb, 0, 0, s1)
                            s1.synchronize()
                            A[bb.start:bb.end, br.start:br.end].copy_(_temp_cpu.T)
                    elif not is_cuda:
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
    is_cuda = A.device.type == "cuda"

    lauum_fn = choose_fn(A.dtype, scll.dlauum, scll.slauum, "Lapack LAUUM")
    trmm_fn = choose_fn(A.dtype, cublasDtrmm, cublasStrmm, "cuBlas TRMM")
    gemm_fn = choose_fn(A.dtype, cublasDgemm, cublasSgemm, "cuBlas GEMM")
    syrk_fn = choose_fn(A.dtype, cublasDsyrk, cublasSsyrk, "cuBlas SYRK")

    tc_device = torch.device('cuda:%d' % (device_id))
    s1 = torch.cuda.Stream(device=tc_device)
    s2 = torch.cuda.Stream(device=tc_device)
    s3 = torch.cuda.Stream(device=tc_device)
    s1_cuda, s2_cuda = s1._as_parameter_, s2._as_parameter_
    cublasSetStream(cublas_handle, s1_cuda)

    max_block_size = max(ba.length for ba in block_allocs)
    my_rows = sorted(my_rows)

    with torch.cuda.device(tc_device), torch.cuda.stream(s1):
        # Preallocate 2 block-columns. The single block is a CPU buffer
        whole_col_b = create_fortran((A.shape[0] * max_block_size,), A.dtype, tc_device)
        whole_col_r = create_fortran((A.shape[0] * max_block_size,), A.dtype, tc_device)
        syrk_out = create_fortran((max_block_size, max_block_size), A.dtype, tc_device)
        lauum_in = create_fortran((max_block_size, max_block_size), A.dtype, tc_device)
        temp_bb = create_fortran((max_block_size, max_block_size), A.dtype, 'cpu', pin_memory=True).T

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
            except ValueError:  # all of `my_rows` are smaller than `b`.
                pass
            if not independent_output:
                barrier.wait()

            for r in my_rows:
                if r < b:
                    continue
                if r == b:
                    is_last_row = b_start + bb.length == N
                    # Sync the load of whole_col_b
                    s1.synchronize()
                    # SYRK on g_b[bb.length:, :] with output replacing g_b[:bb.length, :]
                    # C = beta*C + alpha * op(A) @ op(A).T
                    if not is_last_row:
                        syrk_fn(cublas_handle, uplo='U', trans='N',
                                n=bb.length, k=N - b_start - bb.length,
                                alpha=1.0, A=whole_col_b[bb.length * max_block_size:].data_ptr(),
                                lda=max_block_size,
                                beta=0.0, C=syrk_out.data_ptr(), ldc=max_block_size)

                    with torch.cuda.stream(s3):
                        #lauum_out = whole_col_b[:max_block_size * max_block_size].view(max_block_size, max_block_size)
                        lauum_out = whole_col_b[:bb.length * max_block_size].view(bb.length, max_block_size)[:, :bb.length]
                        # With the copy we go from C-contig to F-contig into lauum_in. This also transposes lauum_out so we get a correct order.
                        cur_lauum_in = lauum_in[:bb.length, :bb.length]
                        cur_lauum_in.copy_(lauum_out)
                        # Since lauum_out is supposed to also be F-contig, we must do another copy from lauum_in to lauum_out.
                        if independent_output:
                            lauum_out.copy_(cur_lauum_in)
                        else:
                            lauum_out.copy_(cur_lauum_in.T)
                        cuda_lauum_lower(n=bb.length, A=cur_lauum_in, lda=max_block_size, B=lauum_out, ldb=max_block_size)

                    s3.synchronize()
                    if not is_last_row:
                        s1.synchronize()
                        lauum_out.add_(syrk_out[:bb.length, :bb.length])

                    # copy back whole_col_b into Abb
                    # Now lauum_out is F-contig, while Abb is C-contig
                    Abb = A[bb.start:bb.end, bb.start:bb.end]
                    if independent_output:
                        Abb.copy_(lauum_out)
                    else:
                        Abb.copy_(lauum_out.T)
                else:  # r > b
                    br = block_allocs[r]

                    # Load column r. Since r > b this column will be shorter than column b
                    cuda_memcpy2d_async(
                        dst=whole_col_r.data_ptr(), dpitch=max_block_size * dts,
                        src=A[br.start, br.start].data_ptr(), spitch=A.shape[1] * dts,
                        width=br.length * dts, height=N - br.start, stream=s1_cuda)
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
