import math
import threading
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from falkon.c_ext import (
    cublas_2d_copy_to_host_async,
    cublas_gemm,
    cublas_syrk,
    cublas_trmm,
    cuda_2d_copy_async,
    lauum_cuda,
)
from falkon.utils.device_copy import copy
from falkon.utils.helpers import sizeof_dtype
from falkon.utils.stream_utils import sync_current_stream
from falkon.utils.tensor_helpers import create_fortran, extract_fortran, extract_same_stride

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


def _extract_flat(flat_tn, size, other, offset):
    struct_tn = extract_same_stride(flat_tn, size=size, other=other, offset=offset)
    offset += np.prod(struct_tn.shape)
    return struct_tn, offset


def par_lauum_f_lower(
    A: torch.Tensor,
    block_allocs: List[BlockAlloc],
    my_rows: List[int],
    barrier: threading.Barrier,
    device_id: int,
    independent_output: bool,
):
    N = A.shape[0]
    is_cuda = A.device.type == "cuda"

    tc_device = torch.device(f"cuda:{device_id}")
    s1 = torch.cuda.Stream(device=tc_device)
    s3 = torch.cuda.Stream(device=tc_device)

    max_block_size = max(ba.length for ba in block_allocs)
    my_rows = sorted(my_rows)

    sync_current_stream(tc_device)
    with torch.cuda.device(tc_device), torch.cuda.stream(s1), torch.inference_mode():
        # Pre allocate b-col, syrk-out, lauum-out
        mem_needed = N * max_block_size + 2 * (max_block_size**2)
        if not is_cuda:
            # Also pre alloc r-col
            mem_needed += N * max_block_size
        f_gpu = torch.empty(size=(mem_needed,), dtype=A.dtype, device=tc_device)
        f_offset = 0
        whole_col_b, f_offset = _extract_flat(f_gpu, (N, max_block_size), other=A, offset=f_offset)
        syrk_out, f_offset = _extract_flat(f_gpu, (max_block_size, max_block_size), other=A, offset=f_offset)
        lauum_out, f_offset = _extract_flat(f_gpu, (max_block_size, max_block_size), other=A, offset=f_offset)
        if not is_cuda:
            temp_bb = create_fortran((max_block_size, max_block_size), A.dtype, "cpu", pin_memory=True)
            whole_col_r, f_offset = _extract_flat(f_gpu, (N, max_block_size), other=A, offset=f_offset)
        syrk_out.fill_(0.0)  # Only needed at the start, since SYRK always only touches one triangle.

        for b in range(len(block_allocs)):
            bb = block_allocs[b]
            # Load col b.
            # Instead of loading the whole column only load the last rows
            # as necessary by inspecting the minimum value in my_rows which is >= b.
            try:
                min_row = min([r for r in my_rows if r >= b])
                b_start = block_allocs[min_row].start
                col_b = copy(A[b_start:N, bb.start : bb.end], whole_col_b[b_start:, : bb.length], non_blocking=True)
            except ValueError:
                pass  # No column here
            if not independent_output:
                # wait for copy to device to succeed. After barrier other threads may modify
                # the part of col_b which we need!
                s1.synchronize()
                barrier.wait()

            for r in my_rows:
                if r == b:
                    # SYRK on col_b[bb.length:, :] with output into syrk_out[:bb.length, :bb.length]
                    # C = beta*C + alpha * op(A) @ op(A).T
                    if b_start + bb.length < N:
                        cur_syrk_out = syrk_out[: bb.length, : bb.length]
                        cublas_syrk(
                            A=col_b[bb.length :, :],
                            lda=col_b.stride(1),
                            alpha=1.0,
                            C=cur_syrk_out,
                            ldc=syrk_out.stride(1),
                            beta=0.0,
                            upper=False,
                            transpose=True,
                            n=bb.length,
                            k=col_b.shape[0] - bb.length,
                        )

                    with torch.cuda.stream(s3):
                        if independent_output:
                            s1.synchronize()  # we need col_b to be loaded
                        cur_lauum_out = lauum_out[: bb.length, : bb.length]
                        # Note that col_b[:bb.length, :bb.length] == Abb
                        if independent_output:
                            # In the independent output case we need to preserve tril(Abb) instead!
                            cur_lauum_out.copy_(col_b[: bb.length, : bb.length].T)
                        else:
                            # In normal case we need triu(Abb) to be preserved in the upper triangle of lauum_out
                            cur_lauum_out.copy_(col_b[: bb.length, : bb.length])

                        # LAUUM on col_b[:bb.length, :bb.length], into cur_lauum_out[:bb.length, :bb.length]
                        # Here the lower-part of cur_lauum_out is over-written
                        lauum_cuda(
                            n=bb.length,
                            A=col_b[: bb.length, : bb.length],
                            lda=col_b.stride(1),
                            B=cur_lauum_out,
                            ldb=max_block_size,
                            lower=True,
                        )
                    s1.wait_stream(s3)  # all subsequent work will need cur_lauum_out

                    # Add outputs of SYRK and LAUUM (only if SYRK was performed)
                    if b_start + bb.length < N:
                        cur_lauum_out.add_(cur_syrk_out)

                    # Copy lauum_out into the original matrix, while preserving the other side
                    # of the triangular matrix. This depends on the `independent_output` flag.
                    Abb = A[bb.start : bb.end, bb.start : bb.end]
                    if independent_output:
                        # cuda and non-cuda cases, since we have different orderings (this will require extra buffer)
                        Abb.copy_(cur_lauum_out.T)
                    else:
                        copy(cur_lauum_out, Abb, non_blocking=True)
                elif r > b:
                    br = block_allocs[r]

                    # Load column r. Since r > b this column will be shorter than column b
                    if is_cuda:  # If col_r is already in GPU no copy needed.
                        col_r = A[br.start : N, br.start : br.end]
                    else:
                        col_r = copy(
                            A[br.start : N, br.start : br.end],
                            whole_col_r[: N - br.start, : br.length],
                            non_blocking=True,
                        )
                    # Restrict column b to only the last 'r' rows
                    ccb = col_b[br.start - b_start :, :]

                    # TRMM on g_r[0:br.length, :] which is triangular (r*r)
                    #         and cur_g_b[0:br.length, :]
                    #         output is a r*b matrix stored in the first rows of ccb
                    # C = alpha * op(A) @ B -- A triangular
                    cublas_trmm(
                        A=col_r,
                        lda=col_r.stride(1),
                        alpha=1.0,
                        B=ccb,
                        ldb=ccb.stride(1),
                        C=ccb,
                        ldc=ccb.stride(1),
                        left=True,
                        upper=False,
                        transpose=True,
                        unitriangular=False,
                        m=br.length,
                        n=bb.length,
                    )

                    # GEMM on g_r[br.length:, :].T and cur_g_b[bb.length:, :]
                    #         output  is the same r*b matrix as before, outputs need to be summed.
                    # C = alpha * op(A) @ op(B) + beta * C
                    if br.end < N:
                        cublas_gemm(
                            A=col_r[br.length :, :],
                            lda=col_r.stride(1),
                            alpha=1.0,  # A is k * m
                            B=ccb[br.length :, :],
                            ldb=ccb.stride(1),  # B is k * n
                            C=ccb,
                            ldc=ccb.stride(1),
                            beta=1.0,  # C is m * n
                            transa=True,
                            transb=False,
                            m=br.length,
                            n=bb.length,
                            k=col_r.shape[0] - br.length,
                        )
                    # Copy back to A[r, b]
                    if independent_output:
                        if is_cuda:
                            A[bb.start : bb.end, br.start : br.end].copy_(ccb[: br.length, : bb.length].T)
                        else:
                            _temp_cpu = copy(
                                ccb[: br.length, : bb.length], temp_bb[: br.length, : bb.length], non_blocking=True
                            )
                            s1.synchronize()  # must wait for data to be onto CPU.
                            A[bb.start : bb.end, br.start : br.end].copy_(_temp_cpu.T)
                    else:
                        copy(ccb[: br.length, : bb.length], A[br.start : br.end, bb.start : bb.end], non_blocking=True)
            s1.synchronize()


def par_lauum_c_lower(
    A: torch.Tensor,
    block_allocs: List[BlockAlloc],
    my_rows: List[int],
    barrier: threading.Barrier,
    device_id: int,
    independent_output: bool,
):
    N = A.shape[0]
    dts = sizeof_dtype(A.dtype)
    is_cuda = A.device.type == "cuda"

    tc_device = torch.device(f"cuda:{device_id}")
    s1 = torch.cuda.Stream(device=tc_device)
    s3 = torch.cuda.Stream(device=tc_device)

    max_block_size = max(ba.length for ba in block_allocs)
    my_rows = sorted(my_rows)

    sync_current_stream(tc_device)
    with torch.cuda.device(tc_device), torch.cuda.stream(s1), torch.inference_mode():
        if not is_cuda:
            temp_bb = create_fortran((max_block_size, max_block_size), A.dtype, "cpu", pin_memory=True).T
        # Pre allocate r-col, b-col, syrk-out, lauum-out
        mem_needed = 2 * N * max_block_size + 2 * (max_block_size**2)
        f_gpu = torch.empty(size=(mem_needed,), dtype=A.dtype, device=tc_device)
        whole_col_b = f_gpu[: N * max_block_size]
        whole_col_r = f_gpu[N * max_block_size : 2 * N * max_block_size]
        syrk_out = extract_fortran(f_gpu, size=(max_block_size, max_block_size), offset=2 * N * max_block_size)
        lauum_out = extract_fortran(
            f_gpu, size=(max_block_size, max_block_size), offset=2 * N * max_block_size + max_block_size**2
        )
        syrk_out.fill_(0.0)

        for b in range(len(block_allocs)):
            bb = block_allocs[b]
            # Load col b.
            # Instead of loading the whole column only load the last rows
            # as necessary by inspecting the minimum value in my_rows which is >= b.
            try:
                min_row = min([r for r in my_rows if r >= b])
                b_start = block_allocs[min_row].start
                cuda_2d_copy_async(
                    src_tensor=A[b_start, bb.start],
                    src_pitch=A.shape[1] * dts,
                    dest_tensor=whole_col_b,
                    dest_pitch=max_block_size * dts,
                    width=bb.length * dts,
                    height=N - b_start,
                )
            except ValueError:  # all of `my_rows` are smaller than `b`.
                pass
            if not independent_output:
                # wait for copy to device to succeed. After barrier other threads may modify
                # the part of col_b which we need!
                s1.synchronize()
                barrier.wait()

            for r in my_rows:
                if r < b:
                    continue
                if r == b:
                    is_last_row = b_start + bb.length == N
                    # SYRK on g_b[bb.length:, :] with output replacing g_b[:bb.length, :]
                    # C = beta*C + alpha * op(A) @ op(A).T
                    if not is_last_row:
                        cublas_syrk(
                            A=whole_col_b[bb.length * max_block_size :],
                            lda=max_block_size,
                            alpha=1.0,
                            C=syrk_out,
                            ldc=max_block_size,
                            beta=0.0,
                            upper=True,
                            transpose=False,
                            n=bb.length,
                            k=N - b_start - bb.length,
                        )

                    with torch.cuda.stream(s3):
                        if independent_output:
                            s1.synchronize()  # we need col_b to be loaded
                        # Lower LAUUM for C-contig is equal to upper LAUUM for F-contig
                        c_lauum_in = whole_col_b[: bb.length * max_block_size].view(bb.length, max_block_size)[
                            :, : bb.length
                        ]
                        c_lauum_out = lauum_out[: bb.length, : bb.length]

                        if independent_output:
                            c_lauum_out.copy_(c_lauum_in)
                        else:
                            c_lauum_out.copy_(c_lauum_in.T)
                        lauum_cuda(
                            n=bb.length,
                            A=c_lauum_in,
                            lda=max_block_size,
                            B=c_lauum_out,
                            ldb=max_block_size,
                            lower=False,
                        )

                    s1.wait_stream(s3)  # all subsequent work on s1 will need cur_lauum_out
                    if not is_last_row:
                        c_lauum_out.add_(syrk_out[: bb.length, : bb.length])

                    # copy back whole_col_b into Abb
                    # Now lauum_out is F-contig, while Abb is C-contig
                    Abb = A[bb.start : bb.end, bb.start : bb.end]
                    if independent_output:
                        Abb.copy_(c_lauum_out)
                    else:
                        Abb.copy_(c_lauum_out.T)
                else:  # r > b
                    br = block_allocs[r]

                    # Load column r. Since r > b this column will be shorter than column b
                    cuda_2d_copy_async(
                        src_tensor=A[br.start, br.start],
                        src_pitch=A.shape[1] * dts,
                        dest_tensor=whole_col_r,
                        dest_pitch=max_block_size * dts,
                        width=br.length * dts,
                        height=N - br.start,
                    )
                    # Restrict column b to only the last 'r' rows
                    ccb = whole_col_b[(br.start - b_start) * max_block_size :]

                    # TRMM on g_r[0:br.length, :] which is triangular (r*r)
                    #         and cur_g_b[0:br.length, :]
                    #         output is a r*b matrix and stored in first rows of ccb
                    # C = alpha * op(A) @ B -- A triangular
                    cublas_trmm(
                        A=whole_col_r,
                        lda=max_block_size,
                        alpha=1.0,
                        B=ccb,
                        ldb=max_block_size,
                        C=ccb,
                        ldc=max_block_size,
                        left=False,
                        upper=True,
                        transpose=True,
                        unitriangular=False,
                        m=bb.length,
                        n=br.length,
                    )

                    # GEMM on g_r[br.length:, :].T and cur_g_b[bb.length:, :]
                    #         output  is the same r*b matrix as before, outputs need to be summed.
                    # C = alpha * op(A) @ op(B) + beta * C
                    if br.end < N:
                        cublas_gemm(
                            A=ccb[br.length * max_block_size :],
                            lda=max_block_size,
                            alpha=1.0,
                            B=whole_col_r[br.length * max_block_size :],
                            ldb=max_block_size,
                            C=ccb,
                            ldc=max_block_size,
                            beta=1.0,
                            transa=False,
                            transb=True,
                            m=bb.length,
                            n=br.length,
                            k=N - br.start - br.length,
                        )

                    # Copy back to A[r, b]
                    if is_cuda:
                        ccb_square = ccb[: max_block_size * br.length].view(br.length, max_block_size)
                        if independent_output:
                            A[bb.start : bb.end, br.start : br.end].copy_(ccb_square[: br.length, : bb.length].T)
                        else:
                            A[br.start : br.end, bb.start : bb.end].copy_(ccb_square[: br.length, : bb.length])
                    elif independent_output:
                        # Copy must be transposed, copy to temp_bb first.
                        cublas_2d_copy_to_host_async(
                            rows=bb.length,
                            cols=br.length,
                            elemSize=dts,
                            dev_tensor=ccb,
                            lda=max_block_size,
                            host_tensor=temp_bb,
                            ldb=max_block_size,
                        )
                        s1.synchronize()
                        A[bb.start : bb.end, br.start : br.end].copy_(temp_bb[: br.length, : bb.length].T)
                    else:
                        cublas_2d_copy_to_host_async(
                            rows=bb.length,
                            cols=br.length,
                            elemSize=dts,
                            dev_tensor=ccb,
                            lda=max_block_size,
                            host_tensor=A[br.start, bb.start],
                            ldb=A.shape[0],
                        )
            s1.synchronize()
