import torch

from falkon.c_ext import (
    cublas_2d_copy_to_host_async,
    cublas_2d_copy_to_dev_async,
    cublas_gemm,
    cublas_syrk,
    cublas_trmm,
    cuda_2d_copy_async,
    lauum_cuda,
)
from falkon.ooc_ops.parallel_lauum import BlockAlloc
from falkon.utils.tensor_helpers import create_fortran, extract_fortran, is_contig, is_f_contig


def ooc_lauum_single_gpu(
    L: torch.Tensor,
    block_allocs: list[BlockAlloc],
):
    # rows are contiguous
    N = L.shape[0]
    dts = L.element_size()

    contig = None
    if is_f_contig(L):
        contig = "F"
    elif is_contig(L):
        contig = "C"
    else:
        raise RuntimeError("OOC LAUUM is only implemented for contiguous matrices")

    max_block_size = max(ba.length for ba in block_allocs)

    # initialize streams & events
    compute_stream = torch.cuda.Stream(device=torch.cuda.current_device())
    copy_stream = torch.cuda.Stream(device=torch.cuda.current_device())
    lauum_d2h_stream = torch.cuda.Stream(device=torch.cuda.current_device())
    r_buffers_ready = [torch.cuda.Event(), torch.cuda.Event()]
    r_buffers_clear = [torch.cuda.Event(), torch.cuda.Event()]
    tmp_buf_ready = [torch.cuda.Event(), torch.cuda.Event()]
    for ev in r_buffers_clear:
        ev.record(compute_stream)
    for ev in r_buffers_ready:
        ev.record(copy_stream)
    for ev in tmp_buf_ready:
        ev.record(copy_stream)

    # initialzie CPU and GPU buffers
    temp_bufs = [
        create_fortran((max_block_size, max_block_size), L.dtype, "cpu", pin_memory=True),
        create_fortran((max_block_size, max_block_size), L.dtype, "cpu", pin_memory=True)
    ]
    mem_needed = 3 * N * max_block_size + 2 * (max_block_size**2)
    f_gpu = torch.empty(size=(mem_needed,), dtype=L.dtype, device=torch.cuda.current_device())
    whole_col_b = f_gpu[: N * max_block_size]
    r_buffers = [
        f_gpu[N * max_block_size : 2 * N * max_block_size],
        f_gpu[2 * N * max_block_size : 3 * N * max_block_size]
    ]
    syrk_out = extract_fortran(f_gpu, size=(max_block_size, max_block_size), offset=3 * N * max_block_size)
    lauum_out = extract_fortran(
        f_gpu, size=(max_block_size, max_block_size), offset=3 * N * max_block_size + max_block_size**2
    )
    syrk_out.fill_(0.0)

    # Outer iteration
    for b in range(len(block_allocs)):
        bb = block_allocs[b]
        with torch.cuda.stream(compute_stream):
            # Load column-block b.
            # Instead of loading the whole column only load the last rows starting from b
            try:
                if contig == "C":
                    # src is row-contiguous. dest is also row-contiguous.
                    cuda_2d_copy_async(
                        src_tensor=L[bb.start, bb.start],
                        src_pitch=L.shape[1] * dts,
                        dest_tensor=whole_col_b,
                        dest_pitch=max_block_size * dts,
                        width=bb.length * dts,
                        height=N - bb.start,
                    )
                    col_b = whole_col_b.as_strided((N - bb.start, bb.length), (max_block_size, 1))
                else:
                    # src is column-contiguous, dest also column-contiguous.
                    cublas_2d_copy_to_dev_async(
                        rows=N - bb.start,
                        cols=bb.length,
                        elemSize=dts,
                        host_tensor=L[bb.start, bb.start],
                        lda=N,
                        dev_tensor=whole_col_b,
                        ldb=N,
                    )
                    col_b = whole_col_b.as_strided((N - bb.start, bb.length), (1, N))
            except ValueError:  # all rows are smaller than `b`?
                raise

        # Start by preloading the first column r = b + 1. 
        # Overlaps with computation of r == b below.
        buf_idx = 0
        if (len(block_allocs) - b) > 1:
            br = block_allocs[b + 1]
            with torch.cuda.stream(copy_stream):
                if contig == "C":
                    cuda_2d_copy_async(
                        src_tensor=L[br.start, br.start],
                        src_pitch=L.shape[1] * dts,
                        dest_tensor=r_buffers[buf_idx],
                        dest_pitch=max_block_size * dts,
                        width=br.length * dts,
                        height=N - br.start,
                    )
                else:
                    cublas_2d_copy_to_dev_async(
                        rows=N - br.start,
                        cols=br.length,
                        elemSize=dts,
                        host_tensor=L[br.start, br.start],
                        lda=N,
                        dev_tensor=r_buffers[buf_idx],
                        ldb=N,
                    )
                r_buffers_ready[buf_idx].record(copy_stream)
        for r in range(b, len(block_allocs)):
            br = block_allocs[r]
            # Triangular block only uses b-column. All on compute_stream
            if r == b:
                with torch.cuda.stream(compute_stream):
                    is_last_row = bb.start + bb.length == N
                    # SYRK on g_b[bb.length:, :] with output replacing g_b[:bb.length, :]
                    # C = beta*C + alpha * op(A) @ op(A).T
                    if not is_last_row:
                        if contig == "C":
                            cublas_syrk(
                                A=col_b[bb.length:, :],
                                # A=whole_col_b[bb.length * max_block_size :],
                                lda=max_block_size,
                                alpha=1.0,
                                C=syrk_out,
                                ldc=max_block_size,
                                beta=0.0,
                                upper=True,
                                transpose=False,
                                n=bb.length,
                                k=N - bb.start - bb.length,
                            )
                        else:
                            cublas_syrk(
                                A=col_b[bb.length:, :],
                                lda=N,
                                alpha=1.0,
                                C=syrk_out,
                                ldc=max_block_size,
                                beta=0.0,
                                upper=False,
                                transpose=True,
                                n=bb.length,
                                k=N - bb.start - bb.length,
                            )
                    # LAUUM
                    c_lauum_in = col_b[:bb.length, :bb.length]
                    c_lauum_out = lauum_out[:bb.length, :bb.length]
                    compute_stream.wait_stream(lauum_d2h_stream)
                    if contig == "C":
                        # Lower LAUUM for C-contig is equal to upper LAUUM for F-contig
                        c_lauum_out.copy_(c_lauum_in)  # TODO: This copy requires intermediate buffers: it's from C->F order
                        lauum_cuda(
                            n=bb.length,
                            A=c_lauum_in,
                            lda=max_block_size,
                            B=c_lauum_out,
                            ldb=max_block_size,
                            lower=False,
                        )
                    else:
                        c_lauum_out.copy_(c_lauum_in.T)  # TODO: This copy requires intermediate buffers: it's from C->F order
                        lauum_cuda(
                            n=bb.length,
                            A=c_lauum_in,
                            lda=N,
                            B=c_lauum_out,
                            ldb=max_block_size,
                            lower=True,
                        )
                    if not is_last_row:
                        c_lauum_out.add_(syrk_out[: bb.length, : bb.length])
                # D2H copy in a separate stream
                with torch.cuda.stream(lauum_d2h_stream):
                    lauum_d2h_stream.wait_stream(compute_stream)
                    # copy back c_lauum_out into temporary buffer on CPU. F->F copy.
                    cublas_2d_copy_to_host_async(
                        rows=bb.length,
                        cols=bb.length,
                        elemSize=dts,
                        dev_tensor=c_lauum_out,
                        lda=max_block_size,
                        host_tensor=temp_bufs[buf_idx],
                        ldb=max_block_size,
                    )
                    tmp_buf_ready[buf_idx].record()
            else:  # r > b
                # Preload the next column r + 1.
                if len(block_allocs) > r + 1:
                    with torch.cuda.stream(copy_stream):
                        # Swap buffer
                        buf_idx = 1 - buf_idx
                        copy_stream.wait_event(r_buffers_clear[buf_idx]) # pyright: ignore[reportArgumentType]
                        br_next = block_allocs[r + 1]
                        # copy L[br_next.start, br_next.start] -> r_buffer
                        if contig == "C":
                            cuda_2d_copy_async(
                                src_tensor=L[br_next.start, br_next.start],
                                src_pitch=L.shape[1] * dts,
                                dest_tensor=r_buffers[buf_idx],
                                dest_pitch=max_block_size * dts,
                                width=br_next.length * dts,
                                height=N - br_next.start,
                            )
                        else:
                            cublas_2d_copy_to_dev_async(
                                rows=N - br_next.start,
                                cols=br_next.length,
                                elemSize=dts,
                                host_tensor=L[br_next.start, br_next.start],
                                lda=N,
                                dev_tensor=r_buffers[buf_idx],
                                ldb=N,
                            )
                        r_buffers_ready[buf_idx].record(copy_stream)
                        # Swap buffer
                        buf_idx = 1 - buf_idx

                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(r_buffers_ready[buf_idx]) # pyright: ignore[reportArgumentType]
                    # Restrict column b to only the last 'r' rows
                    ccb = col_b[br.start - bb.start:, :]
                    # TRMM on g_r[0:br.length, :] which is triangular (r*r)
                    #         and cur_g_b[0:br.length, :]
                    #         output is a r*b matrix and stored in first rows of ccb
                    # C = alpha * op(A) @ B -- A triangular
                    if contig == "C":
                        cublas_trmm(
                            A=r_buffers[buf_idx],
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
                    else:
                        cublas_trmm(
                            A=r_buffers[buf_idx],
                            lda=N,
                            alpha=1.0,
                            B=ccb,
                            ldb=N,
                            C=ccb,
                            ldc=N,
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
                        if contig == "C":
                            cublas_gemm(
                                A=ccb[br.length:, :],
                                lda=max_block_size,
                                alpha=1.0,
                                B=r_buffers[buf_idx][br.length * max_block_size :],
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
                        else:
                            cublas_gemm(
                                A=r_buffers[buf_idx][br.length:],
                                lda=N,
                                alpha=1.0,  # A is k * m
                                B=ccb[br.length:, :],
                                ldb=N,  # B is k * n
                                C=ccb,
                                ldc=N,
                                beta=1.0,  # C is m * n
                                transa=True,
                                transb=False,
                                m=br.length,
                                n=bb.length,
                                k=N - br.start - br.length,
                            )
                    r_buffers_clear[buf_idx].record(compute_stream)

                    if r > b + 1:
                        # Make sure temp_buf is free, and issue CPU copy. 
                        # synchronize() (blocking) is necessary since the operation here is on CPU.
                        tmp_buf_ready[1 - buf_idx].synchronize()
                        # Copy result back to L from temporary CPU buffer. 
                        # Data is written to temp_buf from the compute stream.
                        prev_br = block_allocs[r - 2]
                        if contig == "C":
                            # NOTE: in C-contig case this is a F->C copy (on CPU only)
                            L[
                                bb.start:bb.end, prev_br.start:prev_br.end
                            ].copy_(
                                temp_bufs[1 - buf_idx][:bb.length, :prev_br.length]
                            )
                        else:
                            L[
                                bb.start:bb.end, prev_br.start:prev_br.end
                            ].copy_(
                                temp_bufs[1 - buf_idx][:prev_br.length, :bb.length].T
                            )

                    # Copy back to A[r, b]
                    # Copy must be transposed, copy to temp_buf first.
                    if contig == "C":
                        # This is a transpose copy (note lda=max_block_size) because
                        # ccb is treated as F-contiguous instead of C contiguous.
                        cublas_2d_copy_to_host_async(
                            rows=bb.length,
                            cols=br.length,
                            elemSize=dts,
                            dev_tensor=ccb,
                            lda=max_block_size,
                            host_tensor=temp_bufs[1 - buf_idx],
                            ldb=max_block_size,
                        )
                    else:
                        cublas_2d_copy_to_host_async(
                            rows=br.length,
                            cols=bb.length,
                            elemSize=dts,
                            dev_tensor=ccb,
                            lda=N,
                            host_tensor=temp_bufs[1 - buf_idx],
                            ldb=max_block_size,
                        )
                    tmp_buf_ready[1 - buf_idx].record(compute_stream)
                    # Swap buffer
                    buf_idx = 1 - buf_idx
        # Final copy-backs from the temporary buffer
        if len(block_allocs) - b > 1:
            tmp_buf_ready[1 - buf_idx].synchronize()
            r = len(block_allocs) - 2
            br = block_allocs[r]
            if contig == "C":
                L[
                    bb.start:bb.end, br.start:br.end
                ].copy_(
                    temp_bufs[1 - buf_idx][:bb.length, :br.length]
                )
            else:
                L[
                    bb.start:bb.end, br.start:br.end
                ].copy_(
                    temp_bufs[1 - buf_idx][:br.length, :bb.length].T
                )
        if len(block_allocs) - b > 0:
            r_buffers_clear[buf_idx].synchronize()
            r = len(block_allocs) - 1
            br = block_allocs[r]
            if contig == "C":
                L[
                    bb.start:bb.end, br.start:br.end
                ].copy_(
                    temp_bufs[buf_idx][:bb.length, :br.length]
                )
            else:
                L[
                    bb.start:bb.end, br.start:br.end
                ].copy_(
                    temp_bufs[buf_idx][:br.length, :bb.length].T
                )
