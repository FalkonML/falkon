import math
import threading
from typing import List, Optional

import torch

from falkon.options import FalkonOptions
from falkon.utils import PropagatingThread, devices
from falkon.utils.helpers import sizeof_dtype
from falkon.utils.stream_utils import sync_current_stream
from falkon.utils.tensor_helpers import copy_same_stride, is_contig, is_f_contig

from .ooc_utils import calc_block_sizes3
from .parallel_lauum import BlockAlloc, par_lauum_c_lower, par_lauum_f_lower

__all__ = ("gpu_lauum",)


def _parallel_lauum_runner(A, write_opposite: bool, gpu_info):
    # Choose target:
    if is_f_contig(A):
        target = par_lauum_f_lower
    elif is_contig(A):
        target = par_lauum_c_lower
    else:
        raise NotImplementedError("Parallel LAUUM is only implemented for contiguous matrices")

    N = A.shape[0]
    dt = A.dtype
    dts = sizeof_dtype(dt)
    if A.is_cuda:  # In-core
        sync_current_stream(A.device)
        gpu_info = [g for g in gpu_info if g.Id == A.device.index]
        avail_ram = gpu_info[0].actual_free_mem / dts
        if target.__name__ == "par_lauum_f_lower":
            # Each GPU should hold in memory two additional blocks (2*B^2 <= M)
            # and 1 full column.
            max_block_size = int(math.floor((-N + math.sqrt(N**2 + 8 * avail_ram)) / 4))
        else:
            # Same RAM requirements as the out-of-core version
            max_block_size = int(math.floor((-2 * N + math.sqrt(4 * N**2 + 8 * avail_ram)) / 4))
        if max_block_size < 1:
            raise RuntimeError(
                f"Cannot run parallel LAUUM with minimum available memory of {avail_ram * dts / 2**20:.2f}MB"
            )
        # All computations on the same device (where data is stored). No multi-GPU support!
        block_sizes = calc_block_sizes3(max_block_size, 1, N)
    else:  # Out-of-core
        avail_ram = min([g.actual_free_mem for g in gpu_info]) / dts
        # Each GPU should be able to hold in memory 2 block columns
        # Plus two blocks (=> quadratic equation 2B^2 + 2BN - M <= 0.
        # An additional block is needed whenever write_opposite is True, due to
        # copying blocks between matrices with different strides!
        if write_opposite:
            max_block_size = int(math.floor((-2 * N + math.sqrt(4 * N**2 + 12 * avail_ram)) / 6))
        else:
            max_block_size = int(math.floor((-2 * N + math.sqrt(4 * N**2 + 8 * avail_ram)) / 4))
        if max_block_size < 1:
            raise RuntimeError(
                f"Cannot run parallel LAUUM with minimum available memory of {avail_ram * dts / 2**20:.2f}MB"
            )

        block_sizes = calc_block_sizes3(max_block_size, len(gpu_info), N)

    # Create BlockAlloc objects describing the subdivision of input
    block_allocations: List[BlockAlloc] = []
    cur_n = 0
    for bs in block_sizes:
        block_allocations.append(BlockAlloc(start=cur_n, end=cur_n + bs, length=bs))
        cur_n += bs

    num_gpus = len(gpu_info)
    if num_gpus < 1:
        raise ValueError("Parallel LAUUM can only run when a GPU is available.")
    barrier = threading.Barrier(num_gpus, timeout=1000)
    threads = []
    for _gpu_idx, g in enumerate(gpu_info):
        # Assign rows to GPUs round-robin. Use _gpu_idx instead of g.Id since the latter
        # may not contain all integers from 0.
        gid_allocs = [i for i in range(len(block_allocations)) if i % num_gpus == _gpu_idx]
        t = PropagatingThread(
            target=target,
            name=f"GPU-{g.Id}",
            args=(A, block_allocations, gid_allocs, barrier, g.Id, write_opposite),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return A


def gpu_lauum(
    A: torch.Tensor,
    upper: bool,
    overwrite: bool = True,
    write_opposite: bool = False,
    opt: Optional[FalkonOptions] = None,
):
    """
    Parameters
    -----------
    A : torch.Tensor
        N-by-N triangular matrix.
    upper: bool
        Whether the input matrix is upper or lower triangular.
    overwrite : bool
        Whether to overwrite matrix A or to output the result in a new buffer.
    write_opposite : bool
        Independently of the ``overwrite`` parameter, whether to write the result of the
        triangular multiplication on the 'opposite' side of ``A``. For example, if ``upper == True``
        and ``overwrite == False``, then the result will be written on the lower triangular part
        of the input matrix ``A``.
        While independent, this is mostly useful when ``overwrite == False``, since it can
        effectively avoid allocating a new tensor, and at the same time preserve the original data.
    opt: FalkonOptions or None
        Options for the LAUUM operation. The only relevant options are the one connected to
        GPU memory usage.
    Returns
    -------
    out : torch.Tensor
        A (N x N) tensor. This will share the same memory as the input tensor ``A`` if ``overwrite``
        is set to ``True``, otherwise it will be a newly allocated tensor.
    """
    if opt is None:
        opt = FalkonOptions()
    if not overwrite:
        A = copy_same_stride(A, pin_memory=True)
    # TODO: There is a helper function in mmv_ops for this.
    gpu_info = [v for k, v in devices.get_device_info(opt).items() if k >= 0]
    for g in gpu_info:
        g.actual_free_mem = min((g.free_memory - 300 * 2**20) * 0.95, opt.max_gpu_mem * 0.95)

    # Parallel can only do lower C or F-contiguous arrays
    # By transposing as necessary, it is able to run with every combination of inputs.
    transposed = False
    if upper:
        A = A.T
        transposed = True

    # The parallel runner chooses based on the contiguity pattern of the inputs.
    _parallel_lauum_runner(A, write_opposite, gpu_info)

    if transposed:
        A = A.T
    return A
