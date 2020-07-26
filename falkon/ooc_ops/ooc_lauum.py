import math
import threading
from typing import List, Optional

import numpy as np
import torch

from falkon.cuda import initialization
from falkon.utils import devices, PropagatingThread
from falkon.utils.helpers import sizeof_dtype
from falkon.options import FalkonOptions, LauumOptions
from .ooc_utils import calc_block_sizes3, prepare_matrix
from .parallel_lauum import par_lauum_f_lower, par_lauum_c_lower, BlockAlloc
from ..utils.tensor_helpers import is_f_contig, is_contig

__all__ = ("gpu_lauum",)


def _parallel_lauum_runner(A, write_opposite: bool, opt: LauumOptions, gpu_info):
    # Choose target:
    if is_f_contig(A):
        target = par_lauum_f_lower
    elif is_contig(A):
        target = par_lauum_c_lower
    else:
        raise NotImplementedError("Parallel LAUUM is only implemented for contiguous matrices")

    num_gpus = len(gpu_info)
    if num_gpus < 1:
        raise ValueError(
                "Parallel LAUUM should only be run when some GPU is available.")
    N = A.shape[0]
    dt = A.dtype
    dts = sizeof_dtype(dt)
    avail_ram = min([g.actual_free_mem for g in gpu_info]) / dts
    # Each GPU should be able to hold in memory 2 block columns
    max_block_size = int(math.floor(avail_ram / (2*N)))
    if max_block_size < 1:
        raise RuntimeError(
                "Cannot run parallel LAUUM with minimum "
                "available memory of %.2fMB" % (avail_ram * dts / 2**20))

    block_sizes = calc_block_sizes3(
        max_block_size, num_gpus, N, opt.lauum_par_blk_multiplier)
    block_allocations: List[BlockAlloc] = []
    cur_n = 0
    for bs in block_sizes:
        block_allocations.append(BlockAlloc(start=cur_n, end=cur_n + bs, length=bs))
        cur_n += bs

    barrier = threading.Barrier(num_gpus, timeout=1000)
    threads = []
    for g in gpu_info:
        gid_allocs = [i for i in range(len(block_allocations)) if i % num_gpus == g.Id]
        cublas_handle = initialization.cublas_handle(g.Id)
        if cublas_handle is None:
            raise RuntimeError("CUBLAS must be initialized "
                               "on device %d before running parallel LAUUM." % (g.Id))
        t = PropagatingThread(target=target, name="GPU-%d" % (g.Id), args=(
            A, block_allocations, gid_allocs, barrier, g.Id, cublas_handle, write_opposite))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return A


def gpu_lauum(A, upper, overwrite=True, write_opposite=False, opt: Optional[FalkonOptions] = None):
    """
    Parameters
    -----------
    A : ndarray [N, N]
        2D positive-definite matrix that will be factorized as
        A = U.T @ U (if `upper` is True) or A = L @ L.T if `upper`
        is False.
    overwrite : bool
        Whether to overwrite matrix A or to output the result in a new
        buffer.

    Notes
    ------
    The factorization will always be the 'lower' version of the factorization
    which could however end up on the upper-triangular part of the matrix
    in case A is not Fortran contiguous to begin with.
    """
    if opt is None:
        opt = FalkonOptions()
    gpu_info = [v for k, v in devices.get_device_info(opt).items() if k >= 0]
    for g in gpu_info:
        g.actual_free_mem = min((g.free_memory - 300 * 2 ** 20) * 0.95,
                                opt.max_gpu_mem * 0.95)

    # Start matrix preparations
    if isinstance(A, np.ndarray):
        Anp = A
    elif isinstance(A, torch.Tensor):
        Anp = A.numpy()
    else:
        raise TypeError("Unexpected type encountered for A: %s" % (A.dtype))

    if not overwrite:
        Anp = np.copy(Anp, order='A')

    # Will give a fortran-contiguous numpy array. No copies are performed.
    Anp, transposed = prepare_matrix(Anp)
    if transposed:
        upper = not upper

    # Parallel can only do lower C or F-contiguous arrays
    # But by transposing as necessary, it is able to run with every combination of inputs.
    At = torch.from_numpy(Anp)
    if upper:
        At = At.T
    # The parallel runner chooses based on the contiguity pattern of the inputs.
    _parallel_lauum_runner(At, write_opposite, opt, gpu_info)

    if transposed:
        Anp = Anp.T

    if isinstance(A, np.ndarray):
        return Anp
    else:
        return torch.from_numpy(Anp)
