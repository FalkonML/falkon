import math

import torch
from falkon.options import CholeskyOptions

from falkon.cuda import initialization
from falkon.cuda.cusolver_gpu import *
from falkon.utils import devices
from falkon import la_helpers
from falkon.utils.cuda_helpers import copy_to_device, copy_to_host
from falkon.utils.helpers import choose_fn, sizeof_dtype
from falkon.ooc_ops.multigpu_potrf import parallel_potrf
from falkon.options import FalkonOptions
from .ooc_utils import calc_block_sizes
from ..utils.devices import DeviceInfo
from ..utils.tensor_helpers import create_fortran, is_f_contig, copy_same_stride

__all__ = ("gpu_cholesky",)


def _ic_cholesky(A, upper, device, cusolver_handle):
    """Cholesky factorization of matrix `A` on the GPU

    Uses the cuSOLVER library for implementation of the POTRF function.

    Parameters:
    -----------
    A : [n, n] CPU or GPU array (column-contiguous)
        The (positive definite) matrix which should be factorized
    upper : bool
        Whether we need to factorize the upper of lower portion of `A`. The other side
        of the matrix will not be touched.
    device : int
        The GPU device on which to run the factorization
    cusolver_handle
        Pointer to the cuSOLVER context, which needs to be initialized before calling
        the function.

    Returns:
    --------
    A : [n, n] CPU or GPU array (column-contiguous)
        The factorization of A which overwrites the upper (or lower) triangular part
        of the matrix A. This is not a copy of the original matrix.
    """
    # Check library initialization
    if cusolver_handle is None:
        raise RuntimeError("CuSolver must be initialized "
                           "before running in-core Cholesky.")
    if not is_f_contig(A):
        raise RuntimeError("Cholesky input must be F-contiguous")

    uplo = 'U' if upper else 'L'
    n = A.shape[0]

    tc_device = torch.device("cuda:%d" % (device))
    # Choose functions by dtype
    potrf_buf_size = choose_fn(A.dtype, cusolverDnDpotrf_bufferSize, cusolverDnSpotrf_bufferSize,
                               "POTRF Buffer size")
    potrf_fn = choose_fn(A.dtype, cusolverDnDpotrf, cusolverDnSpotrf, "POTRF")

    # noinspection PyUnresolvedReferences
    with torch.cuda.device(tc_device):
        # Copy A to device memory
        if A.is_cuda:
            Agpu = A
        else:
            Agpu = create_fortran(A.shape, A.dtype, tc_device)
            copy_to_device(n, n, A, 0, 0, Agpu, 0, 0)

        # Create workspace buffer
        potrf_bsize = potrf_buf_size(
            handle=cusolver_handle, uplo=uplo, n=n, A=Agpu.data_ptr(), lda=n)
        potrf_wspace = create_fortran((potrf_bsize,), A.dtype, tc_device)
        dev_info = torch.tensor(4, dtype=torch.int32, device=tc_device)

        # Run cholesky
        potrf_fn(handle=cusolver_handle,
                 uplo=uplo, n=n, A=Agpu.data_ptr(), lda=n,
                 workspace=potrf_wspace.data_ptr(), Lwork=potrf_bsize, devInfo=dev_info)

        # Copy back to CPU
        if not A.is_cuda:
            copy_to_host(n, n, Agpu, 0, 0, A, 0, 0)
            del Agpu
        del potrf_wspace, dev_info
    return A


def _parallel_potrf_runner(A: torch.Tensor, opt: CholeskyOptions, gpu_info) -> torch.Tensor:
    num_gpus = len(gpu_info)
    N = A.shape[0]
    dt = A.dtype
    # Calculate the maximum block size such that we don't run out of GPU
    # RAM on **any** available GPU. We need a total of 2 whole columns and 1 tile:
    # block-size^2 * ((N / block-size) * 2 + 1) floats
    # (plus the cuSOLVER buffer which is small).
    # block_size < (sqrt((2*N)^2 + 4R) - 2*N) / 2
    dts = sizeof_dtype(dt)
    avail_ram = min([g.actual_free_mem for g in gpu_info]) / dts
    max_block_size = (math.sqrt(4 * N ** 2 + 4 * avail_ram) - 2 * N) / 2
    max_block_size = int(math.floor(max_block_size))
    if max_block_size < 1:
        raise RuntimeError(
            "Cannot run parallel POTRF with minimum "
            "available memory of %.2fMB" % (avail_ram * dts / 2 ** 20))

    block_sizes = calc_block_sizes(
        max_block_size, num_gpus, N, opt.chol_par_blk_multiplier)
    block_allocations = []
    cur_n = 0
    for i, bs in enumerate(block_sizes):
        block_allocations.append(
            (cur_n, cur_n + bs, bs, i % num_gpus, i)
        )
        cur_n += bs

    device_info = []
    for g in range(num_gpus):
        device_info.append(
            (0.0, initialization.cusolver_handle(g), g)
        )

    parallel_potrf(device_info, block_allocations, A)
    return A


"""
GPU Cholesky, we implement use cuSOLVER as a backend for POTRF.

 - In-core: Can do upper or lower, must be Fortran
 - Out of core: Can only do lower, Fortran

"""


def can_do_ic(A: torch.Tensor, device: DeviceInfo):
    # noinspection PyUnresolvedReferences
    avail_ram = device.actual_free_mem
    # The multiplier here is a bit tricky since setting it too high results
    # in hard-to-debug cuda errors
    avail_ram *= 0.85

    if A.is_cuda:
        needed_ram = 100 * 8  # Not very much indeed
    else:
        needed_ram = A.shape[0] * A.shape[1] * sizeof_dtype(A.dtype)

    return avail_ram >= needed_ram


def gpu_cholesky(A: torch.Tensor, upper: bool, clean: bool, overwrite: bool, opt: FalkonOptions) -> torch.Tensor:
    """
    Parameters
    -----------
    A : ndarray [N, N]
        2D positive-definite matrix that will be factorized as
        A = U.T @ U (if `upper` is True) or A = L @ L.T if `upper`
        is False.
    upper : bool
        Whether the triangle which should be factorized is the upper or lower of `A`.
    clean : bool
        Whether the "other" triangle of the output matrix (the one that
        does not contain the factorization) will be filled with zeros or
        not.
    overwrite : bool
        Whether to overwrite matrix A or to output the result in a new
        buffer.

    Notes
    ------
    The factorization will always be the 'lower' version of the factorization
    which could however end up on the upper-triangular part of the matrix
    in case A is not Fortran contiguous to begin with.
    """
    # Handle 'overwrite' option immediately so that its usage is reflected in memory
    # availability (in case A is on GPU).
    if not overwrite:
        # We could change the stride to be more favorable to the POTRF requirements
        # but it gets complicated. We leave such decisions to the user!
        A = copy_same_stride(A, pin_memory=True)

    # Decide which version of the algo we run: can be in-core or parallel.
    # (Note that the original OOC version is not going to run).

    # Determine GPU free RAM
    gpu_info = [v for k, v in devices.get_device_info(opt).items() if k >= 0]
    for g in gpu_info:
        g.actual_free_mem = min((g.free_memory - 300 * 2 ** 20) * 0.95,
                                opt.max_gpu_mem * 0.95)

    if A.is_cuda:
        try:
            device = [d for d in gpu_info if d.Id == A.device.index][0]
        except IndexError:
            # This should never happen!
            raise RuntimeError("Device of matrix A (%s) is not recognized" % (A.device))
    else:
        device = max(gpu_info, key=lambda g: g.actual_free_mem)
    ic = can_do_ic(A, device) and not opt.chol_force_ooc
    if opt.chol_force_in_core and not ic:
        raise RuntimeError("Cannot run in-core POTRF but `chol_force_in_core` was specified.")

    f_order = is_f_contig(A)
    transposed = False
    if not f_order:
        A = A.T
        upper = not upper
        transposed = True
    # Now A is always in f_order. So we can only allow upper=False
    if upper:
        # Can do only in-core!
        if not ic:
            raise ValueError("GPU POTRF is only implemented on the "
                             "lower triangle for Fortran-ordered matrices (or on the upper "
                             "triangle for C-ordered matrices)")
    if not ic and A.is_cuda:
        _msg = "Cannot run out-of-core POTRF on CUDA matrix 'A'."
        if opt.chol_force_ooc:
            _msg += " Set the `chol_force_ooc` option to `False` in to allow in-core POTRF."
        raise ValueError(_msg)

    # Handle different implementations for POTRF: in-core and out-of-core
    if ic:
        if opt.debug:
            print("Using in-core POTRF")
        _ic_cholesky(A, upper, device=device.Id,
                     cusolver_handle=initialization.cusolver_handle(device.Id))
    else:
        if opt.debug:
            print("Using parallel POTRF")
        _parallel_potrf_runner(A, opt, gpu_info)

    # Perform cleaning of the 'other side' of the matrix
    if clean:
        la_helpers.zero_triang(A, upper=not upper)
    # Undo previous matrix transformations
    if transposed:
        A = A.T

    return A
