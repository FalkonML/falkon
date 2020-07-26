import math
from typing import List

import torch


def calc_block_sizes(max_block_size: int,
                     num_devices: int,
                     num_rows: int,
                     min_blocks_per_device: int) -> List[int]:
    min_num_blocks = int(math.ceil(num_rows / max_block_size))
    num_blocks = max(min_num_blocks, num_devices, min_blocks_per_device)
    if num_blocks % num_devices != 0:  # even number of blocks per GPU
        num_blocks += num_devices - (num_blocks % num_devices)
    if num_blocks <= 0:
        raise RuntimeError("num_blocks expected > 0, found %d" % (num_blocks))
    # Calculate a block size which evenly splits N
    block_size, extras = divmod(num_rows, num_blocks)
    block_sizes = extras * [block_size+1] + (num_blocks - extras) * [block_size]

    return block_sizes


def calc_block_sizes3(max_block_size: int, num_devices: int, num_rows: int) -> List[int]:
    preferred_block_size = 7000
    # Shortcircuit small matrices
    if num_rows < 1024 and num_rows <= max_block_size:  # Single block on one GPU
        return [num_rows]
    # If we have very small block size, we don't want any block to be larger than it
    if preferred_block_size > max_block_size:
        preferred_block_size = max_block_size

    num_blocks = int(math.ceil(num_rows / preferred_block_size))

    # Ensure an even distribution of blocks between GPUs
    if num_blocks % num_devices != 0 and num_blocks < num_rows:  # even number of blocks per GPU
        added_blocks = num_devices - (num_blocks % num_devices)
        # Ensure that we don't get into num_blocks > num_rows, which then creates blocks of size 0.
        if num_blocks + added_blocks <= num_rows:
            num_blocks += added_blocks

    block_size, extras = divmod(num_rows, num_blocks)
    block_sizes = extras * [block_size+1] + (num_blocks - extras) * [block_size]
    return block_sizes


def prepare_matrix(A):
    # Convert to numpy
    if isinstance(A, torch.Tensor):
        A = A.numpy()

    # Make A Fortran-order
    transpose = False
    if not A.flags['FARRAY']:
        A = A.T
        transpose = not transpose
    if not A.flags['F_CONTIGUOUS']:
        raise RuntimeError("Failed to convert matrix A to Fortran order. "
                           "Matrix is not contiguous in memory.")
    return A, transpose
