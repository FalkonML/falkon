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
    block_sizes = extras * [block_size + 1] + (num_blocks - extras) * [block_size]

    return block_sizes


def calc_block_sizes2(max_block_size: int,
                      num_devices: int,
                      num_rows: int,
                      min_blocks_per_device: int) -> List[int]:
    # 1. Number of blocks must be a multiple of num_devices
    # 2. There must be at least min_blocks_per_device
    # 3. Block sizes should be close to the 'preferred_block_size'
    # 4. Block sizes should be multiples of 32
    preferred_block_size = 7000
    if max_block_size < preferred_block_size + 64:
        preferred_block_size = max_block_size

    min_num_blocks = int(math.ceil(num_rows / preferred_block_size))
    num_blocks = max(min_num_blocks, num_devices * min_blocks_per_device)
    if num_blocks <= 0:
        raise RuntimeError("num_blocks expected > 0, found %d" % (num_blocks))
    if num_blocks % num_devices != 0:  # even number of blocks per GPU
        num_blocks += num_devices - (num_blocks % num_devices)
    block_size, extras = divmod(num_rows, num_blocks)
    round_block_size = block_size + (block_size % 64)
    if round_block_size > max_block_size:
        # This should not happen unless max_block_size is really small.
        round_block_size = block_size

    # Get all block sizes
    pointer = 0
    block_sizes = []
    while pointer < num_rows:
        new_pointer = pointer + min(round_block_size, num_rows - pointer)
        block_sizes.append(new_pointer - pointer)
        pointer = new_pointer

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
