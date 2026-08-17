import math


def calc_block_sizes(max_block_size: int, num_devices: int, num_rows: int, min_blocks_per_device: int) -> list[int]:
    min_num_blocks = int(math.ceil(num_rows / max_block_size))
    num_blocks = max(min_num_blocks, num_devices, min_blocks_per_device)
    if num_blocks % num_devices != 0:  # even number of blocks per GPU
        num_blocks += num_devices - (num_blocks % num_devices)
    if num_blocks <= 0:
        raise RuntimeError(f"num_blocks expected > 0, found {num_blocks}")
    # Calculate a block size which evenly splits N
    block_size, extras = divmod(num_rows, num_blocks)
    block_sizes = extras * [block_size + 1] + (num_blocks - extras) * [block_size]

    return block_sizes


def calc_block_sizes3(max_block_size: int, num_devices: int, num_rows: int) -> list[int]:
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
    block_sizes = extras * [block_size + 1] + (num_blocks - extras) * [block_size]
    return block_sizes


def round_down_to_multiple(num: int, mul: int) -> int:
    return (num // mul) * mul


def calc_block_sizes_serial_ooc_lauum(max_block_size: int, num_rows: int, cuda_compute_capability: int,) -> list[int]:
    # Heuristics for best performance. Tuned on begato so probably not great for newer GPUs.
    if cuda_compute_capability >= 8:
        # a100 (leonardo) has 4.5
        preferred_block_size = max(1, round_down_to_multiple(int(num_rows / 4.5), 32))
        if preferred_block_size > 7712:
            preferred_block_size = 7712
    else:
        # begato has cc 7
        preferred_block_size = max(1, round_down_to_multiple(int(num_rows / 6.5), 32))
        if preferred_block_size > 4640:
            preferred_block_size = 4640

    # Shortcircuit small matrices: 1 block only
    if num_rows < 1024 and num_rows <= max_block_size:
        return [num_rows]
    # If only small blocks allowed, use that size
    if preferred_block_size > max_block_size:
        preferred_block_size = max(1, round_down_to_multiple(max_block_size, 32))
    # multiples of 1024 have some weird performance penalty
    if preferred_block_size % 1024 == 0:
        preferred_block_size -= 32

    block_sizes = []
    for i in range(0, num_rows, preferred_block_size):
        block_sizes.append(min(preferred_block_size, num_rows - i))
    return block_sizes
