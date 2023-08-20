import warnings
from typing import Union, Tuple, Any, Generator

import numpy as np
import torch

__all__ = (
    "create_same_stride",
    "copy_same_stride",
    "extract_same_stride",
    "extract_fortran",
    "extract_C",
    "create_fortran",
    "create_C",
    "is_f_contig",
    "is_contig",
    "is_contig_vec",
    "cast_tensor",
    "move_tensor",
    "batchify_tensors",
)


def _fcontig_strides(sizes) -> Tuple[int, ...]:
    if len(sizes) == 0:
        return ()
    return tuple([1] + np.cumprod(sizes)[:-1].tolist())


def _ccontig_strides(sizes) -> Tuple[int, ...]:
    if len(sizes) == 0:
        return ()
    return tuple(np.cumprod(sizes[1:][::-1])[::-1].tolist() + [1])


def _new_strided_tensor(
    size: Tuple[int], stride: Tuple[int], dtype: torch.dtype, device: Union[str, torch.device], pin_memory: bool
) -> torch.Tensor:
    if not torch.cuda.is_available():
        pin_memory = False
    else:
        if isinstance(device, torch.device):
            pin_memory &= device.type == "cpu"
        else:
            pin_memory &= device.lower() == "cpu"

    # noinspection PyArgumentList
    return torch.empty_strided(
        size=size, stride=stride, dtype=dtype, device=device, requires_grad=False, pin_memory=pin_memory
    )


def extract_fortran(from_tns: torch.Tensor, size: Tuple[int, ...], offset: int) -> torch.Tensor:
    strides = _fcontig_strides(size)
    return from_tns.as_strided(size=size, stride=strides, storage_offset=int(offset))


def extract_C(from_tns: torch.Tensor, size: Tuple[int, ...], offset: int) -> torch.Tensor:
    strides = _ccontig_strides(size)
    return from_tns.as_strided(size=size, stride=strides, storage_offset=int(offset))


def extract_same_stride(
    from_tns: torch.Tensor, size: Tuple[int, ...], other: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    if is_f_contig(other, strict=True):
        return extract_fortran(from_tns, size, offset)
    elif is_contig(other):
        return extract_C(from_tns, size, offset)
    else:
        raise ValueError("Desired stride is not contiguous, cannot extract.")


def create_fortran(
    size: Tuple[int, ...], dtype: torch.dtype, device: Union[str, torch.device], pin_memory: bool = False
) -> torch.Tensor:
    """Allocates an empty, column-contiguous 1 or 2-dimensional tensor

    Parameters
    -----------
    size : tuple of integers
        Must be a tuple of length 1 or 2 indicating the shape of the
        created tensor.
    dtype : torch.dtype
        The type of the new tensor.
    device : str or torch.device
        The device on which the tensor should be allocated (e.g. 'cpu', 'cuda:0')
    pin_memory : bool
        Whether a CPU tensor should be allocated in pinned memory or
        not. If allocating a GPU tensor this flag has no effect.

    Returns
    --------
    t : torch.Tensor
        The allocated tensor
    """
    strides = _fcontig_strides(size)
    return _new_strided_tensor(size, strides, dtype, device, pin_memory)


def create_C(
    size: Tuple[int, ...], dtype: torch.dtype, device: Union[str, torch.device], pin_memory: bool = False
) -> torch.Tensor:
    """Allocates an empty, row-contiguous 1 or 2-dimensional tensor

    Parameters
    -----------
    size : tuple of integers
        Must be a tuple of length 1 or 2 indicating the shape of the
        created tensor.
    dtype : torch.dtype
        The type of the new tensor.
    device : str or torch.device
        The device on which the tensor should be allocated (e.g. 'cpu', 'cuda:0')
    pin_memory : bool
        Whether a CPU tensor should be allocated in pinned memory or
        not. If allocating a GPU tensor this flag has no effect.

    Returns
    --------
    t : torch.Tensor
        The allocated tensor
    """
    strides = _ccontig_strides(size)
    return _new_strided_tensor(size, strides, dtype, device, pin_memory)


def create_same_stride(
    size: Tuple[int, ...],
    other: torch.Tensor,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    pin_memory: bool = False,
) -> torch.Tensor:
    if is_f_contig(other, strict=True):
        return create_fortran(size=size, dtype=dtype, device=device, pin_memory=pin_memory)
    elif is_contig(other):
        return create_C(size=size, dtype=dtype, device=device, pin_memory=pin_memory)
    else:
        raise ValueError("Desired stride is not contiguous, cannot create.")


def copy_same_stride(tensor: torch.Tensor, pin_memory: bool = False) -> torch.Tensor:
    new = create_same_stride(tensor.shape, tensor, tensor.dtype, tensor.device, pin_memory)
    new.copy_(tensor)
    return new


def is_f_contig(tensor: torch.Tensor, strict: bool = False) -> bool:
    """Check if a pytorch Tensor is column-contiguous (Fortran order)

    Column-contiguity means that the stride of the first dimension (of
    a 2D tensor) must be equal to 1.
    In case of 1D tensors we just check contiguity

    Parameters
    -----------
    tensor : torch.Tensor
        1 or 2-dimensional tensor whose stride should be checked.
    strict : bool
        For 1D arrays there is no difference for row and column contiguity.
        2D arrays where one of the dimensions is of size 1 can be either
        treated like 1D arrays (`strict=False`) or like 2D arrays
        (`strict=True`).

    Returns
    --------
    fortran : bool
        Whether the input tensor is column-contiguous
    """
    # noinspection PyArgumentList
    strides = tensor.stride()
    sizes = tensor.shape
    if len(sizes) == 0:
        return True
    if len(sizes) == 1:
        return strides[0] == 1
    # 2 checks for 1D tensors which look 2D
    if sizes[-2] == 1:
        if strict:
            return strides[-2] == 1
        return strides[-1] == 1 or strides[-2] == 1
    if sizes[-1] == 1:
        if strict:
            return strides[-2] == 1 and strides[-1] >= sizes[-2]
        return strides[-2] == 1
    # 2D tensors must have the stride on the first
    # dimension equal to 1 (columns are stored contiguously).
    if strides[-2] != 1 or strides[-1] < strides[-2]:
        return False
    return True


def is_contig_vec(tensor: torch.Tensor) -> bool:
    # noinspection PyArgumentList
    strides = tensor.stride()
    sizes = tensor.shape

    num_not_1 = 0
    for sz, st in zip(sizes, strides):
        if sz != 1:
            num_not_1 += 1
            if st != 1:
                return False
        if num_not_1 > 1:
            return False
    return True


def is_contig(tensor: torch.Tensor) -> bool:
    # noinspection PyArgumentList
    stride = tensor.stride()
    return any(s == 1 for s in stride)


def cast_tensor(tensor: torch.Tensor, dtype: torch.dtype, warn: bool = True) -> torch.Tensor:
    if tensor.is_cuda:
        raise RuntimeError("cast_tensor can only work on CPU tensors")

    if tensor.dtype == dtype:
        return tensor

    if dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.float64:
        np_dtype = np.float64
    else:
        raise RuntimeError("cast_tensor can only cast to float types")

    if warn:
        warnings.warn(
            f"Changing type of {tensor.size()} tensor from {tensor.dtype} to {dtype}. "
            "This will use more memory. If possible change 'inter_type' and "
            "'final_type', or cast the original data to the appropriate type."
        )
    out_np = tensor.numpy().astype(np_dtype, order="K", casting="unsafe", copy=True)
    return torch.from_numpy(out_np)


def move_tensor(tensor: torch.Tensor, device: Union[torch.device, str]) -> torch.Tensor:
    if str(device) == str(tensor.device):
        return tensor

    new_tensor = create_same_stride(tensor.size(), tensor, tensor.dtype, device)
    new_tensor.copy_(tensor)
    return new_tensor


def batchify_tensors(*tensors: torch.Tensor) -> Generator[torch.Tensor, Any, None]:
    for t in tensors:
        if t.dim() == 2:
            yield t.unsqueeze(0)
        else:
            yield t
