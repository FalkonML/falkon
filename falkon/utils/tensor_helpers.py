import warnings
from typing import Union, Tuple

import numpy as np
import torch

__all__ = (
    "create_same_stride", "copy_same_stride", "extract_same_stride",
    "extract_fortran", "extract_C",
    "create_fortran", "create_C", "is_f_contig", "is_contig",
    "cast_tensor", "move_tensor",
)


def _new_strided_tensor(
        size: Tuple[int],
        stride: Tuple[int],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        pin_memory: bool) -> torch.Tensor:
    if isinstance(device, torch.device):
        pin_memory &= device.type == 'cpu'
    else:
        pin_memory &= device.lower() == 'cpu'

    # noinspection PyArgumentList
    return torch.empty_strided(
        size=size, stride=stride,
        dtype=dtype, device=device,
        requires_grad=False,
        pin_memory=pin_memory)


def extract_fortran(from_tns: torch.Tensor,
                    size: Union[Tuple[int, int], Tuple[int]],
                    offset: int) -> torch.Tensor:
    if len(size) == 1:
        stride = (1,)
    elif len(size) == 2:
        stride = (1, size[0])
    else:
        raise ValueError("extract_fortran can only extract 1 or 2D tensors.")
    return from_tns.as_strided(size=size, stride=stride, storage_offset=int(offset))


def extract_C(from_tns: torch.Tensor,
              size: Union[Tuple[int, int], Tuple[int]],
              offset: int) -> torch.Tensor:
    if len(size) == 1:
        stride = (1,)
    elif len(size) == 2:
        stride = (size[1], 1)
    else:
        raise ValueError("extract_C can only extract 1 or 2D tensors.")
    return from_tns.as_strided(size=size, stride=stride, storage_offset=int(offset))


def extract_same_stride(from_tns: torch.Tensor,
                        size: Union[Tuple[int, int], Tuple[int]],
                        other: torch.Tensor,
                        offset: int = 0) -> torch.Tensor:
    if is_f_contig(other, strict=True):
        return extract_fortran(from_tns, size, offset)
    else:
        return extract_C(from_tns, size, offset)


def create_same_stride(size: Union[Tuple[int, int], Tuple[int]],
                       other: torch.Tensor,
                       dtype: torch.dtype,
                       device: Union[str, torch.device],
                       pin_memory: bool = False) -> torch.Tensor:
    if is_f_contig(other, strict=True):
        return create_fortran(size, dtype, device, pin_memory)
    else:
        return create_C(size, dtype, device, pin_memory)


def copy_same_stride(tensor: torch.Tensor, pin_memory: bool = False) -> torch.Tensor:
    size = tensor.shape
    dtype = tensor.dtype
    device = tensor.device
    if is_f_contig(tensor, strict=True):
        new = create_fortran(size, dtype, device, pin_memory)
    else:
        new = create_C(size, dtype, device, pin_memory)
    new.copy_(tensor)
    return new


def create_fortran(size: Union[Tuple[int, ...]],
                   dtype: torch.dtype,
                   device: Union[str, torch.device],
                   pin_memory: bool = False) -> torch.Tensor:
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
    strides = [1] + np.cumprod(size)[:-1].tolist()
    return _new_strided_tensor(tuple(size), strides, dtype, device, pin_memory)


def create_C(size: Union[Tuple[int, int], Tuple[int]],
             dtype: torch.dtype,
             device: Union[str, torch.device],
             pin_memory: bool = False) -> torch.Tensor:
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
    strides = np.cumprod(size[1:][::-1])[::-1].tolist() + [1]
    return _new_strided_tensor(tuple(size), strides, dtype, device, pin_memory)


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
    #if len(tensor.shape) > 2:
    #    raise RuntimeError(
    #        "Cannot check F-contiguity of tensor with %d dimensions" % (len(tensor.shape)))
    strides = tensor.stride()
    if tensor.dim() == 1:
        return strides[0] == 1
    # 2 checks for 1D tensors which look 2D
    if tensor.shape[-2] == 1:
        if strict:
            return strides[-2] == 1
        return strides[-1] == 1 or strides[-2] == 1
    if tensor.shape[-1] == 1:
        if strict:
            return strides[-2] == 1 and strides[-1] >= tensor.shape[-2]
        return strides[-2] == 1
    # 2D tensors must have the stride on the first
    # dimension equal to 1 (columns are stored contiguously).
    if strides[-2] != 1 or strides[-1] < strides[-2]:
        return False
    return True


def is_contig(tensor: torch.Tensor) -> bool:
    stride = tensor.stride()
    for s in stride:
        if s == 1:
            return True
    return False


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
        warnings.warn("Changing type of %s tensor from %s to %s. "
                      "This will use more memory. If possible change 'inter_type' and "
                      "'final_type', or cast the original data to the appropriate type." %
                      (tensor.size(), tensor.dtype, dtype))
    out_np = tensor.numpy().astype(
        np_dtype, order='K', casting='unsafe', copy=True)
    return torch.from_numpy(out_np)


def move_tensor(tensor: torch.Tensor, device: Union[torch.device, str]) -> torch.Tensor:
    if str(device) == str(tensor.device):
        return tensor

    new_tensor = create_same_stride(tensor.size(), tensor, tensor.dtype, device)
    new_tensor.copy_(tensor)
    return new_tensor
