import threading
from typing import Tuple, Optional, Union

import numpy as np
import torch

from falkon.utils import CompOpt, devices, PropagatingThread
from falkon.utils.helpers import FakeQueue
from falkon.utils.tensor_helpers import is_contig

__all__ = ("_setup_opt", "_check_contiguity", "_get_gpu_info", "_get_cpu_ram",
           "_start_wait_processes", "_gpu_tns_same_memory")


def _setup_opt(opt, is_cpu=False):
    if opt is not None:
        opt = CompOpt(opt)
    else:
        opt = CompOpt()
    opt.setdefault('max_gpu_mem', np.inf)
    opt.setdefault('max_cpu_mem', np.inf)
    opt.setdefault('no_single_kernel', True)
    if is_cpu:
        opt['use_cpu'] = True
    return opt


def _check_contiguity(*args: Tuple[Optional[torch.Tensor], str]) -> None:
    for tensor, name in args:
        if tensor is not None and not is_contig(tensor):
            raise ValueError(f"Tensor '{name}' must be memory contiguous")


def _get_gpu_info(opt, slack=0.9):
    # List available devices, get their relative speed and split
    # computations based on device relative speed.
    gpu_info = [v for k, v in devices.get_device_info(opt).items() if k >= 0]
    for g in gpu_info:
        g.usable_ram = min(g.free_memory * slack, opt.max_gpu_mem * slack)
    return gpu_info


def _get_cpu_ram(opt, slack=0.9):
    cpu_info = devices.get_device_info(opt)[-1]
    avail_mem = min(cpu_info.free_memory, opt.max_cpu_mem - cpu_info.used_memory)
    return avail_mem * slack


def _start_wait_processes(target, args):
    processes = []
    for i, a in enumerate(args):
        args_queue = FakeQueue()
        args_queue.put(a[0])
        new_args_tuple = (i, args_queue, a[1])
        # PropagatingThread throws any exception which happened in the thread on join
        process = PropagatingThread(target=target, name=f'GPU-{a[1]}', args=new_args_tuple)
        processes.append(process)
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def _gpu_tns_same_memory(A: torch.Tensor, B: torch.Tensor) -> bool:
    return (A.dtype == B.dtype) and \
           (A.shape == B.shape) and \
           (A.data_ptr() == B.data_ptr()) and \
           (A.stride() == B.stride())
