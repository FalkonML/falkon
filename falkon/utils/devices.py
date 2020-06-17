import os
from dataclasses import dataclass
from typing import Dict, Union

import psutil
import resource
import torch
import torch.cuda as tcd

from . import CompOpt, TicToc

__all__ = ("get_device_info", "DeviceInfo", "num_gpus")

__COMP_DATA = {}


@dataclass
class DeviceInfo:
    Id: int
    speed: float = 0
    total_memory: float = 0
    used_memory: float = 0
    free_memory: float = 0
    gpu_name: str = ''

    def update_memory(self, total_memory=0, used_memory=0, free_memory=0):
        self.total_memory = total_memory
        self.used_memory = used_memory
        self.free_memory = free_memory

    @property
    def isCPU(self):
        return self.Id == -1

    @property
    def isGPU(self):
        return self.Id >= 0

    def __str__(self):
        if self.isCPU:
            return 'cpu'
        else:
            return 'cuda:%s' % self.Id

    def __repr__(self):
        return ("DeviceInfo(Id={Id}, speed={speed}, total_memory={total_memory}, "
                "used_memory={used_memory}, free_memory={free_memory})".format(
            Id=self.Id, speed=self.speed, total_memory=self.total_memory,
            used_memory=self.used_memory, free_memory=self.free_memory))


def _get_cpu_device_info(opt: CompOpt, data_dict: Dict[int, DeviceInfo]) -> Dict[int, DeviceInfo]:
    cpu_free_mem = _cpu_available_mem()
    cpu_used_mem = _cpu_used_mem()
    if -1 in data_dict:
        data_dict[-1].update_memory(
            free_memory=cpu_free_mem, used_memory=cpu_used_mem)
    else:
        if opt.compute_arch_speed:
            cpu_speed = _measure_performance(-1, cpu_free_mem)
        else:
            cpu_speed = psutil.cpu_count()
        data_dict[-1] = DeviceInfo(
            Id=-1,
            used_memory=cpu_used_mem,
            free_memory=cpu_free_mem,
            speed=cpu_speed
        )
    return data_dict


def _get_gpu_device_info(opt: CompOpt,
                         g: int,
                         data_dict: Dict[int, DeviceInfo]) -> Dict[int, DeviceInfo]:
    try:
        from ..cuda.cudart_gpu import cuda_meminfo
    except Exception as e:
        raise ValueError("Failed to import cudart_gpu module. "
                         "Please check dependencies.") from e

    mem_free, mem_total = cuda_meminfo(g)
    mem_used = mem_total - mem_free
    cached_free_mem = tcd.memory_reserved(g) - tcd.memory_allocated(g)

    if g in data_dict:
        data_dict[g].update_memory(
            total_memory=mem_total,
            used_memory=mem_used - cached_free_mem,
            free_memory=mem_free + cached_free_mem)
    else:
        properties = tcd.get_device_properties(g)
        if opt.compute_arch_speed:
            gpu_speed = _measure_performance(g, mem_free)
        else:
            gpu_speed = properties.multi_processor_count

        data_dict[g] = DeviceInfo(
            Id=g,
            speed=float(gpu_speed),
            total_memory=mem_total,
            used_memory=mem_used - cached_free_mem,
            free_memory=mem_free + cached_free_mem,
            gpu_name=properties.name)

    return data_dict


def _measure_performance(g, mem):
    tm = TicToc()
    tt = 0
    f = 1
    if g == -1:
        dev = torch.device('cpu')
    else:
        dev = torch.device('cuda:%s' % g)
    dtt = torch.double

    a = torch.eye(1024, 1024, dtype=dtt, device=dev)
    a.addmm_(a, a)
    if g >= 0:
        tcd.synchronize(device=dev)

    while tt < 1.0 and mem > 8.0 * (f * 2048.0) ** 2:
        tm.tic()
        a = torch.eye(f * 2048, f * 2048, dtype=dtt, device=dev)
        a.addmm_(a, a)
        if g >= 0:
            tcd.synchronize(device=dev)
        tt = tm.toc_val()
        f *= 2

    print('%s:%s - speed: %s' % (dev.type, dev.index, (float(f) ** 3) / tt))

    del a
    if g >= 0:
        tcd.synchronize(device=dev)
    tcd.empty_cache()

    return (float(f) ** 3) / tt


def _cpu_available_mem() -> int:
    return psutil.virtual_memory().available


def _max_used_mem() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _cpu_used_mem(uss=True) -> int:
    process = psutil.Process(os.getpid())
    if not uss:
        return process.memory_info().rss  # in bytes
    try:
        # http://grodola.blogspot.com/2016/02/psutil-4-real-process-memory-and-environ.html
        return process.memory_full_info().uss  # Unique set size
    except:  # Typically a permission error
        return process.memory_info().rss  # in bytes


def get_device_info(opt: Union[None, Dict, CompOpt]) -> Dict[int, DeviceInfo]:
    """Retrieve speed and memory information about CPU and GPU devices on the system

    The behaviour of this function is influenced by the `opt` parameter:
     - if 'use_cpu' (default False) is set to True, then only CPU devices will be probed.
       Please set this to True if your machine does not have any GPUs.
     - if 'compute_arch_speed' (default True) is set to True we will measure performance
       of each device by running a compute workload. This should not take more than 10 seconds,
       but disable for benchmarking and when a single GPU is present.
       The speed of a device is cached in between calls to the function.

    Parameters:
    -----------
    opt :
        Options for fetching device information. Supported options are described above.

    Returns:
    --------
    device_info : Dict[int, DeviceInfo]
        A dictionary mapping device IDs to their DeviceInfo object (which contains information
        about available memory and speed). The IDs of CPU devices are negative.
    """
    if opt is None:
        opt = CompOpt()
    else:
        opt = CompOpt(opt)
    opt.setdefault('use_cpu', False)
    opt.setdefault('compute_arch_speed', False)

    global __COMP_DATA
    # List all devices.
    # If they are already cached we can update memory characteristics
    # Otherwise we can also compute performance measure
    if opt.use_cpu:
        __COMP_DATA = _get_cpu_device_info(opt, __COMP_DATA)
        return __COMP_DATA

    for g in range(0, tcd.device_count()):
        __COMP_DATA = _get_gpu_device_info(opt, g, __COMP_DATA)

    if len(__COMP_DATA) == 0:
        raise RuntimeError("No suitable device found. Enable option 'use_cpu' "
                           "if no GPU is available.")

    return __COMP_DATA


def num_gpus(opt: Union[None, Dict, CompOpt]) -> int:
    global __COMP_DATA
    if len(__COMP_DATA) == 0:
        get_device_info(opt)
    return len([c for c in __COMP_DATA.keys() if c >= 0])
