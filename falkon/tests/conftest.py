import dataclasses
from contextlib import contextmanager

import numpy as np
import pytest
import torch
import torch.cuda as tcd

from falkon.utils.tensor_helpers import move_tensor
from falkon.options import FalkonOptions
from falkon.utils import decide_cuda
from falkon.utils.devices import _cpu_used_mem
from falkon.sparse import SparseTensor

if decide_cuda():
    @pytest.fixture(scope="session", autouse=True)
    def initialize_cuda():
        torch.cuda.init()


@contextmanager
def memory_checker(opt: FalkonOptions, extra_mem=0, check_cpu=True):
    is_cpu = opt.use_cpu
    mem_check = False
    if (is_cpu and check_cpu and opt.max_cpu_mem < np.inf) or (not is_cpu and opt.max_gpu_mem < np.inf):
        mem_check = True

    start_ram = None
    if mem_check and not is_cpu:
        devices = list(range(torch.cuda.device_count()))
        start_ram = {}
        for dev in devices:
            tcd.reset_peak_memory_stats(dev)
            # We have to work around buggy memory stats: sometimes reset doesn't work as expected.
            start_ram[dev] = torch.cuda.max_memory_allocated(dev)
            #print("Start RAM dev %s: %.5fMB" % (dev, torch.cuda.max_memory_allocated(dev) / 2**20))
    elif mem_check:
        start_ram = _cpu_used_mem(uss=True)
        opt = dataclasses.replace(opt, max_cpu_mem=opt.max_cpu_mem + start_ram)

    yield opt

    # Check memory usage
    if mem_check and not is_cpu:
        devices = list(range(torch.cuda.device_count()))
        for dev in devices:
            used_ram = tcd.max_memory_allocated(dev) - start_ram[dev] - extra_mem
            if used_ram > opt.max_gpu_mem:
                raise MemoryError(
                        "DEV %d - Memory usage (%.2fMB) exceeds allowed usage (%.2fMB)" %
                        (dev, used_ram / 2 ** 20, opt.max_gpu_mem / 2 ** 20))
    elif mem_check:
        used_ram = _cpu_used_mem(uss=True) - start_ram - extra_mem
        if used_ram > opt.max_cpu_mem:
            raise MemoryError(
                "Memory usage (%.2fMB) exceeds allowed usage (%.2fMB)" %
                (used_ram / 2 ** 20, opt.max_cpu_mem / 2 ** 20))


def numpy_to_torch_type(dt):
    if dt == np.float32:
        return torch.float32
    elif dt == np.float64:
        return torch.float64
    return dt

def torch_to_numpy_type(dt):
    if dt == torch.float32:
        return np.float32
    elif dt == torch.float64:
        return np.float64
    return dt


def fix_mat(t, dtype, order, device="cpu", copy=False, numpy=False):
    if dtype is None or order is None:
        return None
    requires_grad = False
    if isinstance(t, torch.Tensor):
        requires_grad = t.requires_grad
        t = t.detach().numpy()
    if isinstance(t, np.ndarray):
        t = np.array(t, dtype=dtype, order=order, copy=copy)
        if numpy:
            return t
        t = move_tensor(torch.from_numpy(t), device)
        if t.is_cuda:
            torch.cuda.synchronize()
        if requires_grad:
            t.requires_grad_()
    return t


def fix_mats(*mats, order, device, dtype):
    order = make_tuple(order, len(mats))
    device = make_tuple(device, len(mats))
    dtype = make_tuple(dtype, len(mats))
    for i, m in enumerate(mats):
        if isinstance(m, SparseTensor):
            yield fix_sparse_mat(m, dtype=dtype[i], device=device[i])
        else:
            yield fix_mat(m, order=order[i], device=device[i], dtype=dtype[i])


def fix_sparse_mat(t, dtype, device="cpu"):
    out = t.to(dtype=numpy_to_torch_type(dtype), device=device)
    return out


def fix_mat_dt(t, dtype=None, numpy=False):
    if isinstance(t, SparseTensor):
        out = t.to(dtype=numpy_to_torch_type(dtype))
        if numpy:
            raise RuntimeError("Cannot convert SparseTensor to numpy")
    elif isinstance(t, torch.Tensor):
        out = t.to(dtype=numpy_to_torch_type(dtype))
        if numpy:
            out = out.numpy()
    else:
        out = t.astype(torch_to_numpy_type(dtype))
        if not numpy:
            out = torch.from_numpy(out)
    return out


def make_tuple(val, length):
    if isinstance(val, tuple) or isinstance(val, list):
        assert len(val) == length, "Input to `make_tuple` is already a list of the incorrect length."
    return tuple([val] * length)
