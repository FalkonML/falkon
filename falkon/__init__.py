import os
import torch

# Check torch version vs. compilation version
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
# https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/__init__.py
from falkon.c_ext import cuda_version
flk_cuda_version = cuda_version()
if torch.cuda.is_available() and flk_cuda_version != -1:
    if cuda_version < 10000:
        f_major, f_minor = int(str(flk_cuda_version)[0]), int(str(flk_cuda_version)[2])
    else:
        f_major, f_minor = int(str(flk_cuda_version)[0:2]), int(str(flk_cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != f_major:
        raise RuntimeError(
            f'PyTorch and Falkon were compiled with different CUDA versions. '
            f'PyTorch has CUDA version {t_major}.{t_minor} and Falkon has CUDA version '
            f'{f_major}.{f_minor}. Please reinstall Falkon such that its version matches '
            f'your PyTorch install.')

# Library exports
from . import kernels, sparse, center_selection, preconditioner, optim, gsc_losses, hopt  # noqa: E402
from .options import FalkonOptions  # noqa: E402
from .models import Falkon, LogisticFalkon, InCoreFalkon  # noqa: E402

# Set __version__ attribute on the package
init_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(init_dir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

__all__ = (
    'Falkon',
    'LogisticFalkon',
    'InCoreFalkon',
    'FalkonOptions',
    'kernels',
    'optim',
    'preconditioner',
    'center_selection',
    'sparse',
    'gsc_losses',
    'hopt',
    '__version__',
)
